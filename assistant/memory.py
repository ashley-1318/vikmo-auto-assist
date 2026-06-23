"""
assistant/memory.py - Conversation Memory Manager

Design Philosophy:
-----------------
Effective multi-turn conversation requires remembering:
1. The full chat history (messages sent/received)
2. Entity context — which vehicle was mentioned, which SKU was referenced
3. Order history for the session

Why custom memory instead of LangChain's built-in memory?
- Full control over what gets preserved and what gets summarized
- Explicit entity tracking (vehicle, SKU) avoids re-asking the user
- Easy to inspect and debug during evaluation
- Can extend to persistent storage (database) without changing the interface

The EntityMemory concept:
    User says "Show brake pads for KTM Duke 200"
    → vehicle_context = "KTM Duke 200"
    → last_mentioned_skus = []

    User says "Check stock"
    → memory injects vehicle_context → looks up brake pads for KTM Duke 200
    → last_mentioned_skus = ["BRK-1005", "BRK-1007"]

    User says "Order 2 of those"
    → memory injects last_mentioned_skus[0] → creates order
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum number of turns to keep in full context window.
# Beyond this, we truncate to avoid exceeding LLM token limits.
MAX_HISTORY_TURNS = 20


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_llm_format(self) -> Dict:
        """Returns the minimal dict format that LangChain/Groq expects."""
        return {"role": self.role, "content": self.content}


@dataclass
class EntityContext:
    """
    Tracks entities mentioned in the conversation.

    This is the core of our context-carryover system.
    When the user says "check stock" without specifying a product,
    we look at last_mentioned_skus to infer what they mean.
    """
    vehicle: Optional[str] = None          # e.g., "KTM Duke 200"
    last_mentioned_skus: List[str] = field(default_factory=list)  # Most recent SKUs discussed
    last_mentioned_category: Optional[str] = None  # e.g., "Brakes"
    dealer_name: Optional[str] = None      # Set when dealer introduces themselves


class ConversationMemory:
    """
    Manages conversation history and entity context for a single session.

    Thread Safety: Not thread-safe. Each session (user tab) should have its own instance.
    Streamlit handles this via st.session_state.

    Methods:
    - add_user_message() / add_assistant_message() — record turns
    - update_entity_context() — extract and store entities from content
    - get_history_for_llm() — format history for Groq API
    - get_context_summary() — brief string of current context (injected into prompts)
    """

    def __init__(self):
        self.messages: List[Message] = []
        self.entities: EntityContext = EntityContext()
        self.order_history: List[Dict] = []
        self.session_id: str = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # ─────────────────────────────────────────────────────────
    # Message Management
    # ─────────────────────────────────────────────────────────

    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Record a user message and attempt to extract entities from it."""
        msg = Message(role="user", content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._extract_entities_from_user_message(content)
        logger.debug(f"[Memory] User: {content[:80]}...")

    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Record an assistant message and extract any SKUs mentioned in the response."""
        msg = Message(role="assistant", content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._extract_skus_from_assistant_message(content)
        logger.debug(f"[Memory] Assistant: {content[:80]}...")

    def add_order(self, order: Dict) -> None:
        """Track a confirmed order in session history."""
        self.order_history.append({**order, "timestamp": datetime.utcnow().isoformat()})
        logger.info(f"[Memory] Order recorded: {order.get('order_id')}")

    # ─────────────────────────────────────────────────────────
    # Entity Extraction
    # ─────────────────────────────────────────────────────────

    def _extract_entities_from_user_message(self, content: str) -> None:
        """
        Heuristic entity extraction from user messages.

        We look for:
        1. Vehicle names (common Indian motorcycle brands/models)
        2. Explicit SKU patterns (e.g., BRK-1001)
        3. Category hints (tyres, brake, oil, etc.)

        Note: For production, replace with a dedicated NER model or
        let the LLM return structured entity extractions in its response.
        """
        content_lower = content.lower()

        # SKU pattern detection (e.g., BRK-1001, TYR-2001)
        import re
        sku_pattern = r'\b([A-Z]{2,4}-\d{4})\b'
        skus = re.findall(sku_pattern, content.upper())
        if skus:
            self.entities.last_mentioned_skus = skus[:3]  # Keep up to 3 recent SKUs

        # Vehicle name detection — ordered by specificity (longer matches first)
        vehicle_keywords = [
            "royal enfield himalayan", "royal enfield classic 350", "royal enfield meteor 350",
            "royal enfield thunderbird", "royal enfield bullet 350",
            "ktm duke 390", "ktm duke 250", "ktm duke 200", "ktm rc 390",
            "honda cbr650r", "honda cbr600rr", "honda cbr",
            "yamaha r15 v4", "yamaha r15", "yamaha mt-15", "yamaha fz-s",
            "bajaj pulsar ns200", "bajaj pulsar rs200", "bajaj pulsar 150", "bajaj pulsar 180",
            "tvs apache rtr 160", "tvs apache rtr 180", "tvs apache",
            "bmw g310gs",
        ]

        for vehicle in vehicle_keywords:
            if vehicle in content_lower:
                # Capitalize properly for display
                self.entities.vehicle = vehicle.title()
                logger.debug(f"[Memory] Vehicle context set: {self.entities.vehicle}")
                break

        # Category detection
        category_map = {
            "brake": "Brakes",
            "tyre": "Tyres", "tire": "Tyres",
            "engine oil": "Engine", "oil": "Engine",
            "spark plug": "Engine",
            "filter": "Filters",
            "suspension": "Suspension", "shock": "Suspension", "fork": "Suspension",
            "chain": "Drivetrain", "sprocket": "Drivetrain",
            "battery": "Electrical", "headlight": "Electrical",
            "accessories": "Accessories", "grip": "Accessories",
        }

        for keyword, category in category_map.items():
            if keyword in content_lower:
                self.entities.last_mentioned_category = category
                break

    def _extract_skus_from_assistant_message(self, content: str) -> None:
        """
        Extract SKUs from assistant responses so we can track what was last shown.

        E.g., if the assistant listed BRK-1001 and BRK-1003, and the user
        says "check stock", we know which brake pads to check.
        """
        import re
        sku_pattern = r'\b([A-Z]{2,4}-\d{4})\b'
        skus = re.findall(sku_pattern, content.upper())
        if skus:
            # Preserve existing user-mentioned SKUs; add assistant-mentioned ones
            existing = set(self.entities.last_mentioned_skus)
            new_skus = [s for s in skus if s not in existing]
            self.entities.last_mentioned_skus = (
                list(existing) + new_skus
            )[:5]  # Keep at most 5

    def update_entity_context(self, updates: Dict[str, Any]) -> None:
        """
        Explicitly update entity context (called by the agent after tool responses).

        Args:
            updates: Dict with keys like 'vehicle', 'skus', 'category', 'dealer_name'
        """
        if "vehicle" in updates and updates["vehicle"]:
            self.entities.vehicle = updates["vehicle"]
        if "skus" in updates and updates["skus"]:
            self.entities.last_mentioned_skus = updates["skus"]
        if "category" in updates and updates["category"]:
            self.entities.last_mentioned_category = updates["category"]
        if "dealer_name" in updates and updates["dealer_name"]:
            self.entities.dealer_name = updates["dealer_name"]

    # ─────────────────────────────────────────────────────────
    # History Formatting for LLM
    # ─────────────────────────────────────────────────────────

    def get_history_for_llm(self, max_turns: int = MAX_HISTORY_TURNS) -> List[Dict]:
        """
        Returns conversation history in the format expected by Groq's chat API.

        Truncation strategy: Keep the most recent `max_turns` messages.
        This prevents token limit errors on long conversations while
        preserving the most relevant recent context.
        """
        # Filter out system messages (the system prompt is injected separately)
        chat_messages = [m for m in self.messages if m.role in ("user", "assistant")]

        # Truncate to last N turns (each turn = user + assistant = 2 messages)
        if len(chat_messages) > max_turns * 2:
            chat_messages = chat_messages[-(max_turns * 2):]

        return [m.to_llm_format() for m in chat_messages]

    def get_context_summary(self) -> str:
        """
        Returns a brief context summary injected into each LLM prompt.

        This tells the LLM what context has been established so it can
        answer follow-up questions without re-asking for vehicle/product info.
        """
        parts = []

        if self.entities.dealer_name:
            parts.append(f"Dealer: {self.entities.dealer_name}")

        if self.entities.vehicle:
            parts.append(f"Current vehicle context: {self.entities.vehicle}")

        if self.entities.last_mentioned_skus:
            skus_str = ", ".join(self.entities.last_mentioned_skus[:3])
            parts.append(f"Recently discussed SKUs: {skus_str}")

        if self.entities.last_mentioned_category:
            parts.append(f"Last category: {self.entities.last_mentioned_category}")

        if not parts:
            return "No conversation context established yet."

        return " | ".join(parts)

    def get_primary_sku(self) -> Optional[str]:
        """Returns the most recently discussed SKU, if any."""
        if self.entities.last_mentioned_skus:
            return self.entities.last_mentioned_skus[0]
        return None

    def get_vehicle_context(self) -> Optional[str]:
        """Returns the current vehicle context, if any."""
        return self.entities.vehicle

    def clear(self) -> None:
        """Resets the conversation (used for 'new chat' button)."""
        self.messages.clear()
        self.entities = EntityContext()
        self.order_history.clear()
        logger.info("[Memory] Conversation cleared.")

    def to_dict(self) -> Dict:
        """Serialize full memory state (for export/debugging)."""
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "entities": {
                "vehicle": self.entities.vehicle,
                "last_mentioned_skus": self.entities.last_mentioned_skus,
                "last_mentioned_category": self.entities.last_mentioned_category,
                "dealer_name": self.entities.dealer_name,
            },
            "order_history": self.order_history,
        }
