"""
assistant/agent.py - Groq LLM Agent with Tool Calling and RAG

Architecture:
-------------
The agent orchestrates the full RAG + Tool Calling loop:
1. Receive user message
2. Retrieve relevant products from FAISS (RAG)
3. Build prompt = system_prompt + rag_context + conversation_history + user_message
4. Call Groq Llama 3.3 70B with tool definitions
5. If tool call → execute tool → feed result back → call LLM again
6. Return final grounded response

Prompt Engineering Strategy:
- System prompt defines domain restriction (automotive parts only)
- RAG context is injected before conversation history
- Context summary from memory informs the LLM of established entities
- Tool use is encouraged via explicit instructions in the system prompt
- Guardrails prevent hallucination via explicit "use only provided context" instructions

Multi-turn: The full conversation history is passed every turn. The LLM sees
what was said before and can resolve pronouns ("it", "those") to entities
established earlier.
"""

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from groq import Groq

from assistant.memory import ConversationMemory
from assistant.rag import get_rag_engine
from assistant.tools import (
    TOOL_DEFINITIONS,
    check_stock,
    create_order,
    find_parts_by_vehicle,
    find_similar_products,
    get_low_stock_alerts,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
# Vision model — llama-4-scout supports image_url content blocks on Groq
# Falls back gracefully if the model is unavailable
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOOL_ITERATIONS = 3  # Prevent infinite tool call loops
RETRIEVAL_TOP_K = 5

# Prompt sent to the vision model for part identification
VISION_SYSTEM_PROMPT = """You are an expert automotive parts identification specialist.
Analyse the provided image and identify the automotive part shown.

Respond in this exact JSON format (no markdown, raw JSON only):
{
  "part_name": "<common name of the part>",
  "category": "<one of: Brakes | Tyres | Engine | Suspension | Electrical | Drivetrain | Filters | Accessories>",
  "brand_if_visible": "<brand name or null>",
  "condition": "<New | Used | Unknown>",
  "confidence": "<High | Medium | Low>",
  "description": "<1-2 sentences describing what you see>",
  "search_query": "<short keyword string to search the catalogue e.g. front brake pad disc>"
}"""


# ─────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """You are VIKMO Assistant, an expert AI-powered dealer assistant for an automotive parts catalogue.

## Your Role
You help dealerships find parts, check stock, look up vehicle compatibility, and place orders.

## Domain Restriction (CRITICAL)
You MUST only answer questions about automotive parts, vehicle fitment, inventory, and dealer orders.
If asked about anything else (weather, sports, politics, general knowledge), respond:
"I can only assist with automotive parts, inventory, vehicle fitment and dealer orders."

## Grounding Rules (CRITICAL - NO HALLUCINATION)
- Use ONLY the information provided in the "Retrieved Product Context" below.
- NEVER invent product names, SKUs, prices, or stock values.
- If information is not in the context, say: "I could not find matching catalogue information."
- Do NOT use your training knowledge for product details.

## Tool Usage Rules
- Call `find_parts_by_vehicle` when a vehicle model is mentioned and parts are requested.
- Call `check_stock` when asked about availability or stock of a specific SKU.
- Call `create_order` ONLY when user explicitly asks to place/confirm an order.
- Always check stock before confirming an order recommendation.

## Clarification Rules
- Ask for vehicle model when user requests parts without specifying one.
- Ask for confirmation before creating an order.
- Ask for dealer name if not established in conversation.

## Conversation Context
{context_summary}

## Response Format
- Be concise and professional.
- For product listings, show SKU, name, price (₹), and stock.
- For orders, confirm line items and total amount.
- Highlight low stock warnings (stock ≤ 10 units).
- For alternatives when out of stock, say "Alternative options available: ..."
"""


def _build_system_prompt(memory: ConversationMemory, rag_context: str) -> str:
    """Builds the full system prompt with context summary and RAG results."""
    context_summary = memory.get_context_summary()
    base_prompt = SYSTEM_PROMPT_TEMPLATE.format(context_summary=context_summary)
    return base_prompt + f"\n## Retrieved Product Context\n{rag_context}"


def _execute_tool(tool_name: str, tool_args: Dict, catalogue_df: pd.DataFrame) -> str:
    """
    Dispatches a tool call to the appropriate function.
    Returns the result as a JSON string for the LLM's tool message.
    """
    logger.info(f"[Agent] Executing tool: {tool_name}({tool_args})")

    try:
        if tool_name == "check_stock":
            result = check_stock(tool_args["sku"], catalogue_df)
            # If out of stock, automatically find alternatives
            if result.get("found") and not result.get("available"):
                from pathlib import Path
                row = catalogue_df[catalogue_df["sku"].str.upper() == tool_args["sku"].upper()]
                if not row.empty:
                    category = str(row.iloc[0]["category"])
                    alts = find_similar_products(tool_args["sku"], category, catalogue_df)
                    result["alternatives"] = alts

        elif tool_name == "find_parts_by_vehicle":
            result = find_parts_by_vehicle(tool_args["vehicle_name"], catalogue_df)

        elif tool_name == "create_order":
            result = create_order(tool_args, catalogue_df)

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        logger.error(f"[Agent] Tool {tool_name} raised exception: {e}")
        result = {"error": str(e)}

    return json.dumps(result, ensure_ascii=False)


class DealerAgent:
    """
    The core agent that orchestrates RAG + Tool Calling + Memory.

    Usage:
        agent = DealerAgent()
        agent.initialize()  # loads RAG engine
        response = agent.chat("Show brake pads for KTM Duke 200")
    """

    def __init__(self):
        self.client: Optional[Groq] = None
        self.rag_engine = None
        self.catalogue_df: Optional[pd.DataFrame] = None
        self._initialized = False

    def initialize(self) -> None:
        """Load all components. Call once at application startup."""
        logger.info("[Agent] Initializing DealerAgent...")

        # Groq client reads GROQ_API_KEY from environment (set via .env)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Load RAG engine (embeds catalogue, builds FAISS index)
        self.rag_engine = get_rag_engine()

        # Load catalogue DataFrame (for tool use)
        from pathlib import Path
        data_path = Path(__file__).parent.parent / "data" / "catalogue.csv"
        self.catalogue_df = pd.read_csv(data_path)

        self._initialized = True
        logger.info("[Agent] DealerAgent initialized.")

    def chat(
        self,
        user_message: str,
        memory: ConversationMemory,
    ) -> Dict[str, Any]:
        """
        Process a single user turn. Returns a rich response dict.

        Args:
            user_message: The user's input text.
            memory: The session's ConversationMemory instance.

        Returns:
            {
                "response": str,           # Final text response
                "tool_calls": list,        # Tools that were called
                "retrieved_products": list, # RAG results shown
                "order": dict | None,      # Order confirmation if order was placed
            }
        """
        if not self._initialized:
            raise RuntimeError("DealerAgent not initialized. Call .initialize() first.")

        # ─── Step 1: RAG Retrieval ───────────────────────────────────────────
        # Augment query with vehicle context for better retrieval
        vehicle_ctx = memory.get_vehicle_context()
        retrieval_query = user_message
        if vehicle_ctx and vehicle_ctx.lower() not in user_message.lower():
            retrieval_query = f"{user_message} {vehicle_ctx}"

        rag_results = self.rag_engine.search(retrieval_query, top_k=RETRIEVAL_TOP_K)
        rag_context = self.rag_engine.format_context(rag_results)
        retrieved_products = [doc.to_dict() for doc, _ in rag_results]

        # ─── Step 2: Build Messages ──────────────────────────────────────────
        system_prompt = _build_system_prompt(memory, rag_context)
        history = memory.get_history_for_llm()

        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

        # ─── Step 3: LLM + Tool Calling Loop ────────────────────────────────
        tool_calls_log = []
        final_response = ""
        order_result = None

        for iteration in range(MAX_TOOL_ITERATIONS):
            logger.info(f"[Agent] LLM call iteration {iteration + 1}")

            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.1,       # Low temperature → more deterministic, less hallucination
                max_tokens=2048,
            )

            choice = response.choices[0]
            assistant_message = choice.message

            # If no tool call, we have the final response
            if not assistant_message.tool_calls:
                final_response = assistant_message.content or ""
                break

            # Process tool calls
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            })

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_result = _execute_tool(tool_name, tool_args, self.catalogue_df)
                tool_result_dict = json.loads(tool_result)

                # Track order confirmations for UI display
                if tool_name == "create_order" and tool_result_dict.get("success"):
                    order_result = tool_result_dict
                    memory.add_order(order_result)

                # Log tool call for UI transparency
                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result_dict,
                })

                # Feed tool result back to LLM
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.id,
                })

        else:
            # Reached max iterations — get final response
            final_response_obj = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            final_response = final_response_obj.choices[0].message.content or ""

        # ─── Step 4: Update Memory ───────────────────────────────────────────
        memory.add_user_message(user_message)
        memory.add_assistant_message(final_response)

        # Update entity context from tool calls
        for tc in tool_calls_log:
            if tc["tool"] == "find_parts_by_vehicle":
                memory.update_entity_context({"vehicle": tc["args"].get("vehicle_name")})
            elif tc["tool"] == "check_stock":
                sku = tc["args"].get("sku")
                if sku:
                    current = memory.entities.last_mentioned_skus
                    if sku not in current:
                        memory.update_entity_context({"skus": [sku] + current})

        return {
            "response": final_response,
            "tool_calls": tool_calls_log,
            "retrieved_products": retrieved_products,
            "order": order_result,
        }

    def identify_part_from_image(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> Dict[str, Any]:
        """
        Identify an automotive part from an uploaded image using Groq vision.

        Flow:
        1. Encode image to base64 data-URI.
        2. Send to Groq vision model with a structured automotive prompt.
        3. Parse the JSON response to get part name, category, confidence.
        4. Run a RAG search using the model's suggested search_query to find
           matching products in the catalogue.
        5. Return combined result: vision analysis + matched catalogue products.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            mime_type: MIME type e.g. "image/jpeg", "image/png".

        Returns:
            Dict with keys:
              - vision_result: parsed JSON from the vision model
              - matched_products: list of catalogue products found via RAG
              - raw_text: raw LLM output (for debugging)
              - error: str if something failed, else None
        """
        if not self._initialized:
            raise RuntimeError("DealerAgent not initialized.")

        # Step 1: Base64-encode the image
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64}"

        # Step 2: Call Groq vision model
        try:
            vision_response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Please identify this automotive part and respond "
                                    "in the exact JSON format specified in the system prompt."
                                ),
                            },
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=512,
            )
            raw_text = vision_response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[Vision] Groq vision API error: {e}")
            return {
                "vision_result": None,
                "matched_products": [],
                "raw_text": "",
                "error": f"Vision model error: {str(e)}",
            }

        # Step 3: Parse the JSON response
        vision_result = None
        try:
            # Strip markdown code fences if the model added them
            clean = raw_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            vision_result = json.loads(clean.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning(f"[Vision] Could not parse JSON: {raw_text[:200]}")
            # Build a fallback result from the raw text
            vision_result = {
                "part_name": "Unknown",
                "category": "Unknown",
                "brand_if_visible": None,
                "condition": "Unknown",
                "confidence": "Low",
                "description": raw_text[:300],
                "search_query": "automotive part",
            }

        # Step 4: RAG search using the vision model's suggested search query
        search_query = vision_result.get("search_query", vision_result.get("part_name", ""))
        matched_products = []
        if search_query and search_query != "automotive part":
            try:
                rag_hits = self.rag_engine.search(search_query, top_k=4)
                matched_products = [doc.to_dict() for doc, _ in rag_hits]
            except Exception as e:
                logger.warning(f"[Vision] RAG search failed: {e}")

        logger.info(
            f"[Vision] Identified: {vision_result.get('part_name')} "
            f"(confidence={vision_result.get('confidence')}) | "
            f"{len(matched_products)} catalogue matches"
        )

        return {
            "vision_result": vision_result,
            "matched_products": matched_products,
            "raw_text": raw_text,
            "error": None,
        }

    def get_low_stock_alerts(self, threshold: int = 10) -> List[Dict]:
        """Returns all products with stock at or below threshold."""
        return get_low_stock_alerts(threshold, self.catalogue_df)


# ─────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────
_agent_instance: Optional[DealerAgent] = None


def get_agent() -> DealerAgent:
    """Returns the singleton DealerAgent, initializing if needed."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = DealerAgent()
        _agent_instance.initialize()
    return _agent_instance
