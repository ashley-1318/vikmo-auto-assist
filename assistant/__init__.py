# assistant/__init__.py
# Package marker - exposes key components for easy import
from assistant.rag import RAGEngine, get_rag_engine
from assistant.memory import ConversationMemory
from assistant.agent import DealerAgent, get_agent
from assistant.tools import (
    check_stock,
    find_parts_by_vehicle,
    create_order,
    get_low_stock_alerts,
    TOOL_DEFINITIONS,
)

__all__ = [
    "RAGEngine",
    "get_rag_engine",
    "ConversationMemory",
    "DealerAgent",
    "get_agent",
    "check_stock",
    "find_parts_by_vehicle",
    "create_order",
    "get_low_stock_alerts",
    "TOOL_DEFINITIONS",
]
