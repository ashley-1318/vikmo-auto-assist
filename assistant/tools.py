"""
assistant/tools.py - Structured Tool Definitions

Tools give the LLM deterministic operations:
- Stock checks (exact, from DataFrame - no hallucination possible)
- Vehicle-fitment lookups (exact string matching)
- Order creation (validated via Pydantic)

Why Pydantic for Order Validation?
- Runtime type checking catches malformed LLM outputs
- Validation errors provide clear feedback to retry logic
- Schema is auto-documented for the LLM's tool descriptor
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_order_counter = 1000

def _generate_order_id() -> str:
    global _order_counter
    _order_counter += 1
    return f"ORD{_order_counter}"


# ─────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────

class OrderItem(BaseModel):
    sku: str = Field(..., description="Product SKU identifier e.g. BRK-1001")
    quantity: int = Field(..., ge=1, le=1000, description="Units to order")

    @field_validator("sku")
    @classmethod
    def validate_sku_format(cls, v: str) -> str:
        import re
        v = v.strip().upper()
        if not re.match(r'^[A-Z]{2,4}-\d{4}$', v):
            raise ValueError(f"Invalid SKU format: '{v}'. Expected format: BRK-1001")
        return v


class Order(BaseModel):
    dealer_name: str = Field(..., min_length=2, max_length=100)
    items: List[OrderItem] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_skus(self) -> "Order":
        skus = [item.sku for item in self.items]
        if len(skus) != len(set(skus)):
            raise ValueError("Duplicate SKUs in order. Combine quantities instead.")
        return self


# ─────────────────────────────────────────────────────────
# Tool Functions
# ─────────────────────────────────────────────────────────

def _load_catalogue():
    import pandas as pd
    from pathlib import Path
    data_path = Path(__file__).parent.parent / "data" / "catalogue.csv"
    return pd.read_csv(data_path)


def check_stock(sku: str, catalogue_df=None) -> Dict[str, Any]:
    """
    Check real-time stock for a specific SKU.
    Reads directly from DataFrame - cannot hallucinate values.
    """
    sku = sku.strip().upper()
    logger.info(f"[Tool] check_stock: {sku}")
    if catalogue_df is None:
        catalogue_df = _load_catalogue()

    row = catalogue_df[catalogue_df["sku"].str.upper() == sku]
    if row.empty:
        return {"sku": sku, "found": False, "error": f"SKU '{sku}' not found in catalogue."}

    item = row.iloc[0]
    stock_val = int(item["stock"])
    return {
        "sku": sku,
        "name": str(item["name"]),
        "stock": stock_val,
        "available": stock_val > 0,
        "low_stock": 0 < stock_val <= 10,
        "price_inr": float(item["price_inr"]),
        "found": True,
    }


def find_parts_by_vehicle(vehicle_name: str, catalogue_df=None) -> List[Dict[str, Any]]:
    """
    Find all compatible parts for a given vehicle model.
    Uses token-based AND matching for precision over recall.
    """
    logger.info(f"[Tool] find_parts_by_vehicle: {vehicle_name}")
    if catalogue_df is None:
        catalogue_df = _load_catalogue()

    search_tokens = vehicle_name.lower().split()

    def matches(fitment: str) -> bool:
        fitment_lower = fitment.lower()
        return all(token in fitment_lower for token in search_tokens)

    matching = catalogue_df[catalogue_df["vehicle_fitment"].apply(matches)]

    # Broad fallback: any token matches
    if matching.empty:
        broad_tokens = search_tokens[:2] if len(search_tokens) > 2 else search_tokens
        matching = catalogue_df[
            catalogue_df["vehicle_fitment"].apply(
                lambda f: any(tok in f.lower() for tok in broad_tokens)
            )
        ]

    results = []
    for _, row in matching.iterrows():
        results.append({
            "sku": str(row["sku"]),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "brand": str(row["brand"]),
            "price_inr": float(row["price_inr"]),
            "stock": int(row["stock"]),
            "available": int(row["stock"]) > 0,
            "low_stock": 0 < int(row["stock"]) <= 10,
            "vehicle_fitment": str(row["vehicle_fitment"]),
        })
    return results


def find_similar_products(sku: str, category: str, catalogue_df=None) -> List[Dict[str, Any]]:
    """Bonus: Find in-stock alternatives when a SKU is out of stock."""
    if catalogue_df is None:
        catalogue_df = _load_catalogue()

    alternatives = catalogue_df[
        (catalogue_df["category"] == category) &
        (catalogue_df["sku"] != sku) &
        (catalogue_df["stock"] > 0)
    ].sort_values("stock", ascending=False).head(3)

    return [
        {
            "sku": str(row["sku"]),
            "name": str(row["name"]),
            "brand": str(row["brand"]),
            "price_inr": float(row["price_inr"]),
            "stock": int(row["stock"]),
            "vehicle_fitment": str(row["vehicle_fitment"]),
        }
        for _, row in alternatives.iterrows()
    ]


def create_order(order_data: Dict[str, Any], catalogue_df=None) -> Dict[str, Any]:
    """
    Create and validate a dealer order.
    
    Validation pipeline:
    1. Pydantic schema validation (types, formats)
    2. SKU cross-reference against catalogue
    3. Stock availability check (warns but does not block)
    4. Generate order confirmation with real prices from catalogue
    
    Grounding: prices are always read from catalogue, never from LLM.
    """
    logger.info(f"[Tool] create_order: {order_data}")
    if catalogue_df is None:
        catalogue_df = _load_catalogue()

    try:
        order = Order(**order_data)
    except Exception as e:
        return {"success": False, "error": f"Order validation failed: {str(e)}"}

    confirmed_items = []
    total_amount = 0.0
    warnings = []

    for item in order.items:
        row = catalogue_df[catalogue_df["sku"].str.upper() == item.sku]
        if row.empty:
            return {"success": False, "error": f"SKU '{item.sku}' not found. Order cancelled."}

        product = row.iloc[0]
        stock_available = int(product["stock"])
        price = float(product["price_inr"])
        line_total = price * item.quantity

        if stock_available == 0:
            warnings.append(f"⚠️ {item.sku} is out of stock.")
        elif stock_available < item.quantity:
            warnings.append(f"⚠️ Only {stock_available} units of {item.sku} available (requested {item.quantity}).")

        total_amount += line_total
        confirmed_items.append({
            "sku": item.sku,
            "name": str(product["name"]),
            "quantity": item.quantity,
            "unit_price_inr": price,
            "line_total_inr": line_total,
        })

    order_id = _generate_order_id()
    return {
        "success": True,
        "order_id": order_id,
        "dealer_name": order.dealer_name,
        "status": "CONFIRMED",
        "items": confirmed_items,
        "total_amount_inr": round(total_amount, 2),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "warnings": warnings,
        "message": (
            f"Order {order_id} confirmed for {order.dealer_name}. "
            f"Total: ₹{total_amount:,.2f}. "
            + (f"Notes: {'; '.join(warnings)}" if warnings else "All items in stock.")
        ),
    }


def get_low_stock_alerts(threshold: int = 10, catalogue_df=None) -> List[Dict[str, Any]]:
    """Bonus: Get products with stock at or below threshold, sorted ascending."""
    if catalogue_df is None:
        catalogue_df = _load_catalogue()

    low_stock = catalogue_df[
        (catalogue_df["stock"] > 0) & (catalogue_df["stock"] <= threshold)
    ].sort_values("stock")

    return [
        {
            "sku": str(row["sku"]),
            "name": str(row["name"]),
            "stock": int(row["stock"]),
            "price_inr": float(row["price_inr"]),
            "category": str(row["category"]),
            "alert_level": "CRITICAL" if int(row["stock"]) <= 5 else "WARNING",
        }
        for _, row in low_stock.iterrows()
    ]


# ─────────────────────────────────────────────────────────
# Tool Definitions for Groq Function Calling
# ─────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": (
                "Check current stock availability for a specific part SKU. "
                "Use when user asks about stock levels, availability, or before confirming an order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "Product SKU e.g. BRK-1003"}
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_parts_by_vehicle",
            "description": (
                "Find all compatible parts for a specific vehicle model. "
                "Use when user mentions a vehicle name and wants to see compatible parts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_name": {
                        "type": "string",
                        "description": "Vehicle model e.g. KTM Duke 390, Royal Enfield Meteor 350",
                    }
                },
                "required": ["vehicle_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": (
                "Create a confirmed dealer order. Use ONLY when user explicitly requests to place an order. "
                "Requires dealer name and items list with SKU and quantity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dealer_name": {"type": "string", "description": "Name of dealership"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1},
                            },
                            "required": ["sku", "quantity"],
                        },
                    },
                },
                "required": ["dealer_name", "items"],
            },
        },
    },
]
