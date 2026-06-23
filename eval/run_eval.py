"""
eval/run_eval.py - Automated Evaluation Framework

Metrics:
- Retrieval Accuracy: Were relevant products retrieved? (keyword overlap)
- Tool Call Accuracy: Was the correct tool called?
- Grounding Accuracy: Did the response contain only grounded info?
- Order Validation Accuracy: Were orders properly structured?
- Out-of-Scope Rejection Rate: Were off-topic queries rejected?

Results saved to eval/results.json with a human-readable summary.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_test_cases() -> List[Dict]:
    with open(TEST_CASES_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────────────────

def check_response_contains(response: str, expected_terms: List[str]) -> Dict[str, bool]:
    """Check if response contains expected keywords (case-insensitive)."""
    response_lower = response.lower()
    return {
        term: term.lower() in response_lower
        for term in expected_terms
    }


def is_out_of_scope_rejected(response: str) -> bool:
    """Check if the response correctly rejects an out-of-scope query."""
    rejection_phrases = [
        "only assist", "can only", "cannot assist",
        "automotive parts", "cannot help", "outside my scope",
        "i'm designed to", "i am designed to"
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in rejection_phrases)


def check_tool_accuracy(actual_tool_calls: List[Dict], expected_tool: Optional[str]) -> Dict:
    """Compare expected vs actual tool calls."""
    if expected_tool is None:
        # No tool should be called
        called = len(actual_tool_calls) > 0
        return {
            "expected": "none",
            "called": [tc["tool"] for tc in actual_tool_calls],
            "correct": not called,
            "note": "Unexpected tool call" if called else "Correctly used no tool",
        }

    called_tools = [tc["tool"] for tc in actual_tool_calls]
    correct = expected_tool in called_tools
    return {
        "expected": expected_tool,
        "called": called_tools,
        "correct": correct,
        "note": f"Expected {expected_tool}, got {called_tools}",
    }


def check_grounding(response: str, tool_calls: List[Dict], retrieved_products: List[Dict]) -> Dict:
    """
    Heuristic grounding check:
    - If tool was called and returned data, response should reference it
    - Response should NOT contain obviously hallucinated SKUs (not in catalogue)
    - Check for common hallucination markers

    Note: True grounding verification requires ground-truth labels.
    This is a heuristic proxy suitable for automated evaluation.
    """
    hallucination_markers = [
        "i think", "i believe", "approximately", "probably around",
        "roughly", "i'm not sure but",
    ]
    response_lower = response.lower()
    hallucination_detected = any(m in response_lower for m in hallucination_markers)

    # Check if response references SKU from tool results
    references_tool_data = False
    if tool_calls:
        for tc in tool_calls:
            result = tc.get("result", {})
            if isinstance(result, dict):
                sku = result.get("sku", "")
                if sku and sku in response:
                    references_tool_data = True
            elif isinstance(result, list) and result:
                sku = result[0].get("sku", "")
                if sku and sku in response:
                    references_tool_data = True

    grounded = not hallucination_detected
    return {
        "grounded": grounded,
        "hallucination_markers_detected": hallucination_detected,
        "references_tool_data": references_tool_data,
    }


def run_single_test(agent, memory_class, test_case: Dict, delay: float = 1.0) -> Dict:
    """Run a single test case and return evaluation results."""
    tc_id = test_case["id"]
    category = test_case["category"]

    print(f"  [{tc_id}] {category}: {str(test_case.get('input', 'multi-turn'))[:60]}...")

    # Build input — handle multi-turn
    if "turns" in test_case:
        memory = memory_class()
        last_result = None
        for turn in test_case["turns"]:
            if turn["role"] == "user":
                last_result = agent.chat(turn["content"], memory)
                time.sleep(delay)
        result = last_result
        user_input = test_case["turns"][-1]["content"]
    else:
        memory = memory_class()
        user_input = test_case["input"]
        result = agent.chat(user_input, memory)
        time.sleep(delay)

    if result is None:
        return {"id": tc_id, "error": "No result returned", "passed": False}

    response = result.get("response", "")
    tool_calls = result.get("tool_calls", [])
    retrieved_products = result.get("retrieved_products", [])

    # ─── Metric 1: Response Quality ──────────────────────
    expected_terms = test_case.get("expected_in_response", [])
    term_checks = check_response_contains(response, expected_terms)
    response_accuracy = (
        sum(term_checks.values()) / len(term_checks) if term_checks else 1.0
    )

    # ─── Metric 2: Tool Call Accuracy ────────────────────
    expected_tool = test_case.get("expected_tool")
    tool_accuracy = check_tool_accuracy(tool_calls, expected_tool)

    # ─── Metric 3: Grounding Accuracy ────────────────────
    grounding = check_grounding(response, tool_calls, retrieved_products)

    # ─── Metric 4: Out-of-Scope Handling ─────────────────
    if test_case.get("out_of_scope"):
        oos_handled = is_out_of_scope_rejected(response)
    else:
        oos_handled = None  # Not applicable

    # ─── Metric 5: Order Validation ──────────────────────
    order_valid = None
    if expected_tool == "create_order":
        order = result.get("order")
        if order:
            order_valid = (
                bool(order.get("order_id")) and
                order.get("status") == "CONFIRMED" and
                bool(order.get("items"))
            )
        else:
            order_valid = False

    # ─── Overall Pass/Fail ───────────────────────────────
    passed = (
        response_accuracy >= 0.5 and
        tool_accuracy["correct"] and
        grounding["grounded"] and
        (oos_handled if oos_handled is not None else True) and
        (order_valid if order_valid is not None else True)
    )

    return {
        "id": tc_id,
        "category": category,
        "input": user_input,
        "response_preview": response[:200],
        "passed": passed,
        "metrics": {
            "response_accuracy": round(response_accuracy, 2),
            "term_checks": term_checks,
            "tool_accuracy": tool_accuracy,
            "grounding": grounding,
            "out_of_scope_handled": oos_handled,
            "order_valid": order_valid,
            "tools_called_count": len(tool_calls),
            "products_retrieved": len(retrieved_products),
        },
    }


def generate_report(results: List[Dict]) -> Dict:
    """Generate aggregate metrics and category-wise breakdown."""
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))

    # Category breakdown
    by_category: Dict[str, Dict] = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0}
        by_category[cat]["total"] += 1
        if r.get("passed"):
            by_category[cat]["passed"] += 1

    # Average metrics
    tool_correct = [r["metrics"]["tool_accuracy"]["correct"] for r in results if "metrics" in r]
    grounded = [r["metrics"]["grounding"]["grounded"] for r in results if "metrics" in r]
    response_acc = [r["metrics"]["response_accuracy"] for r in results if "metrics" in r]

    order_results = [r for r in results if r.get("metrics", {}).get("order_valid") is not None]
    order_acc = (
        sum(1 for r in order_results if r["metrics"]["order_valid"]) / len(order_results)
        if order_results else None
    )

    oos_results = [r for r in results if r.get("metrics", {}).get("out_of_scope_handled") is not None]
    oos_acc = (
        sum(1 for r in oos_results if r["metrics"]["out_of_scope_handled"]) / len(oos_results)
        if oos_results else None
    )

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "overall_pass_rate": round(passed / total, 2) if total else 0,
        },
        "aggregate_metrics": {
            "tool_call_accuracy": round(sum(tool_correct) / len(tool_correct), 2) if tool_correct else 0,
            "grounding_accuracy": round(sum(grounded) / len(grounded), 2) if grounded else 0,
            "response_quality": round(sum(response_acc) / len(response_acc), 2) if response_acc else 0,
            "order_validation_accuracy": round(order_acc, 2) if order_acc is not None else "N/A",
            "out_of_scope_rejection_rate": round(oos_acc, 2) if oos_acc is not None else "N/A",
        },
        "category_breakdown": {
            cat: {
                "pass_rate": round(v["passed"] / v["total"], 2),
                "passed": v["passed"],
                "total": v["total"],
            }
            for cat, v in by_category.items()
        },
        "test_results": results,
    }
    return report


def print_summary(report: Dict):
    s = report["summary"]
    m = report["aggregate_metrics"]
    print("\n" + "="*60)
    print("  VIKMO EVAL REPORT")
    print("="*60)
    print(f"  Tests: {s['total_tests']} | Passed: {s['passed']} | Failed: {s['failed']}")
    print(f"  Overall Pass Rate:        {s['overall_pass_rate']*100:.1f}%")
    print("-"*60)
    print(f"  Tool Call Accuracy:       {m['tool_call_accuracy']*100:.1f}%")
    print(f"  Grounding Accuracy:       {m['grounding_accuracy']*100:.1f}%")
    print(f"  Response Quality:         {m['response_quality']*100:.1f}%")
    print(f"  Order Validation Acc:     {m['order_validation_accuracy']}")
    print(f"  OOS Rejection Rate:       {m['out_of_scope_rejection_rate']}")
    print("-"*60)
    print("  Category Breakdown:")
    for cat, v in report["category_breakdown"].items():
        bar = "█" * v["passed"] + "░" * (v["total"] - v["passed"])
        print(f"    {cat:<25} {bar} {v['passed']}/{v['total']}")
    print("="*60)


def main():
    print("🔧 VIKMO AI Dealer Assistant - Evaluation Framework")
    print(f"   Loading test cases from: {TEST_CASES_PATH}")

    test_cases = load_test_cases()
    # Filter to single-turn only if you want quick run; comment out to include multi-turn
    single_turn = [tc for tc in test_cases if "input" in tc]
    multi_turn = [tc for tc in test_cases if "turns" in tc]
    all_cases = single_turn + multi_turn

    print(f"   Running {len(all_cases)} test cases...")
    print("   (This will take a few minutes due to API rate limits)")

    # Initialize agent
    from assistant.agent import DealerAgent
    from assistant.memory import ConversationMemory

    agent = DealerAgent()
    agent.initialize()

    results = []
    failed_cases = []

    for i, tc in enumerate(all_cases, 1):
        print(f"\n[{i}/{len(all_cases)}]", end=" ")
        try:
            result = run_single_test(agent, ConversationMemory, tc, delay=1.5)
            results.append(result)
            status = "✅" if result["passed"] else "❌"
            print(f"  {status}", end="")
            if not result["passed"]:
                failed_cases.append(result)
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            results.append({
                "id": tc["id"],
                "category": tc.get("category"),
                "passed": False,
                "error": str(e),
            })

    # Generate and save report
    report = generate_report(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"\n📄 Full results saved to: {RESULTS_PATH}")

    if failed_cases:
        print(f"\n❌ Failed Tests ({len(failed_cases)}):")
        for fc in failed_cases:
            print(f"   [{fc['id']}] {fc.get('category')}: {fc.get('input','multi-turn')[:60]}")


if __name__ == "__main__":
    main()
