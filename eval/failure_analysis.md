# Failure Analysis — VIKMO AI Dealer Assistant

## Overview

This document analyzes expected failure modes of the VIKMO AI Dealer Assistant based on the system architecture, evaluation design, and known limitations of LLM-based RAG systems.

---

## 1. Retrieval Misses

### Problem
The FAISS semantic search retrieves products based on embedding similarity, not exact keyword match. This can cause misses when:
- User uses highly colloquial or regional terminology (e.g., "disc wafer" instead of "brake rotor")
- Query is very short (e.g., "oil") and lacks sufficient semantic signal
- Rare SKU categories have few training examples in the BGE embedding space

### Example
- Query: "I need wafers for my KTM" → May miss "Brake Rotor Disc Front" (BRK-1006) because "wafer" is not in the description
- Query: "disc" → Too ambiguous; retrieves irrelevant items

### Mitigations (Implemented)
- BGE query prefix: `"Represent this sentence for searching relevant passages: "` improves short query recall
- Context augmentation: vehicle context from memory is appended to retrieval query
- Top-K = 5: Provides coverage even with partial matches

### Future Improvements
- Hybrid search: BM25 keyword search + dense retrieval (Reciprocal Rank Fusion)
- Query expansion: Use LLM to expand short queries before retrieval
- Synonym mapping: "wafer" → "rotor", "rim" → "wheel"

---

## 2. Ambiguous Vehicle Names

### Problem
Indian motorcycle market has many overlapping names:
- "Duke" could mean KTM Duke 200, 250, or 390
- "Pulsar" spans 150, 180, NS200, RS200
- "Apache" could be RTR 160 or RTR 180

The `find_parts_by_vehicle` tool uses AND-matching of tokens, which may:
- **Return too broad results**: "Duke" matches all Duke variants
- **Return empty results**: "KTM Duke V4" (doesn't exist) returns nothing

### Example Failure
- User: "Parts for my Duke" → Tool searches "Duke", returns parts for Duke 200, 250, AND 390 mixed
- User: "KTM 390 adventure" → No match (it's listed as "KTM Duke 390")

### Mitigations (Implemented)
- Broad fallback: If AND-matching returns empty, fall back to OR-matching on first 2 tokens
- Agent clarification prompt: Instructs LLM to ask for specific model when ambiguous

### Future Improvements
- Vehicle taxonomy: Map common informal names to canonical model names
- NER model: Fine-tuned entity recognizer for Indian motorcycle brands/models
- Fuzzy matching: Levenshtein distance for typo tolerance

---

## 3. Hallucination Risks

### Problem
LLM hallucination can manifest as:
- Inventing a product SKU that doesn't exist in the catalogue
- Making up a price or stock level
- Describing a product's compatibility incorrectly

### Observed Mitigation Effectiveness
- **System prompt grounding**: "Use ONLY the information provided in Retrieved Product Context" significantly reduces hallucination in Llama 3.3 70B
- **Tool grounding**: Stock values and prices always come from DataFrame reads, never from LLM generation
- **Low temperature (0.1)**: Reduces creative generation, keeps responses factual

### Residual Risk
- LLM may still paraphrase catalogue descriptions inaccurately
- If retrieved context contains partial information, LLM may extrapolate incorrectly
- Multi-hop reasoning (e.g., "is there a better brake pad than BRK-1001 for track use?") can lead to opinion-based answers

### Future Improvements
- Faithfulness scoring: Use RAGAS or TruLens to measure response vs. context faithfulness
- Factual grounding layer: Post-process LLM output to verify all SKUs/prices against catalogue
- Citation format: Require LLM to cite [Product 1], [Product 2] from context

---

## 4. Tool Routing Mistakes

### Problem
The LLM must decide when to call a tool vs. answer from context. Routing failures include:
- **False tool call**: Calling `check_stock` when user asks a general question about brakes
- **Missing tool call**: Not calling `find_parts_by_vehicle` when vehicle is mentioned
- **Wrong tool**: Calling `create_order` when user just said "I want 2 units" (intent, not confirmed order)

### Example Failures
- User: "Tell me about brake pads" → Should answer from RAG, NOT call check_stock
- User: "I want 5 units of BRK-1001" → Ambiguous: is this an inquiry or an order? Should ask for confirmation
- User: "Order confirmed" → Should call create_order; LLM may just respond in text

### Mitigations (Implemented)
- Tool descriptions are carefully worded: "Use ONLY when user explicitly requests to place an order"
- System prompt: "Always ask for confirmation before creating an order"
- Low temperature: Reduces random tool routing decisions

### Future Improvements
- Intent classifier: Binary classifier to distinguish "inquiry" vs "action" intent
- Two-step confirmation: Always ask user to confirm before order creation
- Tool call validation: If LLM routes wrong tool, detect from result format and retry

---

## 5. Multi-turn Context Degradation

### Problem
As conversation grows longer:
- Token budget consumed by history reduces available context for RAG
- Entity context tracking via regex may miss complex entity mentions
- Old vehicle context may inappropriately influence new queries

### Example
- Turn 1: "Show parts for KTM Duke 390"
- Turn 5: "Now I need something for my Royal Enfield"
- Turn 6: "Check stock" → Memory may still have KTM as vehicle context

### Mitigations (Implemented)
- `MAX_HISTORY_TURNS = 20`: Truncates old history to prevent token overflow
- Explicit entity update after each tool call
- Context summary injected into every system prompt

### Future Improvements
- Summarization: Compress old turns into a summary instead of dropping them
- Entity versioning: Track when vehicle context changes, deprecate old context
- Sliding window with overlap: Keep last N turns + every 5th older turn

---

## 6. Out-of-Scope Boundary Cases

### Problem
Some queries are on the boundary:
- "What is the price of petrol?" → Automotive related but not parts-related
- "How do I install brake pads?" → Automotive but not catalogue-related
- "Is KTM better than Royal Enfield?" → Opinion, not factual catalogue lookup

These may slip through the domain restriction guardrail.

### Mitigations (Implemented)
- Guardrail language: "only answer questions about automotive parts, vehicle fitment, inventory, and dealer orders"
- Explicit examples in prompt: weather, sports are clearly rejected

### Future Improvements
- Topic classifier: Pre-filter queries with a lightweight binary classifier before LLM call
- Confidence thresholding: If retrieval returns low-similarity results, trigger clarification

---

## 7. Summary Table

| Failure Mode | Frequency (Expected) | Severity | Mitigation Status |
|---|---|---|---|
| Retrieval miss (short query) | Medium | Medium | Partial (BGE prefix + context augmentation) |
| Ambiguous vehicle name | High | Low | Partial (clarification prompts) |
| LLM hallucination | Low | High | Strong (temperature + grounding prompt) |
| Tool routing mistake | Low-Medium | Medium | Moderate (careful tool descriptions) |
| Multi-turn context degradation | Low | Low | Partial (truncation + summary) |
| Out-of-scope boundary | Low | Low | Strong (explicit system prompt) |

---

## 8. Future Roadmap

1. **Hybrid Retrieval**: Add BM25 to catch exact keyword matches that dense search misses
2. **RAGAS Evaluation**: Integrate RAGAS for automated faithfulness and context precision scoring
3. **Vehicle Taxonomy**: Build a canonical name → aliases mapping for Indian motorcycle market
4. **Persistent Memory**: Store conversation history to a database for cross-session continuity
5. **Fine-tuned Classifier**: Small BERT-based model for intent detection (inquiry/order/out-of-scope)
6. **Streaming Responses**: Use Groq streaming API for better UX on long responses
