# VIKMO AI Dealer Assistant — Design Document

*Written as a technical interview defense of all architectural decisions.*

---

## 1. Retrieval Architecture

### Why RAG?

The catalogue has 46+ products. Stuffing all of them into every prompt would:
- Consume ~12,000 tokens per request (expensive, slow)
- Reduce LLM focus — too much irrelevant context degrades quality
- Break with larger catalogues (10k+ SKUs)

RAG solves this by retrieving only the top-K relevant products per query, keeping the context window lean and focused.

### Retrieval Flow

```
User Query
    ↓ [BGE prefix prepended]
Embed with BAAI/bge-small-en-v1.5
    ↓
FAISS IndexFlatIP search
    ↓
Top-K=5 CatalogueDocuments
    ↓
Format as structured context block
    ↓
Inject into system prompt
    ↓
LLM generates grounded answer
```

---

## 2. Embedding Choice — BAAI/bge-small-en-v1.5

### Why this model?

| Criterion | BAAI/bge-small-en-v1.5 | text-embedding-ada-002 | all-MiniLM-L6-v2 |
|---|---|---|---|
| Size | 33M params | API only | 22M params |
| BEIR Score | 51.68 | 48.99 | 42.69 |
| Speed | Fast (local) | API latency | Very fast |
| Cost | Free | Pay-per-call | Free |
| Offline | ✅ Yes | ❌ No | ✅ Yes |
| Technical text | ✅ Strong | ✅ Strong | ❌ Weak |

**BGE Small** provides the best retrieval benchmark scores in its size class, with zero API cost and full offline capability. It's specifically designed for asymmetric retrieval (short queries vs. longer passages).

### BGE Query Prefix

BGE models require a prefix on queries (but NOT on passages):
```python
query = "Represent this sentence for searching relevant passages: " + user_query
```

This prefix conditions the encoder to produce query-space embeddings rather than passage-space embeddings, significantly improving retrieval accuracy on short queries.

---

## 3. FAISS Design

### Index Type: IndexFlatIP

- **Flat**: Exact search — no approximation. Every query compares against all vectors.
- **IP**: Inner Product similarity. Since all vectors are L2-normalized at index time, IP = cosine similarity.

### Why Flat (not HNSW/IVF)?

Our catalogue has ~50 products. At this scale:
- HNSW overhead exceeds its benefit (optimized for 100k+ vectors)
- Flat search over 50 vectors takes < 1ms
- Exact search guarantees perfect recall

**Upgrade path**: For 10k+ products, switch to `IndexHNSWFlat` (M=32, efConstruction=200) for sub-linear search time while maintaining > 99% recall.

### Persistence

Index is persisted to `.cache/faiss.index` and documents to `.cache/documents.pkl`. This avoids re-embedding on every startup. Cache is invalidated when the number of documents changes.

---

## 4. Tool Calling Design

### Three-Tool Architecture

| Tool | Trigger | Grounding Source |
|---|---|---|
| `check_stock` | "Is X in stock?", "Check availability" | DataFrame read (exact) |
| `find_parts_by_vehicle` | Vehicle model mentioned | DataFrame filter (exact) |
| `create_order` | "Place order", "Order X units" | DataFrame prices (exact) |

### Why Not Use LLM for Stock/Price?

LLMs hallucinate numerical values. A stock level or price returned by an LLM is **unverifiable**. Tools are deterministic functions that read directly from the source DataFrame — there is no possible hallucination path for these values.

### Pydantic Validation on Orders

```python
class Order(BaseModel):
    dealer_name: str
    items: List[OrderItem]  # Each item validated for SKU format + quantity range

    @model_validator(mode="after")
    def validate_unique_skus(self): ...
```

If the LLM generates malformed JSON arguments (wrong SKU format, negative quantity), Pydantic raises a `ValidationError`. We return the error message to the LLM for self-correction.

### Tool Call Loop

```
LLM → tool_call → execute → tool_result → LLM → final_response
```

Maximum 3 iterations to prevent infinite loops. After MAX_TOOL_ITERATIONS, we request a final non-tool response.

---

## 5. Conversation Memory

### Entity Tracking Strategy

Rather than relying solely on full conversation history (which gets expensive as conversations grow), we maintain an explicit `EntityContext`:

```python
@dataclass
class EntityContext:
    vehicle: Optional[str]
    last_mentioned_skus: List[str]
    last_mentioned_category: Optional[str]
    dealer_name: Optional[str]
```

**Why explicit tracking?**
- Resolves pronouns: "Check stock" → looks up last-mentioned SKU
- Survives context truncation: entity context is always injected into every prompt
- Efficient: no expensive summarization needed for simple entity carryover

### Context Injection

Every LLM call includes a context summary line in the system prompt:
```
Current vehicle context: KTM Duke 390 | Recently discussed SKUs: BRK-1005, BRK-1007
```

This gives the LLM the key facts from memory without requiring it to read all previous turns.

### Truncation

When conversation exceeds `MAX_HISTORY_TURNS=20`, older messages are dropped (most recent messages kept). This prevents token budget exhaustion while preserving the most relevant recent context.

---

## 6. Prompt Strategy

### System Prompt Layers

```
SYSTEM PROMPT =
  [Role Definition]           → "You are VIKMO Assistant..."
  [Domain Restriction]        → "Only automotive parts..."
  [Grounding Rules]           → "Use ONLY the context below..."
  [Tool Usage Rules]          → "Call check_stock when..."
  [Clarification Rules]       → "Ask for vehicle model when ambiguous..."
  [Context Summary]           → Current vehicle, SKUs from memory
  [Retrieved Product Context] → RAG results for this query
```

### Temperature = 0.1

Low temperature is critical for:
- Consistent tool routing decisions
- Factual responses (less creative generation = less hallucination)
- Deterministic order formatting

High temperature would cause unpredictable tool routing and numerical hallucination.

### Why Separate System Prompt from History?

The system prompt is rebuilt every turn (with updated context and new RAG results). This ensures the LLM always has the freshest retrieved context — a key correctness requirement for RAG systems.

---

## 7. Grounding Strategy

### Three-Layer Grounding

1. **Prompt-level**: System prompt explicitly forbids answering outside provided context
2. **Tool-level**: Stock, price, and order data come from DataFrame reads only
3. **Temperature-level**: 0.1 temperature minimizes creative/hallucinated generation

### What Can Still Hallucinate?

- Paraphrasing of catalogue descriptions
- Compatibility claims for vehicles not in the fitment field
- Superlative comparisons ("best brake pad for track")

### Anti-hallucination Prompt Phrases

```
"If information is not found, say: 'I could not find matching catalogue information.'"
"NEVER invent product names, SKUs, prices, or stock values."
"Use ONLY the information provided in the 'Retrieved Product Context' below."
```

---

## 8. Evaluation Methodology

### Five Metrics

| Metric | Definition | How Measured |
|---|---|---|
| Tool Call Accuracy | Was correct tool called? | Expected tool vs actual tool calls |
| Response Quality | Were expected keywords in response? | Term overlap score |
| Grounding Accuracy | No hallucination markers detected? | Phrase matching heuristic |
| Order Validation | Was order structured correctly? | Order schema validation |
| OOS Rejection Rate | Were off-topic queries rejected? | Rejection phrase detection |

### Limitations

The current evaluation is **heuristic-based** (keyword matching, phrase detection). Production-grade evaluation would use:
- **RAGAS**: Faithfulness, context precision, context recall
- **Human evaluation**: Adversarial queries, edge cases
- **LLM-as-judge**: Use GPT-4 to judge response quality

---

## 9. Forecasting Methodology

### Leakage Prevention (Critical)

```python
train = sku_data.iloc[:-TEST_WEEKS]  # All rows except last 4
test  = sku_data.iloc[-TEST_WEEKS:]  # Last 4 weeks ONLY
```

Models are fit ONLY on `train`. Test data is never used for fitting, parameter tuning, or normalization. This mirrors real-world deployment where future data is unavailable.

### Three Models

| Model | Mechanism | Best For |
|---|---|---|
| Last Value | Repeat last observation | Random walks, stationary |
| Moving Average (k=4) | Average of last 4 weeks | Stable, low-variance series |
| Holt-Winters (additive, damped) | Level + trend with dampening | Trending series |

### Why Holt-Winters?

Our sales data shows a clear upward trend (typical for growing market). Holt-Winters captures:
- **Level**: Current baseline demand
- **Trend**: Week-over-week growth
- **Damped**: Prevents over-extrapolation of the trend into the future

No seasonal component is used — weekly data with only 20 observations lacks sufficient history to estimate a yearly seasonal cycle reliably.

### Metrics: MAE vs MAPE

- **MAE** (Mean Absolute Error): Units of the original series, interpretable as "average weekly forecast error in units"
- **MAPE** (Mean Absolute Percentage Error): Scale-free, allows comparison across SKUs with different demand magnitudes

---

## 10. Failure Analysis

See [eval/failure_analysis.md](eval/failure_analysis.md) for comprehensive failure mode analysis.

Key failure modes:
1. Retrieval miss on colloquial terms → mitigated by hybrid search roadmap
2. Ambiguous vehicle names → mitigated by clarification prompts
3. Hallucination risk → mitigated by grounding triplex (prompt + tools + temperature)
4. Tool routing mistakes → mitigated by careful tool descriptions
5. Context degradation → mitigated by entity tracking + truncation

---

## 11. Bonus Features Design

### Low Stock Alerts
- Threshold-based filter on `stock <= N` (configurable via `.env`)
- CRITICAL (≤5) vs WARNING (≤10) tiers
- Rendered in sidebar on every page load

### Recommended Alternatives
- Triggered when `check_stock` returns `available: False`
- Same-category, in-stock products sorted by stock descending
- Top 3 alternatives returned alongside stock check result

### Order History
- Stored in `st.session_state.orders` (session-scoped)
- Displayed in sidebar as scrollable list
- Persisted in `ConversationMemory.order_history` for multi-turn reference

### Image-Based Part Identification
- Pluggable architecture: `st.file_uploader` in sidebar
- When Groq releases vision-capable API endpoints (llava-v1.5 or similar), swap in by changing the model ID
- Current implementation prompts user to describe the part in chat as fallback
