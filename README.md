# VIKMO AI Dealer Assistant

> **Production-quality AI-powered dealer assistant** for an automotive parts catalogue — built with RAG, Tool Calling, Conversation Memory, and Demand Forecasting.

---

## Architecture Overview

```
User Query
    ↓
[Streamlit UI] app.py
    ↓
[Agent Orchestrator] assistant/agent.py
    ├── [RAG Engine] assistant/rag.py         ← FAISS + BGE embeddings
    ├── [Tool Calls] assistant/tools.py        ← Stock / Fitment / Orders
    ├── [Memory]     assistant/memory.py       ← Multi-turn context
    └── [Groq LLM]  llama-3.3-70b-versatile  ← Grounded responses

[Demand Forecasting] forecasting/forecast.py
    ← Holt-Winters + Baselines + MAE/MAPE

[Evaluation]         eval/run_eval.py
    ← 28 test cases, 5 metrics, JSON report
```

---

## Features

| Feature | Description |
|---|---|
| 🔍 Product Search | Semantic RAG over 48 catalogue items |
| 🚗 Vehicle Fitment | Parts lookup by motorcycle model |
| 📦 Stock Check | Real-time stock via tool calling |
| 🛒 Order Creation | Pydantic-validated order schema |
| 🧠 Multi-turn Memory | Vehicle + SKU context carryover |
| 🛡️ Guardrails | Domain restriction + anti-hallucination |
| 📊 Demand Forecasting | Holt-Winters vs baselines per SKU |
| 📋 Evaluation | Automated 5-metric framework |
| 🚨 Low Stock Alerts | Sidebar warnings for stock ≤ threshold |
| 🔄 Alternatives | Out-of-stock → similar product suggestions |
| 📷 Multimodal Vision | Part identification via Groq `llama-4-scout` |

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-username/vikmo-ai-dealer-assistant
cd vikmo-ai-dealer-assistant
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Set API Key

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

Get a free Groq API key at: https://console.groq.com

### 3. Run the Assistant

```bash
streamlit run app.py
```

Opens at: http://localhost:8501

**Note:** First run downloads the BGE embedding model (~130MB) and builds the FAISS index. Subsequent runs load from cache and start in ~2 seconds.

---

## Running Components Individually

### Demand Forecasting

```bash
python forecasting/forecast.py
```

Outputs `forecasting/results.csv` with per-SKU MAE/MAPE for Holt-Winters vs baselines.

### Evaluation Framework

```bash
python eval/run_eval.py
```

Runs 28 test cases, prints a report, and saves full results to `eval/results.json`.

> **Note:** Evaluation makes real API calls. Estimated time: 3-5 minutes. Ensure `GROQ_API_KEY` is set.

---

## Project Structure

```
vikmo-ai-dealer-assistant/
├── README.md               ← You are here
├── DESIGN.md               ← Architecture design document
├── requirements.txt
├── .env.example
├── app.py                  ← Streamlit UI
│
├── assistant/
│   ├── __init__.py
│   ├── rag.py              ← FAISS + BGE RAG engine
│   ├── tools.py            ← Tool definitions + Pydantic schemas
│   ├── agent.py            ← Groq LLM agent + tool calling loop
│   └── memory.py           ← Multi-turn conversation memory
│
├── eval/
│   ├── test_cases.json     ← 28 test cases across 7 categories
│   ├── run_eval.py         ← Automated evaluation framework
│   ├── results.json        ← Generated evaluation report
│   └── failure_analysis.md ← Failure mode analysis
│
├── forecasting/
│   ├── forecast.py         ← Holt-Winters + baselines
│   └── results.csv         ← Generated forecast results
│
└── data/
    ├── catalogue.csv       ← 48 automotive parts
    └── sales_history.csv   ← 20-week sales data for 10 SKUs
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key |
| `LOW_STOCK_THRESHOLD` | No | `10` | Alert threshold for stock levels |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |

---

## Sample Conversations

**Vehicle Fitment:**
```
User: Show brake pads for KTM Duke 390
Bot:  [calls find_parts_by_vehicle("KTM Duke 390")]
      Here are compatible brake pads:
      • BRK-1005 — EBC Rear Disc Brake Pad Set — ₹2,100 — ✅ In Stock: 30
      • BRK-1007 — Brembo Brake Master Cylinder — ₹4,500 — ⚠️ Low Stock: 5
```

**Multi-turn Context:**
```
User: Show brake pads for KTM Duke 200
Bot:  [Returns BRK-1007 options]

User: Check stock
Bot:  [Automatically checks BRK-1007 using memory context]
      BRK-1007 has 5 units remaining — low stock alert!
```

**Order Creation:**
```
User: Place an order for 5 BRK-1001 for ABC Motors
Bot:  [calls create_order({dealer_name: "ABC Motors", items: [{sku: "BRK-1001", qty: 5}]})]
      ✅ Order ORD1001 confirmed for ABC Motors.
      Total: ₹14,250.00 | All items in stock.
```

**Out-of-Scope:**
```
User: Who won IPL 2024?
Bot:  I can only assist with automotive parts, inventory, vehicle fitment and dealer orders.
```

---

## Catalogue Coverage

| Category | Count |
|---|---|
| Brakes | 7 products |
| Tyres | 8 products |
| Engine | 8 products |
| Suspension | 5 products |
| Electrical | 6 products |
| Drivetrain | 4 products |
| Filters | 3 products |
| Accessories | 5 products |
| **Total** | **46 products** |

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM (Chat & Tools)| Groq — Llama 3.3 70B Versatile |
| LLM (Vision) | Groq — Llama 4 Scout 17B Instruct |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS (IndexFlatIP) |
| Validation | Pydantic v2 |
| Forecasting | Statsmodels Holt-Winters |
| UI | Streamlit |
| Data | Pandas, NumPy |

---

## Screenshots

*Run `streamlit run app.py` to see the live interface.*

- **Chat Interface**: ChatGPT-style bubbles with product card panel
- **Sidebar**: Live low-stock alerts, session context, order history
- **Tool Logs**: Collapsible tool call transparency panel
- **Order Cards**: Formatted order confirmations with line items

---

## License

MIT — Built for the VIKMO AI/ML Intern Take-Home Assignment.
