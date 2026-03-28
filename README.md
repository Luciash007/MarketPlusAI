# MarketPulse AI 📊
### Financial Sentiment Intelligence Platform

> An AI-powered GenAI agent that processes aggregated textual data to produce sector-wise and ticker-specific market sentiment summaries — with a live dashboard, divergence alerts, and an RAG-powered chat assistant.

---

## 🏗️ Project Structure

```
project/
│
├── .env                  # API credentials (fill in yours)
├── README.md             # This file
├── knowledge_base.json   # Structured KB: sectors, scenarios, FAQs, guardrail rules
├── index.html            # Landing page + Dashboard + Chat (single-file frontend)
├── main.py               # FastAPI backend + synthetic data generator
├── rag.py                # TF-IDF RAG engine over knowledge_base.json
└── requirements.txt      # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
Edit `.env`:
```
API_KEY=your_azure_openai_api_key
API_ENDPOINT=your_azure_openai_endpoint
```

### 3. Run the server
```bash
python main.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the app
Visit: [http://localhost:8000](http://localhost:8000)

---

## 🧠 Architecture

```
┌─────────────┐   HTTP   ┌───────────────────┐   RAG   ┌─────────────────┐
│  index.html │ ────────▶│     main.py        │ ──────▶│    rag.py        │
│  (Frontend) │          │   (FastAPI)        │         │  (TF-IDF + KB)   │
│  Landing    │          │                    │         └─────────────────┘
│  Dashboard  │          │  ┌─────────────┐  │
│  Chat UI    │          │  │ Data Engine │  │   LLM   ┌─────────────────┐
└─────────────┘          │  │ (Synthetic) │  │ ──────▶│  Azure GPT-4.0   │
                         │  └─────────────┘  │         │  GPT-3.5-turbo   │
                         └───────────────────┘         └─────────────────┘
```

---

## 📡 Data Flow

1. **Synthetic Streaming**: Data generator runs every second, simulating news/social posts for 40+ tickers across 6 sectors
2. **5-Minute Batching**: Every 300 ticks, sector averages are recalculated and saved as timestamped snapshots
3. **Dashboard Polling**: Frontend polls `/api/dashboard` every 3 seconds for live updates
4. **Chat RAG Pipeline**: User query → TF-IDF retrieval from KB → Live data snapshot → LLM prompt → Tier-classified response

---

## 🛡️ 3-Tier Response System

| Tier | Type | Strategy |
|------|------|----------|
| **Tier 1** | In-Scope | Direct answer with score, source count, key themes, confidence |
| **Tier 2** | Out-of-Scope | Polite redirection, offer adjacent in-scope topic |
| **Tier 3** | Prohibited | Hard guardrail + pivot to sentiment data + mandatory disclaimer |

**Prohibited queries**: Investment advice, price predictions, personal financial planning, insider information.

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve index.html |
| GET | `/api/dashboard` | Full dashboard data (sectors, movers, news, alerts) |
| GET | `/api/ticker/{ticker}` | Detailed ticker sentiment |
| GET | `/api/sectors` | Sector-level summary |
| GET | `/api/history` | Historical batch snapshots |
| POST | `/api/chat` | Chat with the AI agent |
| GET | `/health` | System health + RAG stats |

---

## 💡 Features

- ✅ **Real-time streaming** — synthetic data every second, timestamped
- ✅ **5-minute batch windows** — sector averages recalculate automatically
- ✅ **6 sectors, 40+ tickers** — Technology, Energy, Financials, Healthcare, Consumer, Crypto
- ✅ **Divergence Alerts** — flags when news vs social sentiment sharply diverge
- ✅ **RAG + LLM Chat** — grounded responses from knowledge base + live data
- ✅ **3-Tier guardrails** — never provides investment advice
- ✅ **Dark / Light mode** — persistent theme preference
- ✅ **Radar + Bar charts** — Chart.js visualizations
- ✅ **Ticker drill-down** — detailed per-ticker sentiment card
- ✅ **Live news feed** — scrollable, color-coded by sentiment

---

## ⚠️ Disclaimer

> This platform is for **informational and research purposes only** and does not constitute financial advice. Always consult a certified financial advisor before making any investment decisions.

---

## 🤖 Models Used

- **Chat**: `azure/genailab-maas-gpt-4.0` (GPT-4, high quality)
- **Fast**: `azure/genailab-maas-gpt-3.5-turbo` (GPT-3.5, lower latency)