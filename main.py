"""
main.py — MarketPulse AI: Financial Sentiment Analysis API (Enhanced v2)

Changes from original:
  - Imports data_context.py for CSV + RSS structured data
  - _build_system_prompt() now takes data_context as 3rd argument
  - /api/chat injects build_data_context(query) into every LLM call
  - /api/data-stats (new route) shows row counts for all data sources
  - Strict tier classification: out-of-scope and forbidden questions
    get clean refusals — NO data dump, NO disclaimer on non-finance replies
"""

import os
import json
import re
import random
import asyncio
import warnings
import httpx
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from collections import deque

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from openai import OpenAI

from rag import retrieve as rag_retrieve, get_stats as rag_stats
from data_context import build_data_context, get_summary_stats   # ← NEW

BASE_DIR = Path(__file__).parent


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_data()
    task = asyncio.create_task(_data_generator_loop())
    print("✅ Synthetic data generator started")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MarketPulse AI — Financial Sentiment API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_landing():
    return FileResponse(str(BASE_DIR / "index.html"))


# ── Model registry ─────────────────────────────────────────────────────────────
CHAT_MODEL = "genailab-maas-gpt-4o"
FAST_MODEL = "genailab-maas-gpt-35-turbo"


def get_client():
    api_key  = os.environ.get("API_KEY", "").strip()
    endpoint = os.environ.get("API_ENDPOINT", "").strip()
    if not api_key or not endpoint:
        return None
    base = endpoint.rstrip("/")
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base,
            http_client=httpx.Client(verify=False, timeout=30.0),
        )
        print(f"✅ OpenAI client ready → base_url={base}")
        return client
    except Exception as e:
        print(f"⚠ OpenAI client init error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TIER CLASSIFICATION  ← Enhanced with comprehensive keyword lists
# ─────────────────────────────────────────────────────────────────────────────

# Tier 3: Forbidden — investment advice, price prediction, insider info
TIER3_KEYWORDS = [
    "should i buy", "should i sell", "when to buy", "when to sell",
    "invest my", "invest for retirement", "price prediction", "price target",
    "will it go up", "will it go down", "will it reach", "will it hit",
    "insider", "leaked info", "leaked information", "non-public",
    "tell me when to", "how much to invest", "portfolio advice",
    "is it a good time to buy", "is it a good time to sell",
    "should i hold", "should i exit", "entry point", "exit point",
    "where should i put my money", "best stock to buy",
    "guaranteed return", "safe investment",
]

# Tier 2: Out-of-scope — general knowledge, politics, geography, sports, weather, war,
#          personal/emotional statements, greetings, random chit-chat, etc.
TIER2_KEYWORDS = [
    # ── Personal / emotional statements ──────────────────────────────────────
    "i am very sad", "i am sad", "i feel sad", "i'm sad", "i'm very sad",
    "i am happy", "i feel happy", "i'm happy", "i am excited", "i feel excited",
    "i am depressed", "i feel depressed", "i am anxious", "i feel anxious",
    "i am angry", "i feel angry", "i am frustrated", "i feel frustrated",
    "i am tired", "i feel tired", "i am bored", "i feel bored",
    "i am lonely", "i feel lonely", "i am stressed", "i feel stressed",
    "i am worried", "i feel worried", "i am scared", "i feel scared",
    "i am confused", "i feel confused", "i am lost", "i feel lost",
    "i am crying", "i want to cry", "i can't stop crying",
    "i hate", "i love", "i miss", "i wish", "i hope",
    "how are you", "how r u", "what are you", "who are you", "are you a robot",
    "are you an ai", "are you human", "what can you do", "what do you do",
    "tell me a joke", "tell me a story", "can you help me",
    "good morning", "good evening", "good night", "hello", "hi there",
    "my life", "my day", "my problem", "my girlfriend", "my boyfriend",
    "my family", "my job", "my career", "i got fired", "i lost my job",
    "i broke up", "breakup", "relationship", "personal life",
    # Personal trading regret / loss (NOT financial analysis)
    "i lost gold", "i lost silver", "i lost money on", "i lost everything",
    "i sold at", "i bought at wrong", "i sold at lower", "i lost by selling",
    "i lost by buying", "selling at lower price", "bought at higher price",
    "sold at lower price", "my trade went wrong", "my investment failed",
    "my portfolio is down", "i am in loss", "i am ruined", "i am broke",
    "i regret buying", "i regret selling", "i should have sold",
    # ── Geography / politics / government ────────────────────────────────────
    "capital of", "cm of", "chief minister", "prime minister", "president of",
    "governor of", "mayor of", "election", "who won the election",
    "who is the", "what is the population", "which country",
    # ── War / geopolitics (unless market-impact framing) ─────────────────────
    "war update", "latest war", "military strike", "troops", "soldiers",
    "why is usa", "why is us", "why is america", "why is iran", "why is russia",
    "why is china", "why is india", "geopolitical reason", "political reason",
    "why did", "history of", "explain the conflict", "what started the war",
    "latest update on war", "latest news on war", "iran nuclear",
    "nato", "un resolution", "sanctions on iran", "sanctions on russia",
    # ── Weather ───────────────────────────────────────────────────────────────
    "weather", "temperature", "rain", "forecast", "climate today",
    # ── Sports ────────────────────────────────────────────────────────────────
    "sports score", "who won the game", "cricket score", "ipl score",
    "football score", "match result",
    # ── Entertainment / general knowledge ────────────────────────────────────
    "recipe", "cooking", "movie review", "song lyrics",
    "who is the cm", "who is the pm", "who is the president",
    "what is the capital", "population of",
    # ── Insults / nonsense ────────────────────────────────────────────────────
    "shut up", "you are stupid", "you are dumb", "you are useless",
    "you suck", "i hate you", "go away",
]

# Keywords that indicate a MARKET-IMPACT framing (override tier2 for war/geopolitics)
MARKET_IMPACT_KEYWORDS = [
    "market impact", "impact on market", "impact on stocks", "impact on oil",
    "impact on energy", "effect on market", "effect on stocks",
    "how does", "how will", "market reaction", "oil price", "energy sector",
    "sector sentiment", "stock market", "equity market", "commodity",
]

# Keywords that are always Tier 1 (Indian market / index queries)
ALWAYS_TIER1_KEYWORDS = [
    "nifty", "nifty 50", "nifty50", "sensex", "bse", "nse", "nse india",
    "nifty bank", "bank nifty", "nifty it", "nifty auto", "nifty pharma",
    "nifty metal", "nifty fmcg", "nifty realty", "nifty energy",
    "indian market", "india market", "dalal street", "bombay stock",
    "national stock exchange", "bombay stock exchange",
    "tcs sentiment", "infy sentiment", "wipro sentiment",
    "reliance sentiment", "hdfc sentiment", "icici sentiment",
]


# ── Whitelist: only these topic keywords are in-scope for MarketPulse AI ────
TIER1_WHITELIST = [
    # Stocks / tickers
    "stock", "share", "equity", "ticker", "scrip", "listed", "ipo",
    "market cap", "pe ratio", "eps", "dividend", "earnings", "revenue",
    "quarterly", "annual report", "filing", "nse", "bse", "nyse", "nasdaq",
    # Indices
    "nifty", "sensex", "nifty 50", "nifty50", "bank nifty", "banknifty",
    "midcap", "smallcap", "large cap", "nifty it", "nifty auto", "nifty pharma",
    "dow jones", "s&p", "s&p 500", "nasdaq", "ftse", "dax", "cac",
    # Sentiment / analysis
    "sentiment", "bullish", "bearish", "positive", "negative", "neutral",
    "trend", "momentum", "divergence", "signal", "alert", "analysis",
    "score", "confidence", "news score", "social score",
    # Sectors
    "sector", "technology", "energy", "financials", "healthcare", "consumer",
    "crypto", "metals", "commodities", "automobile", "auto", "education",
    "mining", "alcohol", "tobacco", "refinery", "refineries", "telecom",
    "logistics", "power", "water", "utilities", "fmcg", "pharma", "banking",
    "realty", "infrastructure", "it sector", "tech sector",
    # Companies / tickers (Indian)
    "tcs", "infosys", "wipro", "hdfc", "reliance", "icici", "sbi", "axis",
    "kotak", "bajaj", "maruti", "tatamotors", "tata", "hcl", "techm",
    "l&t", "ongc", "ntpc", "powergrid", "adani", "ambani", "itc",
    "bhartiartl", "airtel", "jio", "sun pharma", "dr reddy", "cipla",
    # Companies (Global)
    "aapl", "apple", "msft", "microsoft", "nvda", "nvidia", "googl", "google",
    "meta", "amazon", "amzn", "tsla", "tesla", "amd", "intc", "intel",
    "jpmorgan", "goldman", "morgan stanley", "jp morgan",
    "gold", "silver", "oil", "crude", "bitcoin", "btc", "ethereum", "eth",
    # Financial terms
    "ipo", "fii", "dii", "mutual fund", "etf", "bond", "yield", "interest rate",
    "fed", "rbi", "monetary policy", "inflation", "gdp", "fiscal",
    "market sentiment", "market mood", "market pulse", "market analysis",
    "price action", "support", "resistance", "breakout", "volume",
    # RSS / news context
    "news", "announcement", "press release", "earnings call", "analyst",
    "upgrade", "downgrade", "rating", "target price", "recommendation",
    "corporate", "quarterly results", "balance sheet",
    # General market terms
    "market", "markets", "how are markets", "market today", "market now",
    "how is market", "market update", "market overview", "market summary",
    "market analysis", "market data", "stock market", "financial market",
    "trading", "trade", "investor", "portfolio sentiment",
]


def _tier_classify(message: str) -> str:
    """
    3-Tier classification:
      tier3 — Prohibited (investment advice / price prediction / insider info)
      tier2 — Out-of-scope (anything not related to financial markets / sentiment)
      tier1 — In-scope (market sentiment / stock / sector / financial analysis)

    Uses WHITELIST approach: if none of the finance-related keywords match,
    it is automatically Tier 2 (out-of-scope), preventing food / personal /
    general queries from being treated as financial questions.
    """
    msg = message.lower().strip()

    # ── Step 1: Always Tier 1 — explicit Indian market index queries ──────────
    if any(k in msg for k in ALWAYS_TIER1_KEYWORDS):
        return "tier1"

    # ── Step 2: Tier 3 — prohibited queries ───────────────────────────────────
    if any(k in msg for k in TIER3_KEYWORDS):
        return "tier3"

    # ── Step 3: Tier 2 — explicit out-of-scope keyword match ─────────────────
    is_oos = any(k in msg for k in TIER2_KEYWORDS)
    if is_oos:
        has_market_framing = any(k in msg for k in MARKET_IMPACT_KEYWORDS)
        if not has_market_framing:
            return "tier2"

    # ── Step 4: WHITELIST check — must contain at least one finance keyword ───
    # If the message has NO finance-related keyword → Tier 2 (out-of-scope)
    has_finance_keyword = any(k in msg for k in TIER1_WHITELIST)
    if not has_finance_keyword:
        return "tier2"

    return "tier1"


def _tier3_response(message: str) -> str:
    """Hard refusal for forbidden queries — no data dump, no disclaimer."""
    msg = message.lower()
    if "price" in msg or "prediction" in msg or "will it" in msg:
        return (
            "I'm not able to provide price predictions or forecasts. "
            "MarketPulse AI is a sentiment analysis tool only — it reads current market mood, "
            "not future prices.\n\n"
            "I can show you the **current sentiment score** for any ticker or sector instead. "
            "Just ask: *\"What is the current sentiment for Bitcoin?\"*"
        )
    if "should i buy" in msg or "should i sell" in msg or "should i hold" in msg:
        return (
            "I can't provide buy/sell/hold recommendations. "
            "Investment decisions require personalised financial advice from a certified advisor.\n\n"
            "What I *can* do is show you the current market sentiment data for any stock or sector. "
            "Try asking: *\"What is the sentiment for NVIDIA right now?\"*"
        )
    if "insider" in msg or "leaked" in msg:
        return (
            "I don't have access to non-public or insider information, "
            "and sharing such information would be illegal.\n\n"
            "I only analyse publicly available news and social media sentiment. "
            "Would you like to see current public sentiment for a specific stock?"
        )
    if "invest" in msg or "portfolio" in msg or "retirement" in msg:
        return (
            "Personal financial planning and investment advice are outside my scope. "
            "Please consult a SEBI-registered financial advisor or certified financial planner (CFP).\n\n"
            "I can help you explore current market sentiment across sectors like Technology, "
            "Financials, or Healthcare. Just ask!"
        )
    # Generic tier3 fallback
    return (
        "That type of question is outside what I can help with. "
        "MarketPulse AI provides market sentiment analysis only — not investment advice, "
        "price predictions, or personal financial planning.\n\n"
        "Try asking about current sentiment for a stock, sector, or market trend."
    )


def _tier2_response(message: str) -> str:
    """Polite redirect for out-of-scope queries — no data dump."""
    msg = message.lower()

    # ── Emotional / personal statements ──────────────────────────────────────
    EMOTION_WORDS = [
        "i am sad", "i'm sad", "i feel sad", "i am very sad", "i'm very sad",
        "i am happy", "i feel happy", "i am excited", "i feel excited",
        "i am depressed", "i feel depressed", "i am anxious", "i am angry",
        "i am frustrated", "i am tired", "i am bored", "i am lonely",
        "i am stressed", "i am worried", "i am scared", "i am confused",
        "i am crying", "i want to cry", "i hate", "i love", "i miss",
        "my life", "my day", "my problem", "my girlfriend", "my boyfriend",
        "my family", "my job", "my career", "i got fired", "i lost my job",
        "i broke up", "breakup", "relationship", "personal life",
    ]
    if any(k in msg for k in EMOTION_WORDS):
        return (
            "I hear you, and I hope you feel better soon. 💙\n\n"
            "I'm MarketPulse AI — a financial market sentiment assistant, "
            "so personal and emotional topics are outside what I can help with.\n\n"
            "If you'd like to talk to someone, please reach out to a trusted friend, "
            "family member, or a mental health professional.\n\n"
            "When you're ready, I'm here to help you with market and stock sentiment analysis!"
        )

    # ── Greetings / chit-chat ─────────────────────────────────────────────────
    GREET_WORDS = [
        "how are you", "how r u", "what are you", "who are you",
        "are you a robot", "are you an ai", "are you human",
        "good morning", "good evening", "good night", "hello", "hi there",
        "tell me a joke", "tell me a story", "what can you do",
    ]
    if any(k in msg for k in GREET_WORDS):
        return (
            "Hi there! 👋 I'm MarketPulse AI — a financial market sentiment assistant.\n\n"
            "I specialise in analysing live market sentiment across stocks, sectors, and news. "
            "Here's what I can help you with:\n"
            "• *\"What is the current sentiment for NVIDIA / TCS / WIPRO?\"*\n"
            "• *\"Which sectors are bullish today?\"*\n"
            "• *\"Show me divergence alerts between news and social media\"*\n"
            "• *\"What are the latest NSE announcements for Infosys?\"*\n\n"
            "What would you like to explore?"
        )

    # ── Insults / rude messages ───────────────────────────────────────────────
    RUDE_WORDS = ["shut up", "you are stupid", "you are dumb", "you are useless",
                  "you suck", "i hate you", "go away"]
    if any(k in msg for k in RUDE_WORDS):
        return (
            "I'm here to help with financial market sentiment analysis. "
            "Feel free to ask me about stocks, sectors, or market trends anytime!"
        )

    # ── War / geopolitics ─────────────────────────────────────────────────────
    if any(k in msg for k in ["war", "iran", "russia", "ukraine", "military", "troops",
                                "why is usa", "why is us", "why is america", "nato",
                                "sanctions", "conflict", "nuclear", "geopolit"]):
        return (
            "I'm MarketPulse AI — a financial market sentiment assistant. "
            "I'm not able to provide geopolitical analysis or war updates.\n\n"
            "However, I *can* tell you how geopolitical events are affecting markets. "
            "Try asking:\n"
            "• *\"What is the impact of Middle East tensions on oil/energy sector sentiment?\"*\n"
            "• *\"How are geopolitical risks affecting commodity markets?\"*\n"
            "• *\"What is the current sentiment for the Energy sector?\"*"
        )

    # ── Government / politics ─────────────────────────────────────────────────
    if any(k in msg for k in ["cm of", "chief minister", "prime minister", "president of",
                                "governor", "mayor", "election", "who is the"]):
        return (
            "I'm MarketPulse AI — a financial market sentiment assistant. "
            "Questions about political figures or government are outside my scope.\n\n"
            "I can help you with financial market analysis. For example:\n"
            "• *\"What is the sentiment for Indian IT stocks like TCS, INFY, WIPRO?\"*\n"
            "• *\"How is the NSE sector performing today?\"*"
        )

    # ── Weather ───────────────────────────────────────────────────────────────
    if any(k in msg for k in ["weather", "temperature", "rain", "forecast"]):
        return (
            "I'm a financial sentiment assistant — weather queries are outside my scope. "
            "Try a weather app for that!\n\n"
            "I can show you how climate-related news is affecting "
            "Energy or Agriculture sector sentiment if you're interested."
        )

    # ── Sports ────────────────────────────────────────────────────────────────
    if any(k in msg for k in ["sports score", "cricket", "ipl", "football", "match result",
                                "who won the game"]):
        return (
            "Sports scores are outside my scope — I'm a financial market sentiment assistant.\n\n"
            "I can show you sentiment for sports-related stocks like Nike ($NKE) "
            "or Disney ($DIS) if you're interested!"
        )

       # Generic out-of-scope (food, random, personal, unrelated queries)
    return (
        "I cannot answer this type of question. "
        "I am MarketPulse AI — a financial market sentiment analysis agent. "
        "Please ask me about stocks, sectors, indices, or market sentiment."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA ENGINE  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

SECTORS = {
    "Technology": {
        "id": "TECH",
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "INTC"],
        "color": "#6366f1",
        "base_sentiment": 0.45
    },
    "Energy": {
        "id": "ENERGY",
        "tickers": ["XOM", "CVX", "BP", "NEE", "ENPH", "OXY"],
        "color": "#f59e0b",
        "base_sentiment": -0.1
    },
    "Financials": {
        "id": "FIN",
        "tickers": ["JPM", "GS", "BAC", "MS", "WFC", "V", "MA"],
        "color": "#10b981",
        "base_sentiment": 0.25
    },
    "Healthcare": {
        "id": "HEALTH",
        "tickers": ["JNJ", "PFE", "UNH", "MRNA", "ABBV", "LLY"],
        "color": "#06b6d4",
        "base_sentiment": 0.15
    },
    "Consumer": {
        "id": "CONS",
        "tickers": ["AMZN", "TSLA", "WMT", "HD", "MCD", "NKE"],
        "color": "#f43f5e",
        "base_sentiment": 0.1
    },
    "Crypto": {
        "id": "CRYPTO",
        "tickers": ["BTC", "ETH", "SOL", "COIN", "MSTR"],
        "color": "#a855f7",
        "base_sentiment": -0.2
    },
    "Metals & Commodities": {
        "id": "METALS",
        "tickers": ["GOLD", "SILVER", "PLATINUM", "COPPER", "PALLADIUM", "GLD", "SLV", "GDX", "NEM", "WPM"],
        "color": "#eab308",
        "base_sentiment": 0.3
    },
    "Automobile": {
        "id": "AUTO",
        "tickers": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "M&M", "EICHERMOT", "ASHOKLEY", "TVSMOTORS"],
        "color": "#3b82f6",
        "base_sentiment": 0.2
    },
    "Education": {
        "id": "EDU",
        "tickers": ["NAUKRI", "CAREEREDGE", "APTECH", "NIITLTD", "NAVNETEDUL", "ZEEL", "MTSEDUCARE"],
        "color": "#8b5cf6",
        "base_sentiment": 0.15
    },
    "Mining": {
        "id": "MINING",
        "tickers": ["COALINDIA", "NMDC", "VEDL", "HINDALCO", "NATIONALUM", "GMRINFRA", "MOIL"],
        "color": "#78716c",
        "base_sentiment": 0.1
    },
    "Alcohol & Tobacco": {
        "id": "ALCTOB",
        "tickers": ["UBL", "RADICO", "MCDOWELL-N", "VSTIND", "GODFRYPHLP", "ITC"],
        "color": "#dc2626",
        "base_sentiment": 0.05
    },
    "Refineries": {
        "id": "REFINE",
        "tickers": ["RELIANCE", "IOC", "BPCL", "HPCL", "MRPL", "CPCL", "CHENNPETRO"],
        "color": "#ea580c",
        "base_sentiment": 0.1
    },
    "Telecom & Infra": {
        "id": "TELECOM",
        "tickers": ["BHARTIARTL", "IDEA", "TATACOMM", "INDUS", "INFRATEL", "GTLINFRA", "RELIANCE"],
        "color": "#0891b2",
        "base_sentiment": 0.2
    },
    "Logistics": {
        "id": "LOGISTICS",
        "tickers": ["BLUEDART", "DELHIVERY", "CONCOR", "GATI", "MAHLOG", "TCI", "ALLCARGO"],
        "color": "#059669",
        "base_sentiment": 0.15
    },
    "Power": {
        "id": "POWER",
        "tickers": ["NTPC", "POWERGRID", "ADANIPOWER", "TATAPOWER", "CESC", "NHPC", "TORNTPOWER"],
        "color": "#d97706",
        "base_sentiment": 0.25
    },
    "Water & Utilities": {
        "id": "WATER",
        "tickers": ["WABAG", "GMRINFRA", "KRBL", "ION", "OPTIEMUS", "SUPREMEIND", "FLUOROCHEM"],
        "color": "#0284c7",
        "base_sentiment": 0.1
    },
}

METALS_NAMES = {
    "GOLD":      "Gold (XAU/USD)",
    "SILVER":    "Silver (XAG/USD)",
    "PLATINUM":  "Platinum (XPT/USD)",
    "COPPER":    "Copper (HG)",
    "PALLADIUM": "Palladium (XPD/USD)",
    "GLD":       "SPDR Gold ETF",
    "SLV":       "iShares Silver ETF",
    "GDX":       "VanEck Gold Miners ETF",
    "NEM":       "Newmont Corp",
    "WPM":       "Wheaton Precious Metals",
}

NEWS_TEMPLATES = {
    "positive": [
        "{ticker} Q1 2026 earnings beat Wall Street estimates — EPS $2.14 vs $1.98 expected",
        "{ticker} announces $8B AI infrastructure investment, shares surge 4.2%",
        "S&P upgrades {ticker} to 'Overweight' — price target raised to record high",
        "{ticker} reports 18% YoY revenue growth driven by cloud and AI segment expansion",
        "Fed pivot optimism lifts {ticker} — rate-sensitive growth name outperforms sector",
        "{ticker} secures landmark government contract worth $3.2B over 5 years",
        "Institutional investors increase {ticker} position by 12% in latest 13F filings",
        "{ticker} operating margins expand 220bps YoY — cost discipline paying off",
        "Morgan Stanley raises {ticker} target — AI monetisation ahead of schedule",
        "{ticker} free cash flow hits record $4.1B in Q4 2025, dividend hiked 20%",
        "Goldman Sachs adds {ticker} to Conviction Buy List citing strong moat",
        "{ticker} international expansion gains traction — Asia-Pacific revenue up 31%",
    ],
    "negative": [
        "{ticker} Q1 2026 revenue misses consensus by 6% — guidance cut for full year",
        "{ticker} CEO departure triggers sell-off; board cites strategic disagreement",
        "Margin compression hits {ticker} — gross margins fall 180bps amid cost pressures",
        "Regulatory probe into {ticker} pricing practices weighs on shares",
        "{ticker} faces activist pressure as hedge fund builds 5.2% stake, demands changes",
        "Moody's places {ticker} on negative credit watch — leverage concerns cited",
        "Short interest in {ticker} rises to 8.4% of float — highest level in 18 months",
        "{ticker} loses key enterprise customer worth $400M annually to competitor",
        "JPMorgan downgrades {ticker} to Underweight — valuation stretched at 28x forward PE",
        "{ticker} workforce reduction of 8% announced; restructuring charges of $1.2B",
        "Supply chain bottlenecks continue to pressure {ticker} — lead times extended to 20 weeks",
        "Consumer demand slowdown hits {ticker} — same-store sales fall 3.4% in March 2026",
    ],
    "neutral": [
        "{ticker} Q1 2026 results in line with expectations — full-year guidance maintained",
        "{ticker} completes previously announced $2.1B acquisition of analytics startup",
        "Analyst Day: {ticker} management reaffirms 5-year roadmap, no major surprises",
        "{ticker} CFO presents at Deutsche Bank tech conference — steady-state commentary",
        "{ticker} board approves $1B share repurchase programme over 12 months",
        "{ticker} prices secondary offering at $0.50 discount to market; shares flat",
        "Credit Suisse initiates coverage of {ticker} at Neutral with $185 target",
        "{ticker} joint venture with local partner in India enters pilot phase",
        "Management transition at {ticker} complete — incoming CEO has strong track record",
        "{ticker} annual report highlights ESG progress — net-zero target moved up to 2040",
    ],
}

SOCIAL_TEMPLATES = {
    "positive": [
        "Finally, ${ticker} is getting the recognition it deserves 🚀 AI cycle just getting started",
        "Loaded more ${ticker} on the dip. Q1 numbers were incredible. This is a multi-year hold.",
        "${ticker} chart is a thing of beauty — breakout confirmed, volume backing it up 📈",
        "Institutional accumulation in ${ticker} is screaming buy. 13F filings don't lie.",
        "Rate cut cycle = massive tailwind for ${ticker}. Fed has your back, bulls. 🐂",
        "Just listened to ${ticker} earnings call. Management is executing flawlessly. Adding here.",
        "${ticker} just had the cleanest beat and raise I've seen this earnings season. Wow.",
        "Options flow on ${ticker} is extremely bullish today. Smart money is loading up quietly.",
    ],
    "negative": [
        "${ticker} guidance cut is brutal. This management team has zero credibility left.",
        "Sold all my ${ticker} today. Chart broken, fundamentals weakening. Not worth the risk.",
        "Everyone acting surprised by ${ticker} miss — the warning signs were there for months.",
        "${ticker} valuation at 35x forward earnings is insane given the growth slowdown. Avoid.",
        "Short ${ticker} here. Margin compression + slowing growth + crowded long = disaster.",
        "The ${ticker} bull case is falling apart. Competitor eating their lunch in every segment.",
        "Management credibility at ${ticker} is zero after three guidance cuts in 12 months.",
        "${ticker} chart screaming distribution. Big players quietly selling into retail strength.",
    ],
    "neutral": [
        "Watching ${ticker} closely. Key level is $185 — break above that changes everything.",
        "Not touching ${ticker} until after Fed meeting. Too much macro uncertainty right now.",
        "Mixed read on ${ticker} earnings. Top line beat, margins missed. Waiting for more clarity.",
        "${ticker} consolidating in a tight range. Either a coil or a flag — watching volume.",
        "Still on the fence with ${ticker}. Thesis intact but execution risk is real here.",
        "Taking a small starter position in ${ticker}. Will add if it holds this support level.",
        "${ticker} risk/reward feels balanced here. Not a screaming buy or sell — just a hold.",
    ],
}

METALS_NEWS_TEMPLATES = {
    "positive": [
        "Gold prices surge as safe-haven demand rises amid geopolitical tensions",
        "Silver breaks above key resistance; industrial demand from EV sector lifts {ticker}",
        "Central banks increase gold reserves to record levels, {ticker} surges",
        "{ticker} benefits from inflation hedge narrative; analysts raise price targets",
        "Precious metals rally as USD weakens on dovish Fed commentary — {ticker} leads",
        "Gold hits all-time high as investors seek safety amid banking sector fears",
        "Copper demand soars driven by green energy transition; {ticker} climbs 4%",
        "Palladium supply shortage tightens; {ticker} jumps on Russian export concerns",
        "Mining giant reports record quarterly output; {ticker} shares rise",
        "Silver demand from solar panel manufacturing hits 5-year high — {ticker} up",
    ],
    "negative": [
        "Gold slides as risk appetite returns; strong jobs data pressures {ticker}",
        "Silver ETF outflows accelerate as investors rotate into equities from {ticker}",
        "Rising real yields dampen gold appeal; {ticker} falls to 3-month low",
        "Dollar strengthens on hawkish Fed minutes; {ticker} faces headwinds",
        "Copper drops on weak China PMI data; demand outlook dims for {ticker}",
        "Palladium slumps as EV adoption reduces catalytic converter demand — {ticker} hit",
        "{ticker} gold miners miss production targets; shares fall sharply",
        "IMF warns of global slowdown; commodity complex including {ticker} weakens",
        "Hedge funds cut gold long positions to lowest since 2019 — {ticker} under pressure",
        "Stronger-than-expected CPI reduces rate cut hopes; {ticker} gold drops",
    ],
    "neutral": [
        "Gold holds steady ahead of key US inflation data; {ticker} awaits catalyst",
        "Silver consolidates near support; traders watch Fed speakers for {ticker} direction",
        "{ticker} copper prices range-bound amid mixed China economic signals",
        "Gold ETF flows flat for third consecutive week; {ticker} sentiment neutral",
        "Precious metals mixed as dollar index holds steady — {ticker} unchanged",
        "Analysts split on gold outlook; {ticker} sees balanced buy/sell activity",
        "{ticker} platinum group metals see limited movement in thin holiday trading",
    ],
}

METALS_SOCIAL_TEMPLATES = {
    "positive": [
        "Gold is THE hedge right now. Central banks are buying, retail should too. $GOLD 🥇",
        "$SILVER criminally undervalued vs gold. GSR at 85:1 — time to load up $SLV 🚀",
        "Copper demand from EVs + renewables is just getting started. $COPPER bull thesis intact",
        "When inflation stays sticky, $GOLD wins. Simple as that. Adding to position today.",
        "$GDX miners levered play on gold. If gold goes to $2500, miners go 3x. Loading. 📈",
        "Palladium short squeeze incoming. Supply from Russia constrained. $PALLADIUM watch carefully",
        "$NEM earnings beat + raised guidance. Gold miners finally performing. 🔥",
        "Silver's industrial demand story + monetary demand = perfect storm for $SLV bulls",
    ],
    "negative": [
        "Selling my $GOLD position. Risk-on is back, no need for safe havens rn.",
        "$SILVER rejection at $26 resistance again. Third time fails. Getting out.",
        "Gold bugs are delusional. Rate cuts priced in already. $GLD overvalued here.",
        "Copper demand destruction from China slowdown = stay away from $COPPER",
        "$GDX miners keep underperforming physical gold. Management destroying value.",
        "Palladium collapse continues. EVs are killing catalytic converter demand. $PALLADIUM short.",
        "Real yields rising = gold falling. $GOLD below 200DMA is a sell signal.",
        "Hedge fund exodus from $SLV silver ETF. Smart money leaving. Follow them.",
    ],
    "neutral": [
        "Watching $GOLD at the $2050 level. Break above = bull, break below = bear.",
        "$SILVER setting up for a big move either way. Tight range = explosion coming.",
        "Copper chart looks interesting but waiting for China data before touching $COPPER",
        "$GLD holding $180 support. Need confirmation before adding.",
        "Gold miners $GDX — mixed signals. Some beating, some missing. Not touching yet.",
        "Palladium bouncing but not convinced. $PALLADIUM watching for volume confirmation.",
        "$WPM streaming model insulates from mining costs. Interesting at these levels tbh.",
    ],
}

_social_feed:   deque = deque(maxlen=300)
_sentiment_store: Dict[str, deque] = {}
_ticker_store:    Dict[str, deque] = {}
_news_feed:       deque = deque(maxlen=200)
_chat_history:    List[Dict] = []

_live_sector_sentiment: Dict[str, float] = {}
_live_ticker_sentiment: Dict[str, float] = {}
_live_ticker_metadata:  Dict[str, Dict]  = {}

_batch_counter = 0


def _clamp(val: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, val))


def _sentiment_label(score: float) -> str:
    if score >= 0.6:   return "Highly Positive"
    if score >= 0.2:   return "Positive"
    if score >= -0.19: return "Neutral"
    if score >= -0.59: return "Negative"
    return "Highly Negative"


def _sentiment_color(score: float) -> str:
    if score >= 0.6:   return "#10b981"
    if score >= 0.2:   return "#34d399"
    if score >= -0.19: return "#94a3b8"
    if score >= -0.59: return "#f87171"
    return "#ef4444"


METALS_TICKERS = {"GOLD","SILVER","PLATINUM","COPPER","PALLADIUM","GLD","SLV","GDX","NEM","WPM"}


def _generate_news_item(ticker: str, sentiment_type: str) -> Dict:
    if ticker in METALS_TICKERS:
        templates = METALS_NEWS_TEMPLATES.get(sentiment_type, METALS_NEWS_TEMPLATES["neutral"])
    else:
        templates = NEWS_TEMPLATES.get(sentiment_type, NEWS_TEMPLATES["neutral"])
    headline  = random.choice(templates).replace("{ticker}", ticker)
    sources   = ["Reuters", "Bloomberg", "CNBC", "WSJ", "MarketWatch",
                 "Financial Times", "Kitco News", "Mining.com", "Metal Bulletin"]
    score_map = {"positive": random.uniform(0.35, 0.9),
                 "negative": random.uniform(-0.9, -0.35),
                 "neutral":  random.uniform(-0.18, 0.18)}
    return {
        "id":        f"{ticker}_{datetime.utcnow().timestamp():.4f}_{random.randint(0,9999)}",
        "ticker":    ticker,
        "headline":  headline,
        "source":    random.choice(sources),
        "sentiment": sentiment_type,
        "score":     round(score_map[sentiment_type], 3),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type":      "news",
    }


def _generate_social_item(ticker: str, sentiment_type: str) -> Dict:
    if ticker in METALS_TICKERS:
        templates = METALS_SOCIAL_TEMPLATES.get(sentiment_type, METALS_SOCIAL_TEMPLATES["neutral"])
        text = random.choice(templates)
    else:
        templates = SOCIAL_TEMPLATES.get(sentiment_type, SOCIAL_TEMPLATES["neutral"])
        text = random.choice(templates).replace("{ticker}", ticker)
    score_map = {"positive": random.uniform(0.2, 0.85),
                 "negative": random.uniform(-0.85, -0.2),
                 "neutral":  random.uniform(-0.15, 0.15)}
    platforms = ["Twitter/X", "Reddit r/wallstreetbets", "StockTwits",
                 "Reddit r/investing", "Twitter/X FinTwit", "Reddit r/Gold"]
    return {
        "id":        f"soc_{ticker}_{datetime.utcnow().timestamp():.4f}_{random.randint(0,9999)}",
        "ticker":    ticker,
        "text":      text,
        "platform":  random.choice(platforms),
        "sentiment": sentiment_type,
        "score":     round(score_map[sentiment_type], 3),
        "likes":     random.randint(5, 5000),
        "retweets":  random.randint(0, 800),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type":      "social",
    }


def _tick_sentiment(current: float, base: float) -> float:
    noise  = random.gauss(0, 0.03)
    revert = (base - current) * 0.05
    return _clamp(current + noise + revert)


def _init_data():
    for sector_name, sector_data in SECTORS.items():
        base = sector_data["base_sentiment"]
        _live_sector_sentiment[sector_name] = round(_clamp(base + random.gauss(0, 0.05)), 4)
        for ticker in sector_data["tickers"]:
            ticker_base = round(_clamp(base + random.gauss(0, 0.12)), 4)
            _live_ticker_sentiment[ticker] = ticker_base
            _live_ticker_metadata[ticker] = {
                "sector":       sector_name,
                "news_count":   random.randint(5, 40),
                "social_count": random.randint(80, 600),
                "confidence":   random.randint(70, 96),
                "trend":        random.choice(["up", "down", "stable"]),
                "news_score":   round(_clamp(ticker_base + random.gauss(0, 0.06)), 4),
                "social_score": round(_clamp(ticker_base + random.gauss(0, 0.11)), 4),
            }
    all_tickers = [t for s in SECTORS.values() for t in s["tickers"]]
    for _ in range(30):
        t    = random.choice(all_tickers)
        sc   = _live_ticker_sentiment.get(t, 0.0)
        kind = "positive" if sc > 0.15 else ("negative" if sc < -0.15 else "neutral")
        _news_feed.append(_generate_news_item(t, kind))
    for _ in range(40):
        t    = random.choice(all_tickers)
        sc   = _live_ticker_sentiment.get(t, 0.0)
        kind = "positive" if sc > 0.15 else ("negative" if sc < -0.15 else "neutral")
        _social_feed.append(_generate_social_item(t, kind))
    print(f"✅ Pre-initialized: {len(_live_ticker_sentiment)} tickers, "
          f"{len(_news_feed)} news, {len(_social_feed)} social items")


async def _data_generator_loop():
    global _batch_counter
    batch = 0
    while True:
        batch += 1
        all_tickers   = [t for s in SECTORS.values() for t in s["tickers"]]
        active_ticker = random.choice(all_tickers)
        ticker_sector = None
        for sname, sdata in SECTORS.items():
            if active_ticker in sdata["tickers"]:
                ticker_sector = sname
                break
        sector_base   = SECTORS[ticker_sector]["base_sentiment"]
        current_score = _live_ticker_sentiment.get(active_ticker, sector_base)
        new_score     = _tick_sentiment(current_score, sector_base)
        _live_ticker_sentiment[active_ticker] = new_score
        sent_type = "positive" if new_score > 0.15 else ("negative" if new_score < -0.15 else "neutral")
        if random.random() < 0.6:
            item = _generate_news_item(active_ticker, sent_type)
            _news_feed.append(item)
        else:
            item = _generate_social_item(active_ticker, sent_type)
            _social_feed.append(item)
        meta = _live_ticker_metadata.get(active_ticker, {})
        meta["news_count"]   = meta.get("news_count", 10) + (1 if item.get("source") else 0)
        meta["social_count"] = meta.get("social_count", 100) + (1 if item.get("platform") else 0)
        meta["news_score"]   = round(_clamp(meta.get("news_score", new_score) * 0.9 + item["score"] * 0.1), 4)
        if item.get("platform"):
            meta["social_score"] = round(_clamp(meta.get("social_score", new_score) * 0.9 + item["score"] * 0.1), 4)
        prev = current_score
        if   new_score > prev + 0.01: meta["trend"] = "up"
        elif new_score < prev - 0.01: meta["trend"] = "down"
        else:                          meta["trend"] = "stable"
        meta["confidence"] = min(99, meta.get("confidence", 70) + random.randint(-1, 2))
        _live_ticker_metadata[active_ticker] = meta
        if batch >= 300:
            batch = 0
            _batch_counter += 1
            for sector_name, sector_data in SECTORS.items():
                tickers    = sector_data["tickers"]
                scores     = [_live_ticker_sentiment.get(t, 0.0) for t in tickers]
                new_sector = sum(scores) / len(scores) if scores else 0.0
                _live_sector_sentiment[sector_name] = round(_clamp(new_sector), 4)
            snapshot = {
                "timestamp":        datetime.utcnow().isoformat() + "Z",
                "batch":            _batch_counter,
                "sector_sentiment": dict(_live_sector_sentiment),
                "top_tickers":      sorted(_live_ticker_sentiment.items(), key=lambda x: x[1], reverse=True)[:5],
            }
            if "history" not in _sentiment_store:
                _sentiment_store["history"] = deque(maxlen=288)
            _sentiment_store["history"].append(snapshot)
        else:
            if ticker_sector:
                tickers = SECTORS[ticker_sector]["tickers"]
                scores  = [_live_ticker_sentiment.get(t, 0.0) for t in tickers]
                _live_sector_sentiment[ticker_sector] = round(_clamp(sum(scores) / len(scores)), 4)
        await asyncio.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD ROUTES  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard():
    now = datetime.utcnow().isoformat() + "Z"
    sectors_out = []
    for sector_name, sector_data in SECTORS.items():
        score = _live_sector_sentiment.get(sector_name, 0.0)
        tickers_data = []
        for t in sector_data["tickers"]:
            ts = _live_ticker_sentiment.get(t, 0.0)
            tm = _live_ticker_metadata.get(t, {})
            tickers_data.append({
                "ticker":       t,
                "score":        round(ts, 4),
                "label":        _sentiment_label(ts),
                "color":        _sentiment_color(ts),
                "trend":        tm.get("trend", "stable"),
                "news_count":   tm.get("news_count", 0),
                "social_count": tm.get("social_count", 0),
                "confidence":   tm.get("confidence", 70),
                "news_score":   round(tm.get("news_score", ts), 4),
                "social_score": round(tm.get("social_score", ts), 4),
            })
        news_scores   = [d["news_score"]   for d in tickers_data]
        social_scores = [d["social_score"] for d in tickers_data]
        avg_news   = sum(news_scores)   / len(news_scores)   if news_scores   else 0
        avg_social = sum(social_scores) / len(social_scores) if social_scores else 0
        divergence = abs(avg_news - avg_social) > 0.3
        sectors_out.append({
            "sector":     sector_name,
            "id":         sector_data["id"],
            "score":      round(score, 4),
            "label":      _sentiment_label(score),
            "color":      _sentiment_color(score),
            "hex":        sector_data["color"],
            "tickers":    tickers_data,
            "divergence": divergence,
        })
    all_ticker_scores = list(_live_ticker_sentiment.items())
    top_bullish = sorted(all_ticker_scores, key=lambda x: x[1], reverse=True)[:5]
    top_bearish = sorted(all_ticker_scores, key=lambda x: x[1])[:5]
    alerts = []
    for t, ts in _live_ticker_sentiment.items():
        meta = _live_ticker_metadata.get(t, {})
        ns   = meta.get("news_score", ts)
        ss   = meta.get("social_score", ts)
        diff = ns - ss
        if abs(diff) >= 0.35:
            alerts.append({
                "ticker":       t,
                "news_score":   round(ns, 4),
                "social_score": round(ss, 4),
                "divergence":   round(abs(diff), 4),
                "direction":    "News positive / Social negative" if diff > 0 else "News negative / Social positive",
                "sector":       _live_ticker_metadata.get(t, {}).get("sector", "Unknown"),
            })
    alerts.sort(key=lambda x: x["divergence"], reverse=True)
    return JSONResponse({
        "timestamp":         now,
        "batch_count":       _batch_counter,
        "sectors":           sectors_out,
        "top_bullish":       [{"ticker": t, "score": round(s,4), "label": _sentiment_label(s), "color": _sentiment_color(s)} for t,s in top_bullish],
        "top_bearish":       [{"ticker": t, "score": round(s,4), "label": _sentiment_label(s), "color": _sentiment_color(s)} for t,s in top_bearish],
        "news_feed":         list(_news_feed)[-20:][::-1],
        "divergence_alerts": alerts[:5],
        "market_mood":       _sentiment_label(sum(_live_sector_sentiment.values()) / max(1, len(_live_sector_sentiment))),
        "overall_score":     round(sum(_live_sector_sentiment.values()) / max(1, len(_live_sector_sentiment)), 4),
    })


@app.get("/api/ticker/{ticker}")
def get_ticker(ticker: str):
    ticker = ticker.upper()
    if ticker not in _live_ticker_sentiment:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    score = _live_ticker_sentiment[ticker]
    meta  = _live_ticker_metadata.get(ticker, {})
    relevant_news = [n for n in list(_news_feed) if n.get("ticker") == ticker][-10:]
    return JSONResponse({
        "ticker":       ticker,
        "score":        round(score, 4),
        "label":        _sentiment_label(score),
        "color":        _sentiment_color(score),
        "sector":       meta.get("sector", "Unknown"),
        "news_score":   round(meta.get("news_score", score), 4),
        "social_score": round(meta.get("social_score", score), 4),
        "news_count":   meta.get("news_count", 0),
        "social_count": meta.get("social_count", 0),
        "confidence":   meta.get("confidence", 70),
        "trend":        meta.get("trend", "stable"),
        "recent_news":  relevant_news,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    })


@app.get("/api/history")
def get_history():
    history = list(_sentiment_store.get("history", deque()))
    return JSONResponse({"history": history, "count": len(history)})


@app.get("/api/sectors")
def get_sectors():
    result = []
    for sector_name, sector_data in SECTORS.items():
        score = _live_sector_sentiment.get(sector_name, 0.0)
        result.append({
            "sector": sector_name,
            "score":  round(score, 4),
            "label":  _sentiment_label(score),
            "color":  _sentiment_color(score),
            "hex":    sector_data["color"],
        })
    return JSONResponse({"sectors": result, "timestamp": datetime.utcnow().isoformat() + "Z"})


@app.get("/api/social")
def get_social(limit: int = 30):
    posts = list(_social_feed)[-limit:][::-1]
    return JSONResponse({"posts": posts, "total": len(_social_feed), "timestamp": datetime.utcnow().isoformat() + "Z"})


# ─────────────────────────────────────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_FILTER_REPLACEMENTS = {
    "drug": "medication", "drugs": "medications", "overdose": "excess dosage",
    "narcotic": "controlled substance", "opioid": "pain medication",
    "pharmaceutical": "pharma", "prescription": "regulated product",
    "bomb": "explosive event", "attack": "incident", "kill": "eliminate",
    "suicide": "self-harm event", "hack": "breach", "hacked": "breached",
    "exploit": "leverage", "fraud": "misconduct", "criminal": "legal issue",
    "illegal": "non-compliant", "bankrupt": "insolvent", "bankruptcy": "insolvency",
    "lawsuit": "legal action", "scandal": "controversy",
}


def _sanitize_for_llm(text: str) -> str:
    for bad, good in _FILTER_REPLACEMENTS.items():
        text = re.sub(rf"\b{bad}\b", good, text, flags=re.IGNORECASE)
    return text


def _sanitize_messages(messages: list) -> list:
    safe = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            content = _sanitize_for_llm(content)
        safe.append({**m, "content": content})
    return safe


def _build_dashboard_snapshot() -> str:
    lines = [f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"]
    lines.append("=== SECTOR SENTIMENT ===")
    for sector, score in _live_sector_sentiment.items():
        lines.append(f"  {sector}: {score:+.3f} ({_sentiment_label(score)})")
    metals_tickers = ["GOLD", "SILVER", "PLATINUM", "COPPER", "PALLADIUM", "GLD", "SLV", "GDX"]
    lines.append("\n=== METALS & PRECIOUS METALS DETAIL ===")
    for t in metals_tickers:
        if t in _live_ticker_sentiment:
            s = _live_ticker_sentiment[t]
            m = _live_ticker_metadata.get(t, {})
            name = METALS_NAMES.get(t, t)
            lines.append(f"  {name} ({t}): {s:+.3f} ({_sentiment_label(s)}) | News:{m.get('news_count',0)} Social:{m.get('social_count',0)} Conf:{m.get('confidence',70)}%")
    lines.append("\n=== TOP 5 BULLISH TICKERS ===")
    top5 = sorted(_live_ticker_sentiment.items(), key=lambda x: x[1], reverse=True)[:5]
    for t, s in top5:
        meta = _live_ticker_metadata.get(t, {})
        lines.append(f"  ${t}: {s:+.3f} ({_sentiment_label(s)}) | News: {meta.get('news_count',0)} | Social: {meta.get('social_count',0)} | Conf: {meta.get('confidence',70)}%")
    lines.append("\n=== TOP 5 BEARISH TICKERS ===")
    bot5 = sorted(_live_ticker_sentiment.items(), key=lambda x: x[1])[:5]
    for t, s in bot5:
        meta = _live_ticker_metadata.get(t, {})
        lines.append(f"  ${t}: {s:+.3f} ({_sentiment_label(s)}) | News: {meta.get('news_count',0)} | Conf: {meta.get('confidence',70)}%")
    lines.append("\n=== DIVERGENCE ALERTS ===")
    alerts_added = 0
    for t, ts in _live_ticker_sentiment.items():
        meta = _live_ticker_metadata.get(t, {})
        ns = meta.get("news_score", ts)
        ss = meta.get("social_score", ts)
        if abs(ns - ss) >= 0.35:
            lines.append(f"  ⚠ ${t}: News {ns:+.3f} vs Social {ss:+.3f} (Δ={abs(ns-ss):.3f})")
            alerts_added += 1
        if alerts_added >= 3:
            break
    lines.append("\n=== RECENT NEWS ===")
    for news in list(_news_feed)[-5:][::-1]:
        headline = _sanitize_for_llm(news.get("headline", ""))
        lines.append(f"  [{news.get('source','')}] {news.get('ticker','')}: {headline}")
    return _sanitize_for_llm("\n".join(lines))


def _build_system_prompt(rag_context: str, dashboard_snapshot: str, data_context: str) -> str:
    return f"""You are MarketPulse AI, a financial market sentiment research assistant.

SCOPE: Answer ONLY questions about market sentiment, stocks, sectors, financial trends, and NSE/global market data.

You have access to:
  • Live synthetic market sentiment scores
  • NSE India corporate announcements (WIPRO, INFY, TCS, and 1875+ symbols)
  • Yahoo Finance OHLCV price data
  • Tweet/social media sentiment scores
  • Live RSS news from Reuters, CNBC, MarketWatch, Economic Times, Moneycontrol

RESPONSE FORMAT for in-scope questions:
• Lead with score and label
• List sources used
• 3 bullet-point key findings with source citations
• Confidence and data volume
• Divergence note if applicable
• End with: ⚠️ This is for informational purposes only and does not constitute financial advice.

IMPORTANT — The disclaimer line above must ONLY appear on in-scope financial sentiment answers.
Do NOT add it to refusals or redirects.

━━━ LIVE MARKET SENTIMENT ━━━
{dashboard_snapshot}

━━━ STRUCTURED DATA CONTEXT (CSV + RSS) ━━━
{data_context}

━━━ KNOWLEDGE BASE (RAG) ━━━
{rag_context}

Use exact numbers from the data. Never fabricate figures. Cite sources.
"""


def _call_llm(client, model: str, messages: list, max_tokens: int = 1200) -> str:
    print(f"🤖 Calling model: {model}")
    safe_messages = _sanitize_messages(messages)
    response = client.chat.completions.create(
        model=model, messages=safe_messages, max_tokens=max_tokens, temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def _call_llm_with_fallback(client, messages: list) -> str:
    try:
        return _call_llm(client, CHAT_MODEL, messages)
    except Exception as e1:
        print(f"⚠ Primary model failed: {type(e1).__name__}: {e1}")
        try:
            return _call_llm(client, FAST_MODEL, messages)
        except Exception as e2:
            print(f"⚠ Fallback also failed: {type(e2).__name__}: {e2}")
            raise e2


def _build_local_response(query: str) -> str:
    """
    Data-driven fallback for TIER 1 queries when LLM is unavailable.
    Only called for in-scope financial questions — never for tier2/tier3.
    """
    q = query.lower()

    # ── Nifty 50 / Indian Index query ─────────────────────────────────────────
    if any(k in q for k in ["nifty", "sensex", "nse", "bse", "indian market",
                              "dalal street", "nifty 50", "nifty50", "bank nifty"]):
        # Aggregate NSE-listed sectors sentiment
        nse_sectors = ["Automobile", "Refineries", "Telecom & Infra", "Power",
                       "Mining", "Logistics", "Water & Utilities",
                       "Technology", "Financials", "Healthcare", "Consumer",
                       "Metals & Commodities", "Energy", "Alcohol & Tobacco", "Education"]
        nse_scores  = []
        lines_nifty = ["📊 Nifty 50 / Indian Market Sentiment", ""]
        lines_nifty.append("=== SECTOR-WISE SENTIMENT (NSE India) ===")
        for sec in nse_sectors:
            if sec in _live_sector_sentiment:
                sv = _live_sector_sentiment[sec]
                nse_scores.append(sv)
                icon = "▲" if sv >= 0.2 else "▼" if sv <= -0.2 else "●"
                lines_nifty.append(f"  {icon} {sec:<25} {sv:+.3f}  {_sentiment_label(sv)}")
        if nse_scores:
            overall_nse = sum(nse_scores) / len(nse_scores)
            lines_nifty.insert(2, f"Overall NSE Sentiment: {overall_nse:+.3f} ({_sentiment_label(overall_nse)})")
            lines_nifty.insert(3, "")

        # Top NSE tickers
        nse_tickers_all = [t for sec in nse_sectors
                           for t in SECTORS.get(sec, {}).get("tickers", [])
                           if t in _live_ticker_sentiment]
        if nse_tickers_all:
            top_nse = sorted([(t, _live_ticker_sentiment[t]) for t in nse_tickers_all],
                             key=lambda x: x[1], reverse=True)[:5]
            lines_nifty.append("")
            lines_nifty.append("=== TOP 5 NSE BULLISH TICKERS ===")
            for i, (t, s) in enumerate(top_nse, 1):
                m = _live_ticker_metadata.get(t, {})
                lines_nifty.append(f"  {i}. {t}: {s:+.3f} ({_sentiment_label(s)}) | Conf: {m.get('confidence',70)}%")
            bot_nse = sorted([(t, _live_ticker_sentiment[t]) for t in nse_tickers_all],
                             key=lambda x: x[1])[:3]
            lines_nifty.append("")
            lines_nifty.append("=== TOP 3 NSE BEARISH TICKERS ===")
            for i, (t, s) in enumerate(bot_nse, 1):
                lines_nifty.append(f"  {i}. {t}: {s:+.3f} ({_sentiment_label(s)})")

        lines_nifty.append("")
        lines_nifty.append("Sources: NSE India corporate announcements + live RSS (ET, Moneycontrol) + tweet sentiment")
        lines_nifty.append("⚠️ This is for informational purposes only and does not constitute financial advice.")
        return "".join(lines_nifty)

    KEYWORD_TICKERS = {
        "gold": "GOLD", "silver": "SILVER", "platinum": "PLATINUM",
        "copper": "COPPER", "palladium": "PALLADIUM", "gld": "GLD",
        "slv": "SLV", "gdx": "GDX", "nem": "NEM", "wpm": "WPM",
        "aapl": "AAPL", "apple": "AAPL", "msft": "MSFT", "microsoft": "MSFT",
        "nvda": "NVDA", "nvidia": "NVDA", "googl": "GOOGL", "google": "GOOGL",
        "meta": "META", "amd": "AMD", "intc": "INTC", "intel": "INTC",
        "tsla": "TSLA", "tesla": "TSLA", "amzn": "AMZN", "amazon": "AMZN",
        "btc": "BTC", "bitcoin": "BTC", "eth": "ETH", "ethereum": "ETH",
        "sol": "SOL", "coin": "COIN", "mstr": "MSTR",
        "jpm": "JPM", "gs": "GS", "bac": "BAC",
        "xom": "XOM", "cvx": "CVX", "bp": "BP",
        "jnj": "JNJ", "pfe": "PFE", "unh": "UNH", "lly": "LLY",
    }
    target_tickers = list(dict.fromkeys(
        v for k, v in KEYWORD_TICKERS.items() if k in q and v in _live_ticker_sentiment
    ))
    lines = [f"📊 MarketPulse AI — Sentiment Data: \"{query}\"", ""]

    if any(w in q for w in ["overall", "market", "summary", "pulse", "snapshot", "today"]):
        overall = sum(_live_sector_sentiment.values()) / max(1, len(_live_sector_sentiment))
        lines.append(f"Overall Market Sentiment: {overall:+.3f} ({_sentiment_label(overall)})")
        lines.append("")
        for s, v in sorted(_live_sector_sentiment.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {'▲' if v>=0 else '▼'} {s:<25} {v:+.3f}  {_sentiment_label(v)}")

    for ticker in target_tickers[:4]:
        score  = _live_ticker_sentiment[ticker]
        meta   = _live_ticker_metadata.get(ticker, {})
        ns     = meta.get("news_score", score)
        ss     = meta.get("social_score", score)
        nc     = meta.get("news_count", 0)
        sc_cnt = meta.get("social_count", 0)
        conf   = meta.get("confidence", 70)
        trend  = meta.get("trend", "stable")
        name   = METALS_NAMES.get(ticker, ticker)
        div    = abs(ns - ss)
        lines += [
            "",
            f"${ticker} ({name}) — {meta.get('sector','Unknown')}",
            f"  Sentiment Score: {score:+.3f} ({_sentiment_label(score)})",
            f"  News Score:      {ns:+.3f}",
            f"  Social Score:    {ss:+.3f}",
            f"  Trend:           {'▲ Rising' if trend=='up' else '▼ Falling' if trend=='down' else '● Stable'}",
            f"  Confidence:      {conf}%",
            f"  Volume:          {nc} news articles, {sc_cnt} social posts",
        ]
        if div >= 0.2:
            direction = "News positive, social negative" if ns > ss else "Social positive, news negative"
            lines.append(f"  ⚠ Divergence:   Δ{div:.3f} — {direction}")

    if any(w in q for w in ["risk", "red flag", "fear", "warning", "divergen"]):
        lines.append("\nRisk / Divergence Alerts:")
        flagged = []
        for t, ts in _live_ticker_sentiment.items():
            meta = _live_ticker_metadata.get(t, {})
            ns   = meta.get("news_score", ts)
            ss   = meta.get("social_score", ts)
            diff = abs(ns - ss)
            if diff >= 0.3 or (ts < -0.3 and meta.get("trend") == "down"):
                flagged.append((t, ts, ns, ss, diff, meta.get("trend","stable")))
        flagged.sort(key=lambda x: x[4], reverse=True)
        for t, ts, ns, ss, diff, trend in flagged[:5]:
            lines.append(f"  ⚠ ${t}: {ts:+.3f} | News {ns:+.3f} vs Social {ss:+.3f} | Δ{diff:.3f}")
        if not flagged:
            lines.append("  No major divergence alerts at this time.")

    if any(w in q for w in ["top", "best", "bullish", "most positive"]):
        top5 = sorted(_live_ticker_sentiment.items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append("\nTop 5 Bullish Tickers:")
        for i, (t, s) in enumerate(top5, 1):
            meta = _live_ticker_metadata.get(t, {})
            lines.append(f"  {i}. ${t}: {s:+.3f} ({_sentiment_label(s)}) | Conf: {meta.get('confidence',70)}%")

    if any(w in q for w in ["bearish", "worst", "most negative"]):
        bot5 = sorted(_live_ticker_sentiment.items(), key=lambda x: x[1])[:5]
        lines.append("\nTop 5 Bearish Tickers:")
        for i, (t, s) in enumerate(bot5, 1):
            lines.append(f"  {i}. ${t}: {s:+.3f} ({_sentiment_label(s)})")

    if not any(lines[2:]):
        overall = sum(_live_sector_sentiment.values()) / max(1, len(_live_sector_sentiment))
        lines.append(f"Overall: {overall:+.3f} ({_sentiment_label(overall)})")
        for s, v in sorted(_live_sector_sentiment.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {s}: {v:+.3f} ({_sentiment_label(v)})")

    # Disclaimer ONLY on in-scope financial responses
    lines.append("\n⚠️ This is for informational purposes only and does not constitute financial advice.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CHAT ENDPOINT  ← Main logic
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict]] = []


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # ── Step 1: Classify BEFORE calling LLM or building context ──────────────
    tier = _tier_classify(req.message)

    # ── Tier 3: Forbidden — hard refuse immediately, no data, no disclaimer ──
    if tier == "tier3":
        reply = _tier3_response(req.message)
        return JSONResponse({"reply": reply, "tier": "tier3"})

    # ── Tier 2: Out-of-scope — polite redirect, no data, no disclaimer ────────
    if tier == "tier2":
        reply = _tier2_response(req.message)
        return JSONResponse({"reply": reply, "tier": "tier2"})

    # ── Tier 1: In-scope — proceed with full LLM + data context ──────────────
    client = get_client()

    if client is None:
        # Demo mode — show data context but note API key missing
        data_ctx = build_data_context(req.message)
        nvda = _live_ticker_sentiment.get("NVDA", 0.0)
        aapl = _live_ticker_sentiment.get("AAPL", 0.0)
        reply = (
            "⚠️ **Demo Mode** — API key not configured.\n\n"
            "Set `API_KEY` and `API_ENDPOINT` in `.env` and restart to enable AI chat.\n\n"
            f"**Live sentiment preview:**\n"
            f"• $NVDA: {nvda:+.3f} ({_sentiment_label(nvda)})\n"
            f"• $AAPL: {aapl:+.3f} ({_sentiment_label(aapl)})\n\n"
            f"**Data context for your query:**\n{data_ctx[:800]}\n\n"
            "⚠️ This is for informational purposes only and does not constitute financial advice."
        )
        return JSONResponse({"reply": reply, "tier": "demo"})

    # Build all three context blocks
    rag_context = rag_retrieve(req.message)
    dashboard   = _build_dashboard_snapshot()
    data_ctx    = build_data_context(req.message)

    messages = [{"role": "system", "content": _build_system_prompt(rag_context, dashboard, data_ctx)}]
    for h in (req.history or [])[-6:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    try:
        reply = _call_llm_with_fallback(client, messages)
    except Exception as e:
        err_full = str(e)
        err_type = type(e).__name__
        print(f"❌ LLM failed: {err_type}: {err_full}")
        is_403 = "403" in err_full or "PermissionDenied" in err_type or "Content blocked" in err_full
        if is_403:
            local_reply = _build_local_response(req.message)
            return JSONResponse({"reply": local_reply, "tier": "tier1"})
        elif "401" in err_full or "Unauthorized" in err_full:
            hint = "Invalid API key — check API_KEY in .env"
        elif "404" in err_full:
            hint = f"Model not found. Tried: {CHAT_MODEL} then {FAST_MODEL}"
        elif "500" in err_full:
            hint = "Server error 500 — verify model name in TCS GenAI Lab portal"
        elif "timeout" in err_full.lower() or "connect" in err_full.lower():
            hint = "Connection timeout — check network and endpoint"
        else:
            hint = f"{err_type}: {err_full[:180]}"
        overall  = sum(_live_sector_sentiment.values()) / max(1, len(_live_sector_sentiment))
        top5_str = ", ".join(f"${t}: {s:+.3f}" for t, s in sorted(_live_ticker_sentiment.items(), key=lambda x: x[1], reverse=True)[:5])
        reply = (
            f"⚠️ LLM unavailable ({hint})\n\n"
            f"Live market data:\n"
            f"Overall: {overall:+.3f} ({_sentiment_label(overall)})\n"
            f"Top bullish: {top5_str}\n\n"
            "⚠️ This is for informational purposes only and does not constitute financial advice."
        )
        return JSONResponse({"reply": reply, "tier": "tier1", "error": hint})

    return JSONResponse({"reply": reply, "tier": tier})


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/data-stats")
def get_data_stats():
    return JSONResponse({
        "csv_files":    get_summary_stats(),
        "rag_index":    rag_stats(),
        "data_sources": [
            "NSE India — today (20-Mar-2026)",
            "NSE India — WIPRO, INFY, TCS announcements",
            "NSE India — broad 1875-symbol (Feb–Mar 2026)",
            "Yahoo Finance OHLCV — 25 global stocks",
            "Scored tweet sentiment per stock",
            "StockTwits/Twitter archive",
            "Live RSS: Reuters, CNBC, MarketWatch, ET, Moneycontrol",
            "Knowledge base (domain scenarios, FAQs)",
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.get("/api/query-context")
def get_query_context(q: str = ""):
    q_lower = q.lower().strip()

    # ── Comprehensive keyword → tickers map (covers Indian + Global + all sectors) ──
    KEYWORD_MAP = {
        # ── Precious metals & commodities ─────────────────────────────────────
        "gold":        ["GOLD","GLD","GDX","NEM","WPM"],
        "silver":      ["SILVER","SLV","WPM"],
        "platinum":    ["PLATINUM"],
        "palladium":   ["PALLADIUM"],
        "copper":      ["COPPER"],
        "metals":      ["GOLD","SILVER","PLATINUM","COPPER","PALLADIUM","GLD","SLV","GDX","NEM","WPM"],
        "commodities": ["GOLD","SILVER","COPPER","XOM","CVX"],
        "precious":    ["GOLD","SILVER","PLATINUM","PALLADIUM"],
        "mining":      ["GDX","NEM","WPM","COALINDIA","NMDC","VEDL","HINDALCO"],
        # ── Global Tech ────────────────────────────────────────────────────────
        "tech":        ["AAPL","MSFT","NVDA","GOOGL","META","AMD","INTC"],
        "technology":  ["AAPL","MSFT","NVDA","GOOGL","META","AMD","INTC"],
        "apple":       ["AAPL"],
        "microsoft":   ["MSFT"],
        "nvidia":      ["NVDA"],
        "google":      ["GOOGL"],
        "meta":        ["META"],
        "amd":         ["AMD"],
        "intel":       ["INTC"],
        "tesla":       ["TSLA"],
        "amazon":      ["AMZN"],
        "semiconductor":["NVDA","AMD","INTC"],
        # ── Indian IT / Technology ─────────────────────────────────────────────
        "tcs":         ["TCS","WIPRO","INFY"],
        "infosys":     ["INFY","TCS","WIPRO","HCL","TECHM"],
        "infy":        ["INFY","TCS","WIPRO"],
        "wipro":       ["WIPRO","TCS","INFY","HCL"],
        "hcl":         ["HCL","TECHM","TCS","WIPRO"],
        "techm":       ["TECHM","HCL","WIPRO"],
        "indian it":   ["TCS","INFY","WIPRO","HCL","TECHM"],
        "it sector":   ["TCS","INFY","WIPRO","HCL","TECHM","AAPL","MSFT"],
        # ── Indian Indices / Market ────────────────────────────────────────────
        "nifty":       ["TCS","INFY","WIPRO","RELIANCE","HDFC","ICICI","MARUTI","TATAMOTORS","NTPC","BHARTIARTL"],
        "nifty 50":    ["TCS","INFY","WIPRO","RELIANCE","HDFC","ICICI","MARUTI","TATAMOTORS","NTPC","BHARTIARTL"],
        "nifty50":     ["TCS","INFY","WIPRO","RELIANCE","HDFC","ICICI","MARUTI","TATAMOTORS","NTPC","BHARTIARTL"],
        "sensex":      ["TCS","INFY","WIPRO","RELIANCE","HDFC","ICICI","MARUTI","TATAMOTORS","NTPC","BHARTIARTL"],
        "bse":         ["TCS","INFY","WIPRO","RELIANCE","HDFC","ICICI"],
        "nse":         ["TCS","INFY","WIPRO","RELIANCE","HDFC","ICICI","NTPC","BHARTIARTL"],
        "indian market":["TCS","INFY","WIPRO","RELIANCE","HDFC","ICICI","MARUTI","TATAMOTORS"],
        "dalal street":["TCS","INFY","WIPRO","RELIANCE","HDFC","ICICI"],
        # ── Indian Financials ──────────────────────────────────────────────────
        "hdfc":        ["HDFC","ICICI","KOTAK","SBI","AXISBANK"],
        "icici":       ["ICICI","HDFC","KOTAK","SBI","AXISBANK"],
        "sbi":         ["SBI","HDFC","ICICI","KOTAK","AXISBANK"],
        "kotak":       ["KOTAK","HDFC","ICICI","AXISBANK"],
        "axis":        ["AXISBANK","HDFC","ICICI","KOTAK"],
        "indian bank": ["HDFC","ICICI","SBI","KOTAK","AXISBANK"],
        # ── Indian Automobile ──────────────────────────────────────────────────
        "maruti":      ["MARUTI","TATAMOTORS","BAJAJ-AUTO","M&M","HEROMOTOCO"],
        "tata motors": ["TATAMOTORS","MARUTI","BAJAJ-AUTO","M&M"],
        "tatamotors":  ["TATAMOTORS","MARUTI","BAJAJ-AUTO","M&M"],
        "bajaj":       ["BAJAJ-AUTO","HEROMOTOCO","TVSMOTORS","M&M"],
        "auto":        ["MARUTI","TATAMOTORS","BAJAJ-AUTO","M&M","HEROMOTOCO","EICHERMOT"],
        "automobile":  ["MARUTI","TATAMOTORS","BAJAJ-AUTO","M&M","HEROMOTOCO","EICHERMOT"],
        "ev":          ["TATAMOTORS","TSLA","MARUTI"],
        # ── Indian Energy / Refineries ─────────────────────────────────────────
        "reliance":    ["RELIANCE","IOC","BPCL","HPCL"],
        "ongc":        ["ONGC","IOC","BPCL","HPCL","RELIANCE"],
        "refinery":    ["RELIANCE","IOC","BPCL","HPCL","MRPL","CPCL"],
        "refineries":  ["RELIANCE","IOC","BPCL","HPCL","MRPL","CPCL"],
        "ioc":         ["IOC","BPCL","HPCL","RELIANCE"],
        "bpcl":        ["BPCL","IOC","HPCL","RELIANCE"],
        # ── Indian Power / Utilities ───────────────────────────────────────────
        "ntpc":        ["NTPC","POWERGRID","ADANIPOWER","TATAPOWER","NHPC"],
        "powergrid":   ["POWERGRID","NTPC","ADANIPOWER","TATAPOWER"],
        "adani":       ["ADANIPOWER","ADANIENT","ADANIPORTS","ADANIGREEN"],
        "power":       ["NTPC","POWERGRID","ADANIPOWER","TATAPOWER","CESC","NHPC","TORNTPOWER"],
        "utilities":   ["NTPC","POWERGRID","WABAG","TATAPOWER"],
        # ── Indian Telecom ─────────────────────────────────────────────────────
        "airtel":      ["BHARTIARTL","IDEA","TATACOMM","INDUS"],
        "bhartiartl":  ["BHARTIARTL","IDEA","TATACOMM"],
        "jio":         ["RELIANCE","BHARTIARTL","IDEA"],
        "telecom":     ["BHARTIARTL","IDEA","TATACOMM","INDUS","INFRATEL"],
        # ── Indian Pharma / Healthcare ─────────────────────────────────────────
        "cipla":       ["CIPLA","SUNPHARMA","DRREDDY","LUPIN","AUROPHARMA"],
        "sunpharma":   ["SUNPHARMA","CIPLA","DRREDDY","LUPIN"],
        "drreddy":     ["DRREDDY","SUNPHARMA","CIPLA","LUPIN"],
        "indian pharma":["SUNPHARMA","CIPLA","DRREDDY","LUPIN","AUROPHARMA"],
        # ── Indian Logistics ───────────────────────────────────────────────────
        "delhivery":   ["DELHIVERY","BLUEDART","CONCOR","GATI","TCI"],
        "bluedart":    ["BLUEDART","DELHIVERY","CONCOR","GATI"],
        "logistics":   ["BLUEDART","DELHIVERY","CONCOR","GATI","MAHLOG","TCI","ALLCARGO"],
        # ── Indian Education ───────────────────────────────────────────────────
        "naukri":      ["NAUKRI","CAREEREDGE","APTECH","NIITLTD"],
        "education":   ["NAUKRI","CAREEREDGE","APTECH","NIITLTD","NAVNETEDUL"],
        # ── Indian Alcohol & Tobacco ───────────────────────────────────────────
        "itc":         ["ITC","UBL","RADICO","MCDOWELL-N"],
        "ubl":         ["UBL","RADICO","MCDOWELL-N","ITC"],
        "alcohol":     ["UBL","RADICO","MCDOWELL-N","VSTIND","ITC"],
        # ── Global Finance ─────────────────────────────────────────────────────
        "banks":       ["JPM","GS","BAC","MS","WFC"],
        "finance":     ["JPM","GS","BAC","V","MA"],
        "jpmorgan":    ["JPM","GS","BAC"],
        "goldman":     ["GS","JPM","MS"],
        # ── Crypto ─────────────────────────────────────────────────────────────
        "bitcoin":     ["BTC","COIN","MSTR"],
        "crypto":      ["BTC","ETH","SOL","COIN","MSTR"],
        "ethereum":    ["ETH","BTC","COIN"],
        "solana":      ["SOL","ETH","BTC"],
        # ── Global Energy ──────────────────────────────────────────────────────
        "energy":      ["XOM","CVX","BP","NEE","ENPH","OXY"],
        "oil":         ["XOM","CVX","BP","OXY"],
        "solar":       ["ENPH","NEE"],
        # ── Global Healthcare ──────────────────────────────────────────────────
        "pharma":      ["JNJ","PFE","ABBV","LLY","MRNA"],
        "healthcare":  ["JNJ","PFE","UNH","MRNA","LLY"],
        # ── Macro themes ───────────────────────────────────────────────────────
        "inflation":   ["GOLD","GLD","SLV","XOM","CVX"],
        "fed":         ["JPM","GS","BAC","V","MA"],
        "rbi":         ["HDFC","ICICI","SBI","KOTAK","AXISBANK"],
        "rate":        ["JPM","GS","HDFC","ICICI","GOLD","GLD"],
        "market":      [],   # handled by fallback — returns top movers
        "overall":     [],
        "summary":     [],
        "today":       [],
        "bullish":     [],
        "bearish":     [],
        "sentiment":   [],
        "divergence":  [],
        "alert":       [],
    }

    # ── All 16 sector keyword mappings ────────────────────────────────────────
    SECTOR_KEYWORD_MAP = {
        # Global sectors
        "tech":             "Technology",
        "technology":       "Technology",
        "it sector":        "Technology",
        "energy":           "Energy",
        "oil":              "Energy",
        "financial":        "Financials",
        "finance":          "Financials",
        "bank":             "Financials",
        "banking":          "Financials",
        "health":           "Healthcare",
        "pharma":           "Healthcare",
        "consumer":         "Consumer",
        "retail":           "Consumer",
        "crypto":           "Crypto",
        "bitcoin":          "Crypto",
        "metal":            "Metals & Commodities",
        "gold":             "Metals & Commodities",
        "silver":           "Metals & Commodities",
        "commodity":        "Metals & Commodities",
        "precious":         "Metals & Commodities",
        "mining":           "Mining",
        # Indian sectors
        "automobile":       "Automobile",
        "auto":             "Automobile",
        "car":              "Automobile",
        "ev":               "Automobile",
        "education":        "Education",
        "alcohol":          "Alcohol & Tobacco",
        "tobacco":          "Alcohol & Tobacco",
        "refinery":         "Refineries",
        "refineries":       "Refineries",
        "telecom":          "Telecom & Infra",
        "infra":            "Telecom & Infra",
        "infrastructure":   "Telecom & Infra",
        "logistics":        "Logistics",
        "supply chain":     "Logistics",
        "power":            "Power",
        "electricity":      "Power",
        "water":            "Water & Utilities",
        "utilities":        "Water & Utilities",
        # Catch-all Indian market queries → show all sectors
        "nifty":            "Technology",
        "sensex":           "Financials",
        "nse":              "Technology",
        "indian market":    "Automobile",
        "dalal street":     "Refineries",
    }

    matched_tickers = set()
    matched_sectors = set()

    # Step 1: Match keywords from KEYWORD_MAP
    for kw, tickers in KEYWORD_MAP.items():
        if kw in q_lower:
            matched_tickers.update(tickers)

    # Step 2: Match sectors from SECTOR_KEYWORD_MAP
    for kw, sector in SECTOR_KEYWORD_MAP.items():
        if kw in q_lower:
            matched_sectors.add(sector)

    # Step 3: Direct ticker symbol match (e.g. "TCS", "$WIPRO", "infy")
    all_tickers_list = [t for s in SECTORS.values() for t in s["tickers"]]
    for t in all_tickers_list:
        if t.lower() in q_lower or f"${t.lower()}" in q_lower:
            matched_tickers.add(t)

    # Step 4: For Nifty/Sensex/NSE/BSE/Indian market — add ALL Indian sector tickers
    INDIAN_QUERY_KEYWORDS = ["nifty", "sensex", "nse", "bse", "indian market",
                              "dalal street", "nifty 50", "nifty50", "bank nifty"]
    if any(k in q_lower for k in INDIAN_QUERY_KEYWORDS):
        indian_sectors = ["Automobile","Refineries","Telecom & Infra","Power",
                          "Mining","Logistics","Water & Utilities","Technology",
                          "Financials","Healthcare","Consumer","Metals & Commodities",
                          "Energy","Alcohol & Tobacco","Education"]
        for sec in indian_sectors:
            matched_sectors.add(sec)
        # Add top tickers from each Indian sector
        for sec_name, sec_data in SECTORS.items():
            if sec_name in indian_sectors:
                matched_tickers.update(sec_data["tickers"][:3])

    # Step 5: For general market/overall/summary/today queries → top movers
    GENERAL_KEYWORDS = ["market","overall","summary","today","sentiment",
                        "overview","pulse","snapshot","bullish","bearish",
                        "divergence","alert","risk"]
    if any(k in q_lower for k in GENERAL_KEYWORDS) and not matched_tickers:
        all_scores = list(_live_ticker_sentiment.items())
        top5  = sorted(all_scores, key=lambda x: x[1], reverse=True)[:5]
        bot3  = sorted(all_scores, key=lambda x: x[1])[:3]
        matched_tickers.update({t for t, _ in top5})
        matched_tickers.update({t for t, _ in bot3})
        for sec in SECTORS:
            matched_sectors.add(sec)

    # Step 6: Final fallback → top 6 movers if still nothing matched
    if not matched_tickers and not matched_sectors:
        top6 = sorted(_live_ticker_sentiment.items(), key=lambda x: x[1], reverse=True)[:6]
        matched_tickers = {t for t, _ in top6}
    ticker_data = []
    for t in sorted(matched_tickers):
        if t not in _live_ticker_sentiment:
            continue
        score = _live_ticker_sentiment[t]
        meta  = _live_ticker_metadata.get(t, {})
        ticker_data.append({
            "ticker": t, "name": METALS_NAMES.get(t, t), "sector": meta.get("sector","Unknown"),
            "score": round(score,4), "label": _sentiment_label(score), "color": _sentiment_color(score),
            "trend": meta.get("trend","stable"), "news_score": round(meta.get("news_score",score),4),
            "social_score": round(meta.get("social_score",score),4),
            "news_count": meta.get("news_count",0), "social_count": meta.get("social_count",0),
            "confidence": meta.get("confidence",70),
        })
    sector_data = []
    for s in sorted(matched_sectors):
        if s not in _live_sector_sentiment:
            continue
        score = _live_sector_sentiment[s]
        sector_info = SECTORS.get(s, {})
        sector_data.append({
            "sector": s, "score": round(score,4), "label": _sentiment_label(score),
            "color": _sentiment_color(score), "hex": sector_info.get("color","#6366f1"),
        })
    all_news   = list(_news_feed)
    rel_news   = [n for n in all_news if n.get("ticker") in matched_tickers][-15:][::-1]
    if not rel_news:
        rel_news = list(reversed(all_news[-8:]))
    all_social  = list(_social_feed)
    rel_social  = [p for p in all_social if p.get("ticker") in matched_tickers][-15:][::-1]
    if not rel_social:
        rel_social = list(reversed(all_social[-8:]))
    return JSONResponse({
        "query": q, "tickers": ticker_data, "sectors": sector_data,
        "news": rel_news, "social": rel_social,
        "matched_on": list(matched_tickers | matched_sectors),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.get("/health")
def health():
    return {
        "status": "ok", "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "batch_count": _batch_counter, "rag": rag_stats(),
        "tickers_live": len(_live_ticker_sentiment),
    }


@app.get("/api/debug")
def debug():
    api_key  = os.environ.get("API_KEY", "").strip()
    endpoint = os.environ.get("API_ENDPOINT", "").strip()
    return JSONResponse({
        "status":          "ok",
        "tickers_loaded":  len(_live_ticker_sentiment),
        "sectors_loaded":  len(_live_sector_sentiment),
        "news_items":      len(_news_feed),
        "batch_count":     _batch_counter,
        "api_key_set":     bool(api_key) and api_key != "your_actual_api_key_here",
        "api_key_preview": (api_key[:6] + "..." + api_key[-4:]) if len(api_key) > 10 else ("MISSING" if not api_key else "TOO_SHORT"),
        "api_endpoint":    endpoint or "MISSING",
        "rag_chunks":      rag_stats().get("total_chunks", 0),
        "data_sources":    get_summary_stats(),
    })


@app.get("/api/test-llm")
async def test_llm():
    client = get_client()
    if client is None:
        return JSONResponse({"ok": False, "error": "No API credentials in .env", "api_key_set": False})
    api_key  = os.environ.get("API_KEY", "")
    endpoint = os.environ.get("API_ENDPOINT", "")
    results  = {}
    for model_name, model_id in [("GPT-4o (primary)", CHAT_MODEL), ("GPT-3.5 (fallback)", FAST_MODEL)]:
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=10, temperature=0,
            )
            results[model_name] = {"model": model_id, "status": "✅ OK", "reply": resp.choices[0].message.content.strip()}
        except Exception as e:
            results[model_name] = {"model": model_id, "status": "❌ FAILED", "error": str(e)[:300]}
    return JSONResponse({
        "ok": any(r["status"].startswith("✅") for r in results.values()),
        "base_url": endpoint.rstrip("/"), "api_key": api_key[:16] + "..." if api_key else "NOT SET",
        "models": results,
        "fix_hint": "If both fail with 500, the model name is wrong. Check TCS GenAI Lab portal.",
    })


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
      <rect width="64" height="64" rx="10" fill="#0f172a"/>
      <polyline points="8,48 20,32 30,38 44,18 56,24" fill="none" stroke="#10b981" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="56" cy="24" r="4" fill="#10b981"/>
    </svg>"""
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)