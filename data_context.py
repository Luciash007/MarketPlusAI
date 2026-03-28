"""
data_context.py — Structured data summaries for MarketPulse AI
===============================================================

Reads all CSV files + live RSS feeds and returns a compact,
human-readable context string that is injected into the LLM system prompt.

Called from main.py:  build_data_context(user_query)
"""

import csv
import os
import re
import time
import threading
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger("data_context")

# ── Paths ──────────────────────────────────────────────────────────────────────
# DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent))
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "__data__"))

CSV_FILES = {
    "nse_today":     DATA_DIR / "CF-AN-equities-20-Mar-2026.csv",
    "nse_wipro":     DATA_DIR / "CF-AN-equities-WIPRO-20-Mar-2026.csv",
    "nse_infy":      DATA_DIR / "CF-AN-equities-INFY-20-Mar-2026.csv",
    "nse_tcs":       DATA_DIR / "CF-AN-equities-TCS-20-03-2025-to-20-03-2026.csv",
    "nse_broad":     DATA_DIR / "CF-AN-equities-20-02-2026-to-20-03-2026.csv",
    "yfinance":      DATA_DIR / "stock_yfinance_data.csv",
    "scored_tweets": DATA_DIR / "scored_tweets_total.csv",
    "stockerbot":    DATA_DIR / "stockerbot-export1.csv",
    "stock_tweets":  DATA_DIR / "stock_tweets.csv",
}

RSS_URLS = [
    ("Reuters Business",    "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters Wealth",      "https://feeds.reuters.com/news/wealth"),
    ("CNBC Markets",        "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("MarketWatch",         "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Economic Times",      "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Moneycontrol",        "https://www.moneycontrol.com/rss/marketreports.xml"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()[:600]


def _read_csv(path: Path, max_rows: int = 9999, encoding: str = "utf-8-sig") -> List[Dict]:
    if not path or not path.exists():
        return []
    try:
        rows = []
        with open(path, "r", encoding=encoding, errors="replace") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
                if len(rows) >= max_rows:
                    break
        return rows
    except Exception as e:
        logger.warning(f"CSV read error {path.name}: {e}")
        return []


# ── Section builders ───────────────────────────────────────────────────────────

def _section_nse(rows: List[Dict], label: str, max_items: int = 30) -> str:
    if not rows:
        return ""
    lines = [f"\n=== {label} — NSE CORPORATE ANNOUNCEMENTS ({len(rows)} total, latest {min(max_items,len(rows))}) ==="]
    for row in rows[:max_items]:
        sym  = row.get("SYMBOL", "").strip()
        co   = row.get("COMPANY NAME", "").strip()
        subj = row.get("SUBJECT", "").strip()
        dt   = row.get("BROADCAST DATE/TIME", "").strip()
        det  = _clean(row.get("DETAILS", ""))[:220]
        lines.append(f"  [{dt}] {sym} ({co}) | {subj} | {det}")
    return "\n".join(lines)


def _section_yfinance(rows: List[Dict]) -> str:
    if not rows:
        return ""
    stock_data: Dict[str, List] = defaultdict(list)
    for row in rows:
        stock_data[row.get("Stock Name", "")].append(row)
    lines = ["\n=== STOCK PRICE DATA (Yahoo Finance OHLCV — 25 global stocks) ==="]
    for stock, records in sorted(stock_data.items()):
        if not stock:
            continue
        closes = []
        for r in records:
            try:
                closes.append((r["Date"], float(r["Close"])))
            except Exception:
                pass
        if not closes:
            continue
        closes.sort(key=lambda x: x[0])
        s_date, s_close = closes[0]
        e_date, e_close = closes[-1]
        pct = (e_close - s_close) / s_close * 100 if s_close else 0
        try:
            hi = max(float(r["High"]) for r in records if r.get("High"))
            lo = min(float(r["Low"])  for r in records if r.get("Low"))
        except Exception:
            hi, lo = e_close, e_close
        trend = "📈" if pct > 0 else "📉"
        lines.append(
            f"  {trend} {stock}: ${e_close:.2f} (latest {e_date}) | "
            f"Change {pct:+.1f}% since {s_date} | Hi ${hi:.2f} Lo ${lo:.2f} | {len(closes)} sessions"
        )
    return "\n".join(lines)


def _section_tweet_sentiment(rows: List[Dict], max_rows: int = 5000) -> str:
    if not rows:
        return ""
    stock_sent: Dict[str, List[float]] = defaultdict(list)
    for row in rows[:max_rows]:
        stock = row.get("Stock", "").strip().upper()
        try:
            sent = float(row.get("Sentiment", 0))
        except Exception:
            sent = 0.0
        if stock:
            stock_sent[stock].append(sent)
    lines = ["\n=== TWEET SENTIMENT (scored, per stock) ==="]
    for stock, scores in sorted(stock_sent.items()):
        avg   = sum(scores) / len(scores)
        pos   = sum(1 for s in scores if s > 0)
        neg   = sum(1 for s in scores if s < 0)
        icon  = "📈" if avg > 0.1 else "📉" if avg < -0.1 else "➡"
        label = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
        lines.append(
            f"  {icon} {stock}: avg={avg:+.3f} ({label}) | "
            f"{len(scores)} tweets | {pos} pos / {neg} neg"
        )
    return "\n".join(lines)


def _section_stockerbot(rows: List[Dict], max_rows: int = 5000) -> str:
    if not rows:
        return ""
    sym_count:   Dict[str, int]        = defaultdict(int)
    sym_samples: Dict[str, List[str]]  = defaultdict(list)
    for row in rows[:max_rows]:
        syms = row.get("symbols", "").strip()
        text = _clean(row.get("text", ""))
        for sym in syms.split(","):
            sym = sym.strip().upper()
            if sym:
                sym_count[sym] += 1
                if len(sym_samples[sym]) < 2 and text:
                    sym_samples[sym].append(text[:150])
    lines = ["\n=== SOCIAL MEDIA POSTS (StockTwits / Twitter archive — top symbols) ==="]
    top = sorted(sym_count.items(), key=lambda x: x[1], reverse=True)[:30]
    for sym, cnt in top:
        sample = " | ".join(sym_samples[sym])
        lines.append(f"  {sym}: {cnt} posts | Sample: {sample[:200]}")
    return "\n".join(lines)


def _section_stock_tweets(rows: List[Dict], max_rows: int = 2000) -> str:
    if not rows:
        return ""
    stock_map: Dict[str, List[str]] = defaultdict(list)
    for row in rows[:max_rows]:
        stock = row.get("Stock Name", "").strip().upper()
        tweet = _clean(row.get("Tweet", ""))
        if stock and tweet:
            stock_map[stock].append(tweet[:150])
    lines = ["\n=== TWITTER DISCUSSIONS (raw tweets by stock) ==="]
    for stock, tweets in sorted(stock_map.items()):
        sample = " | ".join(tweets[:2])
        lines.append(f"  {stock}: {len(tweets)} tweets | Sample: {sample[:300]}")
    return "\n".join(lines)


def _section_rss() -> str:
    try:
        import feedparser
    except ImportError:
        return "\n=== LIVE RSS NEWS ===\n  (feedparser not installed — run: pip install feedparser)"

    lines = ["\n=== LIVE RSS NEWS HEADLINES (fetched now) ==="]
    total = 0
    for source_name, url in RSS_URLS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                title   = _clean(getattr(entry, "title", ""))
                summary = _clean(getattr(entry, "summary", ""))[:150]
                pub     = getattr(entry, "published", "")[:25]
                if title:
                    lines.append(f"  [{source_name} | {pub}] {title}. {summary}")
                    total += 1
        except Exception as e:
            lines.append(f"  [{source_name}] fetch failed: {e}")
    lines.insert(1, f"  Total: {total} live headlines")
    return "\n".join(lines)


# ── RSS cache (avoid hammering feeds on every chat message) ───────────────────

_rss_cache: Dict = {"text": "", "ts": 0}
_rss_lock         = threading.Lock()
RSS_CACHE_TTL     = 300  # 5 minutes


def _cached_rss() -> str:
    with _rss_lock:
        if time.time() - _rss_cache["ts"] < RSS_CACHE_TTL and _rss_cache["text"]:
            return _rss_cache["text"]
    result = _section_rss()
    with _rss_lock:
        _rss_cache["text"] = result
        _rss_cache["ts"]   = time.time()
    return result


# ── Main public function ───────────────────────────────────────────────────────

def build_data_context(query: str = "") -> str:
    """
    Returns a rich multi-section string summarising all data sources,
    targeted to the user's query. Injected into the LLM system prompt.
    """
    q = query.lower()
    sections: List[str] = []

    # ── Always include: price data + tweet sentiment ──────────────────────────
    yf_rows = _read_csv(CSV_FILES["yfinance"])
    sections.append(_section_yfinance(yf_rows))

    sc_rows = _read_csv(CSV_FILES["scored_tweets"], max_rows=5000)
    sections.append(_section_tweet_sentiment(sc_rows))

    # ── NSE Today — always included ───────────────────────────────────────────
    nse_today = _read_csv(CSV_FILES["nse_today"])
    sections.append(_section_nse(nse_today, "TODAY (20 Mar 2026)", max_items=19))

    # ── NSE Company-specific — based on query ─────────────────────────────────
    if any(k in q for k in ["wipro", "wip"]):
        rows = _read_csv(CSV_FILES["nse_wipro"], max_rows=300)
        sections.append(_section_nse(rows, "WIPRO ANNOUNCEMENTS", max_items=25))

    if any(k in q for k in ["infy", "infosys"]):
        rows = _read_csv(CSV_FILES["nse_infy"], max_rows=300)
        sections.append(_section_nse(rows, "INFOSYS ANNOUNCEMENTS", max_items=25))

    if any(k in q for k in ["tcs", "tata consultancy"]):
        rows = _read_csv(CSV_FILES["nse_tcs"], max_rows=300)
        sections.append(_section_nse(rows, "TCS ANNOUNCEMENTS", max_items=25))

    # ── NSE broad — query-filtered ────────────────────────────────────────────
    broad = _read_csv(CSV_FILES["nse_broad"], max_rows=500)
    if q:
        words    = [w for w in q.split() if len(w) > 3]
        filtered = [
            r for r in broad
            if any(
                w in (r.get("SYMBOL","") + r.get("COMPANY NAME","") + r.get("DETAILS","")).lower()
                for w in words
            )
        ]
        if not filtered:
            filtered = broad[:40]
    else:
        filtered = broad[:40]
    sections.append(_section_nse(filtered, "NSE INDIA BROAD", max_items=min(30, len(filtered))))

    # ── Social media posts ────────────────────────────────────────────────────
    sb_rows = _read_csv(CSV_FILES["stockerbot"], max_rows=5000)
    sections.append(_section_stockerbot(sb_rows))

    # ── Raw stock tweets — only for relevant tickers ──────────────────────────
    tweet_stocks = ["tsla", "tesla", "aapl", "apple", "amzn", "amazon", "meta",
                    "msft", "microsoft", "nvda", "nvidia", "googl", "google"]
    if any(k in q for k in tweet_stocks):
        st_rows = _read_csv(CSV_FILES["stock_tweets"], max_rows=2000)
        sections.append(_section_stock_tweets(st_rows))

    # ── Live RSS news (cached 5 min) ──────────────────────────────────────────
    sections.append(_cached_rss())

    return "\n".join(s for s in sections if s and s.strip())


def get_summary_stats() -> Dict:
    """Returns row counts for all CSV files — used by /api/data-stats."""
    stats = {}
    for key, path in CSV_FILES.items():
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                    stats[key] = sum(1 for _ in f) - 1
            except Exception:
                stats[key] = "error"
        else:
            stats[key] = "file not found"
    return stats