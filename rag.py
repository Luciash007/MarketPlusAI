"""
rag.py — Enhanced Retrieval-Augmented Generation for MarketPulse AI
=====================================================================

Data sources ingested:
  1. knowledge_base.json          — Agent identity, domain scenarios, FAQs
  2. CF-AN-equities-*.csv         — NSE India corporate announcements
  3. stock_yfinance_data.csv      — OHLCV price data for 25 global stocks
  4. scored_tweets_total.csv      — Scored tweet sentiment per stock
  5. tweet_sentiment.csv          — Pre-cleaned tweet sentiment labels
  6. stockerbot-export1.csv       — StockTwits / Twitter financial posts
  7. stock_tweets.csv             — Raw tweet corpus (sampled)
  8. Live RSS feeds               — Reuters, CNBC, ET, Moneycontrol
"""

import json
import csv
import re
import time
import threading
import logging
import os
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    print("⚠ feedparser not installed — live RSS disabled. Run: pip install feedparser")

logger = logging.getLogger("rag")

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE    = Path(__file__).parent
KB_PATH  = _BASE / "knowledge_base.json"
DATA_DIR = Path(os.environ.get("DATA_DIR", _BASE))

CSV_FILES = {
    "nse_today":     DATA_DIR / "CF-AN-equities-20-Mar-2026.csv",
    "nse_wipro":     DATA_DIR / "CF-AN-equities-WIPRO-20-Mar-2026.csv",
    "nse_infy":      DATA_DIR / "CF-AN-equities-INFY-20-Mar-2026.csv",
    "nse_tcs":       DATA_DIR / "CF-AN-equities-TCS-20-03-2025-to-20-03-2026.csv",
    "nse_broad":     DATA_DIR / "CF-AN-equities-20-02-2026-to-20-03-2026.csv",
    "yfinance":      DATA_DIR / "stock_yfinance_data.csv",
    "scored_tweets": DATA_DIR / "scored_tweets_total.csv",
    "tweet_sent":    DATA_DIR / "tweet_sentiment.csv",
    "stockerbot":    DATA_DIR / "stockerbot-export1.csv",
    "stock_tweets":  DATA_DIR / "stock_tweets.csv",
}

RSS_URLS = [
    ("Reuters Business",    "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters Wealth",      "https://feeds.reuters.com/news/wealth"),
    ("CNBC Markets",        "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC Finance",        "https://www.cnbc.com/id/100727362/device/rss/rss.html"),
    ("MarketWatch",         "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Economic Times Mkt",  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Economic Times News", "https://economictimes.indiatimes.com/news/rssfeeds/1715249553.cms"),
    ("Moneycontrol Mkt",    "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("Moneycontrol Top",    "https://www.moneycontrol.com/rss/topnews.xml"),
]

TOP_K     = 8
MIN_SCORE = 0.03


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:800]


def _read_csv_safe(path: Path, max_rows: int = 99999, encoding: str = "utf-8-sig") -> List[Dict]:
    if not path.exists():
        return []
    try:
        rows = []
        with open(path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
                if len(rows) >= max_rows:
                    break
        return rows
    except Exception as e:
        logger.warning(f"CSV read failed {path.name}: {e}")
        return []


# ── Chunk builders ─────────────────────────────────────────────────────────────

def _chunks_from_knowledge_base(kb: dict) -> List[str]:
    chunks = []
    agent = kb.get("agent_identity", {})
    if agent:
        chunks.append(
            f"Agent Identity — Name: {agent.get('name','')} | Role: {agent.get('role','')} | "
            f"Scope: {agent.get('scope','')} | Prohibited: {', '.join(agent.get('prohibited_scope',[]))}"
        )
    for sector in kb.get("sectors", []):
        tickers = ", ".join(sector.get("tickers", []))
        chunks.append(
            f"Sector {sector.get('name','')} ({sector.get('sector_id','')}): "
            f"{sector.get('description','')}. Tickers: {tickers}"
        )
    scoring = kb.get("sentiment_scoring", {})
    if scoring:
        ranges  = "; ".join(f"{k}={v}" for k, v in scoring.get("ranges", {}).items())
        weights = "; ".join(f"{k}={v}" for k, v in scoring.get("sources_weighted", {}).items())
        chunks.append(f"Sentiment scale {scoring.get('scale','')}: {ranges}. Source weights: {weights}")
    for domain in kb.get("domains", []):
        dname = domain.get("domain_name", "")
        for sc in domain.get("scenarios", []):
            chunk = (
                f"Domain: {dname} | Tier: {sc.get('tier','')} | "
                f"Query example: {sc.get('customer_query_sample','')} | "
                f"Strategy: {sc.get('response_strategy','')} | "
                f"Summary: {sc.get('summary_for_agent','')} | "
                f"Sample: {sc.get('sample_response','')[:300]}"
            )
            chunks.append(chunk)
    for faq in kb.get("faqs", []):
        chunks.append(f"FAQ: {faq.get('q','')} Answer: {faq.get('a','')}")
    for ex in kb.get("example_queries_and_responses", []):
        chunks.append(
            f"Example query: {ex.get('query','')} | Type: {ex.get('response_type','')} | "
            f"Template: {ex.get('answer_template','')}"
        )
    return [c.strip() for c in chunks if c.strip()]


def _chunks_from_nse(rows: List[Dict], label: str) -> List[str]:
    chunks = []
    for row in rows:
        sym     = row.get("SYMBOL", "").strip()
        co      = row.get("COMPANY NAME", "").strip()
        subj    = row.get("SUBJECT", "").strip()
        details = _clean(row.get("DETAILS", ""))
        bdt     = row.get("BROADCAST DATE/TIME", "").strip()
        if not sym and not details:
            continue
        chunks.append(
            f"[{label}] NSE Announcement | Symbol: {sym} | Company: {co} | "
            f"Subject: {subj} | Date: {bdt} | Details: {details}"
        )
    return chunks


def _chunks_from_yfinance(rows: List[Dict]) -> List[str]:
    stock_data: Dict[str, List] = defaultdict(list)
    for row in rows:
        stock_data[row.get("Stock Name", "")].append(row)
    chunks = []
    for stock, records in stock_data.items():
        if not stock or not records:
            continue
        closes = []
        for r in records:
            try:
                closes.append((r.get("Date", ""), float(r.get("Close", 0))))
            except Exception:
                pass
        if not closes:
            continue
        closes.sort(key=lambda x: x[0])
        latest_date, latest_close = closes[-1]
        earliest_date, earliest_close = closes[0]
        pct = ((latest_close - earliest_close) / earliest_close * 100) if earliest_close else 0
        try:
            highs = [float(r.get("High", 0)) for r in records if r.get("High")]
            lows  = [float(r.get("Low",  0)) for r in records if r.get("Low")]
            hi, lo = max(highs), min(lows)
        except Exception:
            hi, lo = latest_close, latest_close
        chunks.append(
            f"Stock Price Data ({stock}): Period {earliest_date} to {latest_date} | "
            f"Latest Close: ${latest_close:.2f} | Change: {pct:+.1f}% | "
            f"Period High: ${hi:.2f} | Period Low: ${lo:.2f} | "
            f"Data points: {len(closes)}"
        )
    return chunks


def _chunks_from_scored_tweets(rows: List[Dict], max_rows: int = 5000) -> List[str]:
    stock_sent:  Dict[str, List[float]] = defaultdict(list)
    stock_texts: Dict[str, List[str]]   = defaultdict(list)
    for row in rows[:max_rows]:
        stock = row.get("Stock", "").strip().upper()
        text  = _clean(row.get("text", ""))
        try:
            sent = float(row.get("Sentiment", 0))
        except Exception:
            sent = 0.0
        if stock:
            stock_sent[stock].append(sent)
            if text and len(stock_texts[stock]) < 3:
                stock_texts[stock].append(text)
    chunks = []
    for stock, scores in stock_sent.items():
        avg   = sum(scores) / len(scores)
        pos   = sum(1 for s in scores if s > 0)
        neg   = sum(1 for s in scores if s < 0)
        label = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
        sample = " | ".join(stock_texts[stock][:2])
        chunks.append(
            f"Tweet Sentiment ({stock}): avg={avg:.3f} ({label}) | "
            f"{len(scores)} tweets | {pos} pos / {neg} neg | Sample: {sample[:200]}"
        )
    return chunks


def _chunks_from_stockerbot(rows: List[Dict], max_rows: int = 5000) -> List[str]:
    sym_texts: Dict[str, List[str]] = defaultdict(list)
    for row in rows[:max_rows]:
        syms = row.get("symbols", "").strip()
        text = _clean(row.get("text", ""))
        ts   = row.get("timestamp", "")
        if syms and text:
            for sym in syms.split(","):
                sym = sym.strip().upper()
                if sym:
                    sym_texts[sym].append(f"[{ts[:10]}] {text}")
    chunks = []
    for sym, texts in sym_texts.items():
        combined = " || ".join(texts[:4])
        chunks.append(
            f"Social Media Posts ({sym}): {len(texts)} posts | Sample: {combined[:400]}"
        )
    return chunks


def _chunks_from_stock_tweets(rows: List[Dict], max_rows: int = 3000) -> List[str]:
    stock_tweets: Dict[str, List[str]] = defaultdict(list)
    for row in rows[:max_rows]:
        stock = row.get("Stock Name", "").strip().upper()
        tweet = _clean(row.get("Tweet", ""))
        date  = row.get("Date", "")
        if stock and tweet:
            stock_tweets[stock].append(f"[{str(date)[:10]}] {tweet[:200]}")
    chunks = []
    for stock, tweets in stock_tweets.items():
        sample = " || ".join(tweets[:3])
        chunks.append(
            f"Twitter Discussion ({stock}): {len(tweets)} tweets | Sample: {sample[:400]}"
        )
    return chunks


def _chunks_from_rss(feed_items: List[Dict]) -> List[str]:
    chunks = []
    for item in feed_items:
        source  = item.get("source", "")
        title   = _clean(item.get("title", ""))
        summary = _clean(item.get("summary", ""))
        pub     = item.get("published", "")
        if title:
            chunks.append(
                f"[Live News | {source}] {title} | Published: {pub} | Summary: {summary[:300]}"
            )
    return chunks


# ── RSS Fetcher ────────────────────────────────────────────────────────────────

class RSSFetcher:
    def __init__(self):
        self._items: List[Dict] = []
        self._lock  = threading.Lock()
        self._fetch_now()
        t = threading.Thread(target=self._background_refresh, daemon=True)
        t.start()

    def _fetch_now(self):
        if not HAS_FEEDPARSER:
            return
        items = []
        for source_name, url in RSS_URLS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:15]:
                    items.append({
                        "source":    source_name,
                        "title":     getattr(entry, "title", ""),
                        "summary":   getattr(entry, "summary", ""),
                        "link":      getattr(entry, "link", ""),
                        "published": getattr(entry, "published", ""),
                    })
            except Exception as e:
                logger.warning(f"RSS fetch failed ({source_name}): {e}")
        with self._lock:
            self._items = items
        print(f"✅ RSS: {len(items)} live news items fetched")

    def _background_refresh(self):
        while True:
            time.sleep(600)
            self._fetch_now()

    @property
    def items(self) -> List[Dict]:
        with self._lock:
            return list(self._items)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._items)


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        self._chunks:     List[str]       = []
        self._rss_chunks: List[str]       = []
        self._vectorizer: TfidfVectorizer = None
        self._matrix                      = None
        self._rss_fetcher: RSSFetcher     = None
        self._lock                        = threading.Lock()
        self._stats: Dict                 = {}
        self._build()

    def retrieve(self, query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> str:
        self._maybe_refresh_rss()
        with self._lock:
            chunks = self._chunks
            vec    = self._vectorizer
            matrix = self._matrix
        if not chunks or vec is None:
            return "Knowledge base not available."
        qv     = vec.transform([query])
        scores = cosine_similarity(qv, matrix).flatten()
        top_ix = [i for i in np.argsort(scores)[::-1] if scores[i] >= min_score][:top_k]
        if not top_ix:
            recent = self._rss_chunks[:5]
            if recent:
                return "No high-relevance match. Recent live news:\n\n" + "\n\n---\n\n".join(recent)
            return "No relevant context found for this query."
        parts = []
        for rank, ix in enumerate(top_ix, 1):
            parts.append(f"[Source {rank} | Relevance {scores[ix]:.3f}]\n{chunks[ix]}")
        return "Context from Knowledge Base + Data Sources:\n\n" + "\n\n---\n\n".join(parts)

    def get_stats(self) -> Dict:
        return dict(self._stats)

    def _build(self):
        print("⏳ RAG: Building knowledge index from all data sources...")
        all_chunks: List[str] = []
        stats: Dict = {}

        # 1. Knowledge base JSON
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                kb = json.load(f)
            kb_chunks = _chunks_from_knowledge_base(kb)
            all_chunks.extend(kb_chunks)
            stats["kb_chunks"] = len(kb_chunks)
            print(f"  ✅ knowledge_base.json → {len(kb_chunks)} chunks")
        except Exception as e:
            print(f"  ⚠ KB load failed: {e}")
            stats["kb_chunks"] = 0

        # 2. NSE Today
        rows = _read_csv_safe(CSV_FILES["nse_today"])
        ch = _chunks_from_nse(rows, "NSE Today")
        all_chunks.extend(ch); stats["nse_today"] = len(ch)
        print(f"  ✅ NSE Today → {len(ch)} chunks ({len(rows)} rows)")

        # 3. NSE WIPRO
        rows = _read_csv_safe(CSV_FILES["nse_wipro"], max_rows=500)
        ch = _chunks_from_nse(rows, "NSE WIPRO")
        all_chunks.extend(ch); stats["nse_wipro"] = len(ch)
        print(f"  ✅ NSE WIPRO → {len(ch)} chunks")

        # 4. NSE INFY
        rows = _read_csv_safe(CSV_FILES["nse_infy"], max_rows=500)
        ch = _chunks_from_nse(rows, "NSE INFY")
        all_chunks.extend(ch); stats["nse_infy"] = len(ch)
        print(f"  ✅ NSE INFY → {len(ch)} chunks")

        # 5. NSE TCS
        rows = _read_csv_safe(CSV_FILES["nse_tcs"], max_rows=500)
        ch = _chunks_from_nse(rows, "NSE TCS")
        all_chunks.extend(ch); stats["nse_tcs"] = len(ch)
        print(f"  ✅ NSE TCS → {len(ch)} chunks")

        # 6. NSE Broad
        rows = _read_csv_safe(CSV_FILES["nse_broad"], max_rows=1000)
        ch = _chunks_from_nse(rows, "NSE India")
        all_chunks.extend(ch); stats["nse_broad"] = len(ch)
        print(f"  ✅ NSE Broad → {len(ch)} chunks")

        # 7. YFinance OHLCV
        rows = _read_csv_safe(CSV_FILES["yfinance"])
        ch = _chunks_from_yfinance(rows)
        all_chunks.extend(ch); stats["yfinance"] = len(ch)
        print(f"  ✅ YFinance → {len(ch)} chunks ({len(rows)} rows)")

        # 8. Scored tweets
        rows = _read_csv_safe(CSV_FILES["scored_tweets"])
        ch = _chunks_from_scored_tweets(rows, max_rows=5000)
        all_chunks.extend(ch); stats["scored_tweets"] = len(ch)
        print(f"  ✅ Scored Tweets → {len(ch)} chunks")

        # 9. Stockerbot
        rows = _read_csv_safe(CSV_FILES["stockerbot"])
        ch = _chunks_from_stockerbot(rows, max_rows=5000)
        all_chunks.extend(ch); stats["stockerbot"] = len(ch)
        print(f"  ✅ StockTwits/Stockerbot → {len(ch)} chunks")

        # 10. Raw stock tweets
        rows = _read_csv_safe(CSV_FILES["stock_tweets"])
        ch = _chunks_from_stock_tweets(rows, max_rows=3000)
        all_chunks.extend(ch); stats["stock_tweets"] = len(ch)
        print(f"  ✅ Stock Tweets → {len(ch)} chunks")

        # 11. Tweet sentiment labels
        rows = _read_csv_safe(CSV_FILES["tweet_sent"], max_rows=3000)
        ch = []
        for row in rows:
            t = _clean(row.get("cleaned_tweets", ""))
            s = str(row.get("sentiment", "0")).strip()
            label_str = "Positive" if s in ("1", "1.0") else "Negative"
            if t:
                ch.append(f"Tweet ({label_str}): {t}")
        all_chunks.extend(ch); stats["tweet_sent"] = len(ch)
        print(f"  ✅ Tweet Sentiment Labels → {len(ch)} chunks")

        # 12. Live RSS
        self._rss_fetcher = RSSFetcher()
        rss_chunks = _chunks_from_rss(self._rss_fetcher.items)
        self._rss_chunks = rss_chunks
        all_chunks.extend(rss_chunks)
        stats["live_rss"] = len(rss_chunks)
        print(f"  ✅ Live RSS → {len(rss_chunks)} chunks")

        # Build TF-IDF
        stats["total_chunks"] = len(all_chunks)
        print(f"⏳ RAG: Fitting TF-IDF over {len(all_chunks)} chunks...")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=80_000,
        )
        matrix = vectorizer.fit_transform(all_chunks)
        print(f"✅ RAG: Index ready — {matrix.shape[0]:,} chunks × {matrix.shape[1]:,} terms")

        with self._lock:
            self._chunks     = all_chunks
            self._vectorizer = vectorizer
            self._matrix     = matrix
            self._stats      = stats

    def _maybe_refresh_rss(self):
        if self._rss_fetcher is None:
            return
        new_rss = _chunks_from_rss(self._rss_fetcher.items)
        if len(new_rss) == len(self._rss_chunks):
            return
        with self._lock:
            old_set    = set(self._rss_chunks)
            new_chunks = [c for c in self._chunks if c not in old_set] + new_rss
            self._rss_chunks = new_rss
            self._chunks     = new_chunks
            try:
                self._matrix = self._vectorizer.transform(new_chunks)
            except Exception:
                pass


# ── Singleton ──────────────────────────────────────────────────────────────────
_engine = RAGEngine()


def retrieve(query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> str:
    return _engine.retrieve(query, top_k=top_k, min_score=min_score)


def get_stats() -> Dict:
    return _engine.get_stats()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RAG RETRIEVAL TEST")
    print("=" * 70)
    for k, v in get_stats().items():
        print(f"  {k}: {v}")
    tests = ["TCS latest", "WIPRO AI", "TSLA sentiment", "NSE India today", "gold news"]
    for q in tests:
        print(f"\nQuery: {q}\n" + "-"*40)
        print(retrieve(q, top_k=2)[:500])