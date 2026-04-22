"""
app/routers/api.py
─────────────────────────────────────────────────────────────────────────────
All API route handlers.

Each route is thin — it handles HTTP concerns (status codes, headers, CORS)
and delegates business logic to the service layer.

Routes implemented:
  GET  /api/analyse/:ticker      Full analysis (rate-limited)
  GET  /api/quote/:ticker        Lightweight quote
  GET  /api/search               Ticker autocomplete
  GET  /api/market/overview      Market indices + movers
  GET  /api/screener             Pre-computed screener
  GET  /api/compare              Side-by-side comparison
  POST /api/session/check        Rate limit status
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any

import yfinance as yf
from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.data.nse_stocks import search_stocks, UNIVERSE, TICKER_TO_META
from app.services.analyser import run_analysis
from app.services.fetcher import fetch_quote_only, fetch_multiple_quotes, TickerNotFoundError
from app.services.cache import (
    cache_get, cache_set,
    analysis_key, quote_key, market_key, screener_key, search_key,
    ANALYSIS_TTL, QUOTE_TTL, MARKET_TTL, SCREENER_TTL, SEARCH_TTL,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Rate limiting (in-memory, per session) ────────────────────────────────
#
#  DISABLE_RATE_LIMIT=true  → skips all limiting (use for local dev)
#  RATE_LIMIT=N             → unique tickers per session window (default 10)
#  RATE_WINDOW_SECONDS=N    → window length in seconds (default 18000 = 5h)
#
#  Same ticker re-analysed within the window is FREE — only unique tickers
#  count against the limit. So refreshing RELIANCE doesn't burn extra slots.
#
_rate_store: dict[str, list[float]] = {}    # session_id → [slot timestamps]
_ticker_store: dict[str, set[str]] = {}     # session_id → {tickers seen}

_RATE_LIMIT  = int(os.getenv("RATE_LIMIT", "10"))
_RATE_WINDOW = int(os.getenv("RATE_WINDOW_SECONDS", str(5 * 60 * 60)))
_DISABLE_RL  = os.getenv("DISABLE_RATE_LIMIT", "false").lower() in ("true", "1", "yes")

if _DISABLE_RL:
    logger.warning("Rate limiting DISABLED (DISABLE_RATE_LIMIT=true) — enable for production")


def _get_session_id(request: Request) -> str:
    """
    Extract session identifier.
    Priority: X-Session-ID header → X-Forwarded-For → direct IP.
    """
    sid = request.headers.get("X-Session-ID", "").strip()
    if sid:
        return sid[:64]
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown"
    )
    return f"ip:{ip}"


def _check_rate_limit(session_id: str, ticker: str) -> tuple[bool, int, float | None]:
    """
    Check rate limit for a session.

    Re-analysing the same ticker within the window does NOT consume a new
    slot — only unique tickers count. This prevents punishing users who
    refresh a stock page, while still limiting broad discovery usage.

    Returns:
        (allowed, unique_tickers_used, next_reset_timestamp_or_None)
    """
    if _DISABLE_RL:
        return True, 0, None

    now = time.time()
    window_start = now - _RATE_WINDOW

    timestamps    = [ts for ts in _rate_store.get(session_id, []) if ts > window_start]
    seen_tickers  = _ticker_store.get(session_id, set())

    # Same ticker within the window — free repeat, no slot consumed
    if ticker in seen_tickers:
        return True, len(timestamps), None

    # New ticker — check limit
    if len(timestamps) >= _RATE_LIMIT:
        oldest = min(timestamps)
        return False, len(timestamps), oldest + _RATE_WINDOW

    # Charge one slot
    timestamps.append(now)
    seen_tickers.add(ticker)
    _rate_store[session_id]  = timestamps
    _ticker_store[session_id] = seen_tickers
    return True, len(timestamps), None


# ── Analyse endpoint ──────────────────────────────────────────────────────

@router.get("/api/analyse/{ticker}")
async def analyse_ticker(ticker: str, request: Request) -> dict[str, Any]:
    """
    Full stock analysis — the core product endpoint.
    Rate limited to RATE_LIMIT analyses per RATE_WINDOW per session.
    Responses cached for ANALYSIS_TTL seconds.
    """
    ticker = ticker.upper().strip()

    # ── Cache check ───────────────────────────────────────────────────────
    cache_k = analysis_key(ticker)
    cached = await cache_get(cache_k)
    if cached:
        logger.info("Cache HIT: %s", ticker)
        # Add cache header so frontend knows this was cached
        return cached

    # ── Rate limit check ──────────────────────────────────────────────────
    session_id = _get_session_id(request)
    allowed, used, next_reset = _check_rate_limit(session_id, ticker)

    if not allowed:
        from datetime import datetime, timezone
        reset_dt = datetime.fromtimestamp(next_reset, tz=timezone.utc).isoformat() if next_reset else None
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"You have used {_RATE_LIMIT} free analyses. Please wait before analysing another stock.",
                "used": used,
                "limit": _RATE_LIMIT,
                "nextResetAt": reset_dt,
            },
        )

    # ── Run analysis ──────────────────────────────────────────────────────
    try:
        result = await run_analysis(ticker)
    except TickerNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "ticker_not_found", "message": f"No data found for '{ticker}'. Check the ticker symbol."},
        ) from exc
    except Exception as exc:
        logger.exception("Analysis failed for %s: %s", ticker, exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "analysis_failed", "message": "Analysis failed. Please try again."},
        ) from exc

    # ── Cache result ──────────────────────────────────────────────────────
    await cache_set(cache_k, result, ttl=ANALYSIS_TTL)

    return result


# ── Quote endpoint ────────────────────────────────────────────────────────

@router.get("/api/quote/{ticker}")
async def get_quote(ticker: str) -> dict[str, Any]:
    """
    Lightweight quote endpoint — price, change, metadata.
    Not rate-limited. Cached for 5 minutes.
    """
    ticker = ticker.upper().strip()

    cache_k = quote_key(ticker)
    cached = await cache_get(cache_k)
    if cached:
        return cached

    from app.data.nse_stocks import resolve_yf_symbol
    from app.services.analyser import _build_quote
    from datetime import datetime, timezone
    import math

    yf_symbol = resolve_yf_symbol(ticker)

    try:
        info = await fetch_quote_only(yf_symbol)
    except TickerNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "ticker_not_found", "message": f"No data for '{ticker}'"},
        ) from exc

    result = _build_quote(ticker, yf_symbol, info)
    await cache_set(cache_k, result, ttl=QUOTE_TTL)
    return result


# ── Search endpoint ───────────────────────────────────────────────────────

@router.get("/api/search")
async def search(q: str = Query("", min_length=0)) -> dict[str, Any]:
    """
    Ticker and company name autocomplete.
    Returns up to 10 results, ranked by relevance.
    """
    q = q.strip()
    if not q:
        # Return popular stocks as suggestions
        popular = [
            TICKER_TO_META[t] for t in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
            if t in TICKER_TO_META
        ]
        return {
            "results": [
                {"ticker": s["ticker"], "name": s["name"],
                 "exchange": s["exchange"], "sector": s["sector"]}
                for s in popular
            ],
            "query": "",
        }

    cache_k = search_key(q)
    cached = await cache_get(cache_k)
    if cached:
        return cached

    results = search_stocks(q, limit=10)
    response = {
        "results": [
            {"ticker": s["ticker"], "name": s["name"],
             "exchange": s["exchange"], "sector": s["sector"]}
            for s in results
        ],
        "query": q,
    }

    await cache_set(cache_k, response, ttl=SEARCH_TTL)
    return response


# ── Market overview endpoint ──────────────────────────────────────────────

@router.get("/api/market/overview")
async def market_overview() -> dict[str, Any]:
    """
    Market overview: indices (Nifty 50, Sensex), trending stocks, top movers.
    Cached for 5 minutes.
    """
    cache_k = market_key()
    cached = await cache_get(cache_k)
    if cached:
        return cached

    # Fetch indices and a sample of movers concurrently
    indices_symbols = ["^NSEI", "^BSESN"]  # Nifty 50, Sensex
    mover_symbols = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "BAJFINANCE.NS", "LT.NS", "KOTAKBANK.NS",
    ]

    all_symbols = indices_symbols + mover_symbols

    try:
        all_infos = await fetch_multiple_quotes(all_symbols)
    except Exception as exc:
        logger.warning("Market overview fetch failed: %s", exc)
        all_infos = {}

    # ── Indices ───────────────────────────────────────────────────────────
    indices = []
    index_meta = {"^NSEI": "Nifty 50", "^BSESN": "Sensex"}

    for sym, name in index_meta.items():
        info = all_infos.get(sym, {})
        if not info:
            continue
        price = info.get("regularMarketPrice") or info.get("currentPrice") or 0
        prev  = info.get("previousClose") or price
        chg   = round(price - prev, 2)
        chg_p = round(((price - prev) / prev) * 100, 2) if prev else 0

        indices.append({
            "name": name,
            "value": round(float(price), 2),
            "change": chg,
            "changePercent": chg_p,
        })

    # ── Movers ────────────────────────────────────────────────────────────
    movers = []
    for sym in mover_symbols:
        info = all_infos.get(sym, {})
        if not info:
            continue
        ticker = sym.replace(".NS", "")
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev  = info.get("previousClose") or price
        chg_p = round(((price - prev) / prev) * 100, 2) if prev else 0

        movers.append({
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName") or ticker,
            "price": round(float(price), 2),
            "changePercent": chg_p,
        })

    # Sort movers
    gainers = sorted([m for m in movers if m["changePercent"] > 0],
                     key=lambda x: x["changePercent"], reverse=True)[:5]
    losers  = sorted([m for m in movers if m["changePercent"] < 0],
                     key=lambda x: x["changePercent"])[:5]
    trending = sorted(movers, key=lambda x: abs(x["changePercent"]), reverse=True)[:5]

    result = {
        "indices":       indices,
        "trending":      trending,
        "topGainers":    gainers,
        "topLosers":     losers,
        "popularSearches": [
            "Infosys", "ICICI Bank", "Larsen & Toubro",
            "Tata Motors", "Wipro", "Bharti Airtel",
        ],
    }

    await cache_set(cache_k, result, ttl=MARKET_TTL)
    return result


# ── Screener endpoint ─────────────────────────────────────────────────────

@router.get("/api/screener")
async def screener(
    sector: str = Query(""),
    min_pe: float = Query(0),
    max_pe: float = Query(1000),
    min_roe: float = Query(0),
    min_revenue_growth: float = Query(-100),
    max_de: float = Query(100),
    min_rsi: float = Query(0),
    max_rsi: float = Query(100),
    supertrend: str = Query(""),    # "Bullish" | "Bearish" | ""
    sort_by: str = Query("marketCap"),
    sort_dir: str = Query("desc"),
    limit: int = Query(100),
) -> dict[str, Any]:
    """
    Screener with filters.
    Results are cached; filtered/sorted on each request.
    """
    # Build a hash of filter params to create a cache key for exact filter combos
    params_hash = hashlib.md5(
        json.dumps(locals(), sort_keys=True, default=str).encode()
    ).hexdigest()[:8]

    # Try full screener cache first
    base_cache_k = screener_key()
    all_stocks = await cache_get(base_cache_k)

    if not all_stocks:
        # Build screener data by fetching a subset of stocks
        all_stocks = await _build_screener_data()
        await cache_set(base_cache_k, all_stocks, ttl=SCREENER_TTL)

    # Apply filters
    filtered = all_stocks
    if sector:
        filtered = [s for s in filtered if s.get("sector", "").lower() == sector.lower()]
    filtered = [s for s in filtered if min_pe <= s.get("pe", 0) <= max_pe]
    filtered = [s for s in filtered if s.get("roe", 0) >= min_roe]
    filtered = [s for s in filtered if s.get("revenueGrowth", 0) >= min_revenue_growth]
    filtered = [s for s in filtered if s.get("debtEquity", 0) <= max_de]
    filtered = [s for s in filtered if min_rsi <= s.get("rsi", 50) <= max_rsi]
    if supertrend:
        filtered = [s for s in filtered if s.get("supertrend", "") == supertrend]

    # Sort
    reverse = sort_dir.lower() == "desc"
    try:
        filtered.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)
    except TypeError:
        pass

    return {
        "results": filtered[:limit],
        "total": len(filtered),
    }


async def _build_screener_data() -> list[dict]:
    """
    Build screener data for top 100 stocks.
    Fetches quotes concurrently in batches of 20.
    Only lightweight quote data — no full analysis.
    """
    # Pick top 100 stocks from universe for screener
    screener_tickers = [s for s in UNIVERSE[:100]]
    symbols = [s["yf_symbol"] for s in screener_tickers]

    # Fetch in batches of 20 to avoid hammering Yahoo
    BATCH = 20
    all_infos: dict = {}
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        try:
            batch_infos = await fetch_multiple_quotes(batch)
            all_infos.update(batch_infos)
        except Exception as exc:
            logger.warning("Screener batch %d failed: %s", i, exc)
        # Small delay between batches to be polite to Yahoo
        if i + BATCH < len(symbols):
            await asyncio.sleep(0.5)

    results = []
    for stock_meta in screener_tickers:
        sym   = stock_meta["yf_symbol"]
        info  = all_infos.get(sym, {})
        if not info:
            continue

        ticker = stock_meta["ticker"]
        price  = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev   = info.get("previousClose") or price
        chg_p  = round(((price - prev) / prev) * 100, 2) if prev else 0

        # D/E normalisation
        de = info.get("debtToEquity")
        if de is not None:
            try:
                de = float(de)
                de = de / 100 if de > 10 else de
            except (TypeError, ValueError):
                de = 0.0
        else:
            de = 0.0

        market_cap = info.get("marketCap") or 0
        market_cap_cr = round(market_cap / 10_000_000, 0)

        # RSI: not in yfinance info — approximate from momentum
        # True RSI requires OHLCV; use 50 as neutral default
        rsi_approx = 50.0

        # Revenue growth
        rev_growth = 0.0
        rg = info.get("revenueGrowth")
        if rg is not None:
            try:
                rev_growth = round(float(rg) * 100, 1)
            except (TypeError, ValueError):
                pass

        roe = 0.0
        roe_raw = info.get("returnOnEquity")
        if roe_raw is not None:
            try:
                roe = round(float(roe_raw) * 100, 1)
            except (TypeError, ValueError):
                pass

        pe = 0.0
        pe_raw = info.get("trailingPE") or info.get("forwardPE")
        if pe_raw is not None:
            try:
                pe_val = float(pe_raw)
                if 0 < pe_val < 1000:
                    pe = round(pe_val, 1)
            except (TypeError, ValueError):
                pass

        results.append({
            "ticker":         ticker,
            "name":           info.get("shortName") or info.get("longName") or ticker,
            "sector":         stock_meta["sector"],
            "marketCap":      market_cap_cr,
            "price":          round(float(price), 2),
            "changePercent":  chg_p,
            "pe":             pe,
            "roe":            roe,
            "revenueGrowth":  rev_growth,
            "debtEquity":     round(de, 2),
            "rsi":            rsi_approx,
            "supertrend":     "Bullish" if chg_p > 0 else "Bearish",  # Proxy
            "topPersona":     "Growth Investor",  # Default until full analysis runs
            "personaScore":   0,
        })

    return results


# ── Compare endpoint ──────────────────────────────────────────────────────

@router.get("/api/compare")
async def compare_stocks(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. RELIANCE,TCS"),
) -> dict[str, Any]:
    """
    Side-by-side comparison of 2–3 stocks.
    Runs full analysis for each (uses cache when available).
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:3]

    if len(ticker_list) < 2:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": "Provide at least 2 tickers."},
        )

    # Run analyses concurrently
    tasks = [run_analysis(t) for t in ticker_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    stocks = []
    for ticker, result in zip(ticker_list, results):
        if isinstance(result, Exception):
            logger.warning("Compare failed for %s: %s", ticker, result)
            continue
        stocks.append(result)

    if not stocks:
        raise HTTPException(
            status_code=500,
            detail={"error": "compare_failed", "message": "All requested tickers failed."},
        )

    return {"stocks": stocks}


# ── Session check endpoint ────────────────────────────────────────────────

@router.post("/api/session/check")
async def session_check(request: Request) -> dict[str, Any]:
    """
    Returns the current rate limit status for this session.
    Does NOT consume a rate limit slot.
    Reports unique tickers used (not raw request count).
    """
    if _DISABLE_RL:
        return {"used": 0, "remaining": _RATE_LIMIT, "limit": _RATE_LIMIT, "nextResetAt": None}

    session_id = _get_session_id(request)
    now = time.time()
    window_start = now - _RATE_WINDOW

    timestamps   = [ts for ts in _rate_store.get(session_id, []) if ts > window_start]
    used         = len(timestamps)           # unique tickers charged
    remaining    = max(0, _RATE_LIMIT - used)

    next_reset = None
    if used >= _RATE_LIMIT and timestamps:
        from datetime import datetime, timezone
        reset_ts   = min(timestamps) + _RATE_WINDOW
        next_reset = datetime.fromtimestamp(reset_ts, tz=timezone.utc).isoformat()

    return {
        "used":        used,
        "remaining":   remaining,
        "limit":       _RATE_LIMIT,
        "nextResetAt": next_reset,
    }


# ── Health check ──────────────────────────────────────────────────────────

@router.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health check for Railway/Render uptime monitoring."""
    return {"status": "ok", "service": "alphavibes-backend"}
