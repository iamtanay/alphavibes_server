"""
app/services/analyser.py
─────────────────────────────────────────────────────────────────────────────
Analysis orchestrator.

This is the single function that the /api/analyse/:ticker route calls.
It:
  1. Fetches raw data from yfinance (via fetcher.py)
  2. Runs all engines in parallel where possible
  3. Assembles the final AnalysisResponse dict
  4. Is completely decoupled from HTTP concerns

FIXES (Roadmap Phase 1 + 4):
  • FIX #3  — promoter_holding and promoter_pledge are now patched into
    StockMetrics AFTER compute_shareholding() runs, before compute_personas().
    This makes Jhunjhunwala + Lynch personas use real promoter data.
  • FIX #6  — Request deduplication via asyncio.Event(). If two requests
    for the same ticker arrive simultaneously (both miss cache), only ONE
    actually runs the yfinance fetch. The second waits on an asyncio.Event
    and reads the result when the first completes. Prevents double-fetching
    and race conditions under concurrent load.
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.services.fetcher import fetch_stock_data, TickerNotFoundError
from app.services.technicals import compute_technicals, build_chart_data
from app.services.fundamentals import compute_fundamentals
from app.services.personas import compute_personas, extract_metrics
from app.services.shareholding import compute_shareholding
from app.data.nse_stocks import TICKER_TO_META, get_peers, resolve_yf_symbol

logger = logging.getLogger(__name__)

# 1 Crore = 10,000,000 INR
_CR = 10_000_000

# ── Request deduplication ──────────────────────────────────────────────────
#
# Maps ticker → asyncio.Event that is set once the analysis completes.
# A concurrent request for the same ticker waits on this event, then reads
# the result from _dedup_results.
#
# This handles the thundering herd problem: many simultaneous requests for
# a popular stock all miss the cache and would otherwise each spawn a 20s
# yfinance fetch. With deduplication, only the first request runs the fetch;
# all others wait and get the same result instantly.
#
_dedup_events: dict[str, asyncio.Event] = {}
_dedup_results: dict[str, Any] = {}
_dedup_lock = asyncio.Lock()


def _safe_float(v: Any, d: int = 2) -> float:
    """Safe float with fallback to 0.0."""
    try:
        import math
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, d)
    except (TypeError, ValueError):
        return 0.0


async def run_analysis(ticker: str) -> dict[str, Any]:
    """
    Full analysis pipeline for one stock, with request deduplication.

    Args:
        ticker: NSE ticker symbol, e.g. "RELIANCE" or "RELIANCE.NS"

    Returns:
        Dict matching the TypeScript AnalysisResponse interface exactly.

    Raises:
        TickerNotFoundError: if the ticker is invalid
    """
    yf_symbol = resolve_yf_symbol(ticker)
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")

    # ── Deduplication check ───────────────────────────────────────────────
    async with _dedup_lock:
        if clean_ticker in _dedup_events:
            # Another coroutine is already fetching this ticker — wait for it
            event = _dedup_events[clean_ticker]
            logger.info("Dedup wait: %s (another fetch in progress)", clean_ticker)
            should_wait = True
        else:
            # We are the first — register ourselves as the fetcher
            event = asyncio.Event()
            _dedup_events[clean_ticker] = event
            should_wait = False

    if should_wait:
        # Wait for the primary fetch to complete (with a 60s safety timeout)
        try:
            await asyncio.wait_for(event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("Dedup wait timed out for %s — running own fetch", clean_ticker)
        else:
            # Primary succeeded; return its cached result
            result = _dedup_results.get(clean_ticker)
            if result is not None:
                logger.info("Dedup hit: served %s from in-flight result", clean_ticker)
                return result
            # If result is absent (primary raised), fall through to own fetch

    # ── Primary fetch path ────────────────────────────────────────────────
    try:
        result = await _run_analysis_inner(clean_ticker, yf_symbol)
    except Exception:
        # Signal waiting coroutines so they don't hang forever
        async with _dedup_lock:
            _dedup_events.pop(clean_ticker, None)
            _dedup_results.pop(clean_ticker, None)
        if clean_ticker in _dedup_events:
            event.set()
        raise

    # Store result and signal waiters
    async with _dedup_lock:
        _dedup_results[clean_ticker] = result
        _dedup_events.pop(clean_ticker, None)

    event.set()

    # Cleanup result after a short window (prevent unbounded memory growth)
    async def _cleanup():
        await asyncio.sleep(5)
        _dedup_results.pop(clean_ticker, None)

    asyncio.create_task(_cleanup())

    return result


async def _run_analysis_inner(clean_ticker: str, yf_symbol: str) -> dict[str, Any]:
    """
    Inner analysis pipeline — called by run_analysis() after deduplication.
    This does the actual work.
    """
    logger.info("Starting analysis for %s (%s)", clean_ticker, yf_symbol)

    # ── Step 1: Fetch all raw data ────────────────────────────────────────
    raw = await fetch_stock_data(yf_symbol)
    info = raw.info

    # ── Step 2: Build technical analysis (CPU-bound but fast) ─────────────
    technicals = await asyncio.to_thread(
        compute_technicals,
        raw.history_5y,
        raw.history_1y,
    )

    # ── Step 3: Build chart data ──────────────────────────────────────────
    chart_data = await asyncio.to_thread(
        build_chart_data,
        raw.history_1y,
        raw.history_5y,
    )

    # ── Step 4: Build fundamentals ────────────────────────────────────────
    fundamentals = await asyncio.to_thread(
        compute_fundamentals,
        info,
        raw.financials,
        raw.quarterly_financials,
        raw.balance_sheet,
        raw.quarterly_balance_sheet,
        raw.cashflow,
        raw.quarterly_cashflow,
    )

    # ── Step 5: Shareholding pattern ─────────────────────────────────────
    # Run BEFORE extract_metrics so promoter_holding is available for personas.
    shareholding = await asyncio.to_thread(compute_shareholding, info, clean_ticker)

    # ── Step 6: Extract flat metrics ──────────────────────────────────────
    metrics = await asyncio.to_thread(
        extract_metrics,
        info,
        raw.history_5y,
        raw.financials,
        technicals,
    )

    # ── FIX #3: Patch promoter data into metrics ──────────────────────────
    # extract_metrics() sets promoter_holding = None (cannot get from info).
    # compute_shareholding() already extracted it from heldPercentInsiders.
    # Wire them together here so Jhunjhunwala + Lynch personas get real data.
    try:
        metrics.promoter_holding = shareholding.get("promoter")        # float in %
        metrics.promoter_pledge  = shareholding.get("promoterPledge")  # float in %
        metrics.fii_holding      = shareholding.get("fii")             # float in %
    except Exception as exc:
        logger.warning("Could not patch promoter holding into metrics: %s", exc)

    # ── Step 7: Score all personas ────────────────────────────────────────
    persona_result = await asyncio.to_thread(compute_personas, metrics)
    personas  = persona_result["personas"]
    conflicts = persona_result["conflicts"]

    # ── Step 8: Peer comparison ───────────────────────────────────────────
    peers = await _build_peers(clean_ticker, info)

    # ── Step 9: Build Quote object ────────────────────────────────────────
    quote = _build_quote(clean_ticker, yf_symbol, info)

    # ── Assemble final response ───────────────────────────────────────────
    return {
        "ticker":       clean_ticker,
        "quote":        quote,
        "technical":    technicals,
        "fundamental":  fundamentals,
        "personas":     personas,
        "conflicts":    conflicts,
        "shareholding": shareholding,
        "peers":        peers,
        "chartData":    chart_data,
    }


def _build_quote(ticker: str, yf_symbol: str, info: dict) -> dict[str, Any]:
    """Build the Quote object from yfinance info."""
    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or 0.0
    )

    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or price
    change = _safe_float(price - prev_close, 2)
    change_pct = _safe_float(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0

    market_cap = info.get("marketCap") or 0
    market_cap_cr = round(market_cap / _CR, 0) if market_cap else 0

    return {
        "ticker":        ticker,
        "name":          info.get("longName") or info.get("shortName") or ticker,
        "exchange":      "NSE" if yf_symbol.endswith(".NS") else "BSE",
        "price":         _safe_float(price, 2),
        "change":        change,
        "changePercent": change_pct,
        "volume":        int(info.get("volume") or info.get("regularMarketVolume") or 0),
        "marketCap":     market_cap_cr,
        "sector":        info.get("sector") or info.get("sectorDisp") or "N/A",
        "industry":      info.get("industry") or info.get("industryDisp") or "N/A",
        "week52High":    _safe_float(info.get("fiftyTwoWeekHigh"), 2),
        "week52Low":     _safe_float(info.get("fiftyTwoWeekLow"), 2),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "isDelayed":     True,  # yfinance is always EOD/delayed
    }


async def _build_peers(ticker: str, info: dict) -> list[dict[str, Any]]:
    """
    Build peer comparison data by fetching key metrics for sector peers.
    Uses cached quotes where available to minimise yfinance calls.
    Falls back gracefully if peer data is unavailable.
    """
    from app.services.fetcher import fetch_multiple_quotes

    peer_metas = get_peers(ticker, limit=5)
    if not peer_metas:
        return []

    peer_symbols = [p["yf_symbol"] for p in peer_metas]

    try:
        peer_infos = await fetch_multiple_quotes(peer_symbols)
    except Exception as exc:
        logger.warning("Peer fetch failed: %s", exc)
        peer_infos = {}

    peers = []

    # Add the current stock first (highlighted)
    peers.append({
        "ticker":     ticker,
        "name":       info.get("longName") or info.get("shortName") or ticker,
        "pe":         _safe_float(info.get("trailingPE") or info.get("forwardPE"), 1),
        "roe":        _safe_float((info.get("returnOnEquity") or 0) * 100, 1),
        "netMargin":  _safe_float((info.get("profitMargins") or 0) * 100, 1),
        "debtEquity": _safe_float(_normalise_de(info.get("debtToEquity")), 2),
        "isSelected": True,
    })

    for meta in peer_metas:
        sym = meta["yf_symbol"]
        p_info = peer_infos.get(sym, {})
        if not p_info:
            continue
        peers.append({
            "ticker":     meta["ticker"],
            "name":       p_info.get("longName") or p_info.get("shortName") or meta["name"],
            "pe":         _safe_float(p_info.get("trailingPE") or p_info.get("forwardPE"), 1),
            "roe":        _safe_float((p_info.get("returnOnEquity") or 0) * 100, 1),
            "netMargin":  _safe_float((p_info.get("profitMargins") or 0) * 100, 1),
            "debtEquity": _safe_float(_normalise_de(p_info.get("debtToEquity")), 2),
            "isSelected": False,
        })

    return peers[:6]  # Max 6 including selected stock


def _normalise_de(de: Any) -> float:
    """Normalise D/E — yfinance sometimes returns in % (28 = 0.28)."""
    if de is None:
        return 0.0
    try:
        f = float(de)
        return f / 100 if f > 10 else f
    except (TypeError, ValueError):
        return 0.0
