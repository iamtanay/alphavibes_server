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

Architecture decision — runtime vs pre-computed:
─────────────────────────────────────────────────────────────────────────────
DECISION: Runtime computation with Redis caching (15-minute TTL)

Rationale for runtime (not pre-computed batch):

  PRO runtime:
  • Budget = ₹0 for storage/compute. Pre-computing 500 stocks daily needs
    a persistent server + ~500 yfinance calls every night = hit Yahoo's
    rate limit hard, get blocked.
  • User base is small in V1 — no need to pre-compute everything.
  • Cache handles repeated requests (same stock, multiple users = 1 fetch).
  • Any stock (not just Nifty 500) can be analysed on demand.

  PRO pre-computed:
  • Screener needs all 500 stocks — we DO pre-compute the screener subset
    (see screener.py) but that's just 15 fields per stock, not the full
    AnalysisResponse.
  • Screener batch is separate from the full analysis pipeline.

  CONCLUSION:
  • Full analysis = runtime with 15-min Redis cache.
  • Screener data = nightly batch for the top 200 stocks only.
  • Quote endpoint = runtime with 5-min cache (lightweight).

This means: zero cost for idle time, scales to ~100 requests/day on
Railway's free tier without issues.
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
    Full analysis pipeline for one stock.

    Args:
        ticker: NSE ticker symbol, e.g. "RELIANCE" or "RELIANCE.NS"

    Returns:
        Dict matching the TypeScript AnalysisResponse interface exactly.

    Raises:
        TickerNotFoundError: if the ticker is invalid
    """
    # Resolve ticker to yfinance symbol
    yf_symbol = resolve_yf_symbol(ticker)
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")

    logger.info("Starting analysis for %s (%s)", clean_ticker, yf_symbol)

    # ── Step 1: Fetch all raw data ────────────────────────────────────────
    raw = await fetch_stock_data(yf_symbol)
    info = raw.info

    # ── Step 2: Build technical analysis (CPU-bound but fast) ─────────────
    # Run in thread pool to avoid blocking event loop on large DataFrames
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

    # ── Step 5: Extract flat metrics and score all personas ───────────────
    metrics = await asyncio.to_thread(
        extract_metrics,
        info,
        raw.history_5y,
        raw.financials,
        technicals,
    )

    personas = await asyncio.to_thread(compute_personas, metrics)

    # ── Step 6: Shareholding pattern ─────────────────────────────────────
    shareholding = await asyncio.to_thread(compute_shareholding, info, clean_ticker)

    # ── Step 7: Peer comparison ───────────────────────────────────────────
    peers = await _build_peers(clean_ticker, info)

    # ── Step 8: Build Quote object ────────────────────────────────────────
    quote = _build_quote(clean_ticker, yf_symbol, info)

    # ── Assemble final response ───────────────────────────────────────────
    return {
        "ticker":      clean_ticker,
        "quote":       quote,
        "technical":   technicals,
        "fundamental": fundamentals,
        "personas":    personas,
        "shareholding": shareholding,
        "peers":       peers,
        "chartData":   chart_data,
    }


def _build_quote(ticker: str, yf_symbol: str, info: dict) -> dict[str, Any]:
    """Build the Quote object from yfinance info."""
    # Current price
    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or 0.0
    )

    # Price change
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

    # Fetch peer quotes concurrently (lightweight — just info dict)
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

    # Add peers
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
