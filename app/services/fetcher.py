"""
app/services/fetcher.py
─────────────────────────────────────────────────────────────────────────────
Thin async wrapper around yfinance.

Design decisions:
  • yfinance is synchronous — we run it in a thread-pool via asyncio.to_thread()
    so it never blocks the FastAPI event loop.
  • Every public function returns plain dicts/DataFrames, no yfinance objects
    leak out — so swapping the data provider later is a one-file change.
  • All network errors are caught here and re-raised as TickerNotFoundError
    or DataUnavailableError so the rest of the app has clean error semantics.
  • We fetch the MAXIMUM data in one call and let callers slice — minimises
    the number of round-trips to Yahoo Finance.
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────

class TickerNotFoundError(Exception):
    """Raised when yfinance returns no data for a symbol."""


class DataUnavailableError(Exception):
    """Raised when a specific data field is missing (e.g. no financials)."""


# ── Raw data container ────────────────────────────────────────────────────

@dataclass
class RawStockData:
    """
    Everything we'll ever need from yfinance for one stock, fetched in
    as few network calls as possible (yf.Ticker is lazy — it batches).
    """
    symbol: str
    info: dict[str, Any]               # Metadata + fundamentals
    history_1y: pd.DataFrame           # Daily OHLCV — 1 year (for chart + TA)
    history_5y: pd.DataFrame           # Daily OHLCV — 5 years (for financial trends)
    financials: pd.DataFrame           # Annual income statement
    quarterly_financials: pd.DataFrame # Quarterly income statement
    balance_sheet: pd.DataFrame        # Annual balance sheet
    quarterly_balance_sheet: pd.DataFrame
    cashflow: pd.DataFrame             # Annual cash flow statement
    quarterly_cashflow: pd.DataFrame


# ── Internal sync fetcher (runs in thread pool) ────────────────────────────

def _fetch_sync(symbol: str) -> RawStockData:
    """
    Synchronous yfinance fetch — DO NOT call from async code directly.
    Use fetch_stock_data() which wraps this in asyncio.to_thread().

    Raises TickerNotFoundError if the ticker is invalid.
    """
    ticker = yf.Ticker(symbol)

    # ── info dict ────────────────────────────────────────────────────────
    # This is the most important call — it returns 100+ fields
    try:
        info = ticker.info
    except Exception as exc:
        raise TickerNotFoundError(f"Could not fetch info for {symbol}: {exc}") from exc

    # Validate — yfinance returns a minimal dict for invalid tickers
    if not info or info.get("quoteType") is None:
        raise TickerNotFoundError(f"No data found for ticker: {symbol}")

    # ── OHLCV history ─────────────────────────────────────────────────────
    # We fetch 5 years once — the 1Y slice is derived from it
    try:
        history_5y = ticker.history(period="5y", interval="1d", auto_adjust=True)
    except Exception as exc:
        logger.warning("Could not fetch 5Y history for %s: %s", symbol, exc)
        history_5y = pd.DataFrame()

    # 1Y slice for chart rendering and TA computation
    if not history_5y.empty:
        one_year_ago = pd.Timestamp.now(tz=history_5y.index.tz) - pd.DateOffset(years=1)
        history_1y = history_5y[history_5y.index >= one_year_ago].copy()
    else:
        history_1y = pd.DataFrame()

    # ── Financial statements ──────────────────────────────────────────────
    # These are lazily fetched by yfinance — each property is one HTTP call
    def safe_get(attr: str) -> pd.DataFrame:
        """Fetch a financial statement; return empty DataFrame on failure."""
        try:
            df = getattr(ticker, attr)
            return df if df is not None and not df.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("Could not fetch %s for %s: %s", attr, symbol, exc)
            return pd.DataFrame()

    return RawStockData(
        symbol=symbol,
        info=info,
        history_1y=history_1y,
        history_5y=history_5y,
        financials=safe_get("financials"),              # annual income stmt
        quarterly_financials=safe_get("quarterly_financials"),
        balance_sheet=safe_get("balance_sheet"),
        quarterly_balance_sheet=safe_get("quarterly_balance_sheet"),
        cashflow=safe_get("cashflow"),
        quarterly_cashflow=safe_get("quarterly_cashflow"),
    )


# ── Public async API ──────────────────────────────────────────────────────

async def fetch_stock_data(symbol: str) -> RawStockData:
    """
    Async entry point. Fetches all yfinance data for `symbol` in a
    thread-pool so the event loop is never blocked.

    Args:
        symbol: yfinance symbol, e.g. "RELIANCE.NS"

    Returns:
        RawStockData with all fields populated (some may be empty DataFrames
        if data is unavailable — callers must handle gracefully).

    Raises:
        TickerNotFoundError: if the ticker is invalid or has no data
    """
    logger.info("Fetching data for %s", symbol)
    return await asyncio.to_thread(_fetch_sync, symbol)


async def fetch_quote_only(symbol: str) -> dict[str, Any]:
    """
    Lightweight fetch — only the info dict (price, change, metadata).
    Much faster than full fetch; used for the /api/quote endpoint.
    """
    def _fetch() -> dict[str, Any]:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or info.get("quoteType") is None:
            raise TickerNotFoundError(f"No data for {symbol}")
        return info

    return await asyncio.to_thread(_fetch)


async def fetch_multiple_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """
    Fetch quotes for multiple symbols concurrently.
    Used by screener and market overview endpoints.

    Returns dict mapping symbol → info dict.
    Symbols that fail are silently omitted from the result.
    """
    async def _safe_fetch(sym: str) -> tuple[str, dict | None]:
        try:
            info = await fetch_quote_only(sym)
            return sym, info
        except Exception as exc:
            logger.warning("Failed to fetch quote for %s: %s", sym, exc)
            return sym, None

    tasks = [_safe_fetch(sym) for sym in symbols]
    results = await asyncio.gather(*tasks)

    return {sym: info for sym, info in results if info is not None}
