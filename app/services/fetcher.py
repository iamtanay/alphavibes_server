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

FIX (Roadmap Phase 1):
  • income_stmt is now the PRIMARY source for annual P&L (replaces `financials`
    which returns empty DataFrames for most NSE stocks in yfinance ≥ 0.2.x).
  • When income_stmt is also empty (common for mid-cap NSE), key P&L rows are
    reconstructed from the info dict: totalRevenue, netIncomeToCommon,
    grossProfits, ebitda, operatingCashflow. This ensures at least 1 year of
    data is always available.
  • get_income_stmt(freq="yearly") is tried as a final fallback with explicit
    per-column error handling.
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
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
    financials: pd.DataFrame           # Annual income statement (primary: income_stmt)
    quarterly_financials: pd.DataFrame # Quarterly income statement
    balance_sheet: pd.DataFrame        # Annual balance sheet
    quarterly_balance_sheet: pd.DataFrame
    cashflow: pd.DataFrame             # Annual cash flow statement
    quarterly_cashflow: pd.DataFrame


# ── P&L reconstruction from info dict ─────────────────────────────────────

def _reconstruct_financials_from_info(info: dict) -> pd.DataFrame:
    """
    When income_stmt returns an empty DataFrame (common for mid-cap NSE),
    reconstruct a single-year P&L from the info dict fields.

    This provides at least 1 year of data — better than nothing.
    Fields sourced: totalRevenue, grossProfits, ebitda, netIncomeToCommon,
    operatingCashflow, earningsBeforeTax.
    """
    row_map = {
        "Total Revenue":    info.get("totalRevenue"),
        "Gross Profit":     info.get("grossProfits"),
        "EBITDA":           info.get("ebitda"),
        "Operating Income": info.get("operatingIncome") or info.get("ebit"),
        "Net Income":       info.get("netIncomeToCommon"),
        "Basic EPS":        info.get("trailingEps") or info.get("epsTrailingTwelveMonths"),
    }

    data = {}
    for row_name, val in row_map.items():
        if val is not None:
            try:
                fval = float(val)
                if not (pd.isna(fval) or np.isinf(fval)):
                    data[row_name] = fval
            except (TypeError, ValueError):
                pass

    if not data:
        return pd.DataFrame()

    # Use current year as the single column
    import datetime
    col = pd.Timestamp(datetime.datetime.now().year, 3, 31)  # FY end approx
    df = pd.DataFrame(data, index=[col]).T
    logger.debug("Reconstructed financials from info dict: %d rows", len(df))
    return df


def _try_get_income_stmt(ticker: yf.Ticker, symbol: str) -> pd.DataFrame:
    """
    Try multiple yfinance APIs to get the annual income statement.
    Priority order:
      1. ticker.income_stmt            (yfinance ≥ 0.2.x preferred API)
      2. ticker.financials             (legacy alias — may work on some versions)
      3. ticker.get_income_stmt(freq='yearly')  (explicit freq call)
      4. Reconstruct from info dict    (always returns something)
    """
    # 1. income_stmt (primary — yfinance ≥ 0.2.x)
    try:
        df = ticker.income_stmt
        if df is not None and not df.empty and len(df.columns) > 0:
            logger.debug("income_stmt succeeded for %s (%d cols)", symbol, len(df.columns))
            return df
    except Exception as exc:
        logger.debug("income_stmt failed for %s: %s", symbol, exc)

    # 2. Legacy financials attribute
    try:
        df = ticker.financials
        if df is not None and not df.empty and len(df.columns) > 0:
            logger.debug("financials (legacy) succeeded for %s", symbol)
            return df
    except Exception as exc:
        logger.debug("financials (legacy) failed for %s: %s", symbol, exc)

    # 3. Explicit get_income_stmt call
    try:
        df = ticker.get_income_stmt(freq="yearly")
        if df is not None and not df.empty and len(df.columns) > 0:
            logger.debug("get_income_stmt succeeded for %s", symbol)
            return df
    except Exception as exc:
        logger.debug("get_income_stmt failed for %s: %s", symbol, exc)

    logger.info("All income stmt attempts failed for %s — will reconstruct from info", symbol)
    return pd.DataFrame()


def _try_get_quarterly_financials(ticker: yf.Ticker, symbol: str) -> pd.DataFrame:
    """Try quarterly income statement via multiple APIs."""
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        try:
            df = getattr(ticker, attr)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    try:
        df = ticker.get_income_stmt(freq="quarterly")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


# ── Internal sync fetcher (runs in thread pool) ────────────────────────────

def _fetch_sync(symbol: str) -> RawStockData:
    """
    Synchronous yfinance fetch — DO NOT call from async code directly.
    Use fetch_stock_data() which wraps this in asyncio.to_thread().

    Raises TickerNotFoundError if the ticker is invalid.
    """
    ticker = yf.Ticker(symbol)

    # ── info dict ────────────────────────────────────────────────────────
    try:
        info = ticker.info
    except Exception as exc:
        raise TickerNotFoundError(f"Could not fetch info for {symbol}: {exc}") from exc

    if not info or info.get("quoteType") is None:
        raise TickerNotFoundError(f"No data found for ticker: {symbol}")

    # ── OHLCV history ─────────────────────────────────────────────────────
    try:
        history_5y = ticker.history(period="5y", interval="1d", auto_adjust=True)
    except Exception as exc:
        logger.warning("Could not fetch 5Y history for %s: %s", symbol, exc)
        history_5y = pd.DataFrame()

    if not history_5y.empty:
        one_year_ago = pd.Timestamp.now(tz=history_5y.index.tz) - pd.DateOffset(years=1)
        history_1y = history_5y[history_5y.index >= one_year_ago].copy()
    else:
        history_1y = pd.DataFrame()

    # ── Financial statements ──────────────────────────────────────────────

    def safe_get(attr: str) -> pd.DataFrame:
        """Fetch a financial statement; return empty DataFrame on failure."""
        try:
            df = getattr(ticker, attr)
            return df if df is not None and not df.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("Could not fetch %s for %s: %s", attr, symbol, exc)
            return pd.DataFrame()

    # ── P&L: use income_stmt as primary, reconstruct from info as fallback ─
    annual_financials = _try_get_income_stmt(ticker, symbol)
    if annual_financials.empty:
        annual_financials = _reconstruct_financials_from_info(info)
        if not annual_financials.empty:
            logger.info(
                "Using info-dict reconstructed financials for %s (1 year of data)", symbol
            )

    quarterly_financials = _try_get_quarterly_financials(ticker, symbol)

    return RawStockData(
        symbol=symbol,
        info=info,
        history_1y=history_1y,
        history_5y=history_5y,
        financials=annual_financials,
        quarterly_financials=quarterly_financials,
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
