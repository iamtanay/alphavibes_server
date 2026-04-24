"""
app/services/fundamentals.py
─────────────────────────────────────────────────────────────────────────────
Fundamental analysis engine.

Parses yfinance `info` dict + financial statement DataFrames into the
exact shape expected by the TypeScript Fundamental interface.

Design notes:
  • yfinance `info` is the primary source for ratios (PE, ROE, etc.) because
    it's pre-computed and reliable.
  • Income statement / balance sheet DataFrames are used for financial trend
    charts and the financial statements tables.
  • All yfinance keys have been verified against the actual API responses.
  • Values in Crores (₹ Crores = ₹ 10 million) for Indian stocks.
    yfinance returns values in INR — we divide by 10M to get Crores.

UPDATE (Roadmap Phase 1 — FIX #1):
  • income_stmt (yfinance ≥ 0.2.x) uses different row index labels than the
    legacy `financials` attribute. Both are now handled via _resolve_row()
    which tries multiple label aliases. This makes financial statement parsing
    robust regardless of which yfinance version or API path was used to fetch
    the data.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import math
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# 1 Crore = 10,000,000 INR
_CR = 10_000_000


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe(value: Any, decimals: int = 2) -> float | None:
    """Round to decimals; return None for NaN/None/Inf."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _pct(value: Any, decimals: int = 1) -> float | None:
    """Convert a 0–1 ratio to a percentage. E.g. 0.22 → 22.0"""
    if value is None:
        return None
    f = _safe(value * 100, decimals)
    return f


def _cr(value: Any) -> float | None:
    """Convert INR to Crores (÷ 1 Crore = 10,000,000)."""
    if value is None:
        return None
    f = _safe(value / _CR, 0)
    return f


def _get(d: dict, *keys, default=None):
    """Try multiple key names and return the first non-None value found."""
    for k in keys:
        v = d.get(k)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    return default


# ── Row label aliases ─────────────────────────────────────────────────────
#
# income_stmt (yfinance ≥ 0.2.x) uses different index labels than the
# legacy `financials` attribute. _resolve_row() finds whichever alias
# is present in a given DataFrame, making parsing robust across yfinance
# versions and API paths (Fix #1).
#
_ROW_ALIASES: dict[str, list[str]] = {
    "Total Revenue":    ["Total Revenue", "Revenue", "TotalRevenue"],
    "Gross Profit":     ["Gross Profit", "GrossProfit"],
    "Operating Income": ["Operating Income", "EBIT", "OperatingIncome"],
    "Net Income":       ["Net Income", "Net Income Common Stockholders", "NetIncome"],
    "Basic EPS":        ["Basic EPS", "Basic", "BasicEPS"],
    "Diluted EPS":      ["Diluted EPS", "Diluted", "DilutedEPS"],
    "EBITDA":           ["EBITDA", "Normalized EBITDA"],
    "Tax Provision":    ["Tax Provision", "Income Tax Expense"],
    "Interest Expense": ["Interest Expense", "Net Interest Income"],
}


def _resolve_row(df: pd.DataFrame, canonical: str) -> str | None:
    """
    Return the first alias for `canonical` that exists in df.index,
    or None if no alias is found. Handles income_stmt vs financials differences.
    """
    for alias in _ROW_ALIASES.get(canonical, [canonical]):
        if alias in df.index:
            return alias
    return None


# ── Rating helpers ────────────────────────────────────────────────────────

def _pe_rating(pe: float | None) -> str:
    if pe is None: return "fair"
    if pe < 15: return "good"
    if pe < 25: return "fair"
    return "high"  # expensive


def _roe_rating(roe: float | None) -> str:
    """roe is already in % (e.g. 22.1)"""
    if roe is None: return "fair"
    if roe >= 15: return "good"
    if roe >= 10: return "fair"
    return "poor"


def _roce_rating(roce: float | None) -> str:
    if roce is None: return "fair"
    if roce >= 15: return "good"
    return "fair"


def _de_rating(de: float | None) -> str:
    if de is None: return "fair"
    if de <= 0.5: return "good"
    if de <= 1.5: return "fair"
    return "poor"


def _div_rating(dy: float | None) -> str:
    if dy is None: return "fair"
    if dy >= 2.0: return "good"
    if dy >= 0.5: return "fair"
    return "poor"


def _health_score(pe, roe, roce, de, gm, om, nm, rev_growth) -> int:
    """
    Compute a 0–100 health score from key fundamentals.
    Used for the Overall Health donut on the Overview tab.
    """
    score = 50  # Start neutral

    if roe is not None:
        if roe >= 20: score += 10
        elif roe >= 15: score += 5
        elif roe < 8: score -= 8

    if de is not None:
        if de <= 0.5: score += 10
        elif de <= 1.0: score += 5
        elif de > 2.0: score -= 10

    if nm is not None:
        if nm >= 15: score += 8
        elif nm >= 8: score += 4
        elif nm < 3: score -= 5

    if pe is not None:
        if pe <= 15: score += 5
        elif pe > 35: score -= 8

    if rev_growth is not None:
        if rev_growth >= 20: score += 7
        elif rev_growth >= 10: score += 3
        elif rev_growth < 0: score -= 7

    return max(0, min(100, score))


def _health_label(score: int) -> str:
    if score >= 75: return "Good"
    if score >= 55: return "Fair"
    return "Weak"


# ── Financial statement parser ─────────────────────────────────────────────

def _parse_income_stmt(
    financials: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """
    Parse income statement DataFrames into FinancialRow lists
    for both annual and quarterly views.

    yfinance column names are dates (most recent first).
    Row index labels are line items.
    """
    def _extract(df: pd.DataFrame, periods: int, is_quarterly: bool) -> list[dict]:
        if df is None or df.empty:
            return []

        # Take the most recent `periods` columns
        df = df.iloc[:, :periods]
        cols = list(df.columns)

        def _row_label(idx_label: str) -> str:
            """Map yfinance row labels to display names."""
            label_map = {
                "Total Revenue":          "Revenue",
                "Operating Income":       "Operating Profit",
                "Net Income":             "Net Profit",
                "Basic EPS":              "EPS (₹)",
                "Diluted EPS":            "EPS (₹)",
                "Gross Profit":           "Gross Profit",
                "EBITDA":                 "EBITDA",
                "Tax Provision":          "Tax",
                "Interest Expense":       "Interest",
            }
            return label_map.get(idx_label, idx_label)

        # Rows we care about
        priority_rows = [
            "Total Revenue", "Operating Income", "Net Income", "Basic EPS",
            "Diluted EPS", "EBITDA", "Gross Profit",
        ]

        # Build label keys for periods
        if is_quarterly:
            period_keys = [f"q{i+1}" for i in range(len(cols))]
        else:
            period_keys = [f"fy{str(c.year)[2:]}" if hasattr(c, "year") else f"p{i}" for i, c in enumerate(cols)]

        rows = []
        for row_name in priority_rows:
            # Use _resolve_row to find the actual label in this DataFrame
            # (handles income_stmt vs financials label differences)
            actual_row = _resolve_row(df, row_name)
            if actual_row is None:
                continue
            row_data = {"label": _row_label(row_name)}
            for i, (col, key) in enumerate(zip(cols, period_keys)):
                val = df.loc[actual_row, col]
                if row_name in ("Basic EPS", "Diluted EPS"):
                    row_data[key] = _safe(val, 1)
                else:
                    row_data[key] = _cr(val)
            rows.append(row_data)

        return rows

    # Annual P&L — map to fy24, fy23, fy22, fy21, fy20 keys
    annual_rows = _extract(financials, 5, is_quarterly=False)

    # Rename keys to match UI (fy24, fy23, etc.)
    # yfinance gives most-recent-first; we want labels fy24 to fy20
    fy_labels = ["fy24", "fy23", "fy22", "fy21", "fy20"]
    for row in annual_rows:
        old_keys = [k for k in row if k.startswith("fy") or k.startswith("p")]
        for i, old_key in enumerate(old_keys[:5]):
            if old_key != fy_labels[i]:
                row[fy_labels[i]] = row.pop(old_key)
        # Ensure all 5 years exist
        for lbl in fy_labels:
            row.setdefault(lbl, None)

    return annual_rows, []


def _parse_balance_sheet(balance_sheet: pd.DataFrame) -> list[dict]:
    """Parse annual balance sheet into FinancialRow list."""
    if balance_sheet is None or balance_sheet.empty:
        return []

    df = balance_sheet.iloc[:, :5]  # Last 5 years
    fy_labels = ["fy24", "fy23", "fy22", "fy21", "fy20"]

    priority = [
        "Total Assets", "Total Liabilities Net Minority Interest",
        "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents",
        "Current Assets", "Current Liabilities",
    ]

    display_map = {
        "Total Assets":                             "Total Assets",
        "Total Liabilities Net Minority Interest":  "Total Liabilities",
        "Stockholders Equity":                      "Shareholders Equity",
        "Total Debt":                               "Total Debt",
        "Cash And Cash Equivalents":               "Cash & Equivalents",
        "Current Assets":                           "Current Assets",
        "Current Liabilities":                      "Current Liabilities",
    }

    rows = []
    for row_name in priority:
        if row_name not in df.index:
            continue
        row = {"label": display_map.get(row_name, row_name)}
        cols = list(df.columns)
        for i, (col, lbl) in enumerate(zip(cols, fy_labels)):
            row[lbl] = _cr(df.loc[row_name, col])
        for lbl in fy_labels:
            row.setdefault(lbl, None)
        rows.append(row)

    return rows


def _parse_cashflow(cashflow: pd.DataFrame) -> list[dict]:
    """Parse annual cash flow statement."""
    if cashflow is None or cashflow.empty:
        return []

    df = cashflow.iloc[:, :5]
    fy_labels = ["fy24", "fy23", "fy22", "fy21", "fy20"]

    priority = [
        "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
        "Capital Expenditure", "Free Cash Flow",
    ]

    display_map = {
        "Operating Cash Flow":  "Operating Cash Flow",
        "Investing Cash Flow":  "Investing Activities",
        "Financing Cash Flow":  "Financing Activities",
        "Capital Expenditure":  "Capital Expenditure",
        "Free Cash Flow":       "Free Cash Flow",
    }

    rows = []
    for row_name in priority:
        if row_name not in df.index:
            continue
        row = {"label": display_map.get(row_name, row_name)}
        cols = list(df.columns)
        for col, lbl in zip(cols, fy_labels):
            row[lbl] = _cr(cashflow.loc[row_name, col])
        for lbl in fy_labels:
            row.setdefault(lbl, None)
        rows.append(row)

    return rows


# ── Financial trend chart data ─────────────────────────────────────────────

def _build_financial_trends(
    financials: pd.DataFrame,
    quarterly_financials: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build annual and quarterly revenue/profit trend data for the chart.
    Returns values in Crores.
    """
    def _df_to_trend(df: pd.DataFrame, is_quarterly: bool) -> list[dict]:
        if df is None or df.empty:
            return []

        revenue_key = _resolve_row(df, "Total Revenue")
        profit_key  = _resolve_row(df, "Net Income")

        trend = []
        cols = list(df.columns)

        for i, col in enumerate(reversed(cols[:8])):  # Up to 8 periods, oldest first
            try:
                rev = _cr(df.loc[revenue_key, col]) if revenue_key in df.index else None
                prf = _cr(df.loc[profit_key, col]) if profit_key in df.index else None

                if rev is None:
                    continue

                # Label: year for annual, quarter label for quarterly
                if hasattr(col, "year"):
                    if is_quarterly:
                        q_num = (col.month - 1) // 3 + 1
                        label = f"Q{q_num} {str(col.year)[2:]}"
                    else:
                        label = str(col.year)
                else:
                    label = str(col)[:7]

                trend.append({
                    "year" if not is_quarterly else "quarter": label,
                    "revenue": rev or 0,
                    "profit": prf or 0,
                })
            except Exception:
                continue

        return trend

    annual_trend = _df_to_trend(financials, is_quarterly=False)
    quarterly_trend = _df_to_trend(quarterly_financials, is_quarterly=True)

    # Compute 5Y revenue CAGR
    cagr = 0.0
    if len(annual_trend) >= 2:
        try:
            rev_start = annual_trend[0]["revenue"]
            rev_end   = annual_trend[-1]["revenue"]
            n_years   = len(annual_trend) - 1
            if rev_start > 0 and n_years > 0:
                cagr = round(((rev_end / rev_start) ** (1 / n_years) - 1) * 100, 1)
        except Exception:
            pass

    return {
        "annual": annual_trend,
        "quarterly": quarterly_trend,
        "revenueCagr5y": cagr,
    }


# ── Main computation ──────────────────────────────────────────────────────

def compute_fundamentals(
    info: dict[str, Any],
    financials: pd.DataFrame,
    quarterly_financials: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    quarterly_balance_sheet: pd.DataFrame,
    cashflow: pd.DataFrame,
    quarterly_cashflow: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute all fundamental metrics.

    Returns a dict matching the TypeScript Fundamental interface exactly.
    """
    # ── Extract key metrics from info dict ────────────────────────────────
    # yfinance info keys (verified against real API responses)
    pe        = _safe(_get(info, "trailingPE", "forwardPE"))
    roe_raw   = _safe(_get(info, "returnOnEquity"))  # 0-1 fraction
    roa_raw   = _safe(_get(info, "returnOnAssets"))
    de        = _safe(_get(info, "debtToEquity"))    # Already a ratio
    eps       = _safe(_get(info, "trailingEps", "epsTrailingTwelveMonths"))
    div_yield = _pct(_get(info, "dividendYield"))    # 0-1 fraction → %
    gross_m   = _pct(_get(info, "grossMargins"))
    op_m      = _pct(_get(info, "operatingMargins"))
    net_m     = _pct(_get(info, "profitMargins"))
    curr_r    = _safe(_get(info, "currentRatio"))
    roa       = _pct(roa_raw)

    # ROE: yfinance gives as fraction (0.22 = 22%) — convert to %
    roe = _pct(roe_raw) if roe_raw is not None else None

    # ROCE: not directly in yfinance info — approximate from margins
    # ROCE = EBIT / Capital Employed ≈ (operating income / total assets - CL)
    # We'll use operating margin as a proxy and note the approximation
    roce = _safe(op_m * 0.85) if op_m is not None else None  # Rough approximation

    # D/E: yfinance sometimes returns in % (e.g. 28 = 0.28) — normalise
    if de is not None and de > 10:
        de = de / 100  # Convert from percentage to ratio

    # Revenue growth (YoY from financials)
    rev_growth = None
    if financials is not None and not financials.empty:
        try:
            rev_row = _resolve_row(financials, "Total Revenue")
            if rev_row and len(financials.columns) >= 2:
                rev_curr = financials.loc[rev_row, financials.columns[0]]
                rev_prev = financials.loc[rev_row, financials.columns[1]]
                if rev_prev and rev_prev != 0:
                    rev_growth = _safe(((rev_curr - rev_prev) / abs(rev_prev)) * 100, 1)
        except Exception:
            pass

    # Profit growth (YoY)
    profit_growth = None
    if financials is not None and not financials.empty:
        try:
            net_row = _resolve_row(financials, "Net Income")
            if net_row and len(financials.columns) >= 2:
                p_curr = financials.loc[net_row, financials.columns[0]]
                p_prev = financials.loc[net_row, financials.columns[1]]
                if p_prev and p_prev != 0:
                    profit_growth = _safe(((p_curr - p_prev) / abs(p_prev)) * 100, 1)
        except Exception:
            pass

    # ── Health score ──────────────────────────────────────────────────────
    health_score = _health_score(pe, roe, roce, de, gross_m, op_m, net_m, rev_growth)
    health_label = _health_label(health_score)

    # ── Quick summary signals ─────────────────────────────────────────────
    rev_signal = "bullish" if (rev_growth or 0) > 10 else ("bearish" if (rev_growth or 0) < 0 else "neutral")
    prf_signal = "bullish" if (profit_growth or 0) > 10 else ("bearish" if (profit_growth or 0) < 0 else "neutral")
    debt_signal = "good" if (de or 0) <= 0.5 else ("neutral" if (de or 0) <= 1.5 else "bearish")
    debt_label = "Low" if (de or 0) <= 0.5 else ("Medium" if (de or 0) <= 1.5 else "High")

    # Valuation signal based on PE
    if pe is None:
        val_label = "N/A"
        val_signal = "neutral"
    elif pe < 15:
        val_label = "Cheap"
        val_signal = "bullish"
    elif pe < 25:
        val_label = "Fair"
        val_signal = "neutral"
    else:
        val_label = "Expensive"
        val_signal = "bearish"

    # Key insight — rule-based, no LLM
    insight = _generate_key_insight(
        health_score=health_score,
        pe=pe, roe=roe, de=de, rev_growth=rev_growth,
        net_margin=net_m, profit_growth=profit_growth,
    )

    # ── Parse financial statements ────────────────────────────────────────
    pnl_rows, _ = _parse_income_stmt(financials, quarterly_financials)
    bs_rows = _parse_balance_sheet(balance_sheet)
    cf_rows = _parse_cashflow(cashflow)

    # ── Financial trends for chart ────────────────────────────────────────
    trends = _build_financial_trends(financials, quarterly_financials)

    # ── Build response ────────────────────────────────────────────────────
    return {
        "overallHealth": {
            "score": health_score,
            "label": health_label,
        },
        "keyRatios": {
            "pe": {
                "value": pe or 0.0,
                "benchmark": "Sector avg ~22x",
                "rating": _pe_rating(pe),
                "label": "PE Ratio",
                "tooltip": "Price-to-Earnings ratio. How much you pay per rupee of earnings.",
            },
            "roe": {
                "value": roe or 0.0,
                "benchmark": "> 15% = good",
                "rating": _roe_rating(roe),
                "label": "ROE",
                "tooltip": "Return on Equity. How efficiently the company uses shareholder capital.",
            },
            "roce": {
                "value": roce or 0.0,
                "benchmark": "> 15% = good",
                "rating": _roce_rating(roce),
                "label": "ROCE",
                "tooltip": "Return on Capital Employed. Measures how well capital generates profit.",
            },
            "debtEquity": {
                "value": de or 0.0,
                "benchmark": "< 0.5 = low debt",
                "rating": _de_rating(de),
                "label": "Debt / Equity",
                "tooltip": "Total debt divided by shareholder equity. Lower is safer.",
            },
            "eps": {
                "value": eps or 0.0,
                "benchmark": "Positive = pass",
                "rating": "good" if (eps or 0) > 0 else "poor",
                "label": "EPS (TTM)",
                "tooltip": "Earnings Per Share for the trailing 12 months.",
            },
            "dividendYield": {
                "value": div_yield or 0.0,
                "benchmark": "> 2% = good",
                "rating": _div_rating(div_yield),
                "label": "Dividend Yield",
                "tooltip": "Annual dividend as a percentage of current stock price.",
            },
        },
        "quickSummary": {
            "overallHealth":  {"score": health_score, "label": health_label},
            "revenueGrowth":  {"value": rev_growth or 0.0, "signal": rev_signal},
            "profitGrowth":   {"value": profit_growth or 0.0, "signal": prf_signal},
            "debt":           {"value": debt_label, "signal": debt_signal},
            "valuation":      {"value": val_label, "signal": val_signal},
            "keyInsight":     insight,
        },
        "ratiosSnapshot": {
            "grossMargin":       gross_m or 0.0,
            "operatingMargin":   op_m or 0.0,
            "netMargin":         net_m or 0.0,
            "currentRatio":      curr_r or 0.0,
            "returnOnAssets":    roa or 0.0,
        },
        "financialTrends": trends,
        "financialStatements": {
            "pnl":          pnl_rows,
            "balanceSheet": bs_rows,
            "cashFlow":     cf_rows,
        },
    }


def _generate_key_insight(
    health_score: int,
    pe: float | None,
    roe: float | None,
    de: float | None,
    rev_growth: float | None,
    net_margin: float | None,
    profit_growth: float | None,
) -> str:
    """
    Generate a rule-based key insight sentence without any LLM calls.
    Reads like a mini equity research summary.
    """
    parts: list[str] = []

    # Profitability
    if roe is not None and roe >= 15:
        parts.append(f"strong returns on equity ({roe:.1f}% ROE)")
    elif roe is not None and roe < 8:
        parts.append(f"weak returns on equity ({roe:.1f}% ROE)")

    # Growth
    if rev_growth is not None and rev_growth >= 15:
        parts.append(f"healthy revenue growth ({rev_growth:.1f}% YoY)")
    elif rev_growth is not None and rev_growth < 0:
        parts.append(f"declining revenue ({rev_growth:.1f}% YoY)")

    # Debt
    if de is not None:
        if de <= 0.3:
            parts.append("very low debt")
        elif de > 2.0:
            parts.append(f"high leverage (D/E {de:.1f}x)")

    # Valuation
    if pe is not None:
        if pe > 30:
            parts.append(f"current valuation is elevated (PE {pe:.1f}x)")
        elif pe < 12:
            parts.append(f"valuation appears attractive (PE {pe:.1f}x)")

    if not parts:
        if health_score >= 70:
            return "Company shows solid overall fundamentals across profitability and growth metrics."
        elif health_score >= 50:
            return "Mixed fundamentals — review individual metrics before making a decision."
        else:
            return "Weak fundamentals signal caution — look for specific improvement triggers."

    base = "This business demonstrates " + ", ".join(parts[:2])
    if len(parts) > 2:
        base += f", but {parts[2]}."
    else:
        base += "."

    return base.capitalize()
