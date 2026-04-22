"""
app/services/personas.py
─────────────────────────────────────────────────────────────────────────────
Investor Persona Scoring Engine.

9 personas. Zero LLM calls. Every score is computed from real financial
data using explicit, auditable rules.

Each persona is a class with:
  - WEIGHTS: dict of metric → max_points
  - score(): returns PersonaResult with score, verdict, criteria, summary

Architecture:
  • BasePersona handles verdict mapping, criteria formatting, and summary
    generation — subclasses only define weights and conditions.
  • All scoring is additive: each criterion contributes 0-to-max points.
  • Partial credit is awarded for "almost there" values (not binary).
─────────────────────────────────────────────────────────────────────────────
"""

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ── Data bag passed to every persona ─────────────────────────────────────

@dataclass
class StockMetrics:
    """
    Pre-extracted flat metrics bag. Computed once and shared by all personas.
    All monetary values in original units (not Crores) for ratio consistency.
    Percentages are in human-readable % (e.g. 22.1 not 0.221).
    """
    # Valuation
    pe: float | None = None
    pb: float | None = None
    peg: float | None = None
    ev_ebitda: float | None = None
    p_fcf: float | None = None

    # Profitability (in %)
    roe: float | None = None         # Return on Equity %
    roa: float | None = None         # Return on Assets %
    roic: float | None = None        # ROIC % (approximated)
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None

    # Growth (in %)
    revenue_growth_yoy: float | None = None
    eps_growth_yoy: float | None = None
    revenue_cagr_3y: float | None = None
    eps_cagr_3y: float | None = None

    # Safety
    debt_equity: float | None = None
    current_ratio: float | None = None
    interest_coverage: float | None = None

    # Technical
    rsi: float | None = None
    price_vs_sma200: float | None = None  # % above/below SMA200
    momentum_6m: float | None = None      # 6-month return %
    volume_ratio: float | None = None     # current vol / 20-day avg

    # Shareholding
    promoter_holding: float | None = None
    promoter_pledge: float | None = None
    fii_holding: float | None = None

    # Market
    market_cap_cr: float | None = None   # Market cap in Crores
    beta: float | None = None

    # Company info
    sector: str = ""
    eps: float | None = None
    dividend_yield: float | None = None


# ── Result type ───────────────────────────────────────────────────────────

@dataclass
class PersonaResult:
    id: str
    name: str
    photo_url: str | None
    icon: str | None
    score: int                            # 0–100
    verdict: str                          # "Strong Match" etc.
    verdict_color: str                    # "success" | "info" | "warning" | "danger"
    summary: str                          # 1-line plain-English
    criteria: list[dict] = field(default_factory=list)  # [{label, value, pass}]


# ── Verdict mapping ───────────────────────────────────────────────────────

def _verdict(score: int) -> tuple[str, str]:
    """Return (label, color) for a given score."""
    if score >= 85: return ("Strong Match", "success")
    if score >= 65: return ("Good Match",   "info")
    if score >= 45: return ("Neutral",       "warning")
    if score >= 25: return ("Weak Match",   "danger")
    return ("Poor Match", "danger")


def _safe(v: Any) -> float | None:
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ── Base class ────────────────────────────────────────────────────────────

class BasePersona:
    ID:        str = ""
    NAME:      str = ""
    PHOTO_URL: str | None = None
    ICON:      str | None = None

    def score(self, m: StockMetrics) -> PersonaResult:
        """Override in subclass — return raw 0–100 score and criteria list."""
        raise NotImplementedError

    def _result(self, raw_score: float, criteria: list[dict]) -> PersonaResult:
        score = max(0, min(100, round(raw_score)))
        verdict, color = _verdict(score)

        # Generate summary
        passing = [c for c in criteria if c["pass"]]
        failing = [c for c in criteria if not c["pass"]]
        summary = self._make_summary(score, passing, failing)

        return PersonaResult(
            id=self.ID,
            name=self.NAME,
            photo_url=self.PHOTO_URL,
            icon=self.ICON,
            score=score,
            verdict=verdict,
            verdict_color=color,
            summary=summary,
            criteria=criteria,
        )

    def _make_summary(self, score: int, passing: list, failing: list) -> str:
        """Generate a 10–15 word plain-English summary."""
        if score >= 85:
            return f"Strong alignment with {self.NAME.split()[0]}'s investment philosophy."
        if score >= 65:
            return f"Good match — {len(passing)} of {len(passing)+len(failing)} criteria met."
        if score >= 45:
            return f"Mixed picture — meets some criteria, fails on others."
        return f"Weak alignment — most of {self.NAME.split()[0]}'s criteria are not met."

    def _criterion(self, label: str, actual: Any, passed: bool, format_fn=None) -> dict:
        """Build a single criterion dict for the UI."""
        if actual is None:
            display = "N/A"
        elif format_fn:
            display = format_fn(actual)
        elif isinstance(actual, float):
            display = f"{actual:.1f}"
        else:
            display = str(actual)
        return {"label": label, "value": display, "pass": passed}


# ── Persona implementations ───────────────────────────────────────────────

class WarrenBuffettPersona(BasePersona):
    ID        = "warren-buffett"
    NAME      = "Warren Buffett"
    PHOTO_URL = "/personas/warren-buffett.svg"
    ICON      = None

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # ROE > 15% (20 pts)
        roe = _safe(m.roe)
        if roe is not None:
            pts = 20 if roe >= 15 else (12 if roe >= 12 else (5 if roe >= 8 else 0))
            total += pts
            criteria.append(self._criterion(
                "Consistent ROE > 15%", roe, roe >= 15, lambda v: f"{v:.1f}%"
            ))

        # Low D/E < 0.5 (15 pts)
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 15 if de <= 0.5 else (10 if de <= 1.0 else (5 if de <= 1.5 else 0))
            total += pts
            criteria.append(self._criterion(
                "Low Debt / Equity < 0.5", de, de <= 0.5, lambda v: f"{v:.2f}"
            ))

        # High Operating Margin > 15% (15 pts)
        om = _safe(m.operating_margin)
        if om is not None:
            pts = 15 if om >= 15 else (8 if om >= 10 else 0)
            total += pts
            criteria.append(self._criterion(
                "High Operating Margin > 15%", om, om >= 15, lambda v: f"{v:.1f}%"
            ))

        # ROIC proxy (roe * (1 - de_ratio)) (15 pts)
        roic_approx = _safe(m.roic) or (roe * 0.7 if roe else None)
        if roic_approx is not None:
            pts = 15 if roic_approx >= 15 else (8 if roic_approx >= 10 else 0)
            total += pts

        # Gross Margin > 40% (10 pts)
        gm = _safe(m.gross_margin)
        if gm is not None:
            pts = 10 if gm >= 40 else (5 if gm >= 25 else 0)
            total += pts
            criteria.append(self._criterion(
                "Gross Margin > 40%", gm, gm >= 40, lambda v: f"{v:.1f}%"
            ))

        # Stable Profit Growth > 10% (15 pts)
        eg = _safe(m.eps_growth_yoy)
        if eg is not None:
            pts = 15 if eg >= 10 else (8 if eg >= 5 else 0)
            total += pts
            criteria.append(self._criterion(
                "Stable Profit Growth", eg, eg >= 10, lambda v: f"{v:.1f}% (5Y)"
            ))

        # Reasonable Valuation PE < 25 (10 pts)
        pe = _safe(m.pe)
        if pe is not None and pe > 0:
            pts = 10 if pe < 20 else (6 if pe < 25 else (3 if pe < 35 else 0))
            total += pts
            criteria.append(self._criterion(
                "Reasonable Valuation (PE < 25)", pe, pe < 25, lambda v: f"{v:.1f}"
            ))

        # Scale to 100 — max possible is 100
        return self._result(total, criteria)


class BenjaminGrahamPersona(BasePersona):
    ID        = "benjamin-graham"
    NAME      = "Benjamin Graham"
    PHOTO_URL = "/personas/benjamin-graham.svg"
    ICON      = None

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # EV/EBITDA < 7x (25 pts)
        ev = _safe(m.ev_ebitda)
        if ev is not None and ev > 0:
            pts = 25 if ev < 7 else (15 if ev < 10 else (5 if ev < 15 else 0))
            total += pts
            criteria.append(self._criterion("EV/EBITDA < 7x", ev, ev < 7, lambda v: f"{v:.1f}x"))

        # P/B < 1.5 (20 pts)
        pb = _safe(m.pb)
        if pb is not None and pb > 0:
            pts = 20 if pb < 1.5 else (12 if pb < 2.5 else (4 if pb < 3.5 else 0))
            total += pts
            criteria.append(self._criterion("Price / Book < 1.5", pb, pb < 1.5, lambda v: f"{v:.2f}x"))

        # Current Ratio > 2 (15 pts)
        cr = _safe(m.current_ratio)
        if cr is not None:
            pts = 15 if cr >= 2 else (8 if cr >= 1.5 else (3 if cr >= 1 else 0))
            total += pts
            criteria.append(self._criterion("Current Ratio > 2.0", cr, cr >= 2, lambda v: f"{v:.2f}"))

        # Debt/Equity < 0.5 (15 pts)
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 15 if de <= 0.5 else (8 if de <= 1.0 else 0)
            total += pts
            criteria.append(self._criterion("Debt / Equity < 0.5", de, de <= 0.5, lambda v: f"{v:.2f}"))

        # Positive EPS (consistent earnings) (15 pts)
        eps = _safe(m.eps)
        if eps is not None:
            pts = 15 if eps > 0 else 0
            total += pts
            criteria.append(self._criterion("Positive Earnings (EPS)", eps, eps > 0, lambda v: f"₹{v:.1f}"))

        # Dividend yield > 0 (10 pts)
        dy = _safe(m.dividend_yield)
        if dy is not None:
            pts = 10 if dy > 0 else 0
            total += pts
            criteria.append(self._criterion("Pays Dividend", dy, dy > 0, lambda v: f"{v:.1f}%"))

        return self._result(total, criteria)


class PeterLynchPersona(BasePersona):
    ID        = "peter-lynch"
    NAME      = "Peter Lynch"
    PHOTO_URL = "/personas/peter-lynch.svg"
    ICON      = None

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # PEG < 1.0 (25 pts) — Lynch's single favourite signal
        peg = _safe(m.peg)
        if peg is not None and peg > 0:
            pts = 25 if peg < 1.0 else (15 if peg < 1.5 else (5 if peg < 2.0 else 0))
            total += pts
            criteria.append(self._criterion("PEG Ratio < 1.0", peg, peg < 1.0, lambda v: f"{v:.2f}"))

        # EPS Growth > 15% (20 pts)
        eg = _safe(m.eps_growth_yoy)
        if eg is not None:
            pts = 20 if eg >= 15 else (12 if eg >= 10 else (4 if eg >= 5 else 0))
            total += pts
            criteria.append(self._criterion("EPS Growth > 15%", eg, eg >= 15, lambda v: f"{v:.1f}%"))

        # Revenue Growth > 20% (15 pts)
        rg = _safe(m.revenue_growth_yoy)
        if rg is not None:
            pts = 15 if rg >= 20 else (8 if rg >= 12 else (3 if rg >= 5 else 0))
            total += pts
            criteria.append(self._criterion("Revenue Growth > 20%", rg, rg >= 20, lambda v: f"{v:.1f}%"))

        # Net Margin > 10% (10 pts)
        nm = _safe(m.net_margin)
        if nm is not None:
            pts = 10 if nm >= 10 else (5 if nm >= 5 else 0)
            total += pts
            criteria.append(self._criterion("Net Margin > 10%", nm, nm >= 10, lambda v: f"{v:.1f}%"))

        # Promoter / Insider holding > 20% (10 pts)
        ph = _safe(m.promoter_holding)
        if ph is not None:
            pts = 10 if ph >= 20 else (5 if ph >= 10 else 0)
            total += pts
            criteria.append(self._criterion("Insider Holding > 20%", ph, ph >= 20, lambda v: f"{v:.1f}%"))

        # Debt/Equity < 0.5 (10 pts)
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 10 if de <= 0.5 else (5 if de <= 1.0 else 0)
            total += pts
            criteria.append(self._criterion("Low Debt (D/E < 0.5)", de, de <= 0.5, lambda v: f"{v:.2f}"))

        # Cash > Total Debt proxy: D/E very low (10 pts)
        if de is not None:
            pts = 10 if de <= 0.1 else (5 if de <= 0.3 else 0)
            total += pts

        return self._result(total, criteria)


class GrowthInvestorPersona(BasePersona):
    ID        = "growth-investor"
    NAME      = "Growth Investor"
    PHOTO_URL = None
    ICON      = "zap"

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # Revenue Growth > 25% (25 pts)
        rg = _safe(m.revenue_growth_yoy)
        if rg is not None:
            pts = 25 if rg >= 25 else (16 if rg >= 15 else (8 if rg >= 8 else 0))
            total += pts
            criteria.append(self._criterion("Revenue Growth > 25%", rg, rg >= 25, lambda v: f"{v:.1f}%"))

        # EPS Growth > 20% (20 pts)
        eg = _safe(m.eps_growth_yoy)
        if eg is not None:
            pts = 20 if eg >= 20 else (12 if eg >= 12 else (5 if eg >= 5 else 0))
            total += pts
            criteria.append(self._criterion("EPS Growth > 20%", eg, eg >= 20, lambda v: f"{v:.1f}%"))

        # Gross Margin > 40% (15 pts)
        gm = _safe(m.gross_margin)
        if gm is not None:
            pts = 15 if gm >= 40 else (8 if gm >= 25 else 0)
            total += pts
            criteria.append(self._criterion("Gross Margin > 40%", gm, gm >= 40, lambda v: f"{v:.1f}%"))

        # ROE > 18% (15 pts)
        roe = _safe(m.roe)
        if roe is not None:
            pts = 15 if roe >= 18 else (8 if roe >= 12 else 0)
            total += pts
            criteria.append(self._criterion("ROE > 18%", roe, roe >= 18, lambda v: f"{v:.1f}%"))

        # Low D/E < 1.0 (10 pts)
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 10 if de <= 1.0 else 0
            total += pts
            criteria.append(self._criterion("Manageable Debt (D/E < 1.0)", de, de <= 1.0, lambda v: f"{v:.2f}"))

        # Revenue CAGR 3Y > 20% (15 pts)
        cagr = _safe(m.revenue_cagr_3y)
        if cagr is not None:
            pts = 15 if cagr >= 20 else (8 if cagr >= 12 else 0)
            total += pts
            criteria.append(self._criterion("Revenue CAGR 3Y > 20%", cagr, cagr >= 20, lambda v: f"{v:.1f}%"))

        return self._result(total, criteria)


class MomentumTraderPersona(BasePersona):
    ID        = "momentum-trader"
    NAME      = "Momentum Trader"
    PHOTO_URL = None
    ICON      = "trending-up"

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # Price above SMA200 (20 pts)
        vs200 = _safe(m.price_vs_sma200)
        if vs200 is not None:
            pts = 20 if vs200 > 0 else 0
            total += pts
            criteria.append(self._criterion(
                "Price above 200-day MA", vs200, vs200 > 0,
                lambda v: f"{'+' if v > 0 else ''}{v:.1f}%"
            ))

        # RSI in 40–70 (momentum zone) (20 pts)
        rsi = _safe(m.rsi)
        if rsi is not None:
            pts = 20 if 40 <= rsi <= 70 else (10 if 30 <= rsi <= 75 else 0)
            total += pts
            criteria.append(self._criterion("RSI in Momentum Zone (40–70)", rsi, 40 <= rsi <= 70, lambda v: f"{v:.1f}"))

        # 6-month momentum > 10% (20 pts)
        mom = _safe(m.momentum_6m)
        if mom is not None:
            pts = 20 if mom >= 10 else (12 if mom >= 0 else 0)
            total += pts
            criteria.append(self._criterion("6M Momentum > 10%", mom, mom >= 10, lambda v: f"{v:.1f}%"))

        # Volume > avg (15 pts)
        vr = _safe(m.volume_ratio)
        if vr is not None:
            pts = 15 if vr >= 1.3 else (8 if vr >= 0.8 else 0)
            total += pts
            criteria.append(self._criterion("Volume Above Average", vr, vr >= 1.0, lambda v: f"{v:.2f}x avg"))

        # High beta for momentum (15 pts)
        beta = _safe(m.beta)
        if beta is not None:
            pts = 15 if beta >= 1.1 else (8 if beta >= 0.8 else 0)
            total += pts

        # ROE check (momentum often accompanies improving fundamentals) (10 pts)
        roe = _safe(m.roe)
        if roe is not None:
            pts = 10 if roe >= 15 else (5 if roe >= 10 else 0)
            total += pts

        return self._result(total, criteria)


class JimSimonsPersona(BasePersona):
    ID        = "jim-simons"
    NAME      = "Jim Simons"
    PHOTO_URL = None
    ICON      = "trending-up"

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # RSI in 40–65 (building momentum) (20 pts)
        rsi = _safe(m.rsi)
        if rsi is not None:
            pts = 20 if 40 <= rsi <= 65 else (10 if 35 <= rsi <= 72 else 0)
            total += pts
            criteria.append(self._criterion("RSI 40–65 (Momentum Building)", rsi, 40 <= rsi <= 65, lambda v: f"{v:.1f}"))

        # Price above SMA200 (20 pts)
        vs200 = _safe(m.price_vs_sma200)
        if vs200 is not None:
            pts = 20 if vs200 > 2 else (10 if vs200 > 0 else 0)
            total += pts
            criteria.append(self._criterion(
                "Price above 200-day MA", vs200, vs200 > 0,
                lambda v: f"{'+' if v > 0 else ''}{v:.1f}%"
            ))

        # Volume spike (15 pts)
        vr = _safe(m.volume_ratio)
        if vr is not None:
            pts = 15 if vr >= 1.5 else (8 if vr >= 1.1 else 0)
            total += pts
            criteria.append(self._criterion("Volume Spike (> 1.5x avg)", vr, vr >= 1.5, lambda v: f"{v:.2f}x"))

        # Momentum 6M (15 pts)
        mom = _safe(m.momentum_6m)
        if mom is not None:
            pts = 15 if mom >= 5 else (8 if mom >= 0 else 0)
            total += pts

        # Low D/E (stat arb favors stable balance sheets) (15 pts)
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 15 if de <= 0.5 else (8 if de <= 1.5 else 0)
            total += pts

        # Positive net margin (15 pts)
        nm = _safe(m.net_margin)
        if nm is not None:
            pts = 15 if nm >= 8 else (8 if nm >= 3 else 0)
            total += pts

        return self._result(total, criteria)


class CathieWoodPersona(BasePersona):
    ID        = "cathie-wood"
    NAME      = "Cathie Wood"
    PHOTO_URL = None
    ICON      = "zap"

    GROWTH_SECTORS = {
        "Information Technology", "Technology", "Healthcare",
        "Financial Services", "Retail",  # Fintech proxy
    }

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # Revenue Growth > 40% (25 pts)
        rg = _safe(m.revenue_growth_yoy)
        if rg is not None:
            pts = 25 if rg >= 40 else (15 if rg >= 25 else (5 if rg >= 10 else 0))
            total += pts
            criteria.append(self._criterion("Revenue Growth > 40%", rg, rg >= 40, lambda v: f"{v:.1f}%"))

        # Revenue CAGR 3Y > 30% (20 pts)
        cagr = _safe(m.revenue_cagr_3y)
        if cagr is not None:
            pts = 20 if cagr >= 30 else (10 if cagr >= 20 else 0)
            total += pts
            criteria.append(self._criterion("Revenue CAGR 3Y > 30%", cagr, cagr >= 30, lambda v: f"{v:.1f}%"))

        # Gross Margin > 50% (15 pts)
        gm = _safe(m.gross_margin)
        if gm is not None:
            pts = 15 if gm >= 50 else (8 if gm >= 35 else 0)
            total += pts
            criteria.append(self._criterion("Gross Margin > 50%", gm, gm >= 50, lambda v: f"{v:.1f}%"))

        # Sector qualifier (15 pts)
        in_sector = m.sector in self.GROWTH_SECTORS
        total += 15 if in_sector else 0
        criteria.append(self._criterion(
            "Growth/Disruptive Sector",
            m.sector or "Unknown",
            in_sector,
        ))

        # Strong momentum (15 pts)
        mom = _safe(m.momentum_6m)
        if mom is not None:
            pts = 15 if mom >= 20 else (8 if mom >= 10 else 0)
            total += pts

        # Price above SMA200 (10 pts)
        vs200 = _safe(m.price_vs_sma200)
        if vs200 is not None:
            total += 10 if vs200 > 0 else 0

        return self._result(total, criteria)


class MarkMinerviniPersona(BasePersona):
    ID        = "mark-minervini"
    NAME      = "Mark Minervini"
    PHOTO_URL = None
    ICON      = "trending-up"

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # Price above SMA200 (20 pts) — Stage 2 non-negotiable
        vs200 = _safe(m.price_vs_sma200)
        if vs200 is not None:
            pts = 20 if vs200 > 0 else 0
            total += pts
            criteria.append(self._criterion(
                "Price above 200-day MA (Stage 2)",
                vs200, vs200 > 0,
                lambda v: f"{'+' if v > 0 else ''}{v:.1f}%"
            ))

        # Relative strength 6M > 10% (20 pts)
        mom = _safe(m.momentum_6m)
        if mom is not None:
            pts = 20 if mom >= 10 else (10 if mom >= 0 else 0)
            total += pts
            criteria.append(self._criterion("6M Relative Strength > 10%", mom, mom >= 10, lambda v: f"{v:.1f}%"))

        # EPS Growth > 20% (15 pts)
        eg = _safe(m.eps_growth_yoy)
        if eg is not None:
            pts = 15 if eg >= 20 else (8 if eg >= 10 else 0)
            total += pts
            criteria.append(self._criterion("EPS Growth > 20%", eg, eg >= 20, lambda v: f"{v:.1f}%"))

        # Revenue Growth > 15% (15 pts)
        rg = _safe(m.revenue_growth_yoy)
        if rg is not None:
            pts = 15 if rg >= 15 else (8 if rg >= 8 else 0)
            total += pts
            criteria.append(self._criterion("Revenue Growth > 15%", rg, rg >= 15, lambda v: f"{v:.1f}%"))

        # RSI in momentum zone (15 pts)
        rsi = _safe(m.rsi)
        if rsi is not None:
            pts = 15 if 50 <= rsi <= 70 else (8 if 40 <= rsi <= 75 else 0)
            total += pts

        # Volume above avg (15 pts)
        vr = _safe(m.volume_ratio)
        if vr is not None:
            pts = 15 if vr >= 1.3 else (8 if vr >= 0.9 else 0)
            total += pts

        return self._result(total, criteria)


class RakeshJhunjhunwalaPersona(BasePersona):
    """India-exclusive persona."""
    ID        = "rakesh-jhunjhunwala"
    NAME      = "Rakesh Jhunjhunwala"
    PHOTO_URL = "/personas/rakesh-jhunjhunwala.svg"
    ICON      = None

    # Mid/small cap sweet spot in Crores
    _MCAP_LOW  = 500
    _MCAP_HIGH = 25_000

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # Revenue CAGR 3Y > 20% (20 pts)
        cagr = _safe(m.revenue_cagr_3y)
        if cagr is not None:
            pts = 20 if cagr >= 20 else (12 if cagr >= 12 else (5 if cagr >= 5 else 0))
            total += pts
            criteria.append(self._criterion("Revenue CAGR 3Y > 20%", cagr, cagr >= 20, lambda v: f"{v:.1f}%"))

        # ROE > 20% (15 pts)
        roe = _safe(m.roe)
        if roe is not None:
            pts = 15 if roe >= 20 else (8 if roe >= 15 else 0)
            total += pts
            criteria.append(self._criterion("ROE > 20%", roe, roe >= 20, lambda v: f"{v:.1f}%"))

        # EPS CAGR 3Y > 20% (15 pts)
        ep_cagr = _safe(m.eps_cagr_3y) or _safe(m.eps_growth_yoy)
        if ep_cagr is not None:
            pts = 15 if ep_cagr >= 20 else (8 if ep_cagr >= 12 else 0)
            total += pts
            criteria.append(self._criterion("EPS CAGR 3Y > 20%", ep_cagr, ep_cagr >= 20, lambda v: f"{v:.1f}%"))

        # Promoter holding > 50%, low pledge (15 pts)
        ph = _safe(m.promoter_holding)
        pp = _safe(m.promoter_pledge) or 0
        if ph is not None:
            pts = 15 if ph >= 50 and pp < 10 else (8 if ph >= 40 else 0)
            total += pts
            criteria.append(self._criterion(
                "Promoter Holding > 50%, Low Pledge",
                ph, ph >= 50 and pp < 10,
                lambda v: f"{v:.1f}% (pledge: {pp:.1f}%)"
            ))

        # D/E < 1.0 (15 pts)
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 15 if de <= 1.0 else (8 if de <= 1.5 else 0)
            total += pts
            criteria.append(self._criterion("Debt / Equity < 1.0", de, de <= 1.0, lambda v: f"{v:.2f}"))

        # Market cap in sweet spot (10 pts)
        mc = _safe(m.market_cap_cr)
        if mc is not None:
            in_range = self._MCAP_LOW <= mc <= self._MCAP_HIGH
            total += 10 if in_range else 3
            criteria.append(self._criterion(
                f"Market Cap ₹{self._MCAP_LOW}–{self._MCAP_HIGH} Cr",
                mc, in_range,
                lambda v: f"₹{v:,.0f} Cr"
            ))

        return self._result(total, criteria)


# ── Factory and main entry point ──────────────────────────────────────────

_ALL_PERSONAS: list[BasePersona] = [
    WarrenBuffettPersona(),
    BenjaminGrahamPersona(),
    PeterLynchPersona(),
    GrowthInvestorPersona(),
    MomentumTraderPersona(),
    JimSimonsPersona(),
    CathieWoodPersona(),
    MarkMinerviniPersona(),
    RakeshJhunjhunwalaPersona(),
]


def extract_metrics(
    info: dict,
    history_5y: pd.DataFrame,
    financials: pd.DataFrame,
    technicals: dict,
) -> StockMetrics:
    """
    Extract a flat StockMetrics bag from raw yfinance data.
    Called once; the result is passed to all 9 persona scorers.
    """
    def _i(key: str, *fallbacks) -> float | None:
        """Get a numeric value from info dict."""
        for k in [key, *fallbacks]:
            v = info.get(k)
            if v is not None:
                try:
                    f = float(v)
                    if not (math.isnan(f) or math.isinf(f)):
                        return f
                except (TypeError, ValueError):
                    pass
        return None

    def _pct_from_frac(key: str) -> float | None:
        """Convert 0–1 fraction from info to %."""
        v = _i(key)
        return round(v * 100, 1) if v is not None else None

    # ── Price momentum ────────────────────────────────────────────────────
    momentum_6m = None
    price_vs_sma200 = None
    volume_ratio = None

    if history_5y is not None and not history_5y.empty:
        try:
            close = history_5y["Close"].dropna()
            vol   = history_5y["Volume"].dropna()

            if len(close) >= 126:  # ~6 months
                price_6m_ago = close.iloc[-126]
                current_price = close.iloc[-1]
                if price_6m_ago > 0:
                    momentum_6m = round(((current_price / price_6m_ago) - 1) * 100, 1)

            if len(close) >= 200:
                sma200 = close.rolling(200).mean().iloc[-1]
                if sma200 > 0:
                    price_vs_sma200 = round(((close.iloc[-1] / sma200) - 1) * 100, 1)

            if len(vol) >= 20:
                avg_20 = vol.tail(20).mean()
                if avg_20 > 0:
                    volume_ratio = round(vol.iloc[-1] / avg_20, 2)
        except Exception:
            pass

    # ── Revenue CAGR 3Y from financials ──────────────────────────────────
    revenue_cagr_3y = None
    eps_cagr_3y = None

    if financials is not None and not financials.empty:
        try:
            if "Total Revenue" in financials.index and len(financials.columns) >= 3:
                rev_now  = float(financials.loc["Total Revenue", financials.columns[0]])
                rev_3ya  = float(financials.loc["Total Revenue", financials.columns[min(3, len(financials.columns)-1)]])
                if rev_3ya > 0:
                    revenue_cagr_3y = round(((rev_now / rev_3ya) ** (1/3) - 1) * 100, 1)

            if "Basic EPS" in financials.index and len(financials.columns) >= 3:
                eps_now = float(financials.loc["Basic EPS", financials.columns[0]])
                eps_3ya = float(financials.loc["Basic EPS", financials.columns[min(3, len(financials.columns)-1)]])
                if eps_3ya > 0 and eps_now > 0:
                    eps_cagr_3y = round(((eps_now / eps_3ya) ** (1/3) - 1) * 100, 1)
        except Exception:
            pass

    # ── YoY growth from financials ────────────────────────────────────────
    rev_yoy = None
    eps_yoy = None

    if financials is not None and not financials.empty:
        try:
            if "Total Revenue" in financials.index and len(financials.columns) >= 2:
                r_now  = float(financials.loc["Total Revenue", financials.columns[0]])
                r_prev = float(financials.loc["Total Revenue", financials.columns[1]])
                if r_prev != 0:
                    rev_yoy = round(((r_now - r_prev) / abs(r_prev)) * 100, 1)

            if "Basic EPS" in financials.index and len(financials.columns) >= 2:
                e_now  = float(financials.loc["Basic EPS", financials.columns[0]])
                e_prev = float(financials.loc["Basic EPS", financials.columns[1]])
                if e_prev != 0:
                    eps_yoy = round(((e_now - e_prev) / abs(e_prev)) * 100, 1)
        except Exception:
            pass

    # D/E normalisation
    de = _i("debtToEquity")
    if de is not None and de > 10:
        de = de / 100  # yfinance sometimes returns as percentage

    market_cap = _i("marketCap")
    market_cap_cr = round(market_cap / 10_000_000, 0) if market_cap else None

    # RSI from technicals dict
    rsi_val = None
    try:
        rsi_val = technicals.get("summary", {}).get("rsi", {}).get("value")
        if rsi_val == "N/A":
            rsi_val = None
        else:
            rsi_val = float(rsi_val) if rsi_val else None
    except Exception:
        pass

    return StockMetrics(
        pe                = _i("trailingPE", "forwardPE"),
        pb                = _i("priceToBook"),
        peg               = _i("pegRatio"),
        ev_ebitda         = _i("enterpriseToEbitda"),
        roe               = _pct_from_frac("returnOnEquity"),
        roa               = _pct_from_frac("returnOnAssets"),
        gross_margin      = _pct_from_frac("grossMargins"),
        operating_margin  = _pct_from_frac("operatingMargins"),
        net_margin        = _pct_from_frac("profitMargins"),
        revenue_growth_yoy = rev_yoy or _pct_from_frac("revenueGrowth"),
        eps_growth_yoy    = eps_yoy or _pct_from_frac("earningsGrowth"),
        revenue_cagr_3y   = revenue_cagr_3y,
        eps_cagr_3y       = eps_cagr_3y,
        debt_equity       = de,
        current_ratio     = _i("currentRatio"),
        rsi               = rsi_val,
        price_vs_sma200   = price_vs_sma200,
        momentum_6m       = momentum_6m,
        volume_ratio      = volume_ratio,
        promoter_holding  = None,   # Not in yfinance info — set by shareholding module
        promoter_pledge   = None,
        fii_holding       = None,
        market_cap_cr     = market_cap_cr,
        beta              = _i("beta"),
        sector            = info.get("sector", ""),
        eps               = _i("trailingEps", "epsTrailingTwelveMonths"),
        dividend_yield    = _pct_from_frac("dividendYield"),
    )


def compute_personas(metrics: StockMetrics) -> list[dict]:
    """
    Run all 9 personas and return a list of dicts matching the
    TypeScript Persona interface.
    """
    results = []
    for persona in _ALL_PERSONAS:
        try:
            result = persona.score(metrics)
            results.append({
                "id":           result.id,
                "name":         result.name,
                "photoUrl":     result.photo_url,
                "icon":         result.icon,
                "score":        result.score,
                "verdict":      result.verdict,
                "verdictColor": result.verdict_color,
                "summary":      result.summary,
                "criteria":     result.criteria,
            })
        except Exception as exc:
            # Never let one persona crash the whole response
            import logging
            logging.getLogger(__name__).warning("Persona %s failed: %s", persona.ID, exc)

    return results
