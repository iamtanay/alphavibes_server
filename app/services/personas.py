"""
app/services/personas.py
─────────────────────────────────────────────────────────────────────────────
Investor Persona Scoring Engine.

4 personas. Zero LLM calls. Every score is computed from real financial
data using explicit, auditable rules.

REDESIGN (Roadmap Phase 3):
  • Cut from 9 to 4 personas — each with a DISTINCT, non-overlapping thesis:
      1. Warren Buffett  — value + economic moat
      2. Peter Lynch     — GARP (growth at a reasonable price)
      3. Rakesh Jhunjhunwala — India macro + high-conviction growth
      4. Momentum Trader — pure technical, anti-fundamental

  • Removed: Benjamin Graham (overlaps Buffett), Growth Investor (overlaps
    Lynch), Jim Simons (no RSI data quality), Cathie Wood (India context poor),
    Mark Minervini (overlaps Momentum Trader).

  • Added weighted criteria — each criterion now carries an explicit weight
    (max_points) so the UI can show "earned 14/20 pts" per criterion.

  • Added conflict_analysis() function — detects when Buffett and Momentum
    Trader have conflicting verdicts (value vs technical) and generates a
    2-sentence insight automatically. This is AlphaVibes' differentiator.

  • Philosophy text now maps 1:1 to scoring criteria (no false promises).
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
class PersonaCriteria:
    label: str
    value: str           # Display string, e.g. "22.1%"
    passed: bool
    max_points: int      # Weight — how much this criterion is worth
    earned_points: int   # Actual points scored (0 to max_points)


@dataclass
class PersonaResult:
    id: str
    name: str
    photo_url: str | None
    icon: str | None
    score: int                                   # 0–100
    verdict: str                                 # "Strong Match" etc.
    verdict_color: str                           # "success" | "info" | "warning" | "danger"
    summary: str                                 # 1-line plain-English
    criteria: list[PersonaCriteria] = field(default_factory=list)


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
    # Total possible points — must equal sum of all criterion max_points
    MAX_POINTS: int = 100

    def score(self, m: StockMetrics) -> PersonaResult:
        """Override in subclass — return PersonaResult."""
        raise NotImplementedError

    def _result(self, total_earned: float, criteria: list[PersonaCriteria]) -> PersonaResult:
        # Normalise to 0–100 based on MAX_POINTS
        max_achievable = self.MAX_POINTS
        normalised = int(round((total_earned / max_achievable) * 100)) if max_achievable > 0 else 0
        score = max(0, min(100, normalised))
        verdict, color = _verdict(score)
        summary = self._make_summary(score, criteria)

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

    def _make_summary(self, score: int, criteria: list[PersonaCriteria]) -> str:
        """Generate a plain-English 10–15 word summary."""
        passing = [c for c in criteria if c.passed]
        failing = [c for c in criteria if not c.passed]
        first_name = self.NAME.split()[0]
        total = len(passing) + len(failing)

        if score >= 85:
            return f"Strong alignment — meets {len(passing)} of {total} core {first_name} criteria."
        if score >= 65:
            return f"Good match — {len(passing)} of {total} criteria met, with room to improve."
        if score >= 45:
            return f"Mixed signals — meets some but fails on key {first_name} requirements."
        return f"Weak alignment — fails {len(failing)} of {total} {first_name} criteria."

    def _criterion(
        self,
        label: str,
        actual: Any,
        passed: bool,
        max_points: int,
        earned_points: int,
        format_fn=None,
    ) -> PersonaCriteria:
        """Build a weighted criterion for the UI."""
        if actual is None:
            display = "N/A"
        elif format_fn:
            display = format_fn(actual)
        elif isinstance(actual, float):
            display = f"{actual:.1f}"
        else:
            display = str(actual)
        return PersonaCriteria(
            label=label,
            value=display,
            passed=passed,
            max_points=max_points,
            earned_points=max(0, min(earned_points, max_points)),
        )


# ── Persona 1: Warren Buffett ─────────────────────────────────────────────
#
# Thesis: Invest in businesses with durable competitive moats, run by honest
# managers, at a fair price. Hold forever. Low debt, high ROE, consistent
# earnings — the 6 things Buffett actually checks.
#
class WarrenBuffettPersona(BasePersona):
    ID        = "warren-buffett"
    NAME      = "Warren Buffett"
    PHOTO_URL = "/personas/warren-buffett.svg"
    ICON      = None
    MAX_POINTS = 100  # sum of all max_points below

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # 1. ROE > 15% (max 20 pts) — proxy for economic moat
        roe = _safe(m.roe)
        if roe is not None:
            pts = 20 if roe >= 15 else (12 if roe >= 12 else (5 if roe >= 8 else 0))
            total += pts
            criteria.append(self._criterion(
                "ROE consistently above 15%", roe, roe >= 15,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 2. Debt / Equity < 0.5 (max 20 pts) — financial fortress
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 20 if de <= 0.5 else (12 if de <= 1.0 else (5 if de <= 1.5 else 0))
            total += pts
            criteria.append(self._criterion(
                "Low Debt / Equity (< 0.5)", de, de <= 0.5,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.2f}",
            ))

        # 3. Operating Margin > 15% (max 20 pts) — pricing power
        om = _safe(m.operating_margin)
        if om is not None:
            pts = 20 if om >= 15 else (10 if om >= 10 else (3 if om >= 5 else 0))
            total += pts
            criteria.append(self._criterion(
                "Operating Margin above 15%", om, om >= 15,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 4. Gross Margin > 40% (max 15 pts) — product differentiation
        gm = _safe(m.gross_margin)
        if gm is not None:
            pts = 15 if gm >= 40 else (8 if gm >= 25 else (3 if gm >= 15 else 0))
            total += pts
            criteria.append(self._criterion(
                "Gross Margin above 40%", gm, gm >= 40,
                max_points=15, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 5. EPS Growth > 10% (max 15 pts) — consistent earnings power
        eg = _safe(m.eps_growth_yoy) or _safe(m.eps_cagr_3y)
        if eg is not None:
            pts = 15 if eg >= 10 else (8 if eg >= 5 else (3 if eg >= 0 else 0))
            total += pts
            criteria.append(self._criterion(
                "EPS growing at 10%+ annually", eg, eg >= 10,
                max_points=15, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}% YoY",
            ))

        # 6. Reasonable PE < 25 (max 10 pts) — fair price, not momentum price
        pe = _safe(m.pe)
        if pe is not None and pe > 0:
            pts = 10 if pe < 20 else (6 if pe < 25 else (2 if pe < 35 else 0))
            total += pts
            criteria.append(self._criterion(
                "Reasonable valuation (PE < 25)", pe, pe < 25,
                max_points=10, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}x",
            ))

        return self._result(total, criteria)


# ── Persona 2: Peter Lynch ────────────────────────────────────────────────
#
# Thesis: Buy what you understand. GARP — growth at a reasonable price.
# PEG is the single most important number. Look for fast growers with
# manageable debt that insiders believe in enough to own themselves.
#
class PeterLynchPersona(BasePersona):
    ID        = "peter-lynch"
    NAME      = "Peter Lynch"
    PHOTO_URL = "/personas/peter-lynch.svg"
    ICON      = None
    MAX_POINTS = 100

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # 1. PEG < 1.0 (max 25 pts) — Lynch's single favourite signal
        peg = _safe(m.peg)
        if peg is not None and peg > 0:
            pts = 25 if peg < 1.0 else (15 if peg < 1.5 else (5 if peg < 2.0 else 0))
            total += pts
            criteria.append(self._criterion(
                "PEG Ratio below 1.0 (value with growth)", peg, peg < 1.0,
                max_points=25, earned_points=int(pts),
                format_fn=lambda v: f"{v:.2f}",
            ))

        # 2. EPS Growth > 15% (max 25 pts) — the "fast grower" filter
        eg = _safe(m.eps_growth_yoy) or _safe(m.eps_cagr_3y)
        if eg is not None:
            pts = 25 if eg >= 15 else (15 if eg >= 10 else (5 if eg >= 5 else 0))
            total += pts
            criteria.append(self._criterion(
                "EPS growth above 15% annually", eg, eg >= 15,
                max_points=25, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 3. Revenue Growth > 20% (max 20 pts) — growth must be top-line driven
        rg = _safe(m.revenue_growth_yoy) or _safe(m.revenue_cagr_3y)
        if rg is not None:
            pts = 20 if rg >= 20 else (12 if rg >= 12 else (4 if rg >= 5 else 0))
            total += pts
            criteria.append(self._criterion(
                "Revenue growth above 20%", rg, rg >= 20,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 4. Insider / Promoter Holding > 20% (max 15 pts) — skin in the game
        ph = _safe(m.promoter_holding)
        if ph is not None:
            pts = 15 if ph >= 20 else (8 if ph >= 10 else 0)
            total += pts
            criteria.append(self._criterion(
                "Insider holding above 20% (skin in game)", ph, ph >= 20,
                max_points=15, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 5. Net Margin > 10% (max 10 pts) — profitable growth
        nm = _safe(m.net_margin)
        if nm is not None:
            pts = 10 if nm >= 10 else (5 if nm >= 5 else 0)
            total += pts
            criteria.append(self._criterion(
                "Net Profit Margin above 10%", nm, nm >= 10,
                max_points=10, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 6. Debt / Equity < 0.5 (max 5 pts) — Lynch avoids over-leveraged growers
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 5 if de <= 0.5 else (2 if de <= 1.0 else 0)
            total += pts
            criteria.append(self._criterion(
                "Low Debt / Equity (< 0.5)", de, de <= 0.5,
                max_points=5, earned_points=int(pts),
                format_fn=lambda v: f"{v:.2f}",
            ))

        return self._result(total, criteria)


# ── Persona 3: Rakesh Jhunjhunwala ────────────────────────────────────────
#
# Thesis: India-exclusive. High-conviction bets on India's domestic growth
# story. Prefers mid/small caps with 3Y revenue CAGR > 20%, high promoter
# conviction (holding + low pledge), and strong EPS momentum. Macro-driven,
# patient, and concentrated.
#
class RakeshJhunjhunwalaPersona(BasePersona):
    ID        = "rakesh-jhunjhunwala"
    NAME      = "Rakesh Jhunjhunwala"
    PHOTO_URL = "/personas/rakesh-jhunjhunwala.svg"
    ICON      = None
    MAX_POINTS = 100

    _MCAP_LOW  = 500      # ₹500 Cr
    _MCAP_HIGH = 25_000   # ₹25,000 Cr

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # 1. Revenue CAGR 3Y > 20% (max 20 pts) — India growth story
        cagr = _safe(m.revenue_cagr_3y) or _safe(m.revenue_growth_yoy)
        if cagr is not None:
            pts = 20 if cagr >= 20 else (12 if cagr >= 12 else (5 if cagr >= 5 else 0))
            total += pts
            criteria.append(self._criterion(
                "3Y Revenue CAGR above 20%", cagr, cagr >= 20,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 2. EPS CAGR 3Y > 20% (max 20 pts) — earnings must match revenue
        ep_cagr = _safe(m.eps_cagr_3y) or _safe(m.eps_growth_yoy)
        if ep_cagr is not None:
            pts = 20 if ep_cagr >= 20 else (12 if ep_cagr >= 12 else (4 if ep_cagr >= 5 else 0))
            total += pts
            criteria.append(self._criterion(
                "3Y EPS CAGR above 20%", ep_cagr, ep_cagr >= 20,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 3. ROE > 20% (max 20 pts) — high-quality growth
        roe = _safe(m.roe)
        if roe is not None:
            pts = 20 if roe >= 20 else (10 if roe >= 15 else (3 if roe >= 10 else 0))
            total += pts
            criteria.append(self._criterion(
                "ROE above 20%", roe, roe >= 20,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 4. Promoter holding > 50%, low pledge (max 20 pts) — conviction
        ph = _safe(m.promoter_holding)
        pp = _safe(m.promoter_pledge) or 0.0
        if ph is not None:
            in_zone = ph >= 50 and pp < 10
            pts = 20 if in_zone else (10 if ph >= 40 else (3 if ph >= 25 else 0))
            total += pts
            pledge_note = f"{v:.1f}% hold, {pp:.1f}% pledged" if False else None
            criteria.append(self._criterion(
                "Promoter holding > 50% with low pledge", ph,
                in_zone,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}% (pledge: {pp:.1f}%)",
            ))

        # 5. D/E < 1.0 (max 10 pts) — leverage acceptable but not extreme
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 10 if de <= 1.0 else (5 if de <= 1.5 else 0)
            total += pts
            criteria.append(self._criterion(
                "Debt / Equity below 1.0", de, de <= 1.0,
                max_points=10, earned_points=int(pts),
                format_fn=lambda v: f"{v:.2f}",
            ))

        # 6. Mid/small cap sweet spot (max 10 pts) — Jhunjhunwala's edge
        mc = _safe(m.market_cap_cr)
        if mc is not None:
            in_range = self._MCAP_LOW <= mc <= self._MCAP_HIGH
            pts = 10 if in_range else (3 if mc < self._MCAP_LOW * 3 else 0)
            total += pts
            criteria.append(self._criterion(
                f"Market cap ₹{self._MCAP_LOW:,}–{self._MCAP_HIGH:,} Cr",
                mc, in_range,
                max_points=10, earned_points=int(pts),
                format_fn=lambda v: f"₹{v:,.0f} Cr",
            ))

        return self._result(total, criteria)



# ── Factory list (4 active personas) ─────────────────────────────────────

# ── Persona 4: Charlie Munger ────────────────────────────────────────────
#
# Thesis: Invert, always invert. Munger is even stricter than Buffett —
# he demands outstanding ROIC (not just ROE), near-zero debt, operating
# leverage (gross margin expanding), no commodity businesses, and a wide
# moat backed by pricing power. He uses inversion to eliminate bad businesses
# first, then asks: is this truly exceptional? Most companies fail immediately.
#
class CharlieMungerPersona(BasePersona):
    ID        = "charlie-munger"
    NAME      = "Charlie Munger"
    PHOTO_URL = "/personas/charlie-munger.svg"
    ICON      = None
    MAX_POINTS = 100

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # 1. ROIC > 15% (max 25 pts) — Munger's primary quality signal
        #    Uses ROIC if available, falls back to ROCE proxy (ROE * equity ratio)
        roic = _safe(m.roic) or _safe(m.roe)
        if roic is not None:
            pts = 25 if roic >= 20 else (18 if roic >= 15 else (8 if roic >= 10 else 0))
            total += pts
            criteria.append(self._criterion(
                "ROIC above 15% (exceptional capital allocation)", roic, roic >= 15,
                max_points=25, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 2. Gross Margin > 50% (max 20 pts) — Munger demands pricing power
        gm = _safe(m.gross_margin)
        if gm is not None:
            pts = 20 if gm >= 50 else (12 if gm >= 35 else (4 if gm >= 20 else 0))
            total += pts
            criteria.append(self._criterion(
                "Gross Margin above 50% (pricing power moat)", gm, gm >= 50,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 3. Debt / Equity < 0.3 (max 20 pts) — Munger is stricter than Buffett on debt
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 20 if de <= 0.3 else (12 if de <= 0.6 else (3 if de <= 1.0 else 0))
            total += pts
            criteria.append(self._criterion(
                "Very low Debt / Equity (< 0.3) — fortress balance sheet", de, de <= 0.3,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.2f}",
            ))

        # 4. Net Margin > 20% (max 20 pts) — operating leverage indicator
        nm = _safe(m.net_margin)
        if nm is not None:
            pts = 20 if nm >= 20 else (12 if nm >= 12 else (4 if nm >= 6 else 0))
            total += pts
            criteria.append(self._criterion(
                "Net Margin above 20% (operating leverage)", nm, nm >= 20,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 5. ROE > 20% — consistently exceptional (max 15 pts)
        roe = _safe(m.roe)
        if roe is not None:
            pts = 15 if roe >= 20 else (8 if roe >= 15 else (2 if roe >= 10 else 0))
            total += pts
            criteria.append(self._criterion(
                "ROE above 20% consistently", roe, roe >= 20,
                max_points=15, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        return self._result(total, criteria)


# ── Persona 5: Benjamin Graham ────────────────────────────────────────────
#
# Thesis: The father of value investing. Buy a dollar for fifty cents.
# Graham demands a margin of safety so large that even if he's wrong
# about the business, the price protects him. P/B < 1.5, P/E < 15,
# dividend paying, current ratio > 2, net-net territory preferred.
# He has zero interest in growth — only in not losing money.
#
class BenjaminGrahamPersona(BasePersona):
    ID        = "benjamin-graham"
    NAME      = "Benjamin Graham"
    PHOTO_URL = "/personas/benjamin-graham.svg"
    ICON      = None
    MAX_POINTS = 100

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # 1. P/E < 15 (max 25 pts) — the primary valuation gate
        pe = _safe(m.pe)
        if pe is not None and pe > 0:
            pts = 25 if pe < 10 else (18 if pe < 15 else (8 if pe < 20 else 0))
            total += pts
            criteria.append(self._criterion(
                "P/E Ratio below 15 (deep value threshold)", pe, pe < 15,
                max_points=25, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}x",
            ))

        # 2. P/B < 1.5 (max 25 pts) — near net-asset-value
        pb = _safe(m.pb)
        if pb is not None and pb > 0:
            pts = 25 if pb < 1.0 else (15 if pb < 1.5 else (5 if pb < 2.5 else 0))
            total += pts
            criteria.append(self._criterion(
                "P/B Ratio below 1.5 (near asset value)", pb, pb < 1.5,
                max_points=25, earned_points=int(pts),
                format_fn=lambda v: f"{v:.2f}x",
            ))

        # 3. Current Ratio > 2.0 (max 20 pts) — financial safety
        cr = _safe(m.current_ratio)
        if cr is not None:
            pts = 20 if cr >= 2.0 else (10 if cr >= 1.5 else (3 if cr >= 1.0 else 0))
            total += pts
            criteria.append(self._criterion(
                "Current Ratio above 2.0 (financial safety)", cr, cr >= 2.0,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.2f}",
            ))

        # 4. Dividend paying (max 15 pts) — Graham wants cash back
        dy = _safe(m.dividend_yield)
        if dy is not None:
            pays_div = dy > 0
            pts = 15 if dy >= 2.0 else (10 if dy > 0 else 0)
            total += pts
            criteria.append(self._criterion(
                "Dividend paying (income + margin of safety)", dy, pays_div,
                max_points=15, earned_points=int(pts),
                format_fn=lambda v: f"{v:.2f}%" if v > 0 else "None",
            ))

        # 5. D/E < 0.5 (max 15 pts) — Graham is paranoid about leverage
        de = _safe(m.debt_equity)
        if de is not None:
            pts = 15 if de <= 0.5 else (8 if de <= 1.0 else 0)
            total += pts
            criteria.append(self._criterion(
                "Debt / Equity below 0.5 (safety first)", de, de <= 0.5,
                max_points=15, earned_points=int(pts),
                format_fn=lambda v: f"{v:.2f}",
            ))

        return self._result(total, criteria)


# ── Persona 6: Vijay Kedia ────────────────────────────────────────────────
#
# Thesis: SMILE — Small/Medium cap, In a niche, Longevity of business model,
# and Emerging. Kedia hunts for undiscovered Indian niche businesses before
# institutional money arrives. He wants fast CAGR, very high promoter
# conviction, low FII holding (means the big money hasn't found it yet),
# and a clear 5-year compounding thesis. He's willing to pay high multiples
# for truly exceptional small-cap franchises.
#
class VijayKediaPersona(BasePersona):
    ID        = "vijay-kedia"
    NAME      = "Vijay Kedia"
    PHOTO_URL = "/personas/vijay-kedia.svg"
    ICON      = None
    MAX_POINTS = 100

    _MCAP_MAX = 10_000  # ₹10,000 Cr — prefers undiscovered names

    def score(self, m: StockMetrics) -> PersonaResult:
        total = 0.0
        criteria = []

        # 1. Revenue CAGR 3Y > 25% (max 25 pts) — must be a fast compounder
        cagr = _safe(m.revenue_cagr_3y) or _safe(m.revenue_growth_yoy)
        if cagr is not None:
            pts = 25 if cagr >= 25 else (15 if cagr >= 15 else (5 if cagr >= 8 else 0))
            total += pts
            criteria.append(self._criterion(
                "Revenue CAGR > 25% (fast compounder)", cagr, cagr >= 25,
                max_points=25, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 2. Promoter Holding > 60% with very low pledge (max 25 pts)
        #    Kedia's non-negotiable: promoter must own majority and be committed
        ph = _safe(m.promoter_holding)
        pp = _safe(m.promoter_pledge) or 0.0
        if ph is not None:
            high_conviction = ph >= 60 and pp < 5
            pts = 25 if high_conviction else (14 if ph >= 50 and pp < 10 else (5 if ph >= 40 else 0))
            total += pts
            criteria.append(self._criterion(
                "Promoter holding > 60% with pledge < 5%", ph,
                high_conviction,
                max_points=25, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}% (pledge: {pp:.1f}%)",
            ))

        # 3. Low FII holding < 15% (max 20 pts) — undiscovered by big money
        fii = _safe(m.fii_holding)
        if fii is not None:
            pts = 20 if fii < 5 else (14 if fii < 10 else (8 if fii < 15 else 0))
            total += pts
            criteria.append(self._criterion(
                "FII holding below 15% (undiscovered opportunity)", fii, fii < 15,
                max_points=20, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        # 4. Small/mid-cap (< ₹10,000 Cr) (max 15 pts) — Kedia's sweet spot
        mc = _safe(m.market_cap_cr)
        if mc is not None:
            in_zone = mc is not None and mc <= self._MCAP_MAX
            pts = 15 if mc <= 2000 else (10 if mc <= self._MCAP_MAX else 0)
            total += pts
            criteria.append(self._criterion(
                f"Market cap below ₹{self._MCAP_MAX:,} Cr (undiscovered niche)",
                mc, in_zone,
                max_points=15, earned_points=int(pts),
                format_fn=lambda v: f"₹{v:,.0f} Cr",
            ))

        # 5. ROE > 20% (max 15 pts) — quality matters even in small caps
        roe = _safe(m.roe)
        if roe is not None:
            pts = 15 if roe >= 20 else (8 if roe >= 15 else (2 if roe >= 10 else 0))
            total += pts
            criteria.append(self._criterion(
                "ROE above 20%", roe, roe >= 20,
                max_points=15, earned_points=int(pts),
                format_fn=lambda v: f"{v:.1f}%",
            ))

        return self._result(total, criteria)


_ALL_PERSONAS: list[BasePersona] = [
    WarrenBuffettPersona(),
    PeterLynchPersona(),
    RakeshJhunjhunwalaPersona(),
    CharlieMungerPersona(),
    BenjaminGrahamPersona(),
    VijayKediaPersona(),
]


# ── Conflict analysis ─────────────────────────────────────────────────────

def conflict_analysis(persona_results: list[dict]) -> list[dict]:
    """
    Detect ALL meaningful conflicts across the 6 personas and return a list.
    Each conflict is self-contained with both persona IDs/names and full text.

    Meaningful pairs (philosophies that genuinely disagree):
      Buffett     vs Lynch       — stable moat vs fast growth
      Buffett     vs Kedia       — large moat quality vs undiscovered niche
      Munger      vs Lynch       — strict quality vs growth-at-any-price
      Munger      vs Jhunjhunwala— ultra-quality vs India growth conviction
      Graham      vs Lynch       — deep value vs GARP growth
      Graham      vs Jhunjhunwala— cheap by assets vs expensive growth thesis
      Graham      vs Kedia       — margin of safety vs paying up for growth
      Jhunjhunwala vs Kedia      — mid-cap India vs undiscovered small-cap
      Buffett     vs Graham      — fair price for quality vs near net-net only

    Threshold: high side >= 65, low side <= 45 (gap >= 20 pts).
    Returns a (possibly empty) list. Multiple conflicts can fire simultaneously.
    """
    scores   = {r["id"]: r["score"]   for r in persona_results}
    verdicts = {r["id"]: r["verdict"] for r in persona_results}
    names    = {r["id"]: r["name"]    for r in persona_results}

    conflicts: list[dict] = []

    # ── Shared detail templates keyed by frozenset pair ──────────────────

    _DETAIL: dict[frozenset, tuple[str, str]] = {
        # (text_when_A_high, text_when_B_high)  A=first id in pair definition

        frozenset(["warren-buffett", "peter-lynch"]): (
            # Buffett high, Lynch low
            "Buffett sees a durable moat here — consistent margins, low debt, and strong ROE. "
            "But Lynch would call this a 'stalwart' at best: revenue growth is too slow for a "
            "ten-bagger thesis, and the PEG isn't exciting enough to get him off the bench. "
            "Stable long-term hold vs a missed growth opportunity — depends on your time horizon.",
            # Lynch high, Buffett low
            "Lynch sees a fast grower with an attractive PEG and accelerating revenue — exactly "
            "the kind of stock he found walking through a mall. But Buffett doesn't see a moat: "
            "ROE is underwhelming, margins are thin, or the valuation is already stretched. "
            "Growth investors will love it; value investors will pass.",
        ),

        frozenset(["warren-buffett", "vijay-kedia"]): (
            # Buffett high, Kedia low
            "Buffett likes this business — quality moat, honest returns, and reasonable price. "
            "But Kedia won't touch it: it's too large, too widely followed, and FII ownership "
            "means institutional money has already priced in the upside. Kedia wants to be "
            "first in; this stock is already on every large-cap fund's radar.",
            # Kedia high, Buffett low
            "Kedia sees a hidden gem — high promoter conviction, low FII radar, and explosive "
            "revenue growth in a niche Buffett probably can't explain in 10 minutes. Buffett "
            "would pass: the moat isn't proven, margins may be thin, and the business might "
            "not compound for 20 years. High upside vs high uncertainty.",
        ),

        frozenset(["charlie-munger", "peter-lynch"]): (
            # Munger high, Lynch low
            "Munger sees an exceptional business — outstanding ROIC, fortress balance sheet, "
            "and pricing power that compounds quietly. Lynch's problem: the growth rate isn't "
            "exciting enough for his 'fast grower' or 'stalwart' buckets, and the PEG doesn't "
            "scream cheap. Munger would hold forever; Lynch would look elsewhere.",
            # Lynch high, Munger low
            "Lynch sees rapid revenue and EPS growth at a reasonable PEG — the hallmarks of a "
            "ten-bagger. Munger would invert and ask: what could go wrong? The gross margin "
            "isn't wide enough, debt is higher than he'd like, or ROIC doesn't prove an "
            "enduring moat. One buys the growth story; the other demands proof.",
        ),

        frozenset(["charlie-munger", "rakesh-jhunjhunwala"]): (
            # Munger high, Jhunjhunwala low
            "Munger sees exceptional quality — ROIC above 15%, near-zero debt, and gross margin "
            "reflecting real pricing power. Jhunjhunwala's concern: it's either too large a "
            "company to deliver the 5x returns he needs, or the India domestic growth angle "
            "isn't compelling enough. Munger buys quality at any size; RJ wants the India "
            "story to be central.",
            # Jhunjhunwala high, Munger low
            "Jhunjhunwala sees a high-conviction India growth play — strong 3Y CAGR, high "
            "promoter holding, and the tailwind of India's domestic consumption boom. Munger "
            "would invert: gross margin isn't wide enough to call it an exceptional franchise, "
            "and debt is higher than he's comfortable with. Growth riding a macro theme vs "
            "a truly exceptional business.",
        ),

        frozenset(["benjamin-graham", "peter-lynch"]): (
            # Graham high, Lynch low
            "Graham found a margin of safety here — P/E below 15, P/B near book value, "
            "dividend paying, and a solid current ratio. Lynch would consider it a 'slow "
            "grower' or 'asset play' — perfectly acceptable but not a ten-bagger. Lynch "
            "wants growth in the 15–25% range; this stock's best days may be behind it.",
            # Lynch high, Graham low
            "Lynch sees a classic fast grower — high EPS momentum, expanding revenue, and "
            "an attractive PEG. Graham sees a speculation: P/E is above 15, P/B offers no "
            "margin of safety, and there's little or no dividend. For Graham, paying up for "
            "growth violates the first principle of investing: don't lose money.",
        ),

        frozenset(["benjamin-graham", "rakesh-jhunjhunwala"]): (
            # Graham high, Jhunjhunwala low
            "Graham sees a cheap business — near asset value, paying a dividend, low debt. "
            "Jhunjhunwala would call this a 'value trap': no growth story, no India tailwind, "
            "and too small or too slow to ride the structural growth he's positioning for. "
            "Graham protects capital; Jhunjhunwala compounds it.",
            # Jhunjhunwala high, Graham low
            "Jhunjhunwala sees India's next compounding machine — high CAGR, high promoter "
            "skin in the game, and a domestic consumption thesis. Graham sees danger: P/E is "
            "too high, P/B offers no safety net, and the dividend is zero or minimal. "
            "Jhunjhunwala is betting on future growth; Graham is demanding value today.",
        ),

        frozenset(["benjamin-graham", "vijay-kedia"]): (
            # Graham high, Kedia low
            "Graham sees a classic cheap stock — low P/E, low P/B, dividend paying, and a "
            "strong current ratio. Kedia would walk away: it's probably a large, slow, "
            "widely-covered company with no hidden upside, or a small company with no growth "
            "story. Graham buys the balance sheet; Kedia buys the future.",
            # Kedia high, Graham low
            "Kedia sees an undiscovered rocket — high promoter conviction, low FII exposure, "
            "and blistering CAGR in a niche nobody is watching yet. Graham sees speculation: "
            "P/E and P/B are too high for any meaningful margin of safety, and with no "
            "dividend, there's nothing protecting the downside. Kedia bets on growth; "
            "Graham bets on value.",
        ),

        frozenset(["rakesh-jhunjhunwala", "vijay-kedia"]): (
            # Jhunjhunwala high, Kedia low
            "Jhunjhunwala sees a strong India franchise — solid CAGR, high promoter holding, "
            "and a proven mid-cap compounding track record. Kedia would say it's too big and "
            "too well-known: FII ownership is already elevated, the market cap is beyond his "
            "sweet spot, and the easy money has already been made. Jhunjhunwala holds "
            "conviction stocks; Kedia wants to find them before everyone else does.",
            # Kedia high, Jhunjhunwala low
            "Kedia sees a hidden gem — tiny market cap, high promoter ownership, negligible "
            "FII presence, and explosive growth in a niche. Jhunjhunwala would hesitate: "
            "revenue CAGR may be strong, but the business is too small and illiquid for a "
            "high-conviction position, or the India macro theme isn't playing out yet. "
            "Kedia fishes in smaller ponds; Jhunjhunwala needs a bigger wave.",
        ),

        frozenset(["warren-buffett", "benjamin-graham"]): (
            # Buffett high, Graham low
            "Buffett sees a wonderful business at a fair price — strong ROE, durable margins, "
            "and low debt. Graham would call the price too high: P/E above 15 and P/B above "
            "1.5 means there's no margin of safety. Buffett evolved away from Graham's "
            "deep-discount framework; Graham never would.",
            # Graham high, Buffett low
            "Graham sees a cheap, safe business — low P/E, low P/B, dividend, solid current "
            "ratio. But Buffett doesn't see a moat: ROE is mediocre, margins are thin, and "
            "there's no evidence the business can compound capital for 20 years. Graham "
            "protects from loss; Buffett seeks compounding wealth.",
        ),
    }

    def _check(id_a: str, id_b: str) -> None:
        sa = scores.get(id_a, 50)
        sb = scores.get(id_b, 50)
        if abs(sa - sb) < 20:
            return  # Not a significant conflict
        pair = frozenset([id_a, id_b])
        detail_pair = _DETAIL.get(pair)
        if detail_pair is None:
            return  # Pair not defined — no conflict text

        if sa >= 65 and sb <= 45:
            high_id, low_id = id_a, id_b
            detail_text = detail_pair[0]
            headline = f"{names.get(id_a, id_a)} says buy, {names.get(id_b, id_b)} says pass"
        elif sb >= 65 and sa <= 45:
            high_id, low_id = id_b, id_a
            detail_text = detail_pair[1]
            headline = f"{names.get(id_b, id_b)} says buy, {names.get(id_a, id_a)} says pass"
        else:
            return

        conflicts.append({
            "persona_a_id":   high_id,
            "persona_b_id":   low_id,
            "persona_a_name": names.get(high_id, high_id),
            "persona_b_name": names.get(low_id, low_id),
            "headline":       headline,
            "detail":         detail_text,
        })

    # Check all defined pairs
    _check("warren-buffett",      "peter-lynch")
    _check("warren-buffett",      "vijay-kedia")
    _check("warren-buffett",      "benjamin-graham")
    _check("charlie-munger",      "peter-lynch")
    _check("charlie-munger",      "rakesh-jhunjhunwala")
    _check("benjamin-graham",     "peter-lynch")
    _check("benjamin-graham",     "rakesh-jhunjhunwala")
    _check("benjamin-graham",     "vijay-kedia")
    _check("rakesh-jhunjhunwala", "vijay-kedia")

    return conflicts



# ── extract_metrics ───────────────────────────────────────────────────────

def extract_metrics(
    info: dict,
    history_5y: "pd.DataFrame",
    financials: "pd.DataFrame",
    technicals: dict,
) -> StockMetrics:
    """
    Extract a flat StockMetrics bag from raw yfinance data.
    Called once; the result is passed to all persona scorers.
    promoter_holding / promoter_pledge / fii_holding are set to None here —
    they MUST be patched by the caller (analyser.py) after compute_shareholding()
    runs. See analyser.py FIX #3.
    """
    import math as _math
    import pandas as _pd

    def _i(key: str, *fallbacks) -> float | None:
        for k in [key, *fallbacks]:
            v = info.get(k)
            if v is not None:
                try:
                    f = float(v)
                    if not (_math.isnan(f) or _math.isinf(f)):
                        return f
                except (TypeError, ValueError):
                    pass
        return None

    def _pct_from_frac(key: str) -> float | None:
        v = _i(key)
        return round(v * 100, 1) if v is not None else None

    # ── Price momentum ─────────────────────────────────────────────────────
    momentum_6m = None
    price_vs_sma200 = None
    volume_ratio = None

    if history_5y is not None and not history_5y.empty:
        try:
            close = history_5y["Close"].dropna()
            vol   = history_5y["Volume"].dropna()

            if len(close) >= 126:
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

    # ── Revenue CAGR 3Y ───────────────────────────────────────────────────
    revenue_cagr_3y = None
    eps_cagr_3y = None

    if financials is not None and not financials.empty:
        try:
            # Try multiple row labels (income_stmt vs financials use different names)
            rev_row = None
            for label in ("Total Revenue", "Revenue"):
                if label in financials.index:
                    rev_row = label
                    break

            if rev_row and len(financials.columns) >= 3:
                rev_now = float(financials.loc[rev_row, financials.columns[0]])
                rev_3ya = float(financials.loc[rev_row, financials.columns[min(3, len(financials.columns)-1)]])
                if rev_3ya > 0:
                    revenue_cagr_3y = round(((rev_now / rev_3ya) ** (1/3) - 1) * 100, 1)

            eps_row = None
            for label in ("Basic EPS", "Diluted EPS"):
                if label in financials.index:
                    eps_row = label
                    break

            if eps_row and len(financials.columns) >= 3:
                eps_now = float(financials.loc[eps_row, financials.columns[0]])
                eps_3ya = float(financials.loc[eps_row, financials.columns[min(3, len(financials.columns)-1)]])
                if eps_3ya > 0 and eps_now > 0:
                    eps_cagr_3y = round(((eps_now / eps_3ya) ** (1/3) - 1) * 100, 1)
        except Exception:
            pass

    # ── YoY growth ────────────────────────────────────────────────────────
    rev_yoy = None
    eps_yoy = None

    if financials is not None and not financials.empty:
        try:
            rev_row = next((l for l in ("Total Revenue", "Revenue") if l in financials.index), None)
            if rev_row and len(financials.columns) >= 2:
                r_now  = float(financials.loc[rev_row, financials.columns[0]])
                r_prev = float(financials.loc[rev_row, financials.columns[1]])
                if r_prev != 0:
                    rev_yoy = round(((r_now - r_prev) / abs(r_prev)) * 100, 1)

            eps_row = next((l for l in ("Basic EPS", "Diluted EPS") if l in financials.index), None)
            if eps_row and len(financials.columns) >= 2:
                e_now  = float(financials.loc[eps_row, financials.columns[0]])
                e_prev = float(financials.loc[eps_row, financials.columns[1]])
                if e_prev != 0:
                    eps_yoy = round(((e_now - e_prev) / abs(e_prev)) * 100, 1)
        except Exception:
            pass

    # D/E normalisation
    de = _i("debtToEquity")
    if de is not None and de > 10:
        de = de / 100

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
        # NOTE: promoter_holding / promoter_pledge / fii_holding are intentionally
        # None here. The caller (analyser.py) patches them from compute_shareholding()
        # before calling compute_personas().
        promoter_holding  = None,
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
    Run all 4 active personas and return a list of dicts matching the
    TypeScript Persona interface, including weighted criteria and
    conflict analysis.
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
                # totalMaxPoints = the persona's full 100-pt ceiling, NOT the
                # sum of criteria that happened to have data. Some criteria are
                # skipped when yfinance returns None for a metric, so summing
                # returned criteria.maxPoints would give a wrong denominator.
                "totalMaxPoints": persona.MAX_POINTS,
                "criteria":     [
                    {
                        "label":         c.label,
                        "value":         c.value,
                        "pass":          c.passed,
                        "maxPoints":     c.max_points,
                        "earnedPoints":  c.earned_points,
                    }
                    for c in result.criteria
                ],
            })
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Persona %s failed: %s", persona.ID, exc)

    # Compute all conflicts across all persona pairs and return as a
    # separate top-level list — not stapled onto any one persona.
    conflicts = conflict_analysis(results)

    return {"personas": results, "conflicts": conflicts}


def compute_top_persona(info: dict) -> str:
    """
    Lightweight persona scoring from info dict alone — no OHLCV required.
    Used by the screener endpoint to assign a topPersona without a full analysis.

    Returns the persona ID of the highest scorer.
    """
    # Build a minimal StockMetrics from info only
    def _i(key: str, *fallbacks):
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

    def _pct(key: str):
        v = _i(key)
        return round(v * 100, 1) if v is not None else None

    de = _i("debtToEquity")
    if de is not None and de > 10:
        de = de / 100

    market_cap = _i("marketCap")
    market_cap_cr = round(market_cap / 10_000_000, 0) if market_cap else None

    metrics = StockMetrics(
        pe               = _i("trailingPE", "forwardPE"),
        peg              = _i("pegRatio"),
        roe              = _pct("returnOnEquity"),
        gross_margin     = _pct("grossMargins"),
        operating_margin = _pct("operatingMargins"),
        net_margin       = _pct("profitMargins"),
        revenue_growth_yoy = _pct("revenueGrowth"),
        eps_growth_yoy   = _pct("earningsGrowth"),
        debt_equity      = de,
        market_cap_cr    = market_cap_cr,
        promoter_holding = _pct("heldPercentInsiders"),  # approx from info
        beta             = _i("beta"),
        sector           = info.get("sector", ""),
    )

    best_id = "warren-buffett"
    best_score = -1

    for persona in _ALL_PERSONAS:
        try:
            result = persona.score(metrics)
            if result.score > best_score:
                best_score = result.score
                best_id = result.id
        except Exception:
            pass

    return best_id
