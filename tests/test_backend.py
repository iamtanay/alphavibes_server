"""
tests/test_backend.py
─────────────────────────────────────────────────────────────────────────────
Test suite for AlphaVibes backend.

Run with:
    pytest tests/ -v

Tests are organised by service module. They use mocked yfinance data so
they run offline and deterministically — no real network calls.
─────────────────────────────────────────────────────────────────────────────
"""

import math
import sys
import os

# Ensure the backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 300, start_price: float = 1000.0) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame for testing."""
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)

    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(0, 0.015)))

    prices = np.array(prices)
    highs  = prices * (1 + abs(rng.normal(0, 0.008, n)))
    lows   = prices * (1 - abs(rng.normal(0, 0.008, n)))
    opens  = prices * (1 + rng.normal(0, 0.005, n))
    vols   = rng.integers(100_000, 10_000_000, n).astype(float)

    df = pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  prices,
        "Volume": vols,
    }, index=dates)

    return df


def make_info(sector: str = "Information Technology") -> dict:
    """Generate a realistic yfinance info dict."""
    return {
        "longName":              "Test Company Ltd.",
        "shortName":             "TESTCO",
        "quoteType":             "EQUITY",
        "sector":                sector,
        "industry":              "Software",
        "currentPrice":          2500.0,
        "previousClose":         2450.0,
        "regularMarketPrice":    2500.0,
        "volume":                5_000_000,
        "marketCap":             250_000_000_000,  # 25,000 Cr
        "fiftyTwoWeekHigh":      2900.0,
        "fiftyTwoWeekLow":       1800.0,
        "trailingPE":            22.5,
        "forwardPE":             18.0,
        "priceToBook":           4.2,
        "pegRatio":              0.85,
        "enterpriseToEbitda":    14.0,
        "returnOnEquity":        0.221,  # 22.1%
        "returnOnAssets":        0.096,  # 9.6%
        "grossMargins":          0.246,  # 24.6%
        "operatingMargins":      0.166,  # 16.6%
        "profitMargins":         0.112,  # 11.2%
        "revenueGrowth":         0.124,  # 12.4%
        "earningsGrowth":        0.098,  # 9.8%
        "debtToEquity":          28.0,   # → 0.28 after normalisation
        "currentRatio":          1.45,
        "trailingEps":           104.2,
        "dividendYield":         0.0102, # 1.02%
        "heldPercentInsiders":   0.425,  # 42.5% promoter
        "heldPercentInstitutions": 0.38, # 38% institutional
        "beta":                  1.15,
    }


def make_financials(n_years: int = 5) -> pd.DataFrame:
    """
    Generate synthetic annual income statement in yfinance orientation:
      - Index   = line item labels
      - Columns = date Timestamps (most recent first)
    Supports any n_years by repeating values cyclically.
    """
    import numpy as np
    dates = pd.date_range("2024-03-31", periods=n_years, freq="-12ME")
    base = {
        "Total Revenue":    [945e9, 850e9, 760e9, 680e9, 610e9, 550e9, 495e9, 445e9],
        "Operating Income": [157e9, 140e9, 125e9, 110e9,  95e9,  85e9,  76e9,  68e9],
        "Net Income":       [ 94e9,  91e9,  76e9,  67e9,  58e9,  52e9,  47e9,  42e9],
        "Basic EPS":        [104.2, 101.0,  84.2,  74.4,  65.6,  58.9,  53.0,  47.7],
        "Gross Profit":     [232e9, 208e9, 186e9, 167e9, 150e9, 135e9, 121e9, 109e9],
        "EBITDA":           [178e9, 160e9, 143e9, 128e9, 114e9, 102e9,  92e9,  82e9],
    }
    data = {k: v[:n_years] for k, v in base.items()}
    # Build with dates as index (rows), metric names as columns; then transpose
    # so final shape is: index=metric names, columns=date timestamps
    return pd.DataFrame(data, index=dates).T


# ═══════════════════════════════════════════════════════════════════════════
# NSE STOCK DATA TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestNseStocks:
    def test_universe_not_empty(self):
        from app.data.nse_stocks import UNIVERSE
        assert len(UNIVERSE) > 50, "Universe should have many stocks"

    def test_ticker_to_meta_lookup(self):
        from app.data.nse_stocks import TICKER_TO_META
        assert "RELIANCE" in TICKER_TO_META
        meta = TICKER_TO_META["RELIANCE"]
        assert meta["yf_symbol"] == "RELIANCE.NS"
        assert meta["exchange"] == "NSE"

    def test_search_exact_match(self):
        from app.data.nse_stocks import search_stocks
        results = search_stocks("RELIANCE")
        assert len(results) > 0
        assert results[0]["ticker"] == "RELIANCE"

    def test_search_partial_name(self):
        from app.data.nse_stocks import search_stocks
        results = search_stocks("tata")
        assert len(results) > 0
        assert all("tata" in r["name"].lower() or "TATA" in r["ticker"] for r in results)

    def test_search_empty_query(self):
        from app.data.nse_stocks import search_stocks
        results = search_stocks("")
        # Should return empty or small default list
        assert isinstance(results, list)

    def test_search_limit_respected(self):
        from app.data.nse_stocks import search_stocks
        results = search_stocks("a", limit=3)
        assert len(results) <= 3

    def test_get_peers_returns_same_sector(self):
        from app.data.nse_stocks import get_peers, TICKER_TO_META
        peers = get_peers("TCS", limit=4)
        tcs_sector = TICKER_TO_META["TCS"]["sector"]
        for peer in peers:
            assert peer["ticker"] != "TCS"  # Should not include itself
            assert peer["sector"] == tcs_sector

    def test_get_peers_unknown_ticker(self):
        from app.data.nse_stocks import get_peers
        peers = get_peers("NOTEXIST")
        assert peers == []

    def test_resolve_yf_symbol_known(self):
        from app.data.nse_stocks import resolve_yf_symbol
        assert resolve_yf_symbol("RELIANCE") == "RELIANCE.NS"

    def test_resolve_yf_symbol_already_has_suffix(self):
        from app.data.nse_stocks import resolve_yf_symbol
        assert resolve_yf_symbol("RELIANCE.NS") == "RELIANCE.NS"

    def test_resolve_yf_symbol_unknown_appends_ns(self):
        from app.data.nse_stocks import resolve_yf_symbol
        sym = resolve_yf_symbol("NEWSTOCK")
        assert sym == "NEWSTOCK.NS"


# ═══════════════════════════════════════════════════════════════════════════
# TECHNICALS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestTechnicals:
    def setup_method(self):
        self.df_5y = make_ohlcv(300)
        self.df_1y = self.df_5y.tail(252).copy()

    def test_compute_returns_dict(self):
        from app.services.technicals import compute_technicals
        result = compute_technicals(self.df_5y, self.df_1y)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from app.services.technicals import compute_technicals
        result = compute_technicals(self.df_5y, self.df_1y)
        assert "overallSignal" in result
        assert "summary" in result
        assert "movingAverages" in result
        assert "indicators" in result
        assert "strategies" in result
        assert "patterns" in result

    def test_overall_signal_valid(self):
        from app.services.technicals import compute_technicals
        result = compute_technicals(self.df_5y, self.df_1y)
        assert result["overallSignal"] in ("bullish", "bearish", "neutral")

    def test_summary_has_four_tiles(self):
        from app.services.technicals import compute_technicals
        result = compute_technicals(self.df_5y, self.df_1y)
        summary = result["summary"]
        assert "trend" in summary
        assert "rsi" in summary
        assert "macd" in summary
        assert "volume" in summary

    def test_moving_averages_are_numbers(self):
        from app.services.technicals import compute_technicals
        result = compute_technicals(self.df_5y, self.df_1y)
        ma = result["movingAverages"]
        # Should be numbers (0.0 if not enough data)
        assert isinstance(ma["ma20"], (int, float))
        assert isinstance(ma["ma50"], (int, float))
        assert isinstance(ma["ma200"], (int, float))

    def test_indicators_dict_not_empty(self):
        from app.services.technicals import compute_technicals
        result = compute_technicals(self.df_5y, self.df_1y)
        assert len(result["indicators"]) > 0

    def test_each_indicator_has_required_fields(self):
        from app.services.technicals import compute_technicals
        result = compute_technicals(self.df_5y, self.df_1y)
        for key, ind in result["indicators"].items():
            assert "value" in ind, f"Missing 'value' in {key}"
            assert "signal" in ind, f"Missing 'signal' in {key}"
            assert "label" in ind, f"Missing 'label' in {key}"
            assert ind["signal"] in ("bullish", "bearish", "neutral", "caution"), \
                f"Invalid signal '{ind['signal']}' in {key}"

    def test_strategies_list(self):
        from app.services.technicals import compute_technicals
        result = compute_technicals(self.df_5y, self.df_1y)
        strats = result["strategies"]
        assert isinstance(strats, list)
        assert len(strats) <= 3
        for s in strats:
            assert "name" in s
            assert "signal" in s
            assert "description" in s

    def test_empty_dataframe_returns_safe_response(self):
        from app.services.technicals import compute_technicals
        result = compute_technicals(pd.DataFrame(), pd.DataFrame())
        assert result["overallSignal"] == "neutral"
        assert result["indicators"] == {}

    def test_chart_data_structure(self):
        from app.services.technicals import build_chart_data
        result = build_chart_data(self.df_1y, self.df_5y)
        assert "daily" in result
        assert "intraday" in result
        assert isinstance(result["daily"], list)
        assert len(result["daily"]) > 0

    def test_candle_has_required_fields(self):
        from app.services.technicals import build_chart_data
        result = build_chart_data(self.df_1y, self.df_5y)
        candle = result["daily"][0]
        assert "time" in candle
        assert "open" in candle
        assert "high" in candle
        assert "low" in candle
        assert "close" in candle
        assert "volume" in candle

    def test_candle_time_is_date_string(self):
        from app.services.technicals import build_chart_data
        result = build_chart_data(self.df_1y, self.df_5y)
        time_val = result["daily"][0]["time"]
        assert len(time_val) == 10
        assert time_val[4] == "-" and time_val[7] == "-"

    def test_no_nan_in_chart_data(self):
        from app.services.technicals import build_chart_data
        result = build_chart_data(self.df_1y, self.df_5y)
        for candle in result["daily"]:
            for key in ("open", "high", "low", "close"):
                assert candle[key] is not None
                assert not math.isnan(candle[key])


# ═══════════════════════════════════════════════════════════════════════════
# FUNDAMENTALS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFundamentals:
    def setup_method(self):
        self.info = make_info()
        self.fin  = make_financials()
        self.empty_df = pd.DataFrame()

    def test_compute_returns_dict(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        assert "overallHealth" in result
        assert "keyRatios" in result
        assert "quickSummary" in result
        assert "ratiosSnapshot" in result
        assert "financialTrends" in result
        assert "financialStatements" in result

    def test_health_score_in_range(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        score = result["overallHealth"]["score"]
        assert 0 <= score <= 100

    def test_health_label_valid(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        label = result["overallHealth"]["label"]
        assert label in ("Good", "Fair", "Weak")

    def test_key_ratios_all_present(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        ratios = result["keyRatios"]
        for key in ("pe", "roe", "roce", "debtEquity", "eps", "dividendYield"):
            assert key in ratios, f"Missing ratio: {key}"
            assert "value" in ratios[key]
            assert "rating" in ratios[key]
            assert ratios[key]["rating"] in ("good", "fair", "high", "poor")

    def test_pe_extracted_correctly(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        pe = result["keyRatios"]["pe"]["value"]
        assert abs(pe - 22.5) < 0.1, f"Expected PE ~22.5, got {pe}"

    def test_roe_converted_to_percent(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        roe = result["keyRatios"]["roe"]["value"]
        # info has returnOnEquity = 0.221 → should become 22.1%
        assert 20 <= roe <= 23, f"ROE should be ~22.1%, got {roe}"

    def test_de_normalised(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        de = result["keyRatios"]["debtEquity"]["value"]
        # info has debtToEquity = 28 → should normalise to 0.28
        assert de < 5, f"D/E should be ~0.28 after normalisation, got {de}"

    def test_financial_trends_annual(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        annual = result["financialTrends"]["annual"]
        assert isinstance(annual, list)
        assert len(annual) > 0
        for item in annual:
            assert "revenue" in item
            assert "profit" in item
            assert item["revenue"] > 0

    def test_cagr_calculated(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        cagr = result["financialTrends"]["revenueCagr5y"]
        assert isinstance(cagr, (int, float))
        assert not math.isnan(cagr)

    def test_key_insight_is_string(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            self.info, self.fin, self.empty_df,
            self.empty_df, self.empty_df, self.empty_df, self.empty_df
        )
        insight = result["quickSummary"]["keyInsight"]
        assert isinstance(insight, str)
        assert len(insight) > 10

    def test_empty_info_doesnt_crash(self):
        from app.services.fundamentals import compute_fundamentals
        result = compute_fundamentals(
            {}, pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert isinstance(result, dict)
        assert "keyRatios" in result


# ═══════════════════════════════════════════════════════════════════════════
# PERSONA TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPersonas:
    def setup_method(self):
        from app.services.personas import StockMetrics
        # A "good" stock for most personas
        self.good_metrics = StockMetrics(
            pe=18.0, pb=3.5, peg=0.9, ev_ebitda=12.0,
            roe=22.0, roa=9.5, gross_margin=45.0,
            operating_margin=18.0, net_margin=12.0,
            revenue_growth_yoy=20.0, eps_growth_yoy=18.0,
            revenue_cagr_3y=22.0, eps_cagr_3y=20.0,
            debt_equity=0.28, current_ratio=1.8,
            rsi=55.0, price_vs_sma200=8.0,
            momentum_6m=15.0, volume_ratio=1.4,
            promoter_holding=52.0, promoter_pledge=2.0,
            market_cap_cr=25_000.0, beta=1.1,
            sector="Information Technology",
            eps=104.0, dividend_yield=1.5,
        )
        # A "poor" stock
        self.poor_metrics = StockMetrics(
            pe=60.0, pb=15.0, peg=4.5,
            roe=4.0, net_margin=2.0,
            revenue_growth_yoy=-5.0, eps_growth_yoy=-10.0,
            debt_equity=3.5, current_ratio=0.8,
            rsi=78.0, price_vs_sma200=-15.0,
            momentum_6m=-20.0,
            sector="Real Estate",
        )

    def test_compute_personas_returns_list(self):
        from app.services.personas import compute_personas
        results = compute_personas(self.good_metrics)
        assert isinstance(results, list)
        assert len(results) == 9  # All 9 personas

    def test_each_persona_has_required_fields(self):
        from app.services.personas import compute_personas
        results = compute_personas(self.good_metrics)
        for p in results:
            assert "id" in p
            assert "name" in p
            assert "score" in p
            assert "verdict" in p
            assert "verdictColor" in p
            assert "summary" in p
            assert "criteria" in p

    def test_scores_in_valid_range(self):
        from app.services.personas import compute_personas
        results = compute_personas(self.good_metrics)
        for p in results:
            assert 0 <= p["score"] <= 100, f"{p['name']} score {p['score']} out of range"

    def test_verdict_color_valid(self):
        from app.services.personas import compute_personas
        results = compute_personas(self.good_metrics)
        valid_colors = {"success", "info", "warning", "danger"}
        for p in results:
            assert p["verdictColor"] in valid_colors

    def test_good_stock_scores_higher(self):
        from app.services.personas import compute_personas
        good_results  = compute_personas(self.good_metrics)
        poor_results  = compute_personas(self.poor_metrics)
        good_avg = sum(p["score"] for p in good_results) / len(good_results)
        poor_avg = sum(p["score"] for p in poor_results) / len(poor_results)
        assert good_avg > poor_avg, f"Good avg {good_avg:.1f} should > poor avg {poor_avg:.1f}"

    def test_buffett_weights_roe_heavily(self):
        from app.services.personas import WarrenBuffettPersona, StockMetrics
        persona = WarrenBuffettPersona()

        high_roe = StockMetrics(roe=25.0, debt_equity=0.3, operating_margin=20.0,
                                gross_margin=50.0, eps_growth_yoy=15.0, pe=20.0)
        low_roe  = StockMetrics(roe=5.0, debt_equity=0.3, operating_margin=20.0,
                                gross_margin=50.0, eps_growth_yoy=15.0, pe=20.0)

        assert persona.score(high_roe).score > persona.score(low_roe).score

    def test_persona_ids_are_unique(self):
        from app.services.personas import compute_personas
        results = compute_personas(self.good_metrics)
        ids = [p["id"] for p in results]
        assert len(ids) == len(set(ids)), "Duplicate persona IDs"

    def test_criteria_have_pass_field(self):
        from app.services.personas import compute_personas
        results = compute_personas(self.good_metrics)
        for p in results:
            for c in p["criteria"]:
                assert "label" in c
                assert "value" in c
                assert "pass" in c
                assert isinstance(c["pass"], bool)

    def test_none_metrics_dont_crash(self):
        from app.services.personas import compute_personas, StockMetrics
        # All None metrics — should not raise
        empty = StockMetrics()
        results = compute_personas(empty)
        assert len(results) == 9
        for p in results:
            assert 0 <= p["score"] <= 100

    def test_verdict_mapping(self):
        from app.services.personas import _verdict
        assert _verdict(90)[0] == "Strong Match"
        assert _verdict(70)[0] == "Good Match"
        assert _verdict(55)[0] == "Neutral"
        assert _verdict(35)[0] == "Weak Match"
        assert _verdict(10)[0] == "Poor Match"

    def test_extract_metrics_from_real_like_data(self):
        from app.services.personas import extract_metrics
        info = make_info()
        df   = make_ohlcv(300)
        fin  = make_financials()
        tech = {
            "summary": {"rsi": {"value": 55.0}},
        }
        metrics = extract_metrics(info, df, fin, tech)
        assert metrics.pe is not None
        assert metrics.roe is not None
        assert 18 <= metrics.roe <= 26  # ~22.1%


# ═══════════════════════════════════════════════════════════════════════════
# CACHE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCache:
    def test_key_builders(self):
        from app.services.cache import (
            analysis_key, quote_key, market_key, screener_key, search_key
        )
        assert analysis_key("RELIANCE") == "av:analysis:RELIANCE"
        assert quote_key("TCS") == "av:quote:TCS"
        assert market_key() == "av:market:overview"
        assert screener_key() == "av:screener:all"
        assert search_key("Reliance") == "av:search:reliance"

    def test_analysis_key_uppercased(self):
        from app.services.cache import analysis_key
        assert analysis_key("reliance") == analysis_key("RELIANCE")

    @pytest.mark.asyncio
    async def test_cache_noop_without_redis(self):
        """Cache operations should silently succeed without Redis."""
        from app.services.cache import cache_get, cache_set
        # Without Redis, get returns None
        result = await cache_get("test:key:noop")
        assert result is None
        # Set should not raise
        await cache_set("test:key:noop", {"data": "value"}, ttl=60)


# ═══════════════════════════════════════════════════════════════════════════
# SHAREHOLDING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestShareholding:
    def test_compute_shareholding_basic(self):
        from app.services.shareholding import compute_shareholding
        info = make_info()
        result = compute_shareholding(info, "TEST")
        assert "promoter" in result
        assert "fii" in result
        assert "dii" in result
        assert "public" in result
        assert "promoterPledge" in result
        assert "trend" in result

    def test_shareholding_sums_to_100(self):
        from app.services.shareholding import compute_shareholding
        info = make_info()
        result = compute_shareholding(info, "TEST")
        total = result["promoter"] + result["fii"] + result["dii"] + result["public"]
        assert abs(total - 100) < 2, f"Shareholding should sum to ~100, got {total}"

    def test_promoter_percent_correct(self):
        from app.services.shareholding import compute_shareholding
        info = make_info()  # heldPercentInsiders = 0.425
        result = compute_shareholding(info, "TEST")
        # 0.425 * 100 = 42.5%
        assert abs(result["promoter"] - 42.5) < 0.5

    def test_trend_has_4_quarters(self):
        from app.services.shareholding import compute_shareholding
        info = make_info()
        result = compute_shareholding(info, "TEST")
        assert len(result["trend"]) == 4

    def test_trend_quarters_have_required_fields(self):
        from app.services.shareholding import compute_shareholding
        info = make_info()
        result = compute_shareholding(info, "TEST")
        for q in result["trend"]:
            assert "quarter" in q
            assert "promoter" in q
            assert "fii" in q
            assert "dii" in q
            assert "public" in q

    def test_empty_info_doesnt_crash(self):
        from app.services.shareholding import compute_shareholding
        result = compute_shareholding({}, "TEST")
        assert isinstance(result, dict)
        assert result["promoter"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST (no network — mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyserIntegration:
    """
    Integration test for the full analysis pipeline using mocked yfinance.
    Validates the final assembled response matches the TypeScript interface.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mock(self, monkeypatch):
        """Run the full analyser pipeline with mocked data."""
        from app.services import fetcher
        from app.services.fetcher import RawStockData

        # Mock the fetch function
        mock_raw = RawStockData(
            symbol="TEST.NS",
            info=make_info(),
            history_1y=make_ohlcv(252),
            history_5y=make_ohlcv(1260),
            financials=make_financials(),
            quarterly_financials=make_financials(8),
            balance_sheet=pd.DataFrame(),
            quarterly_balance_sheet=pd.DataFrame(),
            cashflow=pd.DataFrame(),
            quarterly_cashflow=pd.DataFrame(),
        )

        async def mock_fetch(symbol: str) -> RawStockData:
            return mock_raw

        monkeypatch.setattr(fetcher, "fetch_stock_data", mock_fetch)

        # Also mock peer fetch to avoid network calls
        async def mock_peer_fetch(symbols):
            return {}
        monkeypatch.setattr(fetcher, "fetch_multiple_quotes", mock_peer_fetch)

        from app.services.analyser import run_analysis
        result = await run_analysis("TEST")

        # Validate top-level structure
        assert "ticker" in result
        assert "quote" in result
        assert "technical" in result
        assert "fundamental" in result
        assert "personas" in result
        assert "shareholding" in result
        assert "peers" in result
        assert "chartData" in result

        # Validate ticker
        assert result["ticker"] == "TEST"

        # Validate personas
        assert len(result["personas"]) == 9

        # Validate chart data
        assert len(result["chartData"]["daily"]) > 0

        # Validate no NaN in response (JSON serialisation would fail)
        import json
        json_str = json.dumps(result, default=str)  # Should not raise
        assert len(json_str) > 100


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
