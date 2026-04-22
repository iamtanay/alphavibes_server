"""
app/services/technicals.py
─────────────────────────────────────────────────────────────────────────────
Technical analysis engine.

Computes 30+ indicators from OHLCV data using pandas-ta.
Returns data shaped exactly to match the TypeScript Technical type.

Key design choices:
  • pandas-ta is used for all standard indicators — it's battle-tested
    and runs in pure Python/NumPy (no native deps).
  • All calculations are on the FULL 5Y history for accuracy, but we
    return the CURRENT (last row) values to the UI.
  • The chart data returned is the 1Y daily OHLCV for rendering.
  • Safe helpers ensure NaN values become None and never crash JSON
    serialisation.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import math
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# pandas-ta import — try to import; gracefully degrade if missing
try:
    import pandas_ta as ta  # type: ignore
    _TA_AVAILABLE = True
except ImportError:
    logger.warning("pandas-ta not available — technical indicators will be limited")
    _TA_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe(value: Any, decimals: int = 2) -> float | None:
    """Round a number; return None if it's NaN, None, or infinite."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, decimals)
    except (TypeError, ValueError):
        return None


def _last(series: pd.Series | None) -> float | None:
    """Return the last non-NaN value of a Series, or None."""
    if series is None or series.empty:
        return None
    val = series.dropna()
    if val.empty:
        return None
    return _safe(val.iloc[-1])


def _rsi_signal(rsi_val: float | None) -> dict[str, str]:
    """Classify RSI value into signal + human label."""
    if rsi_val is None:
        return {"signal": "neutral", "label": "N/A"}
    if rsi_val >= 75:
        return {"signal": "bearish", "label": "Overbought"}
    if rsi_val >= 65:
        return {"signal": "caution", "label": "Near Overbought"}
    if rsi_val <= 25:
        return {"signal": "bullish", "label": "Oversold"}
    if rsi_val <= 35:
        return {"signal": "caution", "label": "Near Oversold"}
    return {"signal": "neutral", "label": "Normal"}


def _macd_signal(macd: float | None, signal: float | None, hist: float | None) -> dict[str, Any]:
    """Classify MACD into signal + label."""
    if macd is None or signal is None:
        return {"value": "N/A", "signal": "neutral"}

    if hist is not None and hist > 0 and macd > signal:
        return {"value": "Bullish Crossover", "signal": "bullish"}
    if hist is not None and hist < 0 and macd < signal:
        return {"value": "Bearish Crossover", "signal": "bearish"}
    if macd > signal:
        return {"value": "Above Signal", "signal": "bullish"}
    return {"value": "Below Signal", "signal": "bearish"}


def _trend_signal(close: pd.Series, sma20: float | None, sma50: float | None, sma200: float | None) -> dict[str, Any]:
    """Determine overall trend from MA alignment."""
    if close.empty or sma200 is None:
        return {"value": "N/A", "signal": "neutral"}

    price = close.iloc[-1]

    above_200 = price > sma200
    above_50 = sma50 is not None and price > sma50
    above_20 = sma20 is not None and price > sma20

    bullish_mas = sum([above_200, above_50, above_20])

    if bullish_mas == 3:
        return {"value": "Uptrend", "signal": "bullish"}
    if bullish_mas == 0:
        return {"value": "Downtrend", "signal": "bearish"}
    return {"value": "Mixed", "signal": "neutral"}


def _volume_signal(volume: pd.Series) -> dict[str, Any]:
    """Compare current volume to 20-day average."""
    if volume.empty or len(volume) < 20:
        return {"value": "N/A", "signal": "neutral"}

    current_vol = volume.iloc[-1]
    avg_20 = volume.tail(20).mean()

    if avg_20 == 0:
        return {"value": "N/A", "signal": "neutral"}

    ratio = current_vol / avg_20

    if ratio >= 1.5:
        return {"value": "Above Average", "signal": "bullish"}
    if ratio <= 0.6:
        return {"value": "Below Average", "signal": "bearish"}
    return {"value": "Average", "signal": "neutral"}


# ── Chart data serialiser ─────────────────────────────────────────────────

def build_chart_data(history_1y: pd.DataFrame, history_5y: pd.DataFrame) -> dict[str, list]:
    """
    Convert OHLCV DataFrames to the Candle[] format expected by
    TradingView Lightweight Charts.

    Time must be a date string 'YYYY-MM-DD' (not a timestamp).
    """
    def _df_to_candles(df: pd.DataFrame) -> list[dict]:
        if df is None or df.empty:
            return []
        candles = []
        for ts, row in df.iterrows():
            try:
                # Handle timezone-aware timestamps
                if hasattr(ts, "date"):
                    date_str = ts.date().isoformat()
                else:
                    date_str = str(ts)[:10]

                o = _safe(row.get("Open"), 2)
                h = _safe(row.get("High"), 2)
                l = _safe(row.get("Low"), 2)
                c = _safe(row.get("Close"), 2)
                v = int(row.get("Volume", 0) or 0)

                # Skip rows with missing OHLC
                if None in (o, h, l, c):
                    continue

                candles.append({
                    "time": date_str,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                })
            except Exception:
                continue
        return candles

    # "intraday" — we serve daily 1Y data for both; true intraday requires
    # a paid feed. The UI uses "intraday" for the 1D timeframe — we send
    # the last 30 trading days as a reasonable proxy.
    daily_candles = _df_to_candles(history_1y)
    intraday_candles = daily_candles[-30:] if len(daily_candles) >= 30 else daily_candles

    return {
        "daily": daily_candles,
        "intraday": intraday_candles,
    }


# ── Main computation ──────────────────────────────────────────────────────

def compute_technicals(history_5y: pd.DataFrame, history_1y: pd.DataFrame) -> dict[str, Any]:
    """
    Compute all technical indicators from OHLCV history.

    Args:
        history_5y: 5-year daily OHLCV DataFrame (for accurate indicators)
        history_1y: 1-year daily OHLCV DataFrame (for chart data)

    Returns:
        Dict matching the TypeScript Technical interface exactly.
    """
    if history_5y is None or history_5y.empty:
        return _empty_technical()

    # ── Normalise column names ────────────────────────────────────────────
    df = history_5y.copy()
    df.columns = [c.strip() for c in df.columns]

    # Ensure we have the required columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        logger.warning("Missing OHLCV columns: %s", required - set(df.columns))
        return _empty_technical()

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Moving Averages ───────────────────────────────────────────────────
    sma20  = _last(close.rolling(20).mean())
    sma50  = _last(close.rolling(50).mean())
    sma200 = _last(close.rolling(200).mean())
    ema9   = _last(close.ewm(span=9, adjust=False).mean())
    ema12  = _last(close.ewm(span=12, adjust=False).mean())
    ema26  = _last(close.ewm(span=26, adjust=False).mean())
    ema50  = _last(close.ewm(span=50, adjust=False).mean())

    # ── pandas-ta indicators ──────────────────────────────────────────────
    rsi_val  = None
    macd_val = None
    macd_sig = None
    macd_hist = None
    bb_upper  = None
    bb_lower  = None
    bb_mid    = None
    adx_val   = None
    stoch_k   = None
    stoch_d   = None
    atr_val   = None
    cci_val   = None
    willr_val = None
    obv_val   = None
    supertrend_dir = None   # +1 = bullish, -1 = bearish

    if _TA_AVAILABLE:
        try:
            # RSI
            rsi_series = ta.rsi(close, length=14)
            rsi_val = _last(rsi_series)

            # MACD
            macd_df = ta.macd(close, fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                cols = list(macd_df.columns)
                macd_val  = _last(macd_df[cols[0]])  # MACD line
                macd_hist = _last(macd_df[cols[1]])  # Histogram
                macd_sig  = _last(macd_df[cols[2]])  # Signal line

            # Bollinger Bands
            bb_df = ta.bbands(close, length=20, std=2.0)
            if bb_df is not None and not bb_df.empty:
                cols = list(bb_df.columns)
                # pandas-ta names: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
                for col in cols:
                    if "BBL" in col: bb_lower = _last(bb_df[col])
                    if "BBM" in col: bb_mid   = _last(bb_df[col])
                    if "BBU" in col: bb_upper = _last(bb_df[col])

            # ADX
            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None and not adx_df.empty:
                for col in adx_df.columns:
                    if col.startswith("ADX_"):
                        adx_val = _last(adx_df[col])
                        break

            # Stochastic
            stoch_df = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
            if stoch_df is not None and not stoch_df.empty:
                cols = list(stoch_df.columns)
                stoch_k = _last(stoch_df[cols[0]])
                stoch_d = _last(stoch_df[cols[1]]) if len(cols) > 1 else None

            # ATR
            atr_series = ta.atr(high, low, close, length=14)
            atr_val = _last(atr_series)

            # CCI
            cci_series = ta.cci(high, low, close, length=20)
            cci_val = _last(cci_series)

            # Williams %R
            willr_series = ta.willr(high, low, close, length=14)
            willr_val = _last(willr_series)

            # OBV
            obv_series = ta.obv(close, vol)
            obv_val = _last(obv_series)

            # Supertrend
            st_df = ta.supertrend(high, low, close, length=10, multiplier=3.0)
            if st_df is not None and not st_df.empty:
                for col in st_df.columns:
                    if "SUPERTd" in col:  # direction column
                        supertrend_dir = _last(st_df[col])
                        break

        except Exception as exc:
            logger.warning("pandas-ta computation error: %s", exc)
    else:
        # Fallback: compute RSI manually without pandas-ta
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("inf"))
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = _last(rsi_series)

        # Basic MACD
        ema12_series = close.ewm(span=12, adjust=False).mean()
        ema26_series = close.ewm(span=26, adjust=False).mean()
        macd_series  = ema12_series - ema26_series
        sig_series   = macd_series.ewm(span=9, adjust=False).mean()
        macd_val  = _last(macd_series)
        macd_sig  = _last(sig_series)
        macd_hist = _safe((macd_val or 0) - (macd_sig or 0))

    # ── Summary signals ───────────────────────────────────────────────────
    trend_info  = _trend_signal(close, sma20, sma50, sma200)
    rsi_info    = _rsi_signal(rsi_val)
    macd_info   = _macd_signal(macd_val, macd_sig, macd_hist)
    vol_info    = _volume_signal(vol)

    # Overall signal: simple majority vote
    signals = [
        trend_info["signal"],
        "bullish" if rsi_info["signal"] in ("neutral", "bullish") else "bearish",
        macd_info["signal"],
        vol_info["signal"],
    ]
    bull_count = signals.count("bullish")
    bear_count = signals.count("bearish")
    overall = "bullish" if bull_count > bear_count else ("bearish" if bear_count > bull_count else "neutral")

    # ── Supertrend label ──────────────────────────────────────────────────
    if supertrend_dir is not None:
        st_label = "Bullish" if supertrend_dir == 1 else "Bearish"
    else:
        # Fallback: price vs SMA200
        st_label = "Bullish" if (sma200 and close.iloc[-1] > sma200) else "Bearish"

    # ── Strategies (rule-based, no LLM) ──────────────────────────────────
    strategies = _compute_strategies(
        close=close,
        rsi_val=rsi_val,
        macd_val=macd_val,
        macd_sig=macd_sig,
        sma20=sma20,
        sma50=sma50,
        sma200=sma200,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        vol=vol,
        atr_val=atr_val,
    )

    # ── Build full indicators dict ────────────────────────────────────────
    indicators = _build_indicators_dict(
        rsi_val=rsi_val,
        macd_val=macd_val,
        macd_sig=macd_sig,
        macd_hist=macd_hist,
        sma20=sma20, sma50=sma50, sma200=sma200,
        ema9=ema9, ema12=ema12, ema26=ema26, ema50=ema50,
        bb_upper=bb_upper, bb_lower=bb_lower, bb_mid=bb_mid,
        adx_val=adx_val,
        stoch_k=stoch_k, stoch_d=stoch_d,
        atr_val=atr_val,
        cci_val=cci_val,
        willr_val=willr_val,
        obv_val=obv_val,
        supertrend_dir=supertrend_dir,
        close=close,
    )

    return {
        "overallSignal": overall,
        "summary": {
            "trend": {
                "value": trend_info["value"],
                "signal": trend_info["signal"],
                "label": trend_info["value"],
                "tooltip": "Based on price position relative to 20/50/200-day moving averages",
            },
            "rsi": {
                "value": _safe(rsi_val, 1) or "N/A",
                "signal": rsi_info["signal"],
                "label": rsi_info["label"],
                "tooltip": "RSI (14) — values above 70 are overbought, below 30 are oversold",
            },
            "macd": {
                "value": macd_info["value"],
                "signal": macd_info["signal"],
                "label": macd_info["value"],
                "tooltip": "MACD (12,26,9) — crossover above signal line is bullish",
            },
            "volume": {
                "value": vol_info["value"],
                "signal": vol_info["signal"],
                "label": vol_info["value"],
                "tooltip": "Current volume vs 20-day average",
            },
        },
        "movingAverages": {
            "ma20":  sma20  or 0.0,
            "ma50":  sma50  or 0.0,
            "ma200": sma200 or 0.0,
        },
        "indicators": indicators,
        "strategies": strategies,
        "patterns": [],  # Pattern detection is a Phase 2 feature
    }


def _compute_strategies(
    close: pd.Series,
    rsi_val: float | None,
    macd_val: float | None,
    macd_sig: float | None,
    sma20: float | None,
    sma50: float | None,
    sma200: float | None,
    bb_upper: float | None,
    bb_lower: float | None,
    vol: pd.Series,
    atr_val: float | None,
) -> list[dict]:
    """
    Generate 3 strategy signals from current indicator state.
    Each strategy has a name, signal, and plain-English description.
    """
    strategies = []
    price = close.iloc[-1] if not close.empty else 0

    # ── Strategy 1: Moving Average Trend ─────────────────────────────────
    if sma20 and sma50 and sma200:
        if price > sma50 > sma200:
            strategies.append({
                "name": "Moving Average Trend",
                "signal": "bullish",
                "description": "Price above both 50 and 200-day MAs in bullish alignment.",
                "thumbnail": "ma_trend",
            })
        elif price < sma50 < sma200:
            strategies.append({
                "name": "Moving Average Trend",
                "signal": "bearish",
                "description": "Price below both 50 and 200-day MAs — bearish alignment.",
                "thumbnail": "ma_trend",
            })
        else:
            strategies.append({
                "name": "Moving Average Trend",
                "signal": "neutral",
                "description": "Mixed MA alignment — market lacks clear direction.",
                "thumbnail": "ma_trend",
            })
    elif sma200:
        strategies.append({
            "name": "Moving Average Trend",
            "signal": "bullish" if price > sma200 else "bearish",
            "description": f"Price {'above' if price > sma200 else 'below'} 200-day MA.",
            "thumbnail": "ma_trend",
        })

    # ── Strategy 2: RSI Insight ───────────────────────────────────────────
    if rsi_val is not None:
        if rsi_val < 35:
            strategies.append({
                "name": "RSI Insight",
                "signal": "bullish",
                "description": f"RSI at {rsi_val:.1f} — oversold zone. Potential mean-reversion opportunity.",
                "thumbnail": "rsi",
            })
        elif rsi_val > 70:
            strategies.append({
                "name": "RSI Insight",
                "signal": "caution",
                "description": f"RSI at {rsi_val:.1f} — overbought zone. Watch for potential pullback.",
                "thumbnail": "rsi",
            })
        else:
            strategies.append({
                "name": "RSI Insight",
                "signal": "neutral",
                "description": f"RSI at {rsi_val:.1f} — neutral momentum zone. No extreme reading.",
                "thumbnail": "rsi",
            })

    # ── Strategy 3: Breakout Setup ────────────────────────────────────────
    if bb_upper and bb_lower and vol is not None and len(vol) >= 20:
        bb_width = bb_upper - bb_lower
        price_in_bb = (price - bb_lower) / bb_width if bb_width > 0 else 0.5
        avg_vol = vol.tail(20).mean()
        vol_ratio = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0

        if price_in_bb > 0.85 and vol_ratio > 1.3:
            strategies.append({
                "name": "Breakout Setup",
                "signal": "bullish",
                "description": "Price near upper Bollinger Band with above-average volume — breakout attempt.",
                "thumbnail": "breakout",
            })
        elif price_in_bb < 0.15 and vol_ratio > 1.2:
            strategies.append({
                "name": "Breakdown Setup",
                "signal": "bearish",
                "description": "Price near lower Bollinger Band with elevated volume — possible breakdown.",
                "thumbnail": "breakout",
            })
        elif bb_width / price < 0.05:  # Tight bands = squeeze
            strategies.append({
                "name": "Volatility Squeeze",
                "signal": "neutral",
                "description": "Bollinger Bands are tightening — a significant move may be imminent.",
                "thumbnail": "squeeze",
            })
        else:
            strategies.append({
                "name": "Breakout Setup",
                "signal": "neutral",
                "description": "No clear breakout setup. Price within normal Bollinger Band range.",
                "thumbnail": "breakout",
            })

    # Return up to 3 strategies
    return strategies[:3]


def _build_indicators_dict(
    rsi_val, macd_val, macd_sig, macd_hist,
    sma20, sma50, sma200, ema9, ema12, ema26, ema50,
    bb_upper, bb_lower, bb_mid,
    adx_val, stoch_k, stoch_d, atr_val, cci_val, willr_val, obv_val,
    supertrend_dir, close: pd.Series,
) -> dict[str, Any]:
    """
    Build the full indicators dict consumed by the "All Indicators" table
    in the Technicals tab.
    """
    price = close.iloc[-1] if not close.empty else 0

    def _ind(label: str, value: Any, signal: str, tooltip: str) -> dict:
        return {
            "value": value if value is not None else "N/A",
            "signal": signal,
            "label": label,
            "tooltip": tooltip,
        }

    def _num_signal(val, bull_above=None, bear_below=None):
        """Generic numeric signal classifier."""
        if val is None:
            return "neutral"
        if bull_above is not None and val > bull_above:
            return "bullish"
        if bear_below is not None and val < bear_below:
            return "bearish"
        return "neutral"

    indicators: dict[str, Any] = {}

    # ── Moving Averages ───────────────────────────────────────────────────
    if sma20:
        indicators["sma20"] = _ind(
            "SMA 20", _safe(sma20),
            "bullish" if price > sma20 else "bearish",
            "Simple 20-day moving average. Price above = short-term bullish.",
        )
    if sma50:
        indicators["sma50"] = _ind(
            "SMA 50", _safe(sma50),
            "bullish" if price > sma50 else "bearish",
            "Simple 50-day moving average. Key medium-term trend indicator.",
        )
    if sma200:
        indicators["sma200"] = _ind(
            "SMA 200", _safe(sma200),
            "bullish" if price > sma200 else "bearish",
            "200-day moving average. The gold standard long-term trend line.",
        )
    if ema12:
        indicators["ema12"] = _ind(
            "EMA 12", _safe(ema12),
            "bullish" if price > ema12 else "bearish",
            "12-day exponential MA. Responds faster to recent price changes.",
        )
    if ema26:
        indicators["ema26"] = _ind(
            "EMA 26", _safe(ema26),
            "bullish" if ema12 and ema12 > ema26 else "bearish",
            "26-day exponential MA. Used in MACD calculation.",
        )

    # ── Momentum ──────────────────────────────────────────────────────────
    if rsi_val is not None:
        rsi_info = _rsi_signal(rsi_val)
        indicators["rsi"] = _ind(
            "RSI (14)", _safe(rsi_val, 1),
            rsi_info["signal"],
            "Relative Strength Index. Above 70 = overbought, below 30 = oversold.",
        )

    if macd_val is not None:
        macd_info = _macd_signal(macd_val, macd_sig, macd_hist)
        indicators["macd"] = _ind(
            "MACD (12,26,9)", _safe(macd_val, 4),
            macd_info["signal"],
            "Moving Average Convergence Divergence. Crossover above signal = buy signal.",
        )

    if stoch_k is not None:
        sk_signal = "bearish" if stoch_k > 80 else ("bullish" if stoch_k < 20 else "neutral")
        indicators["stoch_k"] = _ind(
            "Stochastic %K", _safe(stoch_k, 1),
            sk_signal,
            "Stochastic oscillator. Above 80 = overbought, below 20 = oversold.",
        )

    if cci_val is not None:
        cci_signal = "bearish" if cci_val > 100 else ("bullish" if cci_val < -100 else "neutral")
        indicators["cci"] = _ind(
            "CCI (20)", _safe(cci_val, 1),
            cci_signal,
            "Commodity Channel Index. Above +100 = overbought, below -100 = oversold.",
        )

    if willr_val is not None:
        wr_signal = "bullish" if willr_val < -80 else ("bearish" if willr_val > -20 else "neutral")
        indicators["williams_r"] = _ind(
            "Williams %R", _safe(willr_val, 1),
            wr_signal,
            "Williams %R. Below -80 = oversold (bullish), above -20 = overbought.",
        )

    # ── Trend Strength ────────────────────────────────────────────────────
    if adx_val is not None:
        adx_signal = "bullish" if adx_val > 25 else "neutral"
        indicators["adx"] = _ind(
            "ADX (14)", _safe(adx_val, 1),
            adx_signal,
            "Average Directional Index. Above 25 = strong trend, below 20 = weak/choppy.",
        )

    # ── Volatility ────────────────────────────────────────────────────────
    if bb_upper and bb_lower:
        bb_width_pct = _safe(((bb_upper - bb_lower) / price) * 100, 2) if price > 0 else None
        indicators["bb_width"] = _ind(
            "BB Width %", bb_width_pct,
            "neutral",
            "Bollinger Band width as % of price. Low = squeeze, potential breakout.",
        )

    if atr_val is not None:
        atr_pct = _safe((atr_val / price) * 100, 2) if price > 0 else None
        indicators["atr"] = _ind(
            "ATR (14) %", atr_pct,
            "neutral",
            "Average True Range as % of price. Higher = more volatile stock.",
        )

    # ── Supertrend ────────────────────────────────────────────────────────
    if supertrend_dir is not None:
        indicators["supertrend"] = _ind(
            "Supertrend", "Bullish" if supertrend_dir == 1 else "Bearish",
            "bullish" if supertrend_dir == 1 else "bearish",
            "Supertrend (10, 3.0 ATR). Direction flip signals trend reversal.",
        )

    # ── Volume ────────────────────────────────────────────────────────────
    if obv_val is not None:
        indicators["obv"] = _ind(
            "OBV", _safe(obv_val / 1e6, 2),  # in millions
            "neutral",
            "On-Balance Volume (millions). Rising OBV confirms price uptrend.",
        )

    return indicators


def _empty_technical() -> dict[str, Any]:
    """Return a safe empty Technical response when data is unavailable."""
    neutral = {"value": "N/A", "signal": "neutral", "label": "N/A", "tooltip": "Data unavailable"}
    return {
        "overallSignal": "neutral",
        "summary": {
            "trend":  neutral,
            "rsi":    neutral,
            "macd":   neutral,
            "volume": neutral,
        },
        "movingAverages": {"ma20": 0, "ma50": 0, "ma200": 0},
        "indicators": {},
        "strategies": [],
        "patterns": [],
    }
