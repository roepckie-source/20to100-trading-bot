# ==========================================
# 20to100 Trading Bot
# Strategy V5.0
# Regime Filter + Breakout + Risk Management
# ==========================================

import numpy as np
import pandas as pd


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all V5 indicators on the FULL dataset.

    All indicators are causal:
    they only use current/past candles.
    """

    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]

    # ==========================================
    # EMA 100 / EMA 200
    # ==========================================

    out["ema_100"] = close.ewm(
        span=100,
        adjust=False
    ).mean()

    out["ema_200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # ==========================================
    # EMA 200 Slope
    # ==========================================

    out["ema_200_slope"] = (
        out["ema_200"].pct_change(10)
    )

    # ==========================================
    # DONCHIAN 20
    # Previous 20-candle high
    # ==========================================

    out["donchian_high_20"] = (
        high
        .rolling(20)
        .max()
        .shift(1)
    )

    # ==========================================
    # ATR 14
    # ==========================================

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    out["atr_14"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # ==========================================
    # ADX 14
    # ==========================================

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0
        ),
        index=out.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0
        ),
        index=out.index
    )

    atr = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        / atr
    )

    di_sum = (
        plus_di + minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / di_sum
    )

    out["adx_14"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # ==========================================
    # BULLISH TREND REGIME
    # ==========================================

    out["trend_regime"] = (
        (out["ema_100"] > out["ema_200"])
        &
        (out["ema_200_slope"] > 0)
        &
        (out["close"] > out["ema_200"])
    )

    return out


# ==========================================
# BUY SIGNAL
# ==========================================

def buy_signal(
    row: pd.Series,
    adx_min: float = 20.0
) -> bool:

    required = [
        "close",
        "ema_100",
        "ema_200",
        "ema_200_slope",
        "adx_14",
        "donchian_high_20",
        "atr_14",
    ]

    # Prüfen ob alle Werte vorhanden sind
    for column in required:

        if column not in row:
            return False

        if pd.isna(row[column]):
            return False

    # ==========================================
    # V5 ENTRY CONDITIONS
    # ==========================================

    bullish_trend = (
        row["ema_100"]
        >
        row["ema_200"]
    )

    positive_slope = (
        row["ema_200_slope"]
        >
        0
    )

    price_above_ema200 = (
        row["close"]
        >
        row["ema_200"]
    )

    strong_trend = (
        row["adx_14"]
        >=
        adx_min
    )

    breakout = (
        row["close"]
        >
        row["donchian_high_20"]
    )

    valid_atr = (
        row["atr_14"]
        >
        0
    )

    return bool(
        bullish_trend
        and positive_slope
        and price_above_ema200
        and strong_trend
        and breakout
        and valid_atr
    )
