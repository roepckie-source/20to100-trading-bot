# ==========================================
# 20to100 Trading Bot
# Technical Indicators
#
# V6 / V7-S0 compatible
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    EMA_FAST,
    EMA_MEDIUM,
    EMA_SLOW,
    RSI_PERIOD,
    ATR_PERIOD,
    VOLUME_PERIOD,
)


def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    # ==========================================
    # REQUIRED INPUT
    # ==========================================

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in out.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ==========================================
    # BASIC EMA
    # ==========================================

    out["ema_9"] = (
        out["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False,
        )
        .mean()
    )

    out["ema_21"] = (
        out["close"]
        .ewm(
            span=EMA_MEDIUM,
            adjust=False,
        )
        .mean()
    )

    out["ema_50"] = (
        out["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
        )
        .mean()
    )

    # ==========================================
    # V6 EMA 100
    # ==========================================

    out["ema_100"] = (
        out["close"]
        .ewm(
            span=100,
            adjust=False,
        )
        .mean()
    )

    # ==========================================
    # V6 EMA 200
    # ==========================================

    out["ema_200"] = (
        out["close"]
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    # ==========================================
    # V6 EMA 200 SLOPE
    #
    # 10-Candle percentage change
    # ==========================================

    out["ema_200_slope"] = (
        out["ema_200"]
        .pct_change(10)
    )

    # ==========================================
    # V6 ADAPTIVE EMA SLOPE REFERENCE
    #
    # 50-Candle median
    #
    # shift(1) prevents the current slope
    # from influencing its own reference.
    # ==========================================

    out["ema_200_slope_reference"] = (
        out["ema_200_slope"]
        .rolling(50)
        .median()
        .shift(1)
    )

    # ==========================================
    # V6 DONCHIAN 20
    #
    # Previous 20-Candle High
    #
    # shift(1) prevents current candle
    # from being part of the breakout level.
    # ==========================================

    out["donchian_high_20"] = (
        out["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    # ==========================================
    # TRUE RANGE
    # ==========================================

    previous_close = (
        out["close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            out["high"] - out["low"],

            (
                out["high"]
                - previous_close
            ).abs(),

            (
                out["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # ==========================================
    # ATR 14
    # ==========================================

    out["atr_14"] = (
        tr
        .ewm(
            alpha=1 / 14,
            adjust=False,
        )
        .mean()
    )

    # ==========================================
    # ATR 50 REFERENCE
    #
    # Current ATR is NOT included in the
    # reference used for the current candle.
    # ==========================================

    out["atr_14_ma50"] = (
        out["atr_14"]
        .rolling(50)
        .mean()
        .shift(1)
    )

    # ==========================================
    # ADX 14
    # ==========================================

    up_move = (
        out["high"]
        .diff()
    )

    down_move = (
        -out["low"]
        .diff()
    )

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            &
            (up_move > 0),
            up_move,
            0.0,
        ),
        index=out.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            &
            (down_move > 0),
            down_move,
            0.0,
        ),
        index=out.index,
    )

    adx_atr = (
        tr
        .ewm(
            alpha=1 / 14,
            adjust=False,
        )
        .mean()
    )

    plus_di = (
        100
        *
        plus_dm
        .ewm(
            alpha=1 / 14,
            adjust=False,
        )
        .mean()
        /
        adx_atr
    )

    minus_di = (
        100
        *
        minus_dm
        .ewm(
            alpha=1 / 14,
            adjust=False,
        )
        .mean()
        /
        adx_atr
    )

    di_sum = (
        plus_di
        + minus_di
    ).replace(
        0,
        np.nan,
    )

    dx = (
        100
        *
        (
            plus_di
            - minus_di
        ).abs()
        /
        di_sum
    )

    out["adx_14"] = (
        dx
        .ewm(
            alpha=1 / 14,
            adjust=False,
        )
        .mean()
    )

    # ==========================================
    # RSI - Wilder
    # ==========================================

    delta = (
        out["close"]
        .diff()
    )

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / RSI_PERIOD,
            adjust=False,
            min_periods=RSI_PERIOD,
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / RSI_PERIOD,
            adjust=False,
            min_periods=RSI_PERIOD,
        )
        .mean()
    )

    rs = (
        avg_gain
        / avg_loss
    )

    out["rsi_14"] = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    # ==========================================
    # V6 BULLISH TREND REGIME
    # ==========================================

    out["trend_regime"] = (
        (
            out["ema_100"]
            >
            out["ema_200"]
        )
        &
        (
            out["ema_200_slope"]
            > 0
        )
        &
        (
            out["close"]
            >
            out["ema_200"]
        )
    )

    # ==========================================
    # VOLUME
    # ==========================================

    out["volume_sma_20"] = (
        out["volume"]
        .rolling(
            VOLUME_PERIOD
        )
        .mean()
    )

    out["volume_ratio"] = (
        out["volume"]
        /
        out["volume_sma_20"]
    )

    # ==========================================
    # FINAL COLUMN CHECK
    #
    # These columns are required by V6-C
    # and V7-S0.
    # ==========================================

    v6_required = [
        "ema_100",
        "ema_200",
        "ema_200_slope",
        "ema_200_slope_reference",
        "donchian_high_20",
        "atr_14",
        "atr_14_ma50",
        "adx_14",
    ]

    missing_v6 = [
        column
        for column in v6_required
        if column not in out.columns
    ]

    if missing_v6:
        raise ValueError(
            "V6 indicator calculation failed. "
            f"Missing columns: {missing_v6}"
        )

    return out
