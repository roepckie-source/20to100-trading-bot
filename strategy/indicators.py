# ==========================================
# 20to100 Trading Bot
# Technical Indicators
# ==========================================

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

    df = df.copy()

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
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------
    # EMA
    # --------------------------------------

    df["ema_9"] = (
        df["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False,
        )
        .mean()
    )

    df["ema_21"] = (
        df["close"]
        .ewm(
            span=EMA_MEDIUM,
            adjust=False,
        )
        .mean()
    )

    df["ema_50"] = (
        df["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------
    # RSI - Wilder
    # --------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

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

    rs = avg_gain / avg_loss

    df["rsi_14"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------
    # ATR - Wilder
    # --------------------------------------

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"] -
                previous_close
            ).abs(),
            (
                df["low"] -
                previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = (
        tr
        .ewm(
            alpha=1 / ATR_PERIOD,
            adjust=False,
            min_periods=ATR_PERIOD,
        )
        .mean()
    )

    # --------------------------------------
    # Volume
    # --------------------------------------

    df["volume_sma_20"] = (
        df["volume"]
        .rolling(
            VOLUME_PERIOD
        )
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_sma_20"]
    )

    return df
