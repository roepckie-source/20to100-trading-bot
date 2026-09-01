# ==========================================
# 20to100 Trading Bot
# Technical Indicators
# ==========================================

import pandas as pd


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all indicators required by Strategy v1.0.

    Required columns:
        open
        high
        low
        close
        volume
    """

    df = df.copy()

    # --------------------------------------
    # EMA
    # --------------------------------------

    df["ema_9"] = df["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema_21"] = df["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    df["ema_50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # --------------------------------------
    # RSI 14 - Wilder style
    # --------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss

    df["rsi_14"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------
    # ATR 14 - Wilder style
    # --------------------------------------

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    df["atr_14"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    # --------------------------------------
    # Volume SMA 20
    # --------------------------------------

    df["volume_sma_20"] = (
        df["volume"]
        .rolling(window=20)
        .mean()
    )

    # --------------------------------------
    # Volume ratio
    # --------------------------------------

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_sma_20"]
    )

    return df
