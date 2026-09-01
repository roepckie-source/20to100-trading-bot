# ==========================================
# 20to100 Trading Bot
# Trading Signals - Strategy v1.0
# ==========================================

import pandas as pd


def check_buy_signal(df: pd.DataFrame) -> bool:
    """
    Check whether all Strategy v1.0 BUY conditions are fulfilled.

    The last row must represent a COMPLETED candle.
    """

    if len(df) < 60:
        return False

    row = df.iloc[-1]

    # --------------------------------------
    # Required data
    # --------------------------------------

    required_columns = [
        "close",
        "ema_9",
        "ema_21",
        "ema_50",
        "rsi_14",
        "volume",
        "volume_sma_20",
        "volume_ratio",
    ]

    for column in required_columns:
        if pd.isna(row[column]):
            return False

    # --------------------------------------
    # 1. Trend
    # EMA9 > EMA21 > EMA50
    # --------------------------------------

    trend_ok = (
        row["ema_9"] > row["ema_21"]
        and row["ema_21"] > row["ema_50"]
    )

    if not trend_ok:
        return False

    # --------------------------------------
    # 2. RSI
    # 55 <= RSI <= 70
    # --------------------------------------

    rsi_ok = (
        55 <= row["rsi_14"] <= 70
    )

    if not rsi_ok:
        return False

    # --------------------------------------
    # 3. Volume
    # Current volume >= 1.5 × SMA20
    # --------------------------------------

    volume_ok = (
        row["volume"] >=
        1.5 * row["volume_sma_20"]
    )

    if not volume_ok:
        return False

    # --------------------------------------
    # 4. Price above EMA9
    # --------------------------------------

    price_above_ema = (
        row["close"] > row["ema_9"]
    )

    if not price_above_ema:
        return False

    # --------------------------------------
    # 5. Six-candle breakout
    #
    # Current close must be greater than
    # the highest close of the previous
    # six completed candles.
    # --------------------------------------

    previous_6_closes = df["close"].iloc[-7:-1]

    if len(previous_6_closes) != 6:
        return False

    breakout_level = previous_6_closes.max()

    breakout_ok = (
        row["close"] > breakout_level
    )

    if not breakout_ok:
        return False

    # --------------------------------------
    # 6. Spread
    #
    # Spread is supplied by the live/paper
    # execution layer.
    #
    # Historical OHLCV data does not contain
    # bid/ask, so backtesting will handle this
    # separately.
    # --------------------------------------

    if "spread" in df.columns:
        spread = row["spread"]

        if pd.isna(spread):
            return False

        if spread > 0.001:
            return False

    # --------------------------------------
    # ALL conditions fulfilled
    # --------------------------------------

    return True


def check_ema_exit(df: pd.DataFrame) -> bool:
    """
    Exit when EMA9 falls below EMA21.
    """

    if len(df) < 2:
        return False

    current = df.iloc[-1]

    return (
        current["ema_9"] <
        current["ema_21"]
    )


def check_rsi_exit(df: pd.DataFrame) -> bool:
    """
    Exit when RSI falls below 45.
    """

    if len(df) < 1:
        return False

    current = df.iloc[-1]

    return current["rsi_14"] < 45


def check_time_exit(
    entry_timestamp,
    current_timestamp,
    max_minutes=60
) -> bool:
    """
    Exit after the maximum allowed trade duration.
    """

    duration = (
        current_timestamp -
        entry_timestamp
    ).total_seconds() / 60

    return duration >= max_minutes
