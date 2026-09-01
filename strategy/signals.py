# ==========================================
# 20to100 Trading Bot
# Trading Signals - Strategy v1.1
# ==========================================

import pandas as pd

from config import (
    RSI_MIN,
    RSI_MAX,
    RSI_EXIT,
    VOLUME_MULTIPLIER,
    BREAKOUT_CANDLES,
    MAX_SPREAD,
)


def check_buy_signal(
    df: pd.DataFrame,
    spread: float | None = None,
) -> bool:

    if len(df) < 60:
        return False

    row = df.iloc[-1]

    required = [
        "close",
        "ema_9",
        "ema_21",
        "ema_50",
        "rsi_14",
        "volume",
        "volume_sma_20",
    ]

    for column in required:
        if pd.isna(row[column]):
            return False

    # --------------------------------------
    # 1. Trend
    # --------------------------------------

    if not (
        row["ema_9"] > row["ema_21"]
        and
        row["ema_21"] > row["ema_50"]
    ):
        return False

    # --------------------------------------
    # 2. RSI
    # --------------------------------------

    if not (
        RSI_MIN <=
        row["rsi_14"] <=
        RSI_MAX
    ):
        return False

    # --------------------------------------
    # 3. Volume
    # --------------------------------------

    if not (
        row["volume"] >=
        VOLUME_MULTIPLIER *
        row["volume_sma_20"]
    ):
        return False

    # --------------------------------------
    # 4. Price above EMA9
    # --------------------------------------

    if row["close"] <= row["ema_9"]:
        return False

    # --------------------------------------
    # 5. Breakout
    # --------------------------------------

    previous_closes = df[
        "close"
    ].iloc[
        -(BREAKOUT_CANDLES + 1):-1
    ]

    if len(previous_closes) != BREAKOUT_CANDLES:
        return False

    breakout_level = previous_closes.max()

    if row["close"] <= breakout_level:
        return False

    # --------------------------------------
    # 6. Spread
    # --------------------------------------

    if spread is not None:

        if spread > MAX_SPREAD:
            return False

    return True


def check_ema_exit(
    row: pd.Series,
) -> bool:

    return (
        row["ema_9"] <
        row["ema_21"]
    )


def check_rsi_exit(
    row: pd.Series,
) -> bool:

    return (
        row["rsi_14"] <
        RSI_EXIT
    )
