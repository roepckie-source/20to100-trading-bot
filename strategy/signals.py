# ==========================================
# 20to100 Trading Bot
# Trading Signals + Signal Diagnostics
# Strategy v1.1
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


# ==========================================
# Individual Conditions
# ==========================================

def trend_condition(row: pd.Series) -> bool:
    """
    Bullish EMA alignment.
    """

    return (
        row["ema_9"] > row["ema_21"]
        and
        row["ema_21"] > row["ema_50"]
    )


def rsi_condition(row: pd.Series) -> bool:
    """
    RSI must be inside the permitted entry range.
    """

    return (
        RSI_MIN <=
        row["rsi_14"] <=
        RSI_MAX
    )


def volume_condition(row: pd.Series) -> bool:
    """
    Current volume must exceed the
    configured volume multiplier.
    """

    if pd.isna(row["volume_sma_20"]):
        return False

    return (
        row["volume"] >=
        VOLUME_MULTIPLIER *
        row["volume_sma_20"]
    )


def price_condition(row: pd.Series) -> bool:
    """
    Price must be above the fast EMA.
    """

    return (
        row["close"] >
        row["ema_9"]
    )


def breakout_condition(
    df: pd.DataFrame,
) -> bool:
    """
    Current close must break above
    the highest close of the previous
    BREAKOUT_CANDLES candles.
    """

    if len(df) < BREAKOUT_CANDLES + 1:
        return False

    previous_closes = (
        df["close"]
        .iloc[
            -(BREAKOUT_CANDLES + 1):-1
        ]
    )

    if len(previous_closes) != BREAKOUT_CANDLES:
        return False

    breakout_level = (
        previous_closes.max()
    )

    current_close = float(
        df["close"].iloc[-1]
    )

    return (
        current_close >
        breakout_level
    )


def spread_condition(
    spread: float | None,
) -> bool:
    """
    Spread filter.

    In historical OHLCV backtests the spread
    is normally unavailable. Therefore None
    means that no spread rejection is applied.
    """

    if spread is None:
        return True

    return spread <= MAX_SPREAD


# ==========================================
# Complete BUY Signal
# ==========================================

def check_buy_signal(
    df: pd.DataFrame,
    spread: float | None = None,
) -> bool:
    """
    Check whether all entry conditions
    are satisfied.
    """

    if len(df) < 60:
        return False

    row = df.iloc[-1]

    required_columns = [
        "close",
        "ema_9",
        "ema_21",
        "ema_50",
        "rsi_14",
        "volume",
        "volume_sma_20",
    ]

    for column in required_columns:

        if pd.isna(row[column]):
            return False

    conditions = evaluate_conditions(
        df,
        spread=spread,
    )

    return all(
        conditions.values()
    )


# ==========================================
# Evaluate Individual Conditions
# ==========================================

def evaluate_conditions(
    df: pd.DataFrame,
    spread: float | None = None,
) -> dict[str, bool]:
    """
    Return every entry condition separately.

    This is used by both the strategy and
    the diagnostic system.
    """

    row = df.iloc[-1]

    return {
        "trend": trend_condition(row),

        "rsi": rsi_condition(row),

        "volume": volume_condition(row),

        "price_above_ema9":
            price_condition(row),

        "breakout":
            breakout_condition(df),

        "spread":
            spread_condition(spread),
    }


# ==========================================
# Signal Diagnostics
# ==========================================

def diagnose_signals(
    df: pd.DataFrame,
) -> dict:
    """
    Count how often each entry condition
    is satisfied.

    The function also counts how many candles
    pass progressively more conditions.
    """

    if len(df) < 60:

        return {
            "candles": len(df),
            "trend": 0,
            "rsi": 0,
            "volume": 0,
            "price_above_ema9": 0,
            "breakout": 0,
            "all_conditions": 0,
        }

    counts = {
        "trend": 0,
        "rsi": 0,
        "volume": 0,
        "price_above_ema9": 0,
        "breakout": 0,
        "all_conditions": 0,
    }

    valid_candles = 0

    for i in range(60, len(df)):

        history = df.iloc[: i + 1]

        row = history.iloc[-1]

        required_columns = [
            "close",
            "ema_9",
            "ema_21",
            "ema_50",
            "rsi_14",
            "volume",
            "volume_sma_20",
        ]

        valid = True

        for column in required_columns:

            if pd.isna(row[column]):
                valid = False
                break

        if not valid:
            continue

        valid_candles += 1

        conditions = evaluate_conditions(
            history
        )

        for name in counts:

            if name == "all_conditions":
                continue

            if conditions[name]:

                counts[name] += 1

        if all(
            conditions.values()
        ):

            counts["all_conditions"] += 1

    # --------------------------------------
    # Percentages
    # --------------------------------------

    percentages = {}

    if valid_candles > 0:

        for name, count in counts.items():

            percentages[
                f"{name}_pct"
            ] = (
                count /
                valid_candles
            ) * 100

    return {
        "candles": len(df),
        "valid_candles": valid_candles,
        **counts,
        **percentages,
    }


# ==========================================
# Diagnostic Report
# ==========================================

def print_signal_diagnostics(
    diagnostics: dict,
    symbol: str,
) -> None:
    """
    Print a readable diagnostic report.
    """

    print()
    print("=" * 60)
    print(
        f"SIGNAL DIAGNOSTICS: {symbol}"
    )
    print("=" * 60)

    print(
        f"Candles: "
        f"{diagnostics['candles']}"
    )

    print(
        f"Valid candles: "
        f"{diagnostics['valid_candles']}"
    )

    print("-" * 60)

    conditions = [
        ("Trend", "trend"),
        ("RSI", "rsi"),
        ("Volume", "volume"),
        ("Price > EMA9", "price_above_ema9"),
        ("Breakout", "breakout"),
        ("ALL CONDITIONS", "all_conditions"),
    ]

    for label, key in conditions:

        count = diagnostics[key]

        percentage = diagnostics.get(
            f"{key}_pct",
            0.0,
        )

        print(
            f"{label:<20}"
            f"{count:>6}  "
            f"({percentage:>6.2f}%)"
        )

    print("=" * 60)
    print()


# ==========================================
# EXIT CONDITIONS
# ==========================================

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
