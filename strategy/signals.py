# ==========================================
# 20to100 Trading Bot
# Trading Signals
# Strategy v2.0
#
# Trend + Momentum + Pullback + Confirmation
# ==========================================

import pandas as pd

from config import (
    RSI_MIN,
    RSI_MAX,
    RSI_EXIT,
    VOLUME_MULTIPLIER,
    MAX_SPREAD,
)


# ==========================================
# Helper
# ==========================================

def _valid_number(value) -> bool:
    return (
        value is not None
        and not pd.isna(value)
    )


# ==========================================
# 1. TREND
# ==========================================

def trend_condition(
    row: pd.Series,
) -> bool:

    return (
        _valid_number(row["ema_9"])
        and
        _valid_number(row["ema_21"])
        and
        _valid_number(row["ema_50"])
        and
        row["ema_9"] > row["ema_21"]
        and
        row["ema_21"] > row["ema_50"]
    )


# ==========================================
# 2. PRICE ABOVE EMA50
# ==========================================

def price_above_ema50(
    row: pd.Series,
) -> bool:

    return (
        _valid_number(row["close"])
        and
        _valid_number(row["ema_50"])
        and
        row["close"] > row["ema_50"]
    )


# ==========================================
# 3. RSI
# ==========================================

def rsi_condition(
    row: pd.Series,
) -> bool:

    return (
        _valid_number(row["rsi_14"])
        and
        RSI_MIN <= row["rsi_14"] <= RSI_MAX
    )


# ==========================================
# 4. RSI MOMENTUM
# ==========================================

def rsi_rising(
    df: pd.DataFrame,
) -> bool:

    if len(df) < 2:
        return False

    current = df.iloc[-1]
    previous = df.iloc[-2]

    if not _valid_number(
        current["rsi_14"]
    ):
        return False

    if not _valid_number(
        previous["rsi_14"]
    ):
        return False

    return (
        current["rsi_14"] >
        previous["rsi_14"]
    )


# ==========================================
# 5. PULLBACK
# ==========================================

def pullback_condition(
    df: pd.DataFrame,
) -> bool:

    if len(df) < 3:
        return False

    current = df.iloc[-1]
    previous = df.iloc[-2]

    required = [
        "low",
        "close",
        "ema_9",
        "ema_21",
    ]

    for column in required:

        if not _valid_number(
            current[column]
        ):
            return False

        if not _valid_number(
            previous[column]
        ):
            return False

    # Previous candle must have pulled
    # back towards EMA9 / EMA21.
    previous_pullback = (
        previous["low"] <=
        previous["ema_9"]
        or
        previous["low"] <=
        previous["ema_21"]
    )

    # Current candle must recover above EMA9.
    current_recovery = (
        current["close"] >
        current["ema_9"]
    )

    return (
        previous_pullback
        and
        current_recovery
    )


# ==========================================
# 6. CONFIRMATION CANDLE
# ==========================================

def confirmation_condition(
    df: pd.DataFrame,
) -> bool:

    if len(df) < 2:
        return False

    current = df.iloc[-1]

    if not _valid_number(
        current["open"]
    ):
        return False

    if not _valid_number(
        current["close"]
    ):
        return False

    # Bullish confirmation candle.
    return (
        current["close"] >
        current["open"]
    )


# ==========================================
# 7. VOLUME
# ==========================================

def volume_condition(
    row: pd.Series,
) -> bool:

    if not _valid_number(
        row["volume"]
    ):
        return False

    if not _valid_number(
        row["volume_sma_20"]
    ):
        return False

    return (
        row["volume"] >=
        VOLUME_MULTIPLIER *
        row["volume_sma_20"]
    )


# ==========================================
# 8. SPREAD
# ==========================================

def spread_condition(
    spread: float | None,
) -> bool:

    # Historical OHLCV data normally does
    # not contain bid/ask spread.
    #
    # None therefore means:
    # no spread filter available.

    if spread is None:
        return True

    return (
        spread <= MAX_SPREAD
    )


# ==========================================
# COMPLETE BUY SIGNAL
# ==========================================

def check_buy_signal(
    df: pd.DataFrame,
    spread: float | None = None,
) -> bool:

    if len(df) < 60:
        return False

    row = df.iloc[-1]

    required_columns = [
        "open",
        "close",
        "low",
        "ema_9",
        "ema_21",
        "ema_50",
        "rsi_14",
        "volume",
        "volume_sma_20",
    ]

    for column in required_columns:

        if not _valid_number(
            row[column]
        ):
            return False

    conditions = evaluate_conditions(
        df,
        spread=spread,
    )

    return all(
        conditions.values()
    )


# ==========================================
# CONDITION EVALUATION
# ==========================================

def evaluate_conditions(
    df: pd.DataFrame,
    spread: float | None = None,
) -> dict[str, bool]:

    row = df.iloc[-1]

    return {

        "trend":
            trend_condition(row),

        "price_above_ema50":
            price_above_ema50(row),

        "rsi":
            rsi_condition(row),

        "rsi_rising":
            rsi_rising(df),

        "pullback":
            pullback_condition(df),

        "confirmation":
            confirmation_condition(df),

        "volume":
            volume_condition(row),

        "spread":
            spread_condition(spread),
    }


# ==========================================
# SIGNAL DIAGNOSTICS
# ==========================================

def diagnose_signals(
    df: pd.DataFrame,
) -> dict:

    if len(df) < 60:

        return {
            "candles": len(df),
            "valid_candles": 0,
        }

    counts = {
        "trend": 0,
        "price_above_ema50": 0,
        "rsi": 0,
        "rsi_rising": 0,
        "pullback": 0,
        "confirmation": 0,
        "volume": 0,
        "all_conditions": 0,
    }

    valid_candles = 0

    for i in range(
        60,
        len(df),
    ):

        history = df.iloc[
            : i + 1
        ]

        row = history.iloc[-1]

        required_columns = [
            "open",
            "close",
            "low",
            "ema_9",
            "ema_21",
            "ema_50",
            "rsi_14",
            "volume",
            "volume_sma_20",
        ]

        valid = True

        for column in required_columns:

            if not _valid_number(
                row[column]
            ):

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

            if conditions.get(
                name,
                False,
            ):

                counts[name] += 1

        if all(
            conditions.values()
        ):

            counts[
                "all_conditions"
            ] += 1

    percentages = {}

    if valid_candles > 0:

        for name, count in counts.items():

            percentages[
                f"{name}_pct"
            ] = (
                count /
                valid_candles *
                100
            )

    return {
        "candles":
            len(df),

        "valid_candles":
            valid_candles,

        **counts,

        **percentages,
    }


# ==========================================
# DIAGNOSTIC REPORT
# ==========================================

def print_signal_diagnostics(
    diagnostics: dict,
    symbol: str,
) -> None:

    print()
    print("=" * 60)
    print(
        f"SIGNAL DIAGNOSTICS: {symbol}"
    )
    print("=" * 60)

    print(
        f"Candles: "
        f"{diagnostics.get('candles', 0)}"
    )

    print(
        f"Valid candles: "
        f"{diagnostics.get('valid_candles', 0)}"
    )

    print("-" * 60)

    conditions = [
        ("Trend", "trend"),
        (
            "Price > EMA50",
            "price_above_ema50",
        ),
        ("RSI", "rsi"),
        (
            "RSI rising",
            "rsi_rising",
        ),
        (
            "Pullback",
            "pullback",
        ),
        (
            "Confirmation",
            "confirmation",
        ),
        (
            "Volume",
            "volume",
        ),
        (
            "ALL CONDITIONS",
            "all_conditions",
        ),
    ]

    for label, key in conditions:

        count = diagnostics.get(
            key,
            0,
        )

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
        _valid_number(
            row["ema_9"]
        )
        and
        _valid_number(
            row["ema_21"]
        )
        and
        row["ema_9"] <
        row["ema_21"]
    )


def check_rsi_exit(
    row: pd.Series,
) -> bool:

    return (
        _valid_number(
            row["rsi_14"]
        )
        and
        row["rsi_14"] <
        RSI_EXIT
    )
