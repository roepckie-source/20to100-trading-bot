# ==========================================
# 20to100 Trading Bot
# Strategy V2.0
# Trend + Donchian Breakout + ATR
# ==========================================

import pandas as pd


# ==========================================
# CONFIG
# ==========================================

EMA_FAST = 50
EMA_TREND = 200

BREAKOUT_PERIOD = 20

ATR_PERIOD = 14

ATR_STOP_MULTIPLIER = 2.0
ATR_TRAILING_MULTIPLIER = 2.0

MIN_BODY_ATR = 0.30


# ==========================================
# INDICATORS
# ==========================================

def calculate_v2_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------
    # EMA
    # --------------------------------------

    df["ema_50"] = (
        df["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False,
        )
        .mean()
    )

    df["ema_200"] = (
        df["close"]
        .ewm(
            span=EMA_TREND,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------
    # True Range
    # --------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

    tr1 = (
        df["high"]
        -
        df["low"]
    )

    tr2 = (
        df["high"]
        -
        previous_close
    ).abs()

    tr3 = (
        df["low"]
        -
        previous_close
    ).abs()

    df["true_range"] = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    # --------------------------------------
    # ATR
    # --------------------------------------

    df["atr"] = (
        df["true_range"]
        .rolling(
            ATR_PERIOD
        )
        .mean()
    )

    # --------------------------------------
    # Donchian breakout
    #
    # IMPORTANT:
    # shift(1) means we only use
    # COMPLETED previous candles.
    # --------------------------------------

    df["breakout_high"] = (
        df["high"]
        .rolling(
            BREAKOUT_PERIOD
        )
        .max()
        .shift(1)
    )

    df["breakout_low"] = (
        df["low"]
        .rolling(
            BREAKOUT_PERIOD
        )
        .min()
        .shift(1)
    )

    # --------------------------------------
    # Candle body
    # --------------------------------------

    df["body"] = (
        df["close"]
        -
        df["open"]
    ).abs()

    # --------------------------------------
    # EMA slope
    # --------------------------------------

    df["ema_200_rising"] = (
        df["ema_200"]
        >
        df["ema_200"].shift(3)
    )

    # --------------------------------------
    # Trend
    # --------------------------------------

    df["trend"] = (
        (df["close"] > df["ema_200"])
        &
        (df["ema_50"] > df["ema_200"])
        &
        (df["ema_200_rising"])
    )

    # --------------------------------------
    # Breakout
    # --------------------------------------

    df["breakout"] = (
        df["close"]
        >
        df["breakout_high"]
    )

    # --------------------------------------
    # Bullish candle
    # --------------------------------------

    df["bullish_candle"] = (
        df["close"]
        >
        df["open"]
    )

    # --------------------------------------
    # Minimum candle body
    # --------------------------------------

    df["strong_body"] = (
        df["body"]
        >=
        (
            df["atr"]
            *
            MIN_BODY_ATR
        )
    )

    # --------------------------------------
    # Final signal
    # --------------------------------------

    df["v2_signal"] = (
        df["trend"]
        &
        df["breakout"]
        &
        df["bullish_candle"]
        &
        df["strong_body"]
    )

    return df


# ==========================================
# SIGNAL
# ==========================================

def check_v2_buy_signal(
    df: pd.DataFrame,
) -> bool:

    if df.empty:
        return False

    row = df.iloc[-1]

    required_columns = [
        "ema_50",
        "ema_200",
        "atr",
        "breakout_high",
        "v2_signal",
    ]

    for column in required_columns:

        if column not in df.columns:
            return False

        if pd.isna(row[column]):
            return False

    return bool(
        row["v2_signal"]
    )


# ==========================================
# ENTRY CONDITIONS
# ==========================================

def get_v2_conditions(
    row,
) -> dict:

    return {

        "trend":
            bool(
                row.get(
                    "trend",
                    False,
                )
            ),

        "ema_50_above_ema_200":
            bool(
                row.get(
                    "ema_50",
                    0,
                )
                >
                row.get(
                    "ema_200",
                    float("inf"),
                )
            ),

        "ema_200_rising":
            bool(
                row.get(
                    "ema_200_rising",
                    False,
                )
            ),

        "breakout":
            bool(
                row.get(
                    "breakout",
                    False,
                )
            ),

        "bullish_candle":
            bool(
                row.get(
                    "bullish_candle",
                    False,
                )
            ),

        "strong_body":
            bool(
                row.get(
                    "strong_body",
                    False,
                )
            ),
    }


# ==========================================
# DIAGNOSTICS
# ==========================================

def diagnose_v2_signals(
    df: pd.DataFrame,
) -> dict:

    df = calculate_v2_indicators(
        df
    )

    valid = df.dropna(
        subset=[
            "ema_50",
            "ema_200",
            "atr",
            "breakout_high",
        ]
    )

    if valid.empty:

        return {
            "candles":
                len(df),

            "valid":
                0,

            "signals":
                0,
        }

    signal_count = int(
        valid["v2_signal"]
        .sum()
    )

    return {

        "candles":
            len(df),

        "valid":
            len(valid),

        "trend":
            int(
                valid["trend"].sum()
            ),

        "breakout":
            int(
                valid["breakout"].sum()
            ),

        "strong_body":
            int(
                valid["strong_body"].sum()
            ),

        "signals":
            signal_count,
    }


# ==========================================
# FORWARD RETURN ANALYSIS
# ==========================================

def analyze_v2_forward_returns(
    df: pd.DataFrame,
    periods=None,
) -> pd.DataFrame:

    if periods is None:

        periods = [
            4,
            8,
            16,
        ]

    df = calculate_v2_indicators(
        df
    )

    rows = []

    signals = df.index[
        df["v2_signal"]
    ]

    for timestamp in signals:

        entry_pos = df.index.get_loc(
            timestamp
        )

        entry_price = float(
            df.iloc[
                entry_pos
            ]["close"]
        )

        result = {

            "timestamp":
                timestamp,

            "entry":
                entry_price,
        }

        for period in periods:

            future_pos = (
                entry_pos
                +
                period
            )

            if future_pos >= len(df):

                result[
                    f"{period}c"
                ] = None

                continue

            future_price = float(
                df.iloc[
                    future_pos
                ]["close"]
            )

            forward_return = (
                (
                    future_price
                    /
                    entry_price
                )
                -
                1.0
            )
            * 100.0

            result[
                f"{period}c"
            ] = forward_return

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )
