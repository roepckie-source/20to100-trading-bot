"""
Bitrue Strategy V3

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO REAL ORDERS

V3:
- EMA 9 / 21 / 55
- Momentum filter
- Volume filter
- EMA spread filter
- ATR regime filter
- 1H higher-timeframe trend confirmation
- ATR based SL / TP
"""

from dataclasses import dataclass

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 55

MOMENTUM_LOOKBACK = 3
MIN_MOMENTUM_PCT = 0.15

ATR_PERIOD = 14

MIN_ATR_PCT = 0.075
MAX_ATR_PCT = 0.20

VOLUME_LOOKBACK = 20
MIN_VOLUME_RATIO = 1.20

MIN_EMA_SPREAD_PCT = 0.15

# Higher timeframe
HTF_FAST_EMA = 20
HTF_SLOW_EMA = 50

# ATR-based exits
ATR_STOP_MULTIPLIER = 1.20
ATR_TARGET_MULTIPLIER = 2.00

MIN_POSITION_USD = 5.00
MAX_POSITION_USD = 25.00
POSITION_FRACTION = 0.01

ALLOW_LONG = True
ALLOW_SHORT = True

PAPER_ONLY = True
LIVE_TRADING = False


# ============================================================
# SIGNAL
# ============================================================

@dataclass
class StrategySignal:
    signal: bool
    side: str | None
    price: float

    momentum_pct: float
    atr_pct: float
    volume_ratio: float
    ema_spread_pct: float

    htf_bullish: bool
    htf_bearish: bool

    atr_value: float

    position_size: float

    stop_loss: float | None
    take_profit: float | None

    reason: str


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(bankroll: float) -> float:

    raw_size = bankroll * POSITION_FRACTION

    return max(
        MIN_POSITION_USD,
        min(
            MAX_POSITION_USD,
            raw_size,
        ),
    )


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(
    data: pd.DataFrame,
) -> pd.DataFrame:

    df = data.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # --------------------------------------------------------
    # 5M EMA
    # --------------------------------------------------------

    df["ema_fast"] = (
        close
        .ewm(
            span=EMA_FAST,
            adjust=False,
        )
        .mean()
    )

    df["ema_medium"] = (
        close
        .ewm(
            span=EMA_MEDIUM,
            adjust=False,
        )
        .mean()
    )

    df["ema_slow"] = (
        close
        .ewm(
            span=EMA_SLOW,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["momentum_pct"] = (
        close.pct_change(
            MOMENTUM_LOOKBACK
        )
        * 100.0
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = (
        true_range
        .rolling(
            ATR_PERIOD
        )
        .mean()
    )

    df["atr_pct"] = (
        df["atr"]
        / close
        * 100.0
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    average_volume = (
        volume
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()
    )

    df["volume_ratio"] = (
        volume
        / average_volume
    )

    # --------------------------------------------------------
    # EMA SPREAD
    # --------------------------------------------------------

    df["ema_spread_pct"] = (
        (
            df["ema_fast"]
            - df["ema_medium"]
        ).abs()
        / close
        * 100.0
    )

    # ========================================================
    # 1H HIGHER TIMEFRAME
    # ========================================================

    htf = (
        df[
            [
                "timestamp",
                "close",
            ]
        ]
        .set_index("timestamp")
        .resample("1h")
        .agg(
            {
                "close": "last",
            }
        )
    )

    htf["ema_fast"] = (
        htf["close"]
        .ewm(
            span=HTF_FAST_EMA,
            adjust=False,
        )
        .mean()
    )

    htf["ema_slow"] = (
        htf["close"]
        .ewm(
            span=HTF_SLOW_EMA,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Shift by one full 1H candle.
    #
    # The strategy may only use information that was already
    # known before the currently evaluated 5m candle.
    # --------------------------------------------------------

    htf["htf_bullish"] = (
        htf["ema_fast"]
        > htf["ema_slow"]
    )

    htf["htf_bearish"] = (
        htf["ema_fast"]
        < htf["ema_slow"]
    )

    htf_features = (
        htf[
            [
                "htf_bullish",
                "htf_bearish",
            ]
        ]
        .shift(1)
    )

    df = pd.merge_asof(
        df.sort_values("timestamp"),
        htf_features.sort_index(),
        left_on="timestamp",
        right_index=True,
        direction="backward",
    )

    return df


# ============================================================
# SIGNAL EVALUATION
# ============================================================

def evaluate_precomputed(
    row,
    bankroll: float,
) -> StrategySignal:

    price = float(row.close)

    values = [
        row.ema_fast,
        row.ema_medium,
        row.ema_slow,
        row.momentum_pct,
        row.atr,
        row.atr_pct,
        row.volume_ratio,
        row.ema_spread_pct,
        row.htf_bullish,
        row.htf_bearish,
    ]

    if any(
        pd.isna(value)
        for value in values
    ):

        return StrategySignal(
            False,
            None,
            price,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            False,
            0.0,
            0.0,
            None,
            None,
            "insufficient_data",
        )

    momentum = float(
        row.momentum_pct
    )

    atr_value = float(
        row.atr
    )

    atr_pct = float(
        row.atr_pct
    )

    volume_ratio = float(
        row.volume_ratio
    )

    ema_spread_pct = float(
        row.ema_spread_pct
    )

    htf_bullish = bool(
        row.htf_bullish
    )

    htf_bearish = bool(
        row.htf_bearish
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    if abs(momentum) < MIN_MOMENTUM_PCT:

        return StrategySignal(
            False,
            None,
            price,
            momentum,
            atr_pct,
            volume_ratio,
            ema_spread_pct,
            htf_bullish,
            htf_bearish,
            atr_value,
            0.0,
            None,
            None,
            "momentum_filter",
        )

    # --------------------------------------------------------
    # ATR REGIME
    # --------------------------------------------------------

    if (
        atr_pct < MIN_ATR_PCT
        or atr_pct > MAX_ATR_PCT
    ):

        return StrategySignal(
            False,
            None,
            price,
            momentum,
            atr_pct,
            volume_ratio,
            ema_spread_pct,
            htf_bullish,
            htf_bearish,
            atr_value,
            0.0,
            None,
            None,
            "atr_regime_filter",
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if volume_ratio < MIN_VOLUME_RATIO:

        return StrategySignal(
            False,
            None,
            price,
            momentum,
            atr_pct,
            volume_ratio,
            ema_spread_pct,
            htf_bullish,
            htf_bearish,
            atr_value,
            0.0,
            None,
            None,
            "volume_filter",
        )

    # --------------------------------------------------------
    # EMA SPREAD
    # --------------------------------------------------------

    if ema_spread_pct < MIN_EMA_SPREAD_PCT:

        return StrategySignal(
            False,
            None,
            price,
            momentum,
            atr_pct,
            volume_ratio,
            ema_spread_pct,
            htf_bullish,
            htf_bearish,
            atr_value,
            0.0,
            None,
            None,
            "ema_spread_filter",
        )

    # --------------------------------------------------------
    # POSITION SIZE
    # --------------------------------------------------------

    position_size = calculate_position_size(
        bankroll
    )

    # --------------------------------------------------------
    # TREND STRUCTURE
    # --------------------------------------------------------

    long_signal = (
        ALLOW_LONG
        and momentum > 0
        and float(row.ema_fast)
        > float(row.ema_medium)
        > float(row.ema_slow)
        and price > float(row.ema_fast)
        and htf_bullish
    )

    short_signal = (
        ALLOW_SHORT
        and momentum < 0
        and float(row.ema_fast)
        < float(row.ema_medium)
        < float(row.ema_slow)
        and price < float(row.ema_fast)
        and htf_bearish
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if long_signal:

        stop_distance = (
            atr_value
            * ATR_STOP_MULTIPLIER
        )

        target_distance = (
            atr_value
            * ATR_TARGET_MULTIPLIER
        )

        stop_loss = (
            price
            - stop_distance
        )

        take_profit = (
            price
            + target_distance
        )

        return StrategySignal(
            True,
            "LONG",
            price,
            momentum,
            atr_pct,
            volume_ratio,
            ema_spread_pct,
            htf_bullish,
            htf_bearish,
            atr_value,
            position_size,
            stop_loss,
            take_profit,
            "V3 long",
        )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    if short_signal:

        stop_distance = (
            atr_value
            * ATR_STOP_MULTIPLIER
        )

        target_distance = (
            atr_value
            * ATR_TARGET_MULTIPLIER
        )

        stop_loss = (
            price
            + stop_distance
        )

        take_profit = (
            price
            - target_distance
        )

        return StrategySignal(
            True,
            "SHORT",
            price,
            momentum,
            atr_pct,
            volume_ratio,
            ema_spread_pct,
            htf_bullish,
            htf_bearish,
            atr_value,
            position_size,
            stop_loss,
            take_profit,
            "V3 short",
        )

    return StrategySignal(
        False,
        None,
        price,
        momentum,
        atr_pct,
        volume_ratio,
        ema_spread_pct,
        htf_bullish,
        htf_bearish,
        atr_value,
        0.0,
        None,
        None,
        "trend_filter",
    )
