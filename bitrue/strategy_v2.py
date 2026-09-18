"""
Bitrue Strategy V2
PAPER / BACKTEST ONLY
NO LIVE TRADING
"""

from dataclasses import dataclass

import pandas as pd


# ============================================================
# STRATEGY V2
# ============================================================

EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 55

MOMENTUM_LOOKBACK = 3
MIN_MOMENTUM_PCT = 0.15

ATR_PERIOD = 14
MIN_ATR_PCT = 0.05

VOLUME_LOOKBACK = 20
MIN_VOLUME_RATIO = 1.20

# Neuer Filter:
# EMA 9 und EMA 21 müssen einen Mindestabstand
# relativ zum aktuellen Preis haben.
MIN_EMA_SPREAD_PCT = 0.03

STOP_LOSS_PCT = 0.30
TAKE_PROFIT_PCT = 0.60

MIN_POSITION_USD = 5.00
MAX_POSITION_USD = 25.00

# Für einen fairen Vergleich mit V1
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

    position_size: float

    stop_loss: float | None
    take_profit: float | None

    reason: str


# ============================================================
# INDIKATOREN
# ============================================================

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:

    df = data.copy()

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema_fast"] = (
        close
        .ewm(span=EMA_FAST, adjust=False)
        .mean()
    )

    df["ema_medium"] = (
        close
        .ewm(span=EMA_MEDIUM, adjust=False)
        .mean()
    )

    df["ema_slow"] = (
        close
        .ewm(span=EMA_SLOW, adjust=False)
        .mean()
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["momentum_pct"] = (
        close.pct_change(MOMENTUM_LOOKBACK)
        * 100.0
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        true_range
        .rolling(ATR_PERIOD)
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
        .rolling(VOLUME_LOOKBACK)
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

    return df


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    bankroll: float
) -> float:

    raw_size = (
        bankroll
        * POSITION_FRACTION
    )

    return max(
        MIN_POSITION_USD,
        min(
            MAX_POSITION_USD,
            raw_size
        )
    )


# ============================================================
# SIGNAL EVALUATION
# ============================================================

def evaluate_precomputed(
    row,
    bankroll: float
) -> StrategySignal:

    price = float(row.close)

    values = [
        row.ema_fast,
        row.ema_medium,
        row.ema_slow,
        row.momentum_pct,
        row.atr_pct,
        row.volume_ratio,
        row.ema_spread_pct,
    ]

    # --------------------------------------------------------
    # Noch nicht genug Daten
    # --------------------------------------------------------

    if any(pd.isna(value) for value in values):

        return StrategySignal(
            signal=False,
            side=None,
            price=price,
            momentum_pct=0.0,
            atr_pct=0.0,
            volume_ratio=0.0,
            ema_spread_pct=0.0,
            position_size=0.0,
            stop_loss=None,
            take_profit=None,
            reason="insufficient_data",
        )

    momentum = float(
        row.momentum_pct
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

    # --------------------------------------------------------
    # MOMENTUM FILTER
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
            0.0,
            None,
            None,
            "momentum_filter",
        )

    # --------------------------------------------------------
    # ATR FILTER
    # --------------------------------------------------------

    if atr_pct < MIN_ATR_PCT:

        return StrategySignal(
            False,
            None,
            price,
            momentum,
            atr_pct,
            volume_ratio,
            ema_spread_pct,
            0.0,
            None,
            None,
            "atr_filter",
        )

    # --------------------------------------------------------
    # VOLUME FILTER
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
            0.0,
            None,
            None,
            "volume_filter",
        )

    # --------------------------------------------------------
    # EMA SPREAD FILTER
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
    # LONG
    # --------------------------------------------------------

    long_signal = (
        ALLOW_LONG
        and momentum > 0
        and float(row.ema_fast)
        > float(row.ema_medium)
        > float(row.ema_slow)
        and price > float(row.ema_fast)
    )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    short_signal = (
        ALLOW_SHORT
        and momentum < 0
        and float(row.ema_fast)
        < float(row.ema_medium)
        < float(row.ema_slow)
        and price < float(row.ema_fast)
    )

    # --------------------------------------------------------
    # LONG SIGNAL
    # --------------------------------------------------------

    if long_signal:

        stop_loss = (
            price
            * (1.0 - STOP_LOSS_PCT / 100.0)
        )

        take_profit = (
            price
            * (1.0 + TAKE_PROFIT_PCT / 100.0)
        )

        return StrategySignal(
            True,
            "LONG",
            price,
            momentum,
            atr_pct,
            volume_ratio,
            ema_spread_pct,
            position_size,
            stop_loss,
            take_profit,
            "V2 long",
        )

    # --------------------------------------------------------
    # SHORT SIGNAL
    # --------------------------------------------------------

    if short_signal:

        stop_loss = (
            price
            * (1.0 + STOP_LOSS_PCT / 100.0)
        )

        take_profit = (
            price
            * (1.0 - TAKE_PROFIT_PCT / 100.0)
        )

        return StrategySignal(
            True,
            "SHORT",
            price,
            momentum,
            atr_pct,
            volume_ratio,
            ema_spread_pct,
            position_size,
            stop_loss,
            take_profit,
            "V2 short",
        )

    # --------------------------------------------------------
    # KEIN SIGNAL
    # --------------------------------------------------------

    return StrategySignal(
        False,
        None,
        price,
        momentum,
        atr_pct,
        volume_ratio,
        ema_spread_pct,
        0.0,
        None,
        None,
        "trend_filter",
    )
