"""
Bitrue V7 - Long/Short Regime Strategy

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO WALLET
NO REAL ORDERS

Concept:
    Strong bullish regime  -> LONG
    Strong bearish regime -> SHORT
    Neutral / unclear      -> NO TRADE

V7 is intentionally NOT a parameter sweep.
The parameters are fixed before the OOS test.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ============================================================
# V7 FIXED PARAMETERS
# ============================================================

EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 55

HTF_EMA_FAST = 20
HTF_EMA_SLOW = 50

MOMENTUM_LOOKBACK = 3
MIN_MOMENTUM_PCT = 0.15

ATR_PERIOD = 14

# V7 deliberately uses a broader usable volatility regime
MIN_ATR_PCT = 0.10
MAX_ATR_PCT = 0.30

VOLUME_LOOKBACK = 20
MIN_VOLUME_RATIO = 1.20
MAX_VOLUME_RATIO = 5.00

# Minimum distance between fast and slow trend
MIN_EMA_SPREAD_PCT = 0.15

# ATR based risk/reward
STOP_ATR_MULTIPLIER = 1.40
TAKE_ATR_MULTIPLIER = 2.50

# Position sizing
POSITION_PCT = 0.01
MIN_POSITION_USD = 5.00
MAX_POSITION_USD = 25.00

# Direction
ALLOW_LONG = True
ALLOW_SHORT = True


@dataclass
class StrategySignal:
    signal: str
    side: Optional[str]
    price: float

    momentum_pct: float
    atr_pct: float
    volume_ratio: float
    ema_spread_pct: float

    position_size: float

    stop_loss: Optional[float]
    take_profit: Optional[float]

    regime: str
    reason: str


# ============================================================
# INDICATORS
# ============================================================

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # --------------------------------------------------------
    # 5 MINUTE TREND
    # --------------------------------------------------------

    data["ema_fast"] = (
        data["close"]
        .ewm(span=EMA_FAST, adjust=False)
        .mean()
    )

    data["ema_medium"] = (
        data["close"]
        .ewm(span=EMA_MEDIUM, adjust=False)
        .mean()
    )

    data["ema_slow"] = (
        data["close"]
        .ewm(span=EMA_SLOW, adjust=False)
        .mean()
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    data["momentum_pct"] = (
        data["close"]
        .pct_change(MOMENTUM_LOOKBACK)
        * 100.0
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    data["atr"] = calculate_atr(data)

    data["atr_pct"] = (
        data["atr"]
        / data["close"]
        * 100.0
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    data["volume_mean"] = (
        data["volume"]
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"]
        / data["volume_mean"]
    )

    # --------------------------------------------------------
    # EMA SPREAD
    # --------------------------------------------------------

    data["ema_spread_pct"] = (
        (
            data["ema_fast"]
            - data["ema_slow"]
        ).abs()
        / data["close"]
        * 100.0
    )

    # ========================================================
    # 1H HIGHER TIMEFRAME
    # ========================================================

    htf = (
        data.set_index("timestamp")["close"]
        .resample("1h")
        .last()
        .dropna()
        .to_frame("htf_close")
    )

    htf["htf_ema20"] = (
        htf["htf_close"]
        .ewm(span=HTF_EMA_FAST, adjust=False)
        .mean()
    )

    htf["htf_ema50"] = (
        htf["htf_close"]
        .ewm(span=HTF_EMA_SLOW, adjust=False)
        .mean()
    )

    # IMPORTANT:
    # Use the previous completed 1H candle.
    # This prevents look-ahead bias.
    htf["htf_ema20"] = htf["htf_ema20"].shift(1)
    htf["htf_ema50"] = htf["htf_ema50"].shift(1)

    htf = htf[
        [
            "htf_ema20",
            "htf_ema50"
        ]
    ].reset_index()

    data = pd.merge_asof(
        data.sort_values("timestamp"),
        htf.sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )

    # ========================================================
    # REGIME
    # ========================================================

    bullish = (
        (data["ema_fast"] > data["ema_medium"])
        &
        (data["ema_medium"] > data["ema_slow"])
        &
        (data["close"] > data["ema_fast"])
        &
        (data["htf_ema20"] > data["htf_ema50"])
        &
        (data["momentum_pct"] >= MIN_MOMENTUM_PCT)
        &
        (data["ema_spread_pct"] >= MIN_EMA_SPREAD_PCT)
    )

    bearish = (
        (data["ema_fast"] < data["ema_medium"])
        &
        (data["ema_medium"] < data["ema_slow"])
        &
        (data["close"] < data["ema_fast"])
        &
        (data["htf_ema20"] < data["htf_ema50"])
        &
        (data["momentum_pct"] <= -MIN_MOMENTUM_PCT)
        &
        (data["ema_spread_pct"] >= MIN_EMA_SPREAD_PCT)
    )

    data["regime"] = "NEUTRAL"
    data.loc[bullish, "regime"] = "BULL"
    data.loc[bearish, "regime"] = "BEAR"

    return data


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(balance: float) -> float:
    size = balance * POSITION_PCT

    size = max(
        MIN_POSITION_USD,
        size
    )

    size = min(
        MAX_POSITION_USD,
        size
    )

    return size


# ============================================================
# SIGNAL
# ============================================================

def evaluate_precomputed(
    row,
    balance: float
) -> StrategySignal:

    price = float(row.close)

    momentum = float(row.momentum_pct)
    atr_pct = float(row.atr_pct)
    volume_ratio = float(row.volume_ratio)
    ema_spread = float(row.ema_spread_pct)

    atr = float(row.atr)

    regime = str(row.regime)

    position_size = calculate_position_size(balance)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not all([
        pd.notna(price),
        pd.notna(momentum),
        pd.notna(atr_pct),
        pd.notna(volume_ratio),
        pd.notna(ema_spread),
        pd.notna(atr)
    ]):
        return StrategySignal(
            signal="HOLD",
            side=None,
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            ema_spread_pct=ema_spread,
            position_size=0.0,
            stop_loss=None,
            take_profit=None,
            regime="UNKNOWN",
            reason="insufficient data"
        )

    # --------------------------------------------------------
    # Volatility filter
    # --------------------------------------------------------

    if not (
        MIN_ATR_PCT
        <= atr_pct
        <= MAX_ATR_PCT
    ):
        return StrategySignal(
            signal="HOLD",
            side=None,
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            ema_spread_pct=ema_spread,
            position_size=0.0,
            stop_loss=None,
            take_profit=None,
            regime=regime,
            reason="ATR outside trading regime"
        )

    # --------------------------------------------------------
    # Volume filter
    # --------------------------------------------------------

    if not (
        MIN_VOLUME_RATIO
        <= volume_ratio
        <= MAX_VOLUME_RATIO
    ):
        return StrategySignal(
            signal="HOLD",
            side=None,
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            ema_spread_pct=ema_spread,
            position_size=0.0,
            stop_loss=None,
            take_profit=None,
            regime=regime,
            reason="volume outside trading regime"
        )

    # ========================================================
    # LONG
    # ========================================================

    if (
        regime == "BULL"
        and ALLOW_LONG
    ):

        stop_loss = (
            price
            - STOP_ATR_MULTIPLIER * atr
        )

        take_profit = (
            price
            + TAKE_ATR_MULTIPLIER * atr
        )

        return StrategySignal(
            signal="BUY",
            side="LONG",
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            ema_spread_pct=ema_spread,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime=regime,
            reason="bullish 5m + bullish 1h regime"
        )

    # ========================================================
    # SHORT
    # ========================================================

    if (
        regime == "BEAR"
        and ALLOW_SHORT
    ):

        stop_loss = (
            price
            + STOP_ATR_MULTIPLIER * atr
        )

        take_profit = (
            price
            - TAKE_ATR_MULTIPLIER * atr
        )

        return StrategySignal(
            signal="SELL",
            side="SHORT",
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            ema_spread_pct=ema_spread,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime=regime,
            reason="bearish 5m + bearish 1h regime"
        )

    # --------------------------------------------------------
    # NO TRADE
    # --------------------------------------------------------

    return StrategySignal(
        signal="HOLD",
        side=None,
        price=price,
        momentum_pct=momentum,
        atr_pct=atr_pct,
        volume_ratio=volume_ratio,
        ema_spread_pct=ema_spread,
        position_size=0.0,
        stop_loss=None,
        take_profit=None,
        regime=regime,
        reason="no valid directional regime"
    )
