"""
Bitrue V8 - Pullback Long/Short Strategy

FINAL STRATEGY TEST

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO WALLET
NO REAL ORDERS

Concept:

    1. Establish higher-timeframe trend
    2. Establish 5m trend
    3. Wait for a pullback
    4. Enter only when trend resumes

Long:
    1H bullish
    5m bullish
    recent pullback below fast EMA
    current candle recovers above fast EMA
    positive momentum

Short:
    1H bearish
    5m bearish
    recent pullback above fast EMA
    current candle falls below fast EMA
    negative momentum

No parameter optimization.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ============================================================
# FIXED V8 PARAMETERS
# ============================================================

EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 55

HTF_EMA_FAST = 20
HTF_EMA_SLOW = 50

ATR_PERIOD = 14

MIN_ATR_PCT = 0.10
MAX_ATR_PCT = 0.30

MOMENTUM_LOOKBACK = 3
MIN_MOMENTUM_PCT = 0.10

VOLUME_LOOKBACK = 20
MIN_VOLUME_RATIO = 1.20
MAX_VOLUME_RATIO = 5.00

MIN_EMA_SPREAD_PCT = 0.15

# Pullback must occur within this recent window
PULLBACK_LOOKBACK = 3

# ATR based exits
STOP_ATR_MULTIPLIER = 1.40
TAKE_ATR_MULTIPLIER = 2.50

# Position sizing
POSITION_PCT = 0.01
MIN_POSITION_USD = 5.00
MAX_POSITION_USD = 25.00

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

    pullback_detected: bool
    resumption_detected: bool

    reason: str


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD
) -> pd.Series:

    previous_close = df["close"].shift(1)

    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = (
        df["high"]
        - previous_close
    ).abs()

    tr3 = (
        df["low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    return (
        true_range
        .rolling(period)
        .mean()
    )


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:

    data = df.copy()

    data["ema_fast"] = (
        data["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )

    data["ema_medium"] = (
        data["close"]
        .ewm(
            span=EMA_MEDIUM,
            adjust=False
        )
        .mean()
    )

    data["ema_slow"] = (
        data["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    data["momentum_pct"] = (
        data["close"]
        .pct_change(
            MOMENTUM_LOOKBACK
        )
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
        .rolling(
            VOLUME_LOOKBACK
        )
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

    # --------------------------------------------------------
    # PREVIOUS VALUES
    # --------------------------------------------------------

    data["prev_close"] = (
        data["close"].shift(1)
    )

    data["prev_ema_fast"] = (
        data["ema_fast"].shift(1)
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
        .ewm(
            span=HTF_EMA_FAST,
            adjust=False
        )
        .mean()
    )

    htf["htf_ema50"] = (
        htf["htf_close"]
        .ewm(
            span=HTF_EMA_SLOW,
            adjust=False
        )
        .mean()
    )

    # IMPORTANT:
    # Only use the previous completed 1H candle.
    htf["htf_ema20"] = (
        htf["htf_ema20"].shift(1)
    )

    htf["htf_ema50"] = (
        htf["htf_ema50"].shift(1)
    )

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

    bullish_regime = (
        (data["ema_fast"] > data["ema_medium"])
        &
        (data["ema_medium"] > data["ema_slow"])
        &
        (data["htf_ema20"] > data["htf_ema50"])
        &
        (data["ema_spread_pct"] >= MIN_EMA_SPREAD_PCT)
    )

    bearish_regime = (
        (data["ema_fast"] < data["ema_medium"])
        &
        (data["ema_medium"] < data["ema_slow"])
        &
        (data["htf_ema20"] < data["htf_ema50"])
        &
        (data["ema_spread_pct"] >= MIN_EMA_SPREAD_PCT)
    )

    data["regime"] = "NEUTRAL"

    data.loc[
        bullish_regime,
        "regime"
    ] = "BULL"

    data.loc[
        bearish_regime,
        "regime"
    ] = "BEAR"

    # ========================================================
    # PULLBACK DETECTION
    # ========================================================

    # LONG:
    # Within the last N candles price traded below
    # the fast EMA.
    long_pullback = (
        data["low"]
        .rolling(
            PULLBACK_LOOKBACK
        )
        .min()
        <
        data["ema_fast"]
    )

    # SHORT:
    # Within the last N candles price traded above
    # the fast EMA.
    short_pullback = (
        data["high"]
        .rolling(
            PULLBACK_LOOKBACK
        )
        .max()
        >
        data["ema_fast"]
    )

    data["long_pullback"] = (
        long_pullback
    )

    data["short_pullback"] = (
        short_pullback
    )

    # ========================================================
    # CURRENT RESUMPTION
    # ========================================================

    # Current candle recovers above EMA9
    data["long_resumption"] = (
        (data["close"] > data["ema_fast"])
        &
        (data["close"] > data["open"])
    )

    # Current candle falls below EMA9
    data["short_resumption"] = (
        (data["close"] < data["ema_fast"])
        &
        (data["close"] < data["open"])
    )

    return data


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    balance: float
) -> float:

    size = (
        balance
        * POSITION_PCT
    )

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
    data: pd.DataFrame,
    index: int,
    balance: float
) -> StrategySignal:

    row = data.iloc[index]

    price = float(row["close"])

    momentum = float(
        row["momentum_pct"]
    )

    atr_pct = float(
        row["atr_pct"]
    )

    volume_ratio = float(
        row["volume_ratio"]
    )

    ema_spread = float(
        row["ema_spread_pct"]
    )

    atr = float(
        row["atr"]
    )

    regime = str(
        row["regime"]
    )

    long_pullback = bool(
        row["long_pullback"]
    )

    short_pullback = bool(
        row["short_pullback"]
    )

    long_resumption = bool(
        row["long_resumption"]
    )

    short_resumption = bool(
        row["short_resumption"]
    )

    position_size = (
        calculate_position_size(
            balance
        )
    )

    # --------------------------------------------------------
    # Missing data
    # --------------------------------------------------------

    values = [
        price,
        momentum,
        atr_pct,
        volume_ratio,
        ema_spread,
        atr
    ]

    if not all(
        pd.notna(v)
        for v in values
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
            regime="UNKNOWN",
            pullback_detected=False,
            resumption_detected=False,
            reason="insufficient data"
        )

    # --------------------------------------------------------
    # ATR FILTER
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
            pullback_detected=False,
            resumption_detected=False,
            reason="ATR outside range"
        )

    # --------------------------------------------------------
    # VOLUME FILTER
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
            pullback_detected=False,
            resumption_detected=False,
            reason="volume outside range"
        )

    # ========================================================
    # LONG PULLBACK
    # ========================================================

    if (
        ALLOW_LONG
        and regime == "BULL"
        and long_pullback
        and long_resumption
        and momentum >= MIN_MOMENTUM_PCT
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
            pullback_detected=True,
            resumption_detected=True,
            reason="bull trend pullback + recovery"
        )

    # ========================================================
    # SHORT PULLBACK
    # ========================================================

    if (
        ALLOW_SHORT
        and regime == "BEAR"
        and short_pullback
        and short_resumption
        and momentum <= -MIN_MOMENTUM_PCT
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
            pullback_detected=True,
            resumption_detected=True,
            reason="bear trend pullback + recovery"
        )

    # ========================================================
    # NO TRADE
    # ========================================================

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
        pullback_detected=False,
        resumption_detected=False,
        reason="no valid pullback setup"
    )
