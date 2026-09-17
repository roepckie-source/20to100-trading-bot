"""
Bitrue BTC/USDT 5m Strategy V1

Momentum + EMA + ATR + Volume

BACKTEST / PAPER ONLY
NO LIVE TRADING
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    EMA_FAST,
    EMA_MEDIUM,
    EMA_SLOW,
    MOMENTUM_LOOKBACK,
    MIN_MOMENTUM_PCT,
    ATR_PERIOD,
    MIN_ATR_PCT,
    VOLUME_LOOKBACK,
    MIN_VOLUME_RATIO,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    RISK_PER_TRADE,
    MIN_POSITION_USD,
    MAX_POSITION_USD,
    ALLOW_LONG,
    ALLOW_SHORT,
)


# ============================================================
# RESULT
# ============================================================

@dataclass
class StrategySignal:

    signal: bool = False

    side: Optional[str] = None

    price: float = 0.0

    momentum_pct: float = 0.0

    atr_pct: float = 0.0

    volume_ratio: float = 0.0

    position_size: float = 0.0

    stop_loss: float = 0.0

    take_profit: float = 0.0

    reason: str = ""


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()

    required = ["open", "high", "low", "close", "volume"]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

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
    # Momentum
    # --------------------------------------------------------

    data["momentum_pct"] = (
        data["close"]
        .pct_change(MOMENTUM_LOOKBACK)
        * 100
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = data["close"].shift(1)

    tr1 = data["high"] - data["low"]

    tr2 = (
        data["high"] - previous_close
    ).abs()

    tr3 = (
        data["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    data["atr"] = (
        true_range
        .rolling(ATR_PERIOD)
        .mean()
    )

    data["atr_pct"] = (
        data["atr"]
        / data["close"]
        * 100
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    data["average_volume"] = (
        data["volume"]
        .rolling(VOLUME_LOOKBACK)
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"]
        / data["average_volume"]
    )

    return data


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    bankroll: float,
) -> float:

    if bankroll <= 0:
        return 0.0

    position = (
        bankroll
        * RISK_PER_TRADE
    )

    position = max(
        MIN_POSITION_USD,
        position
    )

    position = min(
        MAX_POSITION_USD,
        position
    )

    return round(position, 2)


# ============================================================
# STRATEGY EVALUATION
# ============================================================

def evaluate(
    df: pd.DataFrame,
    bankroll: float,
) -> StrategySignal:

    if len(df) < max(
        EMA_SLOW,
        ATR_PERIOD,
        VOLUME_LOOKBACK,
        MOMENTUM_LOOKBACK,
    ) + 5:

        return StrategySignal(
            reason="Not enough data"
        )

    data = calculate_indicators(df)

    row = data.iloc[-1]

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

    ema_fast = float(
        row["ema_fast"]
    )

    ema_medium = float(
        row["ema_medium"]
    )

    ema_slow = float(
        row["ema_slow"]
    )

    # --------------------------------------------------------
    # NaN protection
    # --------------------------------------------------------

    values = [
        price,
        momentum,
        atr_pct,
        volume_ratio,
        ema_fast,
        ema_medium,
        ema_slow,
    ]

    if any(
        pd.isna(value)
        for value in values
    ):

        return StrategySignal(
            reason="Indicator data incomplete"
        )

    # --------------------------------------------------------
    # Momentum filter
    # --------------------------------------------------------

    if abs(momentum) < MIN_MOMENTUM_PCT:

        return StrategySignal(
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            reason=(
                f"Momentum too small: "
                f"{momentum:.4f}%"
            ),
        )

    # --------------------------------------------------------
    # ATR filter
    # --------------------------------------------------------

    if atr_pct < MIN_ATR_PCT:

        return StrategySignal(
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            reason=(
                f"Volatility too low: "
                f"{atr_pct:.4f}%"
            ),
        )

    # --------------------------------------------------------
    # Volume filter
    # --------------------------------------------------------

    if volume_ratio < MIN_VOLUME_RATIO:

        return StrategySignal(
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            reason=(
                f"Volume too low: "
                f"{volume_ratio:.2f}x"
            ),
        )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    long_condition = (
        momentum > 0
        and ema_fast > ema_medium
        and ema_medium > ema_slow
        and price > ema_fast
    )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    short_condition = (
        momentum < 0
        and ema_fast < ema_medium
        and ema_medium < ema_slow
        and price < ema_fast
    )

    # --------------------------------------------------------
    # LONG SIGNAL
    # --------------------------------------------------------

    if long_condition and ALLOW_LONG:

        position = calculate_position_size(
            bankroll
        )

        stop_loss = (
            price
            * (1 - STOP_LOSS_PCT / 100)
        )

        take_profit = (
            price
            * (1 + TAKE_PROFIT_PCT / 100)
        )

        return StrategySignal(
            signal=True,
            side="LONG",
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            position_size=position,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="LONG: momentum + EMA trend + volume",
        )

    # --------------------------------------------------------
    # SHORT SIGNAL
    # --------------------------------------------------------

    if short_condition and ALLOW_SHORT:

        position = calculate_position_size(
            bankroll
        )

        stop_loss = (
            price
            * (1 + STOP_LOSS_PCT / 100)
        )

        take_profit = (
            price
            * (1 - TAKE_PROFIT_PCT / 100)
        )

        return StrategySignal(
            signal=True,
            side="SHORT",
            price=price,
            momentum_pct=momentum,
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            position_size=position,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="SHORT: momentum + EMA trend + volume",
        )

    return StrategySignal(
        price=price,
        momentum_pct=momentum,
        atr_pct=atr_pct,
        volume_ratio=volume_ratio,
        reason="No aligned trend signal",
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BITRUE BTC/USDT 5M STRATEGY V1")
    print("=" * 60)
    print()
    print("MOMENTUM + EMA + ATR + VOLUME")
    print()
    print("BACKTEST / PAPER ONLY")
    print("NO LIVE TRADING")
    print("NO ORDERS")
    print()
    print("Configuration loaded successfully.")
    print()
    print(f"Symbol:          BTC/USDT")
    print(f"Timeframe:       5m")
    print(f"EMA:             {EMA_FAST}/{EMA_MEDIUM}/{EMA_SLOW}")
    print(f"Momentum:        {MIN_MOMENTUM_PCT}%")
    print(f"ATR minimum:     {MIN_ATR_PCT}%")
    print(f"Volume minimum:  {MIN_VOLUME_RATIO}x")
    print(f"Stop Loss:       {STOP_LOSS_PCT}%")
    print(f"Take Profit:     {TAKE_PROFIT_PCT}%")
    print()
    print("=" * 60)
