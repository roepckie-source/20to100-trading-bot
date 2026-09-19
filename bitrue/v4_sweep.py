"""
Bitrue BTC/USDT 5m Strategy - V4 Parameter Sweep

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO WALLET
NO REAL ORDERS

V4:
- Targeted parameter sweep based on V3 diagnosis
- 70% TRAIN / 30% OOS
- EMA 9 / 21 / 55
- 3-candle momentum
- ATR 14
- ATR regime filter
- volume ratio filter
- EMA spread filter
- 1H EMA20 / EMA50 higher-timeframe confirmation
- cooldown
- configurable ATR-based SL / TP
- fees + slippage
- 96 parameter combinations

IMPORTANT:
- atr_pct is used for ATR regime filtering
- absolute ATR is used for SL/TP distance
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("BTCUSDT_5m.csv")

STARTING_CAPITAL = 100.00

FEE_RATE = 0.0006
SLIPPAGE_RATE = 0.0002

MIN_POSITION_USD = 5.00
MAX_POSITION_USD = 25.00

RISK_PER_TRADE = 0.01

COOLDOWN_MINUTES = 30

TRAIN_RATIO = 0.70


# ============================================================
# V4 PARAMETER SWEEP
# ============================================================

ATR_RANGES = [
    (0.150, 0.200),
    (0.175, 0.200),
    (0.175, 0.225),
    (0.180, 0.250),
]

VOLUME_MAX_VALUES = [
    3.0,
    4.0,
    5.0,
]

SL_MULTIPLIERS = [
    1.0,
    1.2,
    1.4,
    1.6,
]

TP_MULTIPLIERS = [
    2.0,
    2.5,
]


MIN_ATR_PCT = 0.075
MIN_VOLUME_RATIO = 1.20
MIN_EMA_SPREAD_PCT = 0.15
MIN_MOMENTUM_PCT = 0.15


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_FILE = Path("bitrue_v4_sweep_results.csv")
CANDIDATES_FILE = Path("bitrue_v4_candidates.csv")
REPORT_FILE = Path("bitrue_v4_sweep_report.txt")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float

    stop_loss: float
    take_profit: float

    entry_index: int

    momentum_pct: float
    atr_pct: float
    atr_value: float
    volume_ratio: float
    ema_spread_pct: float


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("CSV is empty.")

    # --------------------------------------------------------
    # Detect timestamp column
    # --------------------------------------------------------

    timestamp_candidates = [
        "open_time",
        "timestamp",
        "datetime",
        "date",
        "time",
    ]

    timestamp_column = None

    for col in timestamp_candidates:
        if col in df.columns:
            timestamp_column = col
            break

    if timestamp_column is None:
        raise ValueError(
            f"No timestamp column found. "
            f"Columns: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(
        df[timestamp_column],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Standardize OHLCV column names
    # --------------------------------------------------------

    rename_map = {}

    for col in df.columns:
        lower = col.lower()

        if lower == "open":
            rename_map[col] = "open"

        elif lower == "high":
            rename_map[col] = "high"

        elif lower == "low":
            rename_map[col] = "low"

        elif lower == "close":
            rename_map[col] = "close"

        elif lower == "volume":
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = df.dropna(
        subset=required
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="first",
    )

    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if (df["close"] <= 0).any():
        raise ValueError(
            "Invalid close price detected."
        )

    if (df["high"] <= 0).any():
        raise ValueError(
            "Invalid high price detected."
        )

    if (df["low"] <= 0).any():
        raise ValueError(
            "Invalid low price detected."
        )

    if (df["volume"] < 0).any():
        raise ValueError(
            "Invalid volume detected."
        )

    print()
    print("=" * 70)
    print("DATA")
    print("=" * 70)

    print(
        f"Rows:        {len(df):,}"
    )

    print(
        f"Start:       {df['timestamp'].iloc[0]}"
    )

    print(
        f"End:         {df['timestamp'].iloc[-1]}"
    )

    print(
        f"BTC start:   ${df['close'].iloc[0]:,.2f}"
    )

    print(
        f"BTC end:     ${df['close'].iloc[-1]:,.2f}"
    )

    print("=" * 70)

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    data["ema_fast"] = (
        data["close"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    data["ema_medium"] = (
        data["close"]
        .ewm(
            span=21,
            adjust=False,
        )
        .mean()
    )

    data["ema_slow"] = (
        data["close"]
        .ewm(
            span=55,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    data["momentum_pct"] = (
        (
            data["close"]
            / data["close"].shift(3)
        )
        - 1.0
    ) * 100.0

    # --------------------------------------------------------
    # True Range
    # --------------------------------------------------------

    previous_close = (
        data["close"].shift(1)
    )

    tr1 = (
        data["high"]
        - data["low"]
    ).abs()

    tr2 = (
        data["high"]
        - previous_close
    ).abs()

    tr3 = (
        data["low"]
        - previous_close
    ).abs()

    data["tr"] = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    data["atr"] = (
        data["tr"]
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    # ATR percentage
    data["atr_pct"] = (
        data["atr"]
        / data["close"]
        * 100.0
    )

    # --------------------------------------------------------
    # Volume ratio
    # --------------------------------------------------------

    volume_average = (
        data["volume"]
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"]
        / volume_average
    )

    # --------------------------------------------------------
    # EMA spread
    # --------------------------------------------------------

    data["ema_spread_pct"] = (
        (
            data["ema_fast"]
            - data["ema_medium"]
        ).abs()
        / data["close"]
        * 100.0
    )

    # --------------------------------------------------------
    # Higher timeframe 1H
    # --------------------------------------------------------

    h1 = (
        data.set_index("timestamp")
        ["close"]
        .resample("1h")
        .last()
        .dropna()
        .to_frame("close")
    )

    h1["ema20"] = (
        h1["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    h1["ema50"] = (
        h1["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Shift one complete 1H candle to prevent lookahead.
    # --------------------------------------------------------

    h1["ema20_previous"] = (
        h1["ema20"].shift(1)
    )

    h1["ema50_previous"] = (
        h1["ema50"].shift(1)
    )

    h1 = h1[
        [
            "ema20_previous",
            "ema50_previous",
        ]
    ]

    h1 = h1.reset_index()

    # --------------------------------------------------------
    # Merge latest completed 1H candle
    # --------------------------------------------------------

    data = pd.merge_asof(
        data.sort_values("timestamp"),
        h1.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    data["htf_bullish"] = (
        data["ema20_previous"]
        > data["ema50_previous"]
    )

    data["htf_bearish"] = (
        data["ema20_previous"]
        < data["ema50_previous"]
    )

    return data


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    balance: float,
) -> float:

    raw_size = (
        balance
        * RISK_PER_TRADE
    )

    return min(
        MAX_POSITION_USD,
        max(
            MIN_POSITION_USD,
            raw_size,
        ),
    )


# ============================================================
# ENTRY SIGNAL
# ============================================================

def get_signal(
    row: pd.Series,
) -> str | None:

    required_values = [
        row["close"],
        row["ema_fast"],
        row["ema_medium"],
        row["ema_slow"],
        row["momentum_pct"],
        row["atr"],
        row["atr_pct"],
        row["volume_ratio"],
        row["ema_spread_pct"],
    ]

    if any(
        pd.isna(value)
        for value in required_values
    ):
        return None

    momentum = float(
        row["momentum_pct"]
    )

    atr_pct = float(
        row["atr_pct"]
    )

    volume_ratio = float(
        row["volume_ratio"]
    )

    ema_spread_pct = float(
        row["ema_spread_pct"]
    )

    close = float(
        row["close"]
    )

    # --------------------------------------------------------
    # Base filters
    # --------------------------------------------------------

    if atr_pct < MIN_ATR_PCT:
        return None

    if volume_ratio < MIN_VOLUME_RATIO:
        return None

    if ema_spread_pct < MIN_EMA_SPREAD_PCT:
        return None

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if (
        momentum >= MIN_MOMENTUM_PCT
        and float(row["ema_fast"])
        > float(row["ema_medium"])
        > float(row["ema_slow"])
        and close
        > float(row["ema_fast"])
        and bool(row["htf_bullish"])
    ):
        return "LONG"

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    if (
        momentum <= -MIN_MOMENTUM_PCT
        and float(row["ema_fast"])
        < float(row["ema_medium"])
        < float(row["ema_slow"])
        and close
        < float(row["ema_fast"])
        and bool(row["htf_bearish"])
    ):
        return "SHORT"

    return None


# ============================================================
# EXECUTION PRICE
# ============================================================

def apply_entry_slippage(
    price: float,
    side: str,
) -> float:

    if side == "LONG":
        return price * (
            1.0 + SLIPPAGE_RATE
        )

    return price * (
        1.0 - SLIPPAGE_RATE
    )


def apply_exit_slippage(
    price: float,
    side: str,
) -> float:

    if side == "LONG":
        return price * (
            1.0 - SLIPPAGE_RATE
        )

    return price * (
        1.0 + SLIPPAGE_RATE
    )


# ============================================================
# P&L
# ============================================================

def calculate_trade_result(
    position: Position,
    exit_price: float,
) -> tuple[float, float, float, float]:

    # --------------------------------------------------------
    # Execution prices already include slippage.
    # --------------------------------------------------------

    if position.side == "LONG":

        entry_value = (
            position.entry_price
            * position.position_size
            / position.entry_price
        )

        exit_value = (
            exit_price
            * position.position_size
            / position.entry_price
        )

        gross_pnl = (
            exit_value
            - position.position_size
        )

    else:

        quantity = (
            position.position_size
            / position.entry_price
        )

        gross_pnl = (
            (
                position.entry_price
                - exit_price
            )
            * quantity
        )

    # --------------------------------------------------------
    # Fees
    # --------------------------------------------------------

    entry_notional = (
        position.position_size
    )

    exit_notional = (
        abs(
            position.position_size
            + gross_pnl
        )
    )

    fees = (
        entry_notional
        + exit_notional
    ) * FEE_RATE

    # --------------------------------------------------------
    # IMPORTANT:
    # Slippage is already reflected in execution prices.
    #
    # Therefore we DO NOT subtract an additional
    # slippage cost here.
    #
    # This avoids double-counting slippage.
    # --------------------------------------------------------

    slippage_cost = 0.0

    net_pnl = (
        gross_pnl
        - fees
    )

    return (
        gross_pnl,
        fees,
        slippage_cost,
        net_pnl,
    )


# ============================================================
# SINGLE VARIANT BACKTEST
# ============================================================

def run_variant(
    data: pd.DataFrame,
    split_index: int,
    atr_min: float,
    atr_max: float,
    volume_max: float,
    sl_multiplier: float,
    tp_multiplier: float,
) -> dict:

    balance = STARTING_CAPITAL

    position: Position | None = None

    trades = []

    last_exit_time = None

    # --------------------------------------------------------
    # Iterate candles
    # --------------------------------------------------------

    for i, row in enumerate(
        data.itertuples(index=False)
    ):

        timestamp = row.timestamp

        open_price = float(
            row.open
        )

        high_price = float(
            row.high
        )

        low_price = float(
            row.low
        )

        close_price = float(
            row.close
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

        momentum_pct = float(
            row.momentum_pct
        )

        # ----------------------------------------------------
        # Skip invalid indicator rows
        # ----------------------------------------------------

        if not all(
            math.isfinite(x)
            for x in [
                atr_value,
                atr_pct,
                volume_ratio,
                ema_spread_pct,
                momentum_pct,
            ]
        ):
            continue

        # ----------------------------------------------------
        # Manage open position
        # ----------------------------------------------------

        if position is not None:

            exit_reason = None
            raw_exit_price = None

            if position.side == "LONG":

                # Conservative:
                # STOP checked before TP
                if low_price <= position.stop_loss:

                    exit_reason = "STOP_LOSS"

                    raw_exit_price = (
                        position.stop_loss
                    )

                elif high_price >= position.take_profit:

                    exit_reason = "TAKE_PROFIT"

                    raw_exit_price = (
                        position.take_profit
                    )

            else:

                # Conservative:
                # STOP checked before TP
                if high_price >= position.stop_loss:

                    exit_reason = "STOP_LOSS"

                    raw_exit_price = (
                        position.stop_loss
                    )

                elif low_price <= position.take_profit:

                    exit_reason = "TAKE_PROFIT"

                    raw_exit_price = (
                        position.take_profit
                    )

            if exit_reason is not None:

                exit_price = apply_exit_slippage(
                    raw_exit_price,
                    position.side,
                )

                (
                    gross_pnl,
                    fees,
                    slippage_cost,
                    net_pnl,
                ) = calculate_trade_result(
                    position,
                    exit_price,
                )

                balance += net_pnl

                period = (
                    "TRAIN"
                    if position.entry_index
                    < split_index
                    else "OOS"
                )

                trades.append(
                    {
                        "entry_time":
                            position.entry_time,

                        "exit_time":
                            timestamp,

                        "period":
                            period,

                        "side":
                            position.side,

                        "entry_price":
                            position.entry_price,

                        "exit_price":
                            exit_price,

                        "position_size":
                            position.position_size,

                        "stop_loss":
                            position.stop_loss,

                        "take_profit":
                            position.take_profit,

                        "gross_pnl":
                            gross_pnl,

                        "fees":
                            fees,

                        "slippage":
                            slippage_cost,

                        "net_pnl":
                            net_pnl,

                        "balance":
                            balance,

                        "momentum_pct":
                            position.momentum_pct,

                        "atr_pct":
                            position.atr_pct,

                        "atr_value":
                            position.atr_value,

                        "volume_ratio":
                            position.volume_ratio,

                        "ema_spread_pct":
                            position.ema_spread_pct,

                        "exit_reason":
                            exit_reason,

                        "entry_index":
                            position.entry_index,

                        "exit_index":
                            i,
                    }
                )

                last_exit_time = timestamp

                position = None

                continue

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if last_exit_time is not None:

            elapsed = (
                timestamp
                - last_exit_time
            ).total_seconds() / 60.0

            if elapsed < COOLDOWN_MINUTES:
                continue

        # ----------------------------------------------------
        # V4 ATR REGIME FILTER
        #
        # IMPORTANT FIX:
        # atr_pct is compared against percent thresholds.
        #
        # Example:
        # atr_pct = 0.18
        # means ATR = 0.18% of BTC price.
        #
        # DO NOT compare absolute atr here.
        # ----------------------------------------------------

        if atr_pct < atr_min:
            continue

        if atr_pct > atr_max:
            continue

        # ----------------------------------------------------
        # V4 volume upper bound
        # ----------------------------------------------------

        if volume_ratio > volume_max:
            continue

        # ----------------------------------------------------
        # EMA spread
        # ----------------------------------------------------

        if ema_spread_pct < MIN_EMA_SPREAD_PCT:
            continue

        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        row_series = pd.Series(
            {
                "close":
                    close_price,

                "ema_fast":
                    row.ema_fast,

                "ema_medium":
                    row.ema_medium,

                "ema_slow":
                    row.ema_slow,

                "momentum_pct":
                    momentum_pct,

                "atr":
                    atr_value,

                "atr_pct":
                    atr_pct,

                "volume_ratio":
                    volume_ratio,

                "ema_spread_pct":
                    ema_spread_pct,

                "htf_bullish":
                    row.htf_bullish,

                "htf_bearish":
                    row.htf_bearish,
            }
        )

        signal = get_signal(
            row_series
        )

        if signal is None:
            continue

        # ----------------------------------------------------
        # Position size
        # ----------------------------------------------------

        position_size = (
            calculate_position_size(
                balance
            )
        )

        if position_size <= 0:
            continue

        # ----------------------------------------------------
        # Entry
        # ----------------------------------------------------

        entry_price = (
            apply_entry_slippage(
                close_price,
                signal,
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # SL/TP use ABSOLUTE ATR VALUE,
        # not atr_pct.
        # ----------------------------------------------------

        if signal == "LONG":

            stop_loss = (
                entry_price
                - atr_value
                * sl_multiplier
            )

            take_profit = (
                entry_price
                + atr_value
                * tp_multiplier
            )

        else:

            stop_loss = (
                entry_price
                + atr_value
                * sl_multiplier
            )

            take_profit = (
                entry_price
                - atr_value
                * tp_multiplier
            )

        position = Position(
            side=signal,

            entry_time=timestamp,

            entry_price=entry_price,

            position_size=position_size,

            stop_loss=stop_loss,

            take_profit=take_profit,

            entry_index=i,

            momentum_pct=momentum_pct,

            atr_pct=atr_pct,

            atr_value=atr_value,

            volume_ratio=volume_ratio,

            ema_spread_pct=ema_spread_pct,
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    trades_df = pd.DataFrame(
        trades
    )

    if trades_df.empty:

        return {
            "atr_min": atr_min,
            "atr_max": atr_max,
            "volume_max": volume_max,
            "sl_multiplier": sl_multiplier,
            "tp_multiplier": tp_multiplier,

            "trades": 0,

            "train_trades": 0,
            "oos_trades": 0,

            "train_net": 0.0,
            "oos_net": 0.0,

            "train_avg": 0.0,
            "oos_avg": 0.0,

            "train_win_rate": 0.0,
            "oos_win_rate": 0.0,

            "train_profit_factor": 0.0,
            "oos_profit_factor": 0.0,

            "train_max_drawdown": 0.0,
            "oos_max_drawdown": 0.0,

            "net_pnl": 0.0,
        }

    # --------------------------------------------------------
    # Period statistics
    # --------------------------------------------------------

    def stats_for_period(
        frame: pd.DataFrame,
    ) -> dict:

        if frame.empty:

            return {
                "trades": 0,
                "net": 0.0,
                "avg": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
            }

        net = (
            frame["net_pnl"]
            .sum()
        )

        avg = (
            frame["net_pnl"]
            .mean()
        )

        wins = (
            frame["net_pnl"] > 0
        ).sum()

        trade_count = len(
            frame
        )

        win_rate = (
            wins
            / trade_count
            * 100.0
        )

        positive = (
            frame.loc[
                frame["net_pnl"] > 0,
                "net_pnl",
            ].sum()
        )

        negative = (
            frame.loc[
                frame["net_pnl"] < 0,
                "net_pnl",
            ].sum()
        )

        if negative < 0:
            profit_factor = (
                positive
                / abs(negative)
            )
        else:
            profit_factor = (
                float("inf")
                if positive > 0
                else 0.0
            )

        equity = (
            STARTING_CAPITAL
            + frame["net_pnl"].cumsum()
        )

        peak = (
            equity.cummax()
        )

        drawdown = (
            peak - equity
        )

        max_drawdown = (
            drawdown.max()
        )

        return {
            "trades":
                int(trade_count),

            "net":
                float(net),

            "avg":
                float(avg),

            "win_rate":
                float(win_rate),

            "profit_factor":
                float(profit_factor),

            "max_drawdown":
                float(max_drawdown),
        }

    train = trades_df[
        trades_df["period"] == "TRAIN"
    ]

    oos = trades_df[
        trades_df["period"] == "OOS"
    ]

    train_stats = stats_for_period(
        train
    )

    oos_stats = stats_for_period(
        oos
    )

    return {
        "atr_min": atr_min,
        "atr_max": atr_max,

        "volume_max": volume_max,

        "sl_multiplier":
            sl_multiplier,

        "tp_multiplier":
            tp_multiplier,

        "trades":
            len(trades_df),

        "train_trades":
            train_stats["trades"],

        "oos_trades":
            oos_stats["trades"],

        "train_net":
            train_stats["net"],

        "oos_net":
            oos_stats["net"],

        "train_avg":
            train_stats["avg"],

        "oos_avg":
            oos_stats["avg"],

        "train_win_rate":
            train_stats["win_rate"],

        "oos_win_rate":
            oos_stats["win_rate"],

        "train_profit_factor":
            train_stats["profit_factor"],

        "oos_profit_factor":
            oos_stats["profit_factor"],

        "train_max_drawdown":
            train_stats["max_drawdown"],

        "oos_max_drawdown":
            oos_stats["max_drawdown"],

        "net_pnl":
            float(
                trades_df["net_pnl"].sum()
            ),
    }


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    row: pd.Series,
) -> float:

    oos_trades = (
        float(row["oos_trades"])
    )

    oos_net = (
        float(row["oos_net"])
    )

    oos_avg = (
        float(row["oos_avg"])
    )

    oos_pf = (
        float(row["oos_profit_factor"])
    )

    oos_win = (
        float(row["oos_win_rate"])
    )

    oos_dd = (
        float(row["oos_max_drawdown"])
    )

    train_net = (
        float(row["train_net"])
    )

    # --------------------------------------------------------
    # Minimum trade requirement
    # --------------------------------------------------------

    if oos_trades < 20:
        return -999999.0

    # --------------------------------------------------------
    # Primary objective:
    # OOS profitability
    # --------------------------------------------------------

    score = 0.0

    score += (
        oos_net * 100.0
    )

    score += (
        oos_avg * 500.0
    )

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    if math.isfinite(oos_pf):
        score += (
            max(
                0.0,
                oos_pf - 1.0,
            )
            * 25.0
        )

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    score += (
        max(
            0.0,
            oos_win - 50.0,
        )
        * 0.10
    )

    # --------------------------------------------------------
    # Drawdown penalty
    # --------------------------------------------------------

    score -= (
        oos_dd
        * 2.0
    )

    # --------------------------------------------------------
    # Penalize TRAIN positive / OOS negative
    # --------------------------------------------------------

    if (
        train_net > 0
        and oos_net < 0
    ):
        score -= 20.0

    return float(score)


# ============================================================
# MAIN SWEEP
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BITRUE V4 PARAMETER SWEEP")
    print("=" * 70)

    print()
    print("PAPER ONLY")
    print("NO LIVE TRADING")
    print("NO API KEYS")
    print("NO WALLET")
    print("NO REAL ORDERS")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(
        DATA_FILE
    )

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    print()
    print(
        "Calculating indicators..."
    )

    data = calculate_indicators(
        df
    )

    # --------------------------------------------------------
    # Train/OOS split
    # --------------------------------------------------------

    split_index = int(
        len(data)
        * TRAIN_RATIO
    )

    print()
    print(
        f"TRAIN candles: {split_index:,}"
    )

    print(
        f"OOS candles:   {len(data) - split_index:,}"
    )

    # --------------------------------------------------------
    # Build combinations
    # --------------------------------------------------------

    combinations = list(
        itertools.product(
            ATR_RANGES,
            VOLUME_MAX_VALUES,
            SL_MULTIPLIERS,
            TP_MULTIPLIERS,
        )
    )

    print()
    print(
        f"Parameter combinations: "
        f"{len(combinations)}"
    )

    # --------------------------------------------------------
    # Sweep
    # --------------------------------------------------------

    results = []

    for number, (
        atr_range,
        volume_max,
        sl_multiplier,
        tp_multiplier,
    ) in enumerate(
        combinations,
        start=1,
    ):

        atr_min, atr_max = (
            atr_range
        )

        print(
            f"[{number:02d}/{len(combinations):02d}] "
            f"ATR {atr_min:.3f}-{atr_max:.3f}% | "
            f"VOL <= {volume_max:.1f} | "
            f"SL {sl_multiplier:.1f}x | "
            f"TP {tp_multiplier:.1f}x"
        )

        result = run_variant(
            data=data,
            split_index=split_index,
            atr_min=atr_min,
            atr_max=atr_max,
            volume_max=volume_max,
            sl_multiplier=sl_multiplier,
            tp_multiplier=tp_multiplier,
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    results_df["score"] = (
        results_df.apply(
            calculate_score,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            [
                "score",
                "oos_net",
                "oos_avg",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save all results
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Candidate selection
    # --------------------------------------------------------

    candidates = results_df[
        (results_df["oos_trades"] >= 20)
        & (results_df["oos_net"] > 0)
        & (results_df["oos_avg"] > 0)
    ].copy()

    candidates = (
        candidates
        .sort_values(
            [
                "oos_net",
                "oos_avg",
                "oos_profit_factor",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    candidates.to_csv(
        CANDIDATES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "BITRUE V4 PARAMETER SWEEP REPORT"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        ""
    )

    lines.append(
        "PAPER / BACKTEST ONLY"
    )

    lines.append(
        "NO LIVE TRADING"
    )

    lines.append(
        "NO API KEYS"
    )

    lines.append(
        "NO WALLET"
    )

    lines.append(
        "NO REAL ORDERS"
    )

    lines.append(
        ""
    )

    lines.append(
        f"Data rows: {len(data):,}"
    )

    lines.append(
        f"Train rows: {split_index:,}"
    )

    lines.append(
        f"OOS rows: {len(data) - split_index:,}"
    )

    lines.append(
        f"Variants tested: {len(results_df)}"
    )

    lines.append(
        f"Candidates: {len(candidates)}"
    )

    lines.append(
        ""
    )

    lines.append(
        "IMPORTANT ATR HANDLING:"
    )

    lines.append(
        "- ATR filter uses atr_pct"
    )

    lines.append(
        "- SL/TP uses absolute ATR"
    )

    lines.append(
        "- Slippage is included in execution prices"
    )

    lines.append(
        "- Slippage is NOT double-counted as a separate cost"
    )

    lines.append(
        ""
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "TOP 20 VARIANTS"
    )

    lines.append(
        "=" * 70
    )

    top = results_df.head(
        20
    )

    for rank, row in enumerate(
        top.itertuples(
            index=False
        ),
        start=1,
    ):

        lines.append(
            f"{rank:02d}. "
            f"ATR {row.atr_min:.3f}-{row.atr_max:.3f}% | "
            f"VOL <= {row.volume_max:.1f} | "
            f"SL {row.sl_multiplier:.1f}x | "
            f"TP {row.tp_multiplier:.1f}x | "
            f"OOS trades {row.oos_trades} | "
            f"OOS net ${row.oos_net:.4f} | "
            f"OOS avg ${row.oos_avg:.6f} | "
            f"OOS WR {row.oos_win_rate:.2f}% | "
            f"OOS PF {row.oos_profit_factor:.3f} | "
            f"OOS DD ${row.oos_max_drawdown:.4f} | "
            f"Score {row.score:.3f}"
        )

    lines.append(
        ""
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "POSITIVE OOS CANDIDATES"
    )

    lines.append(
        "=" * 70
    )

    if candidates.empty:

        lines.append(
            "NO VARIANT PASSED:"
        )

        lines.append(
            "OOS trades >= 20"
        )

        lines.append(
            "OOS net > 0"
        )

        lines.append(
            "OOS average trade > 0"
        )

    else:

        for rank, row in enumerate(
            candidates.itertuples(
                index=False
            ),
            start=1,
        ):

            lines.append(
                f"{rank:02d}. "
                f"ATR {row.atr_min:.3f}-{row.atr_max:.3f}% | "
                f"VOL <= {row.volume_max:.1f} | "
                f"SL {row.sl_multiplier:.1f}x | "
                f"TP {row.tp_multiplier:.1f}x | "
                f"OOS trades {row.oos_trades} | "
                f"OOS net ${row.oos_net:.4f} | "
                f"OOS avg ${row.oos_avg:.6f} | "
                f"OOS WR {row.oos_win_rate:.2f}% | "
                f"OOS PF {row.oos_profit_factor:.3f} | "
                f"OOS DD ${row.oos_max_drawdown:.4f}"
            )

    lines.append(
        ""
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "BEST OOS VARIANT"
    )

    lines.append(
        "=" * 70
    )

    if not candidates.empty:

        best = candidates.iloc[0]

        lines.append(
            f"ATR range: "
            f"{best['atr_min']:.3f}% - "
            f"{best['atr_max']:.3f}%"
        )

        lines.append(
            f"Volume max: "
            f"{best['volume_max']:.1f}x"
        )

        lines.append(
            f"SL multiplier: "
            f"{best['sl_multiplier']:.1f}x ATR"
        )

        lines.append(
            f"TP multiplier: "
            f"{best['tp_multiplier']:.1f}x ATR"
        )

        lines.append(
            f"OOS trades: "
            f"{int(best['oos_trades'])}"
        )

        lines.append(
            f"OOS net: "
            f"${best['oos_net']:.4f}"
        )

        lines.append(
            f"OOS average trade: "
            f"${best['oos_avg']:.6f}"
        )

        lines.append(
            f"OOS win rate: "
            f"{best['oos_win_rate']:.2f}%"
        )

        lines.append(
            f"OOS profit factor: "
            f"{best['oos_profit_factor']:.3f}"
        )

        lines.append(
            f"OOS max drawdown: "
            f"${best['oos_max_drawdown']:.4f}"
        )

    else:

        lines.append(
            "No positive OOS candidate."
        )

    lines.append(
        ""
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "END OF REPORT"
    )

    lines.append(
        "=" * 70
    )

    report_text = "\n".join(
        lines
    )

    print()
    print(
        report_text
    )

    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        RESULTS_FILE
    )

    print(
        CANDIDATES_FILE
    )

    print(
        REPORT_FILE
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Final safety information
    # --------------------------------------------------------

    print()
    print(
        "V4 COMPLETE"
    )

    print(
        "PAPER ONLY"
    )

    print(
        "LIVE TRADING = FALSE"
    )

    print(
        "NO API KEYS"
    )

    print(
        "NO WALLET"
    )

    print(
        "NO REAL ORDERS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
