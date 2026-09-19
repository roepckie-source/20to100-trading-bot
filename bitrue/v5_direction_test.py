"""
Bitrue BTC/USDT 5m Strategy - V5 Direction Test

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO WALLET
NO REAL ORDERS

V5 PURPOSE
----------
Compare:

1. LONG ONLY
2. SHORT ONLY
3. LONG + SHORT

using the best V4 parameter region without
performing another parameter optimization.

V4 parameters used:

ATR:        0.175% - 0.200%
Volume:     <= 3.0x
SL:         1.4x ATR
TP:         2.5x ATR

Data:
BTCUSDT_5m.csv

Split:
70% TRAIN
30% OOS
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np


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

TRAIN_RATIO = 0.70

COOLDOWN_MINUTES = 30

# ------------------------------------------------------------
# V4 BEST REGION
# ------------------------------------------------------------

ATR_MIN = 0.175
ATR_MAX = 0.200

VOLUME_MIN = 1.20
VOLUME_MAX = 3.00

EMA_SPREAD_MIN = 0.15
MOMENTUM_MIN = 0.15

SL_ATR_MULTIPLIER = 1.4
TP_ATR_MULTIPLIER = 2.5


# ============================================================
# POSITION
# ============================================================

@dataclass
class Position:

    side: str

    entry_time: pd.Timestamp
    entry_index: int

    entry_price: float
    position_size: float

    stop_loss: float
    take_profit: float

    momentum_pct: float
    atr_pct: float
    atr_value: float

    volume_ratio: float
    ema_spread_pct: float


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Missing data file: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    timestamp_candidates = [
        "open_time",
        "timestamp",
        "datetime",
        "date",
        "time",
    ]

    timestamp_column = None

    for column in timestamp_candidates:

        if column in df.columns:

            timestamp_column = column
            break

    if timestamp_column is None:

        raise ValueError(
            "No timestamp column found."
        )

    df["timestamp"] = pd.to_datetime(
        df[timestamp_column],
        utc=True,
        errors="coerce",
    )

    rename = {}

    for column in df.columns:

        name = column.lower()

        if name == "open":
            rename[column] = "open"

        elif name == "high":
            rename[column] = "high"

        elif name == "low":
            rename[column] = "low"

        elif name == "close":
            rename[column] = "close"

        elif name == "volume":
            rename[column] = "volume"

    df = df.rename(
        columns=rename
    )

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.reset_index(
        drop=True
    )

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
    # ATR
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

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(
        axis=1
    )

    data["atr"] = (
        true_range
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    data["atr_pct"] = (
        data["atr"]
        / data["close"]
        * 100.0
    )

    # --------------------------------------------------------
    # Volume
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
    # 1H higher timeframe
    # --------------------------------------------------------

    h1 = (
        data
        .set_index("timestamp")
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
    # Prevent lookahead
    # --------------------------------------------------------

    h1["ema20"] = h1["ema20"].shift(1)
    h1["ema50"] = h1["ema50"].shift(1)

    h1 = h1[
        [
            "ema20",
            "ema50",
        ]
    ].reset_index()

    data = pd.merge_asof(
        data.sort_values(
            "timestamp"
        ),
        h1.sort_values(
            "timestamp"
        ),
        on="timestamp",
        direction="backward",
    )

    data["htf_bullish"] = (
        data["ema20"]
        > data["ema50"]
    )

    data["htf_bearish"] = (
        data["ema20"]
        < data["ema50"]
    )

    return data


# ============================================================
# POSITION SIZE
# ============================================================

def position_size(
    balance: float,
) -> float:

    size = (
        balance
        * RISK_PER_TRADE
    )

    return min(
        MAX_POSITION_USD,
        max(
            MIN_POSITION_USD,
            size,
        ),
    )


# ============================================================
# EXECUTION
# ============================================================

def entry_price(
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


def exit_price(
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
# SIGNAL
# ============================================================

def get_signal(
    row,
    direction_mode: str,
):

    required = [
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
        for value in required
    ):

        return None

    close = float(
        row["close"]
    )

    momentum = float(
        row["momentum_pct"]
    )

    atr_pct = float(
        row["atr_pct"]
    )

    volume_ratio = float(
        row["volume_ratio"]
    )

    spread = float(
        row["ema_spread_pct"]
    )

    # --------------------------------------------------------
    # Common filters
    # --------------------------------------------------------

    if not (
        ATR_MIN
        <= atr_pct
        <= ATR_MAX
    ):

        return None

    if not (
        VOLUME_MIN
        <= volume_ratio
        <= VOLUME_MAX
    ):

        return None

    if spread < EMA_SPREAD_MIN:

        return None

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    long_signal = (
        momentum >= MOMENTUM_MIN
        and
        row["ema_fast"]
        > row["ema_medium"]
        > row["ema_slow"]
        and
        close
        > row["ema_fast"]
        and
        bool(row["htf_bullish"])
    )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    short_signal = (
        momentum <= -MOMENTUM_MIN
        and
        row["ema_fast"]
        < row["ema_medium"]
        < row["ema_slow"]
        and
        close
        < row["ema_fast"]
        and
        bool(row["htf_bearish"])
    )

    # --------------------------------------------------------
    # Direction control
    # --------------------------------------------------------

    if (
        direction_mode
        == "LONG_ONLY"
    ):

        if long_signal:

            return "LONG"

        return None

    if (
        direction_mode
        == "SHORT_ONLY"
    ):

        if short_signal:

            return "SHORT"

        return None

    if (
        direction_mode
        == "BOTH"
    ):

        if long_signal:

            return "LONG"

        if short_signal:

            return "SHORT"

    return None


# ============================================================
# TRADE RESULT
# ============================================================

def trade_result(
    position: Position,
    exit_exec: float,
):

    quantity = (
        position.position_size
        / position.entry_price
    )

    if position.side == "LONG":

        gross = (
            exit_exec
            - position.entry_price
        ) * quantity

    else:

        gross = (
            position.entry_price
            - exit_exec
        ) * quantity

    entry_notional = (
        position.position_size
    )

    exit_notional = abs(
        position.position_size
        + gross
    )

    fees = (
        entry_notional
        + exit_notional
    ) * FEE_RATE

    # Slippage is already reflected
    # in execution prices.
    slippage_cost = 0.0

    net = (
        gross
        - fees
    )

    return (
        gross,
        fees,
        slippage_cost,
        net,
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    data: pd.DataFrame,
    direction_mode: str,
):

    split_index = int(
        len(data)
        * TRAIN_RATIO
    )

    balance = STARTING_CAPITAL

    position = None

    last_exit_time = None

    trades = []

    for i, row in enumerate(
        data.itertuples(
            index=False
        )
    ):

        timestamp = row.timestamp

        open_price = float(
            row.open
        )

        high = float(
            row.high
        )

        low = float(
            row.low
        )

        close = float(
            row.close
        )

        # ----------------------------------------------------
        # Manage position
        # ----------------------------------------------------

        if position is not None:

            reason = None
            raw_exit = None

            if position.side == "LONG":

                # Conservative:
                # stop first

                if low <= position.stop_loss:

                    reason = "STOP_LOSS"

                    raw_exit = (
                        position.stop_loss
                    )

                elif (
                    high
                    >= position.take_profit
                ):

                    reason = "TAKE_PROFIT"

                    raw_exit = (
                        position.take_profit
                    )

            else:

                if (
                    high
                    >= position.stop_loss
                ):

                    reason = "STOP_LOSS"

                    raw_exit = (
                        position.stop_loss
                    )

                elif (
                    low
                    <= position.take_profit
                ):

                    reason = "TAKE_PROFIT"

                    raw_exit = (
                        position.take_profit
                    )

            if reason is not None:

                execution = exit_price(
                    raw_exit,
                    position.side,
                )

                (
                    gross,
                    fees,
                    slippage,
                    net,
                ) = trade_result(
                    position,
                    execution,
                )

                balance += net

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
                            execution,

                        "position_size":
                            position.position_size,

                        "gross_pnl":
                            gross,

                        "fees":
                            fees,

                        "slippage":
                            slippage,

                        "net_pnl":
                            net,

                        "balance":
                            balance,

                        "momentum_pct":
                            position.momentum_pct,

                        "atr_pct":
                            position.atr_pct,

                        "volume_ratio":
                            position.volume_ratio,

                        "ema_spread_pct":
                            position.ema_spread_pct,

                        "exit_reason":
                            reason,

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

            minutes = (
                timestamp
                - last_exit_time
            ).total_seconds() / 60.0

            if minutes < COOLDOWN_MINUTES:

                continue

        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        signal = get_signal(
            pd.Series(
                {
                    "close":
                        close,

                    "ema_fast":
                        row.ema_fast,

                    "ema_medium":
                        row.ema_medium,

                    "ema_slow":
                        row.ema_slow,

                    "momentum_pct":
                        row.momentum_pct,

                    "atr":
                        row.atr,

                    "atr_pct":
                        row.atr_pct,

                    "volume_ratio":
                        row.volume_ratio,

                    "ema_spread_pct":
                        row.ema_spread_pct,

                    "htf_bullish":
                        row.htf_bullish,

                    "htf_bearish":
                        row.htf_bearish,
                }
            ),
            direction_mode,
        )

        if signal is None:

            continue

        atr = float(
            row.atr
        )

        atr_pct = float(
            row.atr_pct
        )

        volume_ratio = float(
            row.volume_ratio
        )

        spread = float(
            row.ema_spread_pct
        )

        momentum = float(
            row.momentum_pct
        )

        size = position_size(
            balance
        )

        execution = entry_price(
            close,
            signal,
        )

        # ----------------------------------------------------
        # ATR SL / TP
        # ----------------------------------------------------

        if signal == "LONG":

            stop = (
                execution
                - atr
                * SL_ATR_MULTIPLIER
            )

            target = (
                execution
                + atr
                * TP_ATR_MULTIPLIER
            )

        else:

            stop = (
                execution
                + atr
                * SL_ATR_MULTIPLIER
            )

            target = (
                execution
                - atr
                * TP_ATR_MULTIPLIER
            )

        position = Position(
            side=signal,

            entry_time=timestamp,

            entry_index=i,

            entry_price=execution,

            position_size=size,

            stop_loss=stop,

            take_profit=target,

            momentum_pct=momentum,

            atr_pct=atr_pct,

            atr_value=atr,

            volume_ratio=volume_ratio,

            ema_spread_pct=spread,
        )

    return pd.DataFrame(
        trades
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(
    trades: pd.DataFrame,
):

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross": 0.0,
            "fees": 0.0,
            "slippage": 0.0,
            "net": 0.0,
            "avg_trade": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    wins = int(
        (
            trades["net_pnl"]
            > 0
        ).sum()
    )

    losses = int(
        (
            trades["net_pnl"]
            <= 0
        ).sum()
    )

    count = len(
        trades
    )

    win_rate = (
        wins
        / count
        * 100.0
    )

    gross = float(
        trades["gross_pnl"]
        .sum()
    )

    fees = float(
        trades["fees"]
        .sum()
    )

    slippage = float(
        trades["slippage"]
        .sum()
    )

    net = float(
        trades["net_pnl"]
        .sum()
    )

    avg_trade = (
        net
        / count
    )

    positive = float(
        trades.loc[
            trades["net_pnl"] > 0,
            "net_pnl",
        ].sum()
    )

    negative = float(
        trades.loc[
            trades["net_pnl"] < 0,
            "net_pnl",
        ].sum()
    )

    if negative < 0:

        profit_factor = (
            positive
            / abs(negative)
        )

    else:

        profit_factor = 0.0

    equity = (
        STARTING_CAPITAL
        + trades["net_pnl"].cumsum()
    )

    peak = (
        equity.cummax()
    )

    drawdown = (
        peak
        - equity
    )

    max_drawdown = float(
        drawdown.max()
    )

    return {
        "trades": count,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "gross": gross,
        "fees": fees,
        "slippage": slippage,
        "net": net,
        "avg_trade": avg_trade,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


# ============================================================
# REPORT
# ============================================================

def analyze_direction(
    trades: pd.DataFrame,
    direction: str,
):

    if trades.empty:

        return None

    train = trades[
        trades["period"]
        == "TRAIN"
    ]

    oos = trades[
        trades["period"]
        == "OOS"
    ]

    total = calculate_stats(
        trades
    )

    train_stats = calculate_stats(
        train
    )

    oos_stats = calculate_stats(
        oos
    )

    return {
        "direction": direction,

        "total":
            total,

        "train":
            train_stats,

        "oos":
            oos_stats,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BITRUE V5 - LONG VS SHORT")
    print("=" * 70)

    print()
    print("PAPER ONLY")
    print("NO LIVE TRADING")
    print("NO API KEYS")
    print("NO WALLET")
    print("NO REAL ORDERS")

    print()
    print("V4 PARAMETERS")
    print(
        f"ATR:        {ATR_MIN:.3f}% - "
        f"{ATR_MAX:.3f}%"
    )

    print(
        f"Volume:     "
        f"{VOLUME_MIN:.1f}x - "
        f"{VOLUME_MAX:.1f}x"
    )

    print(
        f"SL:         "
        f"{SL_ATR_MULTIPLIER:.1f}x ATR"
    )

    print(
        f"TP:         "
        f"{TP_ATR_MULTIPLIER:.1f}x ATR"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data(
        DATA_FILE
    )

    data = calculate_indicators(
        data
    )

    print()
    print(
        f"Candles: {len(data):,}"
    )

    print(
        f"TRAIN:   "
        f"{int(len(data) * TRAIN_RATIO):,}"
    )

    print(
        f"OOS:     "
        f"{len(data) - int(len(data) * TRAIN_RATIO):,}"
    )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    modes = [
        "LONG_ONLY",
        "SHORT_ONLY",
        "BOTH",
    ]

    all_results = []

    report_lines = []

    report_lines.append(
        "BITRUE V5 - LONG VS SHORT"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "PAPER / BACKTEST ONLY"
    )

    report_lines.append(
        "NO LIVE TRADING"
    )

    report_lines.append(
        "NO API KEYS"
    )

    report_lines.append(
        "NO WALLET"
    )

    report_lines.append(
        "NO REAL ORDERS"
    )

    report_lines.append(
        ""
    )

    report_lines.append(
        "V4 PARAMETERS"
    )

    report_lines.append(
        f"ATR {ATR_MIN:.3f}-{ATR_MAX:.3f}%"
    )

    report_lines.append(
        f"Volume {VOLUME_MIN:.1f}-{VOLUME_MAX:.1f}x"
    )

    report_lines.append(
        f"SL {SL_ATR_MULTIPLIER:.1f}x ATR"
    )

    report_lines.append(
        f"TP {TP_ATR_MULTIPLIER:.1f}x ATR"
    )

    report_lines.append(
        ""
    )

    for mode in modes:

        print()
        print("=" * 70)
        print(mode)
        print("=" * 70)

        trades = run_backtest(
            data,
            mode,
        )

        if not trades.empty:

            filename = (
                f"bitrue_v5_"
                f"{mode.lower()}_trades.csv"
            )

            trades.to_csv(
                filename,
                index=False,
            )

        result = analyze_direction(
            trades,
            mode,
        )

        all_results.append(
            result
        )

        if result is None:

            print(
                "NO TRADES"
            )

            report_lines.append(
                f"{mode}: NO TRADES"
            )

            continue

        for period in [
            "total",
            "train",
            "oos",
        ]:

            stats = result[
                period
            ]

            label = (
                period.upper()
            )

            print()
            print(
                f"{label}:"
            )

            print(
                f"Trades:       "
                f"{stats['trades']}"
            )

            print(
                f"Win rate:     "
                f"{stats['win_rate']:.2f}%"
            )

            print(
                f"Gross P&L:    "
                f"${stats['gross']:.4f}"
            )

            print(
                f"Fees:         "
                f"${stats['fees']:.4f}"
            )

            print(
                f"Net P&L:      "
                f"${stats['net']:.4f}"
            )

            print(
                f"Avg trade:    "
                f"${stats['avg_trade']:.6f}"
            )

            print(
                f"Profit factor:"
                f" {stats['profit_factor']:.3f}"
            )

            print(
                f"Max DD:       "
                f"${stats['max_drawdown']:.4f}"
            )

            report_lines.append(
                f"{mode} {label}: "
                f"Trades={stats['trades']} | "
                f"Win={stats['win_rate']:.2f}% | "
                f"Gross=${stats['gross']:.4f} | "
                f"Fees=${stats['fees']:.4f} | "
                f"Net=${stats['net']:.4f} | "
                f"Avg=${stats['avg_trade']:.6f} | "
                f"PF={stats['profit_factor']:.3f} | "
                f"DD=${stats['max_drawdown']:.4f}"
            )

    # --------------------------------------------------------
    # Direction comparison
    # --------------------------------------------------------

    report_lines.append(
        ""
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "DIRECTION COMPARISON"
    )

    report_lines.append(
        "=" * 70
    )

    for result in all_results:

        if result is None:

            continue

        oos = result["oos"]

        report_lines.append(
            f"{result['direction']}: "
            f"OOS Trades={oos['trades']} | "
            f"OOS Win={oos['win_rate']:.2f}% | "
            f"OOS Net=${oos['net']:.4f} | "
            f"OOS Avg=${oos['avg_trade']:.6f} | "
            f"OOS PF={oos['profit_factor']:.3f}"
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_file = Path(
        "bitrue_v5_direction_report.txt"
    )

    report_file.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Combined summary CSV
    # --------------------------------------------------------

    summary_rows = []

    for result in all_results:

        if result is None:

            continue

        for period in [
            "total",
            "train",
            "oos",
        ]:

            stats = result[
                period
            ]

            summary_rows.append(
                {
                    "direction":
                        result[
                            "direction"
                        ],

                    "period":
                        period.upper(),

                    "trades":
                        stats[
                            "trades"
                        ],

                    "wins":
                        stats[
                            "wins"
                        ],

                    "losses":
                        stats[
                            "losses"
                        ],

                    "win_rate":
                        stats[
                            "win_rate"
                        ],

                    "gross_pnl":
                        stats[
                            "gross"
                        ],

                    "fees":
                        stats[
                            "fees"
                        ],

                    "slippage":
                        stats[
                            "slippage"
                        ],

                    "net_pnl":
                        stats[
                            "net"
                        ],

                    "avg_trade":
                        stats[
                            "avg_trade"
                        ],

                    "profit_factor":
                        stats[
                            "profit_factor"
                        ],

                    "max_drawdown":
                        stats[
                            "max_drawdown"
                        ],
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        "bitrue_v5_direction_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V5 COMPLETE")
    print("=" * 70)

    print(
        "Created:"
    )

    print(
        "- bitrue_v5_direction_report.txt"
    )

    print(
        "- bitrue_v5_direction_summary.csv"
    )

    print(
        "- bitrue_v5_long_only_trades.csv"
    )

    print(
        "- bitrue_v5_short_only_trades.csv"
    )

    print(
        "- bitrue_v5_both_trades.csv"
    )

    print()
    print(
        "PAPER ONLY"
    )

    print(
        "LIVE TRADING = FALSE"
    )


if __name__ == "__main__":
    main()
