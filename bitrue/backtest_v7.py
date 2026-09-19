"""
Bitrue V7 Regime Backtest

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO WALLET
NO REAL ORDERS
"""

from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from v7_regime_strategy import (
    calculate_indicators,
    evaluate_precomputed,
)


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("BTCUSDT_5m.csv")
TRADE_LOG = Path("bitrue_v7_trade_log.csv")
REPORT_FILE = Path("bitrue_v7_report.txt")

STARTING_CAPITAL = 100.00

FEE_RATE = 0.0006
SLIPPAGE_RATE = 0.0002

COOLDOWN_MINUTES = 30

TRAIN_RATIO = 0.70


# ============================================================
# DATA
# ============================================================

def load_data(path: Path = DATA_FILE) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        "timestamp"
    ).drop_duplicates(
        subset=["timestamp"]
    ).reset_index(drop=True)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# TRADE
# ============================================================

@dataclass
class Position:

    side: str
    entry_time: pd.Timestamp
    entry_price: float

    position_size: float

    stop_loss: float
    take_profit: float

    momentum_pct: float
    atr_pct: float
    volume_ratio: float
    ema_spread_pct: float

    regime: str


# ============================================================
# EXECUTION
# ============================================================

def execute_entry_price(
    price: float,
    side: str
) -> float:

    if side == "LONG":

        return price * (
            1.0 + SLIPPAGE_RATE
        )

    return price * (
        1.0 - SLIPPAGE_RATE
    )


def execute_exit_price(
    price: float,
    side: str
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

def calculate_gross_pnl(
    side: str,
    entry_price: float,
    exit_price: float,
    position_size: float
) -> float:

    if entry_price <= 0:
        return 0.0

    if side == "LONG":

        return (
            (exit_price - entry_price)
            / entry_price
            * position_size
        )

    return (
        (entry_price - exit_price)
        / entry_price
        * position_size
    )


# ============================================================
# STATS
# ============================================================

def calculate_stats(
    trades: pd.DataFrame,
    starting_balance: float
) -> dict:

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "net_pnl": 0.0,
            "avg_trade": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    wins = (
        trades["net_pnl"] > 0
    ).sum()

    losses = (
        trades["net_pnl"] <= 0
    ).sum()

    gross_profit = trades.loc[
        trades["net_pnl"] > 0,
        "net_pnl"
    ].sum()

    gross_loss = abs(
        trades.loc[
            trades["net_pnl"] < 0,
            "net_pnl"
        ].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    equity = (
        starting_balance
        + trades["net_pnl"].cumsum()
    )

    running_max = equity.cummax()

    drawdown = (
        running_max - equity
    )

    max_drawdown = (
        drawdown.max()
        if len(drawdown)
        else 0.0
    )

    return {
        "trades": len(trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": (
            wins / len(trades) * 100
        ),
        "gross_pnl": trades["gross_pnl"].sum(),
        "fees": trades["fees"].sum(),
        "net_pnl": trades["net_pnl"].sum(),
        "avg_trade": trades["net_pnl"].mean(),
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df: pd.DataFrame
) -> pd.DataFrame:

    data = calculate_indicators(df)

    split_index = int(
        len(data) * TRAIN_RATIO
    )

    data["period"] = "TRAIN"

    data.loc[
        data.index >= split_index,
        "period"
    ] = "OOS"

    balance = STARTING_CAPITAL

    position = None

    last_exit_time = None

    trades = []

    for row in data.itertuples():

        timestamp = row.timestamp

        # ====================================================
        # MANAGE EXISTING POSITION
        # ====================================================

        if position is not None:

            exit_price = None
            exit_reason = None

            # ------------------------------------------------
            # LONG
            # ------------------------------------------------

            if position.side == "LONG":

                # Conservative:
                # STOP checked BEFORE TP
                if row.low <= position.stop_loss:

                    exit_price = position.stop_loss
                    exit_reason = "STOP_LOSS"

                elif row.high >= position.take_profit:

                    exit_price = position.take_profit
                    exit_reason = "TAKE_PROFIT"

            # ------------------------------------------------
            # SHORT
            # ------------------------------------------------

            elif position.side == "SHORT":

                # Conservative:
                # STOP checked BEFORE TP
                if row.high >= position.stop_loss:

                    exit_price = position.stop_loss
                    exit_reason = "STOP_LOSS"

                elif row.low <= position.take_profit:

                    exit_price = position.take_profit
                    exit_reason = "TAKE_PROFIT"

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if exit_price is not None:

                executed_exit = execute_exit_price(
                    exit_price,
                    position.side
                )

                gross_pnl = calculate_gross_pnl(
                    position.side,
                    position.entry_price,
                    executed_exit,
                    position.position_size
                )

                # Both entry and exit notional
                total_notional = (
                    position.position_size * 2
                )

                fees = (
                    total_notional
                    * FEE_RATE
                )

                net_pnl = (
                    gross_pnl
                    - fees
                )

                balance += net_pnl

                duration_minutes = (
                    timestamp
                    - position.entry_time
                ).total_seconds() / 60.0

                trades.append({

                    "entry_time":
                        position.entry_time,

                    "exit_time":
                        timestamp,

                    "period":
                        row.period,

                    "side":
                        position.side,

                    "regime":
                        position.regime,

                    "entry_price":
                        position.entry_price,

                    "exit_price":
                        executed_exit,

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

                    "net_pnl":
                        net_pnl,

                    "balance":
                        balance,

                    "duration_minutes":
                        duration_minutes,

                    "momentum_pct":
                        position.momentum_pct,

                    "atr_pct":
                        position.atr_pct,

                    "volume_ratio":
                        position.volume_ratio,

                    "ema_spread_pct":
                        position.ema_spread_pct,

                    "exit_reason":
                        exit_reason,
                })

                last_exit_time = timestamp

                position = None

                continue

        # ====================================================
        # COOLDOWN
        # ====================================================

        if last_exit_time is not None:

            minutes_since_exit = (
                timestamp - last_exit_time
            ).total_seconds() / 60.0

            if minutes_since_exit < COOLDOWN_MINUTES:
                continue

        # ====================================================
        # NEW SIGNAL
        # ====================================================

        signal = evaluate_precomputed(
            row,
            balance
        )

        if signal.signal not in [
            "BUY",
            "SELL"
        ]:
            continue

        if signal.side is None:
            continue

        # ====================================================
        # ENTRY
        # ====================================================

        entry_price = execute_entry_price(
            signal.price,
            signal.side
        )

        position = Position(

            side=signal.side,

            entry_time=timestamp,

            entry_price=entry_price,

            position_size=signal.position_size,

            stop_loss=signal.stop_loss,

            take_profit=signal.take_profit,

            momentum_pct=signal.momentum_pct,

            atr_pct=signal.atr_pct,

            volume_ratio=signal.volume_ratio,

            ema_spread_pct=signal.ema_spread_pct,

            regime=signal.regime,
        )

    return pd.DataFrame(trades)


# ============================================================
# REPORT
# ============================================================

def format_stats(
    name: str,
    stats: dict
) -> str:

    return f"""
{name}
------------------------------------------------------------
Trades:          {stats['trades']}
Wins:            {stats['wins']}
Losses:          {stats['losses']}
Win Rate:        {stats['win_rate']:.2f}%
Gross P&L:       ${stats['gross_pnl']:.4f}
Fees:            ${stats['fees']:.4f}
Net P&L:         ${stats['net_pnl']:.4f}
Avg Trade:       ${stats['avg_trade']:.6f}
Profit Factor:   {stats['profit_factor']:.3f}
Max Drawdown:    ${stats['max_drawdown']:.4f}
"""


def build_report(
    df: pd.DataFrame,
    trades: pd.DataFrame
) -> str:

    report = []

    report.append(
        "BITRUE V7 - LONG/SHORT REGIME STRATEGY"
    )

    report.append("=" * 60)

    report.append(
        "PAPER / BACKTEST ONLY"
    )

    report.append(
        "NO LIVE TRADING"
    )

    report.append(
        "NO API KEYS"
    )

    report.append(
        "NO WALLET"
    )

    report.append(
        "NO REAL ORDERS"
    )

    report.append("")

    report.append(
        f"Data candles: {len(df):,}"
    )

    report.append(
        f"Data start: {df['timestamp'].min()}"
    )

    report.append(
        f"Data end:   {df['timestamp'].max()}"
    )

    report.append("")

    report.append("V7 FIXED PARAMETERS")
    report.append("-" * 60)

    report.append(
        "LONG: bullish 5m + bullish 1h"
    )

    report.append(
        "SHORT: bearish 5m + bearish 1h"
    )

    report.append(
        "Neutral regime: NO TRADE"
    )

    report.append(
        "Momentum minimum: 0.15%"
    )

    report.append(
        "ATR range: 0.10% - 0.30%"
    )

    report.append(
        "Volume: 1.20x - 5.00x"
    )

    report.append(
        "EMA spread minimum: 0.15%"
    )

    report.append(
        "Stop: 1.4x ATR"
    )

    report.append(
        "Take profit: 2.5x ATR"
    )

    report.append(
        "Cooldown: 30 minutes"
    )

    report.append("")

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total_stats = calculate_stats(
        trades,
        STARTING_CAPITAL
    )

    report.append(
        format_stats(
            "TOTAL",
            total_stats
        )
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train = trades[
        trades["period"] == "TRAIN"
    ]

    train_stats = calculate_stats(
        train,
        STARTING_CAPITAL
    )

    report.append(
        format_stats(
            "TRAIN",
            train_stats
        )
    )

    # --------------------------------------------------------
    # OOS
    # --------------------------------------------------------

    oos = trades[
        trades["period"] == "OOS"
    ]

    oos_stats = calculate_stats(
        oos,
        STARTING_CAPITAL
    )

    report.append(
        format_stats(
            "OOS",
            oos_stats
        )
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    long_trades = trades[
        trades["side"] == "LONG"
    ]

    long_stats = calculate_stats(
        long_trades,
        STARTING_CAPITAL
    )

    report.append(
        format_stats(
            "LONG TOTAL",
            long_stats
        )
    )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    short_trades = trades[
        trades["side"] == "SHORT"
    ]

    short_stats = calculate_stats(
        short_trades,
        STARTING_CAPITAL
    )

    report.append(
        format_stats(
            "SHORT TOTAL",
            short_stats
        )
    )

    # --------------------------------------------------------
    # OOS LONG
    # --------------------------------------------------------

    oos_long = trades[
        (trades["period"] == "OOS")
        &
        (trades["side"] == "LONG")
    ]

    oos_long_stats = calculate_stats(
        oos_long,
        STARTING_CAPITAL
    )

    report.append(
        format_stats(
            "OOS LONG",
            oos_long_stats
        )
    )

    # --------------------------------------------------------
    # OOS SHORT
    # --------------------------------------------------------

    oos_short = trades[
        (trades["period"] == "OOS")
        &
        (trades["side"] == "SHORT")
    ]

    oos_short_stats = calculate_stats(
        oos_short,
        STARTING_CAPITAL
    )

    report.append(
        format_stats(
            "OOS SHORT",
            oos_short_stats
        )
    )

    # --------------------------------------------------------
    # EXIT DISTRIBUTION
    # --------------------------------------------------------

    if not trades.empty:

        report.append("")
        report.append("EXIT DISTRIBUTION")
        report.append("-" * 60)

        exits = trades[
            "exit_reason"
        ].value_counts()

        for reason, count in exits.items():

            report.append(
                f"{reason}: {count}"
            )

        # ----------------------------------------------------
        # REGIME DISTRIBUTION
        # ----------------------------------------------------

        report.append("")
        report.append("REGIME / DIRECTION")
        report.append("-" * 60)

        regime_counts = (
            trades.groupby(
                ["period", "side"]
            )
            .size()
        )

        for index, count in regime_counts.items():

            report.append(
                f"{index}: {count}"
            )

    # --------------------------------------------------------
    # FINAL DIAGNOSIS
    # --------------------------------------------------------

    report.append("")
    report.append("=" * 60)
    report.append("V7 DIAGNOSIS")
    report.append("=" * 60)

    if oos_stats["trades"] < 20:

        report.append(
            "WARNING: OOS sample is smaller than 20 trades."
        )

    if oos_stats["net_pnl"] > 0:

        report.append(
            "OOS NET P&L IS POSITIVE."
        )

    else:

        report.append(
            "OOS NET P&L IS NEGATIVE."
        )

    if oos_stats["profit_factor"] > 1:

        report.append(
            "OOS PROFIT FACTOR IS ABOVE 1."
        )

    else:

        report.append(
            "OOS PROFIT FACTOR IS NOT ABOVE 1."
        )

    report.append("")
    report.append(
        "V7 was NOT parameter optimized."
    )

    report.append(
        "V7 is intended as a fixed-regime research test."
    )

    return "\n".join(report)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BITRUE V7 - LONG/SHORT REGIME BACKTEST")
    print("=" * 60)

    print()
    print("PAPER / BACKTEST ONLY")
    print("NO LIVE TRADING")
    print("NO API KEYS")
    print("NO WALLET")
    print("NO REAL ORDERS")
    print()

    df = load_data()

    print(
        f"Loaded {len(df):,} candles"
    )

    trades = run_backtest(df)

    if trades.empty:

        print()
        print("NO TRADES GENERATED.")

        empty_report = build_report(
            df,
            trades
        )

        REPORT_FILE.write_text(
            empty_report,
            encoding="utf-8"
        )

        return

    trades.to_csv(
        TRADE_LOG,
        index=False
    )

    report = build_report(
        df,
        trades
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print()
    print(report)

    print()
    print("=" * 60)
    print("FILES CREATED")
    print("=" * 60)

    print(
        TRADE_LOG
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
