"""
Bitrue V8 - Pullback Backtest

FINAL STRATEGY TEST

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO WALLET
NO REAL ORDERS
"""

from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from v8_pullback_strategy import (
    calculate_indicators,
    evaluate_precomputed,
)


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("BTCUSDT_5m.csv")

TRADE_LOG = Path(
    "bitrue_v8_trade_log.csv"
)

REPORT_FILE = Path(
    "bitrue_v8_report.txt"
)

STARTING_CAPITAL = 100.00

FEE_RATE = 0.0006

SLIPPAGE_RATE = 0.0002

COOLDOWN_MINUTES = 30

TRAIN_RATIO = 0.70


# ============================================================
# POSITION
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
# DATA
# ============================================================

def load_data(
    path: Path = DATA_FILE
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
        .reset_index(drop=True)
    )

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# EXECUTION
# ============================================================

def entry_price(
    price: float,
    side: str
) -> float:

    if side == "LONG":

        return (
            price
            * (1.0 + SLIPPAGE_RATE)
        )

    return (
        price
        * (1.0 - SLIPPAGE_RATE)
    )


def exit_price(
    price: float,
    side: str
) -> float:

    if side == "LONG":

        return (
            price
            * (1.0 - SLIPPAGE_RATE)
        )

    return (
        price
        * (1.0 + SLIPPAGE_RATE)
    )


# ============================================================
# P&L
# ============================================================

def gross_pnl(
    side: str,
    entry: float,
    exit: float,
    size: float
) -> float:

    if side == "LONG":

        return (
            (exit - entry)
            / entry
            * size
        )

    return (
        (entry - exit)
        / entry
        * size
    )


# ============================================================
# STATISTICS
# ============================================================

def stats(
    trades: pd.DataFrame,
    starting_balance: float
) -> dict:

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross": 0.0,
            "fees": 0.0,
            "net": 0.0,
            "avg": 0.0,
            "pf": 0.0,
            "dd": 0.0,
        }

    wins = (
        trades["net_pnl"] > 0
    ).sum()

    losses = (
        trades["net_pnl"] <= 0
    ).sum()

    positive = trades.loc[
        trades["net_pnl"] > 0,
        "net_pnl"
    ].sum()

    negative = abs(
        trades.loc[
            trades["net_pnl"] < 0,
            "net_pnl"
        ].sum()
    )

    if negative > 0:

        pf = (
            positive
            / negative
        )

    else:

        pf = float("inf")

    equity = (
        starting_balance
        + trades["net_pnl"].cumsum()
    )

    peak = equity.cummax()

    drawdown = (
        peak - equity
    )

    max_dd = (
        drawdown.max()
        if len(drawdown)
        else 0.0
    )

    return {
        "trades": len(trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": (
            wins / len(trades) * 100.0
        ),
        "gross": trades["gross_pnl"].sum(),
        "fees": trades["fees"].sum(),
        "net": trades["net_pnl"].sum(),
        "avg": trades["net_pnl"].mean(),
        "pf": pf,
        "dd": max_dd,
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df: pd.DataFrame
) -> pd.DataFrame:

    data = calculate_indicators(df)

    split = int(
        len(data)
        * TRAIN_RATIO
    )

    data["period"] = "TRAIN"

    data.loc[
        data.index >= split,
        "period"
    ] = "OOS"

    balance = STARTING_CAPITAL

    position = None

    last_exit = None

    trades = []

    # --------------------------------------------------------
    # ITERATE
    # --------------------------------------------------------

    for i in range(len(data)):

        row = data.iloc[i]

        timestamp = row["timestamp"]

        # ====================================================
        # MANAGE OPEN POSITION
        # ====================================================

        if position is not None:

            exit_raw = None

            reason = None

            # ------------------------------------------------
            # LONG
            # ------------------------------------------------

            if position.side == "LONG":

                # Conservative:
                # stop before target
                if (
                    row["low"]
                    <= position.stop_loss
                ):

                    exit_raw = (
                        position.stop_loss
                    )

                    reason = "STOP_LOSS"

                elif (
                    row["high"]
                    >= position.take_profit
                ):

                    exit_raw = (
                        position.take_profit
                    )

                    reason = "TAKE_PROFIT"

            # ------------------------------------------------
            # SHORT
            # ------------------------------------------------

            elif position.side == "SHORT":

                if (
                    row["high"]
                    >= position.stop_loss
                ):

                    exit_raw = (
                        position.stop_loss
                    )

                    reason = "STOP_LOSS"

                elif (
                    row["low"]
                    <= position.take_profit
                ):

                    exit_raw = (
                        position.take_profit
                    )

                    reason = "TAKE_PROFIT"

            # =================================================
            # EXIT
            # =================================================

            if exit_raw is not None:

                executed_exit = exit_price(
                    exit_raw,
                    position.side
                )

                pnl = gross_pnl(
                    position.side,
                    position.entry_price,
                    executed_exit,
                    position.position_size
                )

                # Entry + exit
                notional = (
                    position.position_size
                    * 2.0
                )

                fees = (
                    notional
                    * FEE_RATE
                )

                net = (
                    pnl
                    - fees
                )

                balance += net

                duration = (
                    timestamp
                    - position.entry_time
                ).total_seconds() / 60.0

                trades.append({

                    "entry_time":
                        position.entry_time,

                    "exit_time":
                        timestamp,

                    "period":
                        row["period"],

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
                        pnl,

                    "fees":
                        fees,

                    "net_pnl":
                        net,

                    "balance":
                        balance,

                    "duration_minutes":
                        duration,

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
                })

                last_exit = timestamp

                position = None

                continue

        # ====================================================
        # COOLDOWN
        # ====================================================

        if last_exit is not None:

            elapsed = (
                timestamp
                - last_exit
            ).total_seconds() / 60.0

            if elapsed < COOLDOWN_MINUTES:

                continue

        # ====================================================
        # SIGNAL
        # ====================================================

        signal = evaluate_precomputed(
            data,
            i,
            balance
        )

        if signal.signal not in (
            "BUY",
            "SELL"
        ):

            continue

        if signal.side is None:

            continue

        # ====================================================
        # ENTRY
        # ====================================================

        executed_entry = entry_price(
            signal.price,
            signal.side
        )

        position = Position(

            side=signal.side,

            entry_time=timestamp,

            entry_price=executed_entry,

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
    title: str,
    result: dict
) -> str:

    return f"""
{title}
------------------------------------------------------------
Trades:          {result['trades']}
Wins:            {result['wins']}
Losses:          {result['losses']}
Win Rate:        {result['win_rate']:.2f}%
Gross P&L:       ${result['gross']:.4f}
Fees:            ${result['fees']:.4f}
Net P&L:         ${result['net']:.4f}
Avg Trade:       ${result['avg']:.6f}
Profit Factor:   {result['pf']:.3f}
Max Drawdown:    ${result['dd']:.4f}
"""


def create_report(
    df: pd.DataFrame,
    trades: pd.DataFrame
) -> str:

    lines = []

    lines.append(
        "BITRUE V8 - PULLBACK LONG/SHORT"
    )

    lines.append(
        "=" * 60
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

    lines.append("")

    lines.append(
        f"Data candles: {len(df):,}"
    )

    lines.append(
        f"Data start: {df['timestamp'].min()}"
    )

    lines.append(
        f"Data end:   {df['timestamp'].max()}"
    )

    lines.append("")

    lines.append(
        "V8 FIXED STRATEGY"
    )

    lines.append(
        "-" * 60
    )

    lines.append(
        "1H trend + 5m trend"
    )

    lines.append(
        "Wait for pullback"
    )

    lines.append(
        "Enter on trend resumption"
    )

    lines.append(
        "LONG and SHORT"
    )

    lines.append(
        "Neutral = no trade"
    )

    lines.append(
        "Momentum minimum: 0.10%"
    )

    lines.append(
        "ATR range: 0.10% - 0.30%"
    )

    lines.append(
        "Volume: 1.20x - 5.00x"
    )

    lines.append(
        "EMA spread: >= 0.15%"
    )

    lines.append(
        "SL: 1.4x ATR"
    )

    lines.append(
        "TP: 2.5x ATR"
    )

    lines.append(
        "Cooldown: 30 minutes"
    )

    lines.append("")

    # ========================================================
    # TOTAL
    # ========================================================

    total = stats(
        trades,
        STARTING_CAPITAL
    )

    lines.append(
        format_stats(
            "TOTAL",
            total
        )
    )

    # ========================================================
    # TRAIN
    # ========================================================

    train = trades[
        trades["period"] == "TRAIN"
    ]

    train_stats = stats(
        train,
        STARTING_CAPITAL
    )

    lines.append(
        format_stats(
            "TRAIN",
            train_stats
        )
    )

    # ========================================================
    # OOS
    # ========================================================

    oos = trades[
        trades["period"] == "OOS"
    ]

    oos_stats = stats(
        oos,
        STARTING_CAPITAL
    )

    lines.append(
        format_stats(
            "OOS",
            oos_stats
        )
    )

    # ========================================================
    # LONG
    # ========================================================

    long = trades[
        trades["side"] == "LONG"
    ]

    long_stats = stats(
        long,
        STARTING_CAPITAL
    )

    lines.append(
        format_stats(
            "LONG TOTAL",
            long_stats
        )
    )

    # ========================================================
    # SHORT
    # ========================================================

    short = trades[
        trades["side"] == "SHORT"
    ]

    short_stats = stats(
        short,
        STARTING_CAPITAL
    )

    lines.append(
        format_stats(
            "SHORT TOTAL",
            short_stats
        )
    )

    # ========================================================
    # OOS LONG
    # ========================================================

    oos_long = trades[
        (trades["period"] == "OOS")
        &
        (trades["side"] == "LONG")
    ]

    oos_long_stats = stats(
        oos_long,
        STARTING_CAPITAL
    )

    lines.append(
        format_stats(
            "OOS LONG",
            oos_long_stats
        )
    )

    # ========================================================
    # OOS SHORT
    # ========================================================

    oos_short = trades[
        (trades["period"] == "OOS")
        &
        (trades["side"] == "SHORT")
    ]

    oos_short_stats = stats(
        oos_short,
        STARTING_CAPITAL
    )

    lines.append(
        format_stats(
            "OOS SHORT",
            oos_short_stats
        )
    )

    # ========================================================
    # EXIT DISTRIBUTION
    # ========================================================

    if not trades.empty:

        lines.append("")

        lines.append(
            "EXIT DISTRIBUTION"
        )

        lines.append(
            "-" * 60
        )

        exits = (
            trades["exit_reason"]
            .value_counts()
        )

        for reason, count in exits.items():

            lines.append(
                f"{reason}: {count}"
            )

    # ========================================================
    # DIRECTION
    # ========================================================

    if not trades.empty:

        lines.append("")

        lines.append(
            "DIRECTION BY PERIOD"
        )

        lines.append(
            "-" * 60
        )

        direction = (
            trades.groupby(
                ["period", "side"]
            )
            .size()
        )

        for key, count in direction.items():

            lines.append(
                f"{key}: {count}"
            )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    lines.append("")

    lines.append(
        "=" * 60
    )

    lines.append(
        "V8 FINAL DECISION"
    )

    lines.append(
        "=" * 60
    )

    if oos_stats["trades"] < 20:

        lines.append(
            "OOS SAMPLE WARNING: fewer than 20 trades."
        )

    if (
        oos_stats["net"] > 0
        and oos_stats["pf"] > 1
        and oos_stats["avg"] > 0
    ):

        lines.append(
            "V8 OOS PASSED THE BASIC PROFITABILITY TEST."
        )

    else:

        lines.append(
            "V8 OOS DID NOT PASS THE BASIC PROFITABILITY TEST."
        )

    lines.append("")

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "V8 was not parameter optimized."
    )

    lines.append(
        "V8 is the final strategy experiment."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "BITRUE V8 - FINAL PULLBACK BACKTEST"
    )

    print("=" * 60)

    print()

    print(
        "PAPER / BACKTEST ONLY"
    )

    print(
        "NO LIVE TRADING"
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

    print()

    df = load_data()

    print(
        f"Loaded {len(df):,} candles"
    )

    trades = run_backtest(df)

    if trades.empty:

        print()
        print(
            "NO TRADES GENERATED."
        )

        report = create_report(
            df,
            trades
        )

        REPORT_FILE.write_text(
            report,
            encoding="utf-8"
        )

        return

    trades.to_csv(
        TRADE_LOG,
        index=False
    )

    report = create_report(
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

    print(
        "=" * 60
    )

    print(
        "FILES CREATED"
    )

    print(
        "=" * 60
    )

    print(
        TRADE_LOG
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
