"""
Bitrue BTC/USDT 5m Backtester V1

Strategy:
    Momentum + EMA + ATR + Volume

Features:
    - LONG / SHORT
    - Stop Loss
    - Take Profit
    - Taker fees
    - Slippage
    - Daily loss limit
    - Equity tracking
    - Trade log
    - Performance statistics

BACKTEST ONLY
NO LIVE TRADING
NO ORDERS
NO API KEYS
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from config import (
    STARTING_CAPITAL,
    TAKER_FEE,
    SLIPPAGE,
    MAX_DAILY_LOSS,
)

from strategy_v1 import (
    evaluate,
)


# ============================================================
# TRADE
# ============================================================

@dataclass
class Trade:

    entry_time: object
    exit_time: object

    side: str

    entry_price: float
    exit_price: float

    position_size: float

    gross_pnl: float
    fees: float
    slippage_cost: float

    net_pnl: float

    return_pct: float

    exit_reason: str


# ============================================================
# BACKTESTER
# ============================================================

class Backtester:

    def __init__(
        self,
        starting_capital: float = STARTING_CAPITAL,
    ):

        self.starting_capital = starting_capital

        self.balance = starting_capital

        self.equity_curve = []

        self.trades: List[Trade] = []

        self.open_trade = None

        self.daily_start_balance = starting_capital

        self.current_day = None

        self.max_equity = starting_capital

        self.max_drawdown = 0.0


    # ========================================================
    # DAILY LOSS CHECK
    # ========================================================

    def reset_daily_limit_if_needed(
        self,
        timestamp,
    ):

        day = timestamp.date()

        if self.current_day != day:

            self.current_day = day

            self.daily_start_balance = self.balance


    def daily_loss_limit_reached(self) -> bool:

        if self.daily_start_balance <= 0:
            return True

        daily_loss = (
            self.daily_start_balance
            - self.balance
        )

        daily_loss_pct = (
            daily_loss
            / self.daily_start_balance
        )

        return daily_loss_pct >= MAX_DAILY_LOSS


    # ========================================================
    # ENTRY SLIPPAGE
    # ========================================================

    def apply_entry_slippage(
        self,
        price: float,
        side: str,
    ) -> float:

        if side == "LONG":

            return price * (
                1 + SLIPPAGE
            )

        return price * (
            1 - SLIPPAGE
        )


    # ========================================================
    # EXIT SLIPPAGE
    # ========================================================

    def apply_exit_slippage(
        self,
        price: float,
        side: str,
    ) -> float:

        if side == "LONG":

            return price * (
                1 - SLIPPAGE
            )

        return price * (
            1 + SLIPPAGE
        )


    # ========================================================
    # OPEN TRADE
    # ========================================================

    def open_position(
        self,
        timestamp,
        price: float,
        signal,
    ):

        if self.open_trade is not None:
            return

        if signal.position_size <= 0:
            return

        entry_price = self.apply_entry_slippage(
            price,
            signal.side,
        )

        position_size = signal.position_size

        self.open_trade = {
            "entry_time": timestamp,
            "side": signal.side,
            "entry_price": entry_price,
            "position_size": position_size,
        }


    # ========================================================
    # EXIT CHECK
    # ========================================================

    def check_exit(
        self,
        timestamp,
        row,
    ):

        if self.open_trade is None:
            return

        trade = self.open_trade

        side = trade["side"]

        entry_price = trade["entry_price"]

        high = float(row["high"])
        low = float(row["low"])

        position_size = trade["position_size"]

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if side == "LONG":

            stop_price = (
                entry_price
                * (1 - 0.003)
            )

            target_price = (
                entry_price
                * (1 + 0.006)
            )

            stop_hit = low <= stop_price
            target_hit = high >= target_price

            if stop_hit and target_hit:

                # Conservative assumption:
                # stop is hit first
                exit_price = stop_price
                reason = "STOP_LOSS"

            elif stop_hit:

                exit_price = stop_price
                reason = "STOP_LOSS"

            elif target_hit:

                exit_price = target_price
                reason = "TAKE_PROFIT"

            else:

                return

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            stop_price = (
                entry_price
                * (1 + 0.003)
            )

            target_price = (
                entry_price
                * (1 - 0.006)
            )

            stop_hit = high >= stop_price
            target_hit = low <= target_price

            if stop_hit and target_hit:

                # Conservative assumption:
                # stop is hit first
                exit_price = stop_price
                reason = "STOP_LOSS"

            elif stop_hit:

                exit_price = stop_price
                reason = "STOP_LOSS"

            elif target_hit:

                exit_price = target_price
                reason = "TAKE_PROFIT"

            else:

                return

        self.close_position(
            timestamp,
            exit_price,
            reason,
        )


    # ========================================================
    # CLOSE TRADE
    # ========================================================

    def close_position(
        self,
        timestamp,
        price: float,
        reason: str,
    ):

        if self.open_trade is None:
            return

        trade = self.open_trade

        side = trade["side"]

        entry_price = trade["entry_price"]

        position_size = trade["position_size"]

        exit_price = self.apply_exit_slippage(
            price,
            side,
        )

        # ----------------------------------------------------
        # PRICE RETURN
        # ----------------------------------------------------

        if side == "LONG":

            price_return = (
                exit_price
                - entry_price
            ) / entry_price

        else:

            price_return = (
                entry_price
                - exit_price
            ) / entry_price

        # ----------------------------------------------------
        # GROSS PNL
        # ----------------------------------------------------

        gross_pnl = (
            position_size
            * price_return
        )

        # ----------------------------------------------------
        # FEES
        # ----------------------------------------------------

        entry_fee = (
            position_size
            * TAKER_FEE
        )

        exit_notional = (
            position_size
            * (1 + price_return)
        )

        exit_fee = (
            abs(exit_notional)
            * TAKER_FEE
        )

        fees = (
            entry_fee
            + exit_fee
        )

        # ----------------------------------------------------
        # SLIPPAGE COST
        # ----------------------------------------------------

        slippage_cost = (
            position_size
            * SLIPPAGE
            * 2
        )

        # ----------------------------------------------------
        # NET PNL
        # ----------------------------------------------------

        net_pnl = (
            gross_pnl
            - fees
        )

        self.balance += net_pnl

        return_pct = (
            net_pnl
            / position_size
            * 100
        )

        completed_trade = Trade(

            entry_time=trade["entry_time"],

            exit_time=timestamp,

            side=side,

            entry_price=entry_price,

            exit_price=exit_price,

            position_size=position_size,

            gross_pnl=gross_pnl,

            fees=fees,

            slippage_cost=slippage_cost,

            net_pnl=net_pnl,

            return_pct=return_pct,

            exit_reason=reason,
        )

        self.trades.append(
            completed_trade
        )

        self.open_trade = None


    # ========================================================
    # EQUITY
    # ========================================================

    def update_equity(
        self,
        timestamp,
    ):

        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": self.balance,
            }
        )

        if self.balance > self.max_equity:

            self.max_equity = self.balance

        drawdown = (
            self.max_equity
            - self.balance
        ) / self.max_equity

        if drawdown > self.max_drawdown:

            self.max_drawdown = drawdown


    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df.empty:

            raise ValueError(
                "DataFrame is empty"
            )

        required = [
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

        data = df.copy()

        # ----------------------------------------------------
        # Ensure datetime index
        # ----------------------------------------------------

        if not isinstance(
            data.index,
            pd.DatetimeIndex,
        ):

            if "timestamp" in data.columns:

                data["timestamp"] = pd.to_datetime(
                    data["timestamp"],
                    utc=True,
                )

                data = data.set_index(
                    "timestamp"
                )

            else:

                raise ValueError(
                    "Data requires a DatetimeIndex "
                    "or timestamp column"
                )

        data = data.sort_index()

        # ----------------------------------------------------
        # Main loop
        # ----------------------------------------------------

        for timestamp, row in data.iterrows():

            self.reset_daily_limit_if_needed(
                timestamp
            )

            # ------------------------------------------------
            # First manage existing position
            # ------------------------------------------------

            if self.open_trade is not None:

                self.check_exit(
                    timestamp,
                    row,
                )

            # ------------------------------------------------
            # New entry
            # ------------------------------------------------

            if (
                self.open_trade is None
                and not self.daily_loss_limit_reached()
            ):

                history = data.loc[
                    :timestamp
                ].copy()

                signal = evaluate(
                    history,
                    self.balance,
                )

                if signal.signal:

                    self.open_position(
                        timestamp,
                        float(row["close"]),
                        signal,
                    )

            # ------------------------------------------------
            # Equity
            # ------------------------------------------------

            self.update_equity(
                timestamp
            )

        # ----------------------------------------------------
        # Close remaining position
        # ----------------------------------------------------

        if self.open_trade is not None:

            last_timestamp = data.index[-1]

            last_price = float(
                data.iloc[-1]["close"]
            )

            self.close_position(
                last_timestamp,
                last_price,
                "END_OF_BACKTEST",
            )

            self.update_equity(
                last_timestamp
            )

        return pd.DataFrame(
            [
                {
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "side": trade.side,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "position_size": trade.position_size,
                    "gross_pnl": trade.gross_pnl,
                    "fees": trade.fees,
                    "slippage_cost": trade.slippage_cost,
                    "net_pnl": trade.net_pnl,
                    "return_pct": trade.return_pct,
                    "exit_reason": trade.exit_reason,
                }
                for trade in self.trades
            ]
        )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    trades: pd.DataFrame,
    starting_capital: float,
    ending_balance: float,
    max_drawdown: float,
):

    if trades.empty:

        return {
            "starting_capital": starting_capital,
            "ending_balance": ending_balance,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": max_drawdown,
            "total_fees": 0.0,
        }

    wins = trades[
        trades["net_pnl"] > 0
    ]

    losses = trades[
        trades["net_pnl"] < 0
    ]

    gross_profit = wins[
        "net_pnl"
    ].sum()

    gross_loss = abs(
        losses["net_pnl"].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = float("inf")

    net_pnl = (
        ending_balance
        - starting_capital
    )

    return_pct = (
        net_pnl
        / starting_capital
        * 100
    )

    win_rate = (
        len(wins)
        / len(trades)
        * 100
    )

    return {
        "starting_capital": starting_capital,
        "ending_balance": ending_balance,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "total_fees": trades["fees"].sum(),
    }


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(
    filename: str,
) -> pd.DataFrame:

    df = pd.read_csv(filename)

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Common timestamp names
    # --------------------------------------------------------

    timestamp_candidates = [
        "timestamp",
        "time",
        "datetime",
        "date",
    ]

    timestamp_column = None

    for column in timestamp_candidates:

        if column in df.columns:

            timestamp_column = column
            break

    if timestamp_column is None:

        raise ValueError(
            "No timestamp column found. "
            "Expected timestamp, time, datetime or date."
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        utc=True,
    )

    df = df.set_index(
        timestamp_column
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        if column not in df.columns:

            raise ValueError(
                f"Missing CSV column: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    df = df.sort_index()

    return df


# ============================================================
# REPORT
# ============================================================

def print_report(
    statistics,
):

    print()
    print("=" * 60)
    print("BITRUE BTC/USDT 5M BACKTEST")
    print("=" * 60)

    print()

    print("BACKTEST ONLY")
    print("NO LIVE TRADING")
    print("NO ORDERS")
    print("NO API KEYS")

    print()

    print("-" * 60)
    print("CAPITAL")
    print("-" * 60)

    print(
        f"Starting Capital:     "
        f"${statistics['starting_capital']:.2f}"
    )

    print(
        f"Ending Balance:       "
        f"${statistics['ending_balance']:.2f}"
    )

    print(
        f"Net P&L:              "
        f"${statistics['net_pnl']:+.2f}"
    )

    print(
        f"Return:               "
        f"{statistics['return_pct']:+.2f}%"
    )

    print()

    print("-" * 60)
    print("TRADES")
    print("-" * 60)

    print(
        f"Trades:               "
        f"{statistics['trades']}"
    )

    print(
        f"Wins:                 "
        f"{statistics['wins']}"
    )

    print(
        f"Losses:               "
        f"{statistics['losses']}"
    )

    print(
        f"Win Rate:             "
        f"{statistics['win_rate']:.2f}%"
    )

    print(
        f"Profit Factor:        "
        f"{statistics['profit_factor']:.3f}"
    )

    print()

    print("-" * 60)
    print("RISK")
    print("-" * 60)

    print(
        f"Max Drawdown:         "
        f"{statistics['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Total Fees:           "
        f"${statistics['total_fees']:.4f}"
    )

    print()

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import sys

    print("=" * 60)
    print("BITRUE BTC/USDT 5M BACKTESTER V1")
    print("=" * 60)

    print()

    print("MOMENTUM + EMA + ATR + VOLUME")

    print()

    print("BACKTEST ONLY")
    print("NO LIVE TRADING")
    print("NO ORDERS")
    print("NO API KEYS")
    print()

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print()

        print(
            "python backtest.py "
            "BTCUSDT_5m.csv"
        )

        print()

        print(
            "No backtest executed."
        )

        raise SystemExit(0)

    filename = sys.argv[1]

    print(
        f"Loading: {filename}"
    )

    print()

    df = load_csv(
        filename
    )

    print(
        f"Candles: {len(df):,}"
    )

    print(
        f"Start:   {df.index[0]}"
    )

    print(
        f"End:     {df.index[-1]}"
    )

    print()

    backtester = Backtester(
        STARTING_CAPITAL
    )

    trades = backtester.run(
        df
    )

    statistics = calculate_statistics(
        trades=trades,
        starting_capital=(
            STARTING_CAPITAL
        ),
        ending_balance=(
            backtester.balance
        ),
        max_drawdown=(
            backtester.max_drawdown
        ),
    )

    print_report(
        statistics
    )

    # --------------------------------------------------------
    # Save trade log
    # --------------------------------------------------------

    if not trades.empty:

        output_file = (
            "bitrue_v1_trade_log.csv"
        )

        trades.to_csv(
            output_file,
            index=False,
        )

        print()

        print(
            f"Trade log saved: "
            f"{output_file}"
        )

    print()

    print(
        "=" * 60
    )

    print(
        "BACKTEST COMPLETE"
    )

    print(
        "NO REAL TRADING WAS PERFORMED."
    )

    print(
        "=" * 60
    )
