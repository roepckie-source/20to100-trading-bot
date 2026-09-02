# ============================================================
# 20→100 TRADING BOT
# V4 WALK-FORWARD BACKTEST ENGINE
# ============================================================

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Trade:
    entry_time: object
    exit_time: object

    entry_price: float
    exit_price: float

    quantity: float

    gross_profit: float
    fees: float
    slippage_cost: float
    net_profit: float

    return_pct: float
    exit_reason: str


class V4BacktestEngine:

    def __init__(
        self,
        starting_balance: float = 20.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ):

        self.starting_balance = starting_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    # ========================================================
    # ENTRY
    # ========================================================

    def _execute_entry(
        self,
        cash: float,
        price: float,
    ):

        # Slippage verschlechtert den Kaufpreis
        execution_price = (
            price * (1 + self.slippage_rate)
        )

        # Gebühren berücksichtigen
        quantity = (
            cash
            / (
                execution_price
                * (1 + self.fee_rate)
            )
        )

        gross_cost = (
            quantity * execution_price
        )

        entry_fee = (
            gross_cost * self.fee_rate
        )

        total_cost = (
            gross_cost + entry_fee
        )

        return (
            quantity,
            execution_price,
            entry_fee,
            total_cost,
        )

    # ========================================================
    # EXIT
    # ========================================================

    def _execute_exit(
        self,
        quantity: float,
        price: float,
    ):

        # Slippage verschlechtert den Verkaufspreis
        execution_price = (
            price * (1 - self.slippage_rate)
        )

        gross_value = (
            quantity * execution_price
        )

        exit_fee = (
            gross_value * self.fee_rate
        )

        net_value = (
            gross_value - exit_fee
        )

        return (
            execution_price,
            exit_fee,
            net_value,
        )

    # ========================================================
    # METRICS
    # ========================================================

    def _calculate_max_drawdown(
        self,
        equity_curve,
    ):

        if not equity_curve:
            return 0.0

        series = pd.Series(
            equity_curve,
            dtype=float,
        )

        peak = series.cummax()

        drawdown = (
            (series - peak)
            / peak
            * 100
        )

        return float(
            drawdown.min()
        )

    def _profit_factor(
        self,
        trades: List[Trade],
    ):

        gross_wins = sum(
            trade.net_profit
            for trade in trades
            if trade.net_profit > 0
        )

        gross_losses = abs(
            sum(
                trade.net_profit
                for trade in trades
                if trade.net_profit < 0
            )
        )

        if gross_losses == 0:

            if gross_wins > 0:
                return float("inf")

            return 0.0

        return (
            gross_wins
            / gross_losses
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        df: pd.DataFrame,
        strategy,
    ):

        if df is None or df.empty:

            return {
                "starting_balance": self.starting_balance,
                "final": self.starting_balance,
                "profit": 0.0,
                "return_pct": 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "max_drawdown": 0.0,
                "fees": 0.0,
                "slippage": 0.0,
                "trade_list": [],
                "equity_curve": [],
            }

        data = df.copy()

        # ----------------------------------------------------
        # INDICATORS
        # ----------------------------------------------------

        data = strategy.calculate_indicators(
            data
        )

        data = data.sort_index()

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        cash = float(
            self.starting_balance
        )

        quantity = 0.0

        in_position = False

        entry_price = 0.0
        entry_time = None

        entry_fee = 0.0

        stop_price = None

        highest_price = 0.0

        entry_index = None

        total_fees = 0.0
        total_slippage = 0.0

        trades = []

        equity_curve = []

        # ----------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------

        for i in range(
            len(data)
        ):

            row = data.iloc[i]

            current_price = float(
                row["close"]
            )

            # =================================================
            # POSITION MANAGEMENT
            # =================================================

            if in_position:

                # -------------------------------------------------
                # STOP CHECK
                #
                # Der Stop vom vorherigen Zustand wird zuerst
                # geprüft. Dadurch entsteht kein Lookahead.
                # -------------------------------------------------

                candle_low = float(
                    row["low"]
                )

                if (
                    stop_price is not None
                    and candle_low <= stop_price
                ):

                    exit_price = float(
                        stop_price
                    )

                    (
                        executed_exit_price,
                        exit_fee,
                        net_value,
                    ) = self._execute_exit(
                        quantity,
                        exit_price,
                    )

                    gross_profit = (
                        (
                            executed_exit_price
                            - entry_price
                        )
                        * quantity
                    )

                    fees = (
                        entry_fee
                        + exit_fee
                    )

                    # -------------------------------------------------
                    # Slippage nur als Diagnosewert.
                    # Der tatsächliche Profit wird bereits mit den
                    # ausgeführten Preisen berechnet.
                    # -------------------------------------------------

                    theoretical_entry = (
                        entry_price
                        / (
                            1
                            + self.slippage_rate
                        )
                    )

                    theoretical_exit = (
                        executed_exit_price
                        / (
                            1
                            - self.slippage_rate
                        )
                    )

                    slippage_cost = (
                        abs(
                            entry_price
                            - theoretical_entry
                        )
                        * quantity
                        +
                        abs(
                            theoretical_exit
                            - executed_exit_price
                        )
                        * quantity
                    )

                    final_cash = net_value

                    net_profit = (
                        final_cash
                        - (
                            quantity
                            * entry_price
                            + entry_fee
                        )
                    )

                    return_pct = (
                        net_profit
                        / (
                            quantity
                            * entry_price
                            + entry_fee
                        )
                        * 100
                    )

                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=data.index[i],
                        entry_price=entry_price,
                        exit_price=executed_exit_price,
                        quantity=quantity,
                        gross_profit=gross_profit,
                        fees=fees,
                        slippage_cost=slippage_cost,
                        net_profit=net_profit,
                        return_pct=return_pct,
                        exit_reason="ATR_STOP",
                    )

                    trades.append(
                        trade
                    )

                    total_fees += fees
                    total_slippage += (
                        slippage_cost
                    )

                    cash = final_cash

                    quantity = 0.0
                    in_position = False

                    entry_price = 0.0
                    entry_time = None
                    entry_fee = 0.0

                    stop_price = None
                    highest_price = 0.0
                    entry_index = None

                    equity_curve.append(
                        cash
                    )

                    continue

                # -------------------------------------------------
                # TRAILING STOP
                #
                # Erst nach dem Stop-Check wird das aktuelle
                # Candle-High verwendet.
                # -------------------------------------------------

                candle_high = float(
                    row["high"]
                )

                if candle_high > highest_price:

                    highest_price = (
                        candle_high
                    )

                atr = row.get(
                    "atr",
                    None
                )

                if (
                    atr is not None
                    and not pd.isna(atr)
                ):

                    new_stop = (
                        strategy.calculate_trailing_stop(
                            highest_price,
                            float(atr),
                        )
                    )

                    if (
                        stop_price is None
                        or new_stop > stop_price
                    ):

                        stop_price = (
                            new_stop
                        )

            # =================================================
            # ENTRY
            # =================================================

            if (
                not in_position
                and i < len(data) - 1
            ):

                if strategy.check_entry(
                    row
                ):

                    next_row = data.iloc[
                        i + 1
                    ]

                    next_open = float(
                        next_row["open"]
                    )

                    atr = row.get(
                        "atr",
                        None
                    )

                    if (
                        atr is None
                        or pd.isna(atr)
                        or atr <= 0
                    ):

                        equity_curve.append(
                            cash
                        )

                        continue

                    (
                        new_quantity,
                        executed_entry_price,
                        new_entry_fee,
                        total_cost,
                    ) = self._execute_entry(
                        cash,
                        next_open,
                    )

                    if (
                        new_quantity <= 0
                        or total_cost > cash
                    ):

                        equity_curve.append(
                            cash
                        )

                        continue

                    quantity = (
                        new_quantity
                    )

                    entry_price = (
                        executed_entry_price
                    )

                    entry_time = (
                        data.index[i + 1]
                    )

                    entry_fee = (
                        new_entry_fee
                    )

                    cash = (
                        cash
                        - total_cost
                    )

                    in_position = True

                    entry_index = (
                        i + 1
                    )

                    highest_price = (
                        entry_price
                    )

                    stop_price = (
                        strategy.calculate_stop(
                            entry_price,
                            float(atr),
                        )
                    )

                    total_fees += (
                        entry_fee
                    )

            # =================================================
            # EQUITY
            # =================================================

            if in_position:

                equity = (
                    cash
                    + quantity
                    * current_price
                )

            else:

                equity = cash

            equity_curve.append(
                equity
            )

        # =====================================================
        # FORCE CLOSE AT END
        # =====================================================

        if in_position:

            last_index = len(data) - 1

            last_row = data.iloc[
                last_index
            ]

            final_price = float(
                last_row["close"]
            )

            (
                executed_exit_price,
                exit_fee,
                net_value,
            ) = self._execute_exit(
                quantity,
                final_price,
            )

            gross_profit = (
                (
                    executed_exit_price
                    - entry_price
                )
                * quantity
            )

            fees = (
                entry_fee
                + exit_fee
            )

            theoretical_entry = (
                entry_price
                / (
                    1
                    + self.slippage_rate
                )
            )

            theoretical_exit = (
                executed_exit_price
                / (
                    1
                    - self.slippage_rate
                )
            )

            slippage_cost = (
                abs(
                    entry_price
                    - theoretical_entry
                )
                * quantity
                +
                abs(
                    theoretical_exit
                    - executed_exit_price
                )
                * quantity
            )

            final_cash = net_value

            net_profit = (
                final_cash
                - (
                    quantity
                    * entry_price
                    + entry_fee
                )
            )

            return_pct = (
                net_profit
                / (
                    quantity
                    * entry_price
                    + entry_fee
                )
                * 100
            )

            trade = Trade(
                entry_time=entry_time,
                exit_time=data.index[last_index],
                entry_price=entry_price,
                exit_price=executed_exit_price,
                quantity=quantity,
                gross_profit=gross_profit,
                fees=fees,
                slippage_cost=slippage_cost,
                net_profit=net_profit,
                return_pct=return_pct,
                exit_reason="END_OF_DATA",
            )

            trades.append(
                trade
            )

            total_fees += fees
            total_slippage += (
                slippage_cost
            )

            cash = final_cash

            equity_curve.append(
                cash
            )

        # =====================================================
        # FINAL METRICS
        # =====================================================

        final_balance = float(
            cash
        )

        profit = (
            final_balance
            - self.starting_balance
        )

        return_pct = (
            profit
            / self.starting_balance
            * 100
        )

        trade_count = len(
            trades
        )

        winners = sum(
            1
            for trade in trades
            if trade.net_profit > 0
        )

        win_rate = (
            winners
            / trade_count
            * 100
            if trade_count > 0
            else 0.0
        )

        profit_factor = (
            self._profit_factor(
                trades
            )
        )

        expectancy = (
            sum(
                trade.net_profit
                for trade in trades
            )
            / trade_count
            if trade_count > 0
            else 0.0
        )

        max_drawdown = (
            self._calculate_max_drawdown(
                equity_curve
            )
        )

        return {
            "starting_balance": self.starting_balance,
            "final": final_balance,
            "profit": profit,
            "return_pct": return_pct,
            "trades": trade_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown": max_drawdown,
            "fees": total_fees,
            "slippage": total_slippage,
            "trade_list": trades,
            "equity_curve": equity_curve,
        }
