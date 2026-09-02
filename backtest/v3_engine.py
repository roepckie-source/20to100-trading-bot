# ==========================================
# 20to100 Trading Bot
# V3 Backtest Engine
# ==========================================

from dataclasses import dataclass

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

    exit_reason: str


class V3BacktestEngine:

    def __init__(
        self,
        starting_balance=20.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
    ):

        self.starting_balance = float(
            starting_balance
        )

        self.fee_rate = float(
            fee_rate
        )

        self.slippage_rate = float(
            slippage_rate
        )

        self.cash = (
            self.starting_balance
        )

        self.position = False

        self.quantity = 0.0

        self.entry_price = 0.0
        self.entry_time = None

        self.entry_cost = 0.0

        self.highest_price = 0.0
        self.stop_price = 0.0

        self.trades = []

        self.equity_curve = []

    # ======================================
    # BUY
    # ======================================

    def enter_position(
        self,
        timestamp,
        market_price,
        atr,
        stop_multiplier,
    ):

        if self.position:
            return

        # Slippage on entry
        entry_price = (
            market_price
            * (1.0 + self.slippage_rate)
        )

        # Maximum affordable quantity
        quantity = (
            self.cash
            /
            (
                entry_price
                * (1.0 + self.fee_rate)
            )
        )

        if quantity <= 0:
            return

        position_value = (
            quantity
            * entry_price
        )

        entry_fee = (
            position_value
            * self.fee_rate
        )

        total_cost = (
            position_value
            + entry_fee
        )

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        self.position = True

        self.quantity = quantity

        self.entry_price = (
            entry_price
        )

        self.entry_time = timestamp

        self.entry_cost = total_cost

        self.highest_price = (
            entry_price
        )

        self.stop_price = (
            entry_price
            -
            atr
            * stop_multiplier
        )

    # ======================================
    # SELL
    # ======================================

    def exit_position(
        self,
        timestamp,
        market_price,
        reason,
    ):

        if not self.position:
            return

        # Slippage on exit
        exit_price = (
            market_price
            * (1.0 - self.slippage_rate)
        )

        gross_value = (
            self.quantity
            * exit_price
        )

        exit_fee = (
            gross_value
            * self.fee_rate
        )

        net_value = (
            gross_value
            - exit_fee
        )

        gross_profit = (
            gross_value
            -
            (
                self.quantity
                * self.entry_price
            )
        )

        entry_fee = (
            self.entry_cost
            -
            (
                self.quantity
                * self.entry_price
            )
        )

        total_fees = (
            entry_fee
            + exit_fee
        )

        slippage_cost = (
            (
                self.entry_price
                * self.quantity
                * self.slippage_rate
            )
            +
            (
                exit_price
                * self.quantity
                * self.slippage_rate
            )
        )

        net_profit = (
            net_value
            -
            self.entry_cost
        )

        self.cash += net_value

        self.trades.append(
            Trade(
                entry_time=self.entry_time,
                exit_time=timestamp,
                entry_price=self.entry_price,
                exit_price=exit_price,
                quantity=self.quantity,
                gross_profit=gross_profit,
                fees=total_fees,
                slippage_cost=slippage_cost,
                net_profit=net_profit,
                exit_reason=reason,
            )
        )

        self.position = False

        self.quantity = 0.0

        self.entry_price = 0.0

        self.entry_time = None

        self.entry_cost = 0.0

        self.highest_price = 0.0

        self.stop_price = 0.0

    # ======================================
    # EQUITY
    # ======================================

    def get_equity(
        self,
        close_price,
    ):

        if not self.position:
            return self.cash

        return (
            self.quantity
            * close_price
        )

    # ======================================
    # RUN
    # ======================================

    def run(
        self,
        df,
        strategy,
        timeframe_name,
    ):

        data = df.copy()

        data = strategy.calculate_indicators(
            data
        )

        self.cash = (
            self.starting_balance
        )

        self.position = False

        self.quantity = 0.0

        self.trades = []

        self.equity_curve = []

        # ----------------------------------
        # MAIN LOOP
        # ----------------------------------

        for i in range(
            len(data) - 1
        ):

            row = data.iloc[i]

            next_row = data.iloc[i + 1]

            timestamp = row.name

            # ==============================
            # POSITION MANAGEMENT
            # ==============================

            if self.position:

                # Existing stop is checked
                # BEFORE trailing update.

                candle_low = float(
                    row["low"]
                )

                if (
                    candle_low
                    <= self.stop_price
                ):

                    self.exit_position(
                        timestamp=timestamp,
                        market_price=self.stop_price,
                        reason="ATR_STOP",
                    )

                    self.equity_curve.append(
                        self.cash
                    )

                    continue

                # Update highest price

                candle_high = float(
                    row["high"]
                )

                if (
                    candle_high
                    > self.highest_price
                ):

                    self.highest_price = (
                        candle_high
                    )

                # Update trailing stop

                atr = row["atr"]

                if (
                    not pd.isna(atr)
                    and atr > 0
                ):

                    new_stop = (
                        self.highest_price
                        -
                        atr
                        * strategy.atr_stop_multiplier
                    )

                    if (
                        new_stop
                        > self.stop_price
                    ):

                        self.stop_price = (
                            new_stop
                        )

            # ==============================
            # ENTRY
            # ==============================

            if not self.position:

                if strategy.check_entry(
                    row
                ):

                    next_open = float(
                        next_row["open"]
                    )

                    atr = float(
                        row["atr"]
                    )

                    if (
                        not pd.isna(atr)
                        and atr > 0
                    ):

                        self.enter_position(
                            timestamp=next_row.name,
                            market_price=next_open,
                            atr=atr,
                            stop_multiplier=(
                                strategy.atr_stop_multiplier
                            ),
                        )

            # ==============================
            # EQUITY
            # ==============================

            equity = self.get_equity(
                float(row["close"])
            )

            self.equity_curve.append(
                equity
            )

        # ==================================
        # CLOSE OPEN POSITION
        # ==================================

        if self.position:

            last_row = data.iloc[-1]

            self.exit_position(
                timestamp=last_row.name,
                market_price=float(
                    last_row["close"]
                ),
                reason="END_OF_DATA",
            )

            self.equity_curve.append(
                self.cash
            )

        return self.results(
            timeframe_name
        )

    # ======================================
    # RESULTS
    # ======================================

    def results(
        self,
        timeframe_name,
    ):

        final_balance = float(
            self.cash
        )

        profit = (
            final_balance
            -
            self.starting_balance
        )

        return_pct = (
            profit
            /
            self.starting_balance
            * 100.0
        )

        trades = len(
            self.trades
        )

        if trades:

            winners = [
                t
                for t in self.trades
                if t.net_profit > 0
            ]

            losers = [
                t
                for t in self.trades
                if t.net_profit <= 0
            ]

            win_rate = (
                len(winners)
                /
                trades
                * 100.0
            )

            gross_wins = sum(
                t.net_profit
                for t in winners
            )

            gross_losses = abs(
                sum(
                    t.net_profit
                    for t in losers
                )
            )

            if gross_losses > 0:

                profit_factor = (
                    gross_wins
                    /
                    gross_losses
                )

            else:

                profit_factor = float(
                    "inf"
                )

            expectancy = (
                sum(
                    t.net_profit
                    for t in self.trades
                )
                /
                trades
            )

        else:

            win_rate = 0.0
            profit_factor = 0.0
            expectancy = 0.0

        # ----------------------------------
        # Maximum drawdown
        # ----------------------------------

        if self.equity_curve:

            equity = pd.Series(
                self.equity_curve,
                dtype=float,
            )

            peaks = equity.cummax()

            drawdowns = (
                equity
                /
                peaks
                - 1.0
            )

            max_drawdown = (
                drawdowns.min()
                * 100.0
            )

        else:

            max_drawdown = 0.0

        total_fees = sum(
            t.fees
            for t in self.trades
        )

        total_slippage = sum(
            t.slippage_cost
            for t in self.trades
        )

        return {

            "timeframe": (
                timeframe_name
            ),

            "final": (
                final_balance
            ),

            "profit": (
                profit
            ),

            "return_pct": (
                return_pct
            ),

            "trades": (
                trades
            ),

            "win_rate": (
                win_rate
            ),

            "profit_factor": (
                profit_factor
            ),

            "expectancy": (
                expectancy
            ),

            "max_drawdown": (
                max_drawdown
            ),

            "fees": (
                total_fees
            ),

            "slippage": (
                total_slippage
            ),
        }
