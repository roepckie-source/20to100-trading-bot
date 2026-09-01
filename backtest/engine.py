# ==========================================
# 20to100 Trading Bot
# Backtest Engine - Strategy v1.1
# ==========================================

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from strategy.indicators import calculate_indicators
from strategy.signals import check_buy_signal


@dataclass
class Position:
    symbol: str
    entry_time: object
    entry_price: float
    quantity: float

    initial_stop: float
    current_stop: float
    risk_per_unit: float

    remaining_quantity: float
    highest_price: float

    realized_profit: float = 0.0
    total_fees: float = 0.0
    total_slippage_cost: float = 0.0

    partial_1_done: bool = False
    partial_2_done: bool = False


@dataclass
class Trade:
    symbol: str
    entry_time: object
    exit_time: object

    entry_price: float
    final_exit_price: float

    initial_quantity: float
    remaining_quantity: float

    gross_profit: float
    fees: float
    slippage_cost: float
    net_profit: float

    exit_reason: str
    r_multiple: float


class BacktestEngine:

    def __init__(
        self,
        starting_balance: float = 20.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        risk_per_trade: float = 0.015,
        atr_stop_multiplier: float = 1.5,
        take_profit_r: float = 2.0,
    ):

        self.starting_balance = starting_balance
        self.balance = starting_balance

        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

        self.risk_per_trade = risk_per_trade
        self.atr_stop_multiplier = atr_stop_multiplier
        self.take_profit_r = take_profit_r

        self.position: Optional[Position] = None

        self.trades = []
        self.equity_curve = []

    # ======================================
    # BUY
    # ======================================

    def execute_buy(
        self,
        symbol,
        timestamp,
        close,
        atr,
    ):

        if self.position is not None:
            return

        if pd.isna(atr) or atr <= 0:
            return

        # Market BUY with slippage.
        entry_price = (
            close *
            (1 + self.slippage_rate)
        )

        stop_distance = (
            atr *
            self.atr_stop_multiplier
        )

        stop_price = (
            entry_price -
            stop_distance
        )

        # Maximum monetary risk.
        risk_amount = (
            self.balance *
            self.risk_per_trade
        )

        # Quantity based on risk.
        quantity = (
            risk_amount /
            stop_distance
        )

        # No leverage.
        max_quantity = (
            self.balance /
            (
                entry_price *
                (1 + self.fee_rate)
            )
        )

        quantity = min(
            quantity,
            max_quantity,
        )

        if quantity <= 0:
            return

        position_value = (
            quantity *
            entry_price
        )

        entry_fee = (
            position_value *
            self.fee_rate
        )

        total_cost = (
            position_value +
            entry_fee
        )

        if total_cost > self.balance:
            return

        self.balance -= total_cost

        self.position = Position(
            symbol=symbol,
            entry_time=timestamp,
            entry_price=entry_price,
            quantity=quantity,
            initial_stop=stop_price,
            current_stop=stop_price,
            risk_per_unit=stop_distance,
            remaining_quantity=quantity,
            highest_price=entry_price,
            total_fees=entry_fee,
        )

    # ======================================
    # PARTIAL SELL
    # ======================================

    def execute_partial_sell(
        self,
        position: Position,
        market_price: float,
        quantity: float,
        timestamp,
    ) -> float:

        if quantity <= 0:
            return 0.0

        quantity = min(
            quantity,
            position.remaining_quantity,
        )

        # Market SELL with slippage.
        exit_price = (
            market_price *
            (1 - self.slippage_rate)
        )

        gross_value = (
            quantity *
            exit_price
        )

        exit_fee = (
            gross_value *
            self.fee_rate
        )

        net_value = (
            gross_value -
            exit_fee
        )

        self.balance += net_value

        # Proportional entry cost.
        entry_value = (
            quantity *
            position.entry_price
        )

        entry_fee = (
            entry_value *
            self.fee_rate
        )

        gross_profit = (
            exit_price -
            position.entry_price
        ) * quantity

        total_fees = (
            entry_fee +
            exit_fee
        )

        # Approximate round-trip slippage cost.
        theoretical_value = (
            quantity *
            position.entry_price
        )

        actual_entry_value = (
            quantity *
            position.entry_price *
            (1 + self.slippage_rate)
        )

        actual_exit_value = (
            quantity *
            exit_price
        )

        slippage_cost = abs(
            actual_entry_value -
            theoretical_value
        ) + abs(
            theoretical_value -
            actual_exit_value
        )

        net_profit = (
            gross_profit -
            total_fees
        )

        position.realized_profit += net_profit
        position.total_fees += (
            exit_fee +
            entry_fee
        )
        position.total_slippage_cost += (
            slippage_cost
        )

        position.remaining_quantity -= quantity

        return net_profit

    # ======================================
    # CLOSE COMPLETE POSITION
    # ======================================

    def close_position(
        self,
        market_price,
        timestamp,
        reason,
    ):

        if self.position is None:
            return

        position = self.position

        remaining = position.remaining_quantity

        if remaining > 0:

            self.execute_partial_sell(
                position=position,
                market_price=market_price,
                quantity=remaining,
                timestamp=timestamp,
            )

        # Total risk of original position.
        initial_risk = (
            position.risk_per_unit *
            position.quantity
        )

        if initial_risk > 0:
            r_multiple = (
                position.realized_profit /
                initial_risk
            )
        else:
            r_multiple = 0.0

        trade = Trade(
            symbol=position.symbol,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            final_exit_price=market_price,
            initial_quantity=position.quantity,
            remaining_quantity=0.0,
            gross_profit=(
                position.realized_profit +
                position.total_fees
            ),
            fees=position.total_fees,
            slippage_cost=position.total_slippage_cost,
            net_profit=position.realized_profit,
            exit_reason=reason,
            r_multiple=r_multiple,
        )

        self.trades.append(trade)

        self.position = None

    # ======================================
    # MANAGE POSITION
    # ======================================

    def manage_position(
        self,
        row,
        history,
    ):

        if self.position is None:
            return

        position = self.position

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # ----------------------------------
        # Update highest price
        # ----------------------------------

        if high > position.highest_price:
            position.highest_price = high

        # ----------------------------------
        # STOP FIRST
        #
        # Conservative assumption:
        # if both stop and target are touched
        # within the same candle, stop wins.
        # ----------------------------------

        if low <= position.current_stop:

            self.close_position(
                market_price=position.current_stop,
                timestamp=row.name,
                reason="STOP_LOSS",
            )

            return

        # ----------------------------------
        # 1R
        # ----------------------------------

        one_r_price = (
            position.entry_price +
            position.risk_per_unit
        )

        if (
            not position.partial_1_done
            and high >= one_r_price
        ):

            quantity = (
                position.quantity *
                0.50
            )

            self.execute_partial_sell(
                position=position,
                market_price=one_r_price,
                quantity=quantity,
                timestamp=row.name,
            )

            position.partial_1_done = True

            # Break-even including entry fee.
            position.current_stop = (
                position.entry_price *
                (1 + self.fee_rate)
            )

        # ----------------------------------
        # 2R
        # ----------------------------------

        two_r_price = (
            position.entry_price +
            position.risk_per_unit *
            self.take_profit_r
        )

        if (
            not position.partial_2_done
            and high >= two_r_price
        ):

            quantity = (
                position.quantity *
                0.25
            )

            self.execute_partial_sell(
                position=position,
                market_price=two_r_price,
                quantity=quantity,
                timestamp=row.name,
            )

            position.partial_2_done = True

        # ----------------------------------
        # Trailing stop
        # ----------------------------------

        if position.partial_1_done:

            atr = row["atr_14"]

            if (
                not pd.isna(atr)
                and atr > 0
            ):

                trailing_stop = (
                    position.highest_price -
                    atr
                )

                if (
                    trailing_stop >
                    position.current_stop
                ):

                    position.current_stop = (
                        trailing_stop
                    )

        # ----------------------------------
        # EMA EXIT
        # ----------------------------------

        ema_exit = (
            row["ema_9"] <
            row["ema_21"]
        )

        if ema_exit:

            self.close_position(
                market_price=close,
                timestamp=row.name,
                reason="EMA_EXIT",
            )

            return

        # ----------------------------------
        # RSI EXIT
        # ----------------------------------

        if row["rsi_14"] < 45:

            self.close_position(
                market_price=close,
                timestamp=row.name,
                reason="RSI_EXIT",
            )

            return

        # ----------------------------------
        # TIME STOP
        # ----------------------------------

        duration_minutes = (
            row.name -
            position.entry_time
        ).total_seconds() / 60

        if duration_minutes >= 60:

            self.close_position(
                market_price=close,
                timestamp=row.name,
                reason="TIME_STOP",
            )

    # ======================================
    # RUN BACKTEST
    # ======================================

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "BTC/USDT",
    ):

        df = calculate_indicators(df)

        self.balance = self.starting_balance
        self.position = None
        self.trades = []
        self.equity_curve = []

        for i in range(60, len(df)):

            current = df.iloc[i]

            history = df.iloc[: i + 1]

            # --------------------------------
            # Manage existing position
            # --------------------------------

            if self.position is not None:

                self.manage_position(
                    row=current,
                    history=history,
                )

            # --------------------------------
            # New entry
            # --------------------------------

            if self.position is None:

                if check_buy_signal(history):

                    self.execute_buy(
                        symbol=symbol,
                        timestamp=current.name,
                        close=float(
                            current["close"]
                        ),
                        atr=float(
                            current["atr_14"]
                        ),
                    )

            # --------------------------------
            # Equity
            # --------------------------------

            equity = self.balance

            if self.position is not None:

                equity += (
                    self.position.remaining_quantity *
                    float(current["close"])
                )

            self.equity_curve.append(
                {
                    "timestamp": current.name,
                    "equity": equity,
                }
            )

        # ------------------------------------
        # Close open position at end
        # ------------------------------------

        if self.position is not None:

            final_row = df.iloc[-1]

            self.close_position(
                market_price=float(
                    final_row["close"]
                ),
                timestamp=final_row.name,
                reason="END_OF_TEST",
            )

        return {
            "balance": self.balance,
            "trades": self.trades,
            "equity_curve": pd.DataFrame(
                self.equity_curve
            ),
        }
