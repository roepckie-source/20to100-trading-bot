# ==========================================
# 20to100 Trading Bot
# Backtest Engine - Strategy v1.0
# ==========================================

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from strategy.indicators import calculate_indicators
from strategy.signals import (
    check_buy_signal,
    check_ema_exit,
    check_rsi_exit,
)


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

    partial_1_done: bool = False
    partial_2_done: bool = False

    highest_price: float = 0.0


@dataclass
class Trade:
    symbol: str
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

        self.consecutive_losses = 0

    # --------------------------------------
    # BUY execution
    # --------------------------------------

    def execute_buy(
        self,
        symbol,
        timestamp,
        close,
        atr
    ):

        if self.position is not None:
            return

        if pd.isna(atr) or atr <= 0:
            return

        # Simulated market BUY including slippage.
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

        # Quantity calculated from risk.
        quantity = (
            risk_amount /
            stop_distance
        )

        # Never use more than available balance.
        max_quantity = (
            self.balance /
            entry_price
        )

        quantity = min(
            quantity,
            max_quantity
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

            quantity = (
                self.balance /
                (entry_price * (1 + self.fee_rate))
            )

            position_value = (
                quantity *
                entry_price
            )

            entry_fee = (
                position_value *
                self.fee_rate
            )

        self.balance -= (
            position_value +
            entry_fee
        )

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
        )

    # --------------------------------------
    # SELL helper
    # --------------------------------------

    def execute_partial_sell(
        self,
        position: Position,
        price: float,
        quantity: float,
        timestamp,
        reason: str,
    ):

        if quantity <= 0:
            return 0.0

        quantity = min(
            quantity,
            position.remaining_quantity
        )

        # Simulated market SELL including slippage.
        exit_price = (
            price *
            (1 - self.slippage_rate)
        )

        gross_value = (
            quantity *
            exit_price
        )

        fee = (
            gross_value *
            self.fee_rate
        )

        net_value = (
            gross_value -
            fee
        )

        self.balance += net_value

        # Approximate proportional entry cost.
        entry_value = (
            quantity *
            position.entry_price
        )

        entry_fee = (
            entry_value *
            self.fee_rate
        )

        gross_profit = (
            (exit_price - position.entry_price)
            * quantity
        )

        total_fees = (
            fee +
            entry_fee
        )

        slippage_cost = (
            (
                price -
                exit_price
            )
            * quantity
        )

        net_profit = (
            gross_profit -
            total_fees
        )

        r_multiple = (
            net_profit /
            (
                position.risk_per_unit *
                quantity
            )
            if quantity > 0
            else 0.0
        )

        trade = Trade(
            symbol=position.symbol,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=quantity,
            gross_profit=gross_profit,
            fees=total_fees,
            slippage_cost=slippage_cost,
            net_profit=net_profit,
            exit_reason=reason,
            r_multiple=r_multiple,
        )

        self.trades.append(trade)

        position.remaining_quantity -= quantity

        return net_profit

    # --------------------------------------
    # Complete exit
    # --------------------------------------

    def close_position(
        self,
        price,
        timestamp,
        reason,
    ):

        if self.position is None:
            return

        position = self.position

        quantity = position.remaining_quantity

        if quantity > 0:

            self.execute_partial_sell(
                position=position,
                price=price,
                quantity=quantity,
                timestamp=timestamp,
                reason=reason,
            )

        self.position = None

    # --------------------------------------
    # Manage open position
    # --------------------------------------

    def manage_position(
        self,
        row,
        previous_rows,
    ):

        if self.position is None:
            return

        position = self.position

        high = row["high"]
        low = row["low"]
        close = row["close"]

        # ----------------------------------
        # Update highest price
        # ----------------------------------

        if high > position.highest_price:
            position.highest_price = high

        # ----------------------------------
        # STOP LOSS
        # ----------------------------------

        if low <= position.current_stop:

            self.close_position(
                price=position.current_stop,
                timestamp=row.name,
                reason="STOP_LOSS",
            )

            return

        # ----------------------------------
        # 1R partial exit
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
                price=one_r_price,
                quantity=quantity,
                timestamp=row.name,
                reason="PARTIAL_1R",
            )

            position.partial_1_done = True

            # Move stop to break-even
            # plus approximate fees.
            position.current_stop = (
                position.entry_price *
                (1 + self.fee_rate)
            )

        # ----------------------------------
        # 2R partial exit
        # ----------------------------------

        two_r_price = (
            position.entry_price +
            (
                position.risk_per_unit *
                self.take_profit_r
            )
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
                price=two_r_price,
                quantity=quantity,
                timestamp=row.name,
                reason="PARTIAL_2R",
            )

            position.partial_2_done = True

        # ----------------------------------
        # Trailing stop
        # ----------------------------------

        if position.partial_1_done:

            atr = row["atr_14"]

            if not pd.isna(atr) and atr > 0:

                trailing_stop = (
                    position.highest_price -
                    atr
                )

                if trailing_stop > position.current_stop:

                    position.current_stop = (
                        trailing_stop
                    )

        # ----------------------------------
        # EMA exit
        # ----------------------------------

        if check_ema_exit(
            previous_rows
        ):

            self.close_position(
                price=close,
                timestamp=row.name,
                reason="EMA_EXIT",
            )

            return

        # ----------------------------------
        # RSI exit
        # ----------------------------------

        if check_rsi_exit(
            previous_rows
        ):

            self.close_position(
                price=close,
                timestamp=row.name,
                reason="RSI_EXIT",
            )

            return

        # ----------------------------------
        # Time stop
        # ----------------------------------

        duration_minutes = (
            row.name -
            position.entry_time
        ).total_seconds() / 60

        if duration_minutes >= 60:

            self.close_position(
                price=close,
                timestamp=row.name,
                reason="TIME_STOP",
            )

    # --------------------------------------
    # Main backtest
    # --------------------------------------

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

            # ------------------------------
            # Manage existing position
            # ------------------------------

            if self.position is not None:

                self.manage_position(
                    row=current,
                    previous_rows=history,
                )

            # ------------------------------
            # Look for new entry
            # ------------------------------

            if self.position is None:

                signal = check_buy_signal(
                    history
                )

                if signal:

                    self.execute_buy(
                        symbol=symbol,
                        timestamp=current.name,
                        close=current["close"],
                        atr=current["atr_14"],
                    )

            # ------------------------------
            # Equity
            # ------------------------------

            equity = self.balance

            if self.position is not None:

                equity += (
                    self.position.remaining_quantity *
                    current["close"]
                )

            self.equity_curve.append(
                {
                    "timestamp": current.name,
                    "equity": equity,
                }
            )

        # Close remaining position at final price.
        if self.position is not None:

            final_row = df.iloc[-1]

            self.close_position(
                price=final_row["close"],
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
