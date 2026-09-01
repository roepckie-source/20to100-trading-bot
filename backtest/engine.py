# ==========================================
# 20to100 Trading Bot
# Backtest Engine - Strategy v1.1
# ==========================================

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    FEE_RATE,
    SLIPPAGE_RATE,
    RISK_PER_TRADE,
    ATR_STOP_MULTIPLIER,
    TAKE_PROFIT_R,
    PARTIAL_EXIT_1_R,
    PARTIAL_EXIT_1_SIZE,
    PARTIAL_EXIT_2_R,
    PARTIAL_EXIT_2_SIZE,
    TRAILING_ATR_MULTIPLIER,
    MAX_TRADE_MINUTES,
    MAX_DAILY_LOSS,
    MAX_CONSECUTIVE_LOSSES,
    LOSS_COOLDOWN_MINUTES,
    TRADE_COOLDOWN_MINUTES,
)

from strategy.indicators import (
    calculate_indicators,
)

from strategy.signals import (
    check_buy_signal,
    check_ema_exit,
    check_rsi_exit,
)

from risk.risk_manager import (
    RiskManager,
    RiskConfig,
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
    ):

        self.starting_balance = (
            starting_balance
        )

        self.balance = (
            starting_balance
        )

        self.position: Optional[
            Position
        ] = None

        self.trades = []

        self.equity_curve = []

        self.risk_manager = RiskManager(
            starting_balance,
            RiskConfig(
                risk_per_trade=RISK_PER_TRADE,
                max_daily_loss=MAX_DAILY_LOSS,
                max_consecutive_losses=(
                    MAX_CONSECUTIVE_LOSSES
                ),
            ),
        )

        self.cooldown_until = None
        self.loss_cooldown_until = None
        self.current_day = None

    # ======================================
    # DAILY RESET
    # ======================================

    def update_day(
        self,
        timestamp,
    ):

        day = timestamp.date()

        if (
            self.current_day is None
            or day != self.current_day
        ):

            self.current_day = day

            self.risk_manager.reset_daily(
                self.balance
            )

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

        if atr <= 0:
            return

        entry_price = (
            close *
            (1 + SLIPPAGE_RATE)
        )

        stop_distance = (
            atr *
            ATR_STOP_MULTIPLIER
        )

        stop_price = (
            entry_price -
            stop_distance
        )

        quantity = (
            self.risk_manager
            .calculate_position_size(
                balance=self.balance,
                entry_price=entry_price,
                stop_price=stop_price,
            )
        )

        if quantity <= 0:
            return

        position_value = (
            quantity *
            entry_price
        )

        entry_fee = (
            position_value *
            FEE_RATE
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
    # SELL
    # ======================================

    def execute_sell(
        self,
        position,
        market_price,
        quantity,
    ):

        if quantity <= 0:
            return

        quantity = min(
            quantity,
            position.remaining_quantity,
        )

        exit_price = (
            market_price *
            (1 - SLIPPAGE_RATE)
        )

        gross_value = (
            quantity *
            exit_price
        )

        exit_fee = (
            gross_value *
            FEE_RATE
        )

        self.balance += (
            gross_value -
            exit_fee
        )

        entry_value = (
            quantity *
            position.entry_price
        )

        entry_fee = (
            entry_value *
            FEE_RATE
        )

        gross_profit = (
            (
                exit_price -
                position.entry_price
            )
            * quantity
        )

        net_profit = (
            gross_profit -
            entry_fee -
            exit_fee
        )

        # Approximate total slippage.
        entry_slippage = (
            quantity *
            position.entry_price *
            SLIPPAGE_RATE
        )

        exit_slippage = (
            quantity *
            market_price *
            SLIPPAGE_RATE
        )

        position.realized_profit += (
            net_profit
        )

        position.total_fees += (
            entry_fee +
            exit_fee
        )

        position.total_slippage_cost += (
            entry_slippage +
            exit_slippage
        )

        position.remaining_quantity -= (
            quantity
        )

    # ======================================
    # CLOSE TRADE
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

        if position.remaining_quantity > 0:

            self.execute_sell(
                position,
                market_price,
                position.remaining_quantity,
            )

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

        gross_profit = (
            position.realized_profit +
            position.total_fees
        )

        trade = Trade(
            symbol=position.symbol,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            final_exit_price=market_price,
            initial_quantity=position.quantity,
            gross_profit=gross_profit,
            fees=position.total_fees,
            slippage_cost=(
                position.total_slippage_cost
            ),
            net_profit=(
                position.realized_profit
            ),
            exit_reason=reason,
            r_multiple=r_multiple,
        )

        self.trades.append(trade)

        self.risk_manager.record_trade(
            trade.net_profit
        )

        if (
            self.risk_manager
            .loss_streak_limit_reached()
        ):

            self.loss_cooldown_until = (
                timestamp +
                pd.Timedelta(
                    minutes=LOSS_COOLDOWN_MINUTES
                )
            )

        else:

            self.cooldown_until = (
                timestamp +
                pd.Timedelta(
                    minutes=TRADE_COOLDOWN_MINUTES
                )
            )

        self.position = None

    # ======================================
    # POSITION MANAGEMENT
    # ======================================

    def manage_position(
        self,
        row,
    ):

        if self.position is None:
            return

        position = self.position

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # ----------------------------------
        # Highest price
        # ----------------------------------

        position.highest_price = max(
            position.highest_price,
            high,
        )

        # ----------------------------------
        # STOP FIRST
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

        one_r = (
            position.entry_price +
            position.risk_per_unit *
            PARTIAL_EXIT_1_R
        )

        if (
            not position.partial_1_done
            and high >= one_r
        ):

            quantity = (
                position.quantity *
                PARTIAL_EXIT_1_SIZE
            )

            self.execute_sell(
                position,
                one_r,
                quantity,
            )

            position.partial_1_done = True

            # Break-even including approximate fee.
            break_even = (
                position.entry_price *
                (1 + FEE_RATE)
            )

            position.current_stop = max(
                position.current_stop,
                break_even,
            )

        # ----------------------------------
        # 2R
        # ----------------------------------

        two_r = (
            position.entry_price +
            position.risk_per_unit *
            PARTIAL_EXIT_2_R
        )

        if (
            not position.partial_2_done
            and high >= two_r
        ):

            quantity = (
                position.quantity *
                PARTIAL_EXIT_2_SIZE
            )

            self.execute_sell(
                position,
                two_r,
                quantity,
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
                    (
                        atr *
                        TRAILING_ATR_MULTIPLIER
                    )
                )

                position.current_stop = max(
                    position.current_stop,
                    trailing_stop,
                )

        # ----------------------------------
        # EMA exit
        # ----------------------------------

        if check_ema_exit(row):

            self.close_position(
                market_price=close,
                timestamp=row.name,
                reason="EMA_EXIT",
            )

            return

        # ----------------------------------
        # RSI exit
        # ----------------------------------

        if check_rsi_exit(row):

            self.close_position(
                market_price=close,
                timestamp=row.name,
                reason="RSI_EXIT",
            )

            return

        # ----------------------------------
        # TIME STOP
        # ----------------------------------

        duration = (
            row.name -
            position.entry_time
        ).total_seconds() / 60

        if duration >= MAX_TRADE_MINUTES:

            self.close_position(
                market_price=close,
                timestamp=row.name,
                reason="TIME_STOP",
            )

    # ======================================
    # RUN
    # ======================================

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
    ):

        df = calculate_indicators(df)

        self.balance = (
            self.starting_balance
        )

        self.position = None
        self.trades = []
        self.equity_curve = []

        self.current_day = None
        self.cooldown_until = None
        self.loss_cooldown_until = None

        for i in range(60, len(df)):

            current = df.iloc[i]

            timestamp = current.name

            self.update_day(timestamp)

            # ------------------------------
            # Existing position
            # ------------------------------

            if self.position is not None:

                self.manage_position(
                    current
                )

            # ------------------------------
            # Daily loss protection
            # ------------------------------

            trading_blocked = (
                self.risk_manager
                .daily_loss_limit_reached(
                    self.balance
                )
            )

            # ------------------------------
            # Loss cooldown
            # ------------------------------

            if (
                self.loss_cooldown_until
                is not None
                and
                timestamp <
                self.loss_cooldown_until
            ):

                trading_blocked = True

            # ------------------------------
            # Normal cooldown
            # ------------------------------

            if (
                self.cooldown_until
                is not None
                and
                timestamp <
                self.cooldown_until
            ):

                trading_blocked = True

            # ------------------------------
            # New entry
            # ------------------------------

            if (
                self.position is None
                and not trading_blocked
            ):

                if check_buy_signal(
                    df.iloc[: i + 1]
                ):

                    self.execute_buy(
                        symbol=symbol,
                        timestamp=timestamp,
                        close=float(
                            current["close"]
                        ),
                        atr=float(
                            current["atr_14"]
                        ),
                    )

            # ------------------------------
            # Equity
            # ------------------------------

            equity = self.balance

            if self.position is not None:

                equity += (
                    self.position
                    .remaining_quantity
                    *
                    float(current["close"])
                )

            self.equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                }
            )

        # ----------------------------------
        # Final position
        # ----------------------------------

        if self.position is not None:

            final = df.iloc[-1]

            self.close_position(
                market_price=float(
                    final["close"]
                ),
                timestamp=final.name,
                reason="END_OF_TEST",
            )

        return {
            "balance": self.balance,
            "trades": self.trades,
            "equity_curve": pd.DataFrame(
                self.equity_curve
            ),
        }
