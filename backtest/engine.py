# ==========================================
# 20to100 Trading Bot
# Backtest Engine - Exit Grid v3.0
# ==========================================

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    FEE_RATE,
    SLIPPAGE_RATE,
    RISK_PER_TRADE,
    ATR_STOP_MULTIPLIER,
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

from strategy.indicators import calculate_indicators
from strategy.signals import check_buy_signal

from risk.risk_manager import (
    RiskManager,
    RiskConfig,
)


# ==========================================
# POSITION
# ==========================================

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


# ==========================================
# TRADE
# ==========================================

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


# ==========================================
# BACKTEST ENGINE
# ==========================================

class BacktestEngine:

    def __init__(
        self,
        starting_balance: float = 20.0,

        atr_stop_multiplier: float = None,

        partial_exit_1_r: float = None,
        partial_exit_1_size: float = None,

        partial_exit_2_r: float = None,
        partial_exit_2_size: float = None,

        trailing_atr_multiplier: float = None,

        max_trade_minutes: int = None,

        fee_rate: float = None,
        slippage_rate: float = None,
    ):

        self.starting_balance = float(
            starting_balance
        )

        self.balance = float(
            starting_balance
        )

        # ==================================
        # EXIT PARAMETERS
        # ==================================

        self.atr_stop_multiplier = (
            float(ATR_STOP_MULTIPLIER)
            if atr_stop_multiplier is None
            else float(atr_stop_multiplier)
        )

        self.partial_exit_1_r = (
            float(PARTIAL_EXIT_1_R)
            if partial_exit_1_r is None
            else float(partial_exit_1_r)
        )

        self.partial_exit_1_size = (
            float(PARTIAL_EXIT_1_SIZE)
            if partial_exit_1_size is None
            else float(partial_exit_1_size)
        )

        self.partial_exit_2_r = (
            float(PARTIAL_EXIT_2_R)
            if partial_exit_2_r is None
            else float(partial_exit_2_r)
        )

        self.partial_exit_2_size = (
            float(PARTIAL_EXIT_2_SIZE)
            if partial_exit_2_size is None
            else float(partial_exit_2_size)
        )

        self.trailing_atr_multiplier = (
            float(TRAILING_ATR_MULTIPLIER)
            if trailing_atr_multiplier is None
            else float(trailing_atr_multiplier)
        )

        self.max_trade_minutes = (
            int(MAX_TRADE_MINUTES)
            if max_trade_minutes is None
            else int(max_trade_minutes)
        )

        # ==================================
        # COST PARAMETERS
        # ==================================

        self.fee_rate = (
            float(FEE_RATE)
            if fee_rate is None
            else float(fee_rate)
        )

        self.slippage_rate = (
            float(SLIPPAGE_RATE)
            if slippage_rate is None
            else float(slippage_rate)
        )

        # ==================================
        # STATE
        # ==================================

        self.position: Optional[Position] = None

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

        # ==================================
        # DIAGNOSTICS
        # ==================================

        self.signal_count = 0
        self.entry_accepted = 0
        self.entry_rejected = 0

        self.rejection_reasons = {}

    # ======================================
    # DAY RESET
    # ======================================

    def update_day(self, timestamp):

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
            return False

        if pd.isna(atr) or atr <= 0:
            self.entry_rejected += 1
            return False

        if self.balance <= 0:
            self.entry_rejected += 1
            return False

        # ----------------------------------
        # Entry with slippage
        # ----------------------------------

        entry_price = (
            float(close)
            *
            (1 + self.slippage_rate)
        )

        # ----------------------------------
        # ATR stop
        # ----------------------------------

        stop_distance = (
            float(atr)
            *
            self.atr_stop_multiplier
        )

        if stop_distance <= 0:
            self.entry_rejected += 1
            return False

        stop_price = (
            entry_price -
            stop_distance
        )

        if stop_price <= 0:
            self.entry_rejected += 1
            return False

        # ----------------------------------
        # Position size
        # ----------------------------------

        quantity = (
            self.risk_manager
            .calculate_position_size(
                balance=self.balance,
                entry_price=entry_price,
                stop_price=stop_price,
                fee_rate=self.fee_rate,
            )
        )

        if quantity <= 0:
            self.entry_rejected += 1
            return False

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
            self.entry_rejected += 1
            return False

        # ----------------------------------
        # Open position
        # ----------------------------------

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

        self.entry_accepted += 1

        return True

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

        if quantity <= 0:
            return

        # ----------------------------------
        # Exit with slippage
        # ----------------------------------

        exit_price = (
            float(market_price)
            *
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
            self.fee_rate
        )

        gross_profit = (
            exit_price -
            position.entry_price
        ) * quantity

        net_profit = (
            gross_profit
            -
            entry_fee
            -
            exit_fee
        )

        # ----------------------------------
        # Cost accounting
        # ----------------------------------

        entry_slippage = (
            quantity
            *
            position.entry_price
            *
            self.slippage_rate
        )

        exit_slippage = (
            quantity
            *
            float(market_price)
            *
            self.slippage_rate
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
    # CLOSE POSITION
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
                position=position,
                market_price=market_price,
                quantity=(
                    position.remaining_quantity
                ),
            )

        initial_risk = (
            position.risk_per_unit
            *
            position.quantity
        )

        if initial_risk > 0:

            r_multiple = (
                position.realized_profit
                /
                initial_risk
            )

        else:

            r_multiple = 0.0

        gross_profit = (
            position.realized_profit
            +
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

        # ----------------------------------
        # Cooldowns
        # ----------------------------------

        if (
            self.risk_manager
            .loss_streak_limit_reached()
        ):

            self.loss_cooldown_until = (
                timestamp
                +
                pd.Timedelta(
                    minutes=(
                        LOSS_COOLDOWN_MINUTES
                    )
                )
            )

        else:

            self.cooldown_until = (
                timestamp
                +
                pd.Timedelta(
                    minutes=(
                        TRADE_COOLDOWN_MINUTES
                    )
                )
            )

        self.position = None

    # ======================================
    # MANAGE POSITION
    # ======================================

    def manage_position(self, row):

        if self.position is None:
            return

        position = self.position

        timestamp = row.name

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        atr = row.get(
            "atr_14",
            0.0,
        )

        if pd.isna(atr):
            atr = 0.0

        # ----------------------------------
        # Highest price
        # ----------------------------------

        position.highest_price = max(
            position.highest_price,
            high,
        )

        # ==================================
        # STOP LOSS
        # ==================================

        if low <= position.current_stop:

            self.close_position(
                market_price=position.current_stop,
                timestamp=timestamp,
                reason="STOP_LOSS",
            )

            return

        risk_distance = (
            position.risk_per_unit
        )

        if risk_distance <= 0:
            return

        # ==================================
        # R LEVELS
        # ==================================

        one_r_price = (
            position.entry_price
            +
            risk_distance
            *
            self.partial_exit_1_r
        )

        two_r_price = (
            position.entry_price
            +
            risk_distance
            *
            self.partial_exit_2_r
        )

        # ==================================
        # PARTIAL EXIT 1
        # ==================================

        if (
            not position.partial_1_done
            and high >= one_r_price
        ):

            quantity = (
                position.quantity
                *
                self.partial_exit_1_size
            )

            self.execute_sell(
                position=position,
                market_price=one_r_price,
                quantity=quantity,
            )

            position.partial_1_done = True

            # Move stop to break-even
            break_even = (
                position.entry_price
                *
                (1 + self.fee_rate)
            )

            position.current_stop = max(
                position.current_stop,
                break_even,
            )

        # ==================================
        # PARTIAL EXIT 2
        # ==================================

        if (
            not position.partial_2_done
            and high >= two_r_price
        ):

            quantity = (
                position.quantity
                *
                self.partial_exit_2_size
            )

            self.execute_sell(
                position=position,
                market_price=two_r_price,
                quantity=quantity,
            )

            position.partial_2_done = True

        # ==================================
        # TRAILING STOP
        # ==================================

        if (
            position.partial_1_done
            and atr > 0
        ):

            trailing_stop = (
                position.highest_price
                -
                (
                    atr
                    *
                    self.trailing_atr_multiplier
                )
            )

            position.current_stop = max(
                position.current_stop,
                trailing_stop,
            )

        # ==================================
        # TAKE PROFIT
        # ==================================
        #
        # TP = 2 x configured partial exit 2 R
        #
        # Example:
        # partial 2 = 2R
        # final TP  = 4R
        #
        # This makes TP configurable.
        # ==================================

        take_profit_r = (
            self.partial_exit_2_r * 2.0
        )

        take_profit_price = (
            position.entry_price
            +
            risk_distance
            *
            take_profit_r
        )

        if high >= take_profit_price:

            self.close_position(
                market_price=take_profit_price,
                timestamp=timestamp,
                reason="TAKE_PROFIT",
            )

            return

        # ==================================
        # TIME STOP
        # ==================================

        elapsed_minutes = (
            timestamp -
            position.entry_time
        ).total_seconds() / 60.0

        if (
            elapsed_minutes
            >=
            self.max_trade_minutes
        ):

            self.close_position(
                market_price=close,
                timestamp=timestamp,
                reason="TIME_STOP",
            )

    # ======================================
    # RUN
    # ======================================

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        conditions: list[str],
    ):

        df = calculate_indicators(
            df.copy()
        )

        # ----------------------------------
        # Reset
        # ----------------------------------

        self.balance = (
            self.starting_balance
        )

        self.position = None

        self.trades = []

        self.equity_curve = []

        self.current_day = None

        self.cooldown_until = None

        self.loss_cooldown_until = None

        self.signal_count = 0

        self.entry_accepted = 0

        self.entry_rejected = 0

        # ----------------------------------
        # Main loop
        # ----------------------------------

        for i in range(
            60,
            len(df),
        ):

            current = df.iloc[i]

            timestamp = current.name

            self.update_day(
                timestamp
            )

            # ==============================
            # Manage position
            # ==============================

            if self.position is not None:

                self.manage_position(
                    current
                )

            # ==============================
            # Risk checks
            # ==============================

            trading_blocked = (
                self.risk_manager
                .daily_loss_limit_reached(
                    self.balance
                )
            )

            if (
                self.loss_cooldown_until
                is not None
                and
                timestamp <
                self.loss_cooldown_until
            ):
                trading_blocked = True

            if (
                self.cooldown_until
                is not None
                and
                timestamp <
                self.cooldown_until
            ):
                trading_blocked = True

            # ==============================
            # Entry
            # ==============================

            if (
                self.position is None
                and not trading_blocked
            ):

                history = df.iloc[
                    : i + 1
                ]

                try:

                    conditions_result = (
                        check_buy_signal(
                            history
                        )
                    )

                    signal = all(
                        conditions_result.get(
                            condition,
                            False,
                        )
                        for condition in conditions
                    )

                except Exception:

                    signal = False

                if signal:

                    self.signal_count += 1

                    atr = current.get(
                        "atr_14",
                        None,
                    )

                    self.execute_buy(
                        symbol=symbol,
                        timestamp=timestamp,
                        close=float(
                            current["close"]
                        ),
                        atr=atr,
                    )

            # ==============================
            # Equity
            # ==============================

            equity = self.balance

            if self.position is not None:

                equity += (
                    self.position
                    .remaining_quantity
                    *
                    float(
                        current["close"]
                    )
                )

            self.equity_curve.append(
                {
                    "timestamp":
                        timestamp,

                    "equity":
                        equity,
                }
            )

        # ==================================
        # Close final position
        # ==================================

        if self.position is not None:

            final = df.iloc[-1]

            self.close_position(
                market_price=float(
                    final["close"]
                ),
                timestamp=final.name,
                reason="END_OF_TEST",
            )

        return self


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(engine):

    trades = engine.trades

    if not trades:

        return {
            "final":
                engine.balance,

            "return_pct":
                (
                    engine.balance
                    /
                    engine.starting_balance
                    -
                    1
                ) * 100,

            "trades": 0,
            "wins": 0,
            "losses": 0,

            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,

            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_r": 0.0,
            "max_drawdown": 0.0,

            "fees": 0.0,
            "slippage": 0.0,
        }

    profits = [
        float(t.net_profit)
        for t in trades
    ]

    winners = [
        p for p in profits
        if p > 0
    ]

    losers = [
        p for p in profits
        if p < 0
    ]

    gross_profit = sum(
        winners
    )

    gross_loss = abs(
        sum(losers)
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    else:

        profit_factor = 0.0

    win_rate = (
        len(winners)
        /
        len(profits)
        *
        100
    )

    average_win = (
        sum(winners)
        /
        len(winners)
        if winners
        else 0.0
    )

    average_loss = (
        sum(losers)
        /
        len(losers)
        if losers
        else 0.0
    )

    expectancy = (
        sum(profits)
        /
        len(profits)
    )

    average_r = (
        sum(
            t.r_multiple
            for t in trades
        )
        /
        len(trades)
    )

    equity = pd.DataFrame(
        engine.equity_curve
    )

    if equity.empty:

        max_drawdown = 0.0

    else:

        peak = (
            equity["equity"]
            .cummax()
        )

        drawdown = (
            equity["equity"]
            /
            peak
            -
            1
        )

        max_drawdown = (
            abs(
                drawdown.min()
            )
            *
            100
        )

    return {

        "final":
            engine.balance,

        "return_pct":
            (
                engine.balance
                /
                engine.starting_balance
                -
                1
            ) * 100,

        "trades":
            len(trades),

        "wins":
            len(winners),

        "losses":
            len(losers),

        "win_rate":
            win_rate,

        "average_win":
            average_win,

        "average_loss":
            average_loss,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "average_r":
            average_r,

        "max_drawdown":
            max_drawdown,

        "fees":
            sum(
                t.fees
                for t in trades
            ),

        "slippage":
            sum(
                t.slippage_cost
                for t in trades
            ),
    }
