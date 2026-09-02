# ==========================================
# 20to100 Trading Bot
# V5 Backtest Engine
#
# Regime Filter
# 1% Risk per Trade
# ATR Stop
# ATR Trailing
# ==========================================

from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from strategy.strategy_v5 import buy_signal


# ==========================================
# TRADE
# ==========================================

@dataclass
class Trade:

    entry_time: object
    exit_time: object

    entry_price: float
    exit_price: float

    quantity: float

    initial_risk_usdt: float

    gross_profit: float

    fees: float

    slippage_cost: float

    net_profit: float

    r_multiple: float

    exit_reason: str


# ==========================================
# ENGINE
# ==========================================

class V5BacktestEngine:

    def __init__(
        self,
        starting_balance: float = 20.0,

        risk_per_trade: float = 0.01,

        fee_rate: float = 0.001,

        slippage_rate: float = 0.0005,

        atr_stop_multiplier: float = 2.5,

        trailing_atr_multiplier: float = 2.5,

        adx_min: float = 20.0,

        max_daily_loss: float = 0.05,

        max_consecutive_losses: int = 3,

        loss_cooldown_bars: int = 24,

        global_max_drawdown: float = 0.20,
    ):

        self.starting_balance = float(
            starting_balance
        )

        self.risk_per_trade = float(
            risk_per_trade
        )

        self.fee_rate = float(
            fee_rate
        )

        self.slippage_rate = float(
            slippage_rate
        )

        self.atr_stop_multiplier = float(
            atr_stop_multiplier
        )

        self.trailing_atr_multiplier = float(
            trailing_atr_multiplier
        )

        self.adx_min = float(
            adx_min
        )

        self.max_daily_loss = float(
            max_daily_loss
        )

        self.max_consecutive_losses = int(
            max_consecutive_losses
        )

        self.loss_cooldown_bars = int(
            loss_cooldown_bars
        )

        self.global_max_drawdown = float(
            global_max_drawdown
        )

        # ======================================
        # ACCOUNT
        # ======================================

        self.balance = (
            self.starting_balance
        )

        self.position: Optional[dict] = None

        self.trades = []

        self.equity_curve = []

        # ======================================
        # RISK CONTROL
        # ======================================

        self.consecutive_losses = 0

        self.cooldown_until = -1

        self.current_day = None

        self.day_start_balance = (
            self.starting_balance
        )

        self.peak_equity = (
            self.starting_balance
        )

        self.kill_switch = False

        self._current_index = 0

    # ==========================================
    # DAILY RESET
    # ==========================================

    def _reset_day_if_needed(
        self,
        timestamp
    ):

        day = pd.Timestamp(
            timestamp
        ).date()

        if self.current_day != day:

            self.current_day = day

            self.day_start_balance = (
                self.balance
            )

    # ==========================================
    # DAILY LOSS
    # ==========================================

    def _daily_loss_limit_hit(self):

        if self.day_start_balance <= 0:
            return True

        loss = (
            self.day_start_balance
            - self.balance
        ) / self.day_start_balance

        return (
            loss
            >=
            self.max_daily_loss
        )

    # ==========================================
    # GLOBAL DRAWDOWN
    # ==========================================

    def _global_drawdown_hit(
        self,
        equity
    ):

        self.peak_equity = max(
            self.peak_equity,
            equity
        )

        if self.peak_equity <= 0:
            return True

        drawdown = (
            self.peak_equity
            - equity
        ) / self.peak_equity

        return (
            drawdown
            >=
            self.global_max_drawdown
        )

    # ==========================================
    # ENTRY
    # ==========================================

    def _enter(
        self,
        timestamp,
        signal_row,
        entry_row
    ):

        entry_raw = float(
            entry_row["open"]
        )

        # Slippage on BUY
        entry_price = (
            entry_raw
            *
            (1.0 + self.slippage_rate)
        )

        atr = float(
            signal_row["atr_14"]
        )

        # ATR STOP
        stop_price = (
            entry_price
            -
            atr
            *
            self.atr_stop_multiplier
        )

        if stop_price <= 0:
            return

        if stop_price >= entry_price:
            return

        # ======================================
        # RISK MONEY
        # ======================================

        risk_money = (
            self.balance
            *
            self.risk_per_trade
        )

        risk_per_unit = (
            entry_price
            -
            stop_price
        )

        if risk_per_unit <= 0:
            return

        # Position size based on risk
        qty_by_risk = (
            risk_money
            /
            risk_per_unit
        )

        # Position size based on available cash
        qty_by_cash = (
            self.balance
            /
            (
                entry_price
                *
                (1.0 + self.fee_rate)
            )
        )

        quantity = min(
            qty_by_risk,
            qty_by_cash
        )

        if quantity <= 0:
            return

        # ======================================
        # ENTRY FEE
        # ======================================

        notional = (
            quantity
            *
            entry_price
        )

        entry_fee = (
            notional
            *
            self.fee_rate
        )

        if entry_fee >= self.balance:
            return

        self.balance -= entry_fee

        # ======================================
        # STORE POSITION
        # ======================================

        self.position = {

            "entry_time": timestamp,

            "entry_price": entry_price,

            "quantity": quantity,

            "initial_stop": stop_price,

            "stop": stop_price,

            "risk_per_unit": risk_per_unit,

            "entry_fee": entry_fee,

            "highest": entry_price,

            "entry_atr": atr,
        }

    # ==========================================
    # EXIT
    # ==========================================

    def _exit(
        self,
        timestamp,
        raw_exit_price,
        reason
    ):

        pos = self.position

        if pos is None:
            return

        # Slippage on SELL
        exit_price = (
            float(raw_exit_price)
            *
            (1.0 - self.slippage_rate)
        )

        # ======================================
        # GROSS P/L
        # ======================================

        gross = (
            exit_price
            -
            pos["entry_price"]
        ) * pos["quantity"]

        exit_notional = (
            exit_price
            *
            pos["quantity"]
        )

        exit_fee = (
            exit_notional
            *
            self.fee_rate
        )

        fees = (
            pos["entry_fee"]
            +
            exit_fee
        )

        # ======================================
        # SLIPPAGE COST
        # ======================================

        theoretical_entry = (
            pos["entry_price"]
            /
            (1.0 + self.slippage_rate)
        )

        theoretical_exit = (
            exit_price
            /
            (1.0 - self.slippage_rate)
        )

        slippage_cost = (

            abs(
                theoretical_entry
                -
                pos["entry_price"]
            )
            *
            pos["quantity"]

            +

            abs(
                theoretical_exit
                -
                exit_price
            )
            *
            pos["quantity"]
        )

        # ======================================
        # NET PROFIT
        # ======================================

        net = (
            gross
            -
            exit_fee
        )

        # Return position capital
        self.balance += (
            exit_notional
            -
            exit_fee
        )

        # ======================================
        # R MULTIPLE
        # ======================================

        initial_risk = (
            pos["risk_per_unit"]
            *
            pos["quantity"]
        )

        if initial_risk > 0:

            r_multiple = (
                net
                /
                initial_risk
            )

        else:

            r_multiple = 0.0

        # ======================================
        # SAVE TRADE
        # ======================================

        trade = Trade(

            entry_time=pos[
                "entry_time"
            ],

            exit_time=timestamp,

            entry_price=pos[
                "entry_price"
            ],

            exit_price=exit_price,

            quantity=pos[
                "quantity"
            ],

            initial_risk_usdt=initial_risk,

            gross_profit=gross,

            fees=fees,

            slippage_cost=slippage_cost,

            net_profit=net,

            r_multiple=r_multiple,

            exit_reason=reason,
        )

        self.trades.append(
            trade
        )

        # ======================================
        # LOSS CONTROL
        # ======================================

        if net < 0:

            self.consecutive_losses += 1

            if (
                self.consecutive_losses
                >=
                self.max_consecutive_losses
            ):

                self.cooldown_until = (
                    self._current_index
                    +
                    self.loss_cooldown_bars
                )

        else:

            self.consecutive_losses = 0

        self.position = None

    # ==========================================
    # RUN
    # ==========================================

    def run(
        self,
        df: pd.DataFrame
    ):

        if len(df) < 250:
            return self._result()

        data = df.copy()

        # ======================================
        # DATETIME INDEX
        # ======================================

        if not isinstance(
            data.index,
            pd.DatetimeIndex
        ):

            if "timestamp" in data.columns:

                data["timestamp"] = (
                    pd.to_datetime(
                        data["timestamp"]
                    )
                )

                data = data.set_index(
                    "timestamp"
                )

        data = data.sort_index()

        # ======================================
        # MAIN LOOP
        # ======================================

        for i in range(
            1,
            len(data)
        ):

            self._current_index = i

            row = data.iloc[i]

            previous = data.iloc[i - 1]

            timestamp = data.index[i]

            self._reset_day_if_needed(
                timestamp
            )

            # ==================================
            # MARK TO MARKET
            # ==================================

            equity = self.balance

            if self.position is not None:

                equity += (
                    self.position[
                        "quantity"
                    ]
                    *
                    float(row["close"])
                )

            self.equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                }
            )

            # ==================================
            # GLOBAL DD CHECK
            # ==================================

            if self._global_drawdown_hit(
                equity
            ):

                self.kill_switch = True

            # ==================================
            # MANAGE POSITION
            # ==================================

            if self.position is not None:

                pos = self.position

                # STOP FIRST
                if (
                    float(row["low"])
                    <=
                    pos["stop"]
                ):

                    self._exit(
                        timestamp,
                        pos["stop"],
                        "ATR_STOP"
                    )

                    continue

                # Update highest price
                pos["highest"] = max(
                    pos["highest"],
                    float(row["high"])
                )

                # ==================================
                # TRAILING STOP
                # ==================================

                candidate = (
                    pos["highest"]
                    -
                    float(row["atr_14"])
                    *
                    self.trailing_atr_multiplier
                )

                if candidate > pos["stop"]:

                    pos["stop"] = candidate

                # ==================================
                # END OF TEST
                # ==================================

                if (
                    i
                    ==
                    len(data) - 1
                ):

                    self._exit(
                        timestamp,
                        float(row["close"]),
                        "END"
                    )

                continue

            # ==================================
            # RISK BLOCKS
            # ==================================

            if self.kill_switch:
                continue

            if self._daily_loss_limit_hit():
                continue

            if (
                i
                <
                self.cooldown_until
            ):
                continue

            # ==================================
            # SIGNAL ON PREVIOUS CLOSED CANDLE
            #
            # ENTRY ON CURRENT OPEN
            # ==================================

            if buy_signal(
                previous,
                adx_min=self.adx_min
            ):

                self._enter(
                    timestamp,
                    previous,
                    row
                )

        # ======================================
        # FINAL SAFETY EXIT
        # ======================================

        if self.position is not None:

            final_row = data.iloc[-1]

            self._current_index = (
                len(data) - 1
            )

            self._exit(
                data.index[-1],
                float(final_row["close"]),
                "END"
            )

        return self._result()

    # ==========================================
    # RESULT
    # ==========================================

    def _result(self):

        final_balance = float(
            self.balance
        )

        profits = [
            trade.net_profit
            for trade in self.trades
        ]

        wins = [
            profit
            for profit in profits
            if profit > 0
        ]

        losses = [
            profit
            for profit in profits
            if profit < 0
        ]

        # ======================================
        # PROFIT FACTOR
        # ======================================

        gross_profit = sum(wins)

        gross_loss = abs(
            sum(losses)
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                /
                gross_loss
            )

        elif gross_profit > 0:

            profit_factor = float(
                "inf"
            )

        else:

            profit_factor = 0.0

        # ======================================
        # EXPECTANCY
        # ======================================

        if profits:

            expectancy = (
                sum(profits)
                /
                len(profits)
            )

        else:

            expectancy = 0.0

        # ======================================
        # WIN RATE
        # ======================================

        if profits:

            win_rate = (
                len(wins)
                /
                len(profits)
                *
                100
            )

        else:

            win_rate = 0.0

        # ======================================
        # MAX DRAWDOWN
        # ======================================

        if self.equity_curve:

            equity = pd.Series(
                [
                    item["equity"]
                    for item
                    in self.equity_curve
                ],
                dtype=float
            )

            peaks = equity.cummax()

            drawdown = (
                equity
                -
                peaks
            ) / peaks * 100

            max_drawdown = float(
                drawdown.min()
            )

        else:

            max_drawdown = 0.0

        # ======================================
        # RESULT
        # ======================================

        return {

            "final_balance":
                final_balance,

            "profit":
                final_balance
                -
                self.starting_balance,

            "return_pct":
                (
                    (
                        final_balance
                        /
                        self.starting_balance
                    )
                    -
                    1.0
                )
                *
                100,

            "trades":
                len(self.trades),

            "wins":
                len(wins),

            "losses":
                len(losses),

            "win_rate":
                win_rate,

            "profit_factor":
                profit_factor,

            "expectancy":
                expectancy,

            "max_drawdown_pct":
                max_drawdown,

            "fees":
                sum(
                    trade.fees
                    for trade
                    in self.trades
                ),

            "slippage_cost":
                sum(
                    trade.slippage_cost
                    for trade
                    in self.trades
                ),

            "trades_detail":
                [
                    asdict(trade)
                    for trade
                    in self.trades
                ],

            "equity_curve":
                self.equity_curve,
        }
