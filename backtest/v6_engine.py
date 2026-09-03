# ==========================================
# 20to100 Trading Bot
# V6 Backtest Engine
#
# Adaptive Regime Filter
# ADX Rising
# Stronger EMA200 Slope
# ATR Volatility Filter
#
# Based on corrected V5 accounting
#
# IMPORTANT:
# - Cash / Equity accounting
# - Entry fee included
# - Exit fee included
# - Slippage included
# - 1% risk per trade
# - ATR stop
# - ATR trailing
# - Daily loss limit
# - Consecutive loss cooldown
# - Global drawdown kill switch
# - Next-candle entry
# - No look-ahead
# ==========================================

from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from strategy.strategy_v6 import buy_signal


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

class V6BacktestEngine:

    def __init__(
        self,
        starting_balance: float = 20.0,
        risk_per_trade: float = 0.01,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        atr_stop_multiplier: float = 3.0,
        trailing_atr_multiplier: float = 3.0,
        adx_min: float = 20.0,
        max_daily_loss: float = 0.05,
        max_consecutive_losses: int = 3,
        loss_cooldown_bars: int = 24,

        # Backward compatibility:
        # Some V6 multi-asset scripts use
        # "cooldown_bars".
        cooldown_bars: int = None,

        global_max_drawdown: float = 0.20,
        variant: str = "V6_A",
    ):

        # ======================================
        # BASIC PARAMETERS
        # ======================================

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

        # ======================================
        # COOLDOWN COMPATIBILITY
        # ======================================

        self.loss_cooldown_bars = int(
            loss_cooldown_bars
        )

        if cooldown_bars is not None:

            self.loss_cooldown_bars = int(
                cooldown_bars
            )

        self.global_max_drawdown = float(
            global_max_drawdown
        )

        self.variant = str(
            variant
        ).upper()

        # ======================================
        # VALIDATE VARIANT
        # ======================================

        valid_variants = {
            "V6_A",
            "V6_B",
            "V6_C",
        }

        if self.variant not in valid_variants:

            raise ValueError(
                f"Unbekannte V6 Variante: "
                f"{self.variant}. "
                f"Erlaubt: {valid_variants}"
            )

        # ======================================
        # ACCOUNT
        #
        # balance = available cash
        #
        # position value is NOT included in
        # balance while position is open.
        #
        # equity = cash + current position
        # market value.
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

        self.day_start_equity = (
            self.starting_balance
        )

        self.peak_equity = (
            self.starting_balance
        )

        self.kill_switch = False

        self._current_index = 0

    # ==========================================
    # EQUITY
    # ==========================================

    def _calculate_equity(
        self,
        market_price: float
    ):

        equity = self.balance

        if self.position is not None:

            equity += (
                self.position["quantity"]
                *
                float(market_price)
            )

        return float(
            equity
        )

    # ==========================================
    # DAILY RESET
    # ==========================================

    def _reset_day_if_needed(
        self,
        timestamp,
        equity
    ):

        day = pd.Timestamp(
            timestamp
        ).date()

        if self.current_day != day:

            self.current_day = day

            self.day_start_equity = float(
                equity
            )

    # ==========================================
    # DAILY LOSS
    # ==========================================

    def _daily_loss_limit_hit(
        self,
        equity
    ):

        if self.day_start_equity <= 0:

            return True

        loss = (
            self.day_start_equity
            -
            equity
        ) / self.day_start_equity

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
            -
            equity
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

        # ======================================
        # RAW ENTRY
        # ======================================

        entry_raw = float(
            entry_row["open"]
        )

        # ======================================
        # BUY SLIPPAGE
        # ======================================

        entry_price = (
            entry_raw
            *
            (
                1.0
                +
                self.slippage_rate
            )
        )

        # ======================================
        # ATR
        # ======================================

        atr = float(
            signal_row["atr_14"]
        )

        if atr <= 0:

            return

        # ======================================
        # INITIAL ATR STOP
        # ======================================

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
        # MONEY RISK
        # ======================================

        risk_money = (
            self.balance
            *
            self.risk_per_trade
        )

        if risk_money <= 0:

            return

        # ======================================
        # RISK PER UNIT
        # ======================================

        risk_per_unit = (
            entry_price
            -
            stop_price
        )

        if risk_per_unit <= 0:

            return

        # ======================================
        # POSITION SIZE BY RISK
        # ======================================

        qty_by_risk = (
            risk_money
            /
            risk_per_unit
        )

        # ======================================
        # POSITION SIZE BY CASH
        #
        # Cash must cover:
        #
        # notional
        # +
        # entry fee
        # ======================================

        qty_by_cash = (
            self.balance
            /
            (
                entry_price
                *
                (
                    1.0
                    +
                    self.fee_rate
                )
            )
        )

        quantity = min(
            qty_by_risk,
            qty_by_cash
        )

        if quantity <= 0:

            return

        # ======================================
        # ENTRY NOTIONAL
        # ======================================

        notional = (
            quantity
            *
            entry_price
        )

        # ======================================
        # ENTRY FEE
        # ======================================

        entry_fee = (
            notional
            *
            self.fee_rate
        )

        # ======================================
        # TOTAL CASH REQUIRED
        # ======================================

        total_entry_cost = (
            notional
            +
            entry_fee
        )

        if total_entry_cost > self.balance:

            return

        # ======================================
        # REMOVE COMPLETE POSITION VALUE
        # + ENTRY FEE FROM CASH
        # ======================================

        self.balance -= (
            total_entry_cost
        )

        # ======================================
        # STORE POSITION
        # ======================================

        self.position = {

            "entry_time":
                timestamp,

            "entry_price":
                entry_price,

            "quantity":
                quantity,

            "initial_stop":
                stop_price,

            "stop":
                stop_price,

            "risk_per_unit":
                risk_per_unit,

            "entry_fee":
                entry_fee,

            "highest":
                entry_price,

            "entry_atr":
                atr,
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

        # ======================================
        # SELL SLIPPAGE
        # ======================================

        exit_price = (
            float(raw_exit_price)
            *
            (
                1.0
                -
                self.slippage_rate
            )
        )

        # ======================================
        # GROSS PROFIT
        # ======================================

        gross = (
            exit_price
            -
            pos["entry_price"]
        ) * pos["quantity"]

        # ======================================
        # EXIT NOTIONAL
        # ======================================

        exit_notional = (
            exit_price
            *
            pos["quantity"]
        )

        # ======================================
        # EXIT FEE
        # ======================================

        exit_fee = (
            exit_notional
            *
            self.fee_rate
        )

        # ======================================
        # TOTAL FEES
        # ======================================

        entry_fee = float(
            pos["entry_fee"]
        )

        fees = (
            entry_fee
            +
            exit_fee
        )

        # ======================================
        # SLIPPAGE COST
        # ======================================

        theoretical_entry = (
            pos["entry_price"]
            /
            (
                1.0
                +
                self.slippage_rate
            )
        )

        theoretical_exit = (
            exit_price
            /
            (
                1.0
                -
                self.slippage_rate
            )
        )

        entry_slippage = (
            abs(
                theoretical_entry
                -
                pos["entry_price"]
            )
            *
            pos["quantity"]
        )

        exit_slippage = (
            abs(
                theoretical_exit
                -
                exit_price
            )
            *
            pos["quantity"]
        )

        slippage_cost = (
            entry_slippage
            +
            exit_slippage
        )

        # ======================================
        # NET PROFIT
        #
        # Entry + exit fees included.
        # ======================================

        net = (
            gross
            -
            entry_fee
            -
            exit_fee
        )

        # ======================================
        # RETURN SALE PROCEEDS TO CASH
        # ======================================

        self.balance += (
            exit_notional
            -
            exit_fee
        )

        # ======================================
        # INITIAL RISK
        # ======================================

        initial_risk = (
            pos["risk_per_unit"]
            *
            pos["quantity"]
        )

        # ======================================
        # R MULTIPLE
        # ======================================

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

        # ======================================
        # CLOSE POSITION
        # ======================================

        self.position = None

    # ==========================================
    # RUN
    # ==========================================

    def run(
        self,
        df: pd.DataFrame
    ):

        if len(df) < 300:

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
                        data["timestamp"],
                        utc=True
                    )
                )

                data = data.set_index(
                    "timestamp"
                )

            else:

                raise ValueError(
                    "DataFrame benötigt einen "
                    "DatetimeIndex oder eine "
                    "timestamp-Spalte."
                )

        else:

            # ==================================
            # MAKE UTC AWARE
            # ==================================

            if data.index.tz is None:

                data.index = (
                    data.index
                    .tz_localize("UTC")
                )

            else:

                data.index = (
                    data.index
                    .tz_convert("UTC")
                )

        # ======================================
        # SORT
        # ======================================

        data = data.sort_index()

        # ======================================
        # REQUIRED V6 COLUMNS
        # ======================================

        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "atr_14",
            "adx_14",
            "ema_100",
            "ema_200",
            "ema_200_slope",
            "ema_200_slope_reference",
            "atr_14_ma50",
            "donchian_high_20",
        ]

        missing = [
            column
            for column
            in required_columns
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Fehlende V6-Indikator-Spalten: "
                f"{missing}"
            )

        # ======================================
        # MAIN LOOP
        #
        # i-2 = previous previous candle
        # i-1 = last closed candle
        # i   = current candle
        #
        # Signal on i-1
        # Entry on i open
        # ======================================

        for i in range(
            2,
            len(data)
        ):

            self._current_index = i

            row = data.iloc[i]

            previous = data.iloc[
                i - 1
            ]

            previous_previous = data.iloc[
                i - 2
            ]

            timestamp = data.index[i]

            current_price = float(
                row["close"]
            )

            # ==================================
            # CURRENT EQUITY
            # ==================================

            equity = (
                self._calculate_equity(
                    current_price
                )
            )

            # ==================================
            # DAILY RESET
            # ==================================

            self._reset_day_if_needed(
                timestamp,
                equity
            )

            # ==================================
            # EQUITY CURVE
            # ==================================

            equity = (
                self._calculate_equity(
                    current_price
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
            # GLOBAL DD
            # ==================================

            if self._global_drawdown_hit(
                equity
            ):

                self.kill_switch = True

            # ==================================
            # MANAGE OPEN POSITION
            # ==================================

            if self.position is not None:

                pos = self.position

                # ==================================
                # STOP FIRST
                #
                # Conservative:
                # if LOW touches stop,
                # stop executes.
                # ==================================

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

                # ==================================
                # UPDATE HIGHEST PRICE
                # ==================================

                pos["highest"] = max(
                    pos["highest"],
                    float(row["high"])
                )

                # ==================================
                # TRAILING STOP
                # ==================================

                current_atr = float(
                    row["atr_14"]
                )

                if current_atr > 0:

                    candidate = (
                        pos["highest"]
                        -
                        current_atr
                        *
                        self.trailing_atr_multiplier
                    )

                    if candidate > pos["stop"]:

                        pos["stop"] = (
                            candidate
                        )

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
                        current_price,
                        "END"
                    )

                continue

            # ==================================
            # GLOBAL KILL SWITCH
            # ==================================

            if self.kill_switch:

                continue

            # ==================================
            # DAILY LOSS LIMIT
            # ==================================

            if self._daily_loss_limit_hit(
                equity
            ):

                continue

            # ==================================
            # CONSECUTIVE LOSS COOLDOWN
            # ==================================

            if (
                i
                <
                self.cooldown_until
            ):

                continue

            # ==================================
            # V6 SIGNAL
            # ==================================

            signal = buy_signal(
                previous,
                previous_previous,
                variant=self.variant,
                adx_min=self.adx_min
            )

            # ==================================
            # ENTRY
            # ==================================

            if signal:

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

        # ======================================
        # PROFITS
        # ======================================

        profits = [
            trade.net_profit
            for trade in self.trades
        ]

        # ======================================
        # WINS
        # ======================================

        wins = [
            profit
            for profit in profits
            if profit > 0
        ]

        # ======================================
        # LOSSES
        # ======================================

        losses = [
            profit
            for profit in profits
            if profit < 0
        ]

        # ======================================
        # PROFIT FACTOR
        # ======================================

        gross_profit = sum(
            wins
        )

        gross_loss = abs(
            sum(
                losses
            )
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
        # TOTAL FEES
        # ======================================

        total_fees = sum(
            trade.fees
            for trade in self.trades
        )

        # ======================================
        # TOTAL SLIPPAGE
        # ======================================

        total_slippage = sum(
            trade.slippage_cost
            for trade in self.trades
        )

        # ======================================
        # RESULT
        # ======================================

        return {

            "variant":
                self.variant,

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
                total_fees,

            "slippage_cost":
                total_slippage,

            "trades_detail":
                [
                    asdict(trade)
                    for trade in self.trades
                ],

            "equity_curve":
                self.equity_curve,
        }
