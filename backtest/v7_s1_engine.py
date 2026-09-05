# ============================================================
# 20to100 Trading Bot
# V7 SURVIVAL ENGINE - S1
#
# V6-C bleibt vollständig eingefroren.
#
# V7-S1:
# - V6-C Entry Logic
# - zusätzlicher Signal-Quality-Filter
# - Breakout muss mindestens 0.10 ATR
#   über dem vorherigen Donchian-20-High liegen
#
# V7 Survival Layer bleibt unverändert:
# - dynamisches Risiko abhängig vom Drawdown
# - harte Kapital-Schutzstufen
# - tägliches Verlustlimit
# - Verlustserien-Schutz
# - globaler Drawdown-Kill-Switch
# - Gebühren
# - Slippage
# - ATR Stop
# - ATR Trailing Stop
# - Next-candle execution
# - kein Look-ahead
#
# Ziel:
# SURVIVE FIRST
# GROW SECOND
# ============================================================

from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from strategy.strategy_v7_s1 import buy_signal_v7_s1


# ============================================================
# TRADE
# ============================================================

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


# ============================================================
# ENGINE
# ============================================================

class V7SurvivalEngine:

    def __init__(
        self,

        starting_balance: float = 100.0,

        # ----------------------------------------------------
        # BASE RISK
        # ----------------------------------------------------

        base_risk_per_trade: float = 0.01,

        # ----------------------------------------------------
        # COSTS
        # ----------------------------------------------------

        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        atr_stop_multiplier: float = 3.0,
        trailing_atr_multiplier: float = 3.0,

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        adx_min: float = 20.0,
        variant: str = "V7_S1",

        # ----------------------------------------------------
        # DAILY PROTECTION
        # ----------------------------------------------------

        max_daily_loss: float = 0.05,

        # ----------------------------------------------------
        # LOSS STREAK
        # ----------------------------------------------------

        max_consecutive_losses: int = 3,
        loss_cooldown_bars: int = 24,

        # ----------------------------------------------------
        # GLOBAL PROTECTION
        # ----------------------------------------------------

        global_max_drawdown: float = 0.20,

        # ----------------------------------------------------
        # SURVIVAL RISK LEVELS
        # ----------------------------------------------------

        defensive_drawdown: float = 0.05,
        survival_drawdown: float = 0.10,
        critical_drawdown: float = 0.15,

    ):

        # ====================================================
        # BASIC
        # ====================================================

        self.starting_balance = float(
            starting_balance
        )

        self.base_risk_per_trade = float(
            base_risk_per_trade
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

        self.variant = str(
            variant
        ).upper()

        # ====================================================
        # PROTECTION
        # ====================================================

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

        self.defensive_drawdown = float(
            defensive_drawdown
        )

        self.survival_drawdown = float(
            survival_drawdown
        )

        self.critical_drawdown = float(
            critical_drawdown
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        if self.starting_balance <= 0:

            raise ValueError(
                "starting_balance muss > 0 sein."
            )

        if not 0 < self.base_risk_per_trade <= 0.10:

            raise ValueError(
                "base_risk_per_trade muss "
                "zwischen 0 und 10% liegen."
            )

        if self.variant not in {
            "V7_S1",
        }:

            raise ValueError(
                f"V7-S1 verwendet ausschließlich "
                f"V7_S1. Erhalten: {self.variant}"
            )

        # ====================================================
        # ACCOUNT
        # ====================================================

        self.balance = (
            self.starting_balance
        )

        self.position: Optional[dict] = None

        self.trades = []

        self.equity_curve = []

        # ====================================================
        # RISK STATE
        # ====================================================

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

        # ====================================================
        # STATISTICS
        # ====================================================

        self.normal_risk_trades = 0

        self.defensive_risk_trades = 0

        self.survival_risk_trades = 0

        self.critical_risk_trades = 0

    # ========================================================
    # EQUITY
    # ========================================================

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

    # ========================================================
    # CURRENT DRAWDOWN
    # ========================================================

    def _current_drawdown(
        self,
        equity: float
    ):

        if self.peak_equity <= 0:

            return 1.0

        return max(
            0.0,
            (
                self.peak_equity
                -
                equity
            )
            /
            self.peak_equity
        )

    # ========================================================
    # SURVIVAL RISK LEVEL
    # ========================================================

    def _risk_multiplier(
        self,
        equity: float
    ):

        drawdown = self._current_drawdown(
            equity
        )

        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        if drawdown < self.defensive_drawdown:

            return (
                1.00,
                "NORMAL"
            )

        # ----------------------------------------------------
        # DEFENSIVE
        # ----------------------------------------------------

        if drawdown < self.survival_drawdown:

            return (
                0.75,
                "DEFENSIVE"
            )

        # ----------------------------------------------------
        # SURVIVAL
        # ----------------------------------------------------

        if drawdown < self.critical_drawdown:

            return (
                0.50,
                "SURVIVAL"
            )

        # ----------------------------------------------------
        # CRITICAL
        # ----------------------------------------------------

        if drawdown < self.global_max_drawdown:

            return (
                0.25,
                "CRITICAL"
            )

        # ----------------------------------------------------
        # HALT
        # ----------------------------------------------------

        return (
            0.0,
            "HALT"
        )

    # ========================================================
    # DAILY RESET
    # ========================================================

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

    # ========================================================
    # DAILY LOSS
    # ========================================================

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

    # ========================================================
    # GLOBAL DRAWDOWN
    # ========================================================

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

    # ========================================================
    # ENTRY
    # ========================================================

    def _enter(
        self,
        timestamp,
        signal_row,
        entry_row,
        equity
    ):

        # ====================================================
        # RISK LEVEL
        # ====================================================

        risk_multiplier, risk_level = (
            self._risk_multiplier(
                equity
            )
        )

        if risk_multiplier <= 0:

            return

        effective_risk = (
            self.base_risk_per_trade
            *
            risk_multiplier
        )

        # ====================================================
        # ENTRY
        # ====================================================

        raw_entry = float(
            entry_row["open"]
        )

        entry_price = (
            raw_entry
            *
            (
                1.0
                +
                self.slippage_rate
            )
        )

        # ====================================================
        # ATR
        # ====================================================

        atr = float(
            signal_row["atr_14"]
        )

        if atr <= 0:

            return

        # ====================================================
        # STOP
        # ====================================================

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

        # ====================================================
        # MONEY RISK
        # ====================================================

        risk_money = (
            self.balance
            *
            effective_risk
        )

        if risk_money <= 0:

            return

        # ====================================================
        # RISK PER UNIT
        # ====================================================

        risk_per_unit = (
            entry_price
            -
            stop_price
        )

        if risk_per_unit <= 0:

            return

        # ====================================================
        # SIZE BY RISK
        # ====================================================

        qty_by_risk = (
            risk_money
            /
            risk_per_unit
        )

        # ====================================================
        # SIZE BY CASH
        # ====================================================

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

        # ====================================================
        # NOTIONAL
        # ====================================================

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

        total_entry_cost = (
            notional
            +
            entry_fee
        )

        if total_entry_cost > self.balance:

            return

        # ====================================================
        # REMOVE CASH
        # ====================================================

        self.balance -= (
            total_entry_cost
        )

        # ====================================================
        # STATISTICS
        # ====================================================

        if risk_level == "NORMAL":

            self.normal_risk_trades += 1

        elif risk_level == "DEFENSIVE":

            self.defensive_risk_trades += 1

        elif risk_level == "SURVIVAL":

            self.survival_risk_trades += 1

        elif risk_level == "CRITICAL":

            self.critical_risk_trades += 1

        # ====================================================
        # POSITION
        # ====================================================

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

            "risk_level":
                risk_level,

            "effective_risk":
                effective_risk,
        }

    # ========================================================
    # EXIT
    # ========================================================

    def _exit(
        self,
        timestamp,
        raw_exit_price,
        reason
    ):

        pos = self.position

        if pos is None:

            return

        # ====================================================
        # SELL SLIPPAGE
        # ====================================================

        exit_price = (
            float(raw_exit_price)
            *
            (
                1.0
                -
                self.slippage_rate
            )
        )

        # ====================================================
        # GROSS
        # ====================================================

        gross = (
            exit_price
            -
            pos["entry_price"]
        ) * pos["quantity"]

        # ====================================================
        # EXIT FEE
        # ====================================================

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

        entry_fee = float(
            pos["entry_fee"]
        )

        fees = (
            entry_fee
            +
            exit_fee
        )

        # ====================================================
        # SLIPPAGE COST
        # ====================================================

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

        # ====================================================
        # NET
        # ====================================================

        net = (
            gross
            -
            entry_fee
            -
            exit_fee
        )

        # ====================================================
        # RETURN CASH
        # ====================================================

        self.balance += (
            exit_notional
            -
            exit_fee
        )

        # ====================================================
        # INITIAL RISK
        # ====================================================

        initial_risk = (
            pos["risk_per_unit"]
            *
            pos["quantity"]
        )

        # ====================================================
        # R MULTIPLE
        # ====================================================

        if initial_risk > 0:

            r_multiple = (
                net
                /
                initial_risk
            )

        else:

            r_multiple = 0.0

        # ====================================================
        # SAVE TRADE
        # ====================================================

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

        # ====================================================
        # LOSS CONTROL
        # ====================================================

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

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        df: pd.DataFrame
    ):

        if len(df) < 300:

            return self._result()

        data = df.copy()

        # ====================================================
        # DATETIME
        # ====================================================

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
                    "DataFrame benötigt "
                    "DatetimeIndex oder timestamp."
                )

        else:

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

        data = data.sort_index()

        # ====================================================
        # REQUIRED
        # ====================================================

        required = [
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
            c
            for c in required
            if c not in data.columns
        ]

        if missing:

            raise ValueError(
                f"Fehlende V6-C Spalten: "
                f"{missing}"
            )

        # ====================================================
        # MAIN LOOP
        # ====================================================

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

            # =================================================
            # EQUITY
            # =================================================

            equity = (
                self._calculate_equity(
                    current_price
                )
            )

            # =================================================
            # DAILY RESET
            # =================================================

            self._reset_day_if_needed(
                timestamp,
                equity
            )

            # =================================================
            # PEAK
            # =================================================

            self.peak_equity = max(
                self.peak_equity,
                equity
            )

            # =================================================
            # EQUITY CURVE
            # =================================================

            self.equity_curve.append(
                {
                    "timestamp":
                        timestamp,

                    "equity":
                        equity,

                    "drawdown_pct":
                        self._current_drawdown(
                            equity
                        )
                        *
                        100,
                }
            )

            # =================================================
            # GLOBAL DD
            # =================================================

            if self._global_drawdown_hit(
                equity
            ):

                self.kill_switch = True

            # =================================================
            # MANAGE POSITION
            # =================================================

            if self.position is not None:

                pos = self.position

                # ---------------------------------------------
                # STOP FIRST
                # ---------------------------------------------

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

                # ---------------------------------------------
                # HIGHEST
                # ---------------------------------------------

                pos["highest"] = max(
                    pos["highest"],
                    float(row["high"])
                )

                # ---------------------------------------------
                # TRAILING
                # ---------------------------------------------

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

                        pos["stop"] = candidate

                # ---------------------------------------------
                # END
                # ---------------------------------------------

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

            # =================================================
            # KILL SWITCH
            # =================================================

            if self.kill_switch:

                continue

            # =================================================
            # DAILY LOSS
            # =================================================

            if self._daily_loss_limit_hit(
                equity
            ):

                continue

            # =================================================
            # COOLDOWN
            # =================================================

            if (
                i
                <
                self.cooldown_until
            ):

                continue

            # =================================================
            # SURVIVAL RISK LEVEL
            # =================================================

            risk_multiplier, risk_level = (
                self._risk_multiplier(
                    equity
                )
            )

            if risk_multiplier <= 0:

                continue

            # =================================================
            # V7-S1 SIGNAL
            #
            # V6-C + 0.10 ATR breakout filter
            # =================================================

            signal = buy_signal_v7_s1(
                previous,
                previous_previous,
                adx_min=self.adx_min
            )

            if not signal:

                continue

            # =================================================
            # ENTRY
            # =================================================

            self._enter(
                timestamp,
                previous,
                row,
                equity
            )

        # ====================================================
        # FINAL EXIT
        # ====================================================

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