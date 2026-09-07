# ============================================================
# 20to100 Trading Bot
# PAPER TRADER - V7-S0
#
# V6-C bleibt vollständig eingefroren.
#
# PAPER ONLY
# ------------------------------------------------------------
# - Keine echten Orders
# - Keine API Keys notwendig
# - Öffentliche OHLCV-Daten
# - 5m -> 1h
# - V6-C Entry
# - V7-S0 Risk Management
# - ATR Stop
# - ATR Trailing Stop
# - Daily Loss Limit
# - Consecutive Loss Protection
# - Global Drawdown Kill Switch
# - State Persistence
# - CSV Logging
# ============================================================

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest.v7_survival_engine import V7SurvivalEngine
from strategy.indicators import calculate_indicators
from strategy.strategy_v6 import buy_signal


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = ROOT_DIR / "logs"

STATE_DIR = LOG_DIR / "paper_state"
TRADE_DIR = LOG_DIR / "paper_trades"
EQUITY_DIR = LOG_DIR / "paper_equity"
SIGNAL_DIR = LOG_DIR / "paper_signals"

for directory in [
    LOG_DIR,
    STATE_DIR,
    TRADE_DIR,
    EQUITY_DIR,
    SIGNAL_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PAPER TRADER
# ============================================================

class PaperTrader:

    def __init__(
        self,
        symbol: str,
        starting_balance: float = 20.0,
        base_risk_per_trade: float = 0.01,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        atr_stop_multiplier: float = 3.0,
        trailing_atr_multiplier: float = 3.0,
        adx_min: float = 20.0,
        max_daily_loss: float = 0.05,
        max_consecutive_losses: int = 3,
        loss_cooldown_bars: int = 24,
        global_max_drawdown: float = 0.20,
        variant: str = "V6_C",
    ):

        # ====================================================
        # BASIC
        # ====================================================

        self.symbol = str(symbol)

        self.starting_balance = float(
            starting_balance
        )

        self.variant = str(
            variant
        ).upper()

        if self.variant != "V6_C":
            raise ValueError(
                "PaperTrader verwendet ausschließlich V6_C."
            )

        # ====================================================
        # V7-S0 ENGINE
        # ====================================================

        self.engine = V7SurvivalEngine(
            starting_balance=self.starting_balance,
            base_risk_per_trade=base_risk_per_trade,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            atr_stop_multiplier=atr_stop_multiplier,
            trailing_atr_multiplier=trailing_atr_multiplier,
            adx_min=adx_min,
            variant="V6_C",
            max_daily_loss=max_daily_loss,
            max_consecutive_losses=max_consecutive_losses,
            loss_cooldown_bars=loss_cooldown_bars,
            global_max_drawdown=global_max_drawdown,
        )

        # ====================================================
        # STATE
        # ====================================================

        self.last_completed_bar: Optional[pd.Timestamp] = None

        self.last_entry_hour: Optional[pd.Timestamp] = None

        self.initialized = False

        # Number of processed 1h bars.
        #
        # This is intentionally persisted because V7-S0 uses
        # the current bar index for cooldown handling.
        self.bar_counter = 0

        # Number of trades already written to CSV.
        self.logged_trade_count = 0

        # ====================================================
        # FILES
        # ====================================================

        safe_symbol = (
            self.symbol
            .replace("/", "_")
            .replace(":", "_")
        )

        self.state_file = (
            STATE_DIR
            / f"{safe_symbol}.json"
        )

        self.trade_file = (
            TRADE_DIR
            / f"{safe_symbol}.csv"
        )

        self.equity_file = (
            EQUITY_DIR
            / f"{safe_symbol}.csv"
        )

        self.signal_file = (
            SIGNAL_DIR
            / f"{safe_symbol}.csv"
        )

    # ========================================================
    # STATE SAVE
    # ========================================================

    def save_state(self):

        position = self.engine.position

        state = {
            "symbol": self.symbol,

            "variant": self.variant,

            "initialized": self.initialized,

            "last_completed_bar": (
                self._timestamp_to_string(
                    self.last_completed_bar
                )
                if self.last_completed_bar is not None
                else None
            ),

            "last_entry_hour": (
                self._timestamp_to_string(
                    self.last_entry_hour
                )
                if self.last_entry_hour is not None
                else None
            ),

            "bar_counter": self.bar_counter,

            "logged_trade_count":
                self.logged_trade_count,

            "balance":
                float(self.engine.balance),

            "peak_equity":
                float(self.engine.peak_equity),

            "day_start_equity":
                float(self.engine.day_start_equity),

            "current_day":
                (
                    self.engine.current_day.isoformat()
                    if self.engine.current_day is not None
                    else None
                ),

            "consecutive_losses":
                int(self.engine.consecutive_losses),

            "cooldown_until":
                int(self.engine.cooldown_until),

            "kill_switch":
                bool(self.engine.kill_switch),

            "normal_risk_trades":
                int(self.engine.normal_risk_trades),

            "defensive_risk_trades":
                int(self.engine.defensive_risk_trades),

            "survival_risk_trades":
                int(self.engine.survival_risk_trades),

            "critical_risk_trades":
                int(self.engine.critical_risk_trades),

            "position":
                position,
        }

        tmp_file = self.state_file.with_suffix(
            ".tmp"
        )

        with open(
            tmp_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                state,
                f,
                indent=2,
                default=str,
            )

        os.replace(
            tmp_file,
            self.state_file,
        )

    # ========================================================
    # STATE LOAD
    # ========================================================

    def load_state(self) -> bool:

        if not self.state_file.exists():
            return False

        try:

            with open(
                self.state_file,
                "r",
                encoding="utf-8",
            ) as f:

                state = json.load(f)

            # ------------------------------------------------
            # BASIC
            # ------------------------------------------------

            self.initialized = bool(
                state.get(
                    "initialized",
                    False,
                )
            )

            self.last_completed_bar = (
                self._string_to_timestamp(
                    state.get(
                        "last_completed_bar"
                    )
                )
            )

            self.last_entry_hour = (
                self._string_to_timestamp(
                    state.get(
                        "last_entry_hour"
                    )
                )
            )

            self.bar_counter = int(
                state.get(
                    "bar_counter",
                    0,
                )
            )

            self.logged_trade_count = int(
                state.get(
                    "logged_trade_count",
                    0,
                )
            )

            # ------------------------------------------------
            # ENGINE ACCOUNT
            # ------------------------------------------------

            self.engine.balance = float(
                state.get(
                    "balance",
                    self.starting_balance,
                )
            )

            self.engine.peak_equity = float(
                state.get(
                    "peak_equity",
                    self.starting_balance,
                )
            )

            self.engine.day_start_equity = float(
                state.get(
                    "day_start_equity",
                    self.starting_balance,
                )
            )

            # ------------------------------------------------
            # DAY
            # ------------------------------------------------

            current_day = state.get(
                "current_day"
            )

            if current_day:
                self.engine.current_day = (
                    pd.Timestamp(
                        current_day
                    ).date()
                )
            else:
                self.engine.current_day = None

            # ------------------------------------------------
            # LOSS CONTROL
            # ------------------------------------------------

            self.engine.consecutive_losses = int(
                state.get(
                    "consecutive_losses",
                    0,
                )
            )

            self.engine.cooldown_until = int(
                state.get(
                    "cooldown_until",
                    -1,
                )
            )

            self.engine.kill_switch = bool(
                state.get(
                    "kill_switch",
                    False,
                )
            )

            # ------------------------------------------------
            # RISK STATISTICS
            # ------------------------------------------------

            self.engine.normal_risk_trades = int(
                state.get(
                    "normal_risk_trades",
                    0,
                )
            )

            self.engine.defensive_risk_trades = int(
                state.get(
                    "defensive_risk_trades",
                    0,
                )
            )

            self.engine.survival_risk_trades = int(
                state.get(
                    "survival_risk_trades",
                    0,
                )
            )

            self.engine.critical_risk_trades = int(
                state.get(
                    "critical_risk_trades",
                    0,
                )
            )

            # ------------------------------------------------
            # POSITION
            # ------------------------------------------------

            self.engine.position = (
                state.get(
                    "position"
                )
            )

            # ------------------------------------------------
            # ENGINE INDEX
            # ------------------------------------------------

            self.engine._current_index = (
                self.bar_counter
            )

            print(
                f"[{self.symbol}] "
                f"Paper-State geladen."
            )

            print(
                f"[{self.symbol}] "
                f"Balance: "
                f"${self.engine.balance:.4f}"
            )

            print(
                f"[{self.symbol}] "
                f"Bar Counter: "
                f"{self.bar_counter}"
            )

            if self.engine.position is not None:

                print(
                    f"[{self.symbol}] "
                    f"OFFENE POSITION geladen."
                )

            if self.engine.kill_switch:

                print(
                    f"[{self.symbol}] "
                    f"⚠️ KILL SWITCH IST AKTIV."
                )

            return True

        except Exception as exc:

            print(
                f"[{self.symbol}] "
                f"FEHLER beim Laden des State: "
                f"{exc}"
            )

            return False

    # ========================================================
    # INITIALIZE FROM CURRENT MARKET DATA
    # ========================================================

    def initialize(
        self,
        data: pd.DataFrame,
    ):

        data = self._prepare_dataframe(
            data
        )

        if len(data) < 300:

            raise ValueError(
                f"[{self.symbol}] "
                f"Zu wenig Daten zur Initialisierung: "
                f"{len(data)}"
            )

        # ----------------------------------------------------
        # Current 1h candle
        # ----------------------------------------------------

        current_hour = data.index[-1]

        completed = data.iloc[:-1]

        if len(completed) < 1:

            raise ValueError(
                f"[{self.symbol}] "
                "Keine abgeschlossene 1h-Kerze vorhanden."
            )

        last_completed = completed.index[-1]

        # ----------------------------------------------------
        # IMPORTANT
        #
        # We do NOT replay old trades.
        #
        # The paper trader starts from NOW.
        # ----------------------------------------------------

        self.last_completed_bar = (
            last_completed
        )

        self.last_entry_hour = (
            current_hour
        )

        self.bar_counter = (
            len(completed)
        )

        self.engine._current_index = (
            self.bar_counter
        )

        # ----------------------------------------------------
        # Establish initial day/equity state.
        # ----------------------------------------------------

        current_price = float(
            current_hour
            and data.iloc[-1]["close"]
        )

        equity = self.engine._calculate_equity(
            current_price
        )

        self.engine._reset_day_if_needed(
            last_completed,
            equity,
        )

        self.engine.peak_equity = max(
            self.engine.peak_equity,
            equity,
        )

        self.initialized = True

        self.save_state()

        print(
            f"[{self.symbol}] "
            f"Paper-Trader initialisiert."
        )

        print(
            f"[{self.symbol}] "
            f"Letzte abgeschlossene 1h: "
            f"{last_completed}"
        )

        print(
            f"[{self.symbol}] "
            f"Aktuelle 1h: "
            f"{current_hour}"
        )

        print(
            f"[{self.symbol}] "
            f"Start-Balance: "
            f"${self.engine.balance:.4f}"
        )

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(
        self,
        data: pd.DataFrame,
    ):

        data = self._prepare_dataframe(
            data
        )

        if len(data) < 300:

            print(
                f"[{self.symbol}] "
                f"Zu wenig 1h-Daten: "
                f"{len(data)}"
            )

            return

        # ====================================================
        # INITIAL SETUP
        # ====================================================

        if not self.initialized:

            self.initialize(
                data
            )

            return

        current_hour = data.index[-1]

        completed = data.iloc[:-1]

        if len(completed) < 3:

            return

        # ====================================================
        # PROCESS NEW COMPLETED BARS
        # ====================================================

        if self.last_completed_bar is None:

            self.last_completed_bar = (
                completed.index[-1]
            )

        new_completed = completed[
            completed.index
            >
            self.last_completed_bar
        ]

        for timestamp, row in new_completed.iterrows():

            self._process_completed_bar(
                timestamp,
                row,
            )

            self.last_completed_bar = timestamp

        # ====================================================
        # CURRENT HOUR ENTRY
        #
        # This corresponds to the next-candle execution
        # of the original S0 engine.
        # ====================================================

        if (
            self.last_entry_hour is None
            or
            current_hour
            >
            self.last_entry_hour
        ):

            self._process_current_hour_entry(
                data
            )

            self.last_entry_hour = (
                current_hour
            )

        # ====================================================
        # SAVE
        # ====================================================

        self.save_state()

    # ========================================================
    # COMPLETED BAR
    # ========================================================

    def _process_completed_bar(
        self,
        timestamp,
        row,
    ):

        self.bar_counter += 1

        self.engine._current_index = (
            self.bar_counter
        )

        current_price = float(
            row["close"]
        )

        # ====================================================
        # EQUITY
        # ====================================================

        equity = (
            self.engine._calculate_equity(
                current_price
            )
        )

        # ====================================================
        # DAILY RESET
        # ====================================================

        self.engine._reset_day_if_needed(
            timestamp,
            equity,
        )

        # ====================================================
        # PEAK
        # ====================================================

        self.engine.peak_equity = max(
            self.engine.peak_equity,
            equity,
        )

        # ====================================================
        # EQUITY LOG
        # ====================================================

        self._log_equity(
            timestamp,
            equity,
        )

        # ====================================================
        # GLOBAL DD
        # ====================================================

        if self.engine._global_drawdown_hit(
            equity
        ):

            self.engine.kill_switch = True

            print(
                f"[{self.symbol}] "
                f"🛑 GLOBAL KILL SWITCH "
                f"bei {timestamp}"
            )

        # ====================================================
        # POSITION MANAGEMENT
        # ====================================================

        if self.engine.position is not None:

            position_before = (
                self.engine.position
            )

            stop = float(
                position_before["stop"]
            )

            # ------------------------------------------------
            # STOP FIRST
            # ------------------------------------------------

            if float(row["low"]) <= stop:

                self.engine._exit(
                    timestamp,
                    stop,
                    "ATR_STOP",
                )

                self._log_new_trades()

                print(
                    f"[{self.symbol}] "
                    f"🔴 EXIT ATR_STOP "
                    f"@ {stop:.8f}"
                )

                return

            # ------------------------------------------------
            # HIGHEST
            # ------------------------------------------------

            position = (
                self.engine.position
            )

            position["highest"] = max(
                position["highest"],
                float(row["high"]),
            )

            # ------------------------------------------------
            # TRAILING
            # ------------------------------------------------

            current_atr = float(
                row["atr_14"]
            )

            if current_atr > 0:

                candidate = (
                    position["highest"]
                    -
                    current_atr
                    *
                    self.engine.trailing_atr_multiplier
                )

                if candidate > position["stop"]:

                    position["stop"] = (
                        candidate
                    )

        # ====================================================
        # NO FORCED END EXIT
        #
        # Unlike the historical backtest, paper trading
        # never closes a position merely because the loop
        # reached the end of a dataset.
        # ====================================================

    # ========================================================
    # CURRENT HOUR ENTRY
    # ========================================================

    def _process_current_hour_entry(
        self,
        data: pd.DataFrame,
    ):

        current = data.iloc[-1]

        previous = data.iloc[-2]

        previous_previous = data.iloc[-3]

        timestamp = data.index[-1]

        current_price = float(
            current["open"]
        )

        # ====================================================
        # CURRENT INDEX
        # ====================================================

        self.bar_counter += 1

        self.engine._current_index = (
            self.bar_counter
        )

        # ====================================================
        # EQUITY
        #
        # Entry uses the current 1h opening price.
        # ====================================================

        equity = (
            self.engine._calculate_equity(
                current_price
            )
        )

        # ====================================================
        # DAILY RESET
        # ====================================================

        self.engine._reset_day_if_needed(
            timestamp,
            equity,
        )

        # ====================================================
        # PEAK
        # ====================================================

        self.engine.peak_equity = max(
            self.engine.peak_equity,
            equity,
        )

        # ====================================================
        # EQUITY LOG
        # ====================================================

        self._log_equity(
            timestamp,
            equity,
        )

        # ====================================================
        # ALREADY IN POSITION?
        # ====================================================

        if self.engine.position is not None:

            return

        # ====================================================
        # KILL SWITCH
        # ====================================================

        if self.engine.kill_switch:

            return

        # ====================================================
        # DAILY LOSS LIMIT
        # ====================================================

        if self.engine._daily_loss_limit_hit(
            equity
        ):

            print(
                f"[{self.symbol}] "
                f"⚠️ Daily Loss Limit aktiv."
            )

            return

        # ====================================================
        # COOLDOWN
        #
        # Important:
        # V7-S0 uses bar indices.
        # ====================================================

        if (
            self.engine._current_index
            <
            self.engine.cooldown_until
        ):

            return

        # ====================================================
        # SURVIVAL RISK
        # ====================================================

        risk_multiplier, risk_level = (
            self.engine._risk_multiplier(
                equity
            )
        )

        if risk_multiplier <= 0:

            return

        # ====================================================
        # V6-C SIGNAL
        #
        # EXACTLY the same signal call as S0.
        # ====================================================

        signal = buy_signal(
            previous,
            previous_previous,
            variant="V6_C",
            adx_min=self.engine.adx_min,
        )

        # ====================================================
        # SIGNAL LOG
        # ====================================================

        self._log_signal(
            timestamp=timestamp,
            signal=signal,
            equity=equity,
            risk_level=risk_level,
            risk_multiplier=risk_multiplier,
            previous=previous,
        )

        if not signal:

            return

        # ====================================================
        # ENTRY
        #
        # Previous candle = signal candle
        # Current candle = execution candle
        # Entry price = current 1h open + slippage
        # ====================================================

        balance_before = (
            self.engine.balance
        )

        self.engine._enter(
            timestamp,
            previous,
            current,
            equity,
        )

        # ====================================================
        # CHECK WHETHER ENTRY ACTUALLY HAPPENED
        # ====================================================

        if self.engine.position is not None:

            position = (
                self.engine.position
            )

            print(
                ""
            )

            print(
                "================================================"
            )

            print(
                f"[{self.symbol}] 🟢 PAPER BUY"
            )

            print(
                f"Time:       {timestamp}"
            )

            print(
                f"Entry:      "
                f"{position['entry_price']:.8f}"
            )

            print(
                f"Stop:       "
                f"{position['stop']:.8f}"
            )

            print(
                f"Quantity:   "
                f"{position['quantity']:.8f}"
            )

            print(
                f"Risk:       "
                f"{position['effective_risk'] * 100:.2f}%"
            )

            print(
                f"Risk Level: "
                f"{position['risk_level']}"
            )

            print(
                f"Balance:    "
                f"${balance_before:.4f}"
                f" -> "
                f"${self.engine.balance:.4f}"
            )

            print(
                "================================================"
            )

            print(
                ""
            )

    # ========================================================
    # DATA PREPARATION
    # ========================================================

    def _prepare_dataframe(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        df = data.copy()

        # ====================================================
        # DATETIME INDEX
        # ====================================================

        if not isinstance(
            df.index,
            pd.DatetimeIndex,
        ):

            if "timestamp" in df.columns:

                df["timestamp"] = (
                    pd.to_datetime(
                        df["timestamp"],
                        utc=True,
                    )
                )

                df = df.set_index(
                    "timestamp"
                )

            else:

                raise ValueError(
                    f"[{self.symbol}] "
                    "DataFrame benötigt "
                    "DatetimeIndex oder timestamp."
                )

        else:

            if df.index.tz is None:

                df.index = (
                    df.index.tz_localize(
                        "UTC"
                    )
                )

            else:

                df.index = (
                    df.index.tz_convert(
                        "UTC"
                    )
                )

        df = df.sort_index()

        # ====================================================
        # NUMERIC
        # ====================================================

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        # ====================================================
        # INDICATORS
        #
        # This uses the same indicator module as the
        # historical V6/V7 backtests.
        # ====================================================

        df = calculate_indicators(
            df
        )

        df = df.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

        return df

    # ========================================================
    # LOG EQUITY
    # ========================================================

    def _log_equity(
        self,
        timestamp,
        equity: float,
    ):

        file_exists = (
            self.equity_file.exists()
        )

        drawdown = (
            self.engine._current_drawdown(
                equity
            )
            * 100.0
        )

        row = {
            "timestamp":
                str(timestamp),

            "symbol":
                self.symbol,

            "equity":
                float(equity),

            "balance":
                float(self.engine.balance),

            "drawdown_pct":
                float(drawdown),

            "kill_switch":
                bool(self.engine.kill_switch),

            "position":
                bool(
                    self.engine.position
                    is not None
                ),
        }

        with open(
            self.equity_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    row.keys()
                ),
            )

            if not file_exists:

                writer.writeheader()

            writer.writerow(
                row
            )

    # ========================================================
    # LOG SIGNAL
    # ========================================================

    def _log_signal(
        self,
        timestamp,
        signal,
        equity,
        risk_level,
        risk_multiplier,
        previous,
    ):

        file_exists = (
            self.signal_file.exists()
        )

        row = {
            "timestamp":
                str(timestamp),

            "symbol":
                self.symbol,

            "signal":
                bool(signal),

            "equity":
                float(equity),

            "risk_level":
                risk_level,

            "risk_multiplier":
                float(risk_multiplier),

            "close":
                float(previous["close"]),

            "atr_14":
                float(previous["atr_14"])
                if pd.notna(
                    previous["atr_14"]
                )
                else None,

            "adx_14":
                float(previous["adx_14"])
                if pd.notna(
                    previous["adx_14"]
                )
                else None,

            "ema_100":
                float(previous["ema_100"])
                if pd.notna(
                    previous["ema_100"]
                )
                else None,

            "ema_200":
                float(previous["ema_200"])
                if pd.notna(
                    previous["ema_200"]
                )
                else None,
        }

        with open(
            self.signal_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    row.keys()
                ),
            )

            if not file_exists:

                writer.writeheader()

            writer.writerow(
                row
            )

    # ========================================================
    # LOG NEW TRADES
    # ========================================================

    def _log_new_trades(self):

        while (
            self.logged_trade_count
            <
            len(self.engine.trades)
        ):

            trade = (
                self.engine.trades[
                    self.logged_trade_count
                ]
            )

            row = asdict(
                trade
            )

            row[
                "symbol"
            ] = self.symbol

            row[
                "variant"
            ] = self.variant

            file_exists = (
                self.trade_file.exists()
            )

            with open(
                self.trade_file,
                "a",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        row.keys()
                    ),
                )

                if not file_exists:

                    writer.writeheader()

                writer.writerow(
                    row
                )

            self.logged_trade_count += 1

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _timestamp_to_string(
        timestamp,
    ):

        if timestamp is None:

            return None

        return pd.Timestamp(
            timestamp
        ).isoformat()

    # ========================================================

    @staticmethod
    def _string_to_timestamp(
        value,
    ):

        if not value:

            return None

        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is None:

            timestamp = timestamp.tz_localize(
                "UTC"
            )

        else:

            timestamp = timestamp.tz_convert(
                "UTC"
            )

        return timestamp

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict:

        position = (
            self.engine.position
        )

        return {
            "symbol":
                self.symbol,

            "variant":
                self.variant,

            "balance":
                float(self.engine.balance),

            "peak_equity":
                float(self.engine.peak_equity),

            "consecutive_losses":
                int(
                    self.engine.consecutive_losses
                ),

            "cooldown_until":
                int(
                    self.engine.cooldown_until
                ),

            "kill_switch":
                bool(
                    self.engine.kill_switch
                ),

            "position":
                position,

            "trades":
                len(
                    self.engine.trades
                ),

            "bar_counter":
                self.bar_counter,

            "last_completed_bar":
                (
                    str(
                        self.last_completed_bar
                    )
                    if self.last_completed_bar
                    else None
                ),
        }
