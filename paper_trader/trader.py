# ============================================================
# 20to100 Trading Bot
# V7-S0 PAPER TRADER
#
# IMPORTANT:
# - PAPER TRADING ONLY
# - NO REAL ORDERS
# - Uses the existing V7 Survival Engine
# - Uses V6-C signal
# - 5m market data -> 1h signal timeframe
# ============================================================

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd

from backtest.v7_survival_engine import V7SurvivalEngine
from strategy.strategy_v6 import buy_signal
from strategy.indicators import calculate_indicators

from config_paper import (
    STARTING_CAPITAL,
    FEE_RATE,
    SLIPPAGE_RATE,
    ATR_STOP_MULTIPLIER,
    TRAILING_ATR_MULTIPLIER,
    ADX_MIN,
    MAX_DAILY_LOSS,
    MAX_CONSECUTIVE_LOSSES,
    LOSS_COOLDOWN_BARS,
    GLOBAL_MAX_DRAWDOWN,
    BASE_RISK_PER_TRADE,
    VARIANT,
    LOG_DIR,
    STATE_DIR,
)


# ============================================================
# PAPER TRADER
# ============================================================

class PaperTrader:
    """
    Incremental paper-trading adapter around V7-S0.

    NO REAL ORDERS ARE SENT.
    """

    def __init__(self, symbol: str):

        self.symbol = symbol

        # ----------------------------------------------------
        # DIRECTORIES
        # ----------------------------------------------------

        self.log_dir = Path(LOG_DIR)
        self.state_dir = Path(STATE_DIR)

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # V7-S0 ENGINE
        # ----------------------------------------------------

        self.engine = V7SurvivalEngine(
            starting_balance=STARTING_CAPITAL,
            base_risk_per_trade=BASE_RISK_PER_TRADE,
            fee_rate=FEE_RATE,
            slippage_rate=SLIPPAGE_RATE,
            atr_stop_multiplier=ATR_STOP_MULTIPLIER,
            trailing_atr_multiplier=TRAILING_ATR_MULTIPLIER,
            adx_min=ADX_MIN,
            variant=VARIANT,
            max_daily_loss=MAX_DAILY_LOSS,
            max_consecutive_losses=MAX_CONSECUTIVE_LOSSES,
            loss_cooldown_bars=LOSS_COOLDOWN_BARS,
            global_max_drawdown=GLOBAL_MAX_DRAWDOWN,
        )

        # ----------------------------------------------------
        # STATE FILE
        # ----------------------------------------------------

        self.state_file = (
            self.state_dir
            / f"{self.symbol.replace('/', '_')}_paper_state.json"
        )

        # ----------------------------------------------------
        # PAPER STATE
        # ----------------------------------------------------

        self.last_completed_bar = None
        self.last_entry_hour = None

        self.bar_counter = -1

        self.balance = float(
            STARTING_CAPITAL
        )

        self.peak_equity = float(
            STARTING_CAPITAL
        )

        self.day_start_equity = float(
            STARTING_CAPITAL
        )

        self.current_day = None

        self.consecutive_losses = 0

        self.cooldown_until = -1

        self.kill_switch = False

        self.position = None

        self.logged_trades = 0

        # ----------------------------------------------------
        # LOAD STATE
        # ----------------------------------------------------

        self.load_state()

        # ----------------------------------------------------
        # RESTORE ENGINE STATE
        # ----------------------------------------------------

        self.engine.balance = self.balance
        self.engine.peak_equity = self.peak_equity
        self.engine.day_start_equity = self.day_start_equity
        self.engine.current_day = self.current_day
        self.engine.consecutive_losses = self.consecutive_losses
        self.engine.cooldown_until = self.cooldown_until
        self.engine.kill_switch = self.kill_switch
        self.engine.position = self.position
        self.engine._current_index = self.bar_counter


    # ========================================================
    # STATE LOAD
    # ========================================================

    def load_state(self):

        if not self.state_file.exists():

            print(
                f"[{self.symbol}] "
                f"No previous paper state found."
            )

            return

        try:

            with open(
                self.state_file,
                "r",
                encoding="utf-8",
            ) as f:

                state = json.load(f)

            self.last_completed_bar = (
                self._parse_timestamp(
                    state.get("last_completed_bar")
                )
            )

            self.last_entry_hour = (
                self._parse_timestamp(
                    state.get("last_entry_hour")
                )
            )

            self.bar_counter = int(
                state.get(
                    "bar_counter",
                    -1,
                )
            )

            self.balance = float(
                state.get(
                    "balance",
                    STARTING_CAPITAL,
                )
            )

            self.peak_equity = float(
                state.get(
                    "peak_equity",
                    self.balance,
                )
            )

            self.day_start_equity = float(
                state.get(
                    "day_start_equity",
                    self.balance,
                )
            )

            current_day = state.get(
                "current_day"
            )

            if current_day:

                self.current_day = (
                    date.fromisoformat(
                        current_day
                    )
                )

            self.consecutive_losses = int(
                state.get(
                    "consecutive_losses",
                    0,
                )
            )

            self.cooldown_until = int(
                state.get(
                    "cooldown_until",
                    -1,
                )
            )

            self.kill_switch = bool(
                state.get(
                    "kill_switch",
                    False,
                )
            )

            self.position = state.get(
                "position"
            )

            self.logged_trades = int(
                state.get(
                    "logged_trades",
                    0,
                )
            )

            print(
                f"[{self.symbol}] "
                f"Paper state restored | "
                f"balance=${self.balance:.4f} | "
                f"last_bar={self.last_completed_bar}"
            )

        except Exception as exc:

            print(
                f"[{self.symbol}] "
                f"WARNING: Could not load state: {exc}"
            )


    # ========================================================
    # STATE SAVE
    # ========================================================

    def save_state(self):

        state = {
            "symbol": self.symbol,

            "last_completed_bar":
                self._serialize_timestamp(
                    self.last_completed_bar
                ),

            "last_entry_hour":
                self._serialize_timestamp(
                    self.last_entry_hour
                ),

            "bar_counter":
                self.bar_counter,

            "balance":
                self.balance,

            "peak_equity":
                self.peak_equity,

            "day_start_equity":
                self.day_start_equity,

            "current_day": (
                self.current_day.isoformat()
                if isinstance(
                    self.current_day,
                    date,
                )
                else None
            ),

            "consecutive_losses":
                self.consecutive_losses,

            "cooldown_until":
                self.cooldown_until,

            "kill_switch":
                self.kill_switch,

            "position":
                self.position,

            "logged_trades":
                self.logged_trades,

            "updated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        temp_file = (
            self.state_file.with_suffix(
                ".tmp"
            )
        )

        with open(
            temp_file,
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
            temp_file,
            self.state_file,
        )


    # ========================================================
    # TIMESTAMP SERIALIZATION
    # ========================================================

    @staticmethod
    def _serialize_timestamp(value):

        if value is None:

            return None

        if isinstance(
            value,
            pd.Timestamp,
        ):

            return value.isoformat()

        if isinstance(
            value,
            datetime,
        ):

            return value.isoformat()

        return str(value)


    @staticmethod
    def _parse_timestamp(value):

        if not value:

            return None

        try:

            return pd.Timestamp(
                value
            )

        except Exception:

            return None


    # ========================================================
    # CSV
    # ========================================================

    def _append_csv(
        self,
        filename,
        row,
    ):

        path = (
            self.log_dir
            / filename
        )

        pd.DataFrame(
            [row]
        ).to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False,
        )


    # ========================================================
    # SIGNAL LOG
    # ========================================================

    def log_signal(
        self,
        timestamp,
        signal,
        row,
        reason="",
    ):

        self._append_csv(
            "paper_signals.csv",
            {
                "timestamp":
                    timestamp,

                "symbol":
                    self.symbol,

                "variant":
                    VARIANT,

                "signal":
                    bool(signal),

                "close":
                    row.get("close"),

                "ema_100":
                    row.get("ema_100"),

                "ema_200":
                    row.get("ema_200"),

                "ema_200_slope":
                    row.get("ema_200_slope"),

                "adx_14":
                    row.get("adx_14"),

                "atr_14":
                    row.get("atr_14"),

                "atr_14_ma50":
                    row.get("atr_14_ma50"),

                "donchian_high_20":
                    row.get("donchian_high_20"),

                "reason":
                    reason,
            },
        )


    # ========================================================
    # EQUITY LOG
    # ========================================================

    def log_equity(
        self,
        timestamp,
        equity,
    ):

        drawdown = 0.0

        if self.peak_equity > 0:

            drawdown = (
                (
                    equity
                    - self.peak_equity
                )
                / self.peak_equity
                * 100.0
            )

        self._append_csv(
            "paper_equity.csv",
            {
                "timestamp":
                    timestamp,

                "symbol":
                    self.symbol,

                "balance":
                    self.balance,

                "equity":
                    equity,

                "peak_equity":
                    self.peak_equity,

                "drawdown_pct":
                    drawdown,

                "position_open":
                    self.position is not None,

                "kill_switch":
                    self.kill_switch,
            },
        )


    # ========================================================
    # TRADE LOGGING
    # ========================================================

    def _log_new_trades(self):

        trades = getattr(
            self.engine,
            "trades",
            [],
        )

        while (
            self.logged_trades
            < len(trades)
        ):

            trade = trades[
                self.logged_trades
            ]

            if is_dataclass(
                trade
            ):

                row = asdict(
                    trade
                )

            elif isinstance(
                trade,
                dict,
            ):

                row = dict(
                    trade
                )

            else:

                row = {
                    "trade":
                        str(trade)
                }

            row["symbol"] = (
                self.symbol
            )

            row["variant"] = (
                VARIANT
            )

            self._append_csv(
                "paper_trades.csv",
                row,
            )

            net_profit = float(
                row.get(
                    "net_profit",
                    0.0,
                )
            )

            if net_profit > 0:

                print(
                    f"[{self.symbol}] "
                    f"✅ PAPER SELL | "
                    f"Net=${net_profit:.4f}"
                )

            else:

                print(
                    f"[{self.symbol}] "
                    f"🔴 PAPER SELL | "
                    f"Net=${net_profit:.4f}"
                )

            self.logged_trades += 1


    # ========================================================
    # EQUITY
    # ========================================================

    def _calculate_equity(
        self,
        row,
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        # Engine expects a market price, not a Series.
        # ----------------------------------------------------

        if isinstance(
            row,
            pd.Series,
        ):

            market_price = float(
                row["close"]
            )

        elif isinstance(
            row,
            dict,
        ):

            market_price = float(
                row["close"]
            )

        else:

            market_price = float(
                row
            )

        return self.engine._calculate_equity(
            market_price
        )


    # ========================================================
    # COMPLETED BAR
    # ========================================================

    def _process_completed_bar(
        self,
        timestamp,
        row,
    ):

        # ----------------------------------------------------
        # Advance engine index
        # ----------------------------------------------------

        self.engine._current_index += 1

        self.bar_counter = (
            self.engine._current_index
        )

        # ----------------------------------------------------
        # Equity
        # ----------------------------------------------------

        equity = self._calculate_equity(
            row
        )

        # ----------------------------------------------------
        # Daily reset
        # ----------------------------------------------------

        self.engine._reset_day_if_needed(
            timestamp,
            equity,
        )

        # ----------------------------------------------------
        # Peak
        # ----------------------------------------------------

        if (
            equity
            > self.engine.peak_equity
        ):

            self.engine.peak_equity = (
                equity
            )

        # ----------------------------------------------------
        # Global drawdown
        # ----------------------------------------------------

        if self.engine._global_drawdown_hit(
            equity
        ):

            if not self.engine.kill_switch:

                print(
                    f"[{self.symbol}] "
                    f"!!! GLOBAL DRAWDOWN "
                    f"KILL SWITCH !!!"
                )

            self.engine.kill_switch = True

        # ----------------------------------------------------
        # POSITION MANAGEMENT
        #
        # Same order as V7-S0:
        #
        # 1. Stop
        # 2. Highest
        # 3. Trailing stop
        # ----------------------------------------------------

        if self.engine.position is not None:

            position = (
                self.engine.position
            )

            # ------------------------------------------------
            # STOP FIRST
            # ------------------------------------------------

            stop_price = float(
                position["stop"]
            )

            if (
                float(row["low"])
                <= stop_price
            ):

                self.engine._exit(
                    timestamp,
                    stop_price,
                    "ATR_STOP",
                )

                self._log_new_trades()

            else:

                # ------------------------------------------------
                # HIGHEST PRICE
                # ------------------------------------------------

                high = float(
                    row["high"]
                )

                if (
                    high
                    > float(
                        position["highest"]
                    )
                ):

                    position["highest"] = (
                        high
                    )

                # ------------------------------------------------
                # TRAILING STOP
                # ------------------------------------------------

                atr = float(
                    row["atr_14"]
                )

                if atr > 0:

                    candidate = (
                        position["highest"]
                        -
                        (
                            atr
                            *
                            self.engine
                                .trailing_atr_multiplier
                        )
                    )

                    if (
                        candidate
                        > float(
                            position["stop"]
                        )
                    ):

                        position["stop"] = (
                            candidate
                        )

                self.engine.position = (
                    position
                )

        # ----------------------------------------------------
        # SYNC STATE
        # ----------------------------------------------------

        self.position = (
            self.engine.position
        )

        self.balance = float(
            self.engine.balance
        )

        self.peak_equity = float(
            self.engine.peak_equity
        )

        self.day_start_equity = float(
            self.engine.day_start_equity
        )

        self.current_day = (
            self.engine.current_day
        )

        self.consecutive_losses = int(
            self.engine.consecutive_losses
        )

        self.cooldown_until = int(
            self.engine.cooldown_until
        )

        self.kill_switch = bool(
            self.engine.kill_switch
        )

        # ----------------------------------------------------
        # EQUITY LOG
        # ----------------------------------------------------

        self.log_equity(
            timestamp,
            equity,
        )


    # ========================================================
    # CURRENT HOUR / ENTRY
    # ========================================================

    def _process_current_hour(
        self,
        timestamp,
        current_row,
        previous_row,
        previous_previous_row,
    ):

        # ----------------------------------------------------
        # Only one entry decision per hour
        # ----------------------------------------------------

        if (
            self.last_entry_hour is not None
            and timestamp
            <= self.last_entry_hour
        ):

            return

        self.last_entry_hour = (
            timestamp
        )

        # ----------------------------------------------------
        # Existing position
        # ----------------------------------------------------

        if self.engine.position is not None:

            self.log_signal(
                timestamp,
                False,
                previous_row,
                "position_already_open",
            )

            return

        # ----------------------------------------------------
        # Current equity
        # ----------------------------------------------------

        equity = self._calculate_equity(
            current_row
        )

        # ----------------------------------------------------
        # Kill switch
        # ----------------------------------------------------

        if self.engine.kill_switch:

            self.log_signal(
                timestamp,
                False,
                previous_row,
                "global_kill_switch",
            )

            return

        # ----------------------------------------------------
        # Daily loss
        # ----------------------------------------------------

        if self.engine._daily_loss_limit_hit(
            equity
        ):

            self.log_signal(
                timestamp,
                False,
                previous_row,
                "daily_loss_limit",
            )

            return

        # ----------------------------------------------------
        # Consecutive losses
        # ----------------------------------------------------

        if (
            self.engine.consecutive_losses
            >= self.engine.max_consecutive_losses
        ):

            self.log_signal(
                timestamp,
                False,
                previous_row,
                "consecutive_loss_limit",
            )

            return

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if (
            self.engine._current_index
            < self.engine.cooldown_until
        ):

            self.log_signal(
                timestamp,
                False,
                previous_row,
                "loss_cooldown",
            )

            return

        # ----------------------------------------------------
        # V6-C SIGNAL
        # ----------------------------------------------------

        try:

            signal = bool(
                buy_signal(
                    previous_row,
                    previous_previous_row,
                    variant="V6_C",
                    adx_min=self.engine.adx_min,
                )
            )

        except Exception as exc:

            print(
                f"[{self.symbol}] "
                f"Signal error: {exc}"
            )

            signal = False

        self.log_signal(
            timestamp,
            signal,
            previous_row,
            "V6_C",
        )

        if not signal:

            return

        # ----------------------------------------------------
        # Risk multiplier
        # ----------------------------------------------------

        risk_multiplier, risk_level = (
            self.engine._risk_multiplier(
                equity
            )
        )

        if risk_multiplier <= 0:

            self.log_signal(
                timestamp,
                False,
                previous_row,
                "risk_multiplier_zero",
            )

            return

        # ----------------------------------------------------
        # ENTER
        #
        # Signal = completed candle
        # Entry = current candle open
        # ----------------------------------------------------

        was_flat = (
            self.engine.position
            is None
        )

        self.engine._enter(
            timestamp,
            previous_row,
            current_row,
            equity,
        )

        # ----------------------------------------------------
        # Detect actual entry
        # ----------------------------------------------------

        if (
            was_flat
            and self.engine.position
            is not None
        ):

            self.position = (
                self.engine.position
            )

            self.balance = float(
                self.engine.balance
            )

            print(
                f"[{self.symbol}] "
                f"🟢 PAPER BUY | "
                f"{timestamp} | "
                f"price="
                f"{self.position.get('entry_price', 0):.4f} | "
                f"qty="
                f"{self.position.get('quantity', 0):.8f}"
            )


    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(
        self,
        hourly_df: pd.DataFrame,
    ):

        if (
            hourly_df is None
            or hourly_df.empty
        ):

            return

        df = hourly_df.copy()

        # ====================================================
        # FIX TIMESTAMP INDEX
        # ====================================================

        # Our resampler returns:
        #
        # timestamp | open | high | ...
        #
        # Therefore the timestamp column MUST become
        # the DatetimeIndex.
        # ====================================================

        if "timestamp" in df.columns:

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True,
                errors="coerce",
            )

            df = df.dropna(
                subset=["timestamp"]
            )

            df = df.set_index(
                "timestamp"
            )

        elif not isinstance(
            df.index,
            pd.DatetimeIndex,
        ):

            raise ValueError(
                "Hourly DataFrame benötigt "
                "eine timestamp-Spalte oder "
                "einen DatetimeIndex."
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

        # ====================================================
        # SORT
        # ====================================================

        df = (
            df
            .sort_index()
            .loc[
                ~df.index.duplicated(
                    keep="last"
                )
            ]
        )

        # ====================================================
        # INDICATORS
        # ====================================================

        try:

            df = calculate_indicators(
                df
            )

        except Exception as exc:

            print(
                f"[{self.symbol}] "
                f"Indicator calculation failed: "
                f"{exc}"
            )

            return

        # ====================================================
        # MINIMUM HISTORY
        # ====================================================

        if len(df) < 250:

            print(
                f"[{self.symbol}] "
                f"Waiting for enough 1h candles: "
                f"{len(df)}/250"
            )

            return

        # ====================================================
        # CURRENT = FORMING
        # PREVIOUS = COMPLETED
        # ====================================================

        current_timestamp = (
            df.index[-1]
        )

        completed_df = (
            df.iloc[:-1].copy()
        )

        if completed_df.empty:

            return

        # ====================================================
        # FIRST STARTUP
        #
        # IMPORTANT:
        # Do NOT replay historical trades.
        # ====================================================

        if self.last_completed_bar is None:

            self.last_completed_bar = (
                completed_df.index[-1]
            )

            self.engine._current_index = (
                len(df) - 2
            )

            self.bar_counter = (
                self.engine._current_index
            )

            # ------------------------------------------------
            # Initialize day state correctly
            # ------------------------------------------------

            current_equity = (
                self._calculate_equity(
                    df.iloc[-1]
                )
            )

            self.engine._reset_day_if_needed(
                current_timestamp,
                current_equity,
            )

            self.balance = float(
                self.engine.balance
            )

            self.peak_equity = max(
                self.peak_equity,
                current_equity,
            )

            self.engine.peak_equity = (
                self.peak_equity
            )

            self.day_start_equity = float(
                self.engine.day_start_equity
            )

            self.current_day = (
                self.engine.current_day
            )

            self.log_equity(
                current_timestamp,
                current_equity,
            )

            self.save_state()

            print(
                f"[{self.symbol}] "
                f"Paper trader initialized at "
                f"{current_timestamp}"
            )

            return

        # ====================================================
        # NEW COMPLETED BARS
        # ====================================================

        new_completed = (
            completed_df[
                completed_df.index
                > self.last_completed_bar
            ]
        )

        for timestamp, row in (
            new_completed.iterrows()
        ):

            self._process_completed_bar(
                timestamp,
                row,
            )

            self.last_completed_bar = (
                timestamp
            )

        # ====================================================
        # ENTRY DECISION
        #
        # Signal = completed candle
        # Entry = current candle open
        # ====================================================

        if len(df) >= 3:

            previous_row = (
                df.iloc[-2]
            )

            previous_previous_row = (
                df.iloc[-3]
            )

            self._process_current_hour(
                current_timestamp,
                df.iloc[-1],
                previous_row,
                previous_previous_row,
            )

        # ====================================================
        # FINAL SYNC
        # ====================================================

        self.balance = float(
            self.engine.balance
        )

        self.peak_equity = float(
            self.engine.peak_equity
        )

        self.day_start_equity = float(
            self.engine.day_start_equity
        )

        self.current_day = (
            self.engine.current_day
        )

        self.consecutive_losses = int(
            self.engine.consecutive_losses
        )

        self.cooldown_until = int(
            self.engine.cooldown_until
        )

        self.kill_switch = bool(
            self.engine.kill_switch
        )

        self.position = (
            self.engine.position
        )

        self.save_state()


    # ========================================================
    # STATUS
    # ========================================================

    def status(self):

        position_text = (
            "OPEN"
            if self.position is not None
            else "FLAT"
        )

        print(
            f"[{self.symbol}] "
            f"STATUS | "
            f"balance=${self.balance:.4f} | "
            f"position={position_text} | "
            f"peak=${self.peak_equity:.4f} | "
            f"losses={self.consecutive_losses} | "
            f"kill={self.kill_switch}"
        )
