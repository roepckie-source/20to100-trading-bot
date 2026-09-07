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

import json
import os
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


class PaperTrader:
    """
    Incremental paper-trading adapter around the existing
    V7-S0 survival engine.

    The backtest engine remains the source of truth for:
        - position sizing
        - risk multiplier
        - stops
        - trailing stops
        - daily loss protection
        - consecutive loss protection
        - global drawdown kill switch
        - fees
        - slippage
        - trade accounting
    """

    def __init__(self, symbol: str):

        self.symbol = symbol

        # ----------------------------------------------------
        # Directories
        # ----------------------------------------------------

        self.log_dir = Path(LOG_DIR)
        self.state_dir = Path(STATE_DIR)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # Engine
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
        # Persistent state
        # ----------------------------------------------------

        self.state_file = (
            self.state_dir
            / f"{self.symbol.replace('/', '_')}_paper_state.json"
        )

        self.last_completed_bar = None
        self.last_entry_hour = None

        self.bar_counter = -1

        self.balance = float(STARTING_CAPITAL)
        self.peak_equity = float(STARTING_CAPITAL)
        self.day_start_equity = float(STARTING_CAPITAL)

        self.current_day = None

        self.consecutive_losses = 0
        self.cooldown_until = -1

        self.kill_switch = False

        self.position = None

        # Number of already logged engine trades
        self.logged_trades = 0

        self.load_state()

        # ----------------------------------------------------
        # Make sure engine state matches restored paper state
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
    # STATE
    # ========================================================

    def load_state(self):

        if not self.state_file.exists():
            print(f"[{self.symbol}] No previous paper state found.")
            return

        try:

            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.last_completed_bar = self._parse_timestamp(
                state.get("last_completed_bar")
            )

            self.last_entry_hour = self._parse_timestamp(
                state.get("last_entry_hour")
            )

            self.bar_counter = int(
                state.get("bar_counter", -1)
            )

            self.balance = float(
                state.get("balance", STARTING_CAPITAL)
            )

            self.peak_equity = float(
                state.get("peak_equity", self.balance)
            )

            self.day_start_equity = float(
                state.get("day_start_equity", self.balance)
            )

            current_day = state.get("current_day")

            if current_day:
                self.current_day = date.fromisoformat(current_day)

            self.consecutive_losses = int(
                state.get("consecutive_losses", 0)
            )

            self.cooldown_until = int(
                state.get("cooldown_until", -1)
            )

            self.kill_switch = bool(
                state.get("kill_switch", False)
            )

            self.position = state.get("position")

            self.logged_trades = int(
                state.get("logged_trades", 0)
            )

            print(
                f"[{self.symbol}] Paper state restored | "
                f"balance=${self.balance:.4f}"
            )

        except Exception as exc:

            print(
                f"[{self.symbol}] WARNING: "
                f"Could not load state: {exc}"
            )

    # --------------------------------------------------------

    def save_state(self):

        state = {
            "symbol": self.symbol,
            "last_completed_bar": self._serialize_timestamp(
                self.last_completed_bar
            ),
            "last_entry_hour": self._serialize_timestamp(
                self.last_entry_hour
            ),
            "bar_counter": self.bar_counter,
            "balance": self.balance,
            "peak_equity": self.peak_equity,
            "day_start_equity": self.day_start_equity,
            "current_day": (
                self.current_day.isoformat()
                if isinstance(self.current_day, date)
                else None
            ),
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until,
            "kill_switch": self.kill_switch,
            "position": self.position,
            "logged_trades": self.logged_trades,
            "updated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        temp_file = self.state_file.with_suffix(".tmp")

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(
                state,
                f,
                indent=2,
                default=str,
            )

        os.replace(temp_file, self.state_file)

    # ========================================================
    # TIMESTAMP HELPERS
    # ========================================================

    @staticmethod
    def _serialize_timestamp(value):

        if value is None:
            return None

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, datetime):
            return value.isoformat()

        return str(value)

    # --------------------------------------------------------

    @staticmethod
    def _parse_timestamp(value):

        if not value:
            return None

        try:
            return pd.Timestamp(value)
        except Exception:
            return None

    # ========================================================
    # LOGGING
    # ========================================================

    def _append_csv(self, filename, row):

        path = self.log_dir / filename

        df = pd.DataFrame([row])

        df.to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False,
        )

    # --------------------------------------------------------

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
                "timestamp": timestamp,
                "symbol": self.symbol,
                "variant": VARIANT,
                "signal": bool(signal),
                "close": row.get("close"),
                "ema_100": row.get("ema_100"),
                "ema_200": row.get("ema_200"),
                "ema_200_slope": row.get("ema_200_slope"),
                "adx_14": row.get("adx_14"),
                "atr_14": row.get("atr_14"),
                "atr_14_ma50": row.get("atr_14_ma50"),
                "donchian_high_20": row.get("donchian_high_20"),
                "reason": reason,
            },
        )

    # --------------------------------------------------------

    def log_equity(self, timestamp, equity):

        drawdown = 0.0

        if self.peak_equity > 0:

            drawdown = (
                (equity - self.peak_equity)
                / self.peak_equity
                * 100.0
            )

        self._append_csv(
            "paper_equity.csv",
            {
                "timestamp": timestamp,
                "symbol": self.symbol,
                "balance": self.balance,
                "equity": equity,
                "peak_equity": self.peak_equity,
                "drawdown_pct": drawdown,
                "position_open": self.position is not None,
                "kill_switch": self.kill_switch,
            },
        )

    # --------------------------------------------------------

    def log_trade(self, trade):

        if not trade:
            return

        row = dict(trade)

        row["symbol"] = self.symbol
        row["variant"] = VARIANT

        self._append_csv(
            "paper_trades.csv",
            row,
        )

    # ========================================================
    # EQUITY
    # ========================================================

    def _calculate_equity(self, row):

        return self.engine._calculate_equity(row)

    # ========================================================
    # PROCESS COMPLETED BAR
    # ========================================================

    def _process_completed_bar(
        self,
        timestamp,
        row,
    ):

        self.engine._current_index += 1
        self.bar_counter = self.engine._current_index

        equity = self._calculate_equity(row)

        # ----------------------------------------------------
        # Day reset
        # ----------------------------------------------------

        self.engine._reset_day_if_needed(timestamp, equity)

        self.day_start_equity = self.engine.day_start_equity
        self.current_day = self.engine.current_day

        # ----------------------------------------------------
        # Peak equity
        # ----------------------------------------------------

        if equity > self.engine.peak_equity:
            self.engine.peak_equity = equity

        self.peak_equity = self.engine.peak_equity

        # ----------------------------------------------------
        # Global drawdown
        # ----------------------------------------------------

        if self.engine._global_drawdown_hit(equity):

            if not self.engine.kill_switch:

                print(
                    f"[{self.symbol}] "
                    f"!!! GLOBAL DRAWDOWN KILL SWITCH !!!"
                )

            self.engine.kill_switch = True
            self.kill_switch = True

        # ----------------------------------------------------
        # Position management
        #
        # Same order as V7-S0:
        #
        # 1. stop
        # 2. trailing stop
        # ----------------------------------------------------

        if self.engine.position is not None:

            position_before = self.engine.position

            self.engine._manage_position(
                timestamp,
                row,
                equity,
            )

            self.position = self.engine.position

            # ------------------------------------------------
            # Trade closed?
            # ------------------------------------------------

            if (
                position_before is not None
                and self.engine.position is None
            ):

                self._log_new_trades()

        # ----------------------------------------------------
        # Equity log
        # ----------------------------------------------------

        self.balance = float(self.engine.balance)

        self.log_equity(
            timestamp,
            equity,
        )

    # ========================================================
    # ENTRY FOR CURRENT HOUR
    # ========================================================

    def _process_current_hour(
        self,
        timestamp,
        current_row,
        previous_row,
        previous_previous_row,
    ):

        # ----------------------------------------------------
        # Only process one entry decision per hour
        # ----------------------------------------------------

        if (
            self.last_entry_hour is not None
            and timestamp <= self.last_entry_hour
        ):
            return

        self.last_entry_hour = timestamp

        self.engine._current_index += 1
        self.bar_counter = self.engine._current_index

        equity = self._calculate_equity(current_row)

        # ----------------------------------------------------
        # Safety: existing position
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
        # Daily loss protection
        # ----------------------------------------------------

        if self.engine._daily_loss_limit_hit(equity):

            self.log_signal(
                timestamp,
                False,
                previous_row,
                "daily_loss_limit",
            )

            return

        # ----------------------------------------------------
        # Consecutive-loss cooldown
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
        # V6-C signal
        #
        # IMPORTANT:
        # Signal is calculated from the COMPLETED candle.
        # Entry is executed on the CURRENT candle open.
        # ----------------------------------------------------

        try:

            signal = bool(
                buy_signal(
                    previous_row,
                    previous_previous_row,
                    adx_min=self.engine.adx_min,
                )
            )

        except TypeError:

            signal = bool(
                buy_signal(
                    previous_row,
                    previous_previous_row,
                )
            )

        except Exception as exc:

            print(
                f"[{self.symbol}] Signal error: {exc}"
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

        risk_multiplier = self.engine._risk_multiplier(
            equity
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
        # Existing V7-S0 engine method.
        # This preserves:
        # - slippage
        # - ATR stop
        # - position sizing
        # - fees
        # ----------------------------------------------------

        entered = self.engine._enter(
            timestamp,
            previous_row,
            current_row,
            equity,
        )

        if entered:

            self.position = self.engine.position

            self.balance = float(
                self.engine.balance
            )

            print(
                f"[{self.symbol}] "
                f"🟢 PAPER BUY | "
                f"{timestamp} | "
                f"price={self.position.get('entry_price', 0):.4f} | "
                f"qty={self.position.get('quantity', 0):.8f}"
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

        while self.logged_trades < len(trades):

            trade = trades[self.logged_trades]

            self.log_trade(trade)

            self.logged_trades += 1

            pnl = float(
                trade.get(
                    "pnl",
                    0.0,
                )
            )

            if pnl > 0:

                print(
                    f"[{self.symbol}] "
                    f"✅ PAPER SELL | "
                    f"PnL=${pnl:.4f}"
                )

            else:

                print(
                    f"[{self.symbol}] "
                    f"🔴 PAPER SELL | "
                    f"PnL=${pnl:.4f}"
                )

    # ========================================================
    # MAIN PROCESS FUNCTION
    # ========================================================

    def process(self, hourly_df: pd.DataFrame):

        if hourly_df is None:
            return

        if hourly_df.empty:
            return

        df = hourly_df.copy()

        # ----------------------------------------------------
        # Normalize index
        # ----------------------------------------------------

        if not isinstance(df.index, pd.DatetimeIndex):

            df.index = pd.to_datetime(
                df.index,
                utc=True,
            )

        else:

            if df.index.tz is None:

                df.index = df.index.tz_localize(
                    "UTC"
                )

            else:

                df.index = df.index.tz_convert(
                    "UTC"
                )

        df = df.sort_index()

        # ----------------------------------------------------
        # Indicators
        # ----------------------------------------------------

        try:

            df = calculate_indicators(df)

        except Exception as exc:

            print(
                f"[{self.symbol}] "
                f"Indicator calculation failed: {exc}"
            )

            return

        if len(df) < 250:

            print(
                f"[{self.symbol}] "
                f"Waiting for enough 1h candles: "
                f"{len(df)}/250"
            )

            return

        # ----------------------------------------------------
        # Current candle = still forming
        # Previous candle = completed
        # ----------------------------------------------------

        current_timestamp = df.index[-1]

        completed_df = df.iloc[:-1].copy()

        if completed_df.empty:
            return

        # ----------------------------------------------------
        # Process newly completed candles
        # ----------------------------------------------------

        if self.last_completed_bar is None:

            # First startup:
            #
            # We do NOT replay history.
            # We initialize from the current market state.

            self.last_completed_bar = (
                completed_df.index[-1]
            )

            self.engine._current_index = (
                len(df) - 2
            )

            self.bar_counter = (
                self.engine._current_index
            )

            current_equity = self._calculate_equity(
                df.iloc[-1]
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

        # ----------------------------------------------------
        # Find all completed bars not processed yet
        # ----------------------------------------------------

        new_completed = completed_df[
            completed_df.index
            > self.last_completed_bar
        ]

        for timestamp, row in new_completed.iterrows():

            self._process_completed_bar(
                timestamp,
                row,
            )

            self.last_completed_bar = timestamp

        # ----------------------------------------------------
        # Current hour entry
        # ----------------------------------------------------

        if len(df) >= 3:

            previous_row = df.iloc[-2]

            previous_previous_row = df.iloc[-3]

            self._process_current_hour(
                current_timestamp,
                df.iloc[-1],
                previous_row,
                previous_previous_row,
            )

        # ----------------------------------------------------
        # Restore references
        # ----------------------------------------------------

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

        self.position = self.engine.position

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

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
