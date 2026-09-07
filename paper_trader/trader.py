# ============================================================
# 20to100 Trading Bot
# V7-S0 PAPER TRADER
# ============================================================

from __future__ import annotations

import json

from pathlib import Path

import pandas as pd

from backtest.v7_survival_engine import (
    V7SurvivalEngine,
)

from strategy.strategy_v6 import (
    buy_signal,
    calculate_indicators,
)

from config_paper import (
    ADX_MIN,
    ATR_STOP_MULTIPLIER,
    BASE_RISK_PER_TRADE,
    FEE_RATE,
    GLOBAL_MAX_DRAWDOWN,
    LOG_DIR,
    LOSS_COOLDOWN_BARS,
    MAX_CONSECUTIVE_LOSSES,
    MAX_DAILY_LOSS,
    SLIPPAGE_RATE,
    STARTING_CAPITAL,
    STATE_FILE,
    SIGNALS_FILE,
    TRADES_FILE,
    EQUITY_FILE,
    TRAILING_ATR_MULTIPLIER,
)


# ============================================================
# PAPER TRADER
# ============================================================

class PaperTrader:

    STATE_VERSION = 1


    def __init__(
        self,
        symbol: str,
        starting_balance:
            float = STARTING_CAPITAL,
    ):

        self.symbol = symbol

        # ----------------------------------------------------
        # EXISTING V7-S0 ENGINE
        # ----------------------------------------------------

        self.engine = V7SurvivalEngine(

            starting_balance=
                starting_balance,

            base_risk_per_trade=
                BASE_RISK_PER_TRADE,

            fee_rate=
                FEE_RATE,

            slippage_rate=
                SLIPPAGE_RATE,

            atr_stop_multiplier=
                ATR_STOP_MULTIPLIER,

            trailing_atr_multiplier=
                TRAILING_ATR_MULTIPLIER,

            adx_min=
                ADX_MIN,

            max_daily_loss=
                MAX_DAILY_LOSS,

            max_consecutive_losses=
                MAX_CONSECUTIVE_LOSSES,

            loss_cooldown_bars=
                LOSS_COOLDOWN_BARS,

            global_max_drawdown=
                GLOBAL_MAX_DRAWDOWN,

            variant=
                "V6_C",
        )

        self.last_completed_bar = None

        self.last_entry_hour = None

        self._ensure_logs()


    # ========================================================
    # LOG DIRECTORY
    # ========================================================

    def _ensure_logs(self):

        Path(
            LOG_DIR
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


    # ========================================================
    # LOAD STATE
    # ========================================================

    def load_state(self):

        path = Path(
            STATE_FILE
        )

        if not path.exists():

            return False

        try:

            state = json.loads(
                path.read_text()
            )

            saved = state.get(
                self.symbol
            )

            if not saved:

                return False

            if (
                saved.get("version")
                != self.STATE_VERSION
            ):

                return False

            if saved.get(
                "last_completed_bar"
            ):

                self.last_completed_bar = (
                    pd.Timestamp(
                        saved[
                            "last_completed_bar"
                        ]
                    )
                )

            if saved.get(
                "last_entry_hour"
            ):

                self.last_entry_hour = (
                    pd.Timestamp(
                        saved[
                            "last_entry_hour"
                        ]
                    )
                )

            e = self.engine

            e.balance = float(
                saved["balance"]
            )

            e.peak_equity = float(
                saved["peak_equity"]
            )

            e.day_start_equity = float(
                saved["day_start_equity"]
            )

            e.consecutive_losses = int(
                saved[
                    "consecutive_losses"
                ]
            )

            e.cooldown_until = int(
                saved[
                    "cooldown_until"
                ]
            )

            e.kill_switch = bool(
                saved[
                    "kill_switch"
                ]
            )

            e.position = saved.get(
                "position"
            )

            return True

        except Exception as exc:

            print(
                f"[{self.symbol}] "
                f"State load failed: "
                f"{exc}"
            )

            return False


    # ========================================================
    # SAVE STATE
    # ========================================================

    def save_state(self):

        path = Path(
            STATE_FILE
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state = {}

        if path.exists():

            try:

                state = json.loads(
                    path.read_text()
                )

            except Exception:

                state = {}

        e = self.engine

        state[
            self.symbol
        ] = {

            "version":
                self.STATE_VERSION,

            "last_completed_bar":
                (
                    self.last_completed_bar.isoformat()
                    if self.last_completed_bar
                    is not None
                    else None
                ),

            "last_entry_hour":
                (
                    self.last_entry_hour.isoformat()
                    if self.last_entry_hour
                    is not None
                    else None
                ),

            "balance":
                e.balance,

            "peak_equity":
                e.peak_equity,

            "day_start_equity":
                e.day_start_equity,

            "consecutive_losses":
                e.consecutive_losses,

            "cooldown_until":
                e.cooldown_until,

            "kill_switch":
                e.kill_switch,

            "position":
                e.position,
        }

        path.write_text(
            json.dumps(
                state,
                indent=2,
                default=str,
            )
        )


    # ========================================================
    # CSV LOGGER
    # ========================================================

    @staticmethod
    def _append_csv(
        path: str,
        rows: list[dict],
    ):

        if not rows:

            return

        target = Path(
            path
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            rows
        ).to_csv(
            target,
            mode="a",
            header=not target.exists(),
            index=False,
        )


    # ========================================================
    # SIGNAL LOG
    # ========================================================

    def _log_signal(
        self,
        timestamp,
        signal,
        row,
    ):

        self._append_csv(
            SIGNALS_FILE,
            [
                {
                    "timestamp":
                        timestamp,

                    "symbol":
                        self.symbol,

                    "signal":
                        bool(signal),

                    "close":
                        float(
                            row["close"]
                        ),

                    "atr_14":
                        float(
                            row["atr_14"]
                        ),

                    "adx_14":
                        float(
                            row["adx_14"]
                        ),

                    "ema_100":
                        float(
                            row["ema_100"]
                        ),

                    "ema_200":
                        float(
                            row["ema_200"]
                        ),

                    "donchian_high_20":
                        float(
                            row[
                                "donchian_high_20"
                            ]
                        ),
                }
            ],
        )


    # ========================================================
    # PROCESS
    # ========================================================

    def process(
        self,
        hourly: pd.DataFrame,
    ):

        if len(hourly) < 300:

            print(
                f"[{self.symbol}] "
                f"Not enough 1h candles: "
                f"{len(hourly)}"
            )

            return

        # ----------------------------------------------------
        # CALCULATE V6-C INDICATORS
        # ----------------------------------------------------

        data = calculate_indicators(
            hourly.copy()
        )

        data = data.sort_index()

        now = pd.Timestamp.now(
            tz="UTC"
        )

        current_hour = now.floor(
            "1h"
        )

        # ----------------------------------------------------
        # ONLY COMPLETED CANDLES
        # ----------------------------------------------------

        completed = data[
            data.index < current_hour
        ].copy()

        current = data[
            data.index == current_hour
        ].copy()

        if (
            len(completed) < 300
            or current.empty
        ):

            return

        # ----------------------------------------------------
        # FIRST START
        # ----------------------------------------------------

        if (
            self.last_completed_bar
            is None
        ):

            self.last_completed_bar = (
                completed.index[-1]
            )

            self.last_entry_hour = (
                current_hour
            )

            self.save_state()

            print(
                f"[{self.symbol}] "
                f"INITIALIZED"
            )

            print(
                f"Completed candle: "
                f"{self.last_completed_bar}"
            )

            print(
                f"Next hour: "
                f"{self.last_entry_hour}"
            )

            return

        # ----------------------------------------------------
        # NEW COMPLETED CANDLES
        # ----------------------------------------------------

        new_completed = completed[
            completed.index
            > self.last_completed_bar
        ]

        for (
            timestamp,
            row
        ) in new_completed.iterrows():

            price = float(
                row["close"]
            )

            equity = (
                self.engine
                ._calculate_equity(
                    price
                )
            )

            self.engine._reset_day_if_needed(
                timestamp,
                equity,
            )

            self.engine.peak_equity = max(
                self.engine.peak_equity,
                equity,
            )

            # ----------------------------------------------
            # POSITION MANAGEMENT
            # ----------------------------------------------

            if (
                self.engine.position
                is not None
            ):

                position = (
                    self.engine.position
                )

                # STOP
                if (
                    float(row["low"])
                    <= position["stop"]
                ):

                    self.engine._exit(
                        timestamp,
                        position["stop"],
                        "ATR_STOP",
                    )

                    print(
                        f"[{self.symbol}] "
                        f"PAPER SELL "
                        f"reason=ATR_STOP"
                    )

                else:

                    # HIGHEST PRICE
                    position[
                        "highest"
                    ] = max(
                        position[
                            "highest"
                        ],
                        float(
                            row["high"]
                        ),
                    )

                    atr = float(
                        row["atr_14"]
                    )

                    if atr > 0:

                        trailing_stop = (
                            position[
                                "highest"
                            ]
                            -
                            atr
                            *
                            self.engine
                            .trailing_atr_multiplier
                        )

                        if (
                            trailing_stop
                            >
                            position["stop"]
                        ):

                            position[
                                "stop"
                            ] = (
                                trailing_stop
                            )

            self.last_completed_bar = (
                timestamp
            )

        # ----------------------------------------------------
        # NEW HOUR
        # ----------------------------------------------------

        if (
            self.last_entry_hour
            is not None
            and
            current_hour
            <=
            self.last_entry_hour
        ):

            self.save_state()

            return

        # ----------------------------------------------------
        # SIGNAL
        #
        # IMPORTANT:
        #
        # Signal = previous completed candle
        #
        # Execution = current hour OPEN
        # ----------------------------------------------------

        previous = data.iloc[-2]

        previous_previous = (
            data.iloc[-3]
        )

        current_row = data.iloc[-1]

        execution_price = float(
            current_row["open"]
        )

        e = self.engine

        equity = e._calculate_equity(
            execution_price
        )

        e._reset_day_if_needed(
            current_hour,
            equity,
        )

        e.peak_equity = max(
            e.peak_equity,
            equity,
        )

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        if (
            e.position is None
            and
            not e.kill_switch
        ):

            if not e._daily_loss_limit_hit(
                equity
            ):

                if (
                    e._current_index
                    >=
                    e.cooldown_until
                ):

                    signal = buy_signal(
                        previous,
                        previous_previous,
                        variant="V6_C",
                        adx_min=ADX_MIN,
                    )

                    self._log_signal(
                        current_hour,
                        signal,
                        previous,
                    )

                    if signal:

                        e._enter(
                            current_hour,
                            previous,
                            current_row,
                            equity,
                        )

                        print()
                        print(
                            "=" * 60
                        )

                        print(
                            f"[{self.symbol}] "
                            f"PAPER BUY"
                        )

                        print(
                            f"Price: "
                            f"{execution_price}"
                        )

                        print(
                            f"Time: "
                            f"{current_hour}"
                        )

                        print(
                            "=" * 60
                        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        self.last_entry_hour = (
            current_hour
        )

        self.save_state()

        print(
            f"[{self.symbol}] "
            f"balance={e.balance:.4f} "
            f"position="
            f"{'OPEN' if e.position else 'NONE'} "
            f"trades={len(e.trades)} "
            f"kill={e.kill_switch}"
        )
