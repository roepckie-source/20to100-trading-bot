# ============================================================
# V2 Backtest Engine
# ============================================================

import pandas as pd
import numpy as np

from strategy.strategy_v2 import StrategyV2


class V2BacktestEngine:

    def __init__(
        self,
        starting_capital=20.0,
        fee_rate=0.001,
        slippage=0.0005,
        donchian_period=20,
        ema_period=200,
        atr_period=14,
        atr_stop_multiplier=2.5,
        atr_trailing_multiplier=2.5,
    ):

        self.starting_capital = float(starting_capital)
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage)

        self.strategy = StrategyV2(
            donchian_period=donchian_period,
            ema_period=ema_period,
            atr_period=atr_period,
            atr_stop_multiplier=atr_stop_multiplier,
            atr_trailing_multiplier=atr_trailing_multiplier,
        )

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    def run(self, df):

        df = df.copy()

        required_columns = [
            "open",
            "high",
            "low",
            "close",
        ]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(
                    f"Fehlende Spalte: {column}"
                )

        df = self.strategy.calculate_indicators(df)
        df = df.reset_index(drop=True)

        capital = self.starting_capital

        position = False

        entry_price = None
        entry_index = None
        stop_price = None
        highest_price = None

        trades = []

        equity_curve = []

        for i in range(len(df)):

            row = df.iloc[i]

            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            atr = row["atr"]

            # ------------------------------------------------
            # EQUITY
            # ------------------------------------------------

            if position:

                current_equity = (
                    capital
                    * close
                    / entry_price
                )

            else:
                current_equity = capital

            equity_curve.append(current_equity)

            # ------------------------------------------------
            # OPEN POSITION
            # ------------------------------------------------

            if position:

                # Highest price aktualisieren
                if high > highest_price:
                    highest_price = high

                # ATR Trailing Stop
                if not pd.isna(atr) and atr > 0:

                    trailing_stop = (
                        highest_price
                        - atr
                        * self.strategy.atr_trailing_multiplier
                    )

                    if trailing_stop > stop_price:
                        stop_price = trailing_stop

                # Stop getroffen?
                if low <= stop_price:

                    exit_price = stop_price

                    # Slippage beim Verkauf
                    exit_price *= (
                        1 - self.slippage
                    )

                    gross_return = (
                        exit_price
                        / entry_price
                        - 1
                    )

                    # Verkaufskosten
                    net_return = (
                        gross_return
                        - self.fee_rate
                        - self.fee_rate
                    )

                    capital *= (
                        1 + net_return
                    )

                    trades.append({
                        "entry_index": entry_index,
                        "exit_index": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": net_return * 100,
                        "exit_reason": "ATR_STOP",
                    })

                    position = False
                    entry_price = None
                    entry_index = None
                    stop_price = None
                    highest_price = None

                    continue

            # ------------------------------------------------
            # ENTRY
            # ------------------------------------------------

            if not position:

                if self.strategy.entry_signal(df, i):

                    # Wir kaufen zum nächsten verfügbaren
                    # Preis mit Slippage.

                    entry_price = close * (
                        1 + self.slippage
                    )

                    # Entry fee
                    capital *= (
                        1 - self.fee_rate
                    )

                    atr_value = float(atr)

                    stop_price = (
                        self.strategy
                        .calculate_initial_stop(
                            entry_price,
                            atr_value
                        )
                    )

                    highest_price = high

                    entry_index = i

                    position = True

        # ----------------------------------------------------
        # POSITION AM ENDE SCHLIESSEN
        # ----------------------------------------------------

        if position:

            exit_price = float(
                df.iloc[-1]["close"]
            )

            exit_price *= (
                1 - self.slippage
            )

            gross_return = (
                exit_price
                / entry_price
                - 1
            )

            net_return = (
                gross_return
                - self.fee_rate
            )

            capital *= (
                1 + net_return
            )

            trades.append({
                "entry_index": entry_index,
                "exit_index": len(df) - 1,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": net_return * 100,
                "exit_reason": "END_OF_DATA",
            })

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        trades_df = pd.DataFrame(trades)

        equity = pd.Series(equity_curve)

        if len(equity) > 0:

            rolling_max = equity.cummax()

            drawdown = (
                equity / rolling_max - 1
            )

            max_drawdown = (
                drawdown.min() * 100
            )

        else:

            max_drawdown = 0.0

        if len(trades_df) > 0:

            wins = trades_df[
                trades_df["return_pct"] > 0
            ]

            losses = trades_df[
                trades_df["return_pct"] <= 0
            ]

            win_rate = (
                len(wins)
                / len(trades_df)
                * 100
            )

            gross_profit = (
                wins["return_pct"]
                .sum()
            )

            gross_loss = abs(
                losses["return_pct"]
                .sum()
            )

            if gross_loss > 0:
                profit_factor = (
                    gross_profit
                    / gross_loss
                )
            else:
                profit_factor = np.inf

            expectancy = (
                trades_df["return_pct"]
                .mean()
            )

        else:

            win_rate = 0.0
            profit_factor = 0.0
            expectancy = 0.0

        result = {
            "starting_capital": self.starting_capital,
            "final_capital": capital,
            "return_pct": (
                (capital / self.starting_capital - 1)
                * 100
            ),
            "trades": len(trades_df),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_pct": expectancy,
            "max_drawdown_pct": max_drawdown,
        }

        return result, trades_df
