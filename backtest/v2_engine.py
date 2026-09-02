# ============================================================
# V2.1 Backtest Engine
#
# Donchian Breakout + EMA200 + ATR
#
# Realistische Modellierung:
# - All-in Position
# - Entry Fee
# - Exit Fee
# - Entry Slippage
# - Exit Slippage
# - Entry am nächsten Kerzen-Open
# - Initialer ATR Stop
# - ATR Trailing Stop
# - Keine Verwendung des zukünftigen Kerzenverlaufs
# - Equity Curve
# - Max Drawdown
# - Profit Factor
# - Expectancy
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

    # ========================================================
    # BACKTEST
    # ========================================================

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

        # ----------------------------------------------------
        # Indicators
        # ----------------------------------------------------

        df = self.strategy.calculate_indicators(df)

        df = df.reset_index(drop=True)

        # ----------------------------------------------------
        # Capital
        # ----------------------------------------------------

        cash = self.starting_capital

        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        position = False

        quantity = 0.0

        entry_price = None
        entry_index = None

        initial_stop = None
        trailing_stop = None
        stop_price = None

        highest_price = None

        entry_fee = 0.0
        exit_fee = 0.0

        trades = []

        equity_curve = []

        # ====================================================
        # MAIN LOOP
        # ====================================================

        for i in range(len(df)):

            row = df.iloc[i]

            open_price = float(row["open"])
            high_price = float(row["high"])
            low_price = float(row["low"])
            close_price = float(row["close"])

            atr = row["atr"]

            # =================================================
            # EQUITY BEFORE ACTION
            # =================================================

            if position:

                current_equity = (
                    quantity * close_price
                )

            else:

                current_equity = cash

            equity_curve.append(
                current_equity
            )

            # =================================================
            # MANAGE EXISTING POSITION
            #
            # IMPORTANT:
            # The stop used for this candle was already
            # determined before this candle.
            # This prevents look-ahead bias.
            # =================================================

            if position:

                # ---------------------------------------------
                # STOP CHECK
                # ---------------------------------------------

                if low_price <= stop_price:

                    # Stop execution.
                    #
                    # We assume the stop can be filled at the
                    # stop price. This is already conservative
                    # because slippage is added below.

                    exit_price = stop_price

                    # Exit slippage
                    exit_price *= (
                        1.0 - self.slippage
                    )

                    gross_proceeds = (
                        quantity * exit_price
                    )

                    exit_fee = (
                        gross_proceeds
                        * self.fee_rate
                    )

                    net_proceeds = (
                        gross_proceeds
                        - exit_fee
                    )

                    # -----------------------------------------
                    # Trade return
                    # -----------------------------------------

                    total_entry_cost = (
                        quantity * entry_price
                        + entry_fee
                    )

                    trade_profit = (
                        net_proceeds
                        - total_entry_cost
                    )

                    trade_return_pct = (
                        trade_profit
                        / total_entry_cost
                        * 100.0
                    )

                    cash = net_proceeds

                    trades.append({
                        "entry_index": entry_index,
                        "exit_index": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": quantity,
                        "entry_fee": entry_fee,
                        "exit_fee": exit_fee,
                        "profit": trade_profit,
                        "return_pct": trade_return_pct,
                        "exit_reason": "ATR_STOP",
                    })

                    # Reset position
                    position = False

                    quantity = 0.0

                    entry_price = None
                    entry_index = None

                    initial_stop = None
                    trailing_stop = None
                    stop_price = None

                    highest_price = None

                    entry_fee = 0.0
                    exit_fee = 0.0

                    continue

                # ---------------------------------------------
                # UPDATE TRAILING STOP
                #
                # We use the current candle's information only
                # to create a stop for the NEXT candle.
                # ---------------------------------------------

                if not pd.isna(atr) and atr > 0:

                    if high_price > highest_price:

                        highest_price = high_price

                    new_trailing_stop = (
                        highest_price
                        - float(atr)
                        * self.strategy.atr_trailing_multiplier
                    )

                    if (
                        trailing_stop is None
                        or new_trailing_stop > trailing_stop
                    ):

                        trailing_stop = (
                            new_trailing_stop
                        )

                    # Stop can only move UP.
                    if trailing_stop > stop_price:

                        stop_price = trailing_stop

            # =================================================
            # ENTRY
            #
            # Signal is generated at the CLOSE of candle i.
            #
            # We therefore enter at the OPEN of candle i+1.
            #
            # This eliminates look-ahead bias.
            # =================================================

            if not position:

                # Last candle cannot have a next candle.
                if i >= len(df) - 1:
                    continue

                if self.strategy.entry_signal(df, i):

                    next_row = df.iloc[i + 1]

                    next_open = float(
                        next_row["open"]
                    )

                    # -----------------------------------------
                    # Entry slippage
                    # -----------------------------------------

                    entry_price = (
                        next_open
                        * (1.0 + self.slippage)
                    )

                    # -----------------------------------------
                    # All-in position
                    #
                    # cash = quantity * entry_price
                    #       + entry fee
                    #
                    # Therefore:
                    #
                    # quantity =
                    # cash / (entry_price * (1+fee))
                    # -----------------------------------------

                    quantity = (
                        cash
                        / (
                            entry_price
                            * (1.0 + self.fee_rate)
                        )
                    )

                    if quantity <= 0:
                        continue

                    entry_notional = (
                        quantity
                        * entry_price
                    )

                    entry_fee = (
                        entry_notional
                        * self.fee_rate
                    )

                    total_entry_cost = (
                        entry_notional
                        + entry_fee
                    )

                    # Safety check
                    if total_entry_cost > cash:

                        quantity = 0.0

                        continue

                    # -----------------------------------------
                    # Calculate initial ATR stop
                    #
                    # Use ATR from the signal candle.
                    # -----------------------------------------

                    signal_atr = df.iloc[i]["atr"]

                    if (
                        pd.isna(signal_atr)
                        or signal_atr <= 0
                    ):

                        quantity = 0.0

                        continue

                    initial_stop = (
                        self.strategy
                        .calculate_initial_stop(
                            entry_price,
                            float(signal_atr)
                        )
                    )

                    # -----------------------------------------
                    # Position opened
                    # -----------------------------------------

                    cash = 0.0

                    position = True

                    entry_index = i + 1

                    highest_price = (
                        entry_price
                    )

                    trailing_stop = None

                    stop_price = initial_stop

        # ====================================================
        # CLOSE OPEN POSITION AT END OF DATA
        # ====================================================

        if position:

            final_close = float(
                df.iloc[-1]["close"]
            )

            # Exit slippage
            exit_price = (
                final_close
                * (1.0 - self.slippage)
            )

            gross_proceeds = (
                quantity * exit_price
            )

            exit_fee = (
                gross_proceeds
                * self.fee_rate
            )

            net_proceeds = (
                gross_proceeds
                - exit_fee
            )

            total_entry_cost = (
                quantity * entry_price
                + entry_fee
            )

            trade_profit = (
                net_proceeds
                - total_entry_cost
            )

            trade_return_pct = (
                trade_profit
                / total_entry_cost
                * 100.0
            )

            cash = net_proceeds

            trades.append({
                "entry_index": entry_index,
                "exit_index": len(df) - 1,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "profit": trade_profit,
                "return_pct": trade_return_pct,
                "exit_reason": "END_OF_DATA",
            })

            position = False

        # ====================================================
        # RESULTS
        # ====================================================

        final_capital = cash

        trades_df = pd.DataFrame(
            trades
        )

        # ====================================================
        # EQUITY / DRAWDOWN
        # ====================================================

        equity = pd.Series(
            equity_curve,
            dtype=float
        )

        # Add final capital so the final state is represented.
        if len(equity) == 0:

            equity = pd.Series(
                [self.starting_capital],
                dtype=float
            )

        else:

            equity = pd.concat(
                [
                    equity,
                    pd.Series(
                        [final_capital]
                    )
                ],
                ignore_index=True
            )

        rolling_max = (
            equity
            .cummax()
        )

        drawdown = (
            equity / rolling_max
            - 1.0
        )

        max_drawdown_pct = (
            drawdown.min()
            * 100.0
        )

        # ====================================================
        # TRADE STATISTICS
        # ====================================================

        if len(trades_df) > 0:

            winners = trades_df[
                trades_df["profit"] > 0
            ]

            losers = trades_df[
                trades_df["profit"] <= 0
            ]

            win_rate = (
                len(winners)
                / len(trades_df)
                * 100.0
            )

            gross_profit = (
                winners["profit"]
                .sum()
            )

            gross_loss = abs(
                losers["profit"]
                .sum()
            )

            if gross_loss > 0:

                profit_factor = (
                    gross_profit
                    / gross_loss
                )

            else:

                profit_factor = np.inf

            expectancy_dollars = (
                trades_df["profit"]
                .mean()
            )

            expectancy_pct = (
                trades_df["return_pct"]
                .mean()
            )

            total_entry_fees = (
                trades_df["entry_fee"]
                .sum()
            )

            total_exit_fees = (
                trades_df["exit_fee"]
                .sum()
            )

            total_fees = (
                total_entry_fees
                + total_exit_fees
            )

        else:

            win_rate = 0.0

            profit_factor = 0.0

            expectancy_dollars = 0.0

            expectancy_pct = 0.0

            total_entry_fees = 0.0

            total_exit_fees = 0.0

            total_fees = 0.0

        # ====================================================
        # RESULT
        # ====================================================

        result = {
            "starting_capital": (
                self.starting_capital
            ),

            "final_capital": (
                final_capital
            ),

            "profit_loss": (
                final_capital
                - self.starting_capital
            ),

            "return_pct": (
                (
                    final_capital
                    / self.starting_capital
                    - 1.0
                )
                * 100.0
            ),

            "trades": len(
                trades_df
            ),

            "win_rate": win_rate,

            "profit_factor": (
                profit_factor
            ),

            "expectancy": (
                expectancy_dollars
            ),

            "expectancy_pct": (
                expectancy_pct
            ),

            "max_drawdown_pct": (
                max_drawdown_pct
            ),

            "entry_fees": (
                total_entry_fees
            ),

            "exit_fees": (
                total_exit_fees
            ),

            "total_fees": (
                total_fees
            ),
        }

        return result, trades_df
