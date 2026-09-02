# ============================================================
# Strategy V2
# Donchian Breakout + EMA200 + ATR
# ============================================================

import pandas as pd
import numpy as np


class StrategyV2:
    """
    V2:
    - Donchian Breakout
    - EMA200 Trendfilter
    - ATR Stop Loss
    - ATR Trailing Stop

    Long-only.
    """

    def __init__(
        self,
        donchian_period=20,
        ema_period=200,
        atr_period=14,
        atr_stop_multiplier=2.5,
        atr_trailing_multiplier=2.5,
    ):
        self.donchian_period = int(donchian_period)
        self.ema_period = int(ema_period)
        self.atr_period = int(atr_period)

        self.atr_stop_multiplier = float(atr_stop_multiplier)
        self.atr_trailing_multiplier = float(atr_trailing_multiplier)

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    def calculate_indicators(self, df):

        df = df.copy()

        # EMA
        df["ema200"] = (
            df["close"]
            .ewm(span=self.ema_period, adjust=False)
            .mean()
        )

        # Donchian Channel
        df["donchian_high"] = (
            df["high"]
            .rolling(self.donchian_period)
            .max()
            .shift(1)
        )

        df["donchian_low"] = (
            df["low"]
            .rolling(self.donchian_period)
            .min()
            .shift(1)
        )

        # True Range
        previous_close = df["close"].shift(1)

        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - previous_close).abs()
        tr3 = (df["low"] - previous_close).abs()

        df["tr"] = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        # ATR
        df["atr"] = (
            df["tr"]
            .rolling(self.atr_period)
            .mean()
        )

        return df

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    def entry_signal(self, df, index):

        if index < max(
            self.ema_period,
            self.donchian_period,
            self.atr_period
        ):
            return False

        row = df.iloc[index]

        if pd.isna(row["ema200"]):
            return False

        if pd.isna(row["donchian_high"]):
            return False

        if pd.isna(row["atr"]) or row["atr"] <= 0:
            return False

        # Trendfilter
        trend_ok = row["close"] > row["ema200"]

        # Breakout
        breakout = row["close"] > row["donchian_high"]

        return bool(trend_ok and breakout)

    # --------------------------------------------------------
    # STOP LOSS
    # --------------------------------------------------------

    def calculate_initial_stop(self, entry_price, atr):

        return (
            entry_price
            - atr * self.atr_stop_multiplier
        )

    # --------------------------------------------------------
    # TRAILING STOP
    # --------------------------------------------------------

    def calculate_trailing_stop(
        self,
        highest_price,
        atr
    ):

        return (
            highest_price
            - atr * self.atr_trailing_multiplier
        )

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    def exit_signal(
        self,
        current_price,
        stop_price
    ):

        return current_price <= stop_price
