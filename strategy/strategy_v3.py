# ==========================================
# 20to100 Trading Bot
# Strategy V3
# Robust Trend Following
# ==========================================

import pandas as pd


class StrategyV3:

    def __init__(
        self,
        ema_fast=100,
        ema_slow=200,
        donchian_period=50,
        atr_period=14,
        atr_stop_multiplier=3.0,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.donchian_period = donchian_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier

    # ======================================
    # INDICATORS
    # ======================================

    def calculate_indicators(self, df):

        data = df.copy()

        # -------------------------------
        # EMA
        # -------------------------------

        data["ema_fast"] = (
            data["close"]
            .ewm(
                span=self.ema_fast,
                adjust=False,
            )
            .mean()
        )

        data["ema_slow"] = (
            data["close"]
            .ewm(
                span=self.ema_slow,
                adjust=False,
            )
            .mean()
        )

        # -------------------------------
        # EMA slope
        # -------------------------------

        data["ema_slow_slope"] = (
            data["ema_slow"]
            .diff(10)
        )

        # -------------------------------
        # Donchian
        # -------------------------------

        data["donchian_high"] = (
            data["high"]
            .rolling(
                self.donchian_period
            )
            .max()
            .shift(1)
        )

        data["donchian_low"] = (
            data["low"]
            .rolling(
                self.donchian_period
            )
            .min()
            .shift(1)
        )

        # -------------------------------
        # ATR
        # -------------------------------

        previous_close = (
            data["close"].shift(1)
        )

        tr1 = (
            data["high"] -
            data["low"]
        )

        tr2 = (
            data["high"] -
            previous_close
        ).abs()

        tr3 = (
            data["low"] -
            previous_close
        ).abs()

        true_range = pd.concat(
            [tr1, tr2, tr3],
            axis=1,
        ).max(axis=1)

        data["atr"] = (
            true_range
            .rolling(
                self.atr_period
            )
            .mean()
        )

        return data

    # ======================================
    # ENTRY
    # ======================================

    def check_entry(self, row):

        required = [
            "close",
            "ema_fast",
            "ema_slow",
            "ema_slow_slope",
            "donchian_high",
            "atr",
        ]

        for column in required:
            if pd.isna(row[column]):
                return False

        # -------------------------------
        # Trend
        # -------------------------------

        trend = (
            row["ema_fast"]
            > row["ema_slow"]
        )

        # -------------------------------
        # Strong trend
        # -------------------------------

        slope_positive = (
            row["ema_slow_slope"]
            > 0
        )

        # -------------------------------
        # Price above slow EMA
        # -------------------------------

        price_above_trend = (
            row["close"]
            > row["ema_slow"]
        )

        # -------------------------------
        # Donchian breakout
        # -------------------------------

        breakout = (
            row["close"]
            > row["donchian_high"]
        )

        return (
            trend
            and slope_positive
            and price_above_trend
            and breakout
        )

    # ======================================
    # INITIAL STOP
    # ======================================

    def calculate_stop(
        self,
        entry_price,
        atr,
    ):

        return (
            entry_price
            -
            atr
            * self.atr_stop_multiplier
        )

    # ======================================
    # TRAILING STOP
    # ======================================

    def calculate_trailing_stop(
        self,
        highest_price,
        atr,
    ):

        return (
            highest_price
            -
            atr
            * self.atr_stop_multiplier
        )
