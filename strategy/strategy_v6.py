# ==========================================
# 20to100 Trading Bot
# Strategy V6.0
# Adaptive Regime Filter
#
# V6 basiert vollständig auf V5_C:
# - EMA 100 / EMA 200
# - positive EMA200-Slope
# - Close > EMA200
# - Donchian 20 Breakout
# - ADX >= 20
# - ATR > 0
#
# V6 ergänzt:
# A: ADX muss steigen
# B: zusätzlich stärkerer EMA200-Slope
# C: zusätzlich ATR über ATR-MA50
#
# Wichtig:
# V5 bleibt unverändert.
# V6 verwendet ausschließlich aktuelle und vergangene
# Kerzen -> kein Look-ahead.
# ==========================================

import numpy as np
import pandas as pd


# ==========================================
# INDICATORS
# ==========================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all V6 indicators on the FULL dataset.

    Alle Indikatoren sind kausal und verwenden
    ausschließlich aktuelle bzw. vergangene Daten.
    """

    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]

    # ==========================================
    # EMA 100
    # ==========================================

    out["ema_100"] = close.ewm(
        span=100,
        adjust=False
    ).mean()

    # ==========================================
    # EMA 200
    # ==========================================

    out["ema_200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # ==========================================
    # EMA 200 SLOPE
    # ==========================================

    out["ema_200_slope"] = (
        out["ema_200"].pct_change(10)
    )

    # ==========================================
    # ADAPTIVE EMA SLOPE REFERENCE
    #
    # 50 Stunden Referenz.
    #
    # Wichtig:
    # shift(1) verhindert, dass die aktuelle
    # Slope in ihre eigene Referenz eingeht.
    # ==========================================

    out["ema_200_slope_reference"] = (
        out["ema_200_slope"]
        .rolling(50)
        .median()
        .shift(1)
    )

    # ==========================================
    # DONCHIAN 20
    #
    # Vorheriges 20-Candle-Hoch
    # ==========================================

    out["donchian_high_20"] = (
        high
        .rolling(20)
        .max()
        .shift(1)
    )

    # ==========================================
    # TRUE RANGE
    # ==========================================

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    # ==========================================
    # ATR 14
    # ==========================================

    out["atr_14"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # ==========================================
    # ATR 50 REFERENCE
    #
    # Aktueller ATR wird NICHT verwendet,
    # um seine eigene Referenz zu bestimmen.
    # ==========================================

    out["atr_14_ma50"] = (
        out["atr_14"]
        .rolling(50)
        .mean()
        .shift(1)
    )

    # ==========================================
    # ADX 14
    # ==========================================

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            &
            (up_move > 0),
            up_move,
            0.0
        ),
        index=out.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            &
            (down_move > 0),
            down_move,
            0.0
        ),
        index=out.index
    )

    atr = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    plus_di = (
        100
        *
        plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        /
        atr
    )

    minus_di = (
        100
        *
        minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        /
        atr
    )

    di_sum = (
        plus_di + minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100
        *
        (plus_di - minus_di).abs()
        /
        di_sum
    )

    out["adx_14"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # ==========================================
    # BULLISH TREND REGIME
    # ==========================================

    out["trend_regime"] = (
        (out["ema_100"] > out["ema_200"])
        &
        (out["ema_200_slope"] > 0)
        &
        (out["close"] > out["ema_200"])
    )

    return out


# ==========================================
# COMMON V5 ENTRY
# ==========================================

def _base_entry(
    row: pd.Series,
    adx_min: float = 20.0
) -> bool:

    required = [
        "close",
        "ema_100",
        "ema_200",
        "ema_200_slope",
        "adx_14",
        "donchian_high_20",
        "atr_14",
    ]

    for column in required:

        if column not in row:
            return False

        if pd.isna(row[column]):
            return False

    # ==========================================
    # V5 / V6 BASE CONDITIONS
    # ==========================================

    bullish_trend = (
        row["ema_100"]
        >
        row["ema_200"]
    )

    positive_slope = (
        row["ema_200_slope"]
        >
        0
    )

    price_above_ema200 = (
        row["close"]
        >
        row["ema_200"]
    )

    strong_trend = (
        row["adx_14"]
        >=
        adx_min
    )

    breakout = (
        row["close"]
        >
        row["donchian_high_20"]
    )

    valid_atr = (
        row["atr_14"]
        >
        0
    )

    return bool(
        bullish_trend
        and positive_slope
        and price_above_ema200
        and strong_trend
        and breakout
        and valid_atr
    )


# ==========================================
# V6 A
# ADX RISING
# ==========================================

def buy_signal_v6_a(
    row: pd.Series,
    previous_row: pd.Series,
    adx_min: float = 20.0
) -> bool:
    """
    V6_A:

    V5_C
    +
    ADX muss steigen.
    """

    if not _base_entry(
        row,
        adx_min=adx_min
    ):
        return False

    if "adx_14" not in previous_row:
        return False

    if pd.isna(previous_row["adx_14"]):
        return False

    adx_rising = (
        row["adx_14"]
        >
        previous_row["adx_14"]
    )

    return bool(
        adx_rising
    )


# ==========================================
# V6 B
# ADX RISING + STRONGER SLOPE
# ==========================================

def buy_signal_v6_b(
    row: pd.Series,
    previous_row: pd.Series,
    adx_min: float = 20.0
) -> bool:
    """
    V6_B:

    V5_C
    +
    ADX steigt
    +
    aktuelle EMA200-Slope liegt über
    ihrer adaptiven 50-Candle-Referenz.
    """

    if not _base_entry(
        row,
        adx_min=adx_min
    ):
        return False

    required = [
        "ema_200_slope",
        "ema_200_slope_reference",
    ]

    for column in required:

        if column not in row:
            return False

        if pd.isna(row[column]):
            return False

    if "adx_14" not in previous_row:
        return False

    if pd.isna(previous_row["adx_14"]):
        return False

    adx_rising = (
        row["adx_14"]
        >
        previous_row["adx_14"]
    )

    strong_slope = (
        row["ema_200_slope"]
        >
        row["ema_200_slope_reference"]
    )

    return bool(
        adx_rising
        and strong_slope
    )


# ==========================================
# V6 C
# ADX RISING
# + STRONGER SLOPE
# + VOLATILITY
# ==========================================

def buy_signal_v6_c(
    row: pd.Series,
    previous_row: pd.Series,
    adx_min: float = 20.0
) -> bool:
    """
    V6_C:

    V5_C
    +
    ADX steigt
    +
    stärkere EMA200-Slope
    +
    ATR liegt über ihrer 50-Candle-Referenz.
    """

    if not buy_signal_v6_b(
        row,
        previous_row,
        adx_min=adx_min
    ):
        return False

    required = [
        "atr_14",
        "atr_14_ma50",
    ]

    for column in required:

        if column not in row:
            return False

        if pd.isna(row[column]):
            return False

    volatility_expanding = (
        row["atr_14"]
        >
        row["atr_14_ma50"]
    )

    return bool(
        volatility_expanding
    )


# ==========================================
# GENERIC BUY SIGNAL
# ==========================================

def buy_signal(
    row: pd.Series,
    previous_row: pd.Series = None,
    variant: str = "V6_A",
    adx_min: float = 20.0
) -> bool:
    """
    Generic V6 entry function.

    Varianten:

    V6_A
        ADX steigt

    V6_B
        ADX steigt
        + stärkerer EMA200-Slope

    V6_C
        ADX steigt
        + stärkerer EMA200-Slope
        + ATR über ATR-MA50
    """

    if previous_row is None:
        return False

    variant = str(
        variant
    ).upper()

    if variant == "V6_A":

        return buy_signal_v6_a(
            row,
            previous_row,
            adx_min=adx_min
        )

    if variant == "V6_B":

        return buy_signal_v6_b(
            row,
            previous_row,
            adx_min=adx_min
        )

    if variant == "V6_C":

        return buy_signal_v6_c(
            row,
            previous_row,
            adx_min=adx_min
        )

    raise ValueError(
        f"Unbekannte V6 Variante: {variant}"
    )


# ==========================================
# BACKWARD-COMPATIBILITY HELPER
# ==========================================

def buy_signal_v5_compatible(
    row: pd.Series,
    adx_min: float = 20.0
) -> bool:
    """
    V5-kompatibler Signal-Helper.

    Wird nicht für die eigentliche V6-
    Bewertung benötigt, erleichtert aber
    Tests und Vergleiche.
    """

    return _base_entry(
        row,
        adx_min=adx_min
    )
