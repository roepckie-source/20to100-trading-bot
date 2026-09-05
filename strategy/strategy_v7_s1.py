import pandas as pd

from strategy.strategy_v6 import buy_signal_v6_c


# ============================================================
# V7-S1
# Signal Quality Filter
# ============================================================

S1_BREAKOUT_ATR_MIN = 0.10


def buy_signal_v7_s1(
    row: pd.Series,
    previous_row: pd.Series,
    adx_min: float = 20.0,
) -> bool:
    """
    V7-S1 basiert vollständig auf V6-C.

    Einziger zusätzlicher Filter:
    Der Breakout muss mindestens 0.10 ATR
    über dem vorherigen 20-Candle-Donchian-High liegen.

    Keine Änderung an:
    - Risk Management
    - Stop Loss
    - Trailing Stop
    - Fees
    - Slippage
    - Survival Layer
    """

    # --------------------------------------------------------
    # 1. V6-C muss zunächst ein gültiges BUY-Signal liefern
    # --------------------------------------------------------

    if not buy_signal_v6_c(
        row,
        previous_row,
        adx_min=adx_min,
    ):
        return False

    # --------------------------------------------------------
    # 2. Benötigte Werte prüfen
    # --------------------------------------------------------

    required = [
        "close",
        "donchian_high_20",
        "atr_14",
    ]

    for column in required:
        if column not in row:
            return False

        if pd.isna(row[column]):
            return False

    # --------------------------------------------------------
    # 3. Werte auslesen
    # --------------------------------------------------------

    close = float(row["close"])
    donchian_high = float(row["donchian_high_20"])
    atr = float(row["atr_14"])

    if atr <= 0:
        return False

    # --------------------------------------------------------
    # 4. Breakout-Stärke berechnen
    # --------------------------------------------------------

    breakout_distance = close - donchian_high

    minimum_breakout = (
        atr * S1_BREAKOUT_ATR_MIN
    )

    # --------------------------------------------------------
    # 5. S1 Filter
    # --------------------------------------------------------

    return bool(
        breakout_distance >= minimum_breakout
    )