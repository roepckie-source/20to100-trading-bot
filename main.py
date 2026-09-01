# ==========================================
# 20to100 Trading Bot
# Main Backtest Runner
# Strategy v3.0
# Entry Quality Analyzer
# ==========================================

from pathlib import Path

import pandas as pd

from config import (
    SYMBOLS,
    TIMEFRAME,
    STARTING_CAPITAL,
    HISTORICAL_DAYS,
)

from data.data_manager import (
    fetch_ohlcv,
    save_data,
)

from strategy.indicators import (
    calculate_indicators,
)

from strategy.signals import (
    diagnose_signals,
)

from backtest.engine import (
    BacktestEngine,
)


# ==========================================
# ENTRY COMBINATIONS
# ==========================================

ENTRY_TESTS = {
    "TREND": [
        "trend",
    ],

    "TREND_MOMENTUM": [
        "trend",
        "rsi",
        "rsi_rising",
    ],

    "TREND_PULLBACK": [
        "trend",
        "pullback",
    ],

    "TREND_CONFIRMATION": [
        "trend",
        "confirmation",
    ],

    "TREND_VOLUME": [
        "trend",
        "volume",
    ],

    "TREND_MOMENTUM_PULLBACK": [
        "trend",
        "rsi",
        "rsi_rising",
        "pullback",
    ],

    "TREND_MOMENTUM_CONFIRMATION": [
        "trend",
        "rsi",
        "rsi_rising",
        "confirmation",
    ],

    "TREND_PULLBACK_CONFIRMATION": [
        "trend",
        "pullback",
        "confirmation",
    ],

    "TREND_MOMENTUM_PULLBACK_CONFIRMATION": [
        "trend",
        "rsi",
        "rsi_rising",
        "pullback",
        "confirmation",
    ],

    "ALL_CONDITIONS": [
        "trend",
        "price_above_ema50",
        "rsi",
        "rsi_rising",
        "pullback",
        "confirmation",
        "volume",
    ],
}


# ==========================================
# CONDITION BUILDER
# ==========================================

def build_condition_mask(
    df,
    conditions,
):
    """
    Erstellt eine boolesche Maske für eine
    bestimmte Entry-Kombination.

    Die vorhandenen Spalten werden flexibel
    erkannt, damit der Analyzer nicht an
    kleinen Namensunterschieden scheitert.
    """

    masks = []

    # --------------------------------------
    # TREND
    # --------------------------------------

    if "trend" in conditions:

        if "trend" in df.columns:

            masks.append(
                df["trend"].fillna(False)
                .astype(bool)
            )

        elif (
            "ema_20" in df.columns
            and
            "ema_50" in df.columns
        ):

            masks.append(
                df["ema_20"]
                >
                df["ema_50"]
            )

        else:

            return None

    # --------------------------------------
    # PRICE > EMA50
    # --------------------------------------

    if "price_above_ema50" in conditions:

        if (
            "close" not in df.columns
            or
            "ema_50" not in df.columns
        ):

            return None

        masks.append(
            df["close"]
            >
            df["ema_50"]
        )

    # --------------------------------------
    # RSI
    # --------------------------------------

    if "rsi" in conditions:

        if "rsi" not in df.columns:

            return None

        masks.append(
            df["rsi"] >= 50
        )

    # --------------------------------------
    # RSI RISING
    # --------------------------------------

    if "rsi_rising" in conditions:

        if "rsi" not in df.columns:

            return None

        masks.append(
            df["rsi"]
            >
            df["rsi"].shift(1)
        )

    # --------------------------------------
    # PULLBACK
    # --------------------------------------

    if "pullback" in conditions:

        if "pullback" in df.columns:

            masks.append(
                df["pullback"]
                .fillna(False)
                .astype(bool)
            )

        elif (
            "close" in df.columns
            and
            "ema_20" in df.columns
        ):

            distance = (
                (
                    df["close"] -
                    df["ema_20"]
                ).abs()
                /
                df["ema_20"]
            )

            masks.append(
                distance <= 0.01
            )

        else:

            return None

    # --------------------------------------
    # CONFIRMATION
    # --------------------------------------

    if "confirmation" in conditions:

        if "confirmation" in df.columns:

            masks.append(
                df["confirmation"]
                .fillna(False)
                .astype(bool)
            )

        else:

            masks.append(
                df["close"]
                >
                df["close"].shift(1)
            )

    # --------------------------------------
    # VOLUME
    # --------------------------------------

    if "volume" in conditions:

        if "volume" not in df.columns:

            return None

        volume_ma = (
            df["volume"]
            .rolling(20)
            .mean()
        )

        masks.append(
            df["volume"]
            >
            volume_ma
        )

    # --------------------------------------
    # Combine
    # --------------------------------------

    if not masks:

        return pd.Series(
            True,
            index=df.index,
        )

    result = masks[0].fillna(False)

    for mask in masks[1:]:

        result = (
            result
            &
            mask.fillna(False)
        )

    return result


# ==========================================
# FORWARD RETURN ANALYSIS
# ==========================================

def calculate_forward_returns(
    df,
    signal_mask,
):
    """
    Misst nicht nur, ob ein Signal entsteht,
    sondern was danach mit dem Markt passiert.

    Dadurch bekommen wir einen ersten Hinweis,
    ob die Entry-Bedingung tatsächlich einen
    positiven Edge besitzt.
    """

    close = df["close"]

    results = {}

    for candles in [
        3,
        6,
        12,
        24,
    ]:

        future_price = (
            close.shift(-candles)
        )

        forward_return = (
            future_price /
            close -
            1
        ) * 100

        values = (
            forward_return[
                signal_mask
            ]
            .dropna()
        )

        if len(values) == 0:

            results[candles] = {
                "count": 0,
                "average": 0.0,
                "median": 0.0,
                "positive_pct": 0.0,
            }

            continue

        results[candles] = {
            "count":
                len(values),

            "average":
                float(values.mean()),

            "median":
                float(values.median()),

            "positive_pct":
                float(
                    (
                        values > 0
                    ).mean()
                    * 100
                ),
        }

    return results


# ==========================================
# ENTRY QUALITY TEST
# ==========================================

def run_entry_quality_test(
    df,
    symbol,
):
    """
    Vergleicht die Qualität verschiedener
    Entry-Bedingungskombinationen.
    """

    print()
    print("=" * 100)
    print(
        f"ENTRY QUALITY TEST: {symbol}"
    )
    print("=" * 100)

    print()
    print(
        "Forward returns are measured after:"
    )

    print(
        "3, 6, 12 and 24 candles"
    )

    print()

    print(
        f"{'Setup':<42}"
        f"{'Signals':>10}"
        f"{'3c avg':>11}"
        f"{'6c avg':>11}"
        f"{'12c avg':>11}"
        f"{'24c avg':>11}"
    )

    print("-" * 100)

    results = []

    for name, conditions in ENTRY_TESTS.items():

        mask = build_condition_mask(
            df,
            conditions,
        )

        if mask is None:

            print(
                f"{name:<42}"
                f"{'N/A':>10}"
            )

            continue

        valid = (
            mask
            &
            df["close"].notna()
        )

        forward = (
            calculate_forward_returns(
                df,
                valid,
            )
        )

        signal_count = int(
            valid.sum()
        )

        avg_3 = (
            forward[3]["average"]
        )

        avg_6 = (
            forward[6]["average"]
        )

        avg_12 = (
            forward[12]["average"]
        )

        avg_24 = (
            forward[24]["average"]
        )

        print(
            f"{name:<42}"
            f"{signal_count:>10}"
            f"{avg_3:>10.3f}%"
            f"{avg_6:>10.3f}%"
            f"{avg_12:>10.3f}%"
            f"{avg_24:>10.3f}%"
        )

        results.append(
            {
                "name":
                    name,

                "signals":
                    signal_count,

                "avg_3":
                    avg_3,

                "avg_6":
                    avg_6,

                "avg_12":
                    avg_12,

                "avg_24":
                    avg_24,

                "positive_3":
                    forward[3][
                        "positive_pct"
                    ],

                "positive_6":
                    forward[6][
                        "positive_pct"
                    ],

                "positive_12":
                    forward[12][
                        "positive_pct"
                    ],

                "positive_24":
                    forward[24][
                        "positive_pct"
                    ],
            }
        )

    print("-" * 100)

    if results:

        best = max(
            results,
            key=lambda x:
                x["avg_12"],
        )

        print()
        print(
            "BEST ENTRY BY 12-CANDLE "
            "FORWARD RETURN"
        )

        print(
            f"Setup:             "
            f"{best['name']}"
        )

        print(
            f"Signals:           "
            f"{best['signals']}"
        )

        print(
            f"12-candle average: "
            f"{best['avg_12']:+.3f}%"
        )

        print(
            f"12-candle positive: "
            f"{best['positive_12']:.2f}%"
        )

    print("=" * 100)

    return results


# ==========================================
# MAIN
# ==========================================

def main():

    Path("logs").mkdir(
        exist_ok=True
    )

    Path("data").mkdir(
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("20→100 TRADING BOT")
    print("Strategy v3.0")
    print("ENTRY QUALITY ANALYZER")
    print("=" * 60)

    all_results = []

    for symbol in SYMBOLS:

        print()
        print("-" * 60)
        print(
            f"ANALYSIS: {symbol}"
        )
        print("-" * 60)

        filename = (
            symbol.replace(
                "/",
                "_",
            )
            + "_"
            + TIMEFRAME
            + ".csv"
        )

        data_file = (
            Path("data") /
            filename
        )

        # ==================================
        # DATA
        # ==================================

        if data_file.exists():

            print(
                f"Using cached data: "
                f"{data_file}"
            )

            df = pd.read_csv(
                data_file,
                index_col="timestamp",
                parse_dates=True,
            )

        else:

            print(
                "No local data found."
            )

            print(
                "Downloading historical data..."
            )

            df = fetch_ohlcv(
                symbol=symbol,
                timeframe=TIMEFRAME,
                days=HISTORICAL_DAYS,
            )

            save_data(
                df,
                symbol=symbol,
                timeframe=TIMEFRAME,
            )

        print(
            f"Candles: {len(df)}"
        )

        # ==================================
        # INDICATORS
        # ==================================

        print()
        print(
            "Calculating indicators..."
        )

        indicator_df = (
            calculate_indicators(
                df
            )
        )

        # ==================================
        # SIGNAL DIAGNOSTICS
        # ==================================

        print()
        print(
            "Running signal diagnostics..."
        )

        try:

            diagnostic_data = (
                diagnose_signals(
                    indicator_df
                )
            )

            print(
                f"Valid candles: "
                f"{diagnostic_data.get(
                    'valid_candles',
                    'N/A'
                )}"
            )

        except Exception as exc:

            print(
                "Signal diagnostics skipped:"
            )

            print(
                str(exc)
            )

        # ==================================
        # ENTRY QUALITY
        # ==================================

        result = (
            run_entry_quality_test(
                indicator_df,
                symbol,
            )
        )

        all_results.append(
            {
                "symbol":
                    symbol,

                "results":
                    result,
            }
        )

    # ======================================
    # FINAL SUMMARY
    # ======================================

    print()
    print("=" * 100)
    print(
        "20→100 ENTRY QUALITY SUMMARY"
    )
    print("=" * 100)

    for market in all_results:

        symbol = market[
            "symbol"
        ]

        results = market[
            "results"
        ]

        if not results:
            continue

        best = max(
            results,
            key=lambda x:
                x["avg_12"],
        )

        print()
        print(
            f"{symbol}"
        )

        print(
            f"Best setup:        "
            f"{best['name']}"
        )

        print(
            f"Signals:           "
            f"{best['signals']}"
        )

        print(
            f"3 candles:         "
            f"{best['avg_3']:+.3f}%"
        )

        print(
            f"6 candles:         "
            f"{best['avg_6']:+.3f}%"
        )

        print(
            f"12 candles:        "
            f"{best['avg_12']:+.3f}%"
        )

        print(
            f"24 candles:        "
            f"{best['avg_24']:+.3f}%"
        )

        print(
            f"Positive 12c:      "
            f"{best['positive_12']:.2f}%"
        )

    print()
    print("=" * 100)
    print(
        "ENTRY QUALITY TEST COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
