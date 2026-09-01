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
    evaluate_conditions,
    diagnose_signals,
    print_signal_diagnostics,
)


# ==========================================
# ENTRY TESTS
# ==========================================

ENTRY_TESTS = {

    "TREND": [
        "trend",
    ],

    "TREND + PRICE EMA50": [
        "trend",
        "price_above_ema50",
    ],

    "TREND + RSI": [
        "trend",
        "rsi",
    ],

    "TREND + RSI RISING": [
        "trend",
        "rsi_rising",
    ],

    "TREND + PULLBACK": [
        "trend",
        "pullback",
    ],

    "TREND + CONFIRMATION": [
        "trend",
        "confirmation",
    ],

    "TREND + VOLUME": [
        "trend",
        "volume",
    ],

    "TREND + RSI + PULLBACK": [
        "trend",
        "rsi",
        "pullback",
    ],

    "TREND + RSI + CONFIRMATION": [
        "trend",
        "rsi",
        "confirmation",
    ],

    "TREND + RSI + PULLBACK + CONFIRMATION": [
        "trend",
        "rsi",
        "pullback",
        "confirmation",
    ],

    "TREND + RSI + PULLBACK + VOLUME": [
        "trend",
        "rsi",
        "pullback",
        "volume",
    ],

    "TREND + RSI + PULLBACK + CONFIRMATION + VOLUME": [
        "trend",
        "rsi",
        "pullback",
        "confirmation",
        "volume",
    ],

    "ALL CONDITIONS": [
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
# CONDITION MASK
# ==========================================

def build_condition_masks(df):
    """
    Berechnet jede Entry-Bedingung exakt mit
    der bereits vorhandenen Strategy-Logik.

    Wir verwenden NICHT selbst erfundene
    Bedingungen.
    """

    masks = {
        name: []
        for name in [
            "trend",
            "price_above_ema50",
            "rsi",
            "rsi_rising",
            "pullback",
            "confirmation",
            "volume",
        ]
    }

    valid_rows = []

    for i in range(
        60,
        len(df),
    ):

        history = df.iloc[
            : i + 1
        ]

        row = history.iloc[-1]

        try:

            conditions = evaluate_conditions(
                history,
                spread=None,
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            valid_rows.append(False)

            for name in masks:
                masks[name].append(False)

            continue

        valid = all(
            pd.notna(
                row[column]
            )
            for column in [
                "open",
                "close",
                "low",
                "ema_9",
                "ema_21",
                "ema_50",
                "rsi_14",
                "volume",
                "volume_sma_20",
            ]
        )

        valid_rows.append(valid)

        for name in masks:

            masks[name].append(
                bool(
                    conditions.get(
                        name,
                        False,
                    )
                )
                if valid
                else False
            )

    index = df.index[
        60:
    ]

    for name in masks:

        masks[name] = pd.Series(
            masks[name],
            index=index,
            dtype=bool,
        )

    valid_series = pd.Series(
        valid_rows,
        index=index,
        dtype=bool,
    )

    return (
        masks,
        valid_series,
    )


# ==========================================
# COMBINE CONDITIONS
# ==========================================

def combined_mask(
    masks,
    conditions,
):

    if not conditions:

        return pd.Series(
            False,
            index=next(
                iter(masks.values())
            ).index,
        )

    result = masks[
        conditions[0]
    ].copy()

    for condition in conditions[1:]:

        result = (
            result
            &
            masks[condition]
        )

    return result


# ==========================================
# FORWARD RETURN
# ==========================================

def forward_return_analysis(
    df,
    signal_mask,
):
    """
    Berechnet die durchschnittliche
    Kursentwicklung nach dem Signal.

    3 Kerzen  = 15 Minuten
    6 Kerzen  = 30 Minuten
    12 Kerzen = 60 Minuten
    24 Kerzen = 120 Minuten
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
            close.shift(
                -candles
            )
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

        if values.empty:

            results[candles] = {
                "count": 0,
                "average": 0.0,
                "median": 0.0,
                "positive_pct": 0.0,
            }

        else:

            results[candles] = {

                "count":
                    len(values),

                "average":
                    float(
                        values.mean()
                    ),

                "median":
                    float(
                        values.median()
                    ),

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

    print()
    print("=" * 110)
    print(
        f"ENTRY QUALITY TEST: {symbol}"
    )
    print("=" * 110)

    print()
    print(
        "Forward return:"
    )

    print(
        "3 candles  = 15 min"
    )

    print(
        "6 candles  = 30 min"
    )

    print(
        "12 candles = 60 min"
    )

    print(
        "24 candles = 120 min"
    )

    print()

    (
        masks,
        valid_series,
    ) = build_condition_masks(
        df
    )

    print(
        f"{'Setup':<55}"
        f"{'Signals':>9}"
        f"{'15m':>11}"
        f"{'30m':>11}"
        f"{'60m':>11}"
        f"{'120m':>11}"
    )

    print("-" * 110)

    results = []

    for name, conditions in ENTRY_TESTS.items():

        mask = combined_mask(
            masks,
            conditions,
        )

        # Nur gültige Kerzen
        mask = (
            mask
            &
            valid_series
        )

        analysis = (
            forward_return_analysis(
                df,
                mask,
            )
        )

        signals = int(
            mask.sum()
        )

        avg_3 = (
            analysis[3]["average"]
        )

        avg_6 = (
            analysis[6]["average"]
        )

        avg_12 = (
            analysis[12]["average"]
        )

        avg_24 = (
            analysis[24]["average"]
        )

        print(
            f"{name:<55}"
            f"{signals:>9}"
            f"{avg_3:>10.3f}%"
            f"{avg_6:>10.3f}%"
            f"{avg_12:>10.3f}%"
            f"{avg_24:>10.3f}%"
        )

        results.append(
            {
                "name":
                    name,

                "conditions":
                    conditions,

                "signals":
                    signals,

                "avg_3":
                    avg_3,

                "avg_6":
                    avg_6,

                "avg_12":
                    avg_12,

                "avg_24":
                    avg_24,

                "positive_3":
                    analysis[3][
                        "positive_pct"
                    ],

                "positive_6":
                    analysis[6][
                        "positive_pct"
                    ],

                "positive_12":
                    analysis[12][
                        "positive_pct"
                    ],

                "positive_24":
                    analysis[24][
                        "positive_pct"
                    ],
            }
        )

    print("-" * 110)

    # ======================================
    # BEST SETUPS
    # ======================================

    if not results:

        print(
            "No valid results."
        )

        print("=" * 110)

        return []

    # --------------------------------------
    # Best 15 min
    # --------------------------------------

    best_3 = max(
        results,
        key=lambda x:
            x["avg_3"],
    )

    # --------------------------------------
    # Best 30 min
    # --------------------------------------

    best_6 = max(
        results,
        key=lambda x:
            x["avg_6"],
    )

    # --------------------------------------
    # Best 60 min
    # --------------------------------------

    best_12 = max(
        results,
        key=lambda x:
            x["avg_12"],
    )

    # --------------------------------------
    # Best 120 min
    # --------------------------------------

    best_24 = max(
        results,
        key=lambda x:
            x["avg_24"],
    )

    print()
    print(
        "BEST SETUPS"
    )

    print("-" * 110)

    print(
        f"15 min : "
        f"{best_3['name']} "
        f"({best_3['avg_3']:+.3f}%)"
    )

    print(
        f"30 min : "
        f"{best_6['name']} "
        f"({best_6['avg_6']:+.3f}%)"
    )

    print(
        f"60 min : "
        f"{best_12['name']} "
        f"({best_12['avg_12']:+.3f}%)"
    )

    print(
        f"120 min: "
        f"{best_24['name']} "
        f"({best_24['avg_24']:+.3f}%)"
    )

    print()
    print(
        "BEST SETUP BY 60-MINUTE RETURN:"
    )

    print(
        f"  {best_12['name']}"
    )

    print(
        f"  Signals: "
        f"{best_12['signals']}"
    )

    print(
        f"  Average: "
        f"{best_12['avg_12']:+.3f}%"
    )

    print(
        f"  Positive: "
        f"{best_12['positive_12']:.2f}%"
    )

    print("=" * 110)

    return results


# ==========================================
# SAVE ENTRY RESULTS
# ==========================================

def save_entry_results(
    results,
    symbol,
):

    if not results:
        return

    output = []

    for result in results:

        output.append(
            {
                "symbol":
                    symbol,

                "setup":
                    result["name"],

                "signals":
                    result["signals"],

                "avg_15m_pct":
                    result["avg_3"],

                "avg_30m_pct":
                    result["avg_6"],

                "avg_60m_pct":
                    result["avg_12"],

                "avg_120m_pct":
                    result["avg_24"],

                "positive_15m_pct":
                    result["positive_3"],

                "positive_30m_pct":
                    result["positive_6"],

                "positive_60m_pct":
                    result["positive_12"],

                "positive_120m_pct":
                    result["positive_24"],
            }
        )

    output_df = pd.DataFrame(
        output
    )

    file = (
        Path("logs")
        /
        (
            symbol.replace(
                "/",
                "_",
            )
            +
            "_entry_quality.csv"
        )
    )

    output_df.to_csv(
        file,
        index=False,
    )

    print(
        f"Entry quality log: {file}"
    )


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
    print("=" * 70)
    print(
        "20→100 TRADING BOT"
    )
    print(
        "Strategy v3.0"
    )
    print(
        "ENTRY QUALITY ANALYZER"
    )
    print("=" * 70)

    all_results = []

    # ======================================
    # SYMBOLS
    # ======================================

    for symbol in SYMBOLS:

        print()
        print("-" * 70)
        print(
            f"ANALYSIS: {symbol}"
        )
        print("-" * 70)

        filename = (
            symbol.replace(
                "/",
                "_",
            )
            +
            "_"
            +
            TIMEFRAME
            +
            ".csv"
        )

        data_file = (
            Path("data")
            /
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
        # EXISTING DIAGNOSTICS
        # ==================================

        print()
        print(
            "Running existing signal diagnostics..."
        )

        try:

            diagnostics = (
                diagnose_signals(
                    indicator_df
                )
            )

            print_signal_diagnostics(
                diagnostics,
                symbol,
            )

        except Exception as exc:

            print(
                "Diagnostics error:"
            )

            print(
                str(exc)
            )

        # ==================================
        # ENTRY QUALITY
        # ==================================

        results = (
            run_entry_quality_test(
                indicator_df,
                symbol,
            )
        )

        save_entry_results(
            results,
            symbol,
        )

        all_results.append(
            {
                "symbol":
                    symbol,

                "results":
                    results,
            }
        )

    # ======================================
    # FINAL SUMMARY
    # ======================================

    print()
    print("=" * 110)
    print(
        "20→100 ENTRY QUALITY SUMMARY"
    )
    print("=" * 110)

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
            f"15 min:            "
            f"{best['avg_3']:+.3f}%"
        )

        print(
            f"30 min:            "
            f"{best['avg_6']:+.3f}%"
        )

        print(
            f"60 min:            "
            f"{best['avg_12']:+.3f}%"
        )

        print(
            f"120 min:           "
            f"{best['avg_24']:+.3f}%"
        )

        print(
            f"Positive 60 min:   "
            f"{best['positive_12']:.2f}%"
        )

    print()
    print("=" * 110)
    print(
        "ENTRY QUALITY TEST COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
