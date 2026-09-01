# ==========================================
# 20to100 Trading Bot
# Main Backtest Runner
# Strategy v3.1
# Targeted Entry Combination Analyzer
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
# TEST SETUPS
# ==========================================

ENTRY_TESTS = {

    # --------------------------------------
    # Baseline
    # --------------------------------------

    "TREND": [
        "trend",
    ],

    "TREND + VOLUME": [
        "trend",
        "volume",
    ],

    "TREND + PULLBACK": [
        "trend",
        "pullback",
    ],

    "TREND + CONFIRMATION": [
        "trend",
        "confirmation",
    ],

    # --------------------------------------
    # Core combinations
    # --------------------------------------

    "TREND + PULLBACK + CONFIRMATION": [
        "trend",
        "pullback",
        "confirmation",
    ],

    "TREND + PULLBACK + VOLUME": [
        "trend",
        "pullback",
        "volume",
    ],

    "TREND + CONFIRMATION + VOLUME": [
        "trend",
        "confirmation",
        "volume",
    ],

    "TREND + PULLBACK + CONFIRMATION + VOLUME": [
        "trend",
        "pullback",
        "confirmation",
        "volume",
    ],

    # --------------------------------------
    # RSI variants
    # --------------------------------------

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

    "TREND + RSI + VOLUME": [
        "trend",
        "rsi",
        "volume",
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

    "TREND + RSI + CONFIRMATION + VOLUME": [
        "trend",
        "rsi",
        "confirmation",
        "volume",
    ],

    "TREND + RSI + PULLBACK + CONFIRMATION + VOLUME": [
        "trend",
        "rsi",
        "pullback",
        "confirmation",
        "volume",
    ],

    # --------------------------------------
    # RSI rising variants
    # --------------------------------------

    "TREND + RSI RISING + VOLUME": [
        "trend",
        "rsi_rising",
        "volume",
    ],

    "TREND + RSI + RSI RISING + VOLUME": [
        "trend",
        "rsi",
        "rsi_rising",
        "volume",
    ],

    "TREND + RSI + RSI RISING + PULLBACK + VOLUME": [
        "trend",
        "rsi",
        "rsi_rising",
        "pullback",
        "volume",
    ],

    "TREND + RSI + RSI RISING + PULLBACK + CONFIRMATION": [
        "trend",
        "rsi",
        "rsi_rising",
        "pullback",
        "confirmation",
    ],

    "TREND + RSI + RSI RISING + PULLBACK + CONFIRMATION + VOLUME": [
        "trend",
        "rsi",
        "rsi_rising",
        "pullback",
        "confirmation",
        "volume",
    ],

    # --------------------------------------
    # EMA50 variants
    # --------------------------------------

    "TREND + EMA50 + VOLUME": [
        "trend",
        "price_above_ema50",
        "volume",
    ],

    "TREND + EMA50 + PULLBACK + CONFIRMATION": [
        "trend",
        "price_above_ema50",
        "pullback",
        "confirmation",
    ],

    "TREND + EMA50 + PULLBACK + CONFIRMATION + VOLUME": [
        "trend",
        "price_above_ema50",
        "pullback",
        "confirmation",
        "volume",
    ],

    # --------------------------------------
    # Original strategy
    # --------------------------------------

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
# BUILD CONDITION MASKS
# ==========================================

def build_condition_masks(df):

    condition_names = [
        "trend",
        "price_above_ema50",
        "rsi",
        "rsi_rising",
        "pullback",
        "confirmation",
        "volume",
    ]

    masks = {
        name: []
        for name in condition_names
    }

    valid_rows = []

    required_columns = [
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

    for i in range(60, len(df)):

        history = df.iloc[:i + 1]

        row = history.iloc[-1]

        valid = True

        for column in required_columns:

            if column not in row.index:
                valid = False
                break

            if pd.isna(row[column]):
                valid = False
                break

        if not valid:

            valid_rows.append(False)

            for name in condition_names:
                masks[name].append(False)

            continue

        try:

            conditions = evaluate_conditions(
                history,
                spread=None,
            )

        except Exception:

            valid_rows.append(False)

            for name in condition_names:
                masks[name].append(False)

            continue

        valid_rows.append(True)

        for name in condition_names:

            masks[name].append(
                bool(
                    conditions.get(
                        name,
                        False,
                    )
                )
            )

    index = df.index[60:]

    for name in condition_names:

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

    return masks, valid_series


# ==========================================
# COMBINE CONDITIONS
# ==========================================

def combined_mask(
    masks,
    conditions,
):

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
# FORWARD RETURN ANALYSIS
# ==========================================

def forward_return_analysis(
    df,
    signal_mask,
):

    close = df["close"]

    signal_mask = (
        signal_mask
        .reindex(
            df.index,
            fill_value=False,
        )
        .astype(bool)
    )

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
            (
                future_price /
                close
            )
            - 1
        ) * 100

        valid = (
            signal_mask
            &
            forward_return.notna()
            &
            close.notna()
        )

        values = (
            forward_return.loc[valid]
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
                    int(len(values)),

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
        "3 candles  = 15 minutes"
    )

    print(
        "6 candles  = 30 minutes"
    )

    print(
        "12 candles = 60 minutes"
    )

    print(
        "24 candles = 120 minutes"
    )

    print()

    masks, valid_series = (
        build_condition_masks(
            df
        )
    )

    print(
        f"{'Setup':<62}"
        f"{'Signals':>9}"
        f"{'15m':>10}"
        f"{'30m':>10}"
        f"{'60m':>10}"
        f"{'120m':>10}"
    )

    print("-" * 110)

    results = []

    for name, conditions in ENTRY_TESTS.items():

        mask = combined_mask(
            masks,
            conditions,
        )

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

        avg_3 = analysis[3]["average"]
        avg_6 = analysis[6]["average"]
        avg_12 = analysis[12]["average"]
        avg_24 = analysis[24]["average"]

        positive_12 = (
            analysis[12]["positive_pct"]
        )

        print(
            f"{name:<62}"
            f"{signals:>9}"
            f"{avg_3:>9.3f}%"
            f"{avg_6:>9.3f}%"
            f"{avg_12:>9.3f}%"
            f"{avg_24:>9.3f}%"
        )

        results.append(
            {
                "name": name,
                "conditions": conditions,
                "signals": signals,

                "avg_3": avg_3,
                "avg_6": avg_6,
                "avg_12": avg_12,
                "avg_24": avg_24,

                "positive_3":
                    analysis[3][
                        "positive_pct"
                    ],

                "positive_6":
                    analysis[6][
                        "positive_pct"
                    ],

                "positive_12":
                    positive_12,

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
        return []

    best_15m = max(
        results,
        key=lambda x:
            x["avg_3"],
    )

    best_30m = max(
        results,
        key=lambda x:
            x["avg_6"],
    )

    best_60m = max(
        results,
        key=lambda x:
            x["avg_12"],
    )

    best_120m = max(
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
        f"{best_15m['name']} "
        f"({best_15m['avg_3']:+.3f}%)"
    )

    print(
        f"30 min : "
        f"{best_30m['name']} "
        f"({best_30m['avg_6']:+.3f}%)"
    )

    print(
        f"60 min : "
        f"{best_60m['name']} "
        f"({best_60m['avg_12']:+.3f}%)"
    )

    print(
        f"120 min: "
        f"{best_120m['name']} "
        f"({best_120m['avg_24']:+.3f}%)"
    )

    # ======================================
    # RANKING
    # ======================================

    print()
    print(
        "TOP 5 BY 60-MINUTE FORWARD RETURN"
    )

    print("-" * 110)

    ranked = sorted(
        results,
        key=lambda x:
            x["avg_12"],
        reverse=True,
    )

    for rank, result in enumerate(
        ranked[:5],
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result['name']}"
        )

        print(
            f"   Signals: "
            f"{result['signals']}"
        )

        print(
            f"   60m: "
            f"{result['avg_12']:+.3f}%"
        )

        print(
            f"   Positive: "
            f"{result['positive_12']:.2f}%"
        )

    print()
    print(
        "TOP 5 BY 120-MINUTE FORWARD RETURN"
    )

    print("-" * 110)

    ranked_120 = sorted(
        results,
        key=lambda x:
            x["avg_24"],
        reverse=True,
    )

    for rank, result in enumerate(
        ranked_120[:5],
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result['name']}"
        )

        print(
            f"   Signals: "
            f"{result['signals']}"
        )

        print(
            f"   120m: "
            f"{result['avg_24']:+.3f}%"
        )

        print(
            f"   Positive: "
            f"{result['positive_24']:.2f}%"
        )

    print("=" * 110)

    return results


# ==========================================
# SAVE RESULTS
# ==========================================

def save_entry_results(
    results,
    symbol,
):

    if not results:
        return

    rows = []

    for result in results:

        rows.append(
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
        rows
    )

    output_file = (
        Path("logs")
        /
        (
            symbol.replace(
                "/",
                "_",
            )
            +
            "_entry_quality_v3_1.csv"
        )
    )

    output_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Entry quality log: "
        f"{output_file}"
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
        "Strategy v3.1"
    )
    print(
        "TARGETED ENTRY ANALYZER"
    )
    print("=" * 70)

    all_results = []

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
        # ORIGINAL DIAGNOSTICS
        # ==================================

        print()
        print(
            "Running signal diagnostics..."
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
        # QUALITY ANALYSIS
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
        "20→100 TARGETED ENTRY SUMMARY"
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

        best_60 = max(
            results,
            key=lambda x:
                x["avg_12"],
        )

        best_120 = max(
            results,
            key=lambda x:
                x["avg_24"],
        )

        print()
        print(
            symbol
        )

        print(
            f"Best 60m setup:   "
            f"{best_60['name']}"
        )

        print(
            f"Signals:          "
            f"{best_60['signals']}"
        )

        print(
            f"60m return:       "
            f"{best_60['avg_12']:+.3f}%"
        )

        print(
            f"Positive 60m:     "
            f"{best_60['positive_12']:.2f}%"
        )

        print()

        print(
            f"Best 120m setup:  "
            f"{best_120['name']}"
        )

        print(
            f"Signals:          "
            f"{best_120['signals']}"
        )

        print(
            f"120m return:      "
            f"{best_120['avg_24']:+.3f}%"
        )

        print(
            f"Positive 120m:    "
            f"{best_120['positive_24']:.2f}%"
        )

    print()
    print("=" * 110)
    print(
        "TARGETED ENTRY TEST COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
