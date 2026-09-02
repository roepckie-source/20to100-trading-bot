# ============================================================
# 20→100 TRADING BOT
# V4 WALK-FORWARD BACKTEST
#
# ETH/USDT 1h
#
# Testet:
#   V3_B = Donchian 20 / ATR 2.5
#   V3_C = Donchian 20 / ATR 3.0
#   V3_F = Donchian 50 / ATR 3.5
#
# Walk-Forward:
#   12 Monate TRAIN
#   3 Monate OOS
#
# Ziel:
#   Prüfen, ob V3 über mehrere Marktphasen robust bleibt.
# ============================================================

import os
import pandas as pd

from strategy.strategy_v3 import StrategyV3
from backtest.v4_engine import V4BacktestEngine


# ============================================================
# CONFIG
# ============================================================

STARTING_CAPITAL = 20.0

FEE_RATE = 0.001

SLIPPAGE_RATE = 0.0005

SYMBOL = "ETH/USDT"

TIMEFRAME = "1h"

TRAIN_MONTHS = 12

TEST_MONTHS = 3


# ============================================================
# STRATEGIES
# ============================================================

STRATEGIES = {

    "V3_B": {
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian_period": 20,
        "atr_period": 14,
        "atr_stop_multiplier": 2.5,
    },

    "V3_C": {
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian_period": 20,
        "atr_period": 14,
        "atr_stop_multiplier": 3.0,
    },

    "V3_F": {
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian_period": 50,
        "atr_period": 14,
        "atr_stop_multiplier": 3.5,
    },
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    filename = (
        SYMBOL.replace("/", "_")
        + "_5m.csv"
    )

    path = os.path.join(
        "data",
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Historische Daten nicht gefunden: {path}"
        )

    print(
        f"Loading {path}"
    )

    df = pd.read_csv(
        path
    )

    if "timestamp" not in df.columns:

        raise ValueError(
            "CSV enthält keine timestamp-Spalte."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.set_index(
        "timestamp"
    )

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Spalte fehlt: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    )

    df = df.sort_index()

    return df


# ============================================================
# RESAMPLE
# ============================================================

def resample_to_1h(df):

    result = df.resample(
        "1h"
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )

    result = result.dropna()

    return result


# ============================================================
# CALCULATE INDICATORS
# ============================================================

def calculate_full_indicators(
    df,
    params
):
    """
    Die Indikatoren werden auf dem vollständigen
    Datensatz berechnet.

    Dadurch stehen EMA / ATR / Donchian auch am
    Anfang eines OOS-Fensters mit ihrer historischen
    Warm-up-Historie zur Verfügung.
    """

    strategy = StrategyV3(
        **params
    )

    return strategy.calculate_indicators(
        df.copy()
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df,
    params
):

    strategy = StrategyV3(
        **params
    )

    engine = V4BacktestEngine(
        starting_balance=STARTING_CAPITAL,
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )

    result = engine.run(
        df,
        strategy
    )

    return result


# ============================================================
# RESULT FORMAT
# ============================================================

def format_result(
    label,
    result
):

    print(
        f"{label:<5} | "
        f"Final ${result.get('final', 0):7.2f} | "
        f"Return {result.get('return_pct', 0):+8.2f}% | "
        f"Trades {result.get('trades', 0):4d} | "
        f"Win {result.get('win_rate', 0):6.2f}% | "
        f"PF {result.get('profit_factor', 0):6.3f} | "
        f"DD {result.get('max_drawdown', 0):+8.2f}%"
    )


# ============================================================
# WALK FORWARD
# ============================================================

def run_walk_forward(
    df,
    strategy_name,
    params
):

    results = []

    first_timestamp = df.index.min()

    last_timestamp = df.index.max()

    current_start = first_timestamp

    window = 1

    while True:

        train_start = current_start

        train_end = (
            train_start
            + pd.DateOffset(
                months=TRAIN_MONTHS
            )
        )

        test_start = train_end

        test_end = (
            test_start
            + pd.DateOffset(
                months=TEST_MONTHS
            )
        )

        # ----------------------------------------------------
        # Wenn kein vollständiges OOS-Fenster mehr vorhanden
        # ist, beenden wir den Walk-Forward-Test.
        # ----------------------------------------------------

        if test_end > last_timestamp:

            break

        train_df = df[
            (df.index >= train_start)
            &
            (df.index < train_end)
        ].copy()

        test_df = df[
            (df.index >= test_start)
            &
            (df.index < test_end)
        ].copy()

        if len(train_df) < 500:

            current_start += pd.DateOffset(
                months=TEST_MONTHS
            )

            continue

        if len(test_df) < 100:

            current_start += pd.DateOffset(
                months=TEST_MONTHS
            )

            continue

        print()
        print(
            "-" * 110
        )

        print(
            f"{strategy_name} | WINDOW {window}"
        )

        print(
            f"TRAIN: "
            f"{train_start.date()} → "
            f"{train_end.date()} "
            f"| {len(train_df):,} candles"
        )

        print(
            f"OOS:   "
            f"{test_start.date()} → "
            f"{test_end.date()} "
            f"| {len(test_df):,} candles"
        )

        print(
            "-" * 110
        )

        # ====================================================
        # TRAIN
        # ====================================================

        train_result = run_backtest(
            train_df,
            params
        )

        format_result(
            "TRAIN",
            train_result
        )

        # ====================================================
        # OOS
        # ====================================================

        test_result = run_backtest(
            test_df,
            params
        )

        format_result(
            "OOS",
            test_result
        )

        # ====================================================
        # STORE OOS RESULT
        # ====================================================

        results.append(
            {
                "symbol": SYMBOL,
                "strategy": strategy_name,
                "window": window,

                "train_start": train_start,
                "train_end": train_end,

                "test_start": test_start,
                "test_end": test_end,

                "final": test_result.get(
                    "final",
                    0.0
                ),

                "return_pct": test_result.get(
                    "return_pct",
                    0.0
                ),

                "trades": test_result.get(
                    "trades",
                    0
                ),

                "win_rate": test_result.get(
                    "win_rate",
                    0.0
                ),

                "profit_factor": test_result.get(
                    "profit_factor",
                    0.0
                ),

                "expectancy": test_result.get(
                    "expectancy",
                    0.0
                ),

                "max_drawdown": test_result.get(
                    "max_drawdown",
                    0.0
                ),

                "fees": test_result.get(
                    "fees",
                    0.0
                ),

                "slippage": test_result.get(
                    "slippage",
                    0.0
                ),
            }
        )

        window += 1

        # ----------------------------------------------------
        # Fenster um 3 Monate verschieben
        # ----------------------------------------------------

        current_start += pd.DateOffset(
            months=TEST_MONTHS
        )

    return results


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    results,
    strategy_name
):

    if not results:

        return None

    df = pd.DataFrame(
        results
    )

    positive = (
        df["return_pct"] > 0
    ).sum()

    negative = (
        df["return_pct"] <= 0
    ).sum()

    windows = len(df)

    positive_ratio = (
        positive
        / windows
        * 100
    )

    total_trades = int(
        df["trades"].sum()
    )

    average_return = float(
        df["return_pct"].mean()
    )

    median_return = float(
        df["return_pct"].median()
    )

    average_pf = float(
        df["profit_factor"].mean()
    )

    median_pf = float(
        df["profit_factor"].median()
    )

    average_dd = float(
        df["max_drawdown"].mean()
    )

    worst_dd = float(
        df["max_drawdown"].min()
    )

    return {
        "strategy": strategy_name,

        "windows": windows,

        "positive": int(
            positive
        ),

        "negative": int(
            negative
        ),

        "positive_ratio": positive_ratio,

        "total_trades": total_trades,

        "average_return": average_return,

        "median_return": median_return,

        "average_pf": average_pf,

        "median_pf": median_pf,

        "average_dd": average_dd,

        "worst_dd": worst_dd,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 110
    )

    print(
        "20→100 TRADING BOT"
    )

    print(
        "V4 WALK-FORWARD ROBUSTNESS TEST"
    )

    print(
        "=" * 110
    )

    print()
    print(
        f"Starting capital: ${STARTING_CAPITAL:.2f}"
    )

    print(
        f"Market: {SYMBOL}"
    )

    print(
        f"Timeframe: {TIMEFRAME}"
    )

    print(
        f"TRAIN: {TRAIN_MONTHS} months"
    )

    print(
        f"OOS: {TEST_MONTHS} months"
    )

    print(
        "Strategies: V3_B / V3_C / V3_F"
    )

    print(
        "=" * 110
    )

    # ========================================================
    # LOAD
    # ========================================================

    raw_df = load_data()

    print()
    print(
        f"5m candles: {len(raw_df):,}"
    )

    print(
        f"From: {raw_df.index.min()}"
    )

    print(
        f"To:   {raw_df.index.max()}"
    )

    # ========================================================
    # RESAMPLE
    # ========================================================

    df = resample_to_1h(
        raw_df
    )

    print()
    print(
        f"1h candles: {len(df):,}"
    )

    # ========================================================
    # RUN
    # ========================================================

    all_results = []

    for strategy_name, params in STRATEGIES.items():

        print()
        print(
            "=" * 110
        )

        print(
            f"STARTING {strategy_name}"
        )

        print(
            "=" * 110
        )

        strategy_results = run_walk_forward(
            df,
            strategy_name,
            params
        )

        all_results.extend(
            strategy_results
        )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    if not all_results:

        print()
        print(
            "❌ Keine Ergebnisse."
        )

        return

    results_df = pd.DataFrame(
        all_results
    )

    os.makedirs(
        "logs",
        exist_ok=True
    )

    result_file = (
        "logs/v4_walk_forward_results.csv"
    )

    results_df.to_csv(
        result_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summaries = []

    for strategy_name in STRATEGIES:

        strategy_results = [
            result
            for result in all_results
            if result["strategy"]
            == strategy_name
        ]

        summary = summarize(
            strategy_results,
            strategy_name
        )

        if summary:

            summaries.append(
                summary
            )

    summary_df = pd.DataFrame(
        summaries
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print()
    print(
        "=" * 110
    )

    print(
        "V4 WALK-FORWARD SUMMARY"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"{'Strategy':<10}"
        f"{'Windows':>9}"
        f"{'Positive':>10}"
        f"{'Negative':>10}"
        f"{'Pos %':>9}"
        f"{'Trades':>9}"
        f"{'Avg Ret':>11}"
        f"{'Med Ret':>11}"
        f"{'Avg PF':>9}"
        f"{'Med PF':>9}"
        f"{'Worst DD':>11}"
    )

    print(
        "-" * 110
    )

    for _, row in summary_df.iterrows():

        print(
            f"{row['strategy']:<10}"
            f"{int(row['windows']):>9}"
            f"{int(row['positive']):>10}"
            f"{int(row['negative']):>10}"
            f"{row['positive_ratio']:>8.1f}%"
            f"{int(row['total_trades']):>9}"
            f"{row['average_return']:>10.2f}%"
            f"{row['median_return']:>10.2f}%"
            f"{row['average_pf']:>9.3f}"
            f"{row['median_pf']:>9.3f}"
            f"{row['worst_dd']:>10.2f}%"
        )

    # ========================================================
    # BEST CANDIDATE
    # ========================================================

    print()
    print(
        "=" * 110
    )

    print(
        "V4 BEST CANDIDATE"
    )

    print(
        "=" * 110
    )

    ranked = summary_df.sort_values(
        by=[
            "median_pf",
            "positive_ratio",
            "median_return",
        ],
        ascending=False
    )

    best = ranked.iloc[0]

    print()
    print(
        f"🏆 {best['strategy']}"
    )

    print(
        f"Positive OOS windows: "
        f"{int(best['positive'])}/"
        f"{int(best['windows'])}"
        f" ({best['positive_ratio']:.1f}%)"
    )

    print(
        f"Median OOS return: "
        f"{best['median_return']:+.2f}%"
    )

    print(
        f"Median OOS PF: "
        f"{best['median_pf']:.3f}"
    )

    print(
        f"Average OOS PF: "
        f"{best['average_pf']:.3f}"
    )

    print(
        f"Worst OOS drawdown: "
        f"{best['worst_dd']:.2f}%"
    )

    print(
        f"Total OOS trades: "
        f"{int(best['total_trades'])}"
    )

    print()
    print(
        f"Results saved: {result_file}"
    )

    print()
    print(
        "=" * 110
    )

    print(
        "V4 WALK-FORWARD BACKTEST COMPLETE"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":

    main()
