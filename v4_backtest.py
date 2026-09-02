# ============================================================
# 20→100 TRADING BOT
# V4 WALK-FORWARD BACKTEST
#
# Ziel:
# - V3-Strategien über mehrere Marktphasen testen
# - 12 Monate Training / 3 Monate OOS
# - ETH/USDT 1h
# - V3_B / V3_C / V3_F
# - Keine Optimierung auf den OOS-Zeitraum
# ============================================================

import os
import pandas as pd

from strategy.strategy_v3 import StrategyV3
from backtest.v3_engine import V3BacktestEngine


STARTING_CAPITAL = 20.0
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

TIMEFRAME = "1h"

TRAIN_MONTHS = 12
TEST_MONTHS = 3

SYMBOLS = [
    "ETH/USDT",
]

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


def load_data(symbol):
    filename = symbol.replace("/", "_") + "_5m.csv"
    path = os.path.join("data", filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Historische Daten nicht gefunden: {path}"
        )

    print(f"Loading {path}")

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"{path} enthält keine 'timestamp'-Spalte."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.set_index("timestamp")

    required_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"{path}: Spalte '{column}' fehlt."
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    )

    df = df.sort_index()

    return df


def resample_to_1h(df):
    """
    5m -> 1h
    """

    result = df.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )

    result = result.dropna()

    return result


def calculate_indicators(df, params):
    """
    Berechnet die V3-Indikatoren EINMAL auf dem
    vollständigen Datensatz.

    Wichtig:
    Dadurch besitzt auch das erste OOS-Fenster
    die historische Warm-up-Historie.
    """

    strategy = StrategyV3(**params)

    return strategy.calculate_indicators(df)


def run_backtest(df, params):
    """
    Führt einen einzelnen Backtest aus.
    """

    strategy = StrategyV3(**params)

    engine = V3BacktestEngine(
        starting_balance=STARTING_CAPITAL,
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )

    result = engine.run(
        df,
        strategy,
    )

    return result


def add_result(
    results,
    symbol,
    strategy_name,
    window_number,
    train_start,
    train_end,
    test_start,
    test_end,
    result,
):
    results.append(
        {
            "symbol": symbol,
            "strategy": strategy_name,
            "window": window_number,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "final": result.get("final", 0.0),
            "return_pct": result.get("return_pct", 0.0),
            "trades": result.get("trades", 0),
            "win_rate": result.get("win_rate", 0.0),
            "profit_factor": result.get("profit_factor", 0.0),
            "expectancy": result.get("expectancy", 0.0),
            "max_drawdown": result.get("max_drawdown", 0.0),
            "fees": result.get("fees", 0.0),
            "slippage": result.get("slippage", 0.0),
        }
    )


def run_walk_forward(
    df,
    symbol,
    strategy_name,
    params,
):
    """
    Walk-Forward:

    12 Monate TRAIN
    3 Monate OOS

    Danach wird das Fenster um 3 Monate
    nach vorne verschoben.
    """

    results = []

    start = df.index.min()
    end = df.index.max()

    current_train_start = start

    window_number = 1

    while True:

        train_start = current_train_start

        train_end = (
            train_start
            + pd.DateOffset(months=TRAIN_MONTHS)
        )

        test_start = train_end

        test_end = (
            test_start
            + pd.DateOffset(months=TEST_MONTHS)
        )

        if test_end > end:
            break

        train_df = df[
            (df.index >= train_start)
            & (df.index < train_end)
        ].copy()

        test_df = df[
            (df.index >= test_start)
            & (df.index < test_end)
        ].copy()

        if len(train_df) < 500:
            current_train_start += pd.DateOffset(
                months=TEST_MONTHS
            )
            continue

        if len(test_df) < 100:
            current_train_start += pd.DateOffset(
                months=TEST_MONTHS
            )
            continue

        print()
        print("-" * 110)
        print(
            f"{symbol} | {strategy_name} | "
            f"WINDOW {window_number}"
        )
        print("-" * 110)

        print(
            f"TRAIN: "
            f"{train_start.date()} → "
            f"{train_end.date()} "
            f"({len(train_df):,} candles)"
        )

        print(
            f"OOS:   "
            f"{test_start.date()} → "
            f"{test_end.date()} "
            f"({len(test_df):,} candles)"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_result = run_backtest(
            train_df,
            params,
        )

        print(
            f"TRAIN | "
            f"Final ${train_result.get('final', 0):.2f} | "
            f"Return {train_result.get('return_pct', 0):+.2f}% | "
            f"Trades {train_result.get('trades', 0):3d} | "
            f"PF {train_result.get('profit_factor', 0):.3f}"
        )

        # ----------------------------------------------------
        # OOS
        # ----------------------------------------------------

        test_result = run_backtest(
            test_df,
            params,
        )

        print(
            f"OOS   | "
            f"Final ${test_result.get('final', 0):.2f} | "
            f"Return {test_result.get('return_pct', 0):+.2f}% | "
            f"Trades {test_result.get('trades', 0):3d} | "
            f"Win {test_result.get('win_rate', 0):.2f}% | "
            f"PF {test_result.get('profit_factor', 0):.3f} | "
            f"DD {test_result.get('max_drawdown', 0):+.2f}%"
        )

        add_result(
            results,
            symbol,
            strategy_name,
            window_number,
            train_start,
            train_end,
            test_start,
            test_end,
            test_result,
        )

        window_number += 1

        current_train_start += pd.DateOffset(
            months=TEST_MONTHS
        )

    return results


def summarize_strategy(results):
    """
    Erstellt eine robuste Zusammenfassung
    über alle OOS-Fenster.
    """

    if not results:
        return None

    df = pd.DataFrame(results)

    positive_windows = (
        df["return_pct"] > 0
    ).sum()

    negative_windows = (
        df["return_pct"] <= 0
    ).sum()

    total_windows = len(df)

    total_trades = int(
        df["trades"].sum()
    )

    average_return = (
        df["return_pct"].mean()
    )

    median_return = (
        df["return_pct"].median()
    )

    average_pf = (
        df["profit_factor"].mean()
    )

    median_pf = (
        df["profit_factor"].median()
    )

    average_dd = (
        df["max_drawdown"].mean()
    )

    worst_dd = (
        df["max_drawdown"].min()
    )

    return {
        "windows": total_windows,
        "positive_windows": int(positive_windows),
        "negative_windows": int(negative_windows),
        "positive_ratio": (
            positive_windows / total_windows * 100
        ),
        "total_trades": total_trades,
        "average_return": average_return,
        "median_return": median_return,
        "average_pf": average_pf,
        "median_pf": median_pf,
        "average_dd": average_dd,
        "worst_dd": worst_dd,
    }


def main():

    print("=" * 110)
    print("20→100 TRADING BOT")
    print("V4 WALK-FORWARD ROBUSTNESS TEST")
    print("=" * 110)

    print()
    print(f"Starting capital: ${STARTING_CAPITAL:.2f}")
    print("Market: ETH/USDT")
    print("Timeframe: 1h")
    print("TRAIN: 12 months")
    print("OOS: 3 months")
    print("Strategies: V3_B / V3_C / V3_F")
    print("=" * 110)

    all_results = []

    for symbol in SYMBOLS:

        raw_df = load_data(symbol)

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

        df = resample_to_1h(
            raw_df
        )

        print(
            f"1h candles: {len(df):,}"
        )

        # ----------------------------------------------------
        # Wichtig:
        #
        # Indikatoren werden auf dem gesamten Datensatz
        # berechnet, bevor die Walk-Forward-Fenster
        # ausgewertet werden.
        # ----------------------------------------------------

        for strategy_name, params in STRATEGIES.items():

            print()
            print("=" * 110)
            print(
                f"STARTING {strategy_name}"
            )
            print("=" * 110)

            results = run_walk_forward(
                df,
                symbol,
                strategy_name,
                params,
            )

            all_results.extend(
                results
            )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    if not all_results:

        print()
        print("❌ Keine Walk-Forward-Ergebnisse.")
        return

    results_df = pd.DataFrame(
        all_results
    )

    os.makedirs(
        "logs",
        exist_ok=True
    )

    output_file = (
        "logs/v4_walk_forward_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print()
    print("=" * 110)
    print("V4 WALK-FORWARD SUMMARY")
    print("=" * 110)

    summary_rows = []

    for strategy_name in STRATEGIES:

        strategy_results = [
            r
            for r in all_results
            if r["strategy"] == strategy_name
        ]

        summary = summarize_strategy(
            strategy_results
        )

        if summary is None:
            continue

        summary_rows.append(
            {
                "Strategy": strategy_name,
                "Windows": summary["windows"],
                "Positive": summary["positive_windows"],
                "Negative": summary["negative_windows"],
                "Positive %": summary["positive_ratio"],
                "Trades": summary["total_trades"],
                "Avg Return": summary["average_return"],
                "Median Return": summary["median_return"],
                "Avg PF": summary["average_pf"],
                "Median PF": summary["median_pf"],
                "Avg DD": summary["average_dd"],
                "Worst DD": summary["worst_dd"],
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print()

    print(
        f"{'Strategy':<10}"
        f"{'Windows':>8}"
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

    print("-" * 110)

    for _, row in summary_df.iterrows():

        print(
            f"{row['Strategy']:<10}"
            f"{int(row['Windows']):>8}"
            f"{int(row['Positive']):>10}"
            f"{int(row['Negative']):>10}"
            f"{row['Positive %']:>8.1f}%"
            f"{int(row['Trades']):>9}"
            f"{row['Avg Return']:>10.2f}%"
            f"{row['Median Return']:>10.2f}%"
            f"{row['Avg PF']:>9.3f}"
            f"{row['Median PF']:>9.3f}"
            f"{row['Worst DD']:>10.2f}%"
        )

    # --------------------------------------------------------
    # BEST STRATEGY
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("V4 BEST CANDIDATE")
    print("=" * 110)

    # Robustheitsranking:
    #
    # 1. Median PF
    # 2. Anteil positiver Fenster
    # 3. Median Return
    #
    # Dadurch wird ein einzelner Ausreißer nicht
    # überbewertet.

    ranked = summary_df.sort_values(
        by=[
            "Median PF",
            "Positive %",
            "Median Return",
        ],
        ascending=False,
    )

    best = ranked.iloc[0]

    print()
    print(
        f"🏆 {best['Strategy']}"
    )

    print(
        f"Positive OOS-Fenster: "
        f"{int(best['Positive'])}/"
        f"{int(best['Windows'])} "
        f"({best['Positive %']:.1f}%)"
    )

    print(
        f"Median OOS Return: "
        f"{best['Median Return']:+.2f}%"
    )

    print(
        f"Median OOS PF: "
        f"{best['Median PF']:.3f}"
    )

    print(
        f"Schlechtester Drawdown: "
        f"{best['Worst DD']:.2f}%"
    )

    print(
        f"Gesamte OOS-Trades: "
        f"{int(best['Trades'])}"
    )

    print()
    print(
        f"Results saved: {output_file}"
    )

    print()
    print("=" * 110)
    print("V4 WALK-FORWARD BACKTEST COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
