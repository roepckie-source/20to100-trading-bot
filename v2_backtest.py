# ============================================================
# V2 Backtest Runner
# ============================================================

import os
import pandas as pd

from backtest.v2_engine import V2BacktestEngine


STARTING_CAPITAL = 20.0

FEE_RATE = 0.001
SLIPPAGE = 0.0005


DATA_FILES = {
    "BTC": "data/BTC_USDT_5m.csv",
    "ETH": "data/ETH_USDT_5m.csv",
}


TIMEFRAMES = {
    "5m": 1,
    "15m": 3,
    "1h": 12,
}


def load_data(path):

    if not os.path.exists(path):

        print(
            f"❌ Datei nicht gefunden: {path}"
        )

        return None

    df = pd.read_csv(path)

    # Timestamp erkennen
    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.set_index("timestamp")

    # Alternative Schreibweisen
    rename_map = {}

    for column in df.columns:

        lower = column.lower()

        if lower == "open":
            rename_map[column] = "open"

        elif lower == "high":
            rename_map[column] = "high"

        elif lower == "low":
            rename_map[column] = "low"

        elif lower == "close":
            rename_map[column] = "close"

        elif lower == "volume":
            rename_map[column] = "volume"

    df = df.rename(
        columns=rename_map
    )

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:

        print(
            f"❌ Fehlende Spalten: {missing}"
        )

        return None

    df = df.sort_index()

    return df


def resample_data(df, rule):

    if rule == "5m":
        return df.copy()

    if rule == "15m":

        return (
            df.resample("15min")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
                if "volume" in df.columns
                else "last",
            })
            .dropna()
        )

    if rule == "1h":

        return (
            df.resample("1h")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
                if "volume" in df.columns
                else "last",
            })
            .dropna()
        )

    raise ValueError(
        f"Unbekanntes Timeframe: {rule}"
    )


def run_test(symbol, timeframe, df):

    engine = V2BacktestEngine(
        starting_capital=STARTING_CAPITAL,
        fee_rate=FEE_RATE,
        slippage=SLIPPAGE,

        donchian_period=20,

        ema_period=200,

        atr_period=14,

        atr_stop_multiplier=2.5,

        atr_trailing_multiplier=2.5,
    )

    result, trades = engine.run(df)

    pf = result["profit_factor"]

    if pf == float("inf"):
        pf_text = "∞"
    else:
        pf_text = f"{pf:.3f}"

    print(
        f"{symbol:<5} "
        f"{timeframe:<4} | "
        f"Trades: {result['trades']:<4} | "
        f"Final: ${result['final_capital']:.2f} | "
        f"Return: {result['return_pct']:+.2f}% | "
        f"Win: {result['win_rate']:.2f}% | "
        f"PF: {pf_text} | "
        f"DD: {result['max_drawdown_pct']:.2f}%"
    )

    return result


def main():

    print()
    print("=" * 80)
    print("🚀 STRATEGY V2 – DONCHIAN + EMA200 + ATR")
    print("=" * 80)
    print()

    all_results = []

    for symbol, path in DATA_FILES.items():

        print()
        print("-" * 80)
        print(f"📊 {symbol}")
        print("-" * 80)

        df = load_data(path)

        if df is None:
            continue

        print(
            f"Daten: {len(df):,} Kerzen"
        )

        print(
            f"Von: {df.index.min()}"
        )

        print(
            f"Bis: {df.index.max()}"
        )

        print()

        for timeframe, rule in TIMEFRAMES.items():

            test_df = resample_data(
                df,
                timeframe
            )

            result = run_test(
                symbol,
                timeframe,
                test_df
            )

            all_results.append({
                "symbol": symbol,
                "timeframe": timeframe,
                **result,
            })

    print()
    print("=" * 80)
    print("🏆 ERGEBNISSE")
    print("=" * 80)

    results_df = pd.DataFrame(
        all_results
    )

    if len(results_df) == 0:

        print(
            "Keine Ergebnisse."
        )

        return

    print()

    for _, row in results_df.iterrows():

        print(
            f"{row['symbol']} "
            f"{row['timeframe']}: "
            f"${row['final_capital']:.2f} | "
            f"{row['return_pct']:+.2f}% | "
            f"{int(row['trades'])} Trades | "
            f"PF {row['profit_factor']:.3f}"
        )

    print()

    # Beste Variante
    best = results_df.loc[
        results_df["final_capital"].idxmax()
    ]

    print("=" * 80)
    print("🥇 BESTE V2 VARIANTE")
    print("=" * 80)

    print(
        f"{best['symbol']} "
        f"{best['timeframe']}"
    )

    print(
        f"Startkapital: "
        f"${best['starting_capital']:.2f}"
    )

    print(
        f"Endkapital: "
        f"${best['final_capital']:.2f}"
    )

    print(
        f"Rendite: "
        f"{best['return_pct']:+.2f}%"
    )

    print(
        f"Trades: "
        f"{int(best['trades'])}"
    )

    print(
        f"Win Rate: "
        f"{best['win_rate']:.2f}%"
    )

    print(
        f"Profit Factor: "
        f"{best['profit_factor']:.3f}"
    )

    print(
        f"Max Drawdown: "
        f"{best['max_drawdown_pct']:.2f}%"
    )

    print()


if __name__ == "__main__":
    main()
