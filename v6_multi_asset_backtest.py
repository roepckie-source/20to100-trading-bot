# ============================================================
# V6 MULTI-ASSET WALK-FORWARD BACKTEST
# ============================================================
#
# V6.0 Adaptive Regime Filter
#
# Assets:
#   BTC/USDT
#   ETH/USDT
#   SOL/USDT
#
# Variants:
#   V6_A = ADX Rising
#   V6_B = ADX Rising + Strong EMA200 Slope
#   V6_C = V6_B + ATR Volatility Filter
#
# Timeframe:
#   5m -> 1h
#
# Walk-Forward:
#   12 months TRAIN
#   3 months OOS
#
# IMPORTANT:
# V6 currently uses fixed parameters and does not optimize
# on the TRAIN section. Therefore this is a rolling/fixed-
# parameter OOS robustness test rather than parameter-optimized
# walk-forward optimization.
# ============================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd

from strategy.strategy_v6 import calculate_indicators
from backtest.v6_engine import V6BacktestEngine


warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ASSETS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
]

VARIANTS = [
    "V6_A",
    "V6_B",
    "V6_C",
]

DATA_DIR = "data"
LOG_DIR = "logs"

STARTING_BALANCE = 20.0

RISK_PER_TRADE = 0.01
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

ATR_STOP_MULTIPLIER = 3.0
TRAILING_ATR_MULTIPLIER = 3.0

ADX_MIN = 20.0

MAX_DAILY_LOSS = 0.05
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_BARS = 24

GLOBAL_MAX_DRAWDOWN = 0.20

TRAIN_MONTHS = 12
OOS_MONTHS = 3


# ============================================================
# DATA LOADING
# ============================================================

def find_data_file(symbol: str):
    """
    Find the corresponding 5m CSV for an asset.
    """

    clean = symbol.replace("/", "_")

    candidates = [
        os.path.join(DATA_DIR, f"{clean}_5m.csv"),
        os.path.join(DATA_DIR, f"{clean}.csv"),
        os.path.join(DATA_DIR, f"{clean.lower()}_5m.csv"),
        os.path.join(DATA_DIR, f"{clean.lower()}.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Flexible fallback
    patterns = [
        os.path.join(DATA_DIR, f"*{clean}*5m*.csv"),
        os.path.join(DATA_DIR, f"*{clean}*.csv"),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)

        if matches:
            return matches[0]

    return None


def load_data(symbol: str):
    """
    Load 5m OHLCV data.

    Handles:
    - integer timestamps
    - string timestamps
    - Pandas StringDtype
    - object timestamps
    - milliseconds
    - seconds
    """

    path = find_data_file(symbol)

    if path is None:
        print(f"❌ Keine Daten gefunden für {symbol}")
        print(
            f"   Erwartet z.B.: "
            f"{DATA_DIR}/{symbol.replace('/', '_')}_5m.csv"
        )
        return None

    print(f"📂 {symbol}: {path}")

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    # --------------------------------------------------------
    # Detect timestamp column
    # --------------------------------------------------------

    timestamp_candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "open_time",
    ]

    timestamp_col = None

    for col in timestamp_candidates:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        raise ValueError(
            f"{symbol}: Keine Timestamp-Spalte gefunden. "
            f"Vorhanden: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp_series = df[timestamp_col]

    # Pandas StringDtype / object / numeric sicher behandeln
    if pd.api.types.is_numeric_dtype(timestamp_series):

        numeric_timestamp = pd.to_numeric(
            timestamp_series,
            errors="coerce",
        )

        median_value = numeric_timestamp.median()

        if pd.isna(median_value):
            raise ValueError(
                f"{symbol}: Timestamp-Spalte enthält "
                f"keine gültigen numerischen Werte."
            )

        # Detect milliseconds vs seconds
        unit = (
            "ms"
            if median_value > 10_000_000_000
            else "s"
        )

        df["timestamp"] = pd.to_datetime(
            numeric_timestamp,
            unit=unit,
            utc=True,
            errors="coerce",
        )

    else:

        df["timestamp"] = pd.to_datetime(
            timestamp_series,
            utc=True,
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove invalid timestamps
    # --------------------------------------------------------

    invalid_timestamps = df["timestamp"].isna().sum()

    if invalid_timestamps > 0:
        print(
            f"   ⚠️ {invalid_timestamps:,} "
            f"ungültige Timestamps entfernt."
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    if df.empty:
        raise ValueError(
            f"{symbol}: Nach Timestamp-Konvertierung "
            f"sind keine Daten übrig."
        )

    df = df.set_index("timestamp")

    # --------------------------------------------------------
    # Required OHLC columns
    # --------------------------------------------------------

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol}: Fehlende Spalten: {missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in required:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if "volume" in df.columns:

        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Keep relevant columns
    # --------------------------------------------------------

    columns = required.copy()

    if "volume" in df.columns:
        columns.append("volume")

    df = df[columns]

    # --------------------------------------------------------
    # Remove invalid OHLC rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=required
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_index()

    # --------------------------------------------------------
    # Price validation
    # --------------------------------------------------------

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    ]

    # High must be >= Low
    df = df[
        df["high"] >= df["low"]
    ]

    if df.empty:
        raise ValueError(
            f"{symbol}: Nach Datenvalidierung "
            f"sind keine gültigen Kerzen übrig."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"   Zeitraum: "
        f"{df.index.min()} → "
        f"{df.index.max()}"
    )

    print(
        f"   5m Kerzen: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# RESAMPLE 5m -> 1h
# ============================================================

def resample_to_1h(
    df: pd.DataFrame,
):
    """
    Convert 5m OHLCV data into 1h candles.
    """

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    if "volume" in df.columns:
        agg["volume"] = "sum"

    hourly = (
        df.resample("1h")
        .agg(agg)
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
    )

    return hourly


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================

def create_walk_forward_windows(
    df: pd.DataFrame,
):
    """
    Rolling 12m TRAIN / 3m OOS windows.
    """

    windows = []

    start = df.index.min()

    while True:

        train_start = start

        train_end = (
            train_start
            + pd.DateOffset(
                months=TRAIN_MONTHS
            )
        )

        oos_end = (
            train_end
            + pd.DateOffset(
                months=OOS_MONTHS
            )
        )

        if oos_end > df.index.max():
            break

        train = df[
            (df.index >= train_start)
            & (df.index < train_end)
        ].copy()

        oos = df[
            (df.index >= train_end)
            & (df.index < oos_end)
        ].copy()

        if (
            len(train) == 0
            or len(oos) == 0
        ):
            break

        windows.append(
            {
                "window": len(windows) + 1,

                "train_start":
                    train.index.min(),

                "train_end":
                    train.index.max(),

                "oos_start":
                    oos.index.min(),

                "oos_end":
                    oos.index.max(),

                "train": train,

                "oos": oos,
            }
        )

        # Roll forward by OOS period
        start = train_end

    return windows


# ============================================================
# RUN ONE WINDOW
# ============================================================

def run_window(
    oos_df: pd.DataFrame,
    variant: str,
):
    """
    Run V6 engine on one OOS window.
    """

    engine = V6BacktestEngine(
        starting_balance=STARTING_BALANCE,
        risk_per_trade=RISK_PER_TRADE,
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
        atr_stop_multiplier=ATR_STOP_MULTIPLIER,
        trailing_atr_multiplier=TRAILING_ATR_MULTIPLIER,
        adx_min=ADX_MIN,
        max_daily_loss=MAX_DAILY_LOSS,
        max_consecutive_losses=MAX_CONSECUTIVE_LOSSES,
        cooldown_bars=COOLDOWN_BARS,
        global_max_drawdown=GLOBAL_MAX_DRAWDOWN,
        variant=variant,
    )

    result = engine.run(
        oos_df
    )

    return result


# ============================================================
# SAFE RESULT VALUE
# ============================================================

def safe_float(
    value,
):
    """
    Convert a value safely to float.
    """

    try:

        if value is None:
            return np.nan

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return np.nan


# ============================================================
# EXTRACT RESULT
# ============================================================

def extract_result(
    result,
    symbol,
    variant,
    window_info,
):
    """
    Normalize engine output into one result dictionary.
    """

    if result is None:
        result = {}

    return {
        "symbol": symbol,

        "variant": variant,

        "window":
            window_info["window"],

        "train_start":
            window_info["train_start"],

        "train_end":
            window_info["train_end"],

        "oos_start":
            window_info["oos_start"],

        "oos_end":
            window_info["oos_end"],

        "starting_balance":
            safe_float(
                result.get(
                    "starting_balance",
                    STARTING_BALANCE,
                )
            ),

        "final_balance":
            safe_float(
                result.get(
                    "final_balance",
                    np.nan,
                )
            ),

        "profit":
            safe_float(
                result.get(
                    "profit",
                    np.nan,
                )
            ),

        "return_pct":
            safe_float(
                result.get(
                    "return_pct",
                    np.nan,
                )
            ),

        "trades":
            result.get(
                "trades",
                0,
            ),

        "wins":
            result.get(
                "wins",
                0,
            ),

        "losses":
            result.get(
                "losses",
                0,
            ),

        "win_rate":
            safe_float(
                result.get(
                    "win_rate",
                    np.nan,
                )
            ),

        "profit_factor":
            safe_float(
                result.get(
                    "profit_factor",
                    np.nan,
                )
            ),

        "expectancy":
            safe_float(
                result.get(
                    "expectancy",
                    np.nan,
                )
            ),

        "max_drawdown":
            safe_float(
                result.get(
                    "max_drawdown",
                    np.nan,
                )
            ),

        "fees":
            safe_float(
                result.get(
                    "fees",
                    np.nan,
                )
            ),

        "slippage":
            safe_float(
                result.get(
                    "slippage",
                    np.nan,
                )
            ),
    }


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    results,
):
    """
    Create summary per asset / variant.
    """

    df = pd.DataFrame(
        results
    )

    if df.empty:
        return df

    summaries = []

    grouped = df.groupby(
        [
            "symbol",
            "variant",
        ]
    )

    for (
        symbol,
        variant,
    ), group in grouped:

        returns = pd.to_numeric(
            group["return_pct"],
            errors="coerce",
        )

        pf = pd.to_numeric(
            group["profit_factor"],
            errors="coerce",
        )

        trades = pd.to_numeric(
            group["trades"],
            errors="coerce",
        )

        win_rate = pd.to_numeric(
            group["win_rate"],
            errors="coerce",
        )

        dd = pd.to_numeric(
            group["max_drawdown"],
            errors="coerce",
        )

        summaries.append(
            {
                "symbol": symbol,

                "variant": variant,

                "windows":
                    len(group),

                "positive_windows":
                    int(
                        (returns > 0).sum()
                    ),

                "positive_window_pct":
                    (
                        (returns > 0).mean()
                        * 100
                    ),

                "pf_above_1_windows":
                    int(
                        (pf > 1).sum()
                    ),

                "avg_return_pct":
                    returns.mean(),

                "median_return_pct":
                    returns.median(),

                "best_return_pct":
                    returns.max(),

                "worst_return_pct":
                    returns.min(),

                "avg_profit_factor":
                    pf.mean(),

                "median_profit_factor":
                    pf.median(),

                "total_trades":
                    trades.sum(),

                "avg_trades_per_window":
                    trades.mean(),

                "avg_win_rate":
                    win_rate.mean(),

                "worst_drawdown":
                    dd.min(),
            }
        )

    return pd.DataFrame(
        summaries
    )


# ============================================================
# VARIANT SUMMARY
# ============================================================

def create_variant_summary(
    summary_df,
):
    """
    Aggregate all assets per variant.
    """

    if summary_df.empty:
        return pd.DataFrame()

    rows = []

    for variant in VARIANTS:

        subset = summary_df[
            summary_df["variant"]
            == variant
        ]

        if subset.empty:
            continue

        rows.append(
            {
                "variant":
                    variant,

                "assets_tested":
                    len(subset),

                "avg_asset_return_pct":
                    subset[
                        "avg_return_pct"
                    ].mean(),

                "median_asset_return_pct":
                    subset[
                        "median_return_pct"
                    ].median(),

                "avg_profit_factor":
                    subset[
                        "avg_profit_factor"
                    ].mean(),

                "positive_window_pct_avg":
                    subset[
                        "positive_window_pct"
                    ].mean(),

                "total_trades":
                    subset[
                        "total_trades"
                    ].sum(),

                "worst_drawdown":
                    subset[
                        "worst_drawdown"
                    ].min(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary_df,
    variant_summary,
):
    """
    Print readable V6 summary.
    """

    print()

    print("=" * 90)
    print("V6 MULTI-ASSET SUMMARY")
    print("=" * 90)

    if summary_df.empty:

        print(
            "❌ Keine Ergebnisse."
        )

        return

    for _, row in summary_df.iterrows():

        print()

        print(
            f"{row['symbol']} | "
            f"{row['variant']}"
        )

        print(
            f"   Positive Windows: "
            f"{int(row['positive_windows'])}/"
            f"{int(row['windows'])} "
            f"({row['positive_window_pct']:.1f}%)"
        )

        print(
            f"   PF > 1: "
            f"{int(row['pf_above_1_windows'])}/"
            f"{int(row['windows'])}"
        )

        print(
            f"   Avg Return: "
            f"{row['avg_return_pct']:.2f}%"
        )

        print(
            f"   Median Return: "
            f"{row['median_return_pct']:.2f}%"
        )

        print(
            f"   Best: "
            f"{row['best_return_pct']:.2f}%"
        )

        print(
            f"   Worst: "
            f"{row['worst_return_pct']:.2f}%"
        )

        print(
            f"   Avg PF: "
            f"{row['avg_profit_factor']:.3f}"
        )

        print(
            f"   Median PF: "
            f"{row['median_profit_factor']:.3f}"
        )

        print(
            f"   Trades: "
            f"{int(row['total_trades'])}"
        )

        print(
            f"   Avg Win Rate: "
            f"{row['avg_win_rate']:.2f}%"
        )

        print(
            f"   Worst DD: "
            f"{row['worst_drawdown']:.2f}%"
        )

    print()

    print("=" * 90)
    print("VARIANT COMPARISON")
    print("=" * 90)

    if not variant_summary.empty:

        print()

        for _, row in variant_summary.iterrows():

            print(
                f"{row['variant']}: "
                f"Avg Return "
                f"{row['avg_asset_return_pct']:.2f}% | "
                f"Median Return "
                f"{row['median_asset_return_pct']:.2f}% | "
                f"Avg PF "
                f"{row['avg_profit_factor']:.3f} | "
                f"Positive Windows "
                f"{row['positive_window_pct_avg']:.1f}% | "
                f"Worst DD "
                f"{row['worst_drawdown']:.2f}% | "
                f"Trades "
                f"{int(row['total_trades'])}"
            )

    print()

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 90)
    print(
        "V6 MULTI-ASSET WALK-FORWARD BACKTEST"
    )
    print("=" * 90)

    print()

    print(
        "Assets:",
        ", ".join(ASSETS)
    )

    print(
        "Variants:",
        ", ".join(VARIANTS)
    )

    print(
        f"TRAIN: {TRAIN_MONTHS} Monate | "
        f"OOS: {OOS_MONTHS} Monate"
    )

    print(
        "Parameter: FIXED"
    )

    print(
        "⚠️ Kein Parameter-Fitting "
        "im TRAIN-Bereich."
    )

    print()

    os.makedirs(
        LOG_DIR,
        exist_ok=True,
    )

    all_results = []

    # ========================================================
    # ASSET LOOP
    # ========================================================

    for symbol in ASSETS:

        print()

        print("#" * 90)

        print(
            f"ASSET: {symbol}"
        )

        print("#" * 90)

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        try:

            df_5m = load_data(
                symbol
            )

        except Exception as e:

            print(
                f"❌ Fehler beim Laden "
                f"von {symbol}: "
                f"{type(e).__name__}: {e}"
            )

            continue

        if df_5m is None:

            print(
                f"⏭️ {symbol} übersprungen."
            )

            continue

        # ----------------------------------------------------
        # RESAMPLE
        # ----------------------------------------------------

        print(
            f"🔄 {symbol}: 5m → 1h"
        )

        df = resample_to_1h(
            df_5m
        )

        print(
            f"   1h Kerzen: "
            f"{len(df):,}"
        )

        if len(df) < 5000:

            print(
                f"⚠️ Wenige 1h Kerzen "
                f"für {symbol}: "
                f"{len(df)}"
            )

        # ----------------------------------------------------
        # INDICATORS
        # ----------------------------------------------------

        print(
            f"📊 {symbol}: "
            f"V6 Indikatoren berechnen..."
        )

        try:

            df = calculate_indicators(
                df
            )

        except Exception as e:

            print(
                f"❌ Fehler bei V6 "
                f"Indikatoren für "
                f"{symbol}: "
                f"{type(e).__name__}: {e}"
            )

            continue

        # ----------------------------------------------------
        # WALK-FORWARD WINDOWS
        # ----------------------------------------------------

        windows = (
            create_walk_forward_windows(
                df
            )
        )

        print(
            f"📅 {len(windows)} "
            f"Walk-Forward Windows"
        )

        if not windows:

            print(
                f"❌ Keine vollständigen "
                f"Windows für {symbol}."
            )

            continue

        # ====================================================
        # VARIANT LOOP
        # ====================================================

        for variant in VARIANTS:

            print()

            print(
                "-" * 90
            )

            print(
                f"🚀 {symbol} | {variant}"
            )

            print(
                "-" * 90
            )

            # ------------------------------------------------
            # WINDOW LOOP
            # ------------------------------------------------

            for window in windows:

                window_no = (
                    window["window"]
                )

                print(
                    f"   Window "
                    f"{window_no:02d}/"
                    f"{len(windows)} | "
                    f"OOS "
                    f"{window['oos_start'].date()} → "
                    f"{window['oos_end'].date()}"
                )

                try:

                    result = run_window(
                        window["oos"],
                        variant,
                    )

                    row = extract_result(
                        result,
                        symbol,
                        variant,
                        window,
                    )

                    all_results.append(
                        row
                    )

                    return_pct = safe_float(
                        row["return_pct"]
                    )

                    profit_factor = safe_float(
                        row["profit_factor"]
                    )

                    if pd.isna(
                        return_pct
                    ):
                        return_text = "n/a"
                    else:
                        return_text = (
                            f"{return_pct:.2f}%"
                        )

                    if pd.isna(
                        profit_factor
                    ):
                        pf_text = "n/a"
                    else:
                        pf_text = (
                            f"{profit_factor:.3f}"
                        )

                    print(
                        f"      Return: "
                        f"{return_text} | "
                        f"PF: "
                        f"{pf_text} | "
                        f"Trades: "
                        f"{int(row['trades'])}"
                    )

                except Exception as e:

                    print(
                        f"      ❌ Fehler: "
                        f"{type(e).__name__}: "
                        f"{e}"
                    )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:

        print()

        print(
            "❌ Keine Ergebnisse erzeugt."
        )

        return

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = create_summary(
        all_results
    )

    variant_summary = (
        create_variant_summary(
            summary_df
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_path = os.path.join(
        LOG_DIR,
        "v6_multi_asset_results.csv",
    )

    summary_path = os.path.join(
        LOG_DIR,
        "v6_multi_asset_summary.csv",
    )

    variant_path = os.path.join(
        LOG_DIR,
        "v6_multi_asset_variant_summary.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    variant_summary.to_csv(
        variant_path,
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_summary(
        summary_df,
        variant_summary,
    )

    # ========================================================
    # FILE OUTPUT
    # ========================================================

    print()

    print(
        "💾 Ergebnisse gespeichert:"
    )

    print(
        f"   {results_path}"
    )

    print(
        f"   {summary_path}"
    )

    print(
        f"   {variant_path}"
    )

    print()

    print(
        "✅ V6 Multi-Asset Backtest abgeschlossen."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
