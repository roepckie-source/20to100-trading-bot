# ==========================================
# 20to100 Trading Bot
# V5 Walk-Forward Backtest
#
# ETH/USDT 1H
#
# 12 Monate TRAIN
# 3 Monate OOS
#
# V5 Varianten:
# A = Base
# B = Strict Regime
# C = Wider Stop
#
# WICHTIG:
# - Indikatoren werden auf dem kompletten Datensatz berechnet
# - Danach erfolgt der Train/OOS Split
# - UTC bleibt erhalten
# - Kein Absturz bei zu wenig Daten
# ==========================================

from pathlib import Path
import math

import pandas as pd

from strategy.strategy_v5 import (
    calculate_indicators
)

from backtest.v5_engine import (
    V5BacktestEngine
)


# ==========================================
# SETTINGS
# ==========================================

STARTING_BALANCE = 20.0

TRAIN_MONTHS = 12
OOS_MONTHS = 3


# ==========================================
# V5 VARIANTS
# ==========================================

VARIANTS = {

    "V5_A_BASE": {

        "adx_min": 20.0,

        "atr_stop_multiplier": 2.5,

        "trailing_atr_multiplier": 2.5,

    },

    "V5_B_STRICT_REGIME": {

        "adx_min": 25.0,

        "atr_stop_multiplier": 2.5,

        "trailing_atr_multiplier": 2.5,

    },

    "V5_C_WIDER_STOP": {

        "adx_min": 20.0,

        "atr_stop_multiplier": 3.0,

        "trailing_atr_multiplier": 3.0,

    },

}


# ==========================================
# LOAD ETH 5m
# ==========================================

def load_eth_1h():

    path = Path(
        "data/ETH_USDT_5m.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Datei fehlt: {path}"
        )

    print(
        f"Using cached data: {path}"
    )

    df = pd.read_csv(
        path
    )

    if "timestamp" not in df.columns:

        raise ValueError(
            "CSV besitzt keine timestamp-Spalte."
        )

    # ======================================
    # UTC
    # ======================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    df = df.set_index(
        "timestamp"
    )

    df = df.sort_index()

    # ======================================
    # REQUIRED COLUMNS
    # ======================================

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Fehlende Spalten: {missing}"
        )

    # ======================================
    # NUMERIC
    # ======================================

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    )

    # ======================================
    # 5m -> 1h
    # ======================================

    hourly = (
        df
        .resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    return hourly


# ==========================================
# WALK-FORWARD WINDOWS
# ==========================================

def month_windows(index):

    index = pd.DatetimeIndex(
        index
    )

    if len(index) == 0:

        return

    # ======================================
    # Make sure timestamps are UTC-aware
    # ======================================

    if index.tz is None:

        index = index.tz_localize(
            "UTC"
        )

    else:

        index = index.tz_convert(
            "UTC"
        )

    # ======================================
    # IMPORTANT:
    #
    # Do NOT use:
    #
    # .to_period("M")
    #
    # because that generates timezone warnings.
    #
    # We construct the first day of the month
    # directly instead.
    # ======================================

    first_timestamp = index.min()

    start = pd.Timestamp(
        year=first_timestamp.year,
        month=first_timestamp.month,
        day=1,
        tz="UTC"
    )

    last_timestamp = index.max()

    end = pd.Timestamp(
        year=last_timestamp.year,
        month=last_timestamp.month,
        day=1,
        tz="UTC"
    )

    # ======================================
    # First OOS starts after TRAIN
    # ======================================

    cursor = (
        start
        + pd.DateOffset(
            months=TRAIN_MONTHS
        )
    )

    # ======================================
    # Generate windows
    # ======================================

    while True:

        oos_end = (
            cursor
            + pd.DateOffset(
                months=OOS_MONTHS
            )
        )

        if oos_end > end:

            break

        train_start = (
            cursor
            - pd.DateOffset(
                months=TRAIN_MONTHS
            )
        )

        train_end = cursor

        oos_start = cursor

        yield (
            train_start,
            train_end,
            oos_start,
            oos_end,
        )

        cursor += (
            pd.DateOffset(
                months=OOS_MONTHS
            )
        )


# ==========================================
# RUN ONE WINDOW
# ==========================================

def run_window(
    full_data,
    train_start,
    train_end,
    oos_start,
    oos_end,
    params
):

    # ======================================
    # TRAIN
    # ======================================

    train = full_data[
        (
            full_data.index
            >= train_start
        )
        &
        (
            full_data.index
            < train_end
        )
    ].copy()

    # ======================================
    # OOS
    # ======================================

    oos = full_data[
        (
            full_data.index
            >= oos_start
        )
        &
        (
            full_data.index
            < oos_end
        )
    ].copy()

    # ======================================
    # DATA VALIDATION
    # ======================================

    if len(train) < 300:

        return None

    if len(oos) < 100:

        return None

    # ======================================
    # TRAIN ENGINE
    # ======================================

    train_engine = V5BacktestEngine(

        starting_balance=
            STARTING_BALANCE,

        **params,
    )

    train_result = (
        train_engine.run(
            train
        )
    )

    # ======================================
    # OOS ENGINE
    # ======================================

    oos_engine = V5BacktestEngine(

        starting_balance=
            STARTING_BALANCE,

        **params,
    )

    oos_result = (
        oos_engine.run(
            oos
        )
    )

    # ======================================
    # RESULT
    # ======================================

    return {

        "train_start":
            train_start,

        "train_end":
            train_end,

        "oos_start":
            oos_start,

        "oos_end":
            oos_end,

        "train_return_pct":
            train_result[
                "return_pct"
            ],

        "train_pf":
            train_result[
                "profit_factor"
            ],

        "train_trades":
            train_result[
                "trades"
            ],

        "train_dd_pct":
            train_result[
                "max_drawdown_pct"
            ],

        "oos_return_pct":
            oos_result[
                "return_pct"
            ],

        "oos_pf":
            oos_result[
                "profit_factor"
            ],

        "oos_trades":
            oos_result[
                "trades"
            ],

        "oos_win_rate":
            oos_result[
                "win_rate"
            ],

        "oos_dd_pct":
            oos_result[
                "max_drawdown_pct"
            ],

        "oos_fees":
            oos_result[
                "fees"
            ],

        "oos_slippage":
            oos_result[
                "slippage_cost"
            ],
    }


# ==========================================
# FORMAT PF
# ==========================================

def fmt_pf(value):

    try:

        if math.isinf(
            float(value)
        ):

            return "inf"

    except Exception:

        pass

    return f"{float(value):.3f}"


# ==========================================
# SUMMARY
# ==========================================

def summarize(rows):

    if not rows:

        return None

    df = pd.DataFrame(
        rows
    )

    if df.empty:

        return None

    # ======================================
    # Numeric conversion
    # ======================================

    oos_returns = pd.to_numeric(
        df["oos_return_pct"],
        errors="coerce"
    )

    oos_pf = pd.to_numeric(
        df["oos_pf"],
        errors="coerce"
    )

    oos_pf_clean = (
        oos_pf
        .replace(
            [float("inf"), float("-inf")],
            pd.NA
        )
    )

    positive = (
        oos_returns > 0
    ).sum()

    pf_above_1 = (
        oos_pf > 1
    ).sum()

    return {

        "windows":
            len(df),

        "positive_windows":
            int(
                positive
            ),

        "positive_window_pct":
            (
                positive
                /
                len(df)
                *
                100
            ),

        "pf_gt_1_windows":
            int(
                pf_above_1
            ),

        "pf_gt_1_pct":
            (
                pf_above_1
                /
                len(df)
                *
                100
            ),

        "avg_oos_return_pct":
            oos_returns.mean(),

        "median_oos_return_pct":
            oos_returns.median(),

        "avg_oos_pf":
            oos_pf_clean.mean(),

        "median_oos_pf":
            oos_pf_clean.median(),

        "total_oos_trades":
            int(
                pd.to_numeric(
                    df["oos_trades"],
                    errors="coerce"
                )
                .fillna(0)
                .sum()
            ),

        "worst_oos_dd_pct":
            pd.to_numeric(
                df["oos_dd_pct"],
                errors="coerce"
            ).min(),

    }


# ==========================================
# MAIN
# ==========================================

def main():

    Path(
        "logs"
    ).mkdir(
        exist_ok=True
    )

    print()
    print(
        "=" * 78
    )

    print(
        "20→100 TRADING BOT"
    )

    print(
        "V5 WALK-FORWARD BACKTEST"
    )

    print(
        "ETH/USDT 1H"
    )

    print(
        "REGIME FILTER + RISK MANAGEMENT"
    )

    print(
        "=" * 78
    )

    # ======================================
    # LOAD
    # ======================================

    raw = load_eth_1h()

    print()
    print(
        f"5m -> 1h candles: "
        f"{len(raw):,}"
    )

    if raw.empty:

        raise RuntimeError(
            "ETH-Datensatz ist leer."
        )

    # ======================================
    # INDICATORS
    #
    # IMPORTANT:
    # Full dataset first.
    # ======================================

    print()
    print(
        "Calculating V5 indicators..."
    )

    data = calculate_indicators(
        raw
    )

    print()
    print(
        f"Data range:"
        f" {data.index.min()}"
        f" -> {data.index.max()}"
    )

    print(
        f"Hourly candles:"
        f" {len(data):,}"
    )

    # ======================================
    # CONFIG
    # ======================================

    print()
    print(
        "Starting balance: $20.00"
    )

    print(
        "Risk per trade: 1.00%"
    )

    print(
        "Fee: 0.10%"
    )

    print(
        "Slippage: 0.05%"
    )

    print(
        "Daily loss limit: 5%"
    )

    print(
        "Global max drawdown: 20%"
    )

    print()

    # ======================================
    # WINDOWS
    # ======================================

    windows = list(
        month_windows(
            data.index
        )
    )

    print(
        f"Walk-forward windows:"
        f" {len(windows)}"
    )

    print(
        f"TRAIN:"
        f" {TRAIN_MONTHS} Monate"
    )

    print(
        f"OOS:"
        f" {OOS_MONTHS} Monate"
    )

    # ======================================
    # IMPORTANT:
    # Too little historical data
    # ======================================

    if len(windows) == 0:

        print()
        print(
            "=" * 78
        )

        print(
            "⚠️ KEINE WALK-FORWARD-FENSTER"
        )

        print(
            "=" * 78
        )

        print()
        print(
            "Der vorhandene ETH-Datensatz"
        )

        print(
            "ist zu kurz für:"
        )

        print(
            f"TRAIN = {TRAIN_MONTHS} Monate"
        )

        print(
            f"OOS   = {OOS_MONTHS} Monate"
        )

        print()
        print(
            "Aktueller Zeitraum:"
        )

        print(
            f"{data.index.min()}"
            f" -> "
            f"{data.index.max()}"
        )

        print()
        print(
            "V5 wird deshalb sauber beendet,"
        )

        print(
            "ohne den gesamten GitHub-Workflow"
        )

        print(
            "abzubrechen."
        )

        # ==================================
        # Write empty result files
        # ==================================

        empty_results = pd.DataFrame()

        empty_summary = pd.DataFrame()

        empty_results.to_csv(
            Path("logs")
            /
            "v5_walk_forward_results.csv",
            index=False
        )

        empty_summary.to_csv(
            Path("logs")
            /
            "v5_walk_forward_summary.csv",
            index=False
        )

        print()
        print(
            "Leere V5-Ergebnisdateien wurden"
        )

        print(
            "erzeugt."
        )

        print()
        print(
            "V5 CHECK COMPLETE"
        )

        return

    # ======================================
    # RESULTS
    # ======================================

    all_rows = []

    summaries = []

    # ======================================
    # VARIANTS
    # ======================================

    for (
        variant,
        params
    ) in VARIANTS.items():

        print()
        print(
            "-" * 78
        )

        print(
            variant
        )

        print(
            f"ADX >= "
            f"{params['adx_min']}"
        )

        print(
            f"ATR Stop = "
            f"{params['atr_stop_multiplier']}"
        )

        print(
            f"ATR Trail = "
            f"{params['trailing_atr_multiplier']}"
        )

        print(
            "-" * 78
        )

        rows = []

        # ==================================
        # WINDOWS
        # ==================================

        for number, (
            train_start,
            train_end,
            oos_start,
            oos_end,
        ) in enumerate(
            windows,
            start=1
        ):

            print(
                f"W{number:02d} | "
                f"OOS "
                f"{oos_start.date()} "
                f"-> "
                f"{oos_end.date()}",
                end=" "
            )

            try:

                result = run_window(

                    data,

                    train_start,

                    train_end,

                    oos_start,

                    oos_end,

                    params

                )

            except Exception as exc:

                print()
                print(
                    f"❌ Fehler in W{number:02d}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                continue

            if result is None:

                print(
                    "| übersprungen"
                )

                continue

            result[
                "variant"
            ] = variant

            result[
                "window"
            ] = number

            rows.append(
                result
            )

            all_rows.append(
                result
            )

            print(
                f"| Return "
                f"{result['oos_return_pct']:+7.2f}% "
                f"| PF "
                f"{fmt_pf(result['oos_pf']):>5} "
                f"| Trades "
                f"{result['oos_trades']:>3} "
                f"| DD "
                f"{result['oos_dd_pct']:+7.2f}%"
            )

        # ==================================
        # SUMMARY
        # ==================================

        summary = summarize(
            rows
        )

        # ==================================
        # FIX:
        # Only append a real summary.
        # ==================================

        if summary is None:

            print()
            print(
                f"SUMMARY {variant}"
            )

            print(
                "Keine gültigen Fenster."
            )

            continue

        summary[
            "variant"
        ] = variant

        summaries.append(
            summary
        )

        # ==================================
        # PRINT SUMMARY
        # ==================================

        print()
        print(
            f"SUMMARY {variant}"
        )

        print(
            f"Positive windows: "
            f"{summary['positive_windows']}"
            f"/"
            f"{summary['windows']}"
            f" "
            f"("
            f"{summary['positive_window_pct']:.1f}"
            f"%)"
        )

        print(
            f"PF > 1 windows: "
            f"{summary['pf_gt_1_windows']}"
            f"/"
            f"{summary['windows']}"
            f" "
            f"("
            f"{summary['pf_gt_1_pct']:.1f}"
            f"%)"
        )

        print(
            f"Average OOS return: "
            f"{summary['avg_oos_return_pct']:+.2f}%"
        )

        print(
            f"Median OOS return: "
            f"{summary['median_oos_return_pct']:+.2f}%"
        )

        print(
            f"Median OOS PF: "
            f"{summary['median_oos_pf']:.3f}"
        )

        print(
            f"OOS trades: "
            f"{summary['total_oos_trades']}"
        )

        print(
            f"Worst OOS DD: "
            f"{summary['worst_oos_dd_pct']:.2f}%"
        )

    # ======================================
    # DATAFRAMES
    # ======================================

    results_df = pd.DataFrame(
        all_rows
    )

    summary_df = pd.DataFrame(
        summaries
    )

    # ======================================
    # FILE PATHS
    # ======================================

    results_path = (
        Path("logs")
        /
        "v5_walk_forward_results.csv"
    )

    summary_path = (
        Path("logs")
        /
        "v5_walk_forward_summary.csv"
    )

    # ======================================
    # SAVE
    # ======================================

    results_df.to_csv(
        results_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ======================================
    # FINAL SUMMARY
    # ======================================

    print()
    print(
        "=" * 78
    )

    print(
        "V5 FINAL SUMMARY"
    )

    print(
        "=" * 78
    )

    if summary_df.empty:

        print(
            "Keine gültigen V5-Ergebnisse."
        )

    else:

        columns = [

            "variant",

            "windows",

            "positive_windows",

            "positive_window_pct",

            "pf_gt_1_windows",

            "pf_gt_1_pct",

            "avg_oos_return_pct",

            "median_oos_return_pct",

            "median_oos_pf",

            "total_oos_trades",

            "worst_oos_dd_pct",

        ]

        print(
            summary_df[
                columns
            ].to_string(
                index=False,
                float_format=(
                    lambda x:
                    f"{x:.3f}"
                )
            )
        )

    # ======================================
    # FILES
    # ======================================

    print()
    print(
        f"Results saved:"
        f" {results_path}"
    )

    print(
        f"Summary saved:"
        f" {summary_path}"
    )

    print()
    print(
        "V5 WALK-FORWARD COMPLETE"
    )


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()
