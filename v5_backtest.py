# ==========================================
# 20to100 Trading Bot
# V5 Walk-Forward Backtest
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
#
# Nur kleine Sensitivitätsprüfung.
# KEIN aggressives Optimieren.
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
    # IMPORTANT:
    # Always use UTC-aware timestamps.
    # ======================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.set_index(
        "timestamp"
    )

    df = df.sort_index()

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column
        in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Fehlende Spalten: {missing}"
        )

    # ======================================
    # 5m -> 1h
    # ======================================

    hourly = (
        df.resample("1h")
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

    # ======================================
    # IMPORTANT:
    # Keep all timestamps timezone-aware
    # because market data uses UTC.
    # ======================================

    index = pd.DatetimeIndex(index)

    if index.tz is None:

        index = index.tz_localize(
            "UTC"
        )

    else:

        index = index.tz_convert(
            "UTC"
        )

    # ======================================
    # Start of first month
    # ======================================

    start = (
        index.min()
        .to_period("M")
        .to_timestamp()
        .tz_localize("UTC")
    )

    # ======================================
    # End of last month
    # ======================================

    end = (
        index.max()
        .to_period("M")
        .to_timestamp()
        .tz_localize("UTC")
    )

    # ======================================
    # First OOS period starts after
    # TRAIN_MONTHS
    # ======================================

    cursor = (
        start
        + pd.DateOffset(
            months=TRAIN_MONTHS
        )
    )

    # ======================================
    # Generate rolling windows
    # ======================================

    while (
        cursor
        + pd.DateOffset(
            months=OOS_MONTHS
        )
        <= end
    ):

        train_start = (
            cursor
            - pd.DateOffset(
                months=TRAIN_MONTHS
            )
        )

        train_end = cursor

        oos_start = cursor

        oos_end = (
            cursor
            + pd.DateOffset(
                months=OOS_MONTHS
            )
        )

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
# RUN WINDOW
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
    # IMPORTANT:
    #
    # Indicators were calculated BEFORE
    # the train/OOS split.
    #
    # This fixes the V4 warm-up problem.
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
    ]

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
    ]

    if len(train) < 300:

        return None

    if len(oos) < 100:

        return None

    # ======================================
    # TRAIN
    # ======================================

    train_engine = (
        V5BacktestEngine(
            starting_balance=(
                STARTING_BALANCE
            ),
            **params,
        )
    )

    train_result = (
        train_engine.run(
            train
        )
    )

    # ======================================
    # OOS
    # ======================================

    oos_engine = (
        V5BacktestEngine(
            starting_balance=(
                STARTING_BALANCE
            ),
            **params,
        )
    )

    oos_result = (
        oos_engine.run(
            oos
        )
    )

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

    if math.isinf(value):

        return "inf"

    return f"{value:.3f}"


# ==========================================
# SUMMARY
# ==========================================

def summarize(rows):

    df = pd.DataFrame(
        rows
    )

    if df.empty:

        return {}

    positive = (
        df[
            "oos_return_pct"
        ]
        > 0
    ).sum()

    pf_above_1 = (
        df[
            "oos_pf"
        ]
        > 1
    ).sum()

    return {

        "windows":
            len(df),

        "positive_windows":
            int(
                positive
            ),

        "positive_window_pct":
            positive
            /
            len(df)
            *
            100,

        "pf_gt_1_windows":
            int(
                pf_above_1
            ),

        "pf_gt_1_pct":
            pf_above_1
            /
            len(df)
            *
            100,

        "avg_oos_return_pct":
            df[
                "oos_return_pct"
            ].mean(),

        "median_oos_return_pct":
            df[
                "oos_return_pct"
            ].median(),

        "avg_oos_pf":
            df[
                "oos_pf"
            ]
            .replace(
                float("inf"),
                pd.NA
            )
            .mean(),

        "median_oos_pf":
            df[
                "oos_pf"
            ]
            .replace(
                float("inf"),
                pd.NA
            )
            .median(),

        "total_oos_trades":
            int(
                df[
                    "oos_trades"
                ].sum()
            ),

        "worst_oos_dd_pct":
            df[
                "oos_dd_pct"
            ].min(),
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
        "REGIME FILTER + 1% RISK"
    )

    print(
        "=" * 78
    )

    # ======================================
    # LOAD DATA
    # ======================================

    raw = load_eth_1h()

    print(
        f"5m -> 1h candles: "
        f"{len(raw):,}"
    )

    # ======================================
    # CALCULATE INDICATORS ON FULL DATA
    # ======================================

    print(
        "Calculating V5 indicators..."
    )

    data = (
        calculate_indicators(
            raw
        )
    )

    print(
        f"Data range:"
        f" {data.index.min()}"
        f" -> {data.index.max()}"
    )

    print(
        f"Hourly candles:"
        f" {len(data):,}"
    )

    print()

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

    print()

    all_rows = []

    summaries = []

    # ======================================
    # VARIANTS
    # ======================================

    for (
        variant,
        params
    ) in VARIANTS.items():

        print(
            "-" * 78
        )

        print(
            f"{variant}"
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
            "-" * 78
        )

        rows = []

        for number, (
            train_start,
            train_end,
            oos_start,
            oos_end,
        ) in enumerate(
            windows,
            start=1
        ):

            result = run_window(
                data,
                train_start,
                train_end,
                oos_start,
                oos_end,
                params
            )

            if result is None:

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

                f"W{number:02d} | "

                f"OOS "
                f"{oos_start.date()} "
                f"-> "
                f"{oos_end.date()} | "

                f"Return "
                f"{result['oos_return_pct']:+7.2f}% | "

                f"PF "
                f"{fmt_pf(result['oos_pf']):>5} | "

                f"Trades "
                f"{result['oos_trades']:>3} | "

                f"DD "
                f"{result['oos_dd_pct']:+7.2f}%"
            )

        summary = summarize(
            rows
        )

        summary[
            "variant"
        ] = variant

        summaries.append(
            summary
        )

        if summary:

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

        print()

    # ======================================
    # SAVE RESULTS
    # ======================================

    results_df = pd.DataFrame(
        all_rows
    )

    summary_df = pd.DataFrame(
        summaries
    )

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

    print(
        "=" * 78
    )

    print(
        "V5 FINAL SUMMARY"
    )

    print(
        "=" * 78
    )

    if not summary_df.empty:

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
        "V5 is NOT validated merely"
        " because one variant is profitable."
    )

    print(
        "The important metric is robustness"
        " across OOS windows."
    )


if __name__ == "__main__":

    main()
