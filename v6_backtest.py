# ==========================================
# 20to100 Trading Bot
# V6 Walk-Forward Backtest
#
# ETH/USDT
# 5m -> 1h
#
# 12 Monate TRAIN
# 3 Monate OOS
#
# Varianten:
# V6_A - ADX Rising
# V6_B - ADX Rising + Stronger Slope
# V6_C - ADX Rising + Stronger Slope + Volatility
# ==========================================

from pathlib import Path

import pandas as pd

from strategy.strategy_v6 import calculate_indicators
from backtest.v6_engine import V6BacktestEngine


# ==========================================
# CONFIG
# ==========================================

SYMBOL = "ETH/USDT"

STARTING_BALANCE = 20.0

RISK_PER_TRADE = 0.01

FEE_RATE = 0.001

SLIPPAGE_RATE = 0.0005

ATR_STOP_MULTIPLIER = 3.0

TRAILING_ATR_MULTIPLIER = 3.0

ADX_MIN = 20.0

MAX_DAILY_LOSS = 0.05

MAX_CONSECUTIVE_LOSSES = 3

LOSS_COOLDOWN_BARS = 24

GLOBAL_MAX_DRAWDOWN = 0.20

TRAIN_MONTHS = 12

OOS_MONTHS = 3


VARIANTS = {
    "V6_A": {
        "name": "ADX_RISING",
    },
    "V6_B": {
        "name": "ADX_RISING_STRONG_SLOPE",
    },
    "V6_C": {
        "name": "ADX_RISING_STRONG_SLOPE_VOLATILITY",
    },
}


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FILE = (
    LOG_DIR
    /
    "v6_walk_forward_results.csv"
)

SUMMARY_FILE = (
    LOG_DIR
    /
    "v6_walk_forward_summary.csv"
)


# ==========================================
# DATA LOADER
# ==========================================

def load_data():

    print()
    print("=" * 70)
    print("V6 WALK-FORWARD BACKTEST")
    print("=" * 70)
    print()

    # ======================================
    # Search possible files
    # ======================================

    possible_files = [
        DATA_DIR / "ETH_USDT_5m.csv",
        DATA_DIR / "ETH-USDT-5m.csv",
        DATA_DIR / "ETHUSDT_5m.csv",
        DATA_DIR / "eth_usdt_5m.csv",
        DATA_DIR / "eth_5m.csv",
        DATA_DIR / "ETH_USDT.csv",
    ]

    data_file = None

    for file in possible_files:

        if file.exists():

            data_file = file
            break

    if data_file is None:

        csv_files = list(
            DATA_DIR.glob("*.csv")
        )

        if not csv_files:

            raise FileNotFoundError(
                "Keine CSV-Datei im data/ "
                "Verzeichnis gefunden."
            )

        # Try to find ETH data
        eth_files = [
            f for f in csv_files
            if "eth" in f.name.lower()
        ]

        if eth_files:

            data_file = eth_files[0]

        else:

            raise FileNotFoundError(
                "Keine ETH-CSV-Datei gefunden."
            )

    print(
        f"📂 Daten: {data_file}"
    )

    df = pd.read_csv(
        data_file
    )

    # ======================================
    # Normalize columns
    # ======================================

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    # ======================================
    # Timestamp
    # ======================================

    timestamp_candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
    ]

    timestamp_column = None

    for column in timestamp_candidates:

        if column in df.columns:

            timestamp_column = column
            break

    if timestamp_column is None:

        raise ValueError(
            "Keine Timestamp-Spalte gefunden."
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        utc=True
    )

    df = df.set_index(
        timestamp_column
    )

    # ======================================
    # Required OHLC
    # ======================================

    required = [
        "open",
        "high",
        "low",
        "close",
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

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ======================================
    # Remove invalid rows
    # ======================================

    df = df.dropna(
        subset=required
    )

    # ======================================
    # Sort
    # ======================================

    df = df.sort_index()

    # ======================================
    # Remove duplicates
    # ======================================

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    print(
        f"📊 5m Candles: {len(df):,}"
    )

    print(
        f"📅 Start: {df.index[0]}"
    )

    print(
        f"📅 Ende : {df.index[-1]}"
    )

    return df


# ==========================================
# RESAMPLE 5m -> 1h
# ==========================================

def resample_to_1h(
    df
):

    print()
    print(
        "🔄 Resample 5m → 1h ..."
    )

    hourly = (
        df[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )
        .dropna()
    )

    print(
        f"📊 1h Candles: {len(hourly):,}"
    )

    print(
        f"📅 Start: {hourly.index[0]}"
    )

    print(
        f"📅 Ende : {hourly.index[-1]}"
    )

    return hourly


# ==========================================
# MONTH STARTS
# ==========================================

def month_start(
    timestamp
):

    timestamp = pd.Timestamp(
        timestamp
    )

    if timestamp.tzinfo is None:

        timestamp = timestamp.tz_localize(
            "UTC"
        )

    else:

        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return timestamp.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )


# ==========================================
# ADD MONTHS
# ==========================================

def add_months(
    timestamp,
    months
):

    timestamp = pd.Timestamp(
        timestamp
    )

    if timestamp.tzinfo is None:

        timestamp = timestamp.tz_localize(
            "UTC"
        )

    else:

        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return (
        timestamp
        +
        pd.DateOffset(
            months=months
        )
    )


# ==========================================
# WALK-FORWARD WINDOWS
# ==========================================

def create_windows(
    data
):

    start = month_start(
        data.index[0]
    )

    end = month_start(
        data.index[-1]
    )

    windows = []

    train_start = start

    while True:

        train_end = add_months(
            train_start,
            TRAIN_MONTHS
        )

        oos_end = add_months(
            train_end,
            OOS_MONTHS
        )

        if oos_end > end:

            break

        windows.append(
            {
                "train_start":
                    train_start,

                "train_end":
                    train_end,

                "oos_start":
                    train_end,

                "oos_end":
                    oos_end,
            }
        )

        train_start = add_months(
            train_start,
            OOS_MONTHS
        )

    return windows


# ==========================================
# PREPARE FULL DATA
# ==========================================

def prepare_data(
    data
):

    print()
    print(
        "🧮 Berechne V6-Indikatoren "
        "auf dem kompletten Datensatz ..."
    )

    # ======================================
    # IMPORTANT
    #
    # Indicators are calculated BEFORE
    # train/OOS splitting.
    #
    # This preserves warm-up history.
    # ======================================

    data = calculate_indicators(
        data
    )

    return data


# ==========================================
# RUN ONE WINDOW
# ==========================================

def run_window(
    data,
    window,
    variant
):

    train_start = window[
        "train_start"
    ]

    train_end = window[
        "train_end"
    ]

    oos_start = window[
        "oos_start"
    ]

    oos_end = window[
        "oos_end"
    ]

    # ======================================
    # TRAIN
    #
    # Included for diagnostics.
    # V6 parameters are intentionally fixed.
    # ======================================

    train = data[
        (
            data.index >= train_start
        )
        &
        (
            data.index < train_end
        )
    ].copy()

    # ======================================
    # OOS
    # ======================================

    oos = data[
        (
            data.index >= oos_start
        )
        &
        (
            data.index < oos_end
        )
    ].copy()

    if len(oos) < 10:

        return None

    # ======================================
    # Engine
    # ======================================

    engine = V6BacktestEngine(

        starting_balance=STARTING_BALANCE,

        risk_per_trade=RISK_PER_TRADE,

        fee_rate=FEE_RATE,

        slippage_rate=SLIPPAGE_RATE,

        atr_stop_multiplier=(
            ATR_STOP_MULTIPLIER
        ),

        trailing_atr_multiplier=(
            TRAILING_ATR_MULTIPLIER
        ),

        adx_min=ADX_MIN,

        max_daily_loss=(
            MAX_DAILY_LOSS
        ),

        max_consecutive_losses=(
            MAX_CONSECUTIVE_LOSSES
        ),

        loss_cooldown_bars=(
            LOSS_COOLDOWN_BARS
        ),

        global_max_drawdown=(
            GLOBAL_MAX_DRAWDOWN
        ),

        variant=variant,
    )

    result = engine.run(
        oos
    )

    return {
        "variant": variant,

        "train_start": train_start,

        "train_end": train_end,

        "oos_start": oos_start,

        "oos_end": oos_end,

        "train_candles": len(train),

        "oos_candles": len(oos),

        "final_balance":
            result[
                "final_balance"
            ],

        "profit":
            result[
                "profit"
            ],

        "return_pct":
            result[
                "return_pct"
            ],

        "trades":
            result[
                "trades"
            ],

        "wins":
            result[
                "wins"
            ],

        "losses":
            result[
                "losses"
            ],

        "win_rate":
            result[
                "win_rate"
            ],

        "profit_factor":
            result[
                "profit_factor"
            ],

        "expectancy":
            result[
                "expectancy"
            ],

        "max_drawdown_pct":
            result[
                "max_drawdown_pct"
            ],

        "fees":
            result[
                "fees"
            ],

        "slippage_cost":
            result[
                "slippage_cost"
            ],
    }


# ==========================================
# RUN ALL VARIANTS
# ==========================================

def run_backtest(
    data
):

    windows = create_windows(
        data
    )

    print()
    print(
        f"🧪 Walk-Forward Fenster: "
        f"{len(windows)}"
    )

    print(
        f"📈 Training: "
        f"{TRAIN_MONTHS} Monate"
    )

    print(
        f"🎯 OOS: "
        f"{OOS_MONTHS} Monate"
    )

    print()

    results = []

    for variant in VARIANTS:

        print()
        print(
            "=" * 70
        )

        print(
            f"🚀 TESTE {variant}"
        )

        print(
            VARIANTS[
                variant
            ]["name"]
        )

        print(
            "=" * 70
        )

        for number, window in enumerate(
            windows,
            start=1
        ):

            result = run_window(
                data,
                window,
                variant
            )

            if result is None:

                continue

            results.append(
                result
            )

            pf = result[
                "profit_factor"
            ]

            if pd.isna(pf):

                pf_text = "N/A"

            elif pf == float(
                "inf"
            ):

                pf_text = "INF"

            else:

                pf_text = (
                    f"{pf:.3f}"
                )

            print(
                f"{variant} "
                f"W{number:02d} | "
                f"OOS "
                f"{result['oos_start'].date()} → "
                f"{result['oos_end'].date()} | "
                f"Return "
                f"{result['return_pct']:+.2f}% | "
                f"PF "
                f"{pf_text} | "
                f"Trades "
                f"{result['trades']} | "
                f"DD "
                f"{result['max_drawdown_pct']:.2f}%"
            )

    return pd.DataFrame(
        results
    )


# ==========================================
# SUMMARY
# ==========================================

def create_summary(
    results
):

    summary_rows = []

    for variant in VARIANTS:

        subset = results[
            results["variant"]
            ==
            variant
        ].copy()

        if subset.empty:
            continue

        positive = (
            subset["return_pct"]
            >
            0
        ).sum()

        pf_positive = (
            subset["profit_factor"]
            >
            1
        ).sum()

        total_trades = (
            subset["trades"]
            .sum()
        )

        summary_rows.append(
            {
                "variant":
                    variant,

                "windows":
                    len(subset),

                "positive_windows":
                    int(positive),

                "positive_window_pct":
                    (
                        positive
                        /
                        len(subset)
                        *
                        100
                    ),

                "pf_gt_1_windows":
                    int(pf_positive),

                "pf_gt_1_pct":
                    (
                        pf_positive
                        /
                        len(subset)
                        *
                        100
                    ),

                "avg_return_pct":
                    subset[
                        "return_pct"
                    ].mean(),

                "median_return_pct":
                    subset[
                        "return_pct"
                    ].median(),

                "best_return_pct":
                    subset[
                        "return_pct"
                    ].max(),

                "worst_return_pct":
                    subset[
                        "return_pct"
                    ].min(),

                "avg_profit_factor":
                    subset[
                        "profit_factor"
                    ].replace(
                        float("inf"),
                        pd.NA
                    ).mean(),

                "median_profit_factor":
                    subset[
                        "profit_factor"
                    ].replace(
                        float("inf"),
                        pd.NA
                    ).median(),

                "total_trades":
                    int(total_trades),

                "avg_trades":
                    subset[
                        "trades"
                    ].mean(),

                "avg_win_rate":
                    subset[
                        "win_rate"
                    ].mean(),

                "avg_expectancy":
                    subset[
                        "expectancy"
                    ].mean(),

                "worst_drawdown_pct":
                    subset[
                        "max_drawdown_pct"
                    ].min(),

                "avg_drawdown_pct":
                    subset[
                        "max_drawdown_pct"
                    ].mean(),

                "total_fees":
                    subset[
                        "fees"
                    ].sum(),

                "total_slippage":
                    subset[
                        "slippage_cost"
                    ].sum(),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# ==========================================
# PRINT SUMMARY
# ==========================================

def print_summary(
    summary
):

    print()
    print()
    print("=" * 100)
    print("V6 WALK-FORWARD SUMMARY")
    print("=" * 100)

    for _, row in summary.iterrows():

        print()
        print(
            f"🔹 {row['variant']}"
        )

        print(
            f"   Positive Fenster : "
            f"{int(row['positive_windows'])}/"
            f"{int(row['windows'])} "
            f"({row['positive_window_pct']:.1f}%)"
        )

        print(
            f"   PF > 1           : "
            f"{int(row['pf_gt_1_windows'])}/"
            f"{int(row['windows'])} "
            f"({row['pf_gt_1_pct']:.1f}%)"
        )

        print(
            f"   Ø Return         : "
            f"{row['avg_return_pct']:+.3f}%"
        )

        print(
            f"   Median Return    : "
            f"{row['median_return_pct']:+.3f}%"
        )

        print(
            f"   Ø Profit Factor  : "
            f"{row['avg_profit_factor']:.3f}"
        )

        print(
            f"   Median PF        : "
            f"{row['median_profit_factor']:.3f}"
        )

        print(
            f"   Trades gesamt    : "
            f"{int(row['total_trades'])}"
        )

        print(
            f"   Ø Win Rate       : "
            f"{row['avg_win_rate']:.2f}%"
        )

        print(
            f"   Ø Expectancy     : "
            f"{row['avg_expectancy']:.5f}"
        )

        print(
            f"   Worst DD         : "
            f"{row['worst_drawdown_pct']:.3f}%"
        )

        print(
            f"   Ø DD             : "
            f"{row['avg_drawdown_pct']:.3f}%"
        )

    print()
    print("=" * 100)


# ==========================================
# MAIN
# ==========================================

def main():

    # ======================================
    # Load
    # ======================================

    raw = load_data()

    # ======================================
    # Resample
    # ======================================

    hourly = resample_to_1h(
        raw
    )

    # ======================================
    # Indicators BEFORE split
    # ======================================

    data = prepare_data(
        hourly
    )

    # ======================================
    # Backtest
    # ======================================

    results = run_backtest(
        data
    )

    if results.empty:

        print(
            "❌ Keine Backtest-Ergebnisse."
        )

        return

    # ======================================
    # Save detailed results
    # ======================================

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    print()
    print(
        f"💾 Ergebnisse gespeichert:"
    )

    print(
        RESULT_FILE
    )

    # ======================================
    # Summary
    # ======================================

    summary = create_summary(
        results
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"💾 Summary gespeichert:"
    )

    print(
        SUMMARY_FILE
    )

    # ======================================
    # Display
    # ======================================

    print_summary(
        summary
    )

    print()
    print(
        "✅ V6 BACKTEST ABGESCHLOSSEN"
    )


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()
