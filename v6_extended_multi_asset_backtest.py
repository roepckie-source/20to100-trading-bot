# ==========================================
# 20to100 Trading Bot
# V6 EXTENDED MULTI-ASSET WALK-FORWARD
#
# BTC + ETH + SOL
#
# 5 Jahre historische Daten
# 12 Monate TRAIN
# 3 Monate OOS
# Rolling alle 3 Monate
#
# V6_A
# V6_B
# V6_C
#
# WICHTIG:
# - Keine Parameteroptimierung
# - V6 Parameter bleiben fix
# - Indikatoren werden VOR dem Split berechnet
# - OOS ist vollständig out-of-sample
# ==========================================

from pathlib import Path
import math

import pandas as pd

from strategy.strategy_v6 import calculate_indicators
from backtest.v6_engine import V6BacktestEngine


# ==========================================
# CONFIG
# ==========================================

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


# ==========================================
# ASSETS
# ==========================================

ASSETS = {
    "BTC/USDT": "BTC_USDT_5m.csv",
    "ETH/USDT": "ETH_USDT_5m.csv",
    "SOL/USDT": "SOL_USDT_5m.csv",
}


# ==========================================
# VARIANTS
# ==========================================

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
    LOG_DIR /
    "v6_extended_multi_asset_results.csv"
)


SUMMARY_FILE = (
    LOG_DIR /
    "v6_extended_multi_asset_summary.csv"
)


ASSET_SUMMARY_FILE = (
    LOG_DIR /
    "v6_extended_asset_summary.csv"
)


# ==========================================
# FIND CSV
# ==========================================

def find_data_file(filename, symbol):

    direct = DATA_DIR / filename

    if direct.exists():
        return direct

    symbol_lower = symbol.split("/")[0].lower()

    candidates = list(
        DATA_DIR.glob("*.csv")
    )

    matches = [
        file
        for file in candidates
        if symbol_lower in file.name.lower()
    ]

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Keine CSV-Datei für {symbol} gefunden."
    )


# ==========================================
# LOAD DATA
# ==========================================

def load_asset(symbol, filename):

    print()
    print("=" * 90)
    print(f"📂 LOADING {symbol}")
    print("=" * 90)

    data_file = find_data_file(
        filename,
        symbol
    )

    print(
        f"Datei: {data_file}"
    )

    df = pd.read_csv(
        data_file
    )

    # ======================================
    # NORMALIZE COLUMNS
    # ======================================

    df.columns = [
        str(column)
        .strip()
        .lower()
        for column in df.columns
    ]

    # ======================================
    # TIMESTAMP
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
            f"{symbol}: Keine Timestamp-Spalte gefunden."
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=[timestamp_column]
    )

    df = df.set_index(
        timestamp_column
    )

    # ======================================
    # OHLC
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
            f"{symbol}: Fehlende Spalten: {missing}"
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
    # SORT
    # ======================================

    df = df.sort_index()

    # ======================================
    # REMOVE DUPLICATES
    # ======================================

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    print(
        f"5m Candles : {len(df):,}"
    )

    print(
        f"Start      : {df.index[0]}"
    )

    print(
        f"Ende       : {df.index[-1]}"
    )

    return df


# ==========================================
# RESAMPLE 5m -> 1h
# ==========================================

def resample_to_1h(df):

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

    return hourly


# ==========================================
# MONTH START
# ==========================================

def month_start(timestamp):

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

def add_months(timestamp, months):

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
# CREATE ROLLING WINDOWS
#
# 12 MONTH TRAIN
# 3 MONTH OOS
# STEP = 3 MONTHS
# ==========================================

def create_windows(data):

    start = month_start(
        data.index[0]
    )

    end = month_start(
        data.index[-1]
    )

    windows = []

    train_start = start

    window_number = 1

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
                "window":
                    window_number,

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

        window_number += 1

        # ==================================
        # IMPORTANT:
        # MOVE 3 MONTHS
        # ==================================

        train_start = add_months(
            train_start,
            OOS_MONTHS
        )

    return windows


# ==========================================
# PREPARE INDICATORS
#
# IMPORTANT:
# COMPLETE DATASET FIRST
# ==========================================

def prepare_data(data):

    print()
    print(
        "🧮 Berechne V6-Indikatoren "
        "auf dem kompletten Datensatz..."
    )

    prepared = calculate_indicators(
        data
    )

    return prepared


# ==========================================
# SAFE NUMBER
# ==========================================

def safe_float(value):

    if value is None:
        return float("nan")

    try:

        value = float(value)

        if math.isinf(value):
            return float("nan")

        return value

    except Exception:

        return float("nan")


# ==========================================
# RUN ONE WINDOW
# ==========================================

def run_window(
    data,
    symbol,
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

    window_number = window[
        "window"
    ]

    # ======================================
    # TRAIN
    #
    # Only diagnostic.
    #
    # NO parameter fitting.
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

    if len(oos) < 300:

        print(
            f"⚠️ {symbol} {variant} "
            f"W{window_number:02d}: "
            f"zu wenig OOS-Daten."
        )

        return None

    # ======================================
    # ENGINE
    # ======================================

    engine = V6BacktestEngine(

        starting_balance=
            STARTING_BALANCE,

        risk_per_trade=
            RISK_PER_TRADE,

        fee_rate=
            FEE_RATE,

        slippage_rate=
            SLIPPAGE_RATE,

        atr_stop_multiplier=
            ATR_STOP_MULTIPLIER,

        trailing_atr_multiplier=
            TRAILING_ATR_MULTIPLIER,

        adx_min=
            ADX_MIN,

        max_daily_loss=
            MAX_DAILY_LOSS,

        max_consecutive_losses=
            MAX_CONSECUTIVE_LOSSES,

        loss_cooldown_bars=
            LOSS_COOLDOWN_BARS,

        global_max_drawdown=
            GLOBAL_MAX_DRAWDOWN,

        variant=
            variant,
    )

    # ======================================
    # RUN
    # ======================================

    result = engine.run(
        oos
    )

    return {

        "symbol":
            symbol,

        "variant":
            variant,

        "window":
            window_number,

        "train_start":
            train_start,

        "train_end":
            train_end,

        "oos_start":
            oos_start,

        "oos_end":
            oos_end,

        "train_candles":
            len(train),

        "oos_candles":
            len(oos),

        "final_balance":
            safe_float(
                result.get(
                    "final_balance"
                )
            ),

        "profit":
            safe_float(
                result.get(
                    "profit"
                )
            ),

        "return_pct":
            safe_float(
                result.get(
                    "return_pct"
                )
            ),

        "trades":
            int(
                result.get(
                    "trades",
                    0
                )
            ),

        "wins":
            int(
                result.get(
                    "wins",
                    0
                )
            ),

        "losses":
            int(
                result.get(
                    "losses",
                    0
                )
            ),

        "win_rate":
            safe_float(
                result.get(
                    "win_rate"
                )
            ),

        "profit_factor":
            safe_float(
                result.get(
                    "profit_factor"
                )
            ),

        "expectancy":
            safe_float(
                result.get(
                    "expectancy"
                )
            ),

        "max_drawdown_pct":
            safe_float(
                result.get(
                    "max_drawdown_pct"
                )
            ),

        "fees":
            safe_float(
                result.get(
                    "fees"
                )
            ),

        "slippage_cost":
            safe_float(
                result.get(
                    "slippage_cost"
                )
            ),
    }


# ==========================================
# PRINT RESULT
# ==========================================

def print_window_result(result):

    pf = result[
        "profit_factor"
    ]

    if pd.isna(pf):

        pf_text = "N/A"

    elif math.isinf(pf):

        pf_text = "INF"

    else:

        pf_text = f"{pf:.3f}"

    print(
        f"{result['symbol']:9s} "
        f"{result['variant']:5s} "
        f"W{int(result['window']):02d} | "
        f"OOS "
        f"{pd.Timestamp(result['oos_start']).date()} → "
        f"{pd.Timestamp(result['oos_end']).date()} | "
        f"Return "
        f"{result['return_pct']:+.2f}% | "
        f"PF "
        f"{pf_text:>6s} | "
        f"Trades "
        f"{result['trades']:3d} | "
        f"Win "
        f"{result['win_rate']:.1f}% | "
        f"DD "
        f"{result['max_drawdown_pct']:.2f}%"
    )


# ==========================================
# ASSET / VARIANT SUMMARY
# ==========================================

def create_asset_summary(results):

    rows = []

    for (
        symbol,
        variant
    ), subset in results.groupby(
        [
            "symbol",
            "variant"
        ]
    ):

        subset = subset.copy()

        windows = len(
            subset
        )

        positive_windows = int(
            (
                subset[
                    "return_pct"
                ]
                > 0
            ).sum()
        )

        pf_gt_1_windows = int(
            (
                subset[
                    "profit_factor"
                ]
                > 1
            ).sum()
        )

        rows.append({

            "symbol":
                symbol,

            "variant":
                variant,

            "windows":
                windows,

            "positive_windows":
                positive_windows,

            "positive_window_pct":
                (
                    positive_windows
                    /
                    windows
                    *
                    100
                ),

            "pf_gt_1_windows":
                pf_gt_1_windows,

            "pf_gt_1_pct":
                (
                    pf_gt_1_windows
                    /
                    windows
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
                ].mean(),

            "median_profit_factor":
                subset[
                    "profit_factor"
                ].median(),

            "total_trades":
                int(
                    subset[
                        "trades"
                    ].sum()
                ),

            "avg_trades_per_window":
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
        })

    return pd.DataFrame(
        rows
    )


# ==========================================
# GLOBAL VARIANT SUMMARY
# ==========================================

def create_global_summary(
    asset_summary
):

    rows = []

    for variant in VARIANTS:

        subset = asset_summary[
            asset_summary[
                "variant"
            ]
            ==
            variant
        ].copy()

        if subset.empty:
            continue

        rows.append({

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

            "median_profit_factor":
                subset[
                    "median_profit_factor"
                ].median(),

            "avg_positive_window_pct":
                subset[
                    "positive_window_pct"
                ].mean(),

            "avg_pf_gt_1_pct":
                subset[
                    "pf_gt_1_pct"
                ].mean(),

            "total_trades":
                int(
                    subset[
                        "total_trades"
                    ].sum()
                ),

            "worst_asset_return_pct":
                subset[
                    "worst_return_pct"
                ].min(),

            "best_asset_return_pct":
                subset[
                    "best_return_pct"
                ].max(),

            "worst_asset_drawdown_pct":
                subset[
                    "worst_drawdown_pct"
                ].min(),
        })

    return pd.DataFrame(
        rows
    )


# ==========================================
# PRINT ASSET SUMMARY
# ==========================================

def print_asset_summary(
    summary
):

    print()
    print()
    print("=" * 120)
    print("V6 EXTENDED ASSET SUMMARY")
    print("=" * 120)

    for _, row in summary.iterrows():

        print()

        print(
            f"🔹 {row['symbol']} "
            f"{row['variant']}"
        )

        print(
            f"   Fenster              : "
            f"{int(row['windows'])}"
        )

        print(
            f"   Positive Fenster     : "
            f"{int(row['positive_windows'])}/"
            f"{int(row['windows'])} "
            f"({row['positive_window_pct']:.1f}%)"
        )

        print(
            f"   PF > 1               : "
            f"{int(row['pf_gt_1_windows'])}/"
            f"{int(row['windows'])} "
            f"({row['pf_gt_1_pct']:.1f}%)"
        )

        print(
            f"   Ø Return             : "
            f"{row['avg_return_pct']:+.3f}%"
        )

        print(
            f"   Median Return        : "
            f"{row['median_return_pct']:+.3f}%"
        )

        print(
            f"   Beste OOS-Periode    : "
            f"{row['best_return_pct']:+.3f}%"
        )

        print(
            f"   Schlechteste OOS     : "
            f"{row['worst_return_pct']:+.3f}%"
        )

        print(
            f"   Ø Profit Factor      : "
            f"{row['avg_profit_factor']:.3f}"
        )

        print(
            f"   Median PF            : "
            f"{row['median_profit_factor']:.3f}"
        )

        print(
            f"   Trades gesamt        : "
            f"{int(row['total_trades'])}"
        )

        print(
            f"   Ø Trades/Fenster     : "
            f"{row['avg_trades_per_window']:.2f}"
        )

        print(
            f"   Ø Win Rate           : "
            f"{row['avg_win_rate']:.2f}%"
        )

        print(
            f"   Schlechtester DD     : "
            f"{row['worst_drawdown_pct']:.2f}%"
        )


# ==========================================
# PRINT GLOBAL SUMMARY
# ==========================================

def print_global_summary(
    summary
):

    print()
    print()
    print("=" * 120)
    print("🏆 V6 EXTENDED MULTI-ASSET SUMMARY")
    print("=" * 120)

    for _, row in summary.iterrows():

        print()

        print(
            f"🔹 {row['variant']}"
        )

        print(
            f"   Assets getestet      : "
            f"{int(row['assets_tested'])}"
        )

        print(
            f"   Ø Asset Return       : "
            f"{row['avg_asset_return_pct']:+.3f}%"
        )

        print(
            f"   Median Asset Return  : "
            f"{row['median_asset_return_pct']:+.3f}%"
        )

        print(
            f"   Ø Profit Factor      : "
            f"{row['avg_profit_factor']:.3f}"
        )

        print(
            f"   Median Profit Factor : "
            f"{row['median_profit_factor']:.3f}"
        )

        print(
            f"   Ø positive Fenster   : "
            f"{row['avg_positive_window_pct']:.1f}%"
        )

        print(
            f"   Ø PF > 1 Fenster     : "
            f"{row['avg_pf_gt_1_pct']:.1f}%"
        )

        print(
            f"   Trades gesamt        : "
            f"{int(row['total_trades'])}"
        )

        print(
            f"   Beste Asset-Periode  : "
            f"{row['best_asset_return_pct']:+.3f}%"
        )

        print(
            f"   Schlechteste Asset- : "
            f"{row['worst_asset_return_pct']:+.3f}%"
        )

        print(
            f"   Schlechtester DD     : "
            f"{row['worst_asset_drawdown_pct']:.2f}%"
        )


# ==========================================
# MAIN
# ==========================================

def main():

    print()
    print("=" * 120)
    print("20→100 TRADING BOT")
    print("V6 EXTENDED MULTI-ASSET WALK-FORWARD")
    print("=" * 120)

    print()
    print(
        "BTC + ETH + SOL"
    )

    print(
        "12 Monate TRAIN / 3 Monate OOS"
    )

    print(
        "Rolling Step: 3 Monate"
    )

    print(
        "V6 Parameter FIX"
    )

    print()
    print(
        f"Starting Balance : ${STARTING_BALANCE:.2f}"
    )

    print(
        f"Risk / Trade     : {RISK_PER_TRADE * 100:.2f}%"
    )

    print(
        f"Fee              : {FEE_RATE * 100:.2f}%"
    )

    print(
        f"Slippage         : {SLIPPAGE_RATE * 100:.2f}%"
    )

    print(
        f"ATR Stop         : {ATR_STOP_MULTIPLIER:.1f}"
    )

    print(
        f"ATR Trailing     : {TRAILING_ATR_MULTIPLIER:.1f}"
    )

    print(
        f"ADX Minimum      : {ADX_MIN:.1f}"
    )

    print(
        f"Daily Loss       : {MAX_DAILY_LOSS * 100:.1f}%"
    )

    print(
        f"Loss Cooldown    : {LOSS_COOLDOWN_BARS} Bars"
    )

    print(
        f"Max Drawdown     : {GLOBAL_MAX_DRAWDOWN * 100:.1f}%"
    )

    all_results = []

    # ======================================
    # EACH ASSET
    # ======================================

    for symbol, filename in ASSETS.items():

        try:

            raw = load_asset(
                symbol,
                filename
            )

            hourly = resample_to_1h(
                raw
            )

            print(
                f"1h Candles  : {len(hourly):,}"
            )

            prepared = prepare_data(
                hourly
            )

            windows = create_windows(
                prepared
            )

            print()

            print(
                f"🧪 {symbol}: "
                f"{len(windows)} Walk-Forward Fenster"
            )

            # ==================================
            # ALL VARIANTS
            # ==================================

            for variant in VARIANTS:

                print()
                print(
                    "=" * 100
                )

                print(
                    f"🚀 {symbol} → {variant}"
                )

                print(
                    VARIANTS[
                        variant
                    ]["name"]
                )

                print(
                    "=" * 100
                )

                for window in windows:

                    result = run_window(
                        prepared,
                        symbol,
                        window,
                        variant
                    )

                    if result is None:
                        continue

                    all_results.append(
                        result
                    )

                    print_window_result(
                        result
                    )

        except Exception as exc:

            print()
            print(
                f"❌ FEHLER bei {symbol}:"
            )

            print(
                repr(exc)
            )

    # ======================================
    # NO RESULTS
    # ======================================

    if not all_results:

        raise RuntimeError(
            "Keine Backtest-Ergebnisse erzeugt."
        )

    # ======================================
    # DATAFRAME
    # ======================================

    results = pd.DataFrame(
        all_results
    )

    # ======================================
    # SORT
    # ======================================

    results = results.sort_values(
        [
            "symbol",
            "variant",
            "window",
        ]
    )

    # ======================================
    # SAVE DETAILED RESULTS
    # ======================================

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    # ======================================
    # ASSET SUMMARY
    # ======================================

    asset_summary = create_asset_summary(
        results
    )

    asset_summary.to_csv(
        ASSET_SUMMARY_FILE,
        index=False
    )

    # ======================================
    # GLOBAL SUMMARY
    # ======================================

    global_summary = create_global_summary(
        asset_summary
    )

    global_summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # ======================================
    # PRINT
    # ======================================

    print_asset_summary(
        asset_summary
    )

    print_global_summary(
        global_summary
    )

    # ======================================
    # WINNER
    # ======================================

    if not global_summary.empty:

        ranking = (
            global_summary
            .sort_values(
                [
                    "median_asset_return_pct",
                    "median_profit_factor",
                    "avg_positive_window_pct",
                ],
                ascending=False
            )
        )

        winner = ranking.iloc[0]

        print()
        print()
        print("=" * 120)
        print("🏆 V6 CURRENT VALIDATION WINNER")
        print("=" * 120)

        print()

        print(
            f"Strategie: {winner['variant']}"
        )

        print(
            f"Ø Asset Return: "
            f"{winner['avg_asset_return_pct']:+.3f}%"
        )

        print(
            f"Median Asset Return: "
            f"{winner['median_asset_return_pct']:+.3f}%"
        )

        print(
            f"Ø Profit Factor: "
            f"{winner['avg_profit_factor']:.3f}"
        )

        print(
            f"Median Profit Factor: "
            f"{winner['median_profit_factor']:.3f}"
        )

        print(
            f"Ø positive OOS-Fenster: "
            f"{winner['avg_positive_window_pct']:.1f}%"
        )

        print(
            f"Ø PF > 1: "
            f"{winner['avg_pf_gt_1_pct']:.1f}%"
        )

        print()

    # ======================================
    # FILES
    # ======================================

    print()
    print("=" * 120)
    print("💾 ERGEBNISDATEIEN")
    print("=" * 120)

    print()
    print(
        f"Details : {RESULT_FILE}"
    )

    print(
        f"Assets  : {ASSET_SUMMARY_FILE}"
    )

    print(
        f"Summary : {SUMMARY_FILE}"
    )

    print()
    print(
        "✅ V6 EXTENDED MULTI-ASSET BACKTEST COMPLETE"
    )

    print()


if __name__ == "__main__":
    main()