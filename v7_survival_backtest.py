# ============================================================
# V7 SURVIVAL BACKTEST
# ============================================================
#
# V7-S0
#
# Entry:
#   V6-C
#
# Survival layer:
#   V7SurvivalEngine
#
# Assets:
#   BTC/USDT
#   ETH/USDT
#   SOL/USDT
#
# Capital scenarios:
#   20 / 50 / 100 / 250 USDT
#
# Method:
#   Rolling OOS
#   12 months training period
#   3 months OOS period
#
# IMPORTANT:
#   V6 strategy remains unchanged.
#   V6 engine remains unchanged.
#
# ============================================================

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

from strategy.strategy_v6 import calculate_indicators

from backtest.v7_survival_engine import (
    V7SurvivalEngine,
)


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = ROOT / "data"

ASSETS = [
    "BTC_USDT_5m",
    "ETH_USDT_5m",
    "SOL_USDT_5m",
]

CAPITALS = [
    20.0,
    50.0,
    100.0,
    250.0,
]

VARIANT = "V6_C"

TRAIN_MONTHS = 12
OOS_MONTHS = 3

RISK_PER_TRADE = 0.01

FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

ATR_STOP_MULTIPLIER = 3.0
TRAILING_ATR_MULTIPLIER = 3.0

ADX_MIN = 20.0

MAX_DAILY_LOSS_PCT = 5.0
MAX_CONSECUTIVE_LOSSES = 3

COOLDOWN_BARS = 24

GLOBAL_MAX_DRAWDOWN_PCT = 20.0


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_FILE = ROOT / "v7_survival_results.csv"
SUMMARY_FILE = ROOT / "v7_survival_summary.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data(asset_name: str) -> pd.DataFrame:

    path = DATA_DIR / f"{asset_name}.csv"

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    print(f"\nLoading {path}")

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:

        raise ValueError(
            f"{asset_name}: missing timestamp column"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["timestamp"]
    ).copy()

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"{asset_name}: missing column {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    ).copy()

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.set_index(
        "timestamp"
    )

    df = df[
        required
    ].copy()

    return df


# ============================================================
# RESAMPLE 5m -> 1h
# ============================================================

def resample_to_1h(
    df: pd.DataFrame,
) -> pd.DataFrame:

    hourly = df.resample(
        "1h"
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )

    hourly = hourly.dropna()

    return hourly


# ============================================================
# PREPARE STRATEGY DATA
# ============================================================

def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    hourly = resample_to_1h(
        df
    )

    if len(hourly) < 1000:

        raise ValueError(
            "Not enough hourly data."
        )

    hourly = calculate_indicators(
        hourly.copy()
    )

    return hourly


# ============================================================
# CREATE ROLLING WINDOWS
# ============================================================

def create_windows(
    df: pd.DataFrame,
):

    train_delta = pd.DateOffset(
        months=TRAIN_MONTHS
    )

    oos_delta = pd.DateOffset(
        months=OOS_MONTHS
    )

    start = df.index.min()

    end = df.index.max()

    current_train_start = start

    windows = []

    while True:

        train_end = (
            current_train_start
            + train_delta
        )

        oos_end = (
            train_end
            + oos_delta
        )

        if oos_end > end:

            break

        train_data = df[
            (df.index >= current_train_start)
            & (df.index < train_end)
        ].copy()

        oos_data = df[
            (df.index >= train_end)
            & (df.index < oos_end)
        ].copy()

        if (
            len(train_data) > 0
            and len(oos_data) > 0
        ):

            windows.append(
                {
                    "train_start": current_train_start,
                    "train_end": train_end,
                    "oos_start": train_end,
                    "oos_end": oos_end,
                    "train": train_data,
                    "oos": oos_data,
                }
            )

        current_train_start = (
            current_train_start
            + oos_delta
        )

    return windows


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(
    value,
    default=np.nan,
):

    try:

        value = float(value)

        if np.isfinite(value):

            return value

    except Exception:

        pass

    return default


# ============================================================
# RUN ONE BACKTEST
# ============================================================

def run_single_backtest(
    oos_data: pd.DataFrame,
    starting_balance: float,
):

    # ========================================================
    # IMPORTANT
    #
    # These argument names MUST match
    # V7SurvivalEngine.__init__()
    #
    # Engine expects:
    #
    #   base_risk_per_trade
    #   max_daily_loss
    #   loss_cooldown_bars
    #   global_max_drawdown
    #
    # ========================================================

    engine = V7SurvivalEngine(

        starting_balance=starting_balance,

        base_risk_per_trade=RISK_PER_TRADE,

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
            MAX_DAILY_LOSS_PCT / 100.0
        ),

        max_consecutive_losses=(
            MAX_CONSECUTIVE_LOSSES
        ),

        loss_cooldown_bars=(
            COOLDOWN_BARS
        ),

        global_max_drawdown=(
            GLOBAL_MAX_DRAWDOWN_PCT / 100.0
        ),

        variant=VARIANT,
    )

    result = engine.run(
        oos_data
    )

    return result


# ============================================================
# EXTRACT RESULT
# ============================================================

def extract_result(
    result,
    asset: str,
    capital: float,
    window_number: int,
    window: dict,
):

    if result is None:

        result = {}

    return {

        "asset": asset,

        "capital": capital,

        "variant": VARIANT,

        "window": window_number,

        "train_start": (
            window["train_start"]
        ),

        "train_end": (
            window["train_end"]
        ),

        "oos_start": (
            window["oos_start"]
        ),

        "oos_end": (
            window["oos_end"]
        ),

        "starting_balance": (
            capital
        ),

        "final_balance": safe_float(
            result.get(
                "final_balance"
            )
        ),

        "profit": safe_float(
            result.get(
                "profit"
            )
        ),

        "return_pct": safe_float(
            result.get(
                "return_pct"
            )
        ),

        "trades": safe_float(
            result.get(
                "trades"
            ),
            0,
        ),

        "wins": safe_float(
            result.get(
                "wins"
            ),
            0,
        ),

        "losses": safe_float(
            result.get(
                "losses"
            ),
            0,
        ),

        # ====================================================
        # FIX:
        #
        # V7SurvivalEngine returns:
        #
        #   "win_rate"
        #
        # not:
        #
        #   "win_rate_pct"
        #
        # ====================================================

        "win_rate_pct": safe_float(
            result.get(
                "win_rate"
            )
        ),

        "profit_factor": safe_float(
            result.get(
                "profit_factor"
            )
        ),

        "expectancy": safe_float(
            result.get(
                "expectancy"
            )
        ),

        "max_drawdown_pct": safe_float(
            result.get(
                "max_drawdown_pct"
            )
        ),

        "fees": safe_float(
            result.get(
                "fees"
            )
        ),

        "slippage_cost": safe_float(
            result.get(
                "slippage_cost"
            )
        ),
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    if results.empty:

        return pd.DataFrame()

    rows = []

    grouped = results.groupby(
        [
            "capital",
            "variant",
        ]
    )

    for (
        capital,
        variant,
    ), group in grouped:

        returns = group[
            "return_pct"
        ].dropna()

        pf = group[
            "profit_factor"
        ].dropna()

        dd = group[
            "max_drawdown_pct"
        ].dropna()

        expectancy = group[
            "expectancy"
        ].dropna()

        trades = group[
            "trades"
        ].fillna(0)

        positive_windows = (
            returns > 0
        ).sum()

        pf_gt_1 = (
            pf > 1
        ).sum()

        total_windows = len(
            group
        )

        rows.append(
            {

                "variant": variant,

                "capital": capital,

                "assets_tested": (
                    group["asset"]
                    .nunique()
                ),

                "windows_tested": (
                    total_windows
                ),

                "avg_return_pct": (
                    returns.mean()
                    if len(returns)
                    else np.nan
                ),

                "median_return_pct": (
                    returns.median()
                    if len(returns)
                    else np.nan
                ),

                "avg_profit_factor": (
                    pf.mean()
                    if len(pf)
                    else np.nan
                ),

                "median_profit_factor": (
                    pf.median()
                    if len(pf)
                    else np.nan
                ),

                "positive_window_pct": (
                    positive_windows
                    / total_windows
                    * 100
                    if total_windows
                    else np.nan
                ),

                "pf_gt_1_pct": (
                    pf_gt_1
                    / total_windows
                    * 100
                    if total_windows
                    else np.nan
                ),

                "avg_expectancy": (
                    expectancy.mean()
                    if len(expectancy)
                    else np.nan
                ),

                "avg_trades": (
                    trades.mean()
                ),

                "total_trades": (
                    trades.sum()
                ),

                "worst_return_pct": (
                    returns.min()
                    if len(returns)
                    else np.nan
                ),

                "best_return_pct": (
                    returns.max()
                    if len(returns)
                    else np.nan
                ),

                "avg_drawdown_pct": (
                    dd.mean()
                    if len(dd)
                    else np.nan
                ),

                "worst_drawdown_pct": (
                    dd.min()
                    if len(dd)
                    else np.nan
                ),

                "assets_positive_pct": (
                    group.groupby(
                        "asset"
                    )["return_pct"]
                    .mean()
                    .gt(0)
                    .mean()
                    * 100
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "V7 SURVIVAL BACKTEST"
    )
    print("=" * 70)

    print(
        "Strategy: V6-C"
    )

    print(
        "Survival Layer: V7-S0"
    )

    print(
        "Assets: BTC + ETH + SOL"
    )

    print(
        "Capital: 20 / 50 / 100 / 250"
    )

    print(
        "Rolling OOS: 12M train / 3M OOS"
    )

    print("=" * 70)

    all_results = []

    for asset in ASSETS:

        print()
        print("-" * 70)
        print(
            f"PROCESSING {asset}"
        )
        print("-" * 70)

        try:

            raw = load_data(
                asset
            )

            print(
                f"5m rows: {len(raw):,}"
            )

            data = prepare_data(
                raw
            )

            print(
                f"1h rows: {len(data):,}"
            )

            windows = create_windows(
                data
            )

            print(
                f"OOS windows: {len(windows)}"
            )

        except Exception as exc:

            print(
                f"ERROR loading {asset}: {exc}"
            )

            continue

        for capital in CAPITALS:

            print()
            print(
                f"Capital: {capital:.2f} USDT"
            )

            for window_number, window in enumerate(
                windows,
                start=1,
            ):

                print(
                    f"  Window {window_number:02d}: "
                    f"{window['oos_start']} -> "
                    f"{window['oos_end']}"
                )

                try:

                    result = run_single_backtest(
                        window["oos"],
                        capital,
                    )

                    row = extract_result(
                        result=result,
                        asset=asset,
                        capital=capital,
                        window_number=window_number,
                        window=window,
                    )

                    all_results.append(
                        row
                    )

                    print(
                        f"     Return: "
                        f"{row['return_pct']:.3f}% | "
                        f"PF: "
                        f"{row['profit_factor']:.3f} | "
                        f"Trades: "
                        f"{int(row['trades'])}"
                    )

                except Exception as exc:

                    print(
                        f"     ERROR: {exc}"
                    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if not all_results:

        print()
        print(
            "NO RESULTS GENERATED."
        )

        return

    results_df = pd.DataFrame(
        all_results
    )

    summary_df = build_summary(
        results_df
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "V7 SURVIVAL SUMMARY"
    )
    print("=" * 70)

    if not summary_df.empty:

        display_columns = [

            "variant",

            "capital",

            "assets_tested",

            "windows_tested",

            "avg_return_pct",

            "median_return_pct",

            "avg_profit_factor",

            "median_profit_factor",

            "positive_window_pct",

            "pf_gt_1_pct",

            "avg_expectancy",

            "total_trades",

            "worst_return_pct",

            "best_return_pct",

            "avg_drawdown_pct",

            "worst_drawdown_pct",

            "assets_positive_pct",
        ]

        print(
            summary_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_FILE}"
    )

    print()

    print("=" * 70)
    print(
        "V7 SURVIVAL BACKTEST COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()