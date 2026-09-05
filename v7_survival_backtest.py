# ============================================================
# V7 SURVIVAL BACKTEST
# ============================================================
#
# V7-S0 = V6-C ENTRY ENGINE + SURVIVAL RISK MANAGEMENT
#
# Philosophy:
#   SURVIVE FIRST. GROW SECOND.
#
# Assets:
#   BTC / ETH / SOL
#
# Capitals:
#   20 / 50 / 100 / 250 USDT
#
# Walk Forward:
#   12 months train
#   3 months OOS
#   3 months rolling step
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

try:
    from strategy.strategy_v6 import calculate_indicators
except ImportError as exc:
    print("ERROR: Could not import calculate_indicators")
    print(exc)
    raise


try:
    from backtest.v7_survival_engine import V7SurvivalEngine
except ImportError as exc:
    print("ERROR: Could not import V7SurvivalEngine")
    print(exc)
    raise


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = ROOT / "data"

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
]

TIMEFRAME = "5m"

DATA_FILES = {
    "BTC": DATA_DIR / "BTC_USDT_5m.csv",
    "ETH": DATA_DIR / "ETH_USDT_5m.csv",
    "SOL": DATA_DIR / "SOL_USDT_5m.csv",
}


# ============================================================
# V7 CONFIG
# ============================================================

STARTING_CAPITALS = [
    20.0,
    50.0,
    100.0,
    250.0,
]


RISK_PER_TRADE = 0.01

FEE_RATE = 0.001

SLIPPAGE_RATE = 0.0005

ATR_STOP_MULTIPLIER = 3.0

TRAILING_ATR_MULTIPLIER = 3.0

ADX_MIN = 20.0


# ============================================================
# SURVIVAL CONFIG
# ============================================================

MAX_DAILY_LOSS_PCT = 5.0

MAX_CONSECUTIVE_LOSSES = 3

COOLDOWN_BARS = 24

GLOBAL_MAX_DRAWDOWN_PCT = 20.0


# ============================================================
# STRATEGY
# ============================================================

VARIANT = "V6_C"


# ============================================================
# WALK-FORWARD CONFIG
# ============================================================

TRAIN_MONTHS = 12

OOS_MONTHS = 3

STEP_MONTHS = 3


# ============================================================
# OUTPUT
# ============================================================

RESULTS_FILE = ROOT / "v7_survival_results.csv"

SUMMARY_FILE = ROOT / "v7_survival_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    """
    Safely convert a value to float.
    """

    try:

        if value is None:
            return default

        if isinstance(value, str):

            if value.strip().lower() in {
                "",
                "nan",
                "none",
                "n/a",
                "na",
            }:
                return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:

        return default


# ============================================================
# LOAD DATA
# ============================================================

def load_data(asset):
    """
    Load historical 5m OHLCV data.
    """

    file_path = DATA_FILES[asset]

    print()
    print("=" * 70)
    print(f"LOADING {asset}")
    print("=" * 70)

    if not file_path.exists():

        raise FileNotFoundError(
            f"Data file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print(
        f"{asset}: loaded {len(df):,} rows"
    )

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
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

    timestamp_column = None

    for candidate in timestamp_candidates:

        if candidate in df.columns:

            timestamp_column = candidate
            break

    if timestamp_column is None:

        raise ValueError(
            f"{asset}: No timestamp column found. "
            f"Columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # Required OHLC columns
    # --------------------------------------------------------

    required_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{asset}: Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if "volume" in df.columns:

        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"{asset}: cleaned {len(df):,} rows"
    )

    if len(df) > 0:

        print(
            f"{asset}: "
            f"{df['timestamp'].iloc[0]} -> "
            f"{df['timestamp'].iloc[-1]}"
        )

    return df


# ============================================================
# RESAMPLE 5m -> 1h
# ============================================================

def resample_to_1h(df):
    """
    Convert 5m OHLCV data to 1h candles.
    """

    data = df.copy()

    data = data.set_index(
        "timestamp"
    )

    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    if "volume" in data.columns:

        aggregation["volume"] = "sum"

    hourly = (
        data
        .resample("1h")
        .agg(aggregation)
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .reset_index()
    )

    return hourly


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(asset):
    """
    Load 5m data, convert to 1h and calculate V6 indicators.
    """

    df_5m = load_data(asset)

    print(
        f"{asset}: resampling 5m -> 1h..."
    )

    df_1h = resample_to_1h(
        df_5m
    )

    print(
        f"{asset}: {len(df_1h):,} hourly candles"
    )

    if len(df_1h) < 500:

        raise ValueError(
            f"{asset}: Not enough hourly data."
        )

    print(
        f"{asset}: calculating V6 indicators..."
    )

    df_1h = calculate_indicators(
        df_1h
    )

    # --------------------------------------------------------
    # Clean indicator rows
    # --------------------------------------------------------

    df_1h = (
        df_1h
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .reset_index(drop=True)
    )

    print(
        f"{asset}: final prepared rows "
        f"{len(df_1h):,}"
    )

    return df_1h


# ============================================================
# CREATE WALK-FORWARD WINDOWS
# ============================================================

def create_windows(df):
    """
    Create rolling:

        12 month TRAIN
        3 month OOS
        3 month STEP
    """

    if df.empty:

        return []

    timestamps = df["timestamp"]

    start_date = timestamps.iloc[0]

    end_date = timestamps.iloc[-1]

    windows = []

    train_start = start_date

    window_id = 0

    while True:

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

        if oos_end > end_date:

            break

        train_mask = (
            (timestamps >= train_start)
            &
            (timestamps < train_end)
        )

        oos_mask = (
            (timestamps >= train_end)
            &
            (timestamps < oos_end)
        )

        train_df = df.loc[
            train_mask
        ].copy()

        oos_df = df.loc[
            oos_mask
        ].copy()

        if len(train_df) > 0 and len(oos_df) > 0:

            windows.append(
                {
                    "window_id": window_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "oos_start": train_end,
                    "oos_end": oos_end,
                    "train_df": train_df,
                    "oos_df": oos_df,
                }
            )

            window_id += 1

        train_start = (
            train_start
            + pd.DateOffset(
                months=STEP_MONTHS
            )
        )

    return windows


# ============================================================
# RUN SINGLE BACKTEST
# ============================================================

def run_single_backtest(
    asset,
    capital,
    oos_df,
    window_id,
):
    """
    Run one V7-S0 OOS backtest.

    IMPORTANT:
    The parameter names here must exactly match
    V7SurvivalEngine.__init__().
    """

    # --------------------------------------------------------
    # IMPORTANT PARAMETER MAPPING
    # --------------------------------------------------------
    #
    # Runner:
    #
    #   RISK_PER_TRADE
    #   MAX_DAILY_LOSS_PCT
    #   COOLDOWN_BARS
    #   GLOBAL_MAX_DRAWDOWN_PCT
    #
    # Engine:
    #
    #   base_risk_per_trade
    #   max_daily_loss
    #   loss_cooldown_bars
    #   global_max_drawdown
    #
    # Percent values from runner are converted to fractions
    # where required by the engine.
    # --------------------------------------------------------

    engine = V7SurvivalEngine(

        starting_balance=capital,

        base_risk_per_trade=RISK_PER_TRADE,

        fee_rate=FEE_RATE,

        slippage_rate=SLIPPAGE_RATE,

        atr_stop_multiplier=ATR_STOP_MULTIPLIER,

        trailing_atr_multiplier=TRAILING_ATR_MULTIPLIER,

        adx_min=ADX_MIN,

        variant=VARIANT,

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
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    result = engine.run(
        oos_df
    )

    return result


# ============================================================
# EXTRACT RESULT
# ============================================================

def extract_result(
    asset,
    capital,
    window,
    result,
):
    """
    Convert engine result into one flat CSV row.
    """

    return {

        "asset": asset,

        "capital": capital,

        "variant": VARIANT,

        "strategy": result.get(
            "strategy",
            "V7_S0",
        ),

        "window_id": window[
            "window_id"
        ],

        "train_start": window[
            "train_start"
        ],

        "train_end": window[
            "train_end"
        ],

        "oos_start": window[
            "oos_start"
        ],

        "oos_end": window[
            "oos_end"
        ],

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

        # ----------------------------------------------------
        # IMPORTANT:
        # Engine returns "win_rate",
        # not "win_rate_pct".
        # ----------------------------------------------------

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

        "normal_trades": safe_float(
            result.get(
                "normal_trades"
            ),
            0,
        ),

        "defensive_trades": safe_float(
            result.get(
                "defensive_trades"
            ),
            0,
        ),

        "survival_trades": safe_float(
            result.get(
                "survival_trades"
            ),
            0,
        ),

        "critical_trades": safe_float(
            result.get(
                "critical_trades"
            ),
            0,
        ),

        "kill_switch_triggered": result.get(
            "kill_switch_triggered",
            False,
        ),
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(results_df):
    """
    Build capital-level summary.
    """

    if results_df.empty:

        return pd.DataFrame()

    summaries = []

    grouped = results_df.groupby(
        [
            "capital",
            "variant",
        ],
        dropna=False,
    )

    for (
        capital,
        variant,
    ), group in grouped:

        returns = group[
            "return_pct"
        ].dropna()

        profit_factors = group[
            "profit_factor"
        ].dropna()

        expectancies = group[
            "expectancy"
        ].dropna()

        drawdowns = group[
            "max_drawdown_pct"
        ].dropna()

        trades = group[
            "trades"
        ].fillna(0)

        # ----------------------------------------------------
        # Positive windows
        # ----------------------------------------------------

        positive_windows = (
            group[
                "return_pct"
            ] > 0
        ).sum()

        total_windows = len(group)

        positive_window_pct = (
            positive_windows
            / total_windows
            * 100.0
            if total_windows > 0
            else np.nan
        )

        # ----------------------------------------------------
        # PF > 1
        # ----------------------------------------------------

        pf_gt_one = (
            group[
                "profit_factor"
            ] > 1.0
        ).sum()

        pf_gt_one_pct = (
            pf_gt_one
            / total_windows
            * 100.0
            if total_windows > 0
            else np.nan
        )

        # ----------------------------------------------------
        # Positive assets/windows
        # ----------------------------------------------------

        assets_positive = (
            group[
                "return_pct"
            ] > 0
        ).sum()

        assets_positive_pct = (
            assets_positive
            / total_windows
            * 100.0
            if total_windows > 0
            else np.nan
        )

        summaries.append(
            {

                "capital": capital,

                "variant": variant,

                "windows": total_windows,

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
                    profit_factors.mean()
                    if len(profit_factors)
                    else np.nan
                ),

                "median_profit_factor": (
                    profit_factors.median()
                    if len(profit_factors)
                    else np.nan
                ),

                "avg_expectancy": (
                    expectancies.mean()
                    if len(expectancies)
                    else np.nan
                ),

                "avg_positive_window_pct": (
                    positive_window_pct
                ),

                "avg_pf_gt_1_pct": (
                    pf_gt_one_pct
                ),

                "total_trades": (
                    trades.sum()
                ),

                "avg_trades_per_window": (
                    trades.mean()
                    if len(trades)
                    else np.nan
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
                    drawdowns.mean()
                    if len(drawdowns)
                    else np.nan
                ),

                "worst_drawdown_pct": (
                    drawdowns.max()
                    if len(drawdowns)
                    else np.nan
                ),

                "assets_positive_pct": (
                    assets_positive_pct
                ),
            }
        )

    return pd.DataFrame(
        summaries
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V7 SURVIVAL BACKTEST")
    print("=" * 70)
    print()
    print("V7-S0 = V6-C + SURVIVAL RISK MANAGEMENT")
    print()
    print("Assets:")
    print("  BTC")
    print("  ETH")
    print("  SOL")
    print()
    print("Capital:")
    print(
        "  "
        + ", ".join(
            f"{x:g}"
            for x in STARTING_CAPITALS
        )
        + " USDT"
    )
    print()
    print("Risk per trade:")
    print(
        f"  {RISK_PER_TRADE * 100:.2f}%"
    )
    print()
    print("Daily loss limit:")
    print(
        f"  {MAX_DAILY_LOSS_PCT:.2f}%"
    )
    print()
    print("Max consecutive losses:")
    print(
        f"  {MAX_CONSECUTIVE_LOSSES}"
    )
    print()
    print("Cooldown:")
    print(
        f"  {COOLDOWN_BARS} bars"
    )
    print()
    print("Global max drawdown:")
    print(
        f"  {GLOBAL_MAX_DRAWDOWN_PCT:.2f}%"
    )
    print()
    print("=" * 70)

    # ========================================================
    # PREPARE ALL ASSETS
    # ========================================================

    prepared_data = {}

    for asset in ASSETS:

        try:

            prepared_data[
                asset
            ] = prepare_data(
                asset
            )

        except Exception as exc:

            print()
            print(
                f"ERROR preparing {asset}:"
            )
            print(exc)

            raise

    # ========================================================
    # RUN BACKTESTS
    # ========================================================

    all_results = []

    for asset in ASSETS:

        df = prepared_data[
            asset
        ]

        print()
        print("=" * 70)
        print(
            f"CREATING WALK-FORWARD WINDOWS: {asset}"
        )
        print("=" * 70)

        windows = create_windows(
            df
        )

        print(
            f"{asset}: "
            f"{len(windows)} OOS windows"
        )

        if not windows:

            print(
                f"WARNING: No windows for {asset}"
            )

            continue

        # ====================================================
        # CAPITAL LOOP
        # ====================================================

        for capital in STARTING_CAPITALS:

            print()
            print("-" * 70)
            print(
                f"{asset} | CAPITAL {capital:g} USDT"
            )
            print("-" * 70)

            for window in windows:

                window_id = window[
                    "window_id"
                ]

                oos_df = window[
                    "oos_df"
                ]

                print(
                    f"{asset} | "
                    f"${capital:g} | "
                    f"Window {window_id:02d} | "
                    f"OOS "
                    f"{window['oos_start'].date()} "
                    f"-> "
                    f"{window['oos_end'].date()} | "
                    f"{len(oos_df):,} bars"
                )

                try:

                    result = run_single_backtest(
                        asset=asset,
                        capital=capital,
                        oos_df=oos_df,
                        window_id=window_id,
                    )

                    row = extract_result(
                        asset=asset,
                        capital=capital,
                        window=window,
                        result=result,
                    )

                    all_results.append(
                        row
                    )

                    print(
                        "   "
                        f"Return: "
                        f"{row['return_pct']:.4f}% | "
                        f"PF: "
                        f"{row['profit_factor']:.4f} | "
                        f"Trades: "
                        f"{int(row['trades'])} | "
                        f"DD: "
                        f"{row['max_drawdown_pct']:.4f}%"
                    )

                except Exception as exc:

                    print()
                    print(
                        f"   ERROR "
                        f"{asset} | "
                        f"${capital:g} | "
                        f"Window {window_id}:"
                    )

                    print(
                        f"   {type(exc).__name__}: "
                        f"{exc}"
                    )

                    continue

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING RESULTS")
    print("=" * 70)

    if not all_results:

        print()
        print(
            "ERROR: NO RESULTS WERE GENERATED."
        )

        print()
        print(
            "v7_survival_results.csv NOT FOUND"
        )

        print(
            "v7_survival_summary.csv NOT FOUND"
        )

        return

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    sort_columns = [
        "asset",
        "capital",
        "window_id",
    ]

    existing_sort_columns = [
        column
        for column in sort_columns
        if column in results_df.columns
    ]

    if existing_sort_columns:

        results_df = (
            results_df
            .sort_values(
                existing_sort_columns
            )
            .reset_index(drop=True)
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print(
        f"Saved results:"
    )

    print(
        f"  {RESULTS_FILE}"
    )

    print(
        f"  Rows: {len(results_df):,}"
    )

    # ========================================================
    # BUILD SUMMARY
    # ========================================================

    summary_df = build_summary(
        results_df
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        "Saved summary:"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("V7-S0 SURVIVAL SUMMARY")
    print("=" * 70)

    if summary_df.empty:

        print(
            "SUMMARY IS EMPTY"
        )

    else:

        display_columns = [

            "capital",

            "variant",

            "windows",

            "avg_return_pct",

            "median_return_pct",

            "avg_profit_factor",

            "median_profit_factor",

            "avg_positive_window_pct",

            "avg_pf_gt_1_pct",

            "total_trades",

            "worst_return_pct",

            "best_return_pct",

            "avg_drawdown_pct",

            "worst_drawdown_pct",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in summary_df.columns
        ]

        print()

        print(
            summary_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 70)
    print("V7-S0 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Result rows: "
        f"{len(results_df):,}"
    )

    print(
        f"Summary rows: "
        f"{len(summary_df):,}"
    )

    print()
    print(
        "SURVIVE FIRST. GROW SECOND."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
