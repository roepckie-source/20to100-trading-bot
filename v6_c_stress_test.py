# ============================================================
# 20to100 Trading Bot
# V6_C STRESS TEST
#
# BTC + ETH + SOL
#
# FIXED STRATEGY:
# V6_C
#
# 12 MONTH TRAIN
# 3 MONTH OOS
# ROLLING EVERY 3 MONTHS
#
# Stress parameters:
# Fee:
#   0.10%
#   0.15%
#   0.20%
#
# Slippage:
#   0.05%
#   0.10%
#   0.20%
#
# Risk per trade:
#   0.50%
#   1.00%
#   1.50%
#
# Total combinations:
# 3 x 3 x 3 = 27
#
# Assets:
# BTC + ETH + SOL
#
# 16 OOS windows per asset
#
# Total OOS runs:
# 27 x 3 x 16 = 1296
#
# IMPORTANT:
# No parameter optimization is performed.
# V6_C remains completely frozen.
# ============================================================

from pathlib import Path
from itertools import product

import pandas as pd

from strategy.strategy_v6 import calculate_indicators
from backtest.v6_engine import V6BacktestEngine


# ============================================================
# CONFIG
# ============================================================

STARTING_BALANCE = 20.0

ATR_STOP_MULTIPLIER = 3.0
TRAILING_ATR_MULTIPLIER = 3.0

ADX_MIN = 20.0

MAX_DAILY_LOSS = 0.05
MAX_CONSECUTIVE_LOSSES = 3
LOSS_COOLDOWN_BARS = 24

GLOBAL_MAX_DRAWDOWN = 0.20

VARIANT = "V6_C"

TRAIN_MONTHS = 12
OOS_MONTHS = 3


# ============================================================
# STRESS MATRIX
# ============================================================

FEE_RATES = [
    0.0010,   # 0.10%
    0.0015,   # 0.15%
    0.0020,   # 0.20%
]

SLIPPAGE_RATES = [
    0.0005,   # 0.05%
    0.0010,   # 0.10%
    0.0020,   # 0.20%
]

RISK_PER_TRADE_VALUES = [
    0.005,    # 0.50%
    0.010,    # 1.00%
    0.015,    # 1.50%
]


# ============================================================
# ASSETS
# ============================================================

ASSETS = {
    "BTC/USDT": "BTC_USDT_5m.csv",
    "ETH/USDT": "ETH_USDT_5m.csv",
    "SOL/USDT": "SOL_USDT_5m.csv",
}


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_FILE = (
    LOG_DIR
    / "v6_c_stress_test_results.csv"
)

ASSET_SUMMARY_FILE = (
    LOG_DIR
    / "v6_c_stress_test_asset_summary.csv"
)

SUMMARY_FILE = (
    LOG_DIR
    / "v6_c_stress_test_summary.csv"
)

RANKING_FILE = (
    LOG_DIR
    / "v6_c_stress_test_ranking.csv"
)


# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    rename_map = {}

    for column in df.columns:

        name = str(column).strip().lower()

        if name in ("timestamp", "datetime", "date", "time"):
            rename_map[column] = "timestamp"

        elif name in ("open", "o"):
            rename_map[column] = "open"

        elif name in ("high", "h"):
            rename_map[column] = "high"

        elif name in ("low", "l"):
            rename_map[column] = "low"

        elif name in ("close", "c"):
            rename_map[column] = "close"

        elif name in ("volume", "vol", "v"):
            rename_map[column] = "volume"

    df = df.rename(
        columns=rename_map
    )

    required = [
        "timestamp",
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
            "Fehlende Spalten: "
            + ", ".join(missing)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
    )

    df = df.set_index(
        "timestamp"
    )

    return df


# ============================================================
# LOAD 5M DATA
# ============================================================

def load_asset(
    symbol: str,
    filename: str
) -> pd.DataFrame:

    data_file = (
        DATA_DIR
        / filename
    )

    if not data_file.exists():

        raise FileNotFoundError(
            f"Keine Datei gefunden: {data_file}"
        )

    print()
    print("=" * 70)
    print(f"LADE {symbol}")
    print("=" * 70)

    print(
        f"Datei: {data_file}"
    )

    df = pd.read_csv(
        data_file
    )

    df = normalize_columns(
        df
    )

    print(
        f"5m Candles: {len(df):,}"
    )

    print(
        f"Start: {df.index.min()}"
    )

    print(
        f"Ende : {df.index.max()}"
    )

    # --------------------------------------------------------
    # 5m -> 1h
    # --------------------------------------------------------

    df = (
        df[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
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

    print(
        f"1h Candles: {len(df):,}"
    )

    print(
        f"1h Start: {df.index.min()}"
    )

    print(
        f"1h Ende : {df.index.max()}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Calculate indicators BEFORE splitting.
    #
    # This preserves indicator warm-up across
    # TRAIN/OOS boundaries.
    # --------------------------------------------------------

    print(
        "Berechne V6_C-Indikatoren..."
    )

    df = calculate_indicators(
        df
    )

    return df


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================

def create_windows(
    df: pd.DataFrame
):

    start = df.index.min()

    end = df.index.max()

    windows = []

    train_start = start

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

        if oos_end > end:

            break

        windows.append(
            {
                "window":
                    len(windows) + 1,

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

        train_start = (
            train_start
            + pd.DateOffset(
                months=OOS_MONTHS
            )
        )

    return windows


# ============================================================
# SINGLE BACKTEST
# ============================================================

def run_single_test(
    oos_df: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
    risk_per_trade: float,
):

    engine = V6BacktestEngine(

        starting_balance=
            STARTING_BALANCE,

        risk_per_trade=
            risk_per_trade,

        fee_rate=
            fee_rate,

        slippage_rate=
            slippage_rate,

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
            VARIANT,
    )

    result = engine.run(
        oos_df
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("V6_C STRESS TEST")
    print("=" * 80)

    print()
    print("FIXED STRATEGY : V6_C")
    print("TIMEFRAME       : 1h")
    print("TRAIN           : 12 Monate")
    print("OOS             : 3 Monate")
    print("ROLLING         : 3 Monate")
    print("ASSETS          : BTC + ETH + SOL")

    print()
    print(
        "Fee:",
        [
            f"{x * 100:.2f}%"
            for x in FEE_RATES
        ]
    )

    print(
        "Slippage:",
        [
            f"{x * 100:.2f}%"
            for x in SLIPPAGE_RATES
        ]
    )

    print(
        "Risk:",
        [
            f"{x * 100:.2f}%"
            for x in RISK_PER_TRADE_VALUES
        ]
    )

    combinations = list(
        product(
            FEE_RATES,
            SLIPPAGE_RATES,
            RISK_PER_TRADE_VALUES
        )
    )

    print()
    print(
        f"Stress-Kombinationen: "
        f"{len(combinations)}"
    )

    # --------------------------------------------------------
    # Load all assets first
    # --------------------------------------------------------

    asset_data = {}

    for symbol, filename in ASSETS.items():

        asset_data[symbol] = load_asset(
            symbol,
            filename
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    detailed_results = []

    # --------------------------------------------------------
    # Loop stress combinations
    # --------------------------------------------------------

    for combo_number, (
        fee_rate,
        slippage_rate,
        risk_per_trade
    ) in enumerate(
        combinations,
        start=1
    ):

        combo_name = (
            f"F{fee_rate * 100:.2f}_"
            f"S{slippage_rate * 100:.2f}_"
            f"R{risk_per_trade * 100:.2f}"
        )

        print()
        print("=" * 80)
        print(
            f"STRESS COMBINATION "
            f"{combo_number}/{len(combinations)}"
        )
        print("=" * 80)

        print(
            f"Fee       : {fee_rate * 100:.2f}%"
        )

        print(
            f"Slippage  : "
            f"{slippage_rate * 100:.2f}%"
        )

        print(
            f"Risk      : "
            f"{risk_per_trade * 100:.2f}%"
        )

        print(
            f"Name      : {combo_name}"
        )

        # ----------------------------------------------------
        # Each asset
        # ----------------------------------------------------

        for symbol, df in asset_data.items():

            windows = create_windows(
                df
            )

            print()
            print(
                "-" * 80
            )

            print(
                f"{symbol} | "
                f"{len(windows)} OOS-Fenster"
            )

            print(
                "-" * 80
            )

            for window_data in windows:

                window_number = (
                    window_data["window"]
                )

                train_start = (
                    window_data["train_start"]
                )

                train_end = (
                    window_data["train_end"]
                )

                oos_start = (
                    window_data["oos_start"]
                )

                oos_end = (
                    window_data["oos_end"]
                )

                # ------------------------------------------------
                # Train data is deliberately retained only
                # for window accounting / consistency.
                #
                # Strategy parameters remain FIXED.
                # ------------------------------------------------

                train_df = df[
                    (
                        df.index >= train_start
                    )
                    &
                    (
                        df.index < train_end
                    )
                ].copy()

                oos_df = df[
                    (
                        df.index >= oos_start
                    )
                    &
                    (
                        df.index < oos_end
                    )
                ].copy()

                if len(oos_df) < 300:

                    print(
                        f"  Fenster "
                        f"{window_number:02d}: "
                        f"zu wenig OOS-Daten"
                    )

                    continue

                print(
                    f"  Fenster "
                    f"{window_number:02d}/"
                    f"{len(windows)} | "
                    f"OOS "
                    f"{oos_start.date()} "
                    f"-> "
                    f"{oos_end.date()}"
                )

                try:

                    result = run_single_test(
                        oos_df=oos_df,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                        risk_per_trade=risk_per_trade,
                    )

                    detailed_results.append(
                        {
                            "combo":
                                combo_name,

                            "fee_rate":
                                fee_rate,

                            "fee_pct":
                                fee_rate * 100,

                            "slippage_rate":
                                slippage_rate,

                            "slippage_pct":
                                slippage_rate * 100,

                            "risk_per_trade":
                                risk_per_trade,

                            "risk_pct":
                                risk_per_trade * 100,

                            "symbol":
                                symbol,

                            "variant":
                                VARIANT,

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
                                len(train_df),

                            "oos_candles":
                                len(oos_df),

                            "final_balance":
                                result.get(
                                    "final_balance"
                                ),

                            "profit":
                                result.get(
                                    "profit"
                                ),

                            "return_pct":
                                result.get(
                                    "return_pct"
                                ),

                            "trades":
                                result.get(
                                    "trades"
                                ),

                            "wins":
                                result.get(
                                    "wins"
                                ),

                            "losses":
                                result.get(
                                    "losses"
                                ),

                            "win_rate":
                                result.get(
                                    "win_rate"
                                ),

                            "profit_factor":
                                result.get(
                                    "profit_factor"
                                ),

                            "expectancy":
                                result.get(
                                    "expectancy"
                                ),

                            "max_drawdown_pct":
                                result.get(
                                    "max_drawdown_pct"
                                ),

                            "fees":
                                result.get(
                                    "fees"
                                ),

                            "slippage_cost":
                                result.get(
                                    "slippage_cost"
                                ),
                        }
                    )

                    print(
                        f"       Return "
                        f"{result.get('return_pct', 0):+.2f}% | "
                        f"PF "
                        f"{result.get('profit_factor', 0):.3f} | "
                        f"Trades "
                        f"{result.get('trades', 0)}"
                    )

                except Exception as exc:

                    print(
                        f"       ERROR: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    if not detailed_results:

        print()
        print(
            "❌ Keine Ergebnisse erzeugt."
        )

        return

    results_df = pd.DataFrame(
        detailed_results
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"DETAIL RESULTS gespeichert:"
    )

    print(
        RESULT_FILE
    )

    # ========================================================
    # ASSET SUMMARY
    # ========================================================

    asset_summary = []

    group_columns = [
        "combo",
        "fee_pct",
        "slippage_pct",
        "risk_pct",
        "symbol",
    ]

    grouped = (
        results_df
        .groupby(
            group_columns,
            dropna=False
        )
    )

    for keys, group in grouped:

        (
            combo,
            fee_pct,
            slippage_pct,
            risk_pct,
            symbol,
        ) = keys

        returns = (
            pd.to_numeric(
                group["return_pct"],
                errors="coerce"
            )
            .dropna()
        )

        pfs = (
            pd.to_numeric(
                group["profit_factor"],
                errors="coerce"
            )
            .replace(
                [float("inf"), float("-inf")],
                pd.NA
            )
            .dropna()
        )

        drawdowns = (
            pd.to_numeric(
                group["max_drawdown_pct"],
                errors="coerce"
            )
            .dropna()
        )

        asset_summary.append(
            {
                "combo":
                    combo,

                "fee_pct":
                    fee_pct,

                "slippage_pct":
                    slippage_pct,

                "risk_pct":
                    risk_pct,

                "symbol":
                    symbol,

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
                        (pfs > 1).sum()
                    ),

                "pf_above_1_pct":
                    (
                        (pfs > 1).mean()
                        * 100
                    )
                    if len(pfs)
                    else 0.0,

                "avg_return_pct":
                    returns.mean()
                    if len(returns)
                    else 0.0,

                "median_return_pct":
                    returns.median()
                    if len(returns)
                    else 0.0,

                "best_return_pct":
                    returns.max()
                    if len(returns)
                    else 0.0,

                "worst_return_pct":
                    returns.min()
                    if len(returns)
                    else 0.0,

                "avg_profit_factor":
                    pfs.mean()
                    if len(pfs)
                    else 0.0,

                "median_profit_factor":
                    pfs.median()
                    if len(pfs)
                    else 0.0,

                "total_trades":
                    pd.to_numeric(
                        group["trades"],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum(),

                "avg_win_rate":
                    pd.to_numeric(
                        group["win_rate"],
                        errors="coerce"
                    )
                    .mean(),

                "worst_drawdown_pct":
                    drawdowns.min()
                    if len(drawdowns)
                    else 0.0,
            }
        )

    asset_summary_df = pd.DataFrame(
        asset_summary
    )

    asset_summary_df.to_csv(
        ASSET_SUMMARY_FILE,
        index=False
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    global_summary = []

    grouped_global = (
        results_df
        .groupby(
            [
                "combo",
                "fee_pct",
                "slippage_pct",
                "risk_pct",
            ],
            dropna=False
        )
    )

    for keys, group in grouped_global:

        (
            combo,
            fee_pct,
            slippage_pct,
            risk_pct,
        ) = keys

        returns = (
            pd.to_numeric(
                group["return_pct"],
                errors="coerce"
            )
            .dropna()
        )

        pfs = (
            pd.to_numeric(
                group["profit_factor"],
                errors="coerce"
            )
            .replace(
                [float("inf"), float("-inf")],
                pd.NA
            )
            .dropna()
        )

        drawdowns = (
            pd.to_numeric(
                group["max_drawdown_pct"],
                errors="coerce"
            )
            .dropna()
        )

        total_trades = (
            pd.to_numeric(
                group["trades"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

        global_summary.append(
            {
                "combo":
                    combo,

                "fee_pct":
                    fee_pct,

                "slippage_pct":
                    slippage_pct,

                "risk_pct":
                    risk_pct,

                "assets":
                    group["symbol"]
                    .nunique(),

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
                        (pfs > 1).sum()
                    ),

                "pf_above_1_pct":
                    (
                        (pfs > 1).mean()
                        * 100
                    )
                    if len(pfs)
                    else 0.0,

                "avg_return_pct":
                    returns.mean()
                    if len(returns)
                    else 0.0,

                "median_return_pct":
                    returns.median()
                    if len(returns)
                    else 0.0,

                "best_return_pct":
                    returns.max()
                    if len(returns)
                    else 0.0,

                "worst_return_pct":
                    returns.min()
                    if len(returns)
                    else 0.0,

                "avg_profit_factor":
                    pfs.mean()
                    if len(pfs)
                    else 0.0,

                "median_profit_factor":
                    pfs.median()
                    if len(pfs)
                    else 0.0,

                "total_trades":
                    total_trades,

                "avg_win_rate":
                    pd.to_numeric(
                        group["win_rate"],
                        errors="coerce"
                    ).mean(),

                "avg_drawdown_pct":
                    drawdowns.mean()
                    if len(drawdowns)
                    else 0.0,

                "worst_drawdown_pct":
                    drawdowns.min()
                    if len(drawdowns)
                    else 0.0,
            }
        )

    summary_df = pd.DataFrame(
        global_summary
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # ========================================================
    # ROBUSTNESS SCORE
    # ========================================================

    # We deliberately score robustness rather than
    # simply selecting the highest return.
    #
    # Priority:
    # 1. Positive-window percentage
    # 2. Median PF
    # 3. Median return
    # 4. Worst drawdown
    #
    # No strategy parameters are changed here.

    ranking_df = summary_df.copy()

    if len(ranking_df):

        ranking_df["robustness_score"] = (

            ranking_df[
                "positive_window_pct"
            ]
            * 0.30

            +

            (
                ranking_df[
                    "pf_above_1_pct"
                ]
            )
            * 0.25

            +

            (
                ranking_df[
                    "median_profit_factor"
                ]
                .clip(
                    lower=0,
                    upper=3
                )
                / 3
                * 100
            )
            * 0.20

            +

            (
                ranking_df[
                    "median_return_pct"
                ]
                .clip(
                    lower=-5,
                    upper=5
                )
                + 5
            )
            / 10
            * 100
            * 0.15

            +

            (
                (
                    -ranking_df[
                        "worst_drawdown_pct"
                    ]
                )
                .clip(
                    lower=0,
                    upper=20
                )

            )

        )

        # The previous expression intentionally keeps DD
        # as a separate risk dimension.
        #
        # Convert DD to a positive robustness contribution:
        ranking_df["drawdown_score"] = (
            100
            -
            (
                -ranking_df[
                    "worst_drawdown_pct"
                ]
            ).clip(
                lower=0,
                upper=20
            )
            / 20
            * 100
        )

        ranking_df["robustness_score"] = (
            ranking_df["robustness_score"]
            -
            ranking_df["drawdown_score"]
            * 0.10
        )

        ranking_df = (
            ranking_df
            .sort_values(
                [
                    "robustness_score",
                    "median_return_pct",
                    "median_profit_factor",
                ],
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        ranking_df.insert(
            0,
            "rank",
            range(
                1,
                len(ranking_df) + 1
            )
        )

    ranking_df.to_csv(
        RANKING_FILE,
        index=False
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("V6_C STRESS TEST SUMMARY")
    print("=" * 80)

    display_columns = [
        "rank",
        "combo",
        "positive_window_pct",
        "pf_above_1_pct",
        "avg_return_pct",
        "median_return_pct",
        "avg_profit_factor",
        "median_profit_factor",
        "worst_return_pct",
        "worst_drawdown_pct",
        "total_trades",
        "robustness_score",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in ranking_df.columns
    ]

    print()

    print(
        ranking_df[
            available_columns
        ]
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    # ========================================================
    # MOST ROBUST COMBINATION
    # ========================================================

    if len(ranking_df):

        best = ranking_df.iloc[0]

        print()
        print("=" * 80)
        print("🏆 ROBUSTHEITS-SIEGER")
        print("=" * 80)

        print(
            f"Kombination : "
            f"{best['combo']}"
        )

        print(
            f"Fee         : "
            f"{best['fee_pct']:.2f}%"
        )

        print(
            f"Slippage    : "
            f"{best['slippage_pct']:.2f}%"
        )

        print(
            f"Risk/Trade  : "
            f"{best['risk_pct']:.2f}%"
        )

        print(
            f"Positive OOS: "
            f"{best['positive_window_pct']:.2f}%"
        )

        print(
            f"PF > 1      : "
            f"{best['pf_above_1_pct']:.2f}%"
        )

        print(
            f"Median Return: "
            f"{best['median_return_pct']:+.3f}%"
        )

        print(
            f"Median PF    : "
            f"{best['median_profit_factor']:.3f}"
        )

        print(
            f"Worst Return : "
            f"{best['worst_return_pct']:+.3f}%"
        )

        print(
            f"Worst DD     : "
            f"{best['worst_drawdown_pct']:.3f}%"
        )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 80)
    print("DATEIEN")
    print("=" * 80)

    print(
        f"Details : {RESULT_FILE}"
    )

    print(
        f"Assets  : {ASSET_SUMMARY_FILE}"
    )

    print(
        f"Summary : {SUMMARY_FILE}"
    )

    print(
        f"Ranking : {RANKING_FILE}"
    )

    print()
    print(
        "✅ V6_C Stress-Test abgeschlossen."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
