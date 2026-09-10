from __future__ import annotations

import os
import sys
import traceback

import pandas as pd

from backtest.v7_survival_engine import V7SurvivalEngine
from strategy.strategy_v6 import calculate_indicators


# ============================================================
# V7-S0 FIXED-LOGIC WALK-FORWARD / OUT-OF-SAMPLE TEST
#
# IMPORTANT:
# - V6-C ENTRY LOGIC IS NOT CHANGED
# - V7-S0 RISK/EXIT LOGIC IS NOT CHANGED
# - NO PARAMETER OPTIMIZATION
# - NO LIVE TRADING
# - NO REAL ORDERS
# - NO API KEYS
#
# OUTPUT:
# - v7_walk_forward_oos_results.csv
# - v7_walk_forward_oos_trades.csv
#
# The second file contains every individual OOS trade plus
# the indicator state at the signal/entry candle.
# ============================================================


ASSETS = [
    "BTC_USDT_5m",
    "ETH_USDT_5m",
    "SOL_USDT_5m",
]

STARTING_CAPITALS = [
    20.0,
    50.0,
    100.0,
    250.0,
]

VARIANT = "V6_C"

WARMUP_DAYS = 40
OOS_DAYS = 90
FOLDS = 4

OUTPUT_FILE = "v7_walk_forward_oos_results.csv"
TRADES_OUTPUT_FILE = "v7_walk_forward_oos_trades.csv"


# ============================================================
# PRINT HELPERS
# ============================================================

def header(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(asset: str) -> pd.DataFrame:

    path = f"data/{asset}.csv"

    print()
    print(f"Loading: {path}")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}"
        )

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"{asset}: timestamp column missing"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    df = df.set_index("timestamp")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        if column not in df.columns:
            raise ValueError(
                f"{asset}: missing column {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.sort_index()

    df = df.loc[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    print(
        f"5m rows:   {len(df):,}"
    )

    print(
        f"5m range:  "
        f"{df.index.min()} -> "
        f"{df.index.max()}"
    )

    return df


# ============================================================
# 5M -> 1H
# ============================================================

def resample_to_1h(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
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
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
    )

    return result


# ============================================================
# PREPARE FULL HISTORY
# ============================================================

def prepare_full_history(
    asset: str,
) -> pd.DataFrame:

    header(
        f"PREPARING {asset}"
    )

    df_5m = load_data(asset)

    df_1h = resample_to_1h(
        df_5m
    )

    print(
        f"1h rows:    {len(df_1h):,}"
    )

    print(
        "Calculating V6 indicators "
        "on FULL history..."
    )

    df_1h = calculate_indicators(
        df_1h.copy()
    )

    required_columns = [
        "open",
        "high",
        "low",
        "close",
        "atr_14",
        "adx_14",
        "ema_100",
        "ema_200",
        "ema_200_slope",
        "ema_200_slope_reference",
        "atr_14_ma50",
        "donchian_high_20",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df_1h.columns
    ]

    if missing:

        raise ValueError(
            f"{asset}: missing V7-S0 "
            f"columns: {missing}"
        )

    print(
        "PASS: all required "
        "V7-S0 columns present"
    )

    return df_1h


# ============================================================
# CREATE ENGINE
# ============================================================

def create_engine(
    capital: float,
) -> V7SurvivalEngine:

    return V7SurvivalEngine(
        starting_balance=capital,

        base_risk_per_trade=0.01,

        fee_rate=0.001,

        slippage_rate=0.0005,

        atr_stop_multiplier=3.0,

        trailing_atr_multiplier=3.0,

        adx_min=20.0,

        variant=VARIANT,

        max_daily_loss=0.05,

        max_consecutive_losses=3,

        loss_cooldown_bars=24,

        global_max_drawdown=0.20,
    )


# ============================================================
# BUILD WALK-FORWARD FOLDS
# ============================================================

def build_folds(
    df: pd.DataFrame,
) -> list[
    tuple[
        int,
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
    ]
]:

    latest = df.index.max()

    folds = []

    for fold_number in range(
        1,
        FOLDS + 1,
    ):

        oos_end = (
            latest
            - pd.Timedelta(
                days=(
                    fold_number - 1
                )
                * OOS_DAYS
            )
        )

        oos_start = (
            oos_end
            - pd.Timedelta(
                days=OOS_DAYS
            )
        )

        warmup_start = (
            oos_start
            - pd.Timedelta(
                days=WARMUP_DAYS
            )
        )

        folds.append(
            (
                fold_number,
                warmup_start,
                oos_start,
                oos_end,
            )
        )

    return folds


# ============================================================
# TRADE DIAGNOSTICS
# ============================================================

def build_trade_rows(
    *,
    asset: str,
    capital: float,
    fold_number: int,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    oos: pd.DataFrame,
    engine: V7SurvivalEngine,
) -> list[dict]:

    rows = []

    for trade_number, trade in enumerate(
        engine.trades,
        start=1,
    ):

        entry_time = pd.Timestamp(
            trade.entry_time
        )

        exit_time = pd.Timestamp(
            trade.exit_time
        )

        # ----------------------------------------------------
        # Signal candle
        #
        # V6-C signal is generated from the preceding
        # completed 1H candle before entry.
        # ----------------------------------------------------

        signal_time = (
            entry_time
            - pd.Timedelta(
                hours=1
            )
        )

        signal_row = None

        if signal_time in oos.index:

            signal_row = oos.loc[
                signal_time
            ]

        else:

            previous_rows = oos.loc[
                oos.index < entry_time
            ]

            if not previous_rows.empty:

                signal_row = (
                    previous_rows.iloc[-1]
                )

        # ----------------------------------------------------
        # Default values
        # ----------------------------------------------------

        entry_atr = None
        entry_adx = None

        entry_ema100 = None
        entry_ema200 = None

        entry_ema200_slope = None
        entry_ema200_slope_reference = None

        entry_atr_ma50 = None
        entry_donchian_high_20 = None

        signal_close = None
        signal_high = None
        signal_low = None

        # ----------------------------------------------------
        # Indicator state
        # ----------------------------------------------------

        if signal_row is not None:

            entry_atr = float(
                signal_row["atr_14"]
            )

            entry_adx = float(
                signal_row["adx_14"]
            )

            entry_ema100 = float(
                signal_row["ema_100"]
            )

            entry_ema200 = float(
                signal_row["ema_200"]
            )

            entry_ema200_slope = float(
                signal_row[
                    "ema_200_slope"
                ]
            )

            entry_ema200_slope_reference = float(
                signal_row[
                    "ema_200_slope_reference"
                ]
            )

            entry_atr_ma50 = float(
                signal_row[
                    "atr_14_ma50"
                ]
            )

            entry_donchian_high_20 = float(
                signal_row[
                    "donchian_high_20"
                ]
            )

            signal_close = float(
                signal_row["close"]
            )

            signal_high = float(
                signal_row["high"]
            )

            signal_low = float(
                signal_row["low"]
            )

        # ----------------------------------------------------
        # Basic trade information
        # ----------------------------------------------------

        entry_price = float(
            trade.entry_price
        )

        exit_price = float(
            trade.exit_price
        )

        quantity = float(
            trade.quantity
        )

        # ----------------------------------------------------
        # Initial risk / stop
        # ----------------------------------------------------

        initial_risk = float(
            trade.initial_risk_usdt
        )

        if quantity > 0:

            initial_stop_distance = (
                initial_risk
                / quantity
            )

        else:

            initial_stop_distance = None

        if initial_stop_distance is not None:

            initial_stop_price = (
                entry_price
                - initial_stop_distance
            )

        else:

            initial_stop_price = None

        # ----------------------------------------------------
        # Holding time
        # ----------------------------------------------------

        held_hours = (
            exit_time
            - entry_time
        ).total_seconds() / 3600.0

        held_bars = (
            int(
                round(
                    held_hours
                )
            )
            if held_hours >= 0
            else None
        )

        # ----------------------------------------------------
        # Result classification
        # ----------------------------------------------------

        net_profit = float(
            trade.net_profit
        )

        if net_profit > 0:

            result_class = "WIN"

        elif net_profit < 0:

            result_class = "LOSS"

        else:

            result_class = "FLAT"

        # ----------------------------------------------------
        # Complete diagnostic row
        # ----------------------------------------------------

        rows.append(
            {
                "strategy": "V7_S0",
                "variant": VARIANT,

                "asset": asset,

                "fold": fold_number,

                "starting_capital": capital,

                "trade_number": trade_number,

                "oos_start": str(
                    oos_start
                ),

                "oos_end": str(
                    oos_end
                ),

                "signal_time": str(
                    signal_time
                ),

                "entry_time": str(
                    entry_time
                ),

                "exit_time": str(
                    exit_time
                ),

                "entry_price": entry_price,

                "exit_price": exit_price,

                "quantity": quantity,

                # Indicators
                "entry_atr_14": entry_atr,

                "entry_adx_14": entry_adx,

                "entry_ema_100": entry_ema100,

                "entry_ema_200": entry_ema200,

                "entry_ema_200_slope":
                    entry_ema200_slope,

                "entry_ema_200_slope_reference":
                    entry_ema200_slope_reference,

                "entry_atr_14_ma50":
                    entry_atr_ma50,

                "entry_donchian_high_20":
                    entry_donchian_high_20,

                # Signal candle
                "signal_close":
                    signal_close,

                "signal_high":
                    signal_high,

                "signal_low":
                    signal_low,

                # Risk
                "initial_stop_price":
                    initial_stop_price,

                "initial_risk_usdt":
                    initial_risk,

                # Result
                "gross_profit":
                    float(
                        trade.gross_profit
                    ),

                "fees":
                    float(
                        trade.fees
                    ),

                "slippage_cost":
                    float(
                        trade.slippage_cost
                    ),

                "net_profit":
                    net_profit,

                "r_multiple":
                    float(
                        trade.r_multiple
                    ),

                "win":
                    bool(
                        net_profit > 0
                    ),

                "exit_reason":
                    str(
                        trade.exit_reason
                    ),

                "held_hours":
                    held_hours,

                "held_bars":
                    held_bars,

                "result_class":
                    result_class,
            }
        )

    return rows


# ============================================================
# RUN ONE OOS FOLD
# ============================================================

def run_oos(
    *,
    asset: str,
    capital: float,
    df: pd.DataFrame,
    fold_number: int,
    warmup_start: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> tuple[
    dict,
    list[dict],
]:

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The strategy is completely frozen here.
    #
    # No optimization.
    # No parameter fitting.
    # No strategy changes.
    # --------------------------------------------------------

    oos = df.loc[
        (
            df.index
            >= oos_start
        )
        &
        (
            df.index
            <= oos_end
        )
    ].copy()

    if len(oos) < 50:

        raise ValueError(
            f"{asset} fold "
            f"{fold_number}: "
            f"only {len(oos)} OOS rows"
        )

    # Fresh engine for every OOS fold.
    # No position/risk state can leak
    # between folds.

    engine = create_engine(
        capital
    )

    result = engine.run(
        oos
    )

    trade_rows = build_trade_rows(
        asset=asset,
        capital=capital,
        fold_number=fold_number,
        oos_start=oos_start,
        oos_end=oos_end,
        oos=oos,
        engine=engine,
    )

    # --------------------------------------------------------
    # Result row
    # --------------------------------------------------------

    result_row = {

        "strategy":
            "V7_S0",

        "variant":
            VARIANT,

        "asset":
            asset,

        "fold":
            fold_number,

        "warmup_days":
            WARMUP_DAYS,

        "oos_days":
            OOS_DAYS,

        "starting_capital":
            capital,

        "oos_start":
            str(
                oos.index.min()
            ),

        "oos_end":
            str(
                oos.index.max()
            ),

        "bars":
            len(oos),

        "final_balance":
            float(
                result.get(
                    "final_balance",
                    capital,
                )
            ),

        "return_pct":
            float(
                result.get(
                    "return_pct",
                    0.0,
                )
            ),

        "trades":
            int(
                result.get(
                    "trades",
                    0,
                )
            ),

        "wins":
            int(
                result.get(
                    "wins",
                    0,
                )
            ),

        "losses":
            int(
                result.get(
                    "losses",
                    0,
                )
            ),

        "win_rate_pct":
            float(
                result.get(
                    "win_rate",
                    result.get(
                        "win_rate_pct",
                        0.0,
                    ),
                )
            ),

        "profit_factor":
            float(
                result.get(
                    "profit_factor",
                    result.get(
                        "pf",
                        0.0,
                    ),
                )
            ),

        "expectancy":
            float(
                result.get(
                    "expectancy",
                    0.0,
                )
            ),

        "max_drawdown_pct":
            float(
                result.get(
                    "max_drawdown_pct",
                    0.0,
                )
            ),

        "fees":
            float(
                result.get(
                    "fees",
                    0.0,
                )
            ),

        "slippage_cost":
            float(
                result.get(
                    "slippage_cost",
                    0.0,
                )
            ),

        "kill_switch":
            bool(
                result.get(
                    "kill_switch",
                    False,
                )
            ),
    }

    return (
        result_row,
        trade_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    header(
        "V7-S0 FIXED-LOGIC "
        "WALK-FORWARD / OOS TEST"
    )

    print(
        "Strategy:        V7-S0"
    )

    print(
        "Entry:           V6-C"
    )

    print(
        "Indicators:      V6"
    )

    print(
        "Timeframe:       1H"
    )

    print(
        "Source:          5M"
    )

    print(
        f"Warmup:          "
        f"{WARMUP_DAYS} days"
    )

    print(
        f"OOS window:      "
        f"{OOS_DAYS} days"
    )

    print(
        f"Folds:            "
        f"{FOLDS}"
    )

    print(
        "Assets:          "
        "BTC / ETH / SOL"
    )

    print(
        "Capital:         "
        "$20 / $50 / $100 / $250"
    )

    print()

    print(
        "NO PARAMETER OPTIMIZATION."
    )

    print(
        "NO STRATEGY CHANGES."
    )

    print(
        "NO LIVE TRADING."
    )

    print(
        "NO REAL ORDERS."
    )

    print(
        "NO API KEYS."
    )

    print()

    print(
        "Detailed trade diagnostics:"
    )

    print(
        f"    {TRADES_OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    results = []

    all_trades = []

    failures = []

    # --------------------------------------------------------
    # Assets
    # --------------------------------------------------------

    for asset in ASSETS:

        try:

            df = prepare_full_history(
                asset
            )

            folds = build_folds(
                df
            )

            print()

            print(
                f"Available 1H history: "
                f"{df.index.min()} -> "
                f"{df.index.max()}"
            )

            # ------------------------------------------------
            # Folds
            # ------------------------------------------------

            for (
                fold_number,
                warmup_start,
                oos_start,
                oos_end,
            ) in folds:

                print()

                print(
                    "-" * 78
                )

                print(
                    f"{asset} | "
                    f"FOLD {fold_number}"
                )

                print(
                    f"Warmup: "
                    f"{warmup_start} -> "
                    f"{oos_start}"
                )

                print(
                    f"OOS:    "
                    f"{oos_start} -> "
                    f"{oos_end}"
                )

                print(
                    "-" * 78
                )

                # --------------------------------------------
                # Capitals
                # --------------------------------------------

                for capital in STARTING_CAPITALS:

                    try:

                        (
                            row,
                            trade_rows,
                        ) = run_oos(
                            asset=asset,
                            capital=capital,
                            df=df,
                            fold_number=fold_number,
                            warmup_start=warmup_start,
                            oos_start=oos_start,
                            oos_end=oos_end,
                        )

                        results.append(
                            row
                        )

                        all_trades.extend(
                            trade_rows
                        )

                        print(
                            f"{asset} | "
                            f"Fold "
                            f"{fold_number} | "
                            f"${capital:.2f} | "
                            f"Return "
                            f"{row['return_pct']:.4f}% | "
                            f"Trades "
                            f"{row['trades']} | "
                            f"PF "
                            f"{row['profit_factor']:.4f} | "
                            f"DD "
                            f"{row['max_drawdown_pct']:.4f}% | "
                            f"TradeRows "
                            f"{len(trade_rows)}"
                        )

                    except Exception as exc:

                        failures.append(
                            {
                                "asset":
                                    asset,

                                "fold":
                                    fold_number,

                                "capital":
                                    capital,

                                "error":
                                    str(exc),
                            }
                        )

                        print()

                        print(
                            "OOS TEST FAILED"
                        )

                        print(
                            f"Asset:   "
                            f"{asset}"
                        )

                        print(
                            f"Fold:    "
                            f"{fold_number}"
                        )

                        print(
                            f"Capital: "
                            f"${capital:.2f}"
                        )

                        print(
                            f"Error:   "
                            f"{exc}"
                        )

                        traceback.print_exc()

        except Exception as exc:

            failures.append(
                {
                    "asset":
                        asset,

                    "fold":
                        0,

                    "capital":
                        0.0,

                    "error":
                        str(exc),
                }
            )

            print()

            print(
                "ASSET PREPARATION FAILED"
            )

            print(
                f"Asset: "
                f"{asset}"
            )

            print(
                f"Error: "
                f"{exc}"
            )

            traceback.print_exc()

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not results:

        print()

        print(
            "NO OOS RESULTS PRODUCED."
        )

        return 1

    # --------------------------------------------------------
    # Create dataframes
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    trade_df = pd.DataFrame(
        all_trades
    )

    # --------------------------------------------------------
    # Write CSVs
    # --------------------------------------------------------

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    trade_df.to_csv(
        TRADES_OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    header(
        "OOS FINAL REPORT"
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    # ========================================================
    # TRADE DIAGNOSTICS
    # ========================================================

    header(
        "TRADE DIAGNOSTICS"
    )

    print(
        f"Detailed trade rows: "
        f"{len(trade_df)}"
    )

    if not trade_df.empty:

        total_wins = int(
            trade_df["win"].sum()
        )

        total_losses = int(
            (
                trade_df[
                    "net_profit"
                ]
                < 0
            ).sum()
        )

        print(
            f"Wins:   "
            f"{total_wins}"
        )

        print(
            f"Losses: "
            f"{total_losses}"
        )

        # ----------------------------------------------------
        # Losses by asset
        # ----------------------------------------------------

        print()

        print(
            "Losses by asset:"
        )

        losses_by_asset = (
            trade_df[
                trade_df[
                    "net_profit"
                ] < 0
            ]
            .groupby(
                "asset"
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(
            losses_by_asset.to_string()
        )

        # ----------------------------------------------------
        # Losses by exit reason
        # ----------------------------------------------------

        print()

        print(
            "Losses by exit reason:"
        )

        losses_by_exit = (
            trade_df[
                trade_df[
                    "net_profit"
                ] < 0
            ]
            .groupby(
                "exit_reason"
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(
            losses_by_exit.to_string()
        )

        # ----------------------------------------------------
        # Losses by asset + exit
        # ----------------------------------------------------

        print()

        print(
            "Losses by asset / exit reason:"
        )

        losses_matrix = (
            trade_df[
                trade_df[
                    "net_profit"
                ] < 0
            ]
            .groupby(
                [
                    "asset",
                    "exit_reason",
                ]
            )
            .size()
        )

        print(
            losses_matrix.to_string()
        )

        # ----------------------------------------------------
        # Average R
        # ----------------------------------------------------

        print()

        print(
            "Average R by asset:"
        )

        avg_r = (
            trade_df
            .groupby(
                "asset"
            )[
                "r_multiple"
            ]
            .mean()
        )

        print(
            avg_r.to_string()
        )

        # ----------------------------------------------------
        # Average R by result
        # ----------------------------------------------------

        print()

        print(
            "Average R by result:"
        )

        avg_r_result = (
            trade_df
            .groupby(
                "result_class"
            )[
                "r_multiple"
            ]
            .mean()
        )

        print(
            avg_r_result.to_string()
        )

        # ----------------------------------------------------
        # Exit reasons overall
        # ----------------------------------------------------

        print()

        print(
            "All trades by exit reason:"
        )

        exit_summary = (
            trade_df
            .groupby(
                "exit_reason"
            )
            .agg(
                trades=(
                    "net_profit",
                    "count",
                ),
                net_profit=(
                    "net_profit",
                    "sum",
                ),
                avg_r=(
                    "r_multiple",
                    "mean",
                ),
            )
            .sort_values(
                "net_profit"
            )
        )

        print(
            exit_summary.to_string()
        )

        # ----------------------------------------------------
        # ADX diagnostics
        # ----------------------------------------------------

        if "entry_adx_14" in trade_df:

            print()

            print(
                "Average entry ADX:"
            )

            adx_summary = (
                trade_df
                .groupby(
                    "result_class"
                )[
                    "entry_adx_14"
                ]
                .mean()
            )

            print(
                adx_summary.to_string()
            )

        # ----------------------------------------------------
        # ATR diagnostics
        # ----------------------------------------------------

        if "entry_atr_14" in trade_df:

            print()

            print(
                "Average entry ATR:"
            )

            atr_summary = (
                trade_df
                .groupby(
                    "result_class"
                )[
                    "entry_atr_14"
                ]
                .mean()
            )

            print(
                atr_summary.to_string()
            )

    # ========================================================
    # ROBUSTNESS SUMMARY
    # ========================================================

    header(
        "ROBUSTNESS SUMMARY"
    )

    for asset in ASSETS:

        asset_df = result_df[
            result_df[
                "asset"
            ] == asset
        ]

        if asset_df.empty:
            continue

        # One result per fold.
        # Capital does not change the percentage return
        # for this engine, so first row is representative.

        fold_returns = (
            asset_df
            .groupby(
                "fold"
            )[
                "return_pct"
            ]
            .first()
        )

        positive_folds = int(
            (
                fold_returns > 0
            ).sum()
        )

        total_folds = len(
            fold_returns
        )

        avg_return = float(
            fold_returns.mean()
        )

        print()

        print(
            asset
        )

        print(
            f"Positive OOS folds: "
            f"{positive_folds}/"
            f"{total_folds}"
        )

        print(
            f"Average OOS return: "
            f"{avg_return:.4f}%"
        )

        print(
            "Fold returns: "
            +
            ", ".join(
                f"{value:.4f}%"
                for value
                in fold_returns.tolist()
            )
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()

    print(
        f"Results written to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Trade diagnostics written to: "
        f"{TRADES_OUTPUT_FILE}"
    )

    print()

    print(
        "=" * 78
    )

    if failures:

        print(
            "OOS TEST FINISHED WITH FAILURES"
        )

        print(
            f"Failures: "
            f"{len(failures)}"
        )

        print(
            "NO LIVE TRADING."
        )

        print(
            "NO REAL ORDERS."
        )

        return 1

    print(
        "V7-S0 WALK-FORWARD / "
        "OOS TEST COMPLETED"
    )

    print(
        "NO LIVE TRADING."
    )

    print(
        "NO REAL ORDERS."
    )

    print(
        "=" * 78
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )
