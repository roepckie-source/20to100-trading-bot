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
# The test uses the same frozen strategy on multiple
# out-of-sample windows.
# ============================================================

ASSETS = [
    "BTC_USDT_5m",
    "ETH_USDT_5m",
    "SOL_USDT_5m",
]

STARTING_CAPITALS = [20.0, 50.0, 100.0, 250.0]

VARIANT = "V6_C"

# Existing workflow downloads approximately 400 days.
REQUIRED_SOURCE_DAYS = 400

# Walk-forward design:
#
# 4 independent OOS windows
# 90 days each
#
# Total OOS coverage = 360 days
#
# The strategy itself is NOT optimized between folds.
WARMUP_DAYS = 40
OOS_DAYS = 90
FOLDS = 4

OUTPUT_FILE = "v7_walk_forward_oos_results.csv"


# ============================================================
# PRINT
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

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in numeric:

        if col not in df.columns:
            raise ValueError(
                f"{asset}: missing column {col}"
            )

        df[col] = pd.to_numeric(
            df[col],
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

    # Frozen V6 indicator implementation.
    #
    # We deliberately use the exact V6 calculation
    # already used by V7-S0.
    df_1h = calculate_indicators(
        df_1h.copy()
    )

    required = [
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
        c
        for c in required
        if c not in df_1h.columns
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
# ENGINE
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
# FOLD WINDOWS
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

    # Work backwards from the latest data.
    #
    # Fold 1 = most recent 90 days
    # Fold 2 = preceding 90 days
    # Fold 3 = preceding 90 days
    # Fold 4 = preceding 90 days

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
# ONE OOS TEST
# ============================================================

def run_oos(
    asset: str,
    capital: float,
    df: pd.DataFrame,
    fold_number: int,
    warmup_start: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> dict:

    # --------------------------------------------------------
    # The strategy remains completely frozen.
    #
    # No parameters are fitted.
    # No optimization occurs.
    #
    # The indicator history is already calculated on the full
    # dataset. The engine starts fresh at the OOS boundary so
    # no position or risk state can leak from another fold.
    # --------------------------------------------------------

    oos = df.loc[
        (df.index >= oos_start)
        & (df.index <= oos_end)
    ].copy()

    if len(oos) < 50:

        raise ValueError(
            f"{asset} fold {fold_number}: "
            f"only {len(oos)} OOS rows"
        )

    engine = create_engine(
        capital
    )

    result = engine.run(
        oos
    )

    return {
        "strategy": "V7_S0",
        "variant": VARIANT,
        "asset": asset,
        "fold": fold_number,
        "warmup_days": WARMUP_DAYS,
        "oos_days": OOS_DAYS,
        "starting_capital": capital,
        "oos_start": str(
            oos.index.min()
        ),
        "oos_end": str(
            oos.index.max()
        ),
        "bars": len(oos),

        "final_balance": float(
            result.get(
                "final_balance",
                capital,
            )
        ),

        "return_pct": float(
            result.get(
                "return_pct",
                0.0,
            )
        ),

        "trades": int(
            result.get(
                "trades",
                0,
            )
        ),

        "wins": int(
            result.get(
                "wins",
                0,
            )
        ),

        "losses": int(
            result.get(
                "losses",
                0,
            )
        ),

        "win_rate_pct": float(
            result.get(
                "win_rate",
                result.get(
                    "win_rate_pct",
                    0.0,
                ),
            )
        ),

        "profit_factor": float(
            result.get(
                "profit_factor",
                result.get(
                    "pf",
                    0.0,
                ),
            )
        ),

        "expectancy": float(
            result.get(
                "expectancy",
                0.0,
            )
        ),

        "max_drawdown_pct": float(
            result.get(
                "max_drawdown_pct",
                0.0,
            )
        ),

        "fees": float(
            result.get(
                "fees",
                0.0,
            )
        ),

        "slippage_cost": float(
            result.get(
                "slippage_cost",
                0.0,
            )
        ),

        "kill_switch": bool(
            result.get(
                "kill_switch",
                False,
            )
        ),
    }


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

    results = []

    failures = []

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

                for capital in (
                    STARTING_CAPITALS
                ):

                    try:

                        row = run_oos(
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
                            f"{row['max_drawdown_pct']:.4f}%"
                        )

                    except Exception as exc:

                        failures.append(
                            {
                                "asset": asset,
                                "fold": fold_number,
                                "capital": capital,
                                "error": str(
                                    exc
                                ),
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
                    "asset": asset,
                    "fold": 0,
                    "capital": 0.0,
                    "error": str(exc),
                }
            )

            print()
            print(
                "ASSET PREPARATION FAILED"
            )

            print(
                f"Asset: {asset}"
            )

            print(
                f"Error: {exc}"
            )

            traceback.print_exc()

    if not results:

        print()
        print(
            "NO OOS RESULTS PRODUCED."
        )

        return 1

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    header(
        "OOS FINAL REPORT"
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    # ========================================================
    # ROBUSTNESS SUMMARY
    # ========================================================

    header(
        "ROBUSTNESS SUMMARY"
    )

    for asset in ASSETS:

        asset_df = result_df[
            result_df["asset"]
            == asset
        ]

        if asset_df.empty:
            continue

        # Capital does not change percentage returns because
        # risk is percentage based. Use the first capital for
        # the fold-level summary.
        fold_returns = (
            asset_df
            .groupby("fold")[
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
            + ", ".join(
                f"{x:.4f}%"
                for x in
                fold_returns.tolist()
            )
        )

    print()
    print(
        f"Results written to: "
        f"{OUTPUT_FILE}"
    )

    print()
    print(
        "=" * 78
    )

    if failures:

        print(
            "OOS TEST FINISHED "
            "WITH FAILURES"
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


if __name__ == "__main__":
    sys.exit(
        main()
    )
