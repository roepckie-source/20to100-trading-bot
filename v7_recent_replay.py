# ============================================================
# 20to100 Trading Bot
# V7-S0 RECENT MARKET REPLAY
#
# Purpose:
#   Fast validation of V7-S0 in the most recent market period.
#
# Strategy:
#   V6-C entry
#   V7-S0 survival/risk layer
#
# Assets:
#   BTC / ETH / SOL
#
# Timeframe:
#   5m market data -> 1h strategy timeframe
#
# Period:
#   Last 90 calendar days available in the dataset
#
# IMPORTANT:
#   Uses the EXISTING V7SurvivalEngine.
#   No parameter optimization.
#   No strategy changes.
# ============================================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.v7_survival_engine import V7SurvivalEngine
from strategy.indicators import calculate_indicators


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

ASSETS = [
    "BTC_USDT_5m",
    "ETH_USDT_5m",
    "SOL_USDT_5m",
]

STARTING_BALANCE = 20.0

BASE_RISK_PER_TRADE = 0.01

FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

ATR_STOP_MULTIPLIER = 3.0
TRAILING_ATR_MULTIPLIER = 3.0

ADX_MIN = 20.0

MAX_DAILY_LOSS = 0.05

MAX_CONSECUTIVE_LOSSES = 3
LOSS_COOLDOWN_BARS = 24

GLOBAL_MAX_DRAWDOWN = 0.20

VARIANT = "V6_C"

RECENT_DAYS = 90

RESULTS_FILE = ROOT / "v7_recent_replay_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data(asset_name: str) -> pd.DataFrame:

    path = DATA_DIR / f"{asset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    print()
    print("=" * 70)
    print(f"LOADING {asset_name}")
    print("=" * 70)

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
        subset=["timestamp"],
        keep="last",
    )

    df = df.set_index(
        "timestamp"
    )

    df = df[required].copy()

    print(
        f"5m rows: {len(df):,}"
    )

    print(
        f"Data start: {df.index.min()}"
    )

    print(
        f"Data end:   {df.index.max()}"
    )

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
# PREPARE DATA
# ============================================================

def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    hourly = resample_to_1h(
        df
    )

    if len(hourly) < 300:

        raise ValueError(
            "Not enough hourly data."
        )

    hourly = calculate_indicators(
        hourly.copy()
    )

    return hourly


# ============================================================
# RUN ONE RECENT REPLAY
# ============================================================

def run_recent_replay(
    asset_name: str,
) -> dict:

    df_5m = load_data(
        asset_name
    )

    df_1h = prepare_data(
        df_5m
    )

    # --------------------------------------------------------
    # Select last 90 calendar days
    # --------------------------------------------------------

    end_time = df_1h.index.max()

    start_time = (
        end_time
        - pd.Timedelta(
            days=RECENT_DAYS
        )
    )

    recent = df_1h[
        (df_1h.index >= start_time)
        & (df_1h.index <= end_time)
    ].copy()

    if len(recent) < 100:

        raise ValueError(
            f"{asset_name}: "
            f"not enough recent hourly data "
            f"({len(recent)} rows)"
        )

    print()
    print(
        f"1h rows: {len(df_1h):,}"
    )

    print(
        f"Replay start: {recent.index.min()}"
    )

    print(
        f"Replay end:   {recent.index.max()}"
    )

    print(
        f"Replay bars:  {len(recent):,}"
    )

    # --------------------------------------------------------
    # V7-S0 ENGINE
    #
    # IMPORTANT:
    # base_risk_per_trade is the correct current
    # V7SurvivalEngine argument.
    # --------------------------------------------------------

    engine = V7SurvivalEngine(

        starting_balance=STARTING_BALANCE,

        base_risk_per_trade=BASE_RISK_PER_TRADE,

        fee_rate=FEE_RATE,

        slippage_rate=SLIPPAGE_RATE,

        atr_stop_multiplier=(
            ATR_STOP_MULTIPLIER
        ),

        trailing_atr_multiplier=(
            TRAILING_ATR_MULTIPLIER
        ),

        adx_min=ADX_MIN,

        variant=VARIANT,

        max_daily_loss=MAX_DAILY_LOSS,

        max_consecutive_losses=(
            MAX_CONSECUTIVE_LOSSES
        ),

        loss_cooldown_bars=(
            LOSS_COOLDOWN_BARS
        ),

        global_max_drawdown=(
            GLOBAL_MAX_DRAWDOWN
        ),
    )

    # --------------------------------------------------------
    # RUN ENGINE
    # --------------------------------------------------------

    result = engine.run(
        recent
    )

    if result is None:

        raise RuntimeError(
            f"{asset_name}: "
            "engine returned no result"
        )

    # --------------------------------------------------------
    # Extract metrics safely
    # --------------------------------------------------------

    final_balance = float(
        result.get(
            "final_balance",
            np.nan,
        )
    )

    return_pct = float(
        result.get(
            "return_pct",
            np.nan,
        )
    )

    trades = int(
        result.get(
            "trades",
            0,
        )
    )

    wins = int(
        result.get(
            "wins",
            0,
        )
    )

    losses = int(
        result.get(
            "losses",
            0,
        )
    )

    win_rate = float(
        result.get(
            "win_rate",
            np.nan,
        )
    )

    profit_factor = float(
        result.get(
            "profit_factor",
            np.nan,
        )
    )

    expectancy = float(
        result.get(
            "expectancy",
            np.nan,
        )
    )

    max_drawdown = float(
        result.get(
            "max_drawdown_pct",
            np.nan,
        )
    )

    fees = float(
        result.get(
            "fees",
            0.0,
        )
    )

    slippage = float(
        result.get(
            "slippage",
            0.0,
        )
    )

    kill_switch = bool(
        result.get(
            "kill_switch",
            False,
        )
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    row = {

        "asset": asset_name,

        "variant": VARIANT,

        "start": recent.index.min(),

        "end": recent.index.max(),

        "bars": len(recent),

        "starting_balance":
            STARTING_BALANCE,

        "final_balance":
            final_balance,

        "return_pct":
            return_pct,

        "trades":
            trades,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate_pct":
            win_rate,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "max_drawdown_pct":
            max_drawdown,

        "fees":
            fees,

        "slippage":
            slippage,

        "kill_switch":
            kill_switch,
    }

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(f"RESULT {asset_name}")
    print("-" * 70)

    print(
        f"Start balance : "
        f"{STARTING_BALANCE:.4f}"
    )

    print(
        f"Final balance : "
        f"{final_balance:.4f}"
    )

    print(
        f"Return        : "
        f"{return_pct:.4f}%"
    )

    print(
        f"Trades        : "
        f"{trades}"
    )

    print(
        f"Wins / Losses : "
        f"{wins} / {losses}"
    )

    print(
        f"Win rate      : "
        f"{win_rate:.2f}%"
    )

    print(
        f"Profit Factor : "
        f"{profit_factor:.4f}"
    )

    print(
        f"Expectancy    : "
        f"{expectancy:.6f}"
    )

    print(
        f"Max Drawdown  : "
        f"{max_drawdown:.4f}%"
    )

    print(
        f"Fees          : "
        f"{fees:.6f}"
    )

    print(
        f"Slippage      : "
        f"{slippage:.6f}"
    )

    print(
        f"Kill Switch   : "
        f"{kill_switch}"
    )

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("20to100 TRADING BOT")
    print("V7-S0 RECENT MARKET REPLAY")
    print("=" * 70)

    print()
    print("Strategy : V6-C")
    print("Layer    : V7-S0")
    print("Assets   : BTC + ETH + SOL")
    print("TF       : 5m -> 1h")
    print(
        f"Period   : Last {RECENT_DAYS} days"
    )
    print(
        f"Capital  : {STARTING_BALANCE:.2f} USDT"
    )

    print()
    print("Risk configuration:")
    print(
        f"Base risk       : "
        f"{BASE_RISK_PER_TRADE * 100:.2f}%"
    )
    print(
        f"Daily loss      : "
        f"{MAX_DAILY_LOSS * 100:.2f}%"
    )
    print(
        f"Max loss streak : "
        f"{MAX_CONSECUTIVE_LOSSES}"
    )
    print(
        f"Cooldown bars   : "
        f"{LOSS_COOLDOWN_BARS}"
    )
    print(
        f"Global DD       : "
        f"{GLOBAL_MAX_DRAWDOWN * 100:.2f}%"
    )

    print()
    print("=" * 70)

    results = []

    for asset in ASSETS:

        try:

            result = run_recent_replay(
                asset
            )

            results.append(
                result
            )

        except Exception as exc:

            print()
            print(
                f"ERROR {asset}: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if results:

        results_df = pd.DataFrame(
            results
        )

        results_df.to_csv(
            RESULTS_FILE,
            index=False,
        )

        print()
        print("=" * 70)
        print("V7-S0 RECENT REPLAY SUMMARY")
        print("=" * 70)

        print(
            results_df.to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Aggregate statistics
        # ----------------------------------------------------

        valid_returns = pd.to_numeric(
            results_df["return_pct"],
            errors="coerce",
        )

        valid_pf = pd.to_numeric(
            results_df["profit_factor"],
            errors="coerce",
        )

        valid_dd = pd.to_numeric(
            results_df["max_drawdown_pct"],
            errors="coerce",
        )

        print()
        print("=" * 70)
        print("AGGREGATE")
        print("=" * 70)

        print(
            f"Assets tested       : "
            f"{len(results_df)}"
        )

        print(
            f"Average return      : "
            f"{valid_returns.mean():.4f}%"
        )

        print(
            f"Median return       : "
            f"{valid_returns.median():.4f}%"
        )

        print(
            f"Average PF          : "
            f"{valid_pf.mean():.4f}"
        )

        print(
            f"Median PF           : "
            f"{valid_pf.median():.4f}"
        )

        print(
            f"Positive assets     : "
            f"{(valid_returns > 0).sum()}"
            f"/{len(valid_returns)}"
        )

        print(
            f"Worst return       : "
            f"{valid_returns.min():.4f}%"
        )

        print(
            f"Best return        : "
            f"{valid_returns.max():.4f}%"
        )

        print(
            f"Worst drawdown     : "
            f"{valid_dd.min():.4f}%"
        )

        print()
        print(
            f"Results saved to: "
            f"{RESULTS_FILE}"
        )

    else:

        print()
        print("=" * 70)
        print("NO RESULTS")
        print("=" * 70)

        raise RuntimeError(
            "No asset produced a valid result."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
