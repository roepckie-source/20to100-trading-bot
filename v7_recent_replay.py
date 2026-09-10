from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

from backtest.v7_survival_engine import V7SurvivalEngine
from strategy.indicators import calculate_indicators


# ============================================================
# CONFIG
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

RECENT_DAYS = 90

OUTPUT_FILE = "v7_recent_replay_results.csv"


# ============================================================
# REQUIRED ENGINE API
# ============================================================

REQUIRED_ENGINE_COLUMNS = [
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


# ============================================================
# HELPERS
# ============================================================

def print_header(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def load_data(asset: str) -> pd.DataFrame:
    path = f"data/{asset}.csv"

    print()
    print(f"Loading: {path}")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}"
        )

    df = pd.read_csv(path)

    print(f"Raw rows: {len(df):,}")

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

        df = df.dropna(subset=["timestamp"])

        df = df.set_index("timestamp")

    elif isinstance(df.index, pd.DatetimeIndex):

        df.index = pd.to_datetime(
            df.index,
            utc=True,
            errors="coerce",
        )

        df = df[~df.index.isna()]

    else:

        raise ValueError(
            f"{asset}: no timestamp column and no DatetimeIndex"
        )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = df.sort_index()

    df = df.loc[
        ~df.index.duplicated(keep="last")
    ]

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    print(
        f"Clean rows: {len(df):,}"
    )

    print(
        f"Data range: "
        f"{df.index.min()} -> {df.index.max()}"
    )

    return df


# ============================================================
# RESAMPLE 5M -> 1H
# ============================================================

def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing OHLCV columns: {missing}"
        )

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
        .dropna()
    )

    return result


# ============================================================
# PREPARE INDICATORS
# ============================================================

def prepare_data(asset: str) -> pd.DataFrame:

    print_header(
        f"PREPARING {asset}"
    )

    df_5m = load_data(asset)

    print(
        f"5m candles: {len(df_5m):,}"
    )

    df_1h = resample_to_1h(df_5m)

    print(
        f"1h candles: {len(df_1h):,}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Calculate indicators on FULL history first.
    #
    # This avoids destroying indicator warm-up data.
    # Only AFTER indicators are calculated do we select
    # the recent 90-day replay period.
    # --------------------------------------------------------

    print("Calculating indicators...")

    df_1h = calculate_indicators(
        df_1h.copy()
    )

    # --------------------------------------------------------
    # CHECK ENGINE COLUMNS
    # --------------------------------------------------------

    missing = [
        column
        for column in REQUIRED_ENGINE_COLUMNS
        if column not in df_1h.columns
    ]

    if missing:

        raise ValueError(
            f"{asset}: missing engine columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # RECENT 90 DAYS
    # --------------------------------------------------------

    latest_timestamp = df_1h.index.max()

    start_timestamp = (
        latest_timestamp
        - pd.Timedelta(days=RECENT_DAYS)
    )

    df_recent = df_1h.loc[
        df_1h.index >= start_timestamp
    ].copy()

    print()
    print(
        "Recent replay period:"
    )

    print(
        f"START: {df_recent.index.min()}"
    )

    print(
        f"END:   {df_recent.index.max()}"
    )

    print(
        f"ROWS:  {len(df_recent):,}"
    )

    if len(df_recent) < 100:

        raise ValueError(
            f"{asset}: only "
            f"{len(df_recent)} recent rows"
        )

    return df_recent


# ============================================================
# ENGINE API CHECK
# ============================================================

def check_engine_api() -> None:

    print_header(
        "CHECKING V7-S0 ENGINE API"
    )

    engine_init_text = (
        V7SurvivalEngine.__init__.__code__
        .co_varnames
    )

    print(
        "Constructor arguments:"
    )

    print(
        engine_init_text
    )

    if "base_risk_per_trade" not in engine_init_text:

        raise RuntimeError(
            "V7SurvivalEngine does not expose "
            "'base_risk_per_trade'. "
            "The replay script is incompatible "
            "with the current engine."
        )

    print(
        "PASS: base_risk_per_trade found"
    )


# ============================================================
# RUN ONE BACKTEST
# ============================================================

def run_one(
    asset: str,
    capital: float,
) -> dict:

    print_header(
        f"V7-S0 RECENT REPLAY | "
        f"{asset} | ${capital:.2f}"
    )

    df = prepare_data(asset)

    engine = V7SurvivalEngine(
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

    print(
        f"Starting capital: ${capital:.2f}"
    )

    print(
        f"Variant: {VARIANT}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    result = engine.run(df)

    if not isinstance(result, dict):

        raise TypeError(
            "Engine result is not a dictionary"
        )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    final_balance = float(
        result.get(
            "final_balance",
            capital,
        )
    )

    return_pct = float(
        result.get(
            "return_pct",
            0.0,
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
            0.0,
        )
    )

    profit_factor = float(
        result.get(
            "profit_factor",
            result.get(
                "pf",
                0.0,
            ),
        )
    )

    expectancy = float(
        result.get(
            "expectancy",
            0.0,
        )
    )

    max_drawdown_pct = float(
        result.get(
            "max_drawdown_pct",
            0.0,
        )
    )

    fees = float(
        result.get(
            "fees",
            0.0,
        )
    )

    slippage_cost = float(
        result.get(
            "slippage_cost",
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
    # RESULT
    # --------------------------------------------------------

    row = {
        "strategy": "V7_S0",
        "variant": VARIANT,
        "asset": asset,
        "starting_capital": capital,
        "final_balance": final_balance,
        "return_pct": return_pct,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown_pct": max_drawdown_pct,
        "fees": fees,
        "slippage_cost": slippage_cost,
        "kill_switch": kill_switch,
        "start_date": str(
            df.index.min()
        ),
        "end_date": str(
            df.index.max()
        ),
        "bars": len(df),
    }

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print(
        f"Asset:          {asset}"
    )

    print(
        f"Capital:        ${capital:.2f}"
    )

    print(
        f"Final balance:  ${final_balance:.4f}"
    )

    print(
        f"Return:         {return_pct:.4f}%"
    )

    print(
        f"Trades:         {trades}"
    )

    print(
        f"Wins:           {wins}"
    )

    print(
        f"Losses:         {losses}"
    )

    print(
        f"Win rate:       {win_rate:.2f}%"
    )

    print(
        f"Profit factor:  {profit_factor:.4f}"
    )

    print(
        f"Expectancy:     {expectancy:.6f}"
    )

    print(
        f"Max drawdown:   {max_drawdown_pct:.4f}%"
    )

    print(
        f"Fees:           ${fees:.6f}"
    )

    print(
        f"Slippage:       ${slippage_cost:.6f}"
    )

    print(
        f"Kill switch:    {kill_switch}"
    )

    print("-" * 70)

    return row


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print_header(
        "V7-S0 RECENT 90-DAY REPLAY"
    )

    print(
        "SURVIVE FIRST. GROW SECOND."
    )

    print()
    print(
        "Strategy:       V7-S0"
    )

    print(
        "Entry:          V6-C"
    )

    print(
        "Timeframe:      1H"
    )

    print(
        "Source data:    5M"
    )

    print(
        "Replay period:  90 days"
    )

    print(
        "Assets:         BTC / ETH / SOL"
    )

    print(
        "Capital:        $20 / $50 / $100 / $250"
    )

    print(
        "Live trading:   NO"
    )

    print(
        "Real orders:    NO"
    )

    print(
        "API keys:       NO"
    )

    # --------------------------------------------------------
    # ENGINE CHECK
    # --------------------------------------------------------

    try:

        check_engine_api()

    except Exception as exc:

        print()
        print(
            "ENGINE API CHECK FAILED"
        )

        print(
            repr(exc)
        )

        traceback.print_exc()

        return 1

    # --------------------------------------------------------
    # RUN ALL
    # --------------------------------------------------------

    results = []

    failures = []

    for asset in ASSETS:

        for capital in STARTING_CAPITALS:

            try:

                result = run_one(
                    asset,
                    capital,
                )

                results.append(
                    result
                )

            except Exception as exc:

                print()
                print("=" * 70)
                print("BACKTEST FAILED")
                print("=" * 70)

                print(
                    f"Asset:   {asset}"
                )

                print(
                    f"Capital: ${capital:.2f}"
                )

                print(
                    f"Error:   {exc}"
                )

                traceback.print_exc()

                failures.append(
                    {
                        "asset": asset,
                        "capital": capital,
                        "error": str(exc),
                    }
                )

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        print()
        print("=" * 70)
        print("NO RESULTS PRODUCED")
        print("=" * 70)

        return 1

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_header(
        "FINAL DATA REPORT"
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Results written to: "
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_header(
        "SUMMARY"
    )

    print(
        f"Successful tests: "
        f"{len(results)}"
    )

    print(
        f"Failed tests:     "
        f"{len(failures)}"
    )

    if failures:

        print()

        for failure in failures:

            print(
                f"FAILED: "
                f"{failure['asset']} "
                f"${failure['capital']:.2f}"
            )

            print(
                f"        {failure['error']}"
            )

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    print()

    print(
        "=================================================="
    )

    if failures:

        print(
            "REPLAY FINISHED WITH FAILURES"
        )

        print(
            "NO LIVE TRADING."
        )

        return 1

    print(
        "V7-S0 RECENT REPLAY COMPLETED"
    )

    print(
        "NO LIVE TRADING."
    )

    print(
        "NO REAL ORDERS."
    )

    print(
        "=================================================="
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
