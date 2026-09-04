from pathlib import Path

code = r'''# ============================================================
# 20to100 Trading Bot
# V6_C Monte-Carlo Validation
#
# Frozen strategy:
#   V6_C
#
# Assets:
#   BTC/USDT
#   ETH/USDT
#   SOL/USDT
#
# Walk Forward:
#   12 months TRAIN
#   3 months OOS
#   3 months rolling step
#
# Monte Carlo:
#   10,000 random permutations per OOS window
#
# IMPORTANT:
# This test randomizes the ORDER of the actually realized
# V6_C R-multiples. It tests sequence/path dependency.
# It does NOT create new hypothetical trade outcomes.
# ============================================================

from pathlib import Path
import math
import numpy as np
import pandas as pd

from strategy.strategy_v6 import calculate_indicators
from backtest.v6_engine import V6BacktestEngine


# ============================================================
# SETTINGS
# ============================================================

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
STEP_MONTHS = 3

SIMULATIONS = 10_000

ASSETS = {
    "BTC/USDT": Path("data/BTC_USDT_5m.csv"),
    "ETH/USDT": Path("data/ETH_USDT_5m.csv"),
    "SOL/USDT": Path("data/SOL_USDT_5m.csv"),
}

OUTPUT_DIR = Path("logs")


# ============================================================
# LOAD + RESAMPLE
# ============================================================

def load_asset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"{path} has no timestamp column.")

    required = ["open", "high", "low", "close", "volume"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = (
        df.set_index("timestamp")
        .sort_index()
    )

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


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================

def month_start(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return pd.Timestamp(
        year=ts.year,
        month=ts.month,
        day=1,
        tz="UTC"
    )


def month_windows(index):
    index = pd.DatetimeIndex(index)

    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    start = month_start(index.min())
    end = month_start(index.max())

    windows = []

    train_start = start

    while True:
        train_end = train_start + pd.DateOffset(
            months=TRAIN_MONTHS
        )

        oos_end = train_end + pd.DateOffset(
            months=OOS_MONTHS
        )

        if oos_end > end + pd.Timedelta(hours=1):
            break

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "oos_start": train_end,
                "oos_end": oos_end,
            }
        )

        train_start = train_start + pd.DateOffset(
            months=STEP_MONTHS
        )

    return windows


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def simulate_sequence(
    r_multiples,
    rng,
):
    """
    Compound the sequence at RISK_PER_TRADE per trade.

    Example:
        R = +2.0
        risk = 1%
        balance multiplier = 1 + 0.01 * 2 = 1.02

    Returns:
        final_balance
        return_pct
        max_drawdown_pct
        worst_loss_streak
    """

    if len(r_multiples) == 0:
        return (
            STARTING_BALANCE,
            0.0,
            0.0,
            0,
        )

    sequence = np.asarray(
        r_multiples,
        dtype=float
    ).copy()

    rng.shuffle(sequence)

    balance = STARTING_BALANCE
    peak = balance
    max_drawdown = 0.0

    current_loss_streak = 0
    worst_loss_streak = 0

    for r in sequence:

        balance *= (
            1.0
            + RISK_PER_TRADE * r
        )

        # Numerical safety
        if not np.isfinite(balance):
            balance = 0.0

        balance = max(
            0.0,
            balance
        )

        peak = max(
            peak,
            balance
        )

        if peak > 0:
            drawdown = (
                (balance / peak)
                - 1.0
            ) * 100.0

            max_drawdown = min(
                max_drawdown,
                drawdown
            )

        if r < 0:
            current_loss_streak += 1
            worst_loss_streak = max(
                worst_loss_streak,
                current_loss_streak
            )
        else:
            current_loss_streak = 0

    return (
        balance,
        (
            (balance / STARTING_BALANCE)
            - 1.0
        ) * 100.0,
        max_drawdown,
        worst_loss_streak,
    )


# ============================================================
# RUN REAL V6_C OOS WINDOW
# ============================================================

def run_oos_window(
    symbol,
    full_df,
    window,
):
    oos = full_df[
        (full_df.index >= window["oos_start"])
        & (full_df.index < window["oos_end"])
    ].copy()

    if len(oos) < 10:
        return None

    engine = V6BacktestEngine(
        starting_balance=STARTING_BALANCE,
        risk_per_trade=RISK_PER_TRADE,
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
        atr_stop_multiplier=ATR_STOP_MULTIPLIER,
        trailing_atr_multiplier=TRAILING_ATR_MULTIPLIER,
        adx_min=ADX_MIN,
        max_daily_loss=MAX_DAILY_LOSS,
        max_consecutive_losses=MAX_CONSECUTIVE_LOSSES,
        loss_cooldown_bars=LOSS_COOLDOWN_BARS,
        global_max_drawdown=GLOBAL_MAX_DRAWDOWN,
        variant="V6_C",
    )

    result = engine.run(oos)

    r_values = []

    for trade in getattr(
        engine,
        "trades",
        []
    ):
        r = getattr(
            trade,
            "r_multiple",
            None
        )

        if r is None:
            continue

        try:
            r = float(r)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(r):
            continue

        r_values.append(r)

    return {
        "symbol": symbol,
        "train_start": window["train_start"],
        "train_end": window["train_end"],
        "oos_start": window["oos_start"],
        "oos_end": window["oos_end"],
        "trades": len(r_values),
        "engine_result": result,
        "r_multiples": r_values,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("V6_C MONTE-CARLO VALIDATION")
    print("=" * 70)
    print()
    print("Strategy: V6_C")
    print("Assets: BTC + ETH + SOL")
    print(f"Simulations per OOS window: {SIMULATIONS:,}")
    print("Risk per trade: 1.00%")
    print("Walk Forward: 12m TRAIN / 3m OOS / 3m step")
    print("=" * 70)

    rng = np.random.default_rng(
        20260904
    )

    window_rows = []
    trade_rows = []

    total_windows = 0

    for symbol, path in ASSETS.items():

        print()
        print("=" * 70)
        print(f"LOADING {symbol}")
        print("=" * 70)

        hourly = load_asset(path)

        print(
            f"5m file: {path}"
        )
        print(
            f"1h candles: {len(hourly):,}"
        )
        print(
            f"Range: {hourly.index.min()} -> "
            f"{hourly.index.max()}"
        )

        # CRITICAL:
        # Calculate indicators BEFORE the walk-forward split.
        # This preserves indicator warm-up at the OOS boundary.
        print("Calculating V6 indicators on full dataset...")

        hourly = calculate_indicators(
            hourly
        )

        windows = month_windows(
            hourly.index
        )

        print(
            f"Walk-forward windows: {len(windows)}"
        )

        if not windows:
            print(
                f"WARNING: no windows for {symbol}"
            )
            continue

        for window_number, window in enumerate(
            windows,
            start=1
        ):

            print()
            print(
                f"{symbol} | V6_C | "
                f"W{window_number:02d}/{len(windows)}"
            )
            print(
                f"OOS: "
                f"{window['oos_start'].date()} -> "
                f"{window['oos_end'].date()}"
            )

            try:
                result = run_oos_window(
                    symbol,
                    hourly,
                    window
                )

                if result is None:
                    print(
                        "Skipped: insufficient OOS data."
                    )
                    continue

                r_values = result[
                    "r_multiples"
                ]

                if len(r_values) == 0:
                    print(
                        "No valid R-multiples."
                    )
                    continue

                total_windows += 1

                # Store actual trades
                for trade_number, r in enumerate(
                    r_values,
                    start=1
                ):
                    trade_rows.append(
                        {
                            "symbol": symbol,
                            "window": window_number,
                            "trade": trade_number,
                            "r_multiple": r,
                        }
                    )

                finals = []
                returns = []
                drawdowns = []
                loss_streaks = []

                hits_100 = 0
                below_10 = 0

                for _ in range(
                    SIMULATIONS
                ):

                    (
                        final_balance,
                        return_pct,
                        max_dd,
                        worst_streak,
                    ) = simulate_sequence(
                        r_values,
                        rng
                    )

                    finals.append(
                        final_balance
                    )

                    returns.append(
                        return_pct
                    )

                    drawdowns.append(
                        max_dd
                    )

                    loss_streaks.append(
                        worst_streak
                    )

                    if final_balance >= 100.0:
                        hits_100 += 1

                    if final_balance < 10.0:
                        below_10 += 1

                finals = np.asarray(
                    finals
                )

                returns = np.asarray(
                    returns
                )

                drawdowns = np.asarray(
                    drawdowns
                )

                loss_streaks = np.asarray(
                    loss_streaks
                )

                actual_result = result[
                    "engine_result"
                ]

                row = {
                    "symbol": symbol,
                    "window": window_number,
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
                    "actual_trades": len(
                        r_values
                    ),
                    "actual_final_balance": (
                        actual_result.get(
                            "final_balance",
                            np.nan
                        )
                    ),
                    "actual_return_pct": (
                        actual_result.get(
                            "return_pct",
                            np.nan
                        )
                    ),
                    "actual_profit_factor": (
                        actual_result.get(
                            "profit_factor",
                            np.nan
                        )
                    ),
                    "actual_max_drawdown_pct": (
                        actual_result.get(
                            "max_drawdown_pct",
                            np.nan
                        )
                    ),
                    "median_final_balance": (
                        float(
                            np.median(
                                finals
                            )
                        )
                    ),
                    "p05_final_balance": (
                        float(
                            np.percentile(
                                finals,
                                5
                            )
                        )
                    ),
                    "p95_final_balance": (
                        float(
                            np.percentile(
                                finals,
                                95
                            )
                        )
                    ),
                    "median_return_pct": (
                        float(
                            np.median(
                                returns
                            )
                        )
                    ),
                    "p05_return_pct": (
                        float(
                            np.percentile(
                                returns,
                                5
                            )
                        )
                    ),
                    "p95_return_pct": (
                        float(
                            np.percentile(
                                returns,
                                95
                            )
                        )
                    ),
                    "median_max_drawdown_pct": (
                        float(
                            np.median(
                                drawdowns
                            )
                        )
                    ),
                    "worst_max_drawdown_pct": (
                        float(
                            np.min(
                                drawdowns
                            )
                        )
                    ),
                    "median_worst_loss_streak": (
                        float(
                            np.median(
                                loss_streaks
                            )
                        )
                    ),
                    "worst_loss_streak": (
                        int(
                            np.max(
                                loss_streaks
                            )
                        )
                    ),
                    "probability_reach_100_pct": (
                        hits_100
                        / SIMULATIONS
                        * 100.0
                    ),
                    "probability_below_10_pct": (
                        below_10
                        / SIMULATIONS
                        * 100.0
                    ),
                }

                window_rows.append(
                    row
                )

                print(
                    f"Trades: {len(r_values)}"
                )
                print(
                    f"Actual return: "
                    f"{row['actual_return_pct']:.2f}%"
                )
                print(
                    f"MC median final: "
                    f"${row['median_final_balance']:.2f}"
                )
                print(
                    f"MC P05/P95 final: "
                    f"${row['p05_final_balance']:.2f} / "
                    f"${row['p95_final_balance']:.2f}"
                )
                print(
                    f"MC median DD: "
                    f"{row['median_max_drawdown_pct']:.2f}%"
                )
                print(
                    f"MC worst DD: "
                    f"{row['worst_max_drawdown_pct']:.2f}%"
                )
                print(
                    f"Chance >= $100: "
                    f"{row['probability_reach_100_pct']:.2f}%"
                )
                print(
                    f"Chance < $10: "
                    f"{row['probability_below_10_pct']:.2f}%"
                )

            except Exception as exc:
                print(
                    f"ERROR in {symbol} "
                    f"W{window_number}: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ========================================================
    # SAVE WINDOW RESULTS
    # ========================================================

    window_path = (
        OUTPUT_DIR
        / "v6_c_monte_carlo_window_results.csv"
    )

    trades_path = (
        OUTPUT_DIR
        / "v6_c_monte_carlo_trades.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "v6_c_monte_carlo_summary.csv"
    )

    window_df = pd.DataFrame(
        window_rows
    )

    trades_df = pd.DataFrame(
        trade_rows
    )

    window_df.to_csv(
        window_path,
        index=False
    )

    trades_df.to_csv(
        trades_path,
        index=False
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    summary_rows = []

    if not window_df.empty:

        for symbol in sorted(
            window_df["symbol"].unique()
        ):

            subset = window_df[
                window_df["symbol"] == symbol
            ]

            summary_rows.append(
                {
                    "symbol": symbol,
                    "windows": len(subset),
                    "total_actual_trades": int(
                        subset[
                            "actual_trades"
                        ].sum()
                    ),
                    "median_mc_final_balance": (
                        subset[
                            "median_final_balance"
                        ].median()
                    ),
                    "p05_mc_final_balance": (
                        subset[
                            "p05_final_balance"
                        ].median()
                    ),
                    "p95_mc_final_balance": (
                        subset[
                            "p95_final_balance"
                        ].median()
                    ),
                    "median_mc_return_pct": (
                        subset[
                            "median_return_pct"
                        ].median()
                    ),
                    "median_mc_drawdown_pct": (
                        subset[
                            "median_max_drawdown_pct"
                        ].median()
                    ),
                    "worst_mc_drawdown_pct": (
                        subset[
                            "worst_max_drawdown_pct"
                        ].min()
                    ),
                    "median_probability_reach_100_pct": (
                        subset[
                            "probability_reach_100_pct"
                        ].median()
                    ),
                    "median_probability_below_10_pct": (
                        subset[
                            "probability_below_10_pct"
                        ].median()
                    ),
                }
            )

        # Overall row
        summary_rows.append(
            {
                "symbol": "ALL",
                "windows": len(window_df),
                "total_actual_trades": int(
                    window_df[
                        "actual_trades"
                    ].sum()
                ),
                "median_mc_final_balance": (
                    window_df[
                        "median_final_balance"
                    ].median()
                ),
                "p05_mc_final_balance": (
                    window_df[
                        "p05_final_balance"
                    ].median()
                ),
                "p95_mc_final_balance": (
                    window_df[
                        "p95_final_balance"
                    ].median()
                ),
                "median_mc_return_pct": (
                    window_df[
                        "median_return_pct"
                    ].median()
                ),
                "median_mc_drawdown_pct": (
                    window_df[
                        "median_max_drawdown_pct"
                    ].median()
                ),
                "worst_mc_drawdown_pct": (
                    window_df[
                        "worst_max_drawdown_pct"
                    ].min()
                ),
                "median_probability_reach_100_pct": (
                    window_df[
                        "probability_reach_100_pct"
                    ].median()
                ),
                "median_probability_below_10_pct": (
                    window_df[
                        "probability_below_10_pct"
                    ].median()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("V6_C MONTE-CARLO COMPLETE")
    print("=" * 70)
    print(
        f"Valid OOS windows: {total_windows}"
    )
    print(
        f"Simulations per window: {SIMULATIONS:,}"
    )
    print()
    print(
        f"Window results: {window_path}"
    )
    print(
        f"Trade R-multiples: {trades_path}"
    )
    print(
        f"Summary: {summary_path}"
    )

    if not summary_df.empty:
        print()
        print("=" * 70)
        print("GLOBAL MONTE-CARLO SUMMARY")
        print("=" * 70)
        print(
            summary_df.to_string(
                index=False
            )
        )
    else:
        print()
        print(
            "WARNING: No Monte-Carlo results generated."
        )


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/v6_c_monte_carlo.py")
path.write_text(code, encoding="utf-8")

# Syntax check
compile(code, str(path), "exec")

print(f"Datei erstellt: {path}")
print("Syntaxprüfung: OK")
