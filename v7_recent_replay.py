# ============================================================
# 20to100 Trading Bot
# V7-S0 RECENT MARKET REPLAY - ROBUST
#
# V6-C Entry
# V7-S0 Survival Layer
#
# 5m -> 1h
# Last 90 calendar days
#
# NO optimization
# NO strategy changes
# NO live trading
# ============================================================

from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path

import pandas as pd

from backtest.v7_survival_engine import V7SurvivalEngine
from strategy.indicators import calculate_indicators


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


ENGINE_REQUIRED_COLUMNS = [
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
# LOAD DATA
# ============================================================

def load_data(asset_name: str) -> pd.DataFrame:

    path = DATA_DIR / f"{asset_name}.csv"

    print()
    print("=" * 70)
    print(f"LOADING {asset_name}")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

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
# RESAMPLE
# ============================================================

def resample_to_1h(
    df: pd.DataFrame,
) -> pd.DataFrame:

    hourly = (
        df.resample("1h")
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

    if hourly.empty:
        raise RuntimeError(
            "5m -> 1h resampling produced no data"
        )

    return hourly


# ============================================================
# INDICATORS
# ============================================================

def prepare_data(
    df_5m: pd.DataFrame,
) -> pd.DataFrame:

    hourly = resample_to_1h(
        df_5m
    )

    print(
        f"1h rows: {len(hourly):,}"
    )

    if len(hourly) < 400:

        raise ValueError(
            f"Not enough hourly data: "
            f"{len(hourly)}"
        )

    # VERY IMPORTANT:
    #
    # Indicators are calculated on the FULL
    # historical dataset BEFORE selecting the
    # last 90 days.
    #
    # This preserves EMA / ATR / Donchian
    # warm-up history.

    hourly = calculate_indicators(
        hourly.copy()
    )

    missing = [
        column
        for column in ENGINE_REQUIRED_COLUMNS
        if column not in hourly.columns
    ]

    if missing:

        raise ValueError(
            "Missing indicator columns: "
            f"{missing}"
        )

    return hourly


# ============================================================
# SELECT RECENT PERIOD
# ============================================================

def select_recent(
    df_1h: pd.DataFrame,
):

    end_time = df_1h.index.max()

    start_time = (
        end_time
        - pd.Timedelta(
            days=RECENT_DAYS
        )
    )

    recent = df_1h[
        (
            df_1h.index >= start_time
        )
        &
        (
            df_1h.index <= end_time
        )
    ].copy()

    if len(recent) < 100:

        raise ValueError(
            f"Only {len(recent)} recent "
            f"hourly rows available"
        )

    # Check required indicator data

    bad = (
        recent[
            ENGINE_REQUIRED_COLUMNS
        ]
        .isna()
        .sum()
    )

    bad = bad[
        bad > 0
    ]

    if not bad.empty:

        raise ValueError(
            "NaN values in replay data: "
            f"{bad.to_dict()}"
        )

    return (
        recent,
        start_time,
        end_time,
    )


# ============================================================
# ENGINE
# ============================================================

def build_engine():

    signature = inspect.signature(
        V7SurvivalEngine.__init__
    )

    parameters = signature.parameters

    # This is the important compatibility check.
    #
    # Current V7-S0 uses:
    #
    # base_risk_per_trade
    #
    # NOT:
    #
    # risk_per_trade

    if (
        "base_risk_per_trade"
        not in parameters
    ):

        raise RuntimeError(
            "V7SurvivalEngine API mismatch: "
            "base_risk_per_trade not found."
        )

    return V7SurvivalEngine(

        starting_balance=(
            STARTING_BALANCE
        ),

        base_risk_per_trade=(
            BASE_RISK_PER_TRADE
        ),

        fee_rate=(
            FEE_RATE
        ),

        slippage_rate=(
            SLIPPAGE_RATE
        ),

        atr_stop_multiplier=(
            ATR_STOP_MULTIPLIER
        ),

        trailing_atr_multiplier=(
            TRAILING_ATR_MULTIPLIER
        ),

        adx_min=(
            ADX_MIN
        ),

        variant=(
            VARIANT
        ),

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
    )


# ============================================================
# RUN ONE ASSET
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

    (
        recent,
        start_time,
        end_time,
    ) = select_recent(
        df_1h
    )

    print()
    print(
        f"Replay target start: "
        f"{start_time}"
    )

    print(
        f"Replay target end:   "
        f"{end_time}"
    )

    print(
        f"Replay actual start: "
        f"{recent.index.min()}"
    )

    print(
        f"Replay actual end:   "
        f"{recent.index.max()}"
    )

    print(
        f"Replay bars:         "
        f"{len(recent):,}"
    )

    print()
    print(
        "Starting V7-S0 engine..."
    )

    engine = build_engine()

    result = engine.run(
        recent
    )

    if not isinstance(
        result,
        dict
    ):

        raise RuntimeError(
            "Engine returned "
            f"{type(result).__name__}, "
            "expected dict"
        )

    required_result_keys = [

        "final_balance",
        "return_pct",
        "trades",
        "wins",
        "losses",
        "win_rate",
        "profit_factor",
        "expectancy",
        "max_drawdown_pct",
        "fees",
        "slippage_cost",
        "kill_switch",

    ]

    missing = [
        key
        for key in required_result_keys
        if key not in result
    ]

    if missing:

        raise RuntimeError(
            "Engine result missing keys: "
            f"{missing}"
        )

    final_balance = float(
        result[
            "final_balance"
        ]
    )

    return_pct = float(
        result[
            "return_pct"
        ]
    )

    trades = int(
        result[
            "trades"
        ]
    )

    wins = int(
        result[
            "wins"
        ]
    )

    losses = int(
        result[
            "losses"
        ]
    )

    win_rate = float(
        result[
            "win_rate"
        ]
    )

    profit_factor = float(
        result[
            "profit_factor"
        ]
    )

    expectancy = float(
        result[
            "expectancy"
        ]
    )

    max_drawdown = float(
        result[
            "max_drawdown_pct"
        ]
    )

    fees = float(
        result[
            "fees"
        ]
    )

    slippage = float(
        result[
            "slippage_cost"
        ]
    )

    kill_switch = bool(
        result[
            "kill_switch"
        ]
    )

    row = {

        "asset":
            asset_name,

        "variant":
            VARIANT,

        "start":
            recent.index.min(),

        "end":
            recent.index.max(),

        "bars":
            len(recent),

        "starting_balance":
            STARTING_BALANCE,

        "final_balance":
            final_balance,

        "profit":
            final_balance
            - STARTING_BALANCE,

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

        "slippage_cost":
            slippage,

        "kill_switch":
            kill_switch,
    }

    print()
    print(
        "-" * 70
    )

    print(
        f"RESULT {asset_name}"
    )

    print(
        "-" * 70
    )

    print(
        f"Final balance : "
        f"{final_balance:.6f}"
    )

    print(
        f"Profit        : "
        f"{final_balance - STARTING_BALANCE:.6f}"
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

def main() -> int:

    print()
    print(
        "=" * 70
    )

    print(
        "20to100 TRADING BOT"
    )

    print(
        "V7-S0 RECENT MARKET REPLAY"
    )

    print(
        "=" * 70
    )

    print(
        "Strategy : V6-C"
    )

    print(
        "Layer    : V7-S0"
    )

    print(
        "Assets   : BTC + ETH + SOL"
    )

    print(
        "TF       : 5m -> 1h"
    )

    print(
        f"Period   : "
        f"Last {RECENT_DAYS} days"
    )

    print(
        f"Capital  : "
        f"{STARTING_BALANCE:.2f} USDT"
    )

    print()
    print(
        "NO LIVE TRADING"
    )

    print(
        "NO EXCHANGE API KEYS"
    )

    print(
        "NO REAL ORDERS"
    )

    print(
        "=" * 70
    )

    results = []
    failures = []

    for asset in ASSETS:

        try:

            result = run_recent_replay(
                asset
            )

            results.append(
                result
            )

        except Exception as exc:

            failures.append(
                asset
            )

            print()
            print(
                "!" * 70
            )

            print(
                f"FAILED {asset}"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print()
            traceback.print_exc()

            print(
                "!" * 70
            )

    # ========================================================
    # RESULTS
    # ========================================================

    if results:

        results_df = pd.DataFrame(
            results
        )

        results_df.to_csv(
            RESULTS_FILE,
            index=False,
        )

        print()
        print(
            "=" * 70
        )

        print(
            "V7-S0 RECENT REPLAY SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            results_df.to_string(
                index=False
            )
        )

        valid_returns = pd.to_numeric(
            results_df[
                "return_pct"
            ],
            errors="coerce",
        )

        valid_pf = pd.to_numeric(
            results_df[
                "profit_factor"
            ],
            errors="coerce",
        )

        valid_dd = pd.to_numeric(
            results_df[
                "max_drawdown_pct"
            ],
            errors="coerce",
        )

        print()
        print(
            "=" * 70
        )

        print(
            "AGGREGATE"
        )

        print(
            "=" * 70
        )

        print(
            f"Assets successful : "
            f"{len(results)}/{len(ASSETS)}"
        )

        print(
            f"Average return    : "
            f"{valid_returns.mean():.4f}%"
        )

        print(
            f"Median return     : "
            f"{valid_returns.median():.4f}%"
        )

        print(
            f"Average PF        : "
            f"{valid_pf.mean():.4f}"
        )

        print(
            f"Median PF         : "
            f"{valid_pf.median():.4f}"
        )

        print(
            f"Positive assets   : "
            f"{int((valid_returns > 0).sum())}"
            f"/{len(valid_returns)}"
        )

        print(
            f"Worst return      : "
            f"{valid_returns.min():.4f}%"
        )

        print(
            f"Best return       : "
            f"{valid_returns.max():.4f}%"
        )

        print(
            f"Worst drawdown    : "
            f"{valid_dd.min():.4f}%"
        )

        print()
        print(
            f"Results saved to: "
            f"{RESULTS_FILE}"
        )

    # ========================================================
    # HARD FAILURE
    # ========================================================

    if failures:

        print()
        print(
            "=" * 70
        )

        print(
            "V7-S0 RECENT REPLAY FAILED"
        )

        print(
            "=" * 70
        )

        print(
            "Failed assets: "
            + ", ".join(failures)
        )

        return 1

    if len(results) != len(ASSETS):

        print(
            "ERROR: incomplete asset coverage"
        )

        return 1

    print()
    print(
        "=" * 70
    )

    print(
        "V7-S0 RECENT REPLAY PASSED"
    )

    print(
        "=" * 70
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
