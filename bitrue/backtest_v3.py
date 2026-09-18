"""
Bitrue Strategy V3 - Backtest

PAPER / BACKTEST ONLY
NO API
NO LIVE TRADING
NO REAL ORDERS

Chronological evaluation:
- TRAIN: first 70%
- OOS:   final 30%

The strategy itself is unchanged between TRAIN and OOS.
No parameter optimization is performed inside the test.
"""

import sys
from pathlib import Path

import pandas as pd

from strategy_v3 import (
    calculate_indicators,
    evaluate_precomputed,
)


# ============================================================
# CONFIG
# ============================================================

STARTING_CAPITAL = 100.00

TAKER_FEE = 0.0006
SLIPPAGE = 0.0002

MAX_DAILY_LOSS = 0.05

COOLDOWN_MINUTES = 30

TRAIN_FRACTION = 0.70


# ============================================================
# INPUT
# ============================================================

if len(sys.argv) < 2:

    print(
        "Verwendung:"
    )

    print(
        "python bitrue/backtest_v3.py "
        "BTCUSDT_5m.csv"
    )

    sys.exit(1)


DATA_FILE = Path(
    sys.argv[1]
)


if not DATA_FILE.exists():

    print()
    print(
        "FEHLER: Datendatei nicht gefunden:"
    )

    print(DATA_FILE)

    sys.exit(1)


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 70)
print("BITRUE STRATEGY V3 BACKTEST")
print("=" * 70)
print()

data = pd.read_csv(
    DATA_FILE
)

data.columns = [
    str(column)
    .strip()
    .lower()
    for column in data.columns
]


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

missing = [
    column
    for column in required
    if column not in data.columns
]

if missing:

    print(
        "FEHLER: Fehlende Spalten:"
    )

    print(missing)

    sys.exit(1)


# ============================================================
# TIMESTAMP
# ============================================================

timestamp_column = None

for candidate in [
    "timestamp",
    "open_time",
    "time",
    "datetime",
    "date",
]:

    if candidate in data.columns:

        timestamp_column = candidate

        break


if timestamp_column is None:

    print(
        "FEHLER: Keine Zeitspalte gefunden."
    )

    sys.exit(1)


raw_timestamp = data[
    timestamp_column
]


if pd.api.types.is_numeric_dtype(
    raw_timestamp
):

    median_timestamp = (
        raw_timestamp
        .dropna()
        .median()
    )

    if median_timestamp > 10_000_000_000:

        data["timestamp"] = pd.to_datetime(
            raw_timestamp,
            unit="ms",
            utc=True,
        )

    else:

        data["timestamp"] = pd.to_datetime(
            raw_timestamp,
            unit="s",
            utc=True,
        )

else:

    data["timestamp"] = pd.to_datetime(
        raw_timestamp,
        utc=True,
    )


# ============================================================
# CLEAN
# ============================================================

for column in required:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )


data = data.dropna(
    subset=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
)


data = (
    data
    .sort_values("timestamp")
    .drop_duplicates(
        subset=["timestamp"]
    )
    .reset_index(drop=True)
)


if data.empty:

    print(
        "FEHLER: Keine Daten."
    )

    sys.exit(1)


# ============================================================
# TRAIN / OOS SPLIT
# ============================================================

split_index = int(
    len(data) * TRAIN_FRACTION
)


data["period"] = "TRAIN"

data.loc[
    split_index:,
    "period"
] = "OOS"


print(
    f"TRAIN Kerzen: "
    f"{split_index:,}"
)

print(
    f"OOS Kerzen:   "
    f"{len(data) - split_index:,}"
)

print()

print(
    f"TRAIN Start:  "
    f"{data['timestamp'].iloc[0]}"
)

print(
    f"TRAIN Ende:   "
    f"{data['timestamp'].iloc[split_index - 1]}"
)

print()

print(
    f"OOS Start:    "
    f"{data['timestamp'].iloc[split_index]}"
)

print(
    f"OOS Ende:     "
    f"{data['timestamp'].iloc[-1]}"
)

print()


# ============================================================
# INDICATORS
# ============================================================

print(
    "Berechne V3-Indikatoren..."
)

data = calculate_indicators(
    data
)

print(
    "Indikatoren fertig."
)

print()


# ============================================================
# STATE
# ============================================================

balance = STARTING_CAPITAL

position = None

trades = []

current_day = None

day_start_balance = STARTING_CAPITAL

last_exit_timestamp = None


# ============================================================
# DAILY LOSS
# ============================================================

def reset_day(day):

    global current_day
    global day_start_balance

    if current_day != day:

        current_day = day

        day_start_balance = balance


def daily_loss_limit_reached():

    limit = (
        day_start_balance
        * (1.0 - MAX_DAILY_LOSS)
    )

    return balance <= limit


# ============================================================
# ENTRY
# ============================================================

def execute_entry(
    side,
    market_price,
    position_size,
    timestamp,
):

    if side == "LONG":

        execution_price = (
            market_price
            * (1.0 + SLIPPAGE)
        )

    else:

        execution_price = (
            market_price
            * (1.0 - SLIPPAGE)
        )

    quantity = (
        position_size
        / execution_price
    )

    entry_fee = (
        position_size
        * TAKER_FEE
    )

    return {
        "side": side,
        "entry_time": timestamp,
        "entry_market_price": market_price,
        "entry_price": execution_price,
        "quantity": quantity,
        "position_size": position_size,
        "entry_fee": entry_fee,
    }


# ============================================================
# EXIT
# ============================================================

def execute_exit(
    position_state,
    market_price,
    reason,
):

    side = position_state["side"]

    quantity = position_state[
        "quantity"
    ]

    entry_price = position_state[
        "entry_price"
    ]

    if side == "LONG":

        exit_price = (
            market_price
            * (1.0 - SLIPPAGE)
        )

        gross_pnl = (
            exit_price
            - entry_price
        ) * quantity

    else:

        exit_price = (
            market_price
            * (1.0 + SLIPPAGE)
        )

        gross_pnl = (
            entry_price
            - exit_price
        ) * quantity

    exit_notional = abs(
        exit_price * quantity
    )

    exit_fee = (
        exit_notional
        * TAKER_FEE
    )

    total_fees = (
        position_state["entry_fee"]
        + exit_fee
    )

    entry_slippage_cost = (
        abs(
            position_state[
                "entry_market_price"
            ]
            - entry_price
        )
        * quantity
    )

    exit_slippage_cost = (
        abs(
            market_price
            - exit_price
        )
        * quantity
    )

    slippage_cost = (
        entry_slippage_cost
        + exit_slippage_cost
    )

    net_pnl = (
        gross_pnl
        - total_fees
    )

    return {
        "exit_market_price": market_price,
        "exit_price": exit_price,
        "gross_pnl": gross_pnl,
        "fees": total_fees,
        "slippage_cost": slippage_cost,
        "net_pnl": net_pnl,
        "exit_reason": reason,
    }


# ============================================================
# MAIN LOOP
# ============================================================

for row in data.itertuples(
    index=False
):

    timestamp = row.timestamp

    day = timestamp.date()

    reset_day(day)


    # ========================================================
    # EXISTING POSITION
    # ========================================================

    if position is not None:

        side = position["side"]

        stop_loss = position[
            "stop_loss"
        ]

        take_profit = position[
            "take_profit"
        ]

        exit_reason = None

        exit_price = None


        if side == "LONG":

            # Conservative:
            # SL is checked before TP if both occur
            # inside the same 5m candle.

            if row.low <= stop_loss:

                exit_reason = (
                    "STOP_LOSS"
                )

                exit_price = (
                    stop_loss
                )

            elif row.high >= take_profit:

                exit_reason = (
                    "TAKE_PROFIT"
                )

                exit_price = (
                    take_profit
                )


        else:

            if row.high >= stop_loss:

                exit_reason = (
                    "STOP_LOSS"
                )

                exit_price = (
                    stop_loss
                )

            elif row.low <= take_profit:

                exit_reason = (
                    "TAKE_PROFIT"
                )

                exit_price = (
                    take_profit
                )


        # ====================================================
        # EXIT
        # ====================================================

        if exit_reason is not None:

            result = execute_exit(
                position,
                float(exit_price),
                exit_reason,
            )

            net_pnl = result[
                "net_pnl"
            ]

            balance += net_pnl

            duration_minutes = (
                timestamp
                - position["entry_time"]
            ).total_seconds() / 60.0


            trades.append(
                {
                    "entry_time":
                        position["entry_time"],

                    "exit_time":
                        timestamp,

                    "period":
                        position["period"],

                    "side":
                        side,

                    "entry_price":
                        position["entry_price"],

                    "exit_price":
                        result["exit_price"],

                    "market_entry_price":
                        position[
                            "entry_market_price"
                        ],

                    "market_exit_price":
                        result[
                            "exit_market_price"
                        ],

                    "position_size":
                        position[
                            "position_size"
                        ],

                    "quantity":
                        position["quantity"],

                    "gross_pnl":
                        result["gross_pnl"],

                    "fees":
                        result["fees"],

                    "slippage_cost":
                        result[
                            "slippage_cost"
                        ],

                    "net_pnl":
                        result["net_pnl"],

                    "return_pct":
                        (
                            result["net_pnl"]
                            / position[
                                "position_size"
                            ]
                            * 100.0
                        ),

                    "duration_min":
                        duration_minutes,

                    "momentum_pct":
                        position[
                            "momentum_pct"
                        ],

                    "atr_pct":
                        position[
                            "atr_pct"
                        ],

                    "atr_value":
                        position[
                            "atr_value"
                        ],

                    "volume_ratio":
                        position[
                            "volume_ratio"
                        ],

                    "ema_spread_pct":
                        position[
                            "ema_spread_pct"
                        ],

                    "htf_bullish":
                        position[
                            "htf_bullish"
                        ],

                    "htf_bearish":
                        position[
                            "htf_bearish"
                        ],

                    "exit_reason":
                        result[
                            "exit_reason"
                        ],
                }
            )


            position = None

            last_exit_timestamp = (
                timestamp
            )

            continue


    # ========================================================
    # NO POSITION
    # ========================================================

    if position is not None:
        continue


    # ========================================================
    # DAILY LOSS LIMIT
    # ========================================================

    if daily_loss_limit_reached():

        continue


    # ========================================================
    # COOLDOWN
    # ========================================================

    if last_exit_timestamp is not None:

        minutes_since_exit = (
            timestamp
            - last_exit_timestamp
        ).total_seconds() / 60.0

        if (
            minutes_since_exit
            < COOLDOWN_MINUTES
        ):

            continue


    # ========================================================
    # SIGNAL
    # ========================================================

    signal = evaluate_precomputed(
        row,
        balance,
    )


    if not signal.signal:

        continue


    # ========================================================
    # OPEN
    # ========================================================

    position = execute_entry(
        signal.side,
        signal.price,
        signal.position_size,
        timestamp,
    )


    position["period"] = (
        row.period
    )

    position["stop_loss"] = (
        signal.stop_loss
    )

    position["take_profit"] = (
        signal.take_profit
    )

    position["momentum_pct"] = (
        signal.momentum_pct
    )

    position["atr_pct"] = (
        signal.atr_pct
    )

    position["atr_value"] = (
        signal.atr_value
    )

    position["volume_ratio"] = (
        signal.volume_ratio
    )

    position["ema_spread_pct"] = (
        signal.ema_spread_pct
    )

    position["htf_bullish"] = (
        signal.htf_bullish
    )

    position["htf_bearish"] = (
        signal.htf_bearish
    )


# ============================================================
# END OF BACKTEST
# ============================================================

if position is not None:

    last_row = data.iloc[-1]

    timestamp = last_row[
        "timestamp"
    ]

    market_price = float(
        last_row["close"]
    )

    result = execute_exit(
        position,
        market_price,
        "END_OF_BACKTEST",
    )

    balance += result[
        "net_pnl"
    ]

    duration_minutes = (
        timestamp
        - position["entry_time"]
    ).total_seconds() / 60.0


    trades.append(
        {
            "entry_time":
                position["entry_time"],

            "exit_time":
                timestamp,

            "period":
                position["period"],

            "side":
                position["side"],

            "entry_price":
                position["entry_price"],

            "exit_price":
                result["exit_price"],

            "market_entry_price":
                position["entry_market_price"],

            "market_exit_price":
                result[
                    "exit_market_price"
                ],

            "position_size":
                position[
                    "position_size"
                ],

            "quantity":
                position["quantity"],

            "gross_pnl":
                result["gross_pnl"],

            "fees":
                result["fees"],

            "slippage_cost":
                result[
                    "slippage_cost"
                ],

            "net_pnl":
                result["net_pnl"],

            "return_pct":
                (
                    result["net_pnl"]
                    / position[
                        "position_size"
                    ]
                    * 100.0
                ),

            "duration_min":
                duration_minutes,

            "momentum_pct":
                position[
                    "momentum_pct"
                ],

            "atr_pct":
                position[
                    "atr_pct"
                ],

            "atr_value":
                position[
                    "atr_value"
                ],

            "volume_ratio":
                position[
                    "volume_ratio"
                ],

            "ema_spread_pct":
                position[
                    "ema_spread_pct"
                ],

            "htf_bullish":
                position[
                    "htf_bullish"
                ],

            "htf_bearish":
                position[
                    "htf_bearish"
                ],

            "exit_reason":
                "END_OF_BACKTEST",
        }
    )


# ============================================================
# TRADE LOG
# ============================================================

trade_df = pd.DataFrame(
    trades
)

output_file = Path(
    "bitrue_v3_trade_log.csv"
)

trade_df.to_csv(
    output_file,
    index=False,
)


# ============================================================
# STATISTICS
# ============================================================

print()
print("=" * 70)
print("BITRUE STRATEGY V3 - ERGEBNIS")
print("=" * 70)
print()


def print_stats(
    title,
    df,
):

    if df.empty:

        print()
        print(title)
        print("-" * 60)
        print("Keine Trades.")
        print()

        return


    wins = int(
        (df["net_pnl"] > 0).sum()
    )

    losses = int(
        (df["net_pnl"] <= 0).sum()
    )

    trades_count = len(df)

    win_rate = (
        wins
        / trades_count
        * 100.0
    )

    gross = (
        df["gross_pnl"].sum()
    )

    fees = (
        df["fees"].sum()
    )

    slippage = (
        df["slippage_cost"].sum()
    )

    net = (
        df["net_pnl"].sum()
    )

    avg_trade = (
        df["net_pnl"].mean()
    )

    print()
    print(title)
    print("-" * 60)

    print(
        f"Trades:             "
        f"{trades_count}"
    )

    print(
        f"Gewinner:           "
        f"{wins}"
    )

    print(
        f"Verlierer:          "
        f"{losses}"
    )

    print(
        f"Win Rate:           "
        f"{win_rate:.2f}%"
    )

    print(
        f"Brutto P&L:         "
        f"${gross:.4f}"
    )

    print(
        f"Gebühren:           "
        f"${fees:.4f}"
    )

    print(
        f"Slippage:           "
        f"${slippage:.4f}"
    )

    print(
        f"Netto P&L:          "
        f"${net:.4f}"
    )

    print(
        f"Ø Trade:            "
        f"${avg_trade:.6f}"
    )

    print(
        f"Return auf 100$:    "
        f"{net / STARTING_CAPITAL * 100:.2f}%"
    )


# ============================================================
# FULL
# ============================================================

print_stats(
    "GESAMT",
    trade_df,
)


# ============================================================
# TRAIN
# ============================================================

train_df = trade_df[
    trade_df["period"]
    == "TRAIN"
]


print_stats(
    "TRAIN",
    train_df,
)


# ============================================================
# OOS
# ============================================================

oos_df = trade_df[
    trade_df["period"]
    == "OOS"
]


print_stats(
    "OUT-OF-SAMPLE",
    oos_df,
)


# ============================================================
# OOS STATUS
# ============================================================

print()
print("=" * 70)
print("OOS-BEWERTUNG")
print("=" * 70)
print()


if oos_df.empty:

    print(
        "Keine OOS-Trades."
    )

elif oos_df["net_pnl"].sum() > 0:

    print(
        "OOS ist positiv."
    )

else:

    print(
        "OOS ist negativ."
    )


# ============================================================
# EXIT DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("EXIT VERTEILUNG")
print("=" * 70)

if not trade_df.empty:

    print(
        trade_df[
            "exit_reason"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)

print(
    f"Trade-Log: "
    f"{output_file}"
)

print(
    "PAPER ONLY"
)

print(
    "NO API"
)

print(
    "NO LIVE TRADING"
)

print(
    "NO REAL ORDERS"
)

print("=" * 70)
