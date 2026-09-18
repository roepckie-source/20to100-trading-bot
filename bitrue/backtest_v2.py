"""
Bitrue Strategy V2 - Backtest

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO REAL ORDERS
"""

import sys
from pathlib import Path

import pandas as pd

from strategy_v2 import calculate_indicators, evaluate_precomputed


# ============================================================
# CONFIG
# ============================================================

STARTING_CAPITAL = 100.00

TAKER_FEE = 0.0006
SLIPPAGE = 0.0002

MAX_DAILY_LOSS = 0.05

# V2 cooldown:
# Nach einem abgeschlossenen Trade werden für 30 Minuten
# keine neuen Einstiege erlaubt.
COOLDOWN_MINUTES = 30


# ============================================================
# INPUT FILE
# ============================================================

if len(sys.argv) < 2:
    print("Verwendung:")
    print("python backtest_v2.py BTCUSDT_5m.csv")
    sys.exit(1)

DATA_FILE = Path(sys.argv[1])

if not DATA_FILE.exists():
    print()
    print("FEHLER: Datendatei nicht gefunden:")
    print(DATA_FILE)
    sys.exit(1)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("BITRUE STRATEGY V2 BACKTEST")
print("=" * 70)
print()

print(f"Datenquelle: {DATA_FILE}")
print()

data = pd.read_csv(DATA_FILE)


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

data.columns = [
    str(column).strip().lower()
    for column in data.columns
]

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
    print("FEHLER: Fehlende Spalten:")
    print(missing)
    print()
    print("Vorhandene Spalten:")
    print(list(data.columns))
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
    print("FEHLER: Keine Zeitspalte gefunden.")
    print("Erwartet: timestamp / open_time / time / datetime / date")
    sys.exit(1)


# Binance CSV timestamps können Millisekunden sein.
raw_timestamp = data[timestamp_column]

if pd.api.types.is_numeric_dtype(raw_timestamp):

    median_timestamp = raw_timestamp.dropna().median()

    if median_timestamp > 10_000_000_000:

        data["timestamp"] = pd.to_datetime(
            raw_timestamp,
            unit="ms",
            utc=True,
            errors="coerce",
        )

    else:

        data["timestamp"] = pd.to_datetime(
            raw_timestamp,
            unit="s",
            utc=True,
            errors="coerce",
        )

else:

    data["timestamp"] = pd.to_datetime(
        raw_timestamp,
        utc=True,
        errors="coerce",
    )


# ============================================================
# CLEAN DATA
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

data = data.sort_values(
    "timestamp"
).drop_duplicates(
    subset=["timestamp"]
).reset_index(
    drop=True
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if data.empty:
    print("FEHLER: Keine Daten vorhanden.")
    sys.exit(1)

invalid_prices = (
    (data["open"] <= 0)
    | (data["high"] <= 0)
    | (data["low"] <= 0)
    | (data["close"] <= 0)
)

if invalid_prices.any():
    print(
        "FEHLER: Ungültige Preise gefunden:"
        f" {invalid_prices.sum()}"
    )
    sys.exit(1)


# ============================================================
# INDICATORS
# ============================================================

print("Berechne V2-Indikatoren...")

data = calculate_indicators(data)

print("Indikatoren fertig.")
print()

print(
    f"Kerzen:       {len(data):,}"
)

print(
    f"Start:        {data['timestamp'].iloc[0]}"
)

print(
    f"Ende:         {data['timestamp'].iloc[-1]}"
)

print()


# ============================================================
# STATE
# ============================================================

balance = STARTING_CAPITAL

position = None

trades = []

day_start_balance = STARTING_CAPITAL
current_day = None

last_exit_timestamp = None


# ============================================================
# HELPERS
# ============================================================

def reset_daily_balance(day):
    global current_day
    global day_start_balance

    if current_day != day:

        current_day = day
        day_start_balance = balance


def daily_loss_limit_reached():

    limit = day_start_balance * (
        1.0 - MAX_DAILY_LOSS
    )

    return balance <= limit


def execute_entry(
    side,
    market_price,
    position_size,
    timestamp,
):

    # ----------------------------------------
    # Slippage
    # ----------------------------------------

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

    # ----------------------------------------
    # Coin quantity
    # ----------------------------------------

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


def execute_exit(
    position_state,
    market_price,
    reason,
):

    side = position_state["side"]
    quantity = position_state["quantity"]
    entry_price = position_state["entry_price"]

    # ----------------------------------------
    # Exit slippage
    # ----------------------------------------

    if side == "LONG":

        exit_price = (
            market_price
            * (1.0 - SLIPPAGE)
        )

    else:

        exit_price = (
            market_price
            * (1.0 + SLIPPAGE)
        )

    # ----------------------------------------
    # Gross P&L
    # ----------------------------------------

    if side == "LONG":

        gross_pnl = (
            exit_price
            - entry_price
        ) * quantity

    else:

        gross_pnl = (
            entry_price
            - exit_price
        ) * quantity

    # ----------------------------------------
    # Fees
    # ----------------------------------------

    exit_notional = (
        abs(exit_price * quantity)
    )

    exit_fee = (
        exit_notional
        * TAKER_FEE
    )

    total_fees = (
        position_state["entry_fee"]
        + exit_fee
    )

    # ----------------------------------------
    # Slippage cost estimate
    # ----------------------------------------

    entry_slippage_cost = (
        abs(
            position_state["entry_market_price"]
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

    # ----------------------------------------
    # Net
    # ----------------------------------------

    net_pnl = (
        gross_pnl
        - total_fees
    )

    return {
        "exit_time": None,
        "exit_market_price": market_price,
        "exit_price": exit_price,
        "gross_pnl": gross_pnl,
        "fees": total_fees,
        "slippage_cost": slippage_cost,
        "net_pnl": net_pnl,
        "exit_reason": reason,
    }


# ============================================================
# BACKTEST LOOP
# ============================================================

print("Starte V2 Backtest...")
print()

for row in data.itertuples(index=False):

    timestamp = row.timestamp
    day = timestamp.date()

    reset_daily_balance(day)

    # ========================================================
    # EXISTING POSITION
    # ========================================================

    if position is not None:

        side = position["side"]

        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]

        exit_reason = None
        exit_market_price = None

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if side == "LONG":

            # Stop zuerst prüfen.
            # Das ist die konservative Behandlung,
            # wenn innerhalb derselben Kerze sowohl SL
            # als auch TP erreicht wurden.

            if row.low <= stop_loss:

                exit_reason = "STOP_LOSS"
                exit_market_price = stop_loss

            elif row.high >= take_profit:

                exit_reason = "TAKE_PROFIT"
                exit_market_price = take_profit

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            if row.high >= stop_loss:

                exit_reason = "STOP_LOSS"
                exit_market_price = stop_loss

            elif row.low <= take_profit:

                exit_reason = "TAKE_PROFIT"
                exit_market_price = take_profit

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if exit_reason is not None:

            result = execute_exit(
                position,
                float(exit_market_price),
                exit_reason,
            )

            result["exit_time"] = timestamp

            net_pnl = result["net_pnl"]

            balance += net_pnl

            duration_minutes = (
                timestamp
                - position["entry_time"]
            ).total_seconds() / 60.0

            trade = {
                "entry_time": position["entry_time"],
                "exit_time": timestamp,

                "side": side,

                "entry_price": position["entry_price"],
                "exit_price": result["exit_price"],

                "market_entry_price":
                    position["entry_market_price"],

                "market_exit_price":
                    result["exit_market_price"],

                "position_size":
                    position["position_size"],

                "quantity":
                    position["quantity"],

                "gross_pnl":
                    result["gross_pnl"],

                "fees":
                    result["fees"],

                "slippage_cost":
                    result["slippage_cost"],

                "net_pnl":
                    result["net_pnl"],

                "return_pct":
                    (
                        result["net_pnl"]
                        / position["position_size"]
                        * 100.0
                    ),

                "duration_min":
                    duration_minutes,

                "momentum_pct":
                    position["momentum_pct"],

                "atr_pct":
                    position["atr_pct"],

                "volume_ratio":
                    position["volume_ratio"],

                "ema_spread_pct":
                    position["ema_spread_pct"],

                "exit_reason":
                    result["exit_reason"],
            }

            trades.append(trade)

            position = None

            last_exit_timestamp = timestamp

            continue

    # ========================================================
    # NO POSITION -> CHECK ENTRY
    # ========================================================

    if position is not None:
        continue

    # --------------------------------------------------------
    # DAILY LOSS LIMIT
    # --------------------------------------------------------

    if daily_loss_limit_reached():
        continue

    # --------------------------------------------------------
    # COOLDOWN
    # --------------------------------------------------------

    if last_exit_timestamp is not None:

        minutes_since_exit = (
            timestamp
            - last_exit_timestamp
        ).total_seconds() / 60.0

        if minutes_since_exit < COOLDOWN_MINUTES:
            continue

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    signal = evaluate_precomputed(
        row,
        balance,
    )

    if not signal.signal:
        continue

    # --------------------------------------------------------
    # OPEN POSITION
    # --------------------------------------------------------

    position = execute_entry(
        signal.side,
        signal.price,
        signal.position_size,
        timestamp,
    )

    position["stop_loss"] = signal.stop_loss
    position["take_profit"] = signal.take_profit

    position["momentum_pct"] = signal.momentum_pct
    position["atr_pct"] = signal.atr_pct
    position["volume_ratio"] = signal.volume_ratio
    position["ema_spread_pct"] = signal.ema_spread_pct


# ============================================================
# END OF BACKTEST
# ============================================================

if position is not None:

    last_row = data.iloc[-1]

    timestamp = last_row["timestamp"]
    market_price = float(last_row["close"])

    result = execute_exit(
        position,
        market_price,
        "END_OF_BACKTEST",
    )

    result["exit_time"] = timestamp

    balance += result["net_pnl"]

    duration_minutes = (
        timestamp
        - position["entry_time"]
    ).total_seconds() / 60.0

    trade = {
        "entry_time": position["entry_time"],
        "exit_time": timestamp,

        "side": position["side"],

        "entry_price": position["entry_price"],
        "exit_price": result["exit_price"],

        "market_entry_price":
            position["entry_market_price"],

        "market_exit_price":
            result["exit_market_price"],

        "position_size":
            position["position_size"],

        "quantity":
            position["quantity"],

        "gross_pnl":
            result["gross_pnl"],

        "fees":
            result["fees"],

        "slippage_cost":
            result["slippage_cost"],

        "net_pnl":
            result["net_pnl"],

        "return_pct":
            (
                result["net_pnl"]
                / position["position_size"]
                * 100.0
            ),

        "duration_min":
            duration_minutes,

        "momentum_pct":
            position["momentum_pct"],

        "atr_pct":
            position["atr_pct"],

        "volume_ratio":
            position["volume_ratio"],

        "ema_spread_pct":
            position["ema_spread_pct"],

        "exit_reason":
            "END_OF_BACKTEST",
    }

    trades.append(trade)


# ============================================================
# RESULTS
# ============================================================

trade_df = pd.DataFrame(trades)

output_file = Path(
    "bitrue_v2_trade_log.csv"
)

trade_df.to_csv(
    output_file,
    index=False,
)


# ============================================================
# STATISTICS
# ============================================================

if trade_df.empty:

    print()
    print("=" * 70)
    print("KEINE TRADES")
    print("=" * 70)
    print()
    print(
        "V2 hat im getesteten Zeitraum "
        "kein Signal erzeugt."
    )
    sys.exit(0)


wins = (
    trade_df["net_pnl"] > 0
).sum()

losses = (
    trade_df["net_pnl"] <= 0
).sum()

trade_count = len(trade_df)

win_rate = (
    wins
    / trade_count
    * 100.0
)

gross_pnl = (
    trade_df["gross_pnl"].sum()
)

total_fees = (
    trade_df["fees"].sum()
)

total_slippage = (
    trade_df["slippage_cost"].sum()
)

net_pnl = (
    trade_df["net_pnl"].sum()
)

return_pct = (
    net_pnl
    / STARTING_CAPITAL
    * 100.0
)

avg_trade = (
    trade_df["net_pnl"].mean()
)


# ============================================================
# PROFIT FACTOR
# ============================================================

gross_wins = (
    trade_df.loc[
        trade_df["net_pnl"] > 0,
        "net_pnl",
    ].sum()
)

gross_losses = abs(
    trade_df.loc[
        trade_df["net_pnl"] < 0,
        "net_pnl",
    ].sum()
)

if gross_losses > 0:

    profit_factor = (
        gross_wins
        / gross_losses
    )

else:

    profit_factor = float("inf")


# ============================================================
# EQUITY / MAX DRAWDOWN
# ============================================================

equity = STARTING_CAPITAL

peak = equity
max_drawdown = 0.0

for pnl in trade_df["net_pnl"]:

    equity += pnl

    if equity > peak:
        peak = equity

    if peak > 0:

        drawdown = (
            peak - equity
        ) / peak

        max_drawdown = max(
            max_drawdown,
            drawdown,
        )

max_drawdown_pct = (
    max_drawdown
    * 100.0
)


# ============================================================
# LOSS STREAK
# ============================================================

current_streak = 0
max_loss_streak = 0

for pnl in trade_df["net_pnl"]:

    if pnl <= 0:

        current_streak += 1

        max_loss_streak = max(
            max_loss_streak,
            current_streak,
        )

    else:

        current_streak = 0


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)
print("BITRUE STRATEGY V2 - ERGEBNIS")
print("=" * 70)
print()

print(
    f"Startkapital:       ${STARTING_CAPITAL:.2f}"
)

print(
    f"Endkapital:         ${balance:.2f}"
)

print(
    f"Netto P&L:          ${net_pnl:.4f}"
)

print(
    f"Rendite:            {return_pct:.2f}%"
)

print()

print(
    f"Trades:             {trade_count}"
)

print(
    f"Gewinner:           {wins}"
)

print(
    f"Verlierer:          {losses}"
)

print(
    f"Win Rate:           {win_rate:.2f}%"
)

print(
    f"Profit Factor:      {profit_factor:.4f}"
)

print()

print(
    f"Brutto P&L:         ${gross_pnl:.4f}"
)

print(
    f"Gebühren:           ${total_fees:.4f}"
)

print(
    f"Slippage:           ${total_slippage:.4f}"
)

print(
    f"Ø Trade:            ${avg_trade:.6f}"
)

print()

print(
    f"Max Drawdown:       {max_drawdown_pct:.2f}%"
)

print(
    f"Längste Verlustserie: {max_loss_streak}"
)

print()

print(
    f"Trade-Log:          {output_file}"
)

print()
print("=" * 70)
print("V2 BACKTEST ABGESCHLOSSEN")
print("=" * 70)
print()
print("PAPER ONLY")
print("NO API")
print("NO LIVE TRADING")
print("NO REAL ORDERS")
print()
