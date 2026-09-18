"""
Bitrue Strategy V3 - Detailed Diagnostic Analyzer

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO REAL ORDERS

Analysiert:
- Gesamt
- TRAIN vs OOS
- Take Profit / Stop Loss
- Long / Short
- Long / Short x Exit
- Haltedauer
- ATR-Regime
- Momentum
- EMA-Spread
- Volume Ratio
- Higher-Timeframe-Bestätigung
- Monat
- Verlustserie
- Break-even-Trefferquote
"""

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# DATEI
# ============================================================

if len(sys.argv) > 1:
    FILE = Path(sys.argv[1])
else:
    FILE = Path("bitrue_v3_trade_log.csv")


# ============================================================
# DATEI PRÜFEN
# ============================================================

if not FILE.exists():
    print("=" * 70)
    print("FEHLER: V3 Trade-Log nicht gefunden")
    print("=" * 70)
    print()
    print(f"Erwartete Datei: {FILE}")
    print()
    sys.exit(1)


# ============================================================
# CSV LADEN
# ============================================================

df = pd.read_csv(FILE)


# ============================================================
# ERFORDERLICHE SPALTEN
# ============================================================

required_columns = [
    "entry_time",
    "exit_time",
    "period",
    "side",
    "gross_pnl",
    "fees",
    "slippage_cost",
    "net_pnl",
    "duration_min",
    "momentum_pct",
    "atr_pct",
    "atr_value",
    "volume_ratio",
    "ema_spread_pct",
    "htf_bullish",
    "htf_bearish",
    "exit_reason",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:
    print("=" * 70)
    print("FEHLER: FEHLENDE SPALTEN")
    print("=" * 70)
    print()

    for column in missing_columns:
        print(f"- {column}")

    print()
    sys.exit(1)


# ============================================================
# ZEITEN
# ============================================================

df["entry_time"] = pd.to_datetime(
    df["entry_time"],
    utc=True,
    errors="coerce",
)

df["exit_time"] = pd.to_datetime(
    df["exit_time"],
    utc=True,
    errors="coerce",
)


# ============================================================
# ZUSÄTZLICHE SPALTEN
# ============================================================

df["month"] = (
    df["entry_time"]
    .dt.strftime("%Y-%m")
)

df["hour"] = (
    df["entry_time"]
    .dt.hour
)


# ============================================================
# SUMMARY
# ============================================================

def stats(group: pd.DataFrame) -> dict:

    trades = len(group)

    if trades == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross": 0.0,
            "fees": 0.0,
            "slippage": 0.0,
            "net": 0.0,
            "avg": 0.0,
        }

    wins = int(
        (group["net_pnl"] > 0).sum()
    )

    losses = int(
        (group["net_pnl"] <= 0).sum()
    )

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / trades * 100.0
        ),
        "gross": group["gross_pnl"].sum(),
        "fees": group["fees"].sum(),
        "slippage": group["slippage_cost"].sum(),
        "net": group["net_pnl"].sum(),
        "avg": group["net_pnl"].mean(),
    }


def print_stats(
    title: str,
    group: pd.DataFrame,
) -> None:

    result = stats(group)

    print(title)
    print("-" * 60)

    print(
        f"Trades:       {result['trades']}"
    )

    print(
        f"Gewinner:     {result['wins']}"
    )

    print(
        f"Verlierer:    {result['losses']}"
    )

    print(
        f"Win Rate:     {result['win_rate']:.2f}%"
    )

    print(
        f"Brutto P&L:   ${result['gross']:.4f}"
    )

    print(
        f"Gebühren:     ${result['fees']:.4f}"
    )

    print(
        f"Slippage:     ${result['slippage']:.4f}"
    )

    print(
        f"Netto P&L:    ${result['net']:.4f}"
    )

    print(
        f"Ø Trade:      ${result['avg']:.6f}"
    )

    print()


def section(title: str) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# HEADER
# ============================================================

section("BITRUE STRATEGY V3 - DETAILDIAGNOSE")

print(
    f"Trade-Log: {FILE}"
)

print(
    f"Trades:    {len(df)}"
)

if not df.empty:

    print(
        "Zeitraum:  "
        f"{df['entry_time'].min()} "
        "-> "
        f"{df['exit_time'].max()}"
    )

print()


# ============================================================
# 1. GESAMT
# ============================================================

section("1. V3 GESAMT")

print_stats(
    "GESAMT",
    df,
)


# ============================================================
# 2. TRAIN VS OOS
# ============================================================

section("2. TRAIN vs OOS")

for period in [
    "TRAIN",
    "OOS",
]:

    print_stats(
        period,
        df[df["period"] == period],
    )


# ============================================================
# 3. TAKE PROFIT vs STOP LOSS
# ============================================================

section("3. TAKE PROFIT vs STOP LOSS")

for reason, group in df.groupby(
    "exit_reason"
):

    print_stats(
        str(reason).upper(),
        group,
    )


# ============================================================
# 4. LONG vs SHORT
# ============================================================

section("4. LONG vs SHORT")

for side, group in df.groupby(
    "side"
):

    print_stats(
        str(side).upper(),
        group,
    )


# ============================================================
# 5. LONG / SHORT x EXIT
# ============================================================

section("5. LONG / SHORT x EXIT")

for side in sorted(
    df["side"].dropna().unique()
):

    print()
    print(str(side).upper())
    print("-" * 50)

    side_df = df[
        df["side"] == side
    ]

    for reason, group in side_df.groupby(
        "exit_reason"
    ):

        result = stats(group)

        print(
            f"{str(reason):16s} "
            f"Trades={result['trades']:3d} "
            f"WinRate={result['win_rate']:6.2f}% "
            f"Net=${result['net']:8.4f}"
        )


# ============================================================
# 6. HALTEDAUER
# ============================================================

section("6. HALTEDAUER")

duration_bins = [
    -1,
    15,
    30,
    60,
    120,
    240,
    480,
    float("inf"),
]

duration_labels = [
    "<=15m",
    "16-30m",
    "31-60m",
    "61-120m",
    "121-240m",
    "241-480m",
    ">480m",
]

df["duration_bucket"] = pd.cut(
    df["duration_min"],
    bins=duration_bins,
    labels=duration_labels,
)

rows = []

for bucket, group in df.groupby(
    "duration_bucket",
    observed=True,
):

    result = stats(group)

    rows.append(
        [
            str(bucket),
            result["trades"],
            result["win_rate"],
            result["net"],
            result["avg"],
        ]
    )


if rows:

    result_df = pd.DataFrame(
        rows,
        columns=[
            "Haltedauer",
            "Trades",
            "WinRate_%",
            "Netto_$",
            "ØTrade_$",
        ],
    )

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# 7. ATR-REGIME
# ============================================================

section("7. ATR-REGIME")

atr_categories = pd.Series(
    index=df.index,
    dtype="object",
)

atr_categories.loc[
    (df["atr_pct"] >= 0.075)
    & (df["atr_pct"] < 0.10)
] = "0.075-0.10%"

atr_categories.loc[
    (df["atr_pct"] >= 0.10)
    & (df["atr_pct"] < 0.125)
] = "0.10-0.125%"

atr_categories.loc[
    (df["atr_pct"] >= 0.125)
    & (df["atr_pct"] < 0.15)
] = "0.125-0.15%"

atr_categories.loc[
    (df["atr_pct"] >= 0.15)
    & (df["atr_pct"] < 0.175)
] = "0.15-0.175%"

atr_categories.loc[
    (df["atr_pct"] >= 0.175)
    & (df["atr_pct"] < 0.20)
] = "0.175-0.20%"

atr_categories.loc[
    df["atr_pct"] >= 0.20
] = ">=0.20%"

df["atr_bucket"] = atr_categories

rows = []

for bucket, group in df.groupby(
    "atr_bucket",
    observed=True,
):

    result = stats(group)

    rows.append(
        [
            str(bucket),
            result["trades"],
            result["win_rate"],
            result["net"],
            result["avg"],
        ]
    )


if rows:

    result_df = pd.DataFrame(
        rows,
        columns=[
            "ATR",
            "Trades",
            "WinRate_%",
            "Netto_$",
            "ØTrade_$",
        ],
    )

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# 8. MOMENTUM
# ============================================================

section("8. MOMENTUM")

momentum_bins = [
    -float("inf"),
    -0.50,
    -0.30,
    -0.20,
    0.0,
    0.20,
    0.30,
    0.50,
    float("inf"),
]

momentum_labels = [
    "<=-0.50%",
    "-0.50..-0.30%",
    "-0.30..-0.20%",
    "-0.20..0%",
    "0..+0.20%",
    "+0.20..+0.30%",
    "+0.30..+0.50%",
    ">+0.50%",
]

df["momentum_bucket"] = pd.cut(
    df["momentum_pct"],
    bins=momentum_bins,
    labels=momentum_labels,
)

rows = []

for bucket, group in df.groupby(
    "momentum_bucket",
    observed=True,
):

    result = stats(group)

    rows.append(
        [
            str(bucket),
            result["trades"],
            result["win_rate"],
            result["net"],
            result["avg"],
        ]
    )


if rows:

    result_df = pd.DataFrame(
        rows,
        columns=[
            "Momentum",
            "Trades",
            "WinRate_%",
            "Netto_$",
            "ØTrade_$",
        ],
    )

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# 9. EMA-SPREAD
# ============================================================

section("9. EMA-SPREAD")

spread_bins = [
    0.15,
    0.17,
    0.20,
    0.25,
    0.30,
    float("inf"),
]

spread_labels = [
    "0.15-0.17%",
    "0.17-0.20%",
    "0.20-0.25%",
    "0.25-0.30%",
    ">=0.30%",
]

df["spread_bucket"] = pd.cut(
    df["ema_spread_pct"],
    bins=spread_bins,
    labels=spread_labels,
    right=False,
)

rows = []

for bucket, group in df.groupby(
    "spread_bucket",
    observed=True,
):

    result = stats(group)

    rows.append(
        [
            str(bucket),
            result["trades"],
            result["win_rate"],
            result["net"],
            result["avg"],
        ]
    )


if rows:

    result_df = pd.DataFrame(
        rows,
        columns=[
            "EMA Spread",
            "Trades",
            "WinRate_%",
            "Netto_$",
            "ØTrade_$",
        ],
    )

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# 10. VOLUME RATIO
# ============================================================

section("10. VOLUME RATIO")

volume_bins = [
    1.20,
    1.50,
    2.00,
    3.00,
    5.00,
    float("inf"),
]

volume_labels = [
    "1.20-1.50",
    "1.50-2.00",
    "2.00-3.00",
    "3.00-5.00",
    ">=5.00",
]

df["volume_bucket"] = pd.cut(
    df["volume_ratio"],
    bins=volume_bins,
    labels=volume_labels,
    right=False,
)

rows = []

for bucket, group in df.groupby(
    "volume_bucket",
    observed=True,
):

    result = stats(group)

    rows.append(
        [
            str(bucket),
            result["trades"],
            result["win_rate"],
            result["net"],
            result["avg"],
        ]
    )


if rows:

    result_df = pd.DataFrame(
        rows,
        columns=[
            "Volume",
            "Trades",
            "WinRate_%",
            "Netto_$",
            "ØTrade_$",
        ],
    )

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# 11. HIGHER TIMEFRAME
# ============================================================

section("11. HIGHER-TIMEFRAME-BESTÄTIGUNG")

print(
    "V3 verwendet bereits eine 1H-EMA20/EMA50-Bestätigung."
)
print()

long_mask = (
    (df["side"] == "LONG")
    & (df["htf_bullish"] == True)
)

short_mask = (
    (df["side"] == "SHORT")
    & (df["htf_bearish"] == True)
)

print_stats(
    "LONG + HTF bullish",
    df[long_mask],
)

print_stats(
    "SHORT + HTF bearish",
    df[short_mask],
)


# ============================================================
# 12. MONATLICH
# ============================================================

section("12. MONAT")

rows = []

for month, group in df.groupby(
    "month"
):

    result = stats(group)

    rows.append(
        [
            month,
            result["trades"],
            result["win_rate"],
            result["net"],
            result["avg"],
        ]
    )


if rows:

    result_df = pd.DataFrame(
        rows,
        columns=[
            "Monat",
            "Trades",
            "WinRate_%",
            "Netto_$",
            "ØTrade_$",
        ],
    )

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# 13. VERLUSTSERIE
# ============================================================

section("13. VERLUSTSERIE")

current_streak = 0
maximum_streak = 0

for pnl in df["net_pnl"]:

    if pnl <= 0:

        current_streak += 1

        maximum_streak = max(
            maximum_streak,
            current_streak,
        )

    else:

        current_streak = 0


print(
    f"Längste nicht-positive Serie: "
    f"{maximum_streak}"
)


# ============================================================
# 14. BREAK-EVEN WIN RATE
# ============================================================

section("14. BREAK-EVEN-ANALYSE")

tp = df[
    df["exit_reason"]
    == "TAKE_PROFIT"
]

sl = df[
    df["exit_reason"]
    == "STOP_LOSS"
]

tp_avg = (
    tp["net_pnl"].mean()
    if not tp.empty
    else 0.0
)

sl_avg = (
    sl["net_pnl"].mean()
    if not sl.empty
    else 0.0
)

print(
    f"Ø TP netto: "
    f"${tp_avg:.6f}"
)

print(
    f"Ø SL netto: "
    f"${sl_avg:.6f}"
)

if (
    tp_avg > 0
    and sl_avg < 0
):

    breakeven_rate = (
        abs(sl_avg)
        / (
            tp_avg
            + abs(sl_avg)
        )
        * 100.0
    )

    print(
        f"Break-even Win Rate: "
        f"{breakeven_rate:.2f}%"
    )


# ============================================================
# 15. FINAL
# ============================================================

section("15. V3 SCHLUSSDIAGNOSE")

overall = stats(df)

oos = stats(
    df[df["period"] == "OOS"]
)

print(
    f"Gesamt Trades: "
    f"{overall['trades']}"
)

print(
    f"Gesamt Win Rate: "
    f"{overall['win_rate']:.2f}%"
)

print(
    f"Gesamt Netto: "
    f"${overall['net']:.4f}"
)

print()

print(
    f"OOS Trades: "
    f"{oos['trades']}"
)

print(
    f"OOS Win Rate: "
    f"{oos['win_rate']:.2f}%"
)

print(
    f"OOS Netto: "
    f"${oos['net']:.4f}"
)

print()

if overall["net"] < 0:
    print(
        "V3 ist im getesteten Zeitraum "
        "noch nicht profitabel."
    )

if oos["net"] < 0:
    print(
        "Auch OOS ist V3 noch negativ."
    )
else:
    print(
        "OOS ist positiv."
    )

print()
print(
    "Ziel: Den konkreten Verlusttreiber für "
    "die nächste Strategieversion identifizieren."
)

print()
print("=" * 70)
print("V3 DETAILDIAGNOSE ABGESCHLOSSEN")
print("=" * 70)
