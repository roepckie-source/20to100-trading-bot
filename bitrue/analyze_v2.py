"""
Bitrue Strategy V2 - Diagnostic Analyzer

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO REAL ORDERS
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
    FILE = Path("bitrue_v2_trade_log.csv")


# ============================================================
# CSV LADEN
# ============================================================

if not FILE.exists():
    print("=" * 70)
    print("FEHLER: V2 Trade-Log nicht gefunden")
    print("=" * 70)
    print(f"Datei: {FILE}")
    sys.exit(1)


df = pd.read_csv(FILE)


# ============================================================
# VALIDIERUNG
# ============================================================

required = [
    "entry_time",
    "exit_time",
    "side",
    "gross_pnl",
    "fees",
    "slippage_cost",
    "net_pnl",
    "duration_min",
    "momentum_pct",
    "atr_pct",
    "volume_ratio",
    "ema_spread_pct",
    "exit_reason",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    print("FEHLER: Fehlende Spalten:")
    print(missing)
    sys.exit(1)


# ============================================================
# ZEITSPALTEN
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
# BERECHNUNGEN
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

def summary(group: pd.DataFrame) -> dict:

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
        "gross": (
            group["gross_pnl"].sum()
        ),
        "fees": (
            group["fees"].sum()
        ),
        "slippage": (
            group["slippage_cost"].sum()
        ),
        "net": (
            group["net_pnl"].sum()
        ),
        "avg": (
            group["net_pnl"].mean()
        ),
    }


def print_summary(
    title: str,
    group: pd.DataFrame,
) -> None:

    result = summary(group)

    print(title)
    print("-" * 60)

    print(
        f"Trades:              "
        f"{result['trades']}"
    )

    print(
        f"Gewinner:            "
        f"{result['wins']}"
    )

    print(
        f"Verlierer:           "
        f"{result['losses']}"
    )

    print(
        f"Win Rate:            "
        f"{result['win_rate']:.2f}%"
    )

    print(
        f"Brutto P&L:          "
        f"${result['gross']:.4f}"
    )

    print(
        f"Gebühren:            "
        f"${result['fees']:.4f}"
    )

    print(
        f"Slippage:            "
        f"${result['slippage']:.4f}"
    )

    print(
        f"Netto P&L:           "
        f"${result['net']:.4f}"
    )

    print(
        f"Ø Trade:             "
        f"${result['avg']:.6f}"
    )

    print()


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("BITRUE STRATEGY V2 - DIAGNOSE")
print("=" * 70)
print()

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

print("=" * 70)
print("1. GESAMT")
print("=" * 70)

print_summary(
    "Gesamt",
    df,
)


# ============================================================
# 2. LONG / SHORT
# ============================================================

print("=" * 70)
print("2. LONG vs SHORT")
print("=" * 70)

for side, group in df.groupby("side"):

    print()

    print_summary(
        str(side).upper(),
        group,
    )


# ============================================================
# 3. TAKE PROFIT / STOP LOSS
# ============================================================

print("=" * 70)
print("3. TAKE PROFIT vs STOP LOSS")
print("=" * 70)

for reason, group in df.groupby(
    "exit_reason"
):

    print()

    print_summary(
        str(reason).upper(),
        group,
    )


# ============================================================
# 4. MONATLICHE ENTWICKLUNG
# ============================================================

print("=" * 70)
print("4. MONATLICHE ENTWICKLUNG")
print("=" * 70)

monthly_rows = []

for month, group in df.groupby(
    "month"
):

    result = summary(group)

    monthly_rows.append(
        [
            month,
            result["trades"],
            result["win_rate"],
            result["gross"],
            result["net"],
            result["avg"],
        ]
    )


if monthly_rows:

    monthly_df = pd.DataFrame(
        monthly_rows,
        columns=[
            "Monat",
            "Trades",
            "WinRate_%",
            "Brutto_$",
            "Netto_$",
            "ØTrade_$",
        ],
    )

    print(
        monthly_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

print()


# ============================================================
# 5. HALTEDAUER
# ============================================================

print("=" * 70)
print("5. PERFORMANCE NACH HALTEDAUER")
print("=" * 70)

duration_bins = [
    -1,
    15,
    30,
    60,
    120,
    240,
    float("inf"),
]

duration_labels = [
    "<=15m",
    "16-30m",
    "31-60m",
    "61-120m",
    "121-240m",
    ">240m",
]

df["duration_bucket"] = pd.cut(
    df["duration_min"],
    bins=duration_bins,
    labels=duration_labels,
)

duration_rows = []

for bucket, group in df.groupby(
    "duration_bucket",
    observed=True,
):

    result = summary(group)

    duration_rows.append(
        [
            str(bucket),
            result["trades"],
            result["win_rate"],
            result["net"],
            result["avg"],
        ]
    )


if duration_rows:

    duration_df = pd.DataFrame(
        duration_rows,
        columns=[
            "Haltedauer",
            "Trades",
            "WinRate_%",
            "Netto_$",
            "ØTrade_$",
        ],
    )

    print(
        duration_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

print()


# ============================================================
# GENERISCHE BUCKET-AUSWERTUNG
# ============================================================

def bucket_report(
    title,
    series,
    bins,
    labels,
):

    print("=" * 70)
    print(title)
    print("=" * 70)

    column_name = (
        title
        .lower()
        .replace(" ", "_")
    )

    temp = df.copy()

    temp[column_name] = pd.cut(
        series,
        bins=bins,
        labels=labels,
    )

    rows = []

    for bucket, group in temp.groupby(
        column_name,
        observed=True,
    ):

        result = summary(group)

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
                "Bereich",
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

    print()


# ============================================================
# 6. MOMENTUM
# ============================================================

bucket_report(
    "6. MOMENTUM",
    df["momentum_pct"],
    [
        -float("inf"),
        -0.50,
        -0.30,
        -0.20,
        -0.15,
        0.15,
        0.20,
        0.30,
        0.50,
        float("inf"),
    ],
    [
        "<=-0.50%",
        "-0.50..-0.30",
        "-0.30..-0.20",
        "-0.20..-0.15",
        "-0.15..+0.15",
        "+0.15..+0.20",
        "+0.20..+0.30",
        "+0.30..+0.50",
        ">+0.50%",
    ],
)


# ============================================================
# 7. ATR
# ============================================================

bucket_report(
    "7. ATR",
    df["atr_pct"],
    [
        -float("inf"),
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        float("inf"),
    ],
    [
        "<0.05",
        "0.05-0.075",
        "0.075-0.10",
        "0.10-0.15",
        "0.15-0.20",
        "0.20-0.30",
        ">0.30",
    ],
)


# ============================================================
# 8. VOLUME RATIO
# ============================================================

bucket_report(
    "8. VOLUME RATIO",
    df["volume_ratio"],
    [
        -float("inf"),
        1.20,
        1.50,
        2.00,
        3.00,
        5.00,
        10.00,
        float("inf"),
    ],
    [
        "<1.20",
        "1.20-1.50",
        "1.50-2.00",
        "2.00-3.00",
        "3.00-5.00",
        "5.00-10.00",
        ">10.00",
    ],
)


# ============================================================
# 9. EMA SPREAD
# ============================================================

bucket_report(
    "9. EMA SPREAD",
    df["ema_spread_pct"],
    [
        -float("inf"),
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.25,
        float("inf"),
    ],
    [
        "<0.03",
        "0.03-0.05",
        "0.05-0.075",
        "0.075-0.10",
        "0.10-0.15",
        "0.15-0.25",
        ">0.25",
    ],
)


# ============================================================
# 10. EINSTIEGSSTUNDE UTC
# ============================================================

print("=" * 70)
print("10. PERFORMANCE NACH EINSTIEGSSTUNDE UTC")
print("=" * 70)

hour_rows = []

for hour, group in df.groupby(
    "hour"
):

    result = summary(group)

    hour_rows.append(
        [
            int(hour),
            result["trades"],
            result["win_rate"],
            result["net"],
            result["avg"],
        ]
    )


if hour_rows:

    hour_df = pd.DataFrame(
        hour_rows,
        columns=[
            "UTC-Stunde",
            "Trades",
            "WinRate_%",
            "Netto_$",
            "ØTrade_$",
        ],
    )

    print(
        hour_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

print()


# ============================================================
# 11. LONG / SHORT NACH MOMENTUM
# ============================================================

print("=" * 70)
print("11. LONG / SHORT NACH MOMENTUM")
print("=" * 70)

momentum_bins = [
    -float("inf"),
    -0.50,
    -0.30,
    -0.20,
    -0.15,
    0.15,
    0.20,
    0.30,
    0.50,
    float("inf"),
]

momentum_labels = [
    "<=-0.50",
    "-0.50..-0.30",
    "-0.30..-0.20",
    "-0.20..-0.15",
    "-0.15..+0.15",
    "+0.15..+0.20",
    "+0.20..+0.30",
    "+0.30..+0.50",
    ">+0.50",
]

df["momentum_bucket"] = pd.cut(
    df["momentum_pct"],
    bins=momentum_bins,
    labels=momentum_labels,
)

for side, group_side in df.groupby(
    "side"
):

    print()
    print(str(side).upper())

    for bucket, group in group_side.groupby(
        "momentum_bucket",
        observed=True,
    ):

        result = summary(group)

        if result["trades"] == 0:
            continue

        print(
            f"{str(bucket):18s} "
            f"Trades={result['trades']:4d} "
            f"WinRate={result['win_rate']:6.2f}% "
            f"Net=${result['net']:8.4f}"
        )

print()


# ============================================================
# 12. LONG / SHORT NACH EMA SPREAD
# ============================================================

print("=" * 70)
print("12. LONG / SHORT NACH EMA SPREAD")
print("=" * 70)

spread_bins = [
    -float("inf"),
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.25,
    float("inf"),
]

spread_labels = [
    "<0.03",
    "0.03-0.05",
    "0.05-0.075",
    "0.075-0.10",
    "0.10-0.15",
    "0.15-0.25",
    ">0.25",
]

df["spread_bucket"] = pd.cut(
    df["ema_spread_pct"],
    bins=spread_bins,
    labels=spread_labels,
)

for side, group_side in df.groupby(
    "side"
):

    print()
    print(str(side).upper())

    for bucket, group in group_side.groupby(
        "spread_bucket",
        observed=True,
    ):

        result = summary(group)

        if result["trades"] == 0:
            continue

        print(
            f"{str(bucket):18s} "
            f"Trades={result['trades']:4d} "
            f"WinRate={result['win_rate']:6.2f}% "
            f"Net=${result['net']:8.4f}"
        )

print()


# ============================================================
# 13. VERLUSTSERIEN
# ============================================================

print("=" * 70)
print("13. VERLUSTSERIEN")
print("=" * 70)

current_streak = 0
max_streak = 0

for pnl in df["net_pnl"]:

    if pnl <= 0:

        current_streak += 1

        max_streak = max(
            max_streak,
            current_streak,
        )

    else:

        current_streak = 0


print(
    "Längste nicht-positive Serie: "
    f"{max_streak}"
)

print()


# ============================================================
# 14. HANDELSFREQUENZ
# ============================================================

print("=" * 70)
print("14. HANDELSFREQUENZ")
print("=" * 70)

days = (
    df["entry_time"]
    .dt.date
    .nunique()
)

print(
    f"Handelstage: {days}"
)

print(
    f"Trades:       {len(df)}"
)

if days > 0:

    print(
        f"Trades/Tag:   "
        f"{len(df) / days:.2f}"
    )

else:

    print(
        "Trades/Tag:   0.00"
    )

print()


# ============================================================
# 15. AUTOMATISCHE EINORDNUNG
# ============================================================

print("=" * 70)
print("15. AUTOMATISCHE EINORDNUNG")
print("=" * 70)

total = summary(df)

if total["net"] < 0:

    print(
        "V2 ist im getesteten Zeitraum "
        "netto negativ."
    )

if total["gross"] < 0:

    print(
        "V2 ist bereits vor Gebühren negativ."
    )

print(
    f"Win Rate: "
    f"{total['win_rate']:.2f}%"
)

print(
    f"Netto P&L: "
    f"${total['net']:.4f}"
)

print()

print(
    "V2 wurde NICHT verändert."
)

print(
    "Analyse basiert ausschließlich "
    "auf dem vorhandenen V2-Trade-Log."
)

print()

print("=" * 70)
print("DIAGNOSE ABGESCHLOSSEN")
print("=" * 70)
