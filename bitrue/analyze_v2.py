"""
Bitrue Strategy V2 - Diagnostic Analyzer

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO REAL ORDERS

Analysiert:
- Long / Short
- Take Profit / Stop Loss
- Monate
- Einstiegssunden
- Haltedauer
- Momentum
- ATR
- Volume Ratio
- EMA Spread
- Verlustserien
- Gebühren
- Slippage
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
    print("FEHLER")
    print("=" * 70)
    print()
    print(f"Trade-Log nicht gefunden:")
    print(f"  {FILE}")
    print()
    sys.exit(1)

df = pd.read_csv(FILE)


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

df["duration_min"] = (
    df["exit_time"] - df["entry_time"]
).dt.total_seconds() / 60.0

df["month"] = df["entry_time"].dt.strftime("%Y-%m")

df["hour"] = df["entry_time"].dt.hour


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
            "win_rate_pct": 0.0,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "slippage": 0.0,
            "net_pnl": 0.0,
            "avg_net_trade": 0.0,
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
        "win_rate_pct":
            wins / trades * 100.0,
        "gross_pnl":
            group["gross_pnl"].sum(),
        "fees":
            group["fees"].sum(),
        "slippage":
            group["slippage_cost"].sum(),
        "net_pnl":
            group["net_pnl"].sum(),
        "avg_net_trade":
            group["net_pnl"].mean(),
    }


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("BITRUE STRATEGY V2 - DIAGNOSE")
print("=" * 70)
print()

print(f"Trade-Log: {FILE}")
print(f"Trades:    {len(df)}")

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

total = summary(df)

print(
    f"Trades:               {total['trades']}"
)

print(
    f"Gewinner:             {total['wins']}"
)

print(
    f"Verlierer:            {total['losses']}"
)

print(
    f"Win Rate:             {total['win_rate_pct']:.2f}%"
)

print(
    f"Brutto P&L:           ${total['gross_pnl']:.4f}"
)

print(
    f"Gebühren:             ${total['fees']:.4f}"
)

print(
    f"Slippage:             ${total['slippage']:.4f}"
)

print(
    f"Netto P&L:            ${total['net_pnl']:.4f}"
)

print(
    f"Ø Trade:              ${total['avg_net_trade']:.6f}"
)

print()


# ============================================================
# 2. LONG / SHORT
# ============================================================

print("=" * 70)
print("2. LONG vs SHORT")
print("=" * 70)

for side, group in df.groupby("side"):

    result = summary(group)

    print()
    print(str(side).upper())
    print("-" * 50)

    print(
        f"Trades:               {result['trades']}"
    )

    print(
        f"Gewinner:             {result['wins']}"
    )

    print(
        f"Verlierer:            {result['losses']}"
    )

    print(
        f"Win Rate:             {result['win_rate_pct']:.2f}%"
    )

    print(
        f"Brutto P&L:           ${result['gross_pnl']:.4f}"
    )

    print(
        f"Netto P&L:            ${result['net_pnl']:.4f}"
    )

    print(
        f"Ø Trade:              ${result['avg_net_trade']:.6f}"
    )

print()


# ============================================================
# 3. EXIT REASON
# ============================================================

print("=" * 70)
print("3. TAKE PROFIT vs STOP LOSS")
print("=" * 70)

for reason, group in df.groupby("exit_reason"):

    result = summary(group)

    print()
    print(str(reason).upper())
    print("-" * 50)

    print(
        f"Trades:               {result['trades']}"
    )

    print(
        f"Gewinner:             {result['wins']}"
    )

    print(
        f"Verlierer:            {result['losses']}"
    )

    print(
        f"Win Rate:             {result['win_rate_pct']:.2f}%"
    )

    print(
        f"Brutto P&L:           ${result['gross_pnl']:.4f}"
    )

    print(
        f"Gebühren:             ${result['fees']:.4f}"
    )

    print(
        f"Slippage:             ${result['slippage']:.4f}"
    )

    print(
        f"Netto P&L:            ${result['net_pnl']:.4f}"
    )

print()


# ============================================================
# 4. MONAT
# ============================================================

print("=" * 70)
print("4. MONATLICHE ENTWICKLUNG")
print("=" * 70)

monthly_rows = []

for month, group in df.groupby("month"):

    result = summary(group)

    monthly_rows.append({
        "month": month,
        **result,
    })

monthly_df = pd.DataFrame(
    monthly_rows
)

if not monthly_df.empty:

    print(
        monthly_df[
            [
                "month",
                "trades",
                "win_rate_pct",
                "gross_pnl",
                "fees",
                "slippage",
                "net_pnl",
                "avg_net_trade",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print()


# ============================================================
# 5. EINSTIEGSSTUNDE
# ============================================================

print("=" * 70)
print("5. PERFORMANCE NACH EINSTIEGSSTUNDE UTC")
print("=" * 70)

hour_rows = []

for hour, group in df.groupby("hour"):

    result = summary(group)

    hour_rows.append({
        "hour_utc": int(hour),
        **result,
    })

hour_df = pd.DataFrame(
    hour_rows
)

if not hour_df.empty:

    print(
        hour_df[
            [
                "hour_utc",
                "trades",
                "win_rate_pct",
                "gross_pnl",
                "net_pnl",
                "avg_net_trade",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print()


# ============================================================
# 6. HALTEDAUER
# ============================================================

print("=" * 70)
print("6. PERFORMANCE NACH HALTEDAUER")
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

    duration_rows.append({
        "duration": str(bucket),
        **result,
    })

duration_df = pd.DataFrame(
    duration_rows
)

if not duration_df.empty:

    print(
        duration_df[
            [
                "duration",
                "trades",
                "win_rate_pct",
                "gross_pnl",
                "fees",
                "slippage",
                "net_pnl",
                "avg_net_trade",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print()


# ============================================================
# 7. MOMENTUM
# ============================================================

print("=" * 70)
print("7. PERFORMANCE NACH MOMENTUM")
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
    "<=-0.50%",
    "-0.50 bis -0.30%",
    "-0.30 bis -0.20%",
    "-0.20 bis -0.15%",
    "-0.15 bis +0.15%",
    "+0.15 bis +0.20%",
    "+0.20 bis +0.30%",
    "+0.30 bis +0.50%",
    ">+0.50%",
]

df["momentum_bucket"] = pd.cut(
    df["momentum_pct"],
    bins=momentum_bins,
    labels=momentum_labels,
)

momentum_rows = []

for bucket, group in df.groupby(
    "momentum_bucket",
    observed=True,
):

    result = summary(group)

    momentum_rows.append({
        "momentum": str(bucket),
        **result,
    })

momentum_df = pd.DataFrame(
    momentum_rows
)

if not momentum_df.empty:

    print(
        momentum_df[
            [
                "momentum",
                "trades",
                "win_rate_pct",
                "net_pnl",
                "avg_net_trade",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print()


# ============================================================
# 8. ATR
# ============================================================

print("=" * 70)
print("8. PERFORMANCE NACH ATR")
print("=" * 70)

atr_bins = [
    -float("inf"),
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    float("inf"),
]

atr_labels = [
    "<0.05%",
    "0.05-0.075%",
    "0.075-0.10%",
    "0.10-0.15%",
    "0.15-0.20%",
    "0.20-0.30%",
    ">0.30%",
]

df["atr_bucket"] = pd.cut(
    df["atr_pct"],
    bins=atr_bins,
    labels=atr_labels,
)

atr_rows = []

for bucket, group in df.groupby(
    "atr_bucket",
    observed=True,
):

    result = summary(group)

    atr_rows.append({
        "atr": str(bucket),
        **result,
    })

atr_df = pd.DataFrame(
    atr_rows
)

if not atr_df.empty:

    print(
        atr_df[
            [
                "atr",
                "trades",
                "win_rate_pct",
                "net_pnl",
                "avg_net_trade",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print()


# ============================================================
# 9. VOLUME RATIO
# ============================================================

print("=" * 70)
print("9. PERFORMANCE NACH VOLUME RATIO")
print("=" * 70)

volume_bins = [
    -float("inf"),
    1.20,
    1.50,
    2.00,
    3.00,
    5.00,
    10.00,
    float("inf"),
]

volume_labels = [
    "<1.20",
    "1.20-1.50",
    "1.50-2.00",
    "2.00-3.00",
    "3.00-5.00",
    "5.00-10.00",
    ">10.00",
]

df["volume_bucket"] = pd.cut(
    df["volume_ratio"],
    bins=volume_bins,
    labels=volume_labels,
)

volume_rows = []

for bucket, group in df.groupby(
    "volume_bucket",
    observed=True,
):

    result = summary(group)

    volume_rows.append({
        "volume_ratio": str(bucket),
        **result,
    })

volume_df = pd.DataFrame(
    volume_rows
)

if not volume_df.empty:

    print(
        volume_df[
            [
                "volume_ratio",
                "trades",
                "win_rate_pct",
                "net_pnl",
                "avg_net_trade",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print()


# ============================================================
# 10. EMA SPREAD
# ============================================================

print("=" * 70)
print("10. PERFORMANCE NACH EMA-SPREAD")
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
    "<0.03%",
    "0.03-0.05%",
    "0.05-0.075%",
    "0.075-0.10%",
    "0.10-0.15%",
    "0.15-0.25%",
    ">0.25%",
]

df["spread_bucket"] = pd.cut(
    df["ema_spread_pct"],
    bins=spread_bins,
    labels=spread_labels,
)

spread_rows = []

for bucket, group in df.groupby(
    "spread_bucket",
    observed=True,
):

    result = summary(group)

    spread_rows.append({
        "ema_spread": str(bucket),
        **result,
    })

spread_df = pd.DataFrame(
    spread_rows
)

if not spread_df.empty:

    print(
        spread_df[
            [
                "ema_spread",
                "trades",
                "win_rate_pct",
                "net_pnl",
                "avg_net_trade",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

print()


# ============================================================
# 11. LONG + SHORT x MOMENTUM
# ============================================================

print("=" * 70)
print("11. LONG / SHORT NACH MOMENTUM")
print("=" * 70)

for side, side_group in df.groupby("side"):

    print()
    print(f"{str(side).upper()}")
    print("-" * 50)

    for bucket, group in side_group.groupby(
        "momentum_bucket",
        observed=True,
    ):

        result = summary(group)

        if result["trades"] == 0:
            continue

        print(
            f"{str(bucket):25s} "
            f"Trades={result['trades']:4d} "
            f"WinRate={result['win_rate_pct']:6.2f}% "
            f"Net=${result['net_pnl']:8.4f}"
        )

print()


# ============================================================
# 12. LONG + SHORT x EMA SPREAD
# ============================================================

print("=" * 70)
print("12. LONG / SHORT NACH EMA-SPREAD")
print("=" * 70)

for side, side_group in df.groupby("side"):

    print()
    print(f"{str(side).upper()}")
    print("-" * 50)

    for bucket, group in side_group.groupby(
        "spread_bucket",
        observed=True,
    ):

        result = summary(group)

        if result["trades"] == 0:
            continue

        print(
            f"{str(bucket):25s} "
            f"Trades={result['trades']:4d} "
            f"WinRate={result['win_rate_pct']:6.2f}% "
            f"Net=${result['net_pnl']:8.4f}"
        )

print()


# ============================================================
# 13. VERLUSTSERIE
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
    f"Längste Serie nicht-positiver Trades: "
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
    df["entry_time"].dt.date.nunique()
    if not df.empty
    else 0
)

if days > 0:
    trades_per_day = (
        len(df) / days
    )
else:
    trades_per_day = 0.0

print(
    f"Handelstage:     {days}"
)

print(
    f"Trades gesamt:   {len(df)}"
)

print(
    f"Trades / Tag:    {trades_per_day:.2f}"
)

print()


# ============================================================
# 15. AUTOMATISCHE DIAGNOSE
# ============================================================

print("=" * 70)
print("15. AUTOMATISCHE DIAGNOSE")
print("=" * 70)

print()

if total["net_pnl"] < 0:

    print(
        "❌ V2 ist im getesteten Zeitraum "
        "netto negativ."
    )

else:

    print(
        "✅ V2 ist im getesteten Zeitraum "
        "netto positiv."
    )


if total["gross_pnl"] < 0:

    print(
        "❌ V2 ist bereits vor Gebühren negativ."
    )

else:

    print(
        "⚠️ Brutto ist positiv; Kosten werden "
        "separat bewertet."
    )


print(
    f"Trefferquote: "
    f"{total['win_rate_pct']:.2f}%"
)

if total["win_rate_pct"] < 40:

    print(
        "⚠️ Trefferquote weiterhin relativ niedrig."
    )


if trades_per_day > 10:

    print(
        f"⚠️ Handelsfrequenz weiterhin hoch: "
        f"{trades_per_day:.2f} Trades/Tag."
    )


print()
print("=" * 70)
print("DIAGNOSE ABGESCHLOSSEN")
print("=" * 70)
print()

print("V2 wurde NICHT verändert.")
print("Dieser Analyzer liest ausschließlich den Trade-Log.")
print()
