"""
Bitrue Strategy V1 - Diagnostic Analyzer

PAPER / BACKTEST ONLY
KEIN LIVE TRADING
KEINE API KEYS

Analysiert den bestehenden V1-Trade-Log und zeigt,
wo und warum die Strategie Geld verliert.
"""

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# DATEI
# ============================================================

# Standard:
# analyze_v1.py liegt in /bitrue/
# der Trade-Log liegt im Repository-Hauptverzeichnis.
#
# Optional kann beim Aufruf auch ein anderer Dateipfad
# übergeben werden:
#
# python bitrue/analyze_v1.py bitrue_v1_trade_log.csv

if len(sys.argv) > 1:
    FILE = Path(sys.argv[1])
else:
    FILE = Path("../bitrue_v1_trade_log.csv")


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
    print("Erwartete Datei:")
    print("  bitrue_v1_trade_log.csv")
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
# SUMMARY FUNKTION
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

    wins = int((group["net_pnl"] > 0).sum())
    losses = int((group["net_pnl"] <= 0).sum())

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": 100.0 * wins / trades,
        "gross_pnl": group["gross_pnl"].sum(),
        "fees": group["fees"].sum(),
        "slippage": group["slippage_cost"].sum(),
        "net_pnl": group["net_pnl"].sum(),
        "avg_net_trade": group["net_pnl"].mean(),
    }


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("BITRUE STRATEGY V1 - DIAGNOSE")
print("=" * 70)
print()

print(f"Trade-Log: {FILE}")
print(f"Trades:    {len(df)}")

if not df.empty:
    print(
        f"Zeitraum:  "
        f"{df['entry_time'].min()} -> {df['exit_time'].max()}"
    )

print()


# ============================================================
# GESAMT
# ============================================================

print("=" * 70)
print("1. GESAMT")
print("=" * 70)

total = pd.Series(summary(df))

for key, value in total.items():
    if isinstance(value, float):
        if "pnl" in key or key in ("fees", "slippage"):
            print(f"{key:20s}: ${value:,.4f}")
        else:
            print(f"{key:20s}: {value:,.2f}")
    else:
        print(f"{key:20s}: {value}")

print()


# ============================================================
# LONG / SHORT
# ============================================================

print("=" * 70)
print("2. LONG vs SHORT")
print("=" * 70)

for side, group in df.groupby("side"):
    result = summary(group)

    print()
    print(f"{str(side).upper()}")
    print("-" * 50)

    print(f"Trades:               {result['trades']}")
    print(f"Gewinner:             {result['wins']}")
    print(f"Verlierer:            {result['losses']}")
    print(f"Win Rate:             {result['win_rate_pct']:.2f}%")
    print(f"Brutto P&L:           ${result['gross_pnl']:.4f}")
    print(f"Gebühren:             ${result['fees']:.4f}")
    print(f"Slippage:             ${result['slippage']:.4f}")
    print(f"Netto P&L:            ${result['net_pnl']:.4f}")
    print(f"Ø Trade:              ${result['avg_net_trade']:.6f}")

print()


# ============================================================
# EXIT REASON
# ============================================================

print("=" * 70)
print("3. TAKE PROFIT vs STOP LOSS")
print("=" * 70)

for reason, group in df.groupby("exit_reason"):
    result = summary(group)

    print()
    print(f"{str(reason).upper()}")
    print("-" * 50)

    print(f"Trades:               {result['trades']}")
    print(f"Gewinner:             {result['wins']}")
    print(f"Verlierer:            {result['losses']}")
    print(f"Win Rate:             {result['win_rate_pct']:.2f}%")
    print(f"Brutto P&L:           ${result['gross_pnl']:.4f}")
    print(f"Gebühren:             ${result['fees']:.4f}")
    print(f"Slippage:             ${result['slippage']:.4f}")
    print(f"Netto P&L:            ${result['net_pnl']:.4f}")
    print(f"Ø Trade:              ${result['avg_net_trade']:.6f}")

print()


# ============================================================
# MONATLICH
# ============================================================

print("=" * 70)
print("4. MONATLICHE ENTWICKLUNG")
print("=" * 70)

monthly = []

for month, group in df.groupby("month"):
    result = summary(group)

    monthly.append(
        {
            "month": month,
            **result,
        }
    )

monthly_df = pd.DataFrame(monthly)

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
# EINSTIEGSSTUNDE UTC
# ============================================================

print("=" * 70)
print("5. PERFORMANCE NACH EINSTIEGSSTUNDE UTC")
print("=" * 70)

hourly = []

for hour, group in df.groupby("hour"):
    result = summary(group)

    hourly.append(
        {
            "hour_utc": int(hour),
            **result,
        }
    )

hourly_df = pd.DataFrame(hourly)

if not hourly_df.empty:
    print(
        hourly_df[
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
# HALTEDAUER
# ============================================================

print("=" * 70)
print("6. PERFORMANCE NACH HALTEDAUER")
print("=" * 70)

bins = [
    -1,
    15,
    30,
    60,
    120,
    240,
    float("inf"),
]

labels = [
    "<=15m",
    "16-30m",
    "31-60m",
    "61-120m",
    "121-240m",
    ">240m",
]

df["duration_bucket"] = pd.cut(
    df["duration_min"],
    bins=bins,
    labels=labels,
)

duration_rows = []

for bucket, group in df.groupby(
    "duration_bucket",
    observed=True,
):
    result = summary(group)

    duration_rows.append(
        {
            "duration": str(bucket),
            **result,
        }
    )

duration_df = pd.DataFrame(duration_rows)

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
# LÄNGSTE VERLUSTSERIE
# ============================================================

print("=" * 70)
print("7. VERLUSTSERIEN")
print("=" * 70)

loss_mask = df["net_pnl"] <= 0

current_streak = 0
max_streak = 0

for is_loss in loss_mask:
    if is_loss:
        current_streak += 1
        max_streak = max(max_streak, current_streak)
    else:
        current_streak = 0

print(
    f"Längste Serie nicht-positiver Trades: "
    f"{max_streak}"
)

print()


# ============================================================
# BRUTTO VS KOSTEN VS NETTO
# ============================================================

print("=" * 70)
print("8. BRUTTO / KOSTEN / NETTO")
print("=" * 70)

gross_pnl = df["gross_pnl"].sum()
fees = df["fees"].sum()
slippage = df["slippage_cost"].sum()
costs = fees + slippage
net_pnl = df["net_pnl"].sum()

print(f"Brutto P&L:       ${gross_pnl:,.4f}")
print(f"Gebühren:         ${fees:,.4f}")
print(f"Slippage:         ${slippage:,.4f}")
print(f"Gesamtkosten:     ${costs:,.4f}")
print(f"Netto P&L:        ${net_pnl:,.4f}")

print()


# ============================================================
# KOSTENANTEIL
# ============================================================

print("=" * 70)
print("9. KOSTENANALYSE")
print("=" * 70)

if gross_pnl > 0:
    cost_share = 100.0 * costs / gross_pnl
    print(f"Kostenanteil am Brutto-P&L: {cost_share:.2f}%")
else:
    print(
        "Brutto-P&L ist <= 0. "
        "Die Strategie ist bereits vor Kosten negativ."
    )

print()


# ============================================================
# MITTELWERTE
# ============================================================

print("=" * 70)
print("10. DURCHSCHNITTLICHE TRADE-KENNZAHLEN")
print("=" * 70)

print(
    f"Ø Netto pro Trade:     "
    f"${df['net_pnl'].mean():.6f}"
)

print(
    f"Ø Brutto pro Trade:    "
    f"${df['gross_pnl'].mean():.6f}"
)

print(
    f"Ø Gebühren pro Trade:  "
    f"${df['fees'].mean():.6f}"
)

print(
    f"Ø Slippage pro Trade:  "
    f"${df['slippage_cost'].mean():.6f}"
)

print(
    f"Ø Haltedauer:          "
    f"{df['duration_min'].mean():.2f} Minuten"
)

print()


# ============================================================
# TRADES PRO TAG
# ============================================================

print("=" * 70)
print("11. HANDELSFREQUENZ")
print("=" * 70)

days = (
    df["entry_time"].dt.date.nunique()
    if not df.empty
    else 0
)

if days > 0:
    trades_per_day = len(df) / days
else:
    trades_per_day = 0.0

print(f"Handelstage:     {days}")
print(f"Trades gesamt:   {len(df)}")
print(f"Trades / Tag:    {trades_per_day:.2f}")

print()


# ============================================================
# KURZE AUTOMATISCHE DIAGNOSE
# ============================================================

print("=" * 70)
print("12. AUTOMATISCHE DIAGNOSE")
print("=" * 70)

print()

if net_pnl < 0:
    print("❌ V1 ist im getesteten Zeitraum netto negativ.")
else:
    print("✅ V1 ist im getesteten Zeitraum netto positiv.")

if gross_pnl < 0:
    print(
        "❌ Wichtig: V1 ist bereits VOR Gebühren und Slippage negativ."
    )
else:
    print(
        "⚠️ Brutto ist positiv; Kosten müssen separat bewertet werden."
    )

if len(df) > 0:
    win_rate = 100.0 * (df["net_pnl"] > 0).sum() / len(df)

    print(
        f"Trefferquote: {win_rate:.2f}%"
    )

    if win_rate < 40:
        print(
            "⚠️ Die Trefferquote ist relativ niedrig "
            "für das verwendete TP/SL-Verhältnis."
        )

if trades_per_day > 10:
    print(
        "⚠️ Sehr hohe Handelsfrequenz: "
        f"{trades_per_day:.2f} Trades/Tag."
    )

short_trade_pnl = df.loc[
    df["duration_min"] <= 15,
    "net_pnl",
].sum()

short_trade_count = (
    df["duration_min"] <= 15
).sum()

if short_trade_count > 0:
    print(
        f"Trades <=15 Minuten: {short_trade_count}"
    )

    print(
        f"P&L dieser Trades: ${short_trade_pnl:.4f}"
    )

    if short_trade_pnl < 0:
        print(
            "⚠️ Kurze Trades sind ein wesentlicher "
            "Verlusttreiber."
        )

print()
print("=" * 70)
print("DIAGNOSE ABGESCHLOSSEN")
print("=" * 70)
print()
print("V1 wurde NICHT verändert.")
print("Dieser Analyzer liest ausschließlich den vorhandenen Trade-Log.")
print("===============================================================")
