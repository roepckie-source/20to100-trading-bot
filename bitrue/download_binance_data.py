"""
BTCUSDT 5m historical data downloader
Source: Binance Public Data
Target: Bitrue strategy backtest

PAPER / BACKTEST ONLY
NO API KEY
NO LIVE TRADING
NO ORDERS
"""

import io
import os
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

SYMBOL = "BTCUSDT"
INTERVAL = "5m"

DAYS = int(os.getenv("BITRUE_DAYS", "365"))

OUTPUT_FILE = "BTCUSDT_5m.csv"

BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"

COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


# ============================================================
# DOWNLOAD
# ============================================================

def download_day(session, date):
    date_str = date.strftime("%Y-%m-%d")

    filename = f"{SYMBOL}-{INTERVAL}-{date_str}.zip"

    url = f"{BASE_URL}/{SYMBOL}/{INTERVAL}/{filename}"

    print(f"Download: {date_str}")

    response = session.get(url, timeout=30)

    if response.status_code == 404:
        print(f"  Nicht vorhanden: {date_str}")
        return None

    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise RuntimeError(
                f"Keine CSV-Datei im Archiv: {filename}"
            )

        with z.open(csv_files[0]) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=COLUMNS,
            )

    return df


# ============================================================
# MAIN DOWNLOAD
# ============================================================

def download_history():

    print("=" * 70)
    print("BINANCE BTCUSDT 5m HISTORICAL DATA")
    print("=" * 70)
    print()
    print("Quelle: Binance Public Data")
    print("Markt: USD-M Futures")
    print(f"Symbol: {SYMBOL}")
    print(f"Intervall: {INTERVAL}")
    print(f"Tage: {DAYS}")
    print(f"Ausgabe: {OUTPUT_FILE}")
    print()

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=DAYS)

    print(f"Zeitraum:")
    print(f"  {start_date}")
    print(f"  bis")
    print(f"  {end_date}")
    print()

    frames = []

    session = requests.Session()

    current = start_date

    while current <= end_date:

        try:
            df = download_day(session, current)

            if df is not None and not df.empty:
                frames.append(df)

        except Exception as exc:
            print(
                f"  FEHLER bei {current}: {exc}"
            )
            raise

        current += timedelta(days=1)

    if not frames:
        raise RuntimeError(
            "Es wurden keine historischen Daten heruntergeladen."
        )

    print()
    print("=" * 70)
    print("DATEN WERDEN ZUSAMMENGEFÜHRT")
    print("=" * 70)

    df = pd.concat(frames, ignore_index=True)

    # Timestamp
    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    # Numeric columns
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # Sort
    df = df.sort_values("timestamp")

    # Remove duplicates
    before = len(df)

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    duplicates_removed = before - len(df)

    # Keep only required columns
    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    # ========================================================
    # QUALITY CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    expected_candles = DAYS * 24 * 12

    print()
    print(f"Erwartete Kerzen:        ca. {expected_candles:,}")
    print(f"Tatsächliche Kerzen:     {len(df):,}")
    print(f"Duplikate entfernt:      {duplicates_removed:,}")

    if df.empty:
        raise RuntimeError("Dataset ist leer.")

    # Time differences
    differences = (
        df["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    expected_seconds = 5 * 60

    gaps = differences[
        differences != expected_seconds
    ]

    print(f"Zeitabstände != 5 min:   {len(gaps):,}")

    # Price sanity
    invalid_prices = df[
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
    ]

    invalid_ohlc = df[
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ]

    print(f"Ungültige Preise:        {len(invalid_prices):,}")
    print(f"Ungültige OHLC:          {len(invalid_ohlc):,}")

    print()
    print("Erste Kerze:")
    print(df.iloc[0].to_string())

    print()
    print("Letzte Kerze:")
    print(df.iloc[-1].to_string())

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("DOWNLOAD ABGESCHLOSSEN")
    print("=" * 70)

    print()
    print(f"Datei: {OUTPUT_FILE}")
    print(f"Zeilen: {len(df):,}")

    print()

    if len(df) < expected_candles * 0.98:
        raise RuntimeError(
            "Zu wenige Kerzen vorhanden."
        )

    if len(gaps) > 0:
        print(
            "WARNUNG: Es wurden Zeitlücken gefunden."
        )

    if len(invalid_prices) > 0:
        raise RuntimeError(
            "Ungültige Preise gefunden."
        )

    if len(invalid_ohlc) > 0:
        raise RuntimeError(
            "Ungültige OHLC-Daten gefunden."
        )

    print("DATA QUALITY: PASS")
    print()
    print("=" * 70)


if __name__ == "__main__":
    download_history()
