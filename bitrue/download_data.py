"""
Bitrue BTC/USDT Futures 5m Historical Data Downloader

PAPER / BACKTEST ONLY
NO API KEY
NO ORDERS
NO WALLET
NO LIVE TRADING
"""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://fapi.bitrue.com"
ENDPOINT = "/fapi/v1/klines"

CONTRACT_NAME = "E-BTC-USDT"
INTERVAL = "5min"

LIMIT = 300
DAYS = int(os.getenv("BITRUE_DAYS", "1"))

OUTPUT_FILE = "BTCUSDT_5m.csv"

REQUEST_DELAY = 0.25
MAX_RETRIES = 5


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("BITRUE BTC/USDT FUTURES 5M DATA DOWNLOADER")
print("=" * 60)
print()
print("PUBLIC MARKET DATA ONLY")
print("NO API KEY")
print("NO ORDERS")
print("NO WALLET")
print()
print(f"Contract: {CONTRACT_NAME}")
print(f"Interval: {INTERVAL}")
print(f"Days: {DAYS}")
print()


# ============================================================
# API REQUEST
# ============================================================

def get_klines():

    url = BASE_URL + ENDPOINT

    params = {
        "contractName": CONTRACT_NAME,
        "interval": INTERVAL,
        "limit": LIMIT,
    }

    for attempt in range(1, MAX_RETRIES + 1):

        print(f"Request attempt {attempt}/{MAX_RETRIES}")

        try:

            response = requests.get(
                url,
                params=params,
                timeout=20,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Bitrue-BTC-5m-Paper-Downloader/1.0",
                },
            )

            print(f"HTTP status: {response.status_code}")

            response.raise_for_status()

            data = response.json()

            if not isinstance(data, list):
                raise RuntimeError(
                    f"Unexpected API response type: {type(data).__name__}"
                )

            print(f"Received candles: {len(data)}")

            return data

        except Exception as exc:

            print(f"ERROR: {exc}")

            if attempt < MAX_RETRIES:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise


# ============================================================
# PARSE KLINES
# ============================================================

def parse_klines(data):

    rows = []

    for candle in data:

        if not isinstance(candle, dict):
            continue

        try:

            timestamp = candle["idx"]

            # Bitrue documentation specifies milliseconds,
            # but accept seconds defensively as well.
            timestamp = int(timestamp)

            if timestamp < 10_000_000_000:
                timestamp *= 1000

            rows.append(
                {
                    "timestamp": pd.to_datetime(
                        timestamp,
                        unit="ms",
                        utc=True,
                    ),
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle["vol"]),
                }
            )

        except (KeyError, TypeError, ValueError):
            continue

    return rows


# ============================================================
# DOWNLOAD
# ============================================================

def download_history():

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    print(f"Start: {start_time}")
    print(f"End:   {end_time}")
    print()

    # --------------------------------------------------------
    # IMPORTANT:
    # The public Kline endpoint returns the newest candles.
    # We first perform a connectivity/data-shape test.
    # --------------------------------------------------------

    raw_data = get_klines()

    rows = parse_klines(raw_data)

    if not rows:
        raise RuntimeError("Bitrue returned no valid candles.")

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(subset=["timestamp"])

    df = df.sort_values("timestamp")

    # --------------------------------------------------------
    # Filter requested period
    # --------------------------------------------------------

    df = df[
        (df["timestamp"] >= pd.Timestamp(start_time))
        & (df["timestamp"] <= pd.Timestamp(end_time))
    ].copy()

    if df.empty:
        raise RuntimeError(
            "Bitrue returned candles, but none are inside the requested period."
        )

    # --------------------------------------------------------
    # Validate OHLCV
    # --------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        if df[column].isna().any():
            raise RuntimeError(
                f"Invalid NaN values found in column: {column}"
            )

    if (df["high"] < df["low"]).any():
        raise RuntimeError("Invalid candle data: high < low.")

    if (df["open"] <= 0).any():
        raise RuntimeError("Invalid candle data: open <= 0.")

    if (df["close"] <= 0).any():
        raise RuntimeError("Invalid candle data: close <= 0.")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 60)
    print("DOWNLOAD SUCCESS")
    print("=" * 60)
    print()
    print(f"Rows:       {len(df)}")
    print(f"First:      {df['timestamp'].iloc[0]}")
    print(f"Last:       {df['timestamp'].iloc[-1]}")
    print(f"Output:     {OUTPUT_FILE}")
    print()
    print(df.head())
    print()
    print(df.tail())
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    download_history()
