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

# Can be overridden by GitHub Actions:
# BITRUE_DAYS=1
DAYS = int(os.getenv("BITRUE_DAYS", "365"))

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
print("NO LIVE TRADING")
print()
print(f"Contract: {CONTRACT_NAME}")
print(f"Interval: {INTERVAL}")
print(f"Days: {DAYS}")
print()


# ============================================================
# API REQUEST
# ============================================================

def get_klines(start_time_ms, end_time_ms):
    """
    Download one historical block of candles.
    """

    url = BASE_URL + ENDPOINT

    params = {
        "contractName": CONTRACT_NAME,
        "interval": INTERVAL,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
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
                    f"Unexpected API response type: "
                    f"{type(data).__name__}"
                )

            return data

        except Exception as exc:

            print(f"ERROR: {exc}")

            if attempt < MAX_RETRIES:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise


# ============================================================
# PARSE
# ============================================================

def parse_klines(data):
    """
    Convert Bitrue candle objects into normalized rows.
    """

    rows = []

    for candle in data:

        if not isinstance(candle, dict):
            continue

        try:

            timestamp = int(candle["idx"])

            # Defensive handling:
            # seconds -> milliseconds
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
# DOWNLOAD HISTORY
# ============================================================

def download_history():

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    print(f"Start: {start_time}")
    print(f"End:   {end_time}")
    print()

    all_rows = []

    # --------------------------------------------------------
    # Bitrue returns up to 300 candles.
    #
    # 300 x 5 minutes = 1500 minutes
    #                = 25 hours
    #
    # We therefore move through the requested period
    # block by block.
    # --------------------------------------------------------

    block_duration = timedelta(
        minutes=5 * LIMIT
    )

    current_start = start_time

    request_number = 0

    while current_start < end_time:

        current_end = min(
            current_start + block_duration,
            end_time,
        )

        start_ms = int(
            current_start.timestamp() * 1000
        )

        end_ms = int(
            current_end.timestamp() * 1000
        )

        request_number += 1

        print("=" * 60)
        print(f"REQUEST {request_number}")
        print(f"Start: {current_start}")
        print(f"End:   {current_end}")
        print("=" * 60)

        data = get_klines(
            start_ms,
            end_ms,
        )

        rows = parse_klines(data)

        print(f"Received raw candles: {len(data)}")
        print(f"Valid candles:        {len(rows)}")

        if rows:
            all_rows.extend(rows)

        # ----------------------------------------------------
        # Move forward.
        #
        # Do NOT rely only on returned candle count because
        # exchanges can occasionally return fewer candles.
        # ----------------------------------------------------

        current_start = current_end

        time.sleep(REQUEST_DELAY)

    # ========================================================
    # DATAFRAME
    # ========================================================

    if not all_rows:
        raise RuntimeError(
            "No valid Bitrue candles were downloaded."
        )

    df = pd.DataFrame(all_rows)

    # ========================================================
    # CLEANUP
    # ========================================================

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Exact requested time range
    df = df[
        (df["timestamp"] >= pd.Timestamp(start_time))
        & (df["timestamp"] <= pd.Timestamp(end_time))
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No candles remain after time filtering."
        )

    # ========================================================
    # VALIDATION
    # ========================================================

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
                f"NaN values found in column: {column}"
            )

    if (df["open"] <= 0).any():
        raise RuntimeError(
            "Invalid data: open <= 0."
        )

    if (df["high"] <= 0).any():
        raise RuntimeError(
            "Invalid data: high <= 0."
        )

    if (df["low"] <= 0).any():
        raise RuntimeError(
            "Invalid data: low <= 0."
        )

    if (df["close"] <= 0).any():
        raise RuntimeError(
            "Invalid data: close <= 0."
        )

    if (df["high"] < df["low"]).any():
        raise RuntimeError(
            "Invalid data: high < low."
        )

    if (df["volume"] < 0).any():
        raise RuntimeError(
            "Invalid data: negative volume."
        )

    # ========================================================
    # CHECK TIMESTAMP GAPS
    # ========================================================

    df["time_diff"] = (
        df["timestamp"].diff()
    )

    expected_interval = pd.Timedelta(
        minutes=5
    )

    gaps = df[
        df["time_diff"] > expected_interval
    ]

    print()
    print("=" * 60)
    print("DATA QUALITY")
    print("=" * 60)
    print()

    print(f"Total candles: {len(df)}")
    print(f"Expected approximately: {DAYS * 24 * 12}")
    print(f"Timestamp gaps: {len(gaps)}")

    if len(gaps) > 0:

        print()
        print("WARNING: Timestamp gaps detected.")

        print(
            gaps[
                [
                    "timestamp",
                    "time_diff",
                ]
            ].head(10)
        )

    # Remove helper column before saving
    df = df.drop(
        columns=["time_diff"]
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("DOWNLOAD SUCCESS")
    print("=" * 60)
    print()

    print(f"Rows:   {len(df)}")
    print(
        f"First:  {df['timestamp'].iloc[0]}"
    )
    print(
        f"Last:   {df['timestamp'].iloc[-1]}"
    )
    print(
        f"File:   {OUTPUT_FILE}"
    )

    print()
    print("FIRST 5 ROWS")
    print(df.head())

    print()
    print("LAST 5 ROWS")
    print(df.tail())

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    download_history()
