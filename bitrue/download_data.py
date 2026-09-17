"""
Bitrue BTC/USDT 5m Historical Data Downloader

Downloads public Bitrue Futures Kline data.

PAPER / BACKTEST ONLY
NO API KEY
NO ORDERS
NO WALLET
"""

import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SYMBOL = "BTCUSDT"

INTERVAL = "5m"

# 12 months
DAYS = 365

OUTPUT_FILE = "BTCUSDT_5m.csv"

# Bitrue historical endpoint
BASE_URL = (
    "https://openapi.bitrue.com"
)

ENDPOINT = (
    "/futures/v2/public/candlestick"
)

# Conservative request size
LIMIT = 300

REQUEST_DELAY = 0.25

TIMEOUT = 20


# ============================================================
# API REQUEST
# ============================================================

def get_klines(
    start_time_ms: int,
    end_time_ms: int,
):
    """
    Download one batch of Bitrue Futures candles.
    """

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": LIMIT,
    }

    response = requests.get(
        BASE_URL + ENDPOINT,
        params=params,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    # --------------------------------------------------------
    # Basic response handling
    # --------------------------------------------------------

    if isinstance(payload, dict):

        if "data" in payload:
            data = payload["data"]

        elif "result" in payload:
            data = payload["result"]

        else:
            raise RuntimeError(
                f"Unexpected API response: {payload}"
            )

    elif isinstance(payload, list):

        data = payload

    else:

        raise RuntimeError(
            f"Unexpected API response type: "
            f"{type(payload)}"
        )

    if not data:
        return []

    return data


# ============================================================
# CANDLE PARSER
# ============================================================

def parse_candle(candle):
    """
    Convert Bitrue candle into standard OHLCV format.

    Bitrue API formats may differ between API versions,
    therefore common list/dict structures are supported.
    """

    # --------------------------------------------------------
    # Dictionary response
    # --------------------------------------------------------

    if isinstance(candle, dict):

        timestamp = (
            candle.get("openTime")
            or candle.get("timestamp")
            or candle.get("time")
        )

        open_price = candle.get("open")
        high_price = candle.get("high")
        low_price = candle.get("low")
        close_price = candle.get("close")

        volume = (
            candle.get("volume")
            or candle.get("vol")
        )

        if timestamp is None:
            raise ValueError(
                f"Cannot find timestamp: {candle}"
            )

        return {
            "timestamp": int(timestamp),
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
        }

    # --------------------------------------------------------
    # List response
    # --------------------------------------------------------

    if isinstance(candle, list):

        if len(candle) < 6:

            raise ValueError(
                f"Unexpected candle format: {candle}"
            )

        return {
            "timestamp": int(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        }

    raise ValueError(
        f"Unsupported candle type: "
        f"{type(candle)}"
    )


# ============================================================
# DOWNLOAD
# ============================================================

def download_history():

    print("=" * 60)
    print("BITRUE BTC/USDT 5M DATA DOWNLOADER")
    print("=" * 60)

    print()

    print("PUBLIC MARKET DATA ONLY")
    print("NO API KEY")
    print("NO ORDERS")
    print("NO WALLET")

    print()

    end_time = datetime.now(
        timezone.utc
    )

    start_time = (
        end_time
        - timedelta(days=DAYS)
    )

    print(
        f"Start: {start_time}"
    )

    print(
        f"End:   {end_time}"
    )

    print()

    # --------------------------------------------------------
    # Convert to milliseconds
    # --------------------------------------------------------

    cursor = int(
        start_time.timestamp() * 1000
    )

    final_time = int(
        end_time.timestamp() * 1000
    )

    all_rows = []

    candle_duration_ms = (
        5 * 60 * 1000
    )

    request_count = 0

    # --------------------------------------------------------
    # DOWNLOAD LOOP
    # --------------------------------------------------------

    while cursor < final_time:

        # Maximum time represented by 300 candles
        batch_end = min(
            cursor
            + candle_duration_ms * LIMIT,
            final_time,
        )

        request_count += 1

        print(
            f"Request {request_count:4d} | "
            f"{datetime.fromtimestamp(cursor / 1000, tz=timezone.utc)}"
        )

        try:

            candles = get_klines(
                cursor,
                batch_end,
            )

        except Exception as exc:

            print()
            print(
                "ERROR:"
            )

            print(exc)

            print()

            print(
                "Retrying in 5 seconds..."
            )

            time.sleep(5)

            continue

        if not candles:

            print(
                "No candles returned."
            )

            cursor = batch_end

            time.sleep(
                REQUEST_DELAY
            )

            continue

        parsed = []

        for candle in candles:

            try:

                parsed.append(
                    parse_candle(candle)
                )

            except Exception as exc:

                print(
                    f"Skipping invalid candle: "
                    f"{exc}"
                )

        all_rows.extend(
            parsed
        )

        # ----------------------------------------------------
        # Determine newest timestamp
        # ----------------------------------------------------

        timestamps = [
            row["timestamp"]
            for row in parsed
        ]

        if timestamps:

            newest_timestamp = max(
                timestamps
            )

            next_cursor = (
                newest_timestamp
                + candle_duration_ms
            )

            # Safety against infinite loop
            if next_cursor <= cursor:

                next_cursor = batch_end

            cursor = next_cursor

        else:

            cursor = batch_end

        print(
            f"  Candles received: "
            f"{len(parsed)}"
        )

        print(
            f"  Total candles:     "
            f"{len(all_rows)}"
        )

        time.sleep(
            REQUEST_DELAY
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    if not all_rows:

        raise RuntimeError(
            "No market data downloaded."
        )

    df = pd.DataFrame(
        all_rows
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    )

    # --------------------------------------------------------
    # Keep standard columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if df.empty:

        raise RuntimeError(
            "Downloaded DataFrame is empty."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print()

    print(
        f"File:        {OUTPUT_FILE}"
    )

    print(
        f"Candles:     {len(df):,}"
    )

    print(
        f"First:       {df['timestamp'].iloc[0]}"
    )

    print(
        f"Last:        {df['timestamp'].iloc[-1]}"
    )

    print()

    print(
        f"BTC first:   "
        f"${df['open'].iloc[0]:,.2f}"
    )

    print(
        f"BTC last:    "
        f"${df['close'].iloc[-1]:,.2f}"
    )

    print()

    print(
        "DATA ONLY"
    )

    print(
        "NO REAL TRADING"
    )

    print(
        "NO API KEY"
    )

    print(
        "NO ORDERS"
    )

    print(
        "NO WALLET"
    )

    print()

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    download_history()
