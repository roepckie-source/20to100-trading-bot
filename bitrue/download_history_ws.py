"""
Bitrue BTC/USDT Futures
Historical 5m Kline Downloader via WebSocket

PAPER / BACKTEST DATA ONLY
NO API KEY
NO LIVE TRADING
NO ORDERS
"""

import csv
import gzip
import json
import os
import time
from datetime import datetime, timezone

import websocket


# ============================================================
# CONFIG
# ============================================================

WS_URL = "wss://fmarket-ws.bitrue.com/kline-api/ws"

SYMBOL = "e_btcusdt"
INTERVAL = "5min"
CHANNEL = f"market_{SYMBOL}_kline_{INTERVAL}"

PAGE_SIZE = 300

DAYS = int(os.getenv("BITRUE_DAYS", "1"))

OUTPUT_FILE = os.getenv(
    "BITRUE_OUTPUT",
    "BTCUSDT_5m.csv"
)

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.15

CANDLE_SECONDS = 5 * 60


# ============================================================
# HELPERS
# ============================================================

def decode_message(message):
    """
    Bitrue sends market-data messages gzip-compressed
    binary, except heartbeat messages.
    """

    if isinstance(message, bytes):

        try:
            message = gzip.decompress(message)

        except OSError:
            # Some messages may already be plain bytes
            pass

        message = message.decode("utf-8")

    return json.loads(message)


def utc_string(timestamp):
    """Unix seconds -> UTC string."""

    return datetime.fromtimestamp(
        int(timestamp),
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


def send_pong(ws, value):
    """
    Bitrue heartbeat response.
    """

    pong = {
        "pong": value
    }

    ws.send(json.dumps(pong))


# ============================================================
# REQUEST ONE HISTORICAL PAGE
# ============================================================

def request_page(ws, end_idx):

    request = {
        "event": "req",
        "params": {
            "channel": CHANNEL,
            "cb_id": "",
            "endIdx": str(end_idx),
            "pageSize": PAGE_SIZE
        }
    }

    ws.send(json.dumps(request))

    deadline = time.time() + REQUEST_TIMEOUT

    while time.time() < deadline:

        try:
            message = ws.recv()

        except websocket.WebSocketTimeoutException:
            continue

        if not message:
            continue

        try:
            data = decode_message(message)

        except Exception as exc:
            print(
                f"WARNING: Could not decode message: {exc}"
            )
            continue

        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if isinstance(data, dict) and "ping" in data:

            send_pong(ws, data["ping"])
            continue

        # ----------------------------------------------------
        # HISTORICAL RESPONSE
        # ----------------------------------------------------

        if not isinstance(data, dict):
            continue

        if data.get("event_rep") != "rep":
            continue

        if data.get("channel") != CHANNEL:
            continue

        status = data.get("status")

        if status != "ok":
            raise RuntimeError(
                f"Bitrue returned status={status}: {data}"
            )

        rows = data.get("data")

        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected response: {data}"
            )

        return rows

    raise TimeoutError(
        "Timeout waiting for Bitrue historical data"
    )


# ============================================================
# MAIN DOWNLOAD
# ============================================================

def download_history():

    print("=" * 70)
    print("BITRUE BTC/USDT 5m HISTORICAL DOWNLOADER")
    print("=" * 70)

    print()
    print("Mode:")
    print("  PAPER / BACKTEST DATA ONLY")
    print("  NO API KEY")
    print("  NO LIVE TRADING")
    print("  NO ORDERS")
    print()

    print(f"WebSocket: {WS_URL}")
    print(f"Channel:   {CHANNEL}")
    print(f"Days:      {DAYS}")
    print(f"Page size: {PAGE_SIZE}")
    print()

    target_candles = DAYS * 24 * 60 // 5

    print(f"Expected candles: approximately {target_candles}")
    print()

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    print("Connecting to Bitrue WebSocket...")

    ws = websocket.create_connection(
        WS_URL,
        timeout=REQUEST_TIMEOUT
    )

    print("CONNECTED")
    print()

    all_rows = {}

    # Start from current Unix timestamp in SECONDS.
    #
    # Important:
    # Bitrue WebSocket historical Klines use Unix seconds
    # for endIdx / id.
    #
    end_idx = int(time.time())

    page = 0

    try:

        while len(all_rows) < target_candles:

            page += 1

            print(
                f"Request page {page} | "
                f"endIdx={end_idx} | "
                f"candles={len(all_rows)}"
            )

            rows = request_page(
                ws,
                end_idx
            )

            if not rows:

                print("No more historical data returned.")
                break

            # ------------------------------------------------
            # STORE CANDLES
            # ------------------------------------------------

            for row in rows:

                candle_id = int(row["id"])

                all_rows[candle_id] = {
                    "timestamp": utc_string(candle_id),
                    "timestamp_unix": candle_id,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["vol"])
                }

            ids = [
                int(row["id"])
                for row in rows
                if "id" in row
            ]

            if not ids:
                raise RuntimeError(
                    "Response contained no candle IDs."
                )

            oldest_id = min(ids)

            newest_id = max(ids)

            print(
                f"  received: {len(rows)}"
            )

            print(
                f"  oldest:  {utc_string(oldest_id)}"
            )

            print(
                f"  newest:  {utc_string(newest_id)}"
            )

            print(
                f"  total:   {len(all_rows)}"
            )

            # ------------------------------------------------
            # MOVE BACKWARD
            # ------------------------------------------------

            next_end_idx = oldest_id - 1

            if next_end_idx >= end_idx:

                raise RuntimeError(
                    "Pagination did not move backwards."
                )

            end_idx = next_end_idx

            time.sleep(REQUEST_DELAY)

    finally:

        try:
            ws.close()

        except Exception:
            pass

    # ========================================================
    # SORT
    # ========================================================

    rows = sorted(
        all_rows.values(),
        key=lambda x: x["timestamp_unix"]
    )

    # Only keep requested number of candles if we received
    # slightly more than necessary.

    if len(rows) > target_candles:

        rows = rows[-target_candles:]

    # ========================================================
    # WRITE CSV
    # ========================================================

    print()
    print("Writing CSV...")

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ])

        for row in rows:

            writer.writerow([
                row["timestamp"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"]
            ])

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    print(f"Rows:       {len(rows)}")

    if rows:

        print(
            f"First:      {rows[0]['timestamp']} UTC"
        )

        print(
            f"Last:       {rows[-1]['timestamp']} UTC"
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    timestamps = [
        row["timestamp_unix"]
        for row in rows
    ]

    duplicates = len(timestamps) - len(set(timestamps))

    print(f"Duplicates: {duplicates}")

    # --------------------------------------------------------
    # GAPS
    # --------------------------------------------------------

    gaps = 0

    for previous, current in zip(
        timestamps,
        timestamps[1:]
    ):

        delta = current - previous

        if delta != CANDLE_SECONDS:

            gaps += 1

    print(f"Gaps:       {gaps}")

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print()
    print(f"CSV: {OUTPUT_FILE}")
    print(f"Candles: {len(rows)}")
    print()

    if len(rows) < target_candles * 0.95:

        print(
            "WARNING: Fewer candles than expected."
        )

    elif duplicates > 0:

        print(
            "WARNING: Duplicate timestamps detected."
        )

    elif gaps > 0:

        print(
            "WARNING: Timestamp gaps detected."
        )

    else:

        print(
            "DATA QUALITY: PASS"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    download_history()
