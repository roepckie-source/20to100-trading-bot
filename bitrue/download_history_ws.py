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

# Default: 1 day
# Can be overridden by GitHub Actions:
# BITRUE_DAYS=1
# BITRUE_DAYS=365
DAYS = int(os.getenv("BITRUE_DAYS", "1"))

OUTPUT_FILE = os.getenv(
    "BITRUE_OUTPUT",
    "BTCUSDT_5m.csv"
)

REQUEST_TIMEOUT = 30

# Small pause between historical requests
REQUEST_DELAY = 0.15

# 5-minute candles
CANDLE_SECONDS = 5 * 60


# ============================================================
# MESSAGE DECODER
# ============================================================

def decode_message(message):
    """
    Bitrue market-data messages can arrive as
    gzip-compressed binary data.

    Heartbeat messages may arrive as plain text.
    """

    if isinstance(message, bytes):

        try:
            message = gzip.decompress(message)

        except OSError:
            # Message may already be plain bytes
            pass

        message = message.decode("utf-8")

    return json.loads(message)


# ============================================================
# UTC TIME
# ============================================================

def utc_string(timestamp):
    """
    Convert Unix timestamp in seconds to UTC string.
    """

    return datetime.fromtimestamp(
        int(timestamp),
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# HEARTBEAT / PONG
# ============================================================

def send_pong(ws, data):
    """
    Respond to Bitrue heartbeat messages.

    Supported formats:

        {"ping": "..."} 

    and

        {"event": "ping", "ts": "..."}
    """

    # --------------------------------------------------------
    # FORMAT 1
    # {"ping": "..."}
    # --------------------------------------------------------

    if isinstance(data, dict) and "ping" in data:

        pong = {
            "pong": data["ping"]
        }

        ws.send(json.dumps(pong))

        print("  HEARTBEAT: pong sent")

        return


    # --------------------------------------------------------
    # FORMAT 2
    # {"event": "ping", "ts": "..."}
    # --------------------------------------------------------

    if (
        isinstance(data, dict)
        and data.get("event") == "ping"
    ):

        pong = {
            "event": "pong",
            "ts": data.get("ts")
        }

        ws.send(json.dumps(pong))

        print("  HEARTBEAT: event pong sent")

        return


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

    print()
    print("  Sending historical request...")

    ws.send(json.dumps(request))

    deadline = time.time() + REQUEST_TIMEOUT

    while time.time() < deadline:

        try:

            message = ws.recv()

        except websocket.WebSocketTimeoutException:

            continue

        except websocket.WebSocketConnectionClosedException:

            raise RuntimeError(
                "Bitrue WebSocket connection was closed "
                "while waiting for historical data."
            )

        if not message:
            continue

        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        try:

            data = decode_message(message)

        except Exception as exc:

            print(
                f"  WARNING: Could not decode message: {exc}"
            )

            continue


        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if isinstance(data, dict):

            # {"ping": "..."}
            if "ping" in data:

                send_pong(ws, data)

                continue

            # {"event": "ping", ...}
            if data.get("event") == "ping":

                send_pong(ws, data)

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
                f"Unexpected Bitrue response: {data}"
            )


        return rows


    raise TimeoutError(
        "Timeout waiting for Bitrue historical data."
    )


# ============================================================
# DOWNLOAD HISTORY
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

    # --------------------------------------------------------
    # EXPECTED CANDLES
    # --------------------------------------------------------

    target_candles = DAYS * 24 * 60 // 5

    print(
        f"Expected candles: approximately {target_candles}"
    )

    print()


    # ========================================================
    # CONNECT
    # ========================================================

    print("Connecting to Bitrue WebSocket...")

    try:

        ws = websocket.create_connection(
            WS_URL,
            timeout=REQUEST_TIMEOUT
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not connect to Bitrue WebSocket: {exc}"
        )


    print("CONNECTED")

    print()


    # --------------------------------------------------------
    # INITIAL PONG
    # --------------------------------------------------------

    try:

        ws.send(
            json.dumps(
                {
                    "pong": int(time.time())
                }
            )
        )

        print("Initial PONG sent")

    except Exception as exc:

        print(
            f"WARNING: Initial PONG failed: {exc}"
        )

    print()


    # ========================================================
    # STORAGE
    # ========================================================

    all_rows = {}


    # --------------------------------------------------------
    # START AT CURRENT TIME
    #
    # IMPORTANT:
    # WebSocket historical API uses Unix SECONDS.
    #
    # --------------------------------------------------------

    end_idx = int(time.time())

    page = 0


    try:

        # ====================================================
        # PAGINATION LOOP
        # ====================================================

        while len(all_rows) < target_candles:

            page += 1

            print(
                f"Request page {page} | "
                f"endIdx={end_idx} | "
                f"candles={len(all_rows)}"
            )


            # ------------------------------------------------
            # REQUEST
            # ------------------------------------------------

            rows = request_page(
                ws,
                end_idx
            )


            # ------------------------------------------------
            # NO DATA
            # ------------------------------------------------

            if not rows:

                print(
                    "No more historical data returned."
                )

                break


            # ------------------------------------------------
            # STORE CANDLES
            # ------------------------------------------------

            for row in rows:

                if "id" not in row:

                    continue

                candle_id = int(row["id"])


                try:

                    candle = {
                        "timestamp": utc_string(candle_id),
                        "timestamp_unix": candle_id,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["vol"])
                    }

                except (KeyError, TypeError, ValueError) as exc:

                    print(
                        f"  WARNING: Invalid candle: "
                        f"{row} ({exc})"
                    )

                    continue


                all_rows[candle_id] = candle


            # ------------------------------------------------
            # IDS
            # ------------------------------------------------

            ids = [
                int(row["id"])
                for row in rows
                if "id" in row
            ]


            if not ids:

                raise RuntimeError(
                    "Bitrue response contained no candle IDs."
                )


            oldest_id = min(ids)

            newest_id = max(ids)


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            print(
                f"  received: {len(rows)}"
            )

            print(
                f"  oldest:  {utc_string(oldest_id)} UTC"
            )

            print(
                f"  newest:  {utc_string(newest_id)} UTC"
            )

            print(
                f"  total:   {len(all_rows)}"
            )


            # ------------------------------------------------
            # PAGINATION SAFETY CHECK
            # ------------------------------------------------

            next_end_idx = oldest_id - 1


            if next_end_idx >= end_idx:

                raise RuntimeError(
                    "Pagination did not move backwards."
                )


            # ------------------------------------------------
            # MOVE BACKWARD
            # ------------------------------------------------

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


    # --------------------------------------------------------
    # KEEP TARGET SIZE
    # --------------------------------------------------------

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


        writer.writerow(
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )


        for row in rows:

            writer.writerow(
                [
                    row["timestamp"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"]
                ]
            )


    # ========================================================
    # VALIDATION
    # ========================================================

    print()

    print("=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)


    print(
        f"Rows:       {len(rows)}"
    )


    if rows:

        print(
            f"First:      {rows[0]['timestamp']} UTC"
        )

        print(
            f"Last:       {rows[-1]['timestamp']} UTC"
        )


    # ========================================================
    # DUPLICATES
    # ========================================================

    timestamps = [
        row["timestamp_unix"]
        for row in rows
    ]


    duplicates = (
        len(timestamps)
        -
        len(set(timestamps))
    )


    print(
        f"Duplicates: {duplicates}"
    )


    # ========================================================
    # GAPS
    # ========================================================

    gaps = 0


    for previous, current in zip(
        timestamps,
        timestamps[1:]
    ):

        delta = current - previous


        if delta != CANDLE_SECONDS:

            gaps += 1


    print(
        f"Gaps:       {gaps}"
    )


    # ========================================================
    # RESULT
    # ========================================================

    print()

    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print()

    print(
        f"CSV:      {OUTPUT_FILE}"
    )

    print(
        f"Candles:  {len(rows)}"
    )

    print()


    # --------------------------------------------------------
    # QUALITY CHECK
    # --------------------------------------------------------

    if len(rows) < target_candles * 0.95:

        print(
            "DATA QUALITY: WARNING"
        )

        print(
            "Fewer candles than expected."
        )


    elif duplicates > 0:

        print(
            "DATA QUALITY: WARNING"
        )

        print(
            "Duplicate timestamps detected."
        )


    elif gaps > 0:

        print(
            "DATA QUALITY: WARNING"
        )

        print(
            "Timestamp gaps detected."
        )


    else:

        print(
            "DATA QUALITY: PASS"
        )


    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    download_history()
