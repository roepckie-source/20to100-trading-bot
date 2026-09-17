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
import queue
import threading
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

DAYS = int(
    os.getenv("BITRUE_DAYS", "1")
)

OUTPUT_FILE = os.getenv(
    "BITRUE_OUTPUT",
    "BTCUSDT_5m.csv"
)

REQUEST_TIMEOUT = 30

REQUEST_DELAY = 0.20

CANDLE_SECONDS = 5 * 60

MAX_RECONNECTS = 5


# ============================================================
# MESSAGE QUEUE
# ============================================================

message_queue = queue.Queue()


# ============================================================
# GLOBAL CONNECTION STATE
# ============================================================

receiver_error = None
receiver_running = False


# ============================================================
# TIME HELPERS
# ============================================================

def utc_string(timestamp):
    """
    Unix timestamp in seconds -> UTC string.
    """

    return datetime.fromtimestamp(
        int(timestamp),
        tz=timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# MESSAGE DECODER
# ============================================================

def decode_message(message):
    """
    Bitrue Futures market data is gzip compressed
    binary data, except heartbeat messages.
    """

    if isinstance(message, bytes):

        try:

            message = gzip.decompress(
                message
            )

        except OSError:

            # Already uncompressed
            pass

        message = message.decode(
            "utf-8"
        )

    return json.loads(
        message
    )


# ============================================================
# PONG
# ============================================================

def send_pong(ws, ping_data):
    """
    Respond immediately to Bitrue Futures heartbeat.

    Official documented format:

        {"ping": "..."}
        {"pong": "..."}
    """

    try:

        if (
            isinstance(
                ping_data,
                dict
            )
            and "ping" in ping_data
        ):

            pong_value = ping_data[
                "ping"
            ]

        else:

            pong_value = int(
                time.time()
            )


        pong = {
            "pong": pong_value
        }


        ws.send(
            json.dumps(
                pong
            )
        )


        print(
            f"  HEARTBEAT -> PONG {pong_value}"
        )


    except Exception as exc:

        print(
            f"  HEARTBEAT ERROR: {exc}"
        )


# ============================================================
# RECEIVER THREAD
# ============================================================

def receiver_loop(ws):
    """
    Permanent WebSocket receiver.

    IMPORTANT:
    This runs independently from the historical
    request logic so Bitrue heartbeat messages
    are handled immediately.
    """

    global receiver_error
    global receiver_running

    receiver_running = True

    print(
        "WebSocket receiver thread started."
    )


    while receiver_running:

        try:

            message = ws.recv()


            if not message:

                continue


            # ------------------------------------------------
            # DECODE MESSAGE
            # ------------------------------------------------

            try:

                data = decode_message(
                    message
                )

            except Exception as exc:

                print(
                    "  WARNING: "
                    f"Could not decode message: {exc}"
                )

                continue


            # ------------------------------------------------
            # HEARTBEAT
            # ------------------------------------------------

            if isinstance(
                data,
                dict
            ):

                if "ping" in data:

                    send_pong(
                        ws,
                        data
                    )

                    continue


                # Some Bitrue services use
                # event=ping.

                if data.get(
                    "event"
                ) == "ping":

                    send_pong(
                        ws,
                        data
                    )

                    continue


            # ------------------------------------------------
            # NORMAL MESSAGE
            # ------------------------------------------------

            message_queue.put(
                data
            )


        except (
            websocket.WebSocketConnectionClosedException,
            websocket.WebSocketTimeoutException
        ) as exc:

            receiver_error = exc

            print(
                f"WebSocket receiver stopped: {exc}"
            )

            break


        except Exception as exc:

            receiver_error = exc

            print(
                f"WebSocket receiver error: {exc}"
            )

            break


    receiver_running = False


# ============================================================
# CONNECT
# ============================================================

def connect():

    print(
        "Connecting to Bitrue WebSocket..."
    )


    ws = websocket.create_connection(
        WS_URL,
        timeout=REQUEST_TIMEOUT
    )


    print(
        "CONNECTED"
    )


    # --------------------------------------------------------
    # INITIAL UNSOLICITED PONG
    # --------------------------------------------------------

    try:

        ws.send(
            json.dumps(
                {
                    "pong": int(
                        time.time()
                    )
                }
            )
        )

        print(
            "Initial PONG sent"
        )

    except Exception as exc:

        print(
            f"WARNING: Initial PONG failed: {exc}"
        )


    # --------------------------------------------------------
    # START RECEIVER
    # --------------------------------------------------------

    thread = threading.Thread(
        target=receiver_loop,
        args=(ws,),
        daemon=True
    )

    thread.start()


    return ws


# ============================================================
# REQUEST HISTORICAL PAGE
# ============================================================

def request_page(
    ws,
    end_idx
):

    # --------------------------------------------------------
    # CLEAR OLD QUEUE
    # --------------------------------------------------------

    while True:

        try:

            message_queue.get_nowait()

        except queue.Empty:

            break


    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    request = {
        "event": "req",
        "params": {
            "channel": CHANNEL,
            "cb_id": "",
            "endIdx": str(
                end_idx
            ),
            "pageSize": PAGE_SIZE
        }
    }


    print(
        "  Sending historical request..."
    )


    print(
        f"  endIdx: {end_idx}"
    )


    ws.send(
        json.dumps(
            request
        )
    )


    # --------------------------------------------------------
    # WAIT FOR RESPONSE
    #
    # Heartbeats are handled independently
    # by receiver_loop().
    # --------------------------------------------------------

    deadline = (
        time.time()
        +
        REQUEST_TIMEOUT
    )


    while time.time() < deadline:

        remaining = (
            deadline
            -
            time.time()
        )


        if remaining <= 0:

            break


        try:

            data = message_queue.get(
                timeout=min(
                    1.0,
                    remaining
                )
            )

        except queue.Empty:

            if receiver_error:

                raise RuntimeError(
                    "Bitrue WebSocket receiver "
                    "stopped unexpectedly: "
                    f"{receiver_error}"
                )

            continue


        # ----------------------------------------------------
        # CHECK RESPONSE
        # ----------------------------------------------------

        if not isinstance(
            data,
            dict
        ):

            continue


        if data.get(
            "channel"
        ) != CHANNEL:

            continue


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        status = data.get(
            "status"
        )


        if status != "ok":

            raise RuntimeError(
                "Bitrue returned "
                f"status={status}: {data}"
            )


        # ----------------------------------------------------
        # HISTORICAL DATA
        # ----------------------------------------------------

        rows = data.get(
            "data"
        )


        if not isinstance(
            rows,
            list
        ):

            raise RuntimeError(
                "Unexpected Bitrue response: "
                f"{data}"
            )


        return rows


    raise TimeoutError(
        "Timeout waiting for Bitrue "
        "historical data."
    )


# ============================================================
# DOWNLOAD
# ============================================================

def download_history():

    print(
        "=" * 70
    )

    print(
        "BITRUE BTC/USDT 5m "
        "HISTORICAL DOWNLOADER"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Mode:"
    )

    print(
        "  PAPER / BACKTEST DATA ONLY"
    )

    print(
        "  NO API KEY"
    )

    print(
        "  NO LIVE TRADING"
    )

    print(
        "  NO ORDERS"
    )

    print()

    print(
        f"WebSocket: {WS_URL}"
    )

    print(
        f"Channel:   {CHANNEL}"
    )

    print(
        f"Days:      {DAYS}"
    )

    print(
        f"Page size: {PAGE_SIZE}"
    )

    print()


    # ========================================================
    # EXPECTED CANDLES
    # ========================================================

    target_candles = (
        DAYS
        *
        24
        *
        60
        //
        5
    )


    print(
        "Expected candles: "
        f"approximately {target_candles}"
    )

    print()


    # ========================================================
    # CONNECT
    # ========================================================

    ws = connect()


    # ========================================================
    # STORAGE
    # ========================================================

    all_rows = {}


    # ========================================================
    # START TIMESTAMP
    #
    # Bitrue historical WebSocket uses Unix seconds.
    # ========================================================

    end_idx = int(
        time.time()
    )


    page = 0

    reconnect_count = 0


    # ========================================================
    # PAGINATION
    # ========================================================

    try:

        while (
            len(all_rows)
            <
            target_candles
        ):

            page += 1


            print()

            print(
                f"Request page {page} | "
                f"endIdx={end_idx} | "
                f"candles={len(all_rows)}"
            )


            try:

                rows = request_page(
                    ws,
                    end_idx
                )


            except (
                RuntimeError,
                TimeoutError,
                websocket.WebSocketException
            ) as exc:

                print()

                print(
                    "WARNING: "
                    f"Historical request failed: {exc}"
                )


                reconnect_count += 1


                if reconnect_count > MAX_RECONNECTS:

                    raise RuntimeError(
                        "Maximum WebSocket reconnect "
                        "attempts exceeded."
                    )


                print(
                    f"Reconnecting "
                    f"({reconnect_count}/"
                    f"{MAX_RECONNECTS})..."
                )


                try:

                    ws.close()

                except Exception:

                    pass


                time.sleep(
                    2
                )


                ws = connect()


                page -= 1

                continue


            # ------------------------------------------------
            # SUCCESSFUL REQUEST
            # ------------------------------------------------

            reconnect_count = 0


            if not rows:

                print(
                    "No historical data returned."
                )

                break


            # ------------------------------------------------
            # PROCESS CANDLES
            # ------------------------------------------------

            for row in rows:

                if "id" not in row:

                    continue


                candle_id = int(
                    row["id"]
                )


                try:

                    candle = {
                        "timestamp":
                            utc_string(
                                candle_id
                            ),

                        "timestamp_unix":
                            candle_id,

                        "open":
                            float(
                                row["open"]
                            ),

                        "high":
                            float(
                                row["high"]
                            ),

                        "low":
                            float(
                                row["low"]
                            ),

                        "close":
                            float(
                                row["close"]
                            ),

                        "volume":
                            float(
                                row["vol"]
                            )
                    }


                except (
                    KeyError,
                    TypeError,
                    ValueError
                ) as exc:

                    print(
                        "WARNING: Invalid candle:"
                    )

                    print(
                        row
                    )

                    print(
                        exc
                    )

                    continue


                all_rows[
                    candle_id
                ] = candle


            # ------------------------------------------------
            # IDS
            # ------------------------------------------------

            ids = [
                int(
                    row["id"]
                )

                for row in rows

                if "id" in row
            ]


            if not ids:

                raise RuntimeError(
                    "Bitrue response contained "
                    "no candle IDs."
                )


            oldest_id = min(
                ids
            )


            newest_id = max(
                ids
            )


            print(
                f"  received: {len(rows)}"
            )

            print(
                "  oldest:   "
                f"{utc_string(oldest_id)} UTC"
            )

            print(
                "  newest:   "
                f"{utc_string(newest_id)} UTC"
            )

            print(
                f"  total:    {len(all_rows)}"
            )


            # ------------------------------------------------
            # PAGINATION
            # ------------------------------------------------

            next_end_idx = (
                oldest_id
                -
                1
            )


            if next_end_idx >= end_idx:

                raise RuntimeError(
                    "Pagination did not move backwards."
                )


            end_idx = next_end_idx


            time.sleep(
                REQUEST_DELAY
            )


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
        key=lambda x:
            x["timestamp_unix"]
    )


    # ========================================================
    # LIMIT TO TARGET
    # ========================================================

    if len(rows) > target_candles:

        rows = rows[
            -target_candles:
        ]


    # ========================================================
    # WRITE CSV
    # ========================================================

    print()

    print(
        "Writing CSV..."
    )


    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )


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

    print(
        "=" * 70
    )

    print(
        "DATA VALIDATION"
    )

    print(
        "=" * 70
    )


    print(
        f"Rows:       {len(rows)}"
    )


    if rows:

        print(
            "First:      "
            f"{rows[0]['timestamp']} UTC"
        )

        print(
            "Last:       "
            f"{rows[-1]['timestamp']} UTC"
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
        len(
            set(timestamps)
        )
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

        delta = (
            current
            -
            previous
        )


        if delta != CANDLE_SECONDS:

            gaps += 1


    print(
        f"Gaps:       {gaps}"
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "DOWNLOAD COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"CSV:      {OUTPUT_FILE}"
    )

    print(
        f"Candles:  {len(rows)}"
    )

    print()


    # ========================================================
    # QUALITY
    # ========================================================

    if (
        len(rows)
        <
        target_candles * 0.95
    ):

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
# MAIN
# ============================================================

if __name__ == "__main__":

    download_history()
