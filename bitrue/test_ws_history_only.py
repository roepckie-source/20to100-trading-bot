"""
Bitrue Futures WebSocket
FINAL HISTORICAL KLINE MINIMAL TEST

NO API KEY
NO LIVE TRADING
NO ORDERS
"""

import gzip
import json
import time

import websocket


WS_URL = "wss://fmarket-ws.bitrue.com/kline-api/ws"

CHANNEL = "market_e_btcusdt_kline_5min"


def decode_message(message):

    if isinstance(message, bytes):

        try:
            message = gzip.decompress(message)
        except OSError:
            pass

        message = message.decode("utf-8")

    return json.loads(message)


print("=" * 70)
print("BITRUE HISTORICAL KLINE MINIMAL TEST")
print("=" * 70)

print()
print(f"URL:     {WS_URL}")
print(f"Channel: {CHANNEL}")
print()


# ============================================================
# CONNECT
# ============================================================

print("CONNECTING...")

ws = websocket.create_connection(
    WS_URL,
    timeout=10,
    suppress_origin=True
)

print("CONNECTED")
print()


# ============================================================
# HISTORICAL REQUEST
#
# EXACT FORMAT FROM BITRUE DOCUMENTATION
# ============================================================

end_idx = (
    int(time.time()) // 300
) * 300


request = {
    "event": "req",
    "params": {
        "channel": CHANNEL,
        "cb_id": "",
        "endIdx": str(end_idx),
        "pageSize": 10
    }
}


print("HISTORICAL REQUEST")
print("-" * 70)

print(
    json.dumps(
        request,
        indent=2
    )
)

print()


# ============================================================
# SEND
# ============================================================

try:

    ws.send(
        json.dumps(request)
    )

    print("REQUEST SENT")

except Exception as exc:

    print(
        f"REQUEST SEND ERROR: {exc}"
    )

    ws.close()

    raise SystemExit(1)


# ============================================================
# RECEIVE
# ============================================================

print()
print("WAITING FOR BITRUE RESPONSE...")
print("-" * 70)


ws.settimeout(10)


try:

    while True:

        message = ws.recv()

        print()
        print(
            f"MESSAGE TYPE: {type(message).__name__}"
        )

        try:

            data = decode_message(
                message
            )

            print(
                json.dumps(
                    data,
                    indent=2
                )
            )


            # ------------------------------------------------
            # HEARTBEAT
            # ------------------------------------------------

            if (
                isinstance(data, dict)
                and "ping" in data
            ):

                pong = {
                    "pong": data["ping"]
                }

                print()
                print(
                    "PING RECEIVED -> PONG"
                )

                ws.send(
                    json.dumps(pong)
                )

                continue


            # ------------------------------------------------
            # HISTORICAL RESPONSE
            # ------------------------------------------------

            if (
                isinstance(data, dict)
                and data.get(
                    "event_rep"
                ) == "rep"
            ):

                print()
                print("=" * 70)
                print("HISTORICAL REQUEST SUCCESS")
                print("=" * 70)

                rows = data.get(
                    "data",
                    []
                )

                print(
                    f"Rows received: {len(rows)}"
                )

                if rows:

                    print()
                    print(
                        "FIRST ROW:"
                    )

                    print(
                        json.dumps(
                            rows[0],
                            indent=2
                        )
                    )

                break


        except Exception as exc:

            print(
                f"DECODE ERROR: {exc}"
            )


except websocket.WebSocketConnectionClosedException as exc:

    print()
    print("=" * 70)
    print("BITRUE CLOSED THE CONNECTION")
    print("=" * 70)

    print(
        repr(exc)
    )

    print()
    print(
        "Historical request was NOT answered."
    )


except websocket.WebSocketTimeoutException:

    print()
    print("=" * 70)
    print("TIMEOUT")
    print("=" * 70)

    print(
        "No response within 10 seconds."
    )


except Exception as exc:

    print()
    print("=" * 70)
    print("UNEXPECTED ERROR")
    print("=" * 70)

    print(
        repr(exc)
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("FINAL SOCKET STATUS")
print("=" * 70)

try:

    print(
        f"Connected: {ws.connected}"
    )

except Exception:

    pass


try:

    ws.close()

except Exception:

    pass


print()
print("TEST COMPLETE")
print("=" * 70)
