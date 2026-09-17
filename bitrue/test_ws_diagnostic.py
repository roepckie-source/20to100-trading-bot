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
print("BITRUE WEBSOCKET DIAGNOSTIC")
print("=" * 70)

print()
print(f"URL:     {WS_URL}")
print(f"CHANNEL: {CHANNEL}")
print()


# ============================================================
# CONNECT
# ============================================================

print("1. CONNECT TEST")
print("-" * 70)

try:

    ws = websocket.create_connection(
        WS_URL,
        timeout=10,
        suppress_origin=True
    )

    print("CONNECTED")

except Exception as exc:

    print(f"CONNECT FAILED: {exc}")
    raise SystemExit(1)


# ============================================================
# WAIT FOR SERVER HEARTBEAT / DATA
# ============================================================

print()
print("2. WAITING FOR SERVER MESSAGE")
print("-" * 70)

ws.settimeout(5)

try:

    message = ws.recv()

    print(
        f"Received message type: {type(message).__name__}"
    )

    try:

        data = decode_message(message)

        print(
            "Decoded message:"
        )

        print(
            json.dumps(
                data,
                indent=2
            )
        )

        if (
            isinstance(data, dict)
            and "ping" in data
        ):

            pong = {
                "pong": data["ping"]
            }

            print()
            print(
                "Sending PONG:"
            )

            print(
                json.dumps(pong)
            )

            ws.send(
                json.dumps(pong)
            )

    except Exception as exc:

        print(
            f"Decode error: {exc}"
        )

except Exception as exc:

    print(
        f"No normal message received: {exc}"
    )


# ============================================================
# LIVE KLINE SUBSCRIPTION
# ============================================================

print()
print("3. LIVE KLINE SUBSCRIPTION TEST")
print("-" * 70)

subscription = {
    "event": "sub",
    "params": {
        "channel": CHANNEL,
        "cb_id": "diagnostic"
    }
}

print(
    "Sending:"
)

print(
    json.dumps(
        subscription
    )
)

try:

    ws.send(
        json.dumps(
            subscription
        )
    )

except Exception as exc:

    print(
        f"SUBSCRIBE SEND FAILED: {exc}"
    )

    try:
        ws.close()
    except Exception:
        pass

    raise SystemExit(1)


# ------------------------------------------------------------
# WAIT FOR SUBSCRIPTION RESPONSE
# ------------------------------------------------------------

ws.settimeout(5)

try:

    message = ws.recv()

    print()
    print(
        "Subscription response:"
    )

    try:

        data = decode_message(message)

        print(
            json.dumps(
                data,
                indent=2
            )
        )

    except Exception as exc:

        print(
            f"Could not decode: {exc}"
        )


except Exception as exc:

    print()
    print(
        f"SUBSCRIPTION FAILED: {exc}"
    )


# ============================================================
# HISTORICAL REQUEST
# ============================================================

print()
print("4. HISTORICAL KLINE REQUEST TEST")
print("-" * 70)


# Use an aligned 5-minute timestamp.
now = int(time.time())

end_idx = (
    now // 300
) * 300


historical_request = {
    "event": "req",
    "params": {
        "channel": CHANNEL,
        "cb_id": "diagnostic-history",
        "endIdx": str(end_idx),
        "pageSize": 10
    }
}


print(
    "Sending:"
)

print(
    json.dumps(
        historical_request,
        indent=2
    )
)


try:

    ws.send(
        json.dumps(
            historical_request
        )
    )

except Exception as exc:

    print(
        f"HISTORICAL REQUEST SEND FAILED: {exc}"
    )

    try:
        ws.close()
    except Exception:
        pass

    raise SystemExit(1)


# ------------------------------------------------------------
# WAIT FOR RESPONSE
# ------------------------------------------------------------

ws.settimeout(10)

try:

    while True:

        message = ws.recv()

        print()
        print(
            "Historical response received."
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

            # If this is the requested historical response,
            # we are finished.

            if (
                isinstance(data, dict)
                and data.get(
                    "event_rep"
                ) == "rep"
            ):

                break

            # Handle heartbeat.

            if (
                isinstance(data, dict)
                and "ping" in data
            ):

                pong = {
                    "pong": data["ping"]
                }

                ws.send(
                    json.dumps(
                        pong
                    )
                )

        except Exception as exc:

            print(
                f"Decode error: {exc}"
            )


except Exception as exc:

    print()
    print(
        "HISTORICAL REQUEST FAILED:"
    )

    print(
        repr(exc)
    )


# ============================================================
# CLOSE INFORMATION
# ============================================================

print()
print("5. CONNECTION STATUS")
print("-" * 70)


try:

    print(
        f"Socket connected: {ws.connected}"
    )

    if ws.sock:

        print(
            f"Socket fileno: {ws.sock.fileno()}"
        )

except Exception as exc:

    print(
        f"Could not inspect socket: {exc}"
    )


try:

    ws.close()

except Exception:

    pass


print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
