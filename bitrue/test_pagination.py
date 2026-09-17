"""
Bitrue Futures Kline Pagination Test

PAPER / BACKTEST DATA ONLY
NO LIVE TRADING
NO API KEY REQUIRED

Tests possible historical pagination parameters
against Bitrue Futures public API.
"""

import requests
import pandas as pd
from datetime import datetime, timezone


BASE_URL = "https://fapi.bitrue.com"
ENDPOINT = "/fapi/v1/klines"

CONTRACT_NAME = "E-BTC-USDT"
INTERVAL = "5min"
LIMIT = 300


def request_klines(extra_params=None):
    params = {
        "contractName": CONTRACT_NAME,
        "interval": INTERVAL,
        "limit": LIMIT,
    }

    if extra_params:
        params.update(extra_params)

    response = requests.get(
        BASE_URL + ENDPOINT,
        params=params,
        timeout=30,
    )

    print()
    print("=" * 70)
    print("REQUEST")
    print("=" * 70)
    print("URL:", response.url)
    print("HTTP:", response.status_code)

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        print("Unexpected response:")
        print(data)
        return []

    print("Candles:", len(data))

    if data:
        first = data[0]
        last = data[-1]

        print()
        print("FIRST:")
        print(first)

        print()
        print("LAST:")
        print(last)

        print()
        print("FIRST IDX:")
        print(first.get("idx"))

        print("LAST IDX:")
        print(last.get("idx"))

    return data


def timestamp_to_string(ts):
    """
    Bitrue documentation describes idx as milliseconds.
    Some historical examples in the docs are inconsistent,
    so detect seconds vs milliseconds safely.
    """

    if ts is None:
        return "N/A"

    ts = int(ts)

    if ts < 10_000_000_000:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def analyse(name, data):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    if not data:
        print("NO DATA")
        return

    timestamps = [int(x["idx"]) for x in data]

    print("Candles:", len(data))
    print("Newest:", timestamp_to_string(timestamps[0]))
    print("Oldest:", timestamp_to_string(timestamps[-1]))

    unique = len(set(timestamps))

    print("Unique timestamps:", unique)

    if unique != len(timestamps):
        print("WARNING: duplicate timestamps")

    print("Order:")
    if timestamps[0] > timestamps[-1]:
        print("Newest -> oldest")
    else:
        print("Oldest -> newest")


def main():

    print()
    print("=" * 70)
    print("BITRUE HISTORICAL PAGINATION INVESTIGATION")
    print("=" * 70)
    print("Contract :", CONTRACT_NAME)
    print("Interval :", INTERVAL)
    print("Limit    :", LIMIT)

    # ---------------------------------------------------------
    # TEST 1
    # Normal request
    # ---------------------------------------------------------

    data_normal = request_klines()

    analyse(
        "TEST 1 - NORMAL REQUEST",
        data_normal,
    )

    if not data_normal:
        return

    newest_idx = int(data_normal[0]["idx"])
    oldest_idx = int(data_normal[-1]["idx"])

    print()
    print("Newest IDX:", newest_idx)
    print("Oldest IDX:", oldest_idx)

    # ---------------------------------------------------------
    # TEST 2
    # startTime
    # ---------------------------------------------------------

    data_start = request_klines(
        {
            "startTime": oldest_idx - (300 * 5 * 60 * 1000)
        }
    )

    analyse(
        "TEST 2 - startTime",
        data_start,
    )

    # ---------------------------------------------------------
    # TEST 3
    # endTime
    # ---------------------------------------------------------

    data_end = request_klines(
        {
            "endTime": oldest_idx - 1
        }
    )

    analyse(
        "TEST 3 - endTime",
        data_end,
    )

    # ---------------------------------------------------------
    # TEST 4
    # startTime + endTime
    # ---------------------------------------------------------

    data_range = request_klines(
        {
            "startTime": oldest_idx - (300 * 5 * 60 * 1000),
            "endTime": oldest_idx - 1,
        }
    )

    analyse(
        "TEST 4 - startTime + endTime",
        data_range,
    )

    # ---------------------------------------------------------
    # TEST 5
    # fromId
    # ---------------------------------------------------------

    data_from_id = request_klines(
        {
            "fromId": oldest_idx - 1,
        }
    )

    analyse(
        "TEST 5 - fromId",
        data_from_id,
    )

    # ---------------------------------------------------------
    # TEST 6
    # from
    # ---------------------------------------------------------

    data_from = request_klines(
        {
            "from": oldest_idx - 1,
        }
    )

    analyse(
        "TEST 6 - from",
        data_from,
    )

    # ---------------------------------------------------------
    # TEST 7
    # to
    # ---------------------------------------------------------

    data_to = request_klines(
        {
            "to": oldest_idx - 1,
        }
    )

    analyse(
        "TEST 7 - to",
        data_to,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("PAGINATION TEST COMPLETE")
    print("=" * 70)

    print()
    print("We need a request that returns candles older than:")
    print(timestamp_to_string(oldest_idx))

    print()
    print("If all tests return the same 300 candles,")
    print("Bitrue's public Futures Kline endpoint does not")
    print("provide usable historical pagination through these parameters.")

    print()
    print("NO LIVE TRADING")
    print("NO API KEY")
    print("NO WALLET")
    print("NO REAL ORDERS")


if __name__ == "__main__":
    main()
