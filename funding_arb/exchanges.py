import time
import requests

from .models import MarketSnapshot


TIMEOUT = 10


def get(url, params):
    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# BITRUE
# ============================================================

def bitrue():

    snapshot = MarketSnapshot(
        "bitrue",
        timestamp_ms=int(time.time() * 1000),
    )

    try:

        spot = get(
            "https://www.bitrue.com/api/v1/ticker/bookTicker",
            {
                "symbol": "BTCUSDT"
            },
        )

        futures = get(
            "https://fapi.bitrue.com/fapi/v1/ticker",
            {
                "symbol": "BTCUSDT"
            },
        )

        snapshot.spot_bid = float(
            spot["bidPrice"]
        )

        snapshot.spot_ask = float(
            spot["askPrice"]
        )

        snapshot.perp_bid = float(
            futures.get(
                "bidPrice",
                futures.get(
                    "bid1Price",
                    futures.get("bid"),
                ),
            )
        )

        snapshot.perp_ask = float(
            futures.get(
                "askPrice",
                futures.get(
                    "ask1Price",
                    futures.get("ask"),
                ),
            )
        )

        if futures.get("fundingRate") is not None:

            snapshot.funding_rate = float(
                futures["fundingRate"]
            )

        if futures.get("nextFundingTime") is not None:

            snapshot.next_funding_ms = int(
                futures["nextFundingTime"]
            )

    except Exception as exc:

        snapshot.error = str(exc)

    return snapshot


# ============================================================
# OKX
# ============================================================

def okx():

    snapshot = MarketSnapshot(
        "okx",
        timestamp_ms=int(time.time() * 1000),
    )

    try:

        base = "https://eea.okx.com"

        spot = get(
            base + "/api/v5/market/ticker",
            {
                "instId": "BTC-USDT"
            },
        )["data"][0]

        perp = get(
            base + "/api/v5/market/ticker",
            {
                "instId": "BTC-USDT-SWAP"
            },
        )["data"][0]

        funding = get(
            base + "/api/v5/public/funding-rate",
            {
                "instId": "BTC-USDT-SWAP"
            },
        )["data"][0]

        snapshot.spot_bid = float(
            spot["bidPx"]
        )

        snapshot.spot_ask = float(
            spot["askPx"]
        )

        snapshot.perp_bid = float(
            perp["bidPx"]
        )

        snapshot.perp_ask = float(
            perp["askPx"]
        )

        snapshot.funding_rate = float(
            funding["fundingRate"]
        )

        snapshot.next_funding_ms = int(
            funding["nextFundingTime"]
        )

        if (
            funding.get("fundingTime")
            and funding.get("nextFundingTime")
        ):

            snapshot.funding_interval_hours = (
                int(funding["nextFundingTime"])
                - int(funding["fundingTime"])
            ) / 3600000

    except Exception as exc:

        snapshot.error = str(exc)

    return snapshot


# ============================================================
# MEXC
# ============================================================

def mexc():

    snapshot = MarketSnapshot(
        "mexc",
        timestamp_ms=int(time.time() * 1000),
    )

    try:

        spot = get(
            "https://api.mexc.com/api/v3/ticker/bookTicker",
            {
                "symbol": "BTCUSDT"
            },
        )

        futures = get(
            "https://api.mexc.com/api/v1/contract/ticker",
            {
                "symbol": "BTC_USDT"
            },
        )["data"]

        snapshot.spot_bid = float(
            spot["bidPrice"]
        )

        snapshot.spot_ask = float(
            spot["askPrice"]
        )

        snapshot.perp_bid = float(
            futures.get(
                "bid1",
                futures.get("bid1Price"),
            )
        )

        snapshot.perp_ask = float(
            futures.get(
                "ask1",
                futures.get("ask1Price"),
            )
        )

        snapshot.funding_rate = float(
            futures["fundingRate"]
        )

        snapshot.next_funding_ms = int(
            futures["nextSettleTime"]
        )

        if futures.get("collectCycle") is not None:

            snapshot.funding_interval_hours = float(
                futures["collectCycle"]
            )

    except Exception as exc:

        snapshot.error = str(exc)

    return snapshot


# ============================================================
# EXCHANGE REGISTRY
# ============================================================

FETCHERS = {

    "bitrue": bitrue,
    "okx": okx,
    "mexc": mexc,

}
