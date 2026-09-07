# ============================================================
# 20to100 Trading Bot
# V7-S0 PAPER MARKET DATA
#
# IMPORTANT:
# - PUBLIC MARKET DATA ONLY
# - NO PRIVATE API
# - NO REAL ORDERS
# - 5m data -> 1h signal data
# - Fetches enough history for V7-S0 indicators
# ============================================================

from __future__ import annotations

import ccxt
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "5m"

TIMEFRAME_MINUTES = 5
TIMEFRAME_MS = TIMEFRAME_MINUTES * 60 * 1000

# V7-S0 needs at least 250 hourly candles.
# 5000 x 5m = approximately 416 hourly candles.
DEFAULT_LIMIT = 5000

# Most exchanges limit one OHLCV request to around 1000 candles.
MAX_PER_REQUEST = 1000


# ============================================================
# EXCHANGE
# ============================================================

def create_exchange():

    exchange = ccxt.okx({
        "enableRateLimit": True,
    })

    # Public market data only.
    #
    # No API key.
    # No secret.
    # No private trading permissions.

    exchange.load_markets()

    return exchange


# ============================================================
# FETCH 5m
# ============================================================

def fetch_5m(
    exchange,
    symbol: str,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """
    Fetch approximately `limit` 5m candles.

    CCXT/exchanges often limit one request to ~1000 candles,
    therefore the requested history is fetched in chunks.

    This function ONLY requests public OHLCV data.
    It does NOT place orders.
    """

    limit = max(
        int(limit),
        MAX_PER_REQUEST,
    )

    now_ms = exchange.milliseconds()

    start_ms = (
        now_ms
        - (
            limit
            * TIMEFRAME_MS
        )
    )

    all_candles = []

    remaining = limit

    print(
        f"[{symbol}] "
        f"Fetching {limit} x 5m candles..."
    )

    while remaining > 0:

        request_limit = min(
            MAX_PER_REQUEST,
            remaining,
        )

        try:

            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=start_ms,
                limit=request_limit,
            )

        except Exception as exc:

            print(
                f"[{symbol}] "
                f"OHLCV request failed: {exc}"
            )

            break

        if not candles:
            break

        all_candles.extend(
            candles
        )

        first_timestamp = candles[0][0]
        last_timestamp = candles[-1][0]

        next_start = (
            last_timestamp
            + TIMEFRAME_MS
        )

        # Safety against an exchange returning
        # the same candles repeatedly.
        if next_start <= start_ms:
            break

        start_ms = next_start

        remaining -= len(candles)

        # If the exchange returned fewer candles
        # than requested, there may simply be no
        # more historical data in that range.
        if len(candles) < request_limit:
            break

    # --------------------------------------------------------
    # No data
    # --------------------------------------------------------

    if not all_candles:

        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df = df.set_index(
        "timestamp"
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_index()

    # --------------------------------------------------------
    # Keep requested amount
    # --------------------------------------------------------

    if len(df) > limit:

        df = df.iloc[-limit:]

    print(
        f"[{symbol}] "
        f"Received {len(df)} x 5m candles"
    )

    if not df.empty:

        print(
            f"[{symbol}] "
            f"Data range: "
            f"{df.index[0]} -> "
            f"{df.index[-1]}"
        )

    return df


# ============================================================
# 5m -> 1h
# ============================================================

def resample_5m_to_1h(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert 5m OHLCV data into 1h OHLCV.

    The final hourly candle may still be forming.
    The PaperTrader deliberately uses:
        - completed 1h candles for signals
        - current 1h candle only for current open/entry
    """

    if df.empty:

        return df.copy()

    hourly = (
        df.resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    hourly = hourly.sort_index()

    return hourly