# ============================================================
# 20to100 Trading Bot
# PAPER TRADING MARKET DATA
# V7-S0 / V6-C
#
# IMPORTANT:
# - PUBLIC MARKET DATA ONLY
# - NO API KEYS
# - NO PRIVATE API
# - NO REAL ORDERS
# ============================================================

from __future__ import annotations

import time
from datetime import datetime, timezone

import ccxt
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "5m"

TIMEFRAME_MS = 5 * 60 * 1000

# OKX limits the number of candles returned per request.
MAX_PER_REQUEST = 300


# ============================================================
# CREATE EXCHANGE
# ============================================================

def create_exchange():
    """
    Creates a public OKX CCXT connection.

    No API keys are used.
    No private endpoints are used.
    No orders can be placed through this connection.
    """

    exchange = ccxt.okx({
        "enableRateLimit": True,
    })

    return exchange


# ============================================================
# FETCH 5M DATA
# ============================================================

def fetch_5m(
    exchange,
    symbol,
    limit=5000,
):
    """
    Fetch historical 5m OHLCV data from OKX.

    Uses pagination because OKX limits the amount
    of candles returned per request.
    """

    limit = int(limit)

    if limit <= 0:
        raise ValueError(
            "OHLCV limit muss größer als 0 sein."
        )

    print(
        f"[{symbol}] "
        f"Fetching {limit} x {TIMEFRAME} candles..."
    )

    # ========================================================
    # START TIME
    # ========================================================

    now_ms = int(
        datetime.now(
            timezone.utc
        ).timestamp() * 1000
    )

    since_ms = (
        now_ms
        - limit * TIMEFRAME_MS
    )

    all_candles = []

    # ========================================================
    # PAGINATION
    # ========================================================

    while len(all_candles) < limit:

        remaining = (
            limit - len(all_candles)
        )

        request_limit = min(
            MAX_PER_REQUEST,
            remaining,
        )

        try:

            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=since_ms,
                limit=request_limit,
            )

        except Exception as exc:

            print(
                f"[{symbol}] "
                f"OHLCV Fehler: {exc}"
            )

            raise

        if not batch:

            print(
                f"[{symbol}] "
                "Keine weiteren Candles erhalten."
            )

            break

        print(
            f"[{symbol}] "
            f"API batch: {len(batch)} candles"
        )

        all_candles.extend(batch)

        # ====================================================
        # NEXT PAGE
        # ====================================================

        last_timestamp = int(
            batch[-1][0]
        )

        next_since = (
            last_timestamp
            + TIMEFRAME_MS
        )

        if next_since <= since_ms:

            print(
                f"[{symbol}] "
                "Pagination ohne Fortschritt. "
                "Abbruch."
            )

            break

        since_ms = next_since

        # ====================================================
        # RATE LIMIT
        # ====================================================

        rate_limit_ms = getattr(
            exchange,
            "rateLimit",
            100,
        )

        time.sleep(
            max(
                rate_limit_ms / 1000,
                0.05,
            )
        )

    # ========================================================
    # NO DATA
    # ========================================================

    if not all_candles:

        raise RuntimeError(
            f"[{symbol}] "
            "Keine OHLCV-Daten erhalten."
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

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

    # ========================================================
    # CLEAN DATA
    # ========================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # LIMIT
    # ========================================================

    if len(df) > limit:

        df = (
            df
            .tail(limit)
            .reset_index(drop=True)
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        f"[{symbol}] "
        f"Received {len(df)} x {TIMEFRAME} candles"
    )

    if not df.empty:

        print(
            f"[{symbol}] "
            f"Data range: "
            f"{df['timestamp'].iloc[0]} -> "
            f"{df['timestamp'].iloc[-1]}"
        )

    return df


# ============================================================
# RESAMPLE 5M -> 1H
# ============================================================

def resample_5m_to_1h(
    df_5m,
):
    """
    Converts 5m OHLCV data to 1h OHLCV data.

    The last hourly candle may still be forming.
    The PaperTrader is responsible for treating it
    correctly.
    """

    if df_5m is None:

        raise ValueError(
            "df_5m ist None."
        )

    if df_5m.empty:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df_5m.columns
    ]

    if missing:

        raise ValueError(
            "Fehlende Spalten: "
            + ", ".join(missing)
        )

    data = df_5m.copy()

    # ========================================================
    # TIMESTAMP
    # ========================================================

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    )

    data = (
        data
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"]
        )
    )

    # ========================================================
    # NUMERIC
    # ========================================================

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    # ========================================================
    # INDEX
    # ========================================================

    data = data.set_index(
        "timestamp"
    )

    # ========================================================
    # RESAMPLE
    # ========================================================

    hourly = (
        data
        .resample("1h")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
    )

    # ========================================================
    # REMOVE EMPTY HOURS
    # ========================================================

    hourly = hourly.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    hourly = (
        hourly
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return hourly
