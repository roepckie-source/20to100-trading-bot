# ============================================================
# 20to100 Trading Bot
# PAPER MARKET DATA
# ============================================================

from __future__ import annotations

import ccxt
import pandas as pd


# ============================================================
# EXCHANGE
# ============================================================

def create_exchange():

    exchange = ccxt.okx({
        "enableRateLimit": True,
    })

    exchange.load_markets()

    return exchange


# ============================================================
# FETCH 5m
# ============================================================

def fetch_5m(
    exchange,
    symbol: str,
    limit: int = 1000,
) -> pd.DataFrame:

    candles = exchange.fetch_ohlcv(
        symbol,
        timeframe="5m",
        limit=limit,
    )

    if not candles:

        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df = df.set_index(
        "timestamp"
    )

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

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df.sort_index()


# ============================================================
# 5m -> 1h
# ============================================================

def resample_5m_to_1h(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:

        return df.copy()

    hourly = (
        df.resample(
            "1h"
        )
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

    return hourly.sort_index()
