# ============================================================
# V7-S0 PAPER TRADER
# MARKET DATA
# ============================================================

from __future__ import annotations

import ccxt
import pandas as pd


# ============================================================
# CREATE EXCHANGE
# ============================================================

def create_exchange(
    exchange_id: str
):

    exchange_class = getattr(
        ccxt,
        exchange_id,
        None
    )

    if exchange_class is None:

        raise ValueError(
            f"Unknown CCXT exchange: "
            f"{exchange_id}"
        )

    exchange = exchange_class(
        {
            "enableRateLimit": True
        }
    )

    exchange.load_markets()

    return exchange


# ============================================================
# FETCH 5m DATA
# ============================================================

def fetch_5m(
    exchange,
    symbol: str,
    limit: int = 1000,
) -> pd.DataFrame:

    rows = exchange.fetch_ohlcv(
        symbol,
        timeframe="5m",
        limit=limit,
    )

    if not rows:

        raise RuntimeError(
            f"No OHLCV returned for "
            f"{symbol}"
        )

    df = pd.DataFrame(
        rows,
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

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .drop_duplicates(
            "timestamp"
        )
        .sort_values(
            "timestamp"
        )
        .set_index(
            "timestamp"
        )
    )

    return df


# ============================================================
# RESAMPLE 5m -> 1h
# ============================================================

def resample_5m_to_1h(
    df_5m: pd.DataFrame,
) -> pd.DataFrame:

    hourly = (
        df_5m
        .resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
    )

    return hourly
