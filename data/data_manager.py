# ==========================================
# 20to100 Trading Bot
# Historical Data Manager
# ==========================================

import time
from pathlib import Path

import ccxt
import pandas as pd


DATA_DIR = Path("data")
DATA_DIR.mkdir(
    exist_ok=True
)


def create_exchange():

    return ccxt.okx({
        "enableRateLimit": True,
    })


def fetch_ohlcv(
    symbol="BTC/USDT",
    timeframe="5m",
    days=30,
):

    exchange = create_exchange()

    timeframe_ms = (
        exchange.parse_timeframe(
            timeframe
        ) * 1000
    )

    now = exchange.milliseconds()

    since = (
        now -
        days *
        24 *
        60 *
        60 *
        1000
    )

    candles = []

    limit = 100

    print("=" * 60)
    print("20→100 HISTORICAL DATA")
    print("=" * 60)

    while since < now:

        print(
            "Downloading:",
            pd.to_datetime(
                since,
                unit="ms",
            ),
        )

        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )

        if not batch:
            break

        candles.extend(batch)

        last_timestamp = batch[-1][0]

        next_since = (
            last_timestamp +
            timeframe_ms
        )

        if next_since <= since:
            break

        since = next_since

        time.sleep(
            exchange.rateLimit /
            1000
        )

    if not candles:

        raise RuntimeError(
            "No historical data received."
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

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df = df.set_index(
        "timestamp"
    )

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric
    )

    print(
        f"Downloaded candles: {len(df)}"
    )

    print(
        f"From: {df.index.min()}"
    )

    print(
        f"To:   {df.index.max()}"
    )

    return df


def save_data(
    df,
    symbol,
    timeframe,
):

    filename = (
        symbol.replace("/", "_")
        + "_"
        + timeframe
        + ".csv"
    )

    filepath = (
        DATA_DIR /
        filename
    )

    df.to_csv(filepath)

    print(
        f"Data saved to: {filepath}"
    )

    return filepath


def load_data(
    symbol,
    timeframe,
):

    filename = (
        symbol.replace("/", "_")
        + "_"
        + timeframe
        + ".csv"
    )

    filepath = (
        DATA_DIR /
        filename
    )

    if not filepath.exists():

        raise FileNotFoundError(
            str(filepath)
        )

    return pd.read_csv(
        filepath,
        index_col="timestamp",
        parse_dates=True,
    )
