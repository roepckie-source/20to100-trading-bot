# ==========================================
# 20to100 Trading Bot
# Historical Data Manager
# ==========================================

import time
from pathlib import Path

import ccxt
import pandas as pd


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def create_exchange():
    """
    Create a public OKX exchange connection.

    No API key is required for historical OHLCV data.
    """

    return ccxt.okx({
        "enableRateLimit": True,
    })


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    days: int = 30,
) -> pd.DataFrame:
    """
    Download historical OHLCV data from OKX.

    Parameters
    ----------
    symbol:
        Trading pair, e.g. BTC/USDT

    timeframe:
        Candle timeframe, e.g. 5m

    days:
        Number of historical days to request.
    """

    exchange = create_exchange()

    timeframe_ms = exchange.parse_timeframe(
        timeframe
    ) * 1000

    now = exchange.milliseconds()

    since = (
        now -
        days * 24 * 60 * 60 * 1000
    )

    all_candles = []

    limit = 100

    print()
    print("=" * 60)
    print("20→100 HISTORICAL DATA")
    print("=" * 60)
    print(f"Exchange:  OKX")
    print(f"Symbol:    {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Days:      {days}")
    print("=" * 60)

    while since < now:

        print(
            f"Downloading from "
            f"{pd.to_datetime(since, unit='ms')} ..."
        )

        try:

            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

        except Exception as exc:

            print(
                f"ERROR while downloading data: {exc}"
            )

            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)

        last_timestamp = candles[-1][0]

        # Make sure the loop always moves forward.
        next_since = (
            last_timestamp +
            timeframe_ms
        )

        if next_since <= since:
            break

        since = next_since

        time.sleep(
            exchange.rateLimit / 1000
        )

    if not all_candles:
        raise RuntimeError(
            "No historical data received."
        )

    # --------------------------------------
    # Create DataFrame
    # --------------------------------------

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

    # --------------------------------------
    # Convert timestamp
    # --------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    # --------------------------------------
    # Remove duplicates
    # --------------------------------------

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    # --------------------------------------
    # Sort chronologically
    # --------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------
    # Set timestamp as index
    # --------------------------------------

    df = df.set_index(
        "timestamp"
    )

    # --------------------------------------
    # Numeric columns
    # --------------------------------------

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

    # --------------------------------------
    # Remove invalid rows
    # --------------------------------------

    df = df.dropna(
        subset=numeric_columns
    )

    print()
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
    df: pd.DataFrame,
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
) -> Path:
    """
    Save OHLCV data locally as CSV.
    """

    filename = (
        symbol.replace("/", "_")
        + "_"
        + timeframe
        + ".csv"
    )

    filepath = DATA_DIR / filename

    df.to_csv(filepath)

    print(
        f"Data saved to: {filepath}"
    )

    return filepath


def load_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
) -> pd.DataFrame:
    """
    Load previously downloaded data.
    """

    filename = (
        symbol.replace("/", "_")
        + "_"
        + timeframe
        + ".csv"
    )

    filepath = DATA_DIR / filename

    if not filepath.exists():

        raise FileNotFoundError(
            f"Data file not found: {filepath}"
        )

    df = pd.read_csv(
        filepath,
        index_col="timestamp",
        parse_dates=True,
    )

    return df


if __name__ == "__main__":

    data = fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="5m",
        days=30,
    )

    save_data(
        data,
        symbol="BTC/USDT",
        timeframe="5m",
    )
