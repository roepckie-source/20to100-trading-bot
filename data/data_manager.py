# ==========================================
# 20to100 Trading Bot
# Historical Data Manager V2
# ==========================================

import time
from pathlib import Path

import ccxt
import pandas as pd


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TIMEFRAME = "5m"

# Wir laden zunächst 5 Jahre.
# 2021-01-01 bis heute ist für BTC/ETH sinnvoller,
# falls die Börse die komplette Historie anbietet.
DEFAULT_DAYS = 5 * 365

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
]


# ============================================================
# EXCHANGE
# ============================================================

def create_exchange():

    exchange = ccxt.okx({
        "enableRateLimit": True,
    })

    return exchange


# ============================================================
# FETCH HISTORICAL DATA
# ============================================================

def fetch_ohlcv(
    symbol="BTC/USDT",
    timeframe=DEFAULT_TIMEFRAME,
    days=DEFAULT_DAYS,
):

    exchange = create_exchange()

    timeframe_ms = (
        exchange.parse_timeframe(timeframe)
        * 1000
    )

    now = exchange.milliseconds()

    since = (
        now
        - days
        * 24
        * 60
        * 60
        * 1000
    )

    candles = []

    # OKX unterstützt größere Blöcke als 100.
    # Wir bleiben bewusst bei 1000 für stabile Pagination.
    limit = 1000

    print()
    print("=" * 70)
    print("20→100 HISTORICAL DATA DOWNLOADER")
    print("=" * 70)

    print(
        f"Symbol:    {symbol}"
    )

    print(
        f"Timeframe: {timeframe}"
    )

    print(
        f"Zeitraum:  {days} Tage"
    )

    print(
        f"Von:       "
        f"{pd.to_datetime(since, unit='ms', utc=True)}"
    )

    print(
        f"Bis:       "
        f"{pd.to_datetime(now, unit='ms', utc=True)}"
    )

    print("=" * 70)

    request_count = 0

    while since < now:

        request_count += 1

        print(
            f"[{request_count:04d}] "
            f"Downloading from "
            f"{pd.to_datetime(since, unit='ms', utc=True)}"
        )

        try:

            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

        except Exception as e:

            print(
                f"❌ Fehler beim Download: {e}"
            )

            print(
                "⏳ Warte 5 Sekunden und versuche erneut..."
            )

            time.sleep(5)

            continue

        if not batch:

            print(
                "⚠️ Keine weiteren Daten erhalten."
            )

            break

        candles.extend(batch)

        first_timestamp = batch[0][0]
        last_timestamp = batch[-1][0]

        print(
            f"    → {len(batch)} Kerzen"
        )

        print(
            f"    → bis "
            f"{pd.to_datetime(last_timestamp, unit='ms', utc=True)}"
        )

        # ----------------------------------------------------
        # Pagination-Schutz
        # ----------------------------------------------------

        next_since = (
            last_timestamp
            + timeframe_ms
        )

        if next_since <= since:

            print(
                "⚠️ Pagination konnte nicht fortgesetzt werden."
            )

            break

        since = next_since

        # ----------------------------------------------------
        # Rate limit
        # ----------------------------------------------------

        time.sleep(
            max(
                exchange.rateLimit / 1000,
                0.05,
            )
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    if not candles:

        raise RuntimeError(
            "Keine historischen Daten erhalten."
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

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

    # ========================================================
    # TIMESTAMP
    # ========================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
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

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ========================================================
    # CLEAN
    # ========================================================

    df = (
        df
        .dropna(
            subset=numeric_columns
        )
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ========================================================
    # PRICE VALIDATION
    # ========================================================

    # Keine negativen/Null-Preise
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    ]

    # High muss >= Low sein
    df = df[
        df["high"] >= df["low"]
    ]

    # ========================================================
    # TIME RANGE FILTER
    # ========================================================

    # Falls die Börse etwas außerhalb des gewünschten
    # Zeitraums geliefert hat.
    requested_since = pd.to_datetime(
        now
        - days
        * 24
        * 60
        * 60
        * 1000,
        unit="ms",
        utc=True,
    )

    requested_until = pd.to_datetime(
        now,
        unit="ms",
        utc=True,
    )

    df = df[
        (df["timestamp"] >= requested_since)
        & (df["timestamp"] <= requested_until)
    ]

    # ========================================================
    # INDEX
    # ========================================================

    df = df.set_index(
        "timestamp"
    )

    # ========================================================
    # GAP ANALYSIS
    # ========================================================

    expected_delta = pd.Timedelta(
        milliseconds=timeframe_ms
    )

    deltas = (
        df.index.to_series()
        .diff()
        .dropna()
    )

    gaps = deltas[
        deltas > expected_delta
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Kerzen:     {len(df):,}"
    )

    print(
        f"Von:        {df.index.min()}"
    )

    print(
        f"Bis:        {df.index.max()}"
    )

    print(
        f"Requests:   {request_count}"
    )

    print(
        f"Timeframe:  {timeframe}"
    )

    print(
        f"Lücken:     {len(gaps)}"
    )

    if len(gaps) > 0:

        print()
        print(
            "⚠️ WARNUNG: "
            f"{len(gaps)} Datenlücken gefunden."
        )

        print(
            "Die Daten werden trotzdem gespeichert."
        )

    print("=" * 70)

    return df


# ============================================================
# SAVE DATA
# ============================================================

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
        DATA_DIR
        / filename
    )

    df.to_csv(
        filepath
    )

    print(
        f"💾 Data saved to: {filepath}"
    )

    return filepath


# ============================================================
# LOAD DATA
# ============================================================

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
        DATA_DIR
        / filename
    )

    if not filepath.exists():

        raise FileNotFoundError(
            f"Datei nicht gefunden: {filepath}"
        )

    df = pd.read_csv(
        filepath,
        index_col="timestamp",
        parse_dates=True,
    )

    return df


# ============================================================
# DOWNLOAD ALL
# ============================================================

def download_all(
    days=DEFAULT_DAYS,
    timeframe=DEFAULT_TIMEFRAME,
):

    results = {}

    for symbol in SYMBOLS:

        try:

            print()
            print(
                f"🚀 Starte Download: {symbol}"
            )

            df = fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
            )

            filepath = save_data(
                df,
                symbol,
                timeframe,
            )

            results[symbol] = {
                "rows": len(df),
                "filepath": str(filepath),
                "start": df.index.min(),
                "end": df.index.max(),
            }

        except Exception as e:

            print()
            print(
                f"❌ {symbol} fehlgeschlagen:"
            )

            print(e)

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "🚀 20→100 HISTORICAL DATA"
    )

    results = download_all()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for symbol, info in results.items():

        print(
            f"{symbol}: "
            f"{info['rows']:,} Kerzen | "
            f"{info['start']} → {info['end']}"
        )

    print("=" * 70)
