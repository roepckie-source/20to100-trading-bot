# ==========================================
# 20to100 Trading Bot
# Historical Data Downloader
#
# BTC + ETH + SOL
# 5m
# 5 Jahre
# ==========================================

import time
from pathlib import Path

import ccxt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME = "5m"

# 5 Jahre
HISTORICAL_DAYS = 5 * 365

# Erwartete Größenordnung:
# 5 Jahre * 365 Tage * 24 Stunden * 12 Kerzen/Stunde
EXPECTED_CANDLES = HISTORICAL_DAYS * 24 * 12

# Wir akzeptieren mindestens 95 % der theoretischen Anzahl.
MIN_EXPECTED_CANDLES = int(EXPECTED_CANDLES * 0.95)

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
        "timeout": 30000,
    })

    exchange.load_markets()

    return exchange


# ============================================================
# FETCH HISTORICAL DATA
# ============================================================

def fetch_ohlcv(
    exchange,
    symbol,
    timeframe=TIMEFRAME,
    days=HISTORICAL_DAYS,
):

    timeframe_ms = (
        exchange.parse_timeframe(timeframe)
        * 1000
    )

    now = exchange.milliseconds()

    requested_since_ms = (
        now
        -
        days
        * 24
        * 60
        * 60
        * 1000
    )

    since = requested_since_ms

    candles = []

    # OKX / CCXT
    limit = 1000

    request_count = 0

    print()
    print("=" * 70)
    print("20→100 HISTORICAL DATA DOWNLOADER")
    print("=" * 70)

    print(
        f"Symbol:          {symbol}"
    )

    print(
        f"Timeframe:       {timeframe}"
    )

    print(
        f"Zeitraum:        {days} Tage"
    )

    print(
        f"Erwartete Kerzen: ~{EXPECTED_CANDLES:,}"
    )

    print(
        f"Minimum:          {MIN_EXPECTED_CANDLES:,}"
    )

    print(
        f"Von: "
        f"{pd.to_datetime(since, unit='ms', utc=True)}"
    )

    print(
        f"Bis: "
        f"{pd.to_datetime(now, unit='ms', utc=True)}"
    )

    print("=" * 70)

    # ========================================================
    # PAGINATION
    # ========================================================

    while since < now:

        request_count += 1

        current_time = pd.to_datetime(
            since,
            unit="ms",
            utc=True,
        )

        print(
            f"[{request_count:04d}] "
            f"{symbol} "
            f"ab {current_time}"
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
                f"❌ Download-Fehler: {e}"
            )

            print(
                "⏳ Warte 5 Sekunden..."
            )

            time.sleep(5)

            continue

        # ----------------------------------------------------
        # Keine Daten
        # ----------------------------------------------------

        if not batch:

            print(
                "⚠️ Keine weiteren Daten."
            )

            break

        # ----------------------------------------------------
        # Daten hinzufügen
        # ----------------------------------------------------

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
        # Pagination
        # ----------------------------------------------------

        next_since = (
            last_timestamp
            +
            timeframe_ms
        )

        if next_since <= since:

            print(
                "❌ Pagination konnte nicht "
                "fortgesetzt werden."
            )

            break

        since = next_since

        # ----------------------------------------------------
        # Rate Limit
        # ----------------------------------------------------

        time.sleep(
            max(
                exchange.rateLimit / 1000,
                0.05,
            )
        )

    # ========================================================
    # NO DATA
    # ========================================================

    if not candles:

        raise RuntimeError(
            f"Keine historischen Daten für "
            f"{symbol} erhalten."
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
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # PRICE VALIDATION
    # ========================================================

    df = df[
        (df["open"] > 0)
        &
        (df["high"] > 0)
        &
        (df["low"] > 0)
        &
        (df["close"] > 0)
    ]

    df = df[
        df["high"] >= df["low"]
    ]

    # ========================================================
    # TIME RANGE
    # ========================================================

    requested_since = pd.to_datetime(
        requested_since_ms,
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
        &
        (df["timestamp"] <= requested_until)
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
        df.index
        .to_series()
        .diff()
        .dropna()
    )

    gaps = deltas[
        deltas > expected_delta
    ]

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Symbol:       {symbol}"
    )

    print(
        f"Kerzen:       {len(df):,}"
    )

    print(
        f"Von:          {df.index.min()}"
    )

    print(
        f"Bis:          {df.index.max()}"
    )

    print(
        f"Requests:     {request_count}"
    )

    print(
        f"Datenlücken:  {len(gaps)}"
    )

    # ========================================================
    # CRITICAL VALIDATION
    # ========================================================

    if len(df) < MIN_EXPECTED_CANDLES:

        raise RuntimeError(
            f"ZU WENIG DATEN für {symbol}: "
            f"{len(df):,} statt mindestens "
            f"{MIN_EXPECTED_CANDLES:,}."
        )

    if len(gaps) > 0:

        print()
        print(
            f"⚠️ WARNUNG: "
            f"{len(gaps)} Datenlücken gefunden."
        )

    print("=" * 70)

    return df


# ============================================================
# SAVE
# ============================================================

def save_data(
    df,
    symbol,
):

    filename = (
        symbol.replace("/", "_")
        +
        "_"
        +
        TIMEFRAME
        +
        ".csv"
    )

    filepath = (
        DATA_DIR
        /
        filename
    )

    df.to_csv(
        filepath
    )

    print(
        f"💾 Gespeichert: {filepath}"
    )

    return filepath


# ============================================================
# DOWNLOAD ONE SYMBOL
# ============================================================

def download_symbol(
    exchange,
    symbol,
):

    print()
    print(
        "#" * 70
    )

    print(
        f"# DOWNLOAD {symbol}"
    )

    print(
        "#" * 70
    )

    df = fetch_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=TIMEFRAME,
        days=HISTORICAL_DAYS,
    )

    filepath = save_data(
        df,
        symbol,
    )

    return {
        "symbol": symbol,
        "rows": len(df),
        "filepath": str(filepath),
        "start": df.index.min(),
        "end": df.index.max(),
        "gaps": None,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("20→100 HISTORICAL DATA")
    print("BTC + ETH + SOL")
    print("5m / 5 YEARS")
    print("=" * 70)

    print()
    print(
        f"Expected candles per asset: "
        f"{EXPECTED_CANDLES:,}"
    )

    print(
        f"Minimum accepted: "
        f"{MIN_EXPECTED_CANDLES:,}"
    )

    # --------------------------------------------------------
    # Exchange
    # --------------------------------------------------------

    print()
    print(
        "Initialisiere OKX..."
    )

    exchange = create_exchange()

    print(
        "✅ OKX bereit."
    )

    results = []

    errors = []

    # ========================================================
    # ALL SYMBOLS
    # ========================================================

    for symbol in SYMBOLS:

        try:

            result = download_symbol(
                exchange,
                symbol,
            )

            results.append(
                result
            )

        except Exception as e:

            print()
            print(
                f"❌ FEHLER bei {symbol}"
            )

            print(
                repr(e)
            )

            errors.append(
                (
                    symbol,
                    str(e),
                )
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DATA DOWNLOAD SUMMARY")
    print("=" * 70)

    for result in results:

        print(
            f"{result['symbol']:10s} "
            f"{result['rows']:>10,} Kerzen | "
            f"{result['start']} → "
            f"{result['end']}"
        )

    # ========================================================
    # ERRORS
    # ========================================================

    if errors:

        print()
        print("=" * 70)
        print("FEHLER")
        print("=" * 70)

        for symbol, error in errors:

            print(
                f"❌ {symbol}: {error}"
            )

        raise RuntimeError(
            "Mindestens ein Asset konnte "
            "nicht vollständig geladen werden."
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if len(results) != len(SYMBOLS):

        raise RuntimeError(
            "Nicht alle Assets wurden geladen."
        )

    print()
    print("=" * 70)
    print("✅ ALLE DATEN ERFOLGREICH GELADEN")
    print("=" * 70)

    print()
    print(
        "Bereit für:"
    )

    print(
        "  V2"
    )

    print(
        "  V3"
    )

    print(
        "  V4"
    )

    print(
        "  V5"
    )

    print(
        "  V6 Extended"
    )

    print(
        "  V6_C Stress-Test"
    )

    print()
    print(
        "BTC + ETH + SOL"
    )

    print(
        "5m → 1h"
    )

    print(
        "12M TRAIN / 3M OOS"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
