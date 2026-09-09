# ============================================================
# 20to100 Trading Bot
# PAPER TRADING MARKET DATA
# V7-S0 / V6-C
# ============================================================

import time
from datetime import datetime, timezone

import ccxt
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "5m"

TIMEFRAME_MS = 5 * 60 * 1000

DEFAULT_LIMIT = 5000

# OKX liefert maximal ca. 300 Candles pro Request
MAX_PER_REQUEST = 300


# ============================================================
# MARKET DATA LOADER
# ============================================================

class PaperMarketData:

    def __init__(self, exchange_name="okx"):

        exchange_class = getattr(ccxt, exchange_name)

        self.exchange = exchange_class({
            "enableRateLimit": True,
        })

        print(f"Exchange initialisiert: {exchange_name}")


    # ========================================================
    # FETCH 5M DATA
    # ========================================================

    def fetch_5m(self, symbol, limit=DEFAULT_LIMIT):

        limit = int(limit)

        if limit <= 0:
            raise ValueError("limit muss größer als 0 sein")

        print(
            f"[{symbol}] Fetching {limit} x {TIMEFRAME} candles..."
        )

        # ----------------------------------------------------
        # Startzeit berechnen
        # ----------------------------------------------------

        now_ms = int(
            datetime.now(timezone.utc).timestamp() * 1000
        )

        since_ms = (
            now_ms
            - (limit * TIMEFRAME_MS)
        )

        all_candles = []

        # ----------------------------------------------------
        # Pagination
        # ----------------------------------------------------

        while len(all_candles) < limit:

            remaining = limit - len(all_candles)

            request_limit = min(
                MAX_PER_REQUEST,
                remaining,
            )

            try:

                batch = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=TIMEFRAME,
                    since=since_ms,
                    limit=request_limit,
                )

            except Exception as exc:

                print(
                    f"[{symbol}] ERROR beim Laden: {exc}"
                )

                raise

            if not batch:

                print(
                    f"[{symbol}] Keine weiteren Daten."
                )

                break

            print(
                f"[{symbol}] API batch: "
                f"{len(batch)} candles"
            )

            all_candles.extend(batch)

            # ------------------------------------------------
            # Fortschrittsprüfung
            # ------------------------------------------------

            last_timestamp = int(batch[-1][0])

            next_since = (
                last_timestamp
                + TIMEFRAME_MS
            )

            if next_since <= since_ms:

                print(
                    f"[{symbol}] Pagination ohne Fortschritt."
                )

                break

            since_ms = next_since

            # ------------------------------------------------
            # Kleine Pause zur Schonung der API
            # ------------------------------------------------

            time.sleep(
                self.exchange.rateLimit / 1000
            )

            # ------------------------------------------------
            # Sicherheitslimit
            # ------------------------------------------------

            if len(batch) < request_limit:

                # OKX kann bei historischen Daten weniger
                # liefern. Wir versuchen trotzdem weiter,
                # solange die Zeitachse noch Fortschritt macht.
                pass

        # ====================================================
        # DATAFRAME
        # ====================================================

        if not all_candles:

            raise RuntimeError(
                f"[{symbol}] Keine OHLCV-Daten erhalten."
            )

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

        # ====================================================
        # CLEANUP
        # ====================================================

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

        # Nur die gewünschte Anzahl behalten
        if len(df) > limit:

            df = df.tail(limit).reset_index(
                drop=True
            )

        print(
            f"[{symbol}] Received "
            f"{len(df)} x {TIMEFRAME} candles"
        )

        if len(df) > 0:

            print(
                f"[{symbol}] Data range: "
                f"{df['timestamp'].iloc[0]} -> "
                f"{df['timestamp'].iloc[-1]}"
            )

        return df


    # ========================================================
    # RESAMPLE 5M -> 1H
    # ========================================================

    def resample_5m_to_1h(self, df):

        if df is None or df.empty:

            raise ValueError(
                "Keine 5m-Daten zum Resampling."
            )

        data = df.copy()

        data = data.set_index(
            "timestamp"
        )

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

        # Leere Stunden entfernen
        hourly = hourly.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )

        hourly = hourly.reset_index()

        print(
            f"1h candles: {len(hourly)}"
        )

        return hourly


    # ========================================================
    # FETCH + RESAMPLE
    # ========================================================

    def fetch_1h(
        self,
        symbol,
        limit=DEFAULT_LIMIT,
    ):

        df_5m = self.fetch_5m(
            symbol=symbol,
            limit=limit,
        )

        df_1h = self.resample_5m_to_1h(
            df_5m
        )

        return df_1h
