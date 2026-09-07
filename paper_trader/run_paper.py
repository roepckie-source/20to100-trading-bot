# ============================================================
# 20to100 Trading Bot
# V7-S0 PAPER TRADING RUNNER
#
# IMPORTANT:
# - PAPER TRADING ONLY
# - NO REAL ORDERS
# - NO API KEYS
# - V6-C ENTRY
# - V7-S0 SURVIVAL ENGINE
# ============================================================

from __future__ import annotations

import argparse
import time

import config_paper as config

from paper_trader.market_data import (
    create_exchange,
    fetch_5m,
    resample_5m_to_1h,
)

from paper_trader.trader import PaperTrader


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="20to100 V7-S0 Paper Trader"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Nur einen Datenzyklus ausführen.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print("")
    print("=" * 64)
    print("20to100 TRADING BOT")
    print("V7-S0 PAPER TRADING")
    print("=" * 64)
    print("")
    print("PAPER TRADING ONLY")
    print("KEINE ECHTEN ORDERS")
    print("KEINE API KEYS")
    print("")
    print(f"Exchange:         {config.PAPER_EXCHANGE}")
    print(f"Symbols:          {config.SYMBOLS}")
    print(
        f"Starting Capital: "
        f"${config.STARTING_CAPITAL:.2f}"
    )
    print(
        f"Data TF:          "
        f"{config.DATA_TIMEFRAME}"
    )
    print(
        f"Signal TF:        "
        f"{config.SIGNAL_TIMEFRAME}"
    )
    print(
        f"Poll:             "
        f"{config.POLL_SECONDS}s"
    )
    print(
        f"OHLCV History:    "
        f"{config.OHLCV_LIMIT}"
    )
    print("")
    print("Strategy:         V6-C")
    print("Risk Engine:      V7-S0")
    print("Leverage:         NONE")
    print("Live Trading:     FALSE")
    print("")
    print("=" * 64)
    print("")

    # ========================================================
    # HARD SAFETY CHECK
    # ========================================================

    if getattr(
        config,
        "LIVE_TRADING",
        False,
    ):

        raise RuntimeError(
            "ABBRUCH: LIVE_TRADING muss beim "
            "Paper-Trader FALSE sein."
        )

    # ========================================================
    # EXCHANGE
    # ========================================================

    print("Verbinde mit OKX Public API...")

    exchange = create_exchange()

    print("OKX Verbindung erfolgreich.")
    print("")

    # ========================================================
    # TRADERS
    # ========================================================

    traders = {}

    for symbol in config.SYMBOLS:

        print(
            f"[{symbol}] "
            f"Initialisiere V7-S0 Paper Trader..."
        )

        trader = PaperTrader(
            symbol=symbol
        )

        traders[symbol] = trader

    print("")

    # ========================================================
    # ONE CYCLE
    # ========================================================

    def run_cycle():

        print("")
        print("-" * 64)
        print("PAPER DATA CYCLE")
        print("-" * 64)

        for symbol, trader in traders.items():

            try:

                print("")
                print(
                    f"[{symbol}] "
                    f"Lade {config.DATA_TIMEFRAME} Daten..."
                )

                # ------------------------------------------------
                # FETCH 5m
                # ------------------------------------------------

                df_5m = fetch_5m(
                    exchange,
                    symbol,
                    limit=config.OHLCV_LIMIT,
                )

                if df_5m.empty:

                    print(
                        f"[{symbol}] "
                        "Keine Marktdaten."
                    )

                    continue

                print(
                    f"[{symbol}] "
                    f"5m candles: "
                    f"{len(df_5m)}"
                )

                # ------------------------------------------------
                # 5m -> 1h
                # ------------------------------------------------

                df_1h = (
                    resample_5m_to_1h(
                        df_5m
                    )
                )

                print(
                    f"[{symbol}] "
                    f"1h candles: "
                    f"{len(df_1h)}"
                )

                # ------------------------------------------------
                # MINIMUM HISTORY
                # ------------------------------------------------

                if len(df_1h) < 300:

                    print(
                        f"[{symbol}] "
                        f"NICHT GENUG HISTORIE: "
                        f"{len(df_1h)}/300"
                    )

                    continue

                # ------------------------------------------------
                # PROCESS
                # ------------------------------------------------

                trader.process(
                    df_1h
                )

                # ------------------------------------------------
                # STATUS
                # ------------------------------------------------

                trader.status()

            except Exception as exc:

                print(
                    f"[{symbol}] "
                    f"FEHLER: {exc}"
                )

        print("")
        print("-" * 64)
        print("PAPER DATA CYCLE COMPLETE")
        print("-" * 64)
        print("")

    # ========================================================
    # ONCE
    # ========================================================

    if args.once:

        print(
            "Einmaliger Paper-Test gestartet..."
        )

        run_cycle()

        print(
            "Einmaliger Paper-Test beendet."
        )

        return

    # ========================================================
    # CONTINUOUS
    # ========================================================

    print(
        "Dauerhafter Paper-Trading-Betrieb gestartet."
    )

    print(
        "STRG+C zum Beenden."
    )

    print("")

    while True:

        try:

            run_cycle()

        except KeyboardInterrupt:

            print("")
            print(
                "Paper-Trading beendet."
            )

            break

        except Exception as exc:

            print(
                f"Hauptschleifen-Fehler: {exc}"
            )

        time.sleep(
            config.POLL_SECONDS
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()