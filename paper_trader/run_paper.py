# ============================================================
# 20to100 Trading Bot
# V7-S0 PAPER TRADING RUNNER
# ============================================================

from __future__ import annotations

import time

from config_paper import (
    PAPER_EXCHANGE,
    SYMBOLS,
    OHLCV_LIMIT,
    POLL_SECONDS,
    STARTING_CAPITAL,
)

from paper_trader.market_data import (
    create_exchange,
    fetch_5m,
    resample_5m_to_1h,
)

from paper_trader.trader import (
    PaperTrader,
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 72
    )

    print(
        "20to100 V7-S0 PAPER TRADER"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "MODE:          PAPER ONLY"
    )

    print(
        f"Exchange:      "
        f"{PAPER_EXCHANGE}"
    )

    print(
        f"Symbols:       "
        f"{', '.join(SYMBOLS)}"
    )

    print(
        f"Start capital: "
        f"${STARTING_CAPITAL:.2f}"
    )

    print(
        "Signal TF:     1h"
    )

    print(
        "Data TF:       5m"
    )

    print()

    print(
        "NO REAL ORDERS WILL BE SENT."
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # PUBLIC EXCHANGE CONNECTION
    # --------------------------------------------------------

    exchange = create_exchange(
        PAPER_EXCHANGE
    )

    # --------------------------------------------------------
    # ONE PAPER ACCOUNT PER ASSET
    # --------------------------------------------------------

    traders = {

        symbol:
            PaperTrader(
                symbol=symbol,
                starting_balance=
                    STARTING_CAPITAL,
            )

        for symbol in SYMBOLS
    }

    # --------------------------------------------------------
    # LOAD PREVIOUS STATE
    # --------------------------------------------------------

    for trader in traders.values():

        loaded = trader.load_state()

        if loaded:

            print(
                f"[{trader.symbol}] "
                f"state restored"
            )

        else:

            print(
                f"[{trader.symbol}] "
                f"new paper session"
            )

    # --------------------------------------------------------
    # CONTINUOUS LOOP
    # --------------------------------------------------------

    while True:

        print()

        print(
            "--- refresh --- "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        for (
            symbol,
            trader
        ) in traders.items():

            try:

                # --------------------------------------------
                # FETCH 5m
                # --------------------------------------------

                df_5m = fetch_5m(
                    exchange,
                    symbol,
                    limit=OHLCV_LIMIT,
                )

                # --------------------------------------------
                # 5m -> 1h
                # --------------------------------------------

                hourly = (
                    resample_5m_to_1h(
                        df_5m
                    )
                )

                print(
                    f"[{symbol}] "
                    f"5m={len(df_5m)} "
                    f"1h={len(hourly)}"
                )

                # --------------------------------------------
                # PROCESS V7-S0
                # --------------------------------------------

                trader.process(
                    hourly
                )

            except KeyboardInterrupt:

                raise

            except Exception as exc:

                print(
                    f"[ERROR] "
                    f"{symbol}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        print()

        print(
            f"Sleeping "
            f"{POLL_SECONDS}s..."
        )

        try:

            time.sleep(
                POLL_SECONDS
            )

        except KeyboardInterrupt:

            print()

            print(
                "Paper trader stopped."
            )

            break


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
