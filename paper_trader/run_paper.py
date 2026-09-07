# ============================================================
# 20to100 Trading Bot
# PAPER TRADING RUNNER - V7-S0
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

from paper_trader.trader import (
    PaperTrader,
)


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
    print("============================================================")
    print("20to100 TRADING BOT")
    print("V7-S0 PAPER TRADING")
    print("============================================================")
    print("")
    print("⚠️  PAPER TRADING ONLY")
    print("⚠️  KEINE ECHTEN ORDERS")
    print("⚠️  KEINE API KEYS")
    print("")
    print(f"Exchange:        {config.PAPER_EXCHANGE}")
    print(f"Symbols:         {config.SYMBOLS}")
    print(f"Starting Capital:${config.STARTING_CAPITAL:.2f}")
    print(f"Data TF:         {config.DATA_TIMEFRAME}")
    print(f"Signal TF:       {config.SIGNAL_TIMEFRAME}")
    print(f"Poll:            {config.POLL_SECONDS}s")
    print("")
    print("Strategy:        V6-C")
    print("Risk Engine:     V7-S0")
    print("Leverage:        NONE")
    print("Live Trading:    FALSE")
    print("")
    print("============================================================")
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
            "ABBRUCH: LIVE_TRADING darf beim "
            "Paper-Trader NICHT aktiviert sein."
        )

    # ========================================================
    # EXCHANGE
    # ========================================================

    exchange = create_exchange()

    # ========================================================
    # TRADERS
    # ========================================================

    traders = {}

    for symbol in config.SYMBOLS:

        trader = PaperTrader(
            symbol=symbol,

            starting_balance=(
                config.STARTING_CAPITAL
            ),

            base_risk_per_trade=(
                config.BASE_RISK_PER_TRADE
            ),

            fee_rate=(
                config.FEE_RATE
            ),

            slippage_rate=(
                config.SLIPPAGE_RATE
            ),

            atr_stop_multiplier=(
                config.ATR_STOP_MULTIPLIER
            ),

            trailing_atr_multiplier=(
                config.TRAILING_ATR_MULTIPLIER
            ),

            adx_min=(
                config.ADX_MIN
            ),

            max_daily_loss=(
                config.MAX_DAILY_LOSS
            ),

            max_consecutive_losses=(
                config.MAX_CONSECUTIVE_LOSSES
            ),

            loss_cooldown_bars=(
                config.LOSS_COOLDOWN_BARS
            ),

            global_max_drawdown=(
                config.GLOBAL_MAX_DRAWDOWN
            ),

            variant="V6_C",
        )

        trader.load_state()

        traders[symbol] = trader

    # ========================================================
    # ONE CYCLE
    # ========================================================

    def run_cycle():

        print("")
        print("------------------------------------------------------------")
        print("PAPER DATA CYCLE")
        print("------------------------------------------------------------")

        for symbol, trader in traders.items():

            try:

                print(
                    f"[{symbol}] "
                    f"Lade {config.DATA_TIMEFRAME} Daten..."
                )

                df_5m = fetch_5m(
                    exchange,
                    symbol,
                    limit=config.OHLCV_LIMIT,
                )

                if df_5m.empty:

                    print(
                        f"[{symbol}] "
                        "Keine Daten."
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

                df_1h = resample_5m_to_1h(
                    df_5m
                )

                print(
                    f"[{symbol}] "
                    f"1h candles: "
                    f"{len(df_1h)}"
                )

                if len(df_1h) < 300:

                    print(
                        f"[{symbol}] "
                        "❌ Nicht genug 1h-Daten."
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

                status = trader.status()

                print(
                    f"[{symbol}] "
                    f"Balance: "
                    f"${status['balance']:.4f}"
                )

                print(
                    f"[{symbol}] "
                    f"Trades: "
                    f"{status['trades']}"
                )

                print(
                    f"[{symbol}] "
                    f"Position: "
                    f"{'OPEN' if status['position'] else 'NONE'}"
                )

                print(
                    f"[{symbol}] "
                    f"Kill Switch: "
                    f"{status['kill_switch']}"
                )

            except Exception as exc:

                print(
                    f"[{symbol}] "
                    f"❌ FEHLER: {exc}"
                )

        print("------------------------------------------------------------")
        print("")

    # ========================================================
    # ONCE
    # ========================================================

    if args.once:

        print(
            "▶ Einmaliger Paper-Test gestartet..."
        )

        run_cycle()

        print(
            "✓ Einmaliger Paper-Test beendet."
        )

        return

    # ========================================================
    # CONTINUOUS
    # ========================================================

    print(
        "▶ Dauerhafter Paper-Trading-Betrieb gestartet."
    )

    print(
        "▶ STRG+C zum Beenden."
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
                f"❌ Hauptschleifen-Fehler: {exc}"
            )

        time.sleep(
            config.POLL_SECONDS
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
