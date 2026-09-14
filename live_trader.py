import itertools
import os
import sys
import time

try:
    import ccxt
except ImportError:
    print("❌ CCXT ist nicht installiert! Bitte 'pip install ccxt' ausführen.")
    sys.exit(1)

# ============================================================
# ⚙️ TRADER KONFIGURATION
# ============================================================

# Mindestmarge in % für Arbitrage-Trades
MIN_PROFIT_THRESHOLD_PCT = 0.30

# Fester Order-Betrag in USDT pro Arbitrage-Trade
TRADE_AMOUNT_USDT = 10.0

# Sicherheitsgrenzen für GitHub Actions.
# Ein Workflow-Lauf darf niemals unendlich laufen.
MAX_CYCLES = 28
SCAN_INTERVAL_SECONDS = 15

# Mindest-Gegenwert in USDT für den Altcoin-Auto-Cleanup
MIN_CLEANUP_VALUE_USDT = 5.0

SYMBOLS_TO_SCAN = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "LTC/USDT",
]

# ============================================================
# 🏦 BÖRSEN INITIALISIEREN
# ============================================================

def init_exchanges():
    exchanges = {}

    okx_key, okx_sec, okx_pass = (
        os.getenv("OKX_API_KEY"),
        os.getenv("OKX_API_SECRET"),
        os.getenv("OKX_PASSPHRASE"),
    )

    if okx_key and okx_sec and okx_pass:
        try:
            exchanges["okx"] = ccxt.okx({
                "apiKey": okx_key,
                "secret": okx_sec,
                "password": okx_pass,
                "hostname": "my.okx.com",
                "enableRateLimit": True,
            })
        except Exception as e:
            print(f"❌ Fehler bei OKX: {e}")

    mexc_key, mexc_sec = os.getenv("MEXC_API_KEY"), os.getenv("MEXC_API_SECRET")

    if mexc_key and mexc_sec:
        try:
            exchanges["mexc"] = ccxt.mexc({
                "apiKey": mexc_key,
                "secret": mexc_sec,
                "enableRateLimit": True,
            })
        except Exception as e:
            print(f"❌ Fehler bei MEXC: {e}")

    bit_key, bit_sec = os.getenv("BITRUE_API_KEY"), os.getenv("BITRUE_API_SECRET")

    if bit_key and bit_sec:
        try:
            exchanges["bitrue"] = ccxt.bitrue({
                "apiKey": bit_key,
                "secret": bit_sec,
                "enableRateLimit": True,
            })
        except Exception as e:
            print(f"❌ Fehler bei BITRUE: {e}")

    return exchanges


# ============================================================
# 🧹 ALTCOIN AUTO-CLEANUP
# ============================================================

def cleanup_altcoins_to_usdt(exchanges):
    """Prüft alle Börsen auf vorhandene Altcoins und verkauft sie in USDT."""
    print("\n🧹 Starte Altcoin-Auto-Cleanup auf allen Börsen...")

    for name, ex in exchanges.items():
        ex_name = name.upper()

        try:
            balance = ex.fetch_balance()
            free_balances = balance.get("free", {})

            for coin, amount in free_balances.items():
                if coin in ["USDT", "USD", "USDC"] or amount <= 0:
                    continue

                symbol = f"{coin}/USDT"

                try:
                    ticker = ex.fetch_ticker(symbol)
                    current_price = ticker["last"]
                except Exception:
                    continue

                estimated_value_usdt = amount * current_price

                if estimated_value_usdt < MIN_CLEANUP_VALUE_USDT:
                    print(
                        f"ℹ️ [{ex_name}] {coin}: Wert "
                        f"(${estimated_value_usdt:.2f}) unter Minimum "
                        f"(${MIN_CLEANUP_VALUE_USDT:.2f}). Überspringe."
                    )
                    continue

                print(
                    f"⚡ [{ex_name}] Tausche {amount:.4f} {coin} "
                    f"(~${estimated_value_usdt:.2f} USDT) per Market-Sell in USDT..."
                )

                try:
                    order = ex.create_market_sell_order(symbol, amount)
                    order_id = order.get("id", "N/A")
                    print(
                        f"✅ [{ex_name}] Erfolgreich verkauft! Order-ID: {order_id}"
                    )
                except Exception as e:
                    print(f"❌ [{ex_name}] Fehler beim Verkauf von {coin}: {e}")

        except Exception as e:
            print(f"❌ [{ex_name}] Fehler beim Abrufen des Guthabens: {e}")


# ============================================================
# 📊 ARBITRAGE SCANNER & TRADER LOGIK
# ============================================================

def scan_and_trade_arbitrage(exchanges):
    """Holt Preise ab, berechnet Spreads und führt Signale aus."""

    ex_names = list(exchanges.keys())
    pairs = list(itertools.permutations(ex_names, 2))

    for symbol in SYMBOLS_TO_SCAN:
        tickers = {}

        for name, ex in exchanges.items():
            try:
                tickers[name] = ex.fetch_ticker(symbol)
            except Exception:
                continue

        for buy_ex_name, sell_ex_name in pairs:
            if buy_ex_name not in tickers or sell_ex_name not in tickers:
                continue

            buy_price = tickers[buy_ex_name].get("ask")
            sell_price = tickers[sell_ex_name].get("bid")

            if not buy_price or not sell_price or buy_price <= 0:
                continue

            spread_pct = ((sell_price - buy_price) / buy_price) * 100

            if spread_pct > 0.05:
                print(
                    f"🔍 {symbol} | Buy [{buy_ex_name.upper()} @ ${buy_price:.4f}] "
                    f"➔ Sell [{sell_ex_name.upper()} @ ${sell_price:.4f}] | "
                    f"Spread: {spread_pct:+.2f}%"
                )

            if spread_pct >= MIN_PROFIT_THRESHOLD_PCT:
                print(f"\n🚀 ARBITRAGE-SIGNAL: {symbol} (+{spread_pct:.2f}%)")

                execute_arbitrage_trade(
                    exchanges[buy_ex_name],
                    exchanges[sell_ex_name],
                    symbol,
                    buy_price,
                    TRADE_AMOUNT_USDT,
                )


def execute_arbitrage_trade(
    buy_exchange, sell_exchange, symbol, buy_price, amount_usdt
):
    """Führt Kauf und Verkauf aus."""

    buy_name = buy_exchange.id.upper()
    sell_name = sell_exchange.id.upper()
    coin_amount = amount_usdt / buy_price

    print(
        f"⚡ Führe Trade aus: Kaufe {coin_amount:.4f} {symbol} "
        f"auf {buy_name} & verkaufe auf {sell_name}..."
    )

    try:
        buy_order = buy_exchange.create_market_buy_order(symbol, coin_amount)
        print(f"✅ [{buy_name}] Kauf ausgeführt. Order-ID: {buy_order.get('id')}")
    except Exception as e:
        print(f"❌ [{buy_name}] Kauf fehlgeschlagen: {e}")
        return

    try:
        sell_order = sell_exchange.create_market_sell_order(
            symbol, coin_amount
        )
        print(f"✅ [{sell_name}] Verkauf ausgeführt. Order-ID: {sell_order.get('id')}")
    except Exception as e:
        print(
            f"⚠️ [{sell_name}] Verkauf fehlgeschlagen! "
            f"Coin muss über Auto-Cleanup verkauft werden. Fehler: {e}"
        )


# ============================================================
# 🔄 MAIN LOOP & TRADING ENGINE
# ============================================================

def run_trader():
    exchanges = init_exchanges()

    if not exchanges:
        print("❌ Keine Börsen geladen. Abbruch.")
        return

    print("✅ Börsen erfolgreich initialisiert!")

    cleanup_altcoins_to_usdt(exchanges)

    print("\n============================================================")
    print("🚀 STARTE LIVE TRADING & SCANNING")
    print("============================================================")
    print(f"Maximale Laufzeit: ca. {MAX_CYCLES * SCAN_INTERVAL_SECONDS / 60:.1f} Minuten")

    cycle = 0

    try:
        while cycle < MAX_CYCLES:
            cycle += 1

            print(f"\n--- Durchlauf #{cycle}/{MAX_CYCLES} ---")

            scan_and_trade_arbitrage(exchanges)

            if cycle % 10 == 0:
                cleanup_altcoins_to_usdt(exchanges)

            if cycle < MAX_CYCLES:
                time.sleep(SCAN_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n🛑 Trader manuell beendet.")

    finally:
        print("\n🧹 Abschluss-Cleanup vor Workflow-Ende...")
        cleanup_altcoins_to_usdt(exchanges)

    print("\n============================================================")
    print("✅ LIVE TRADER WORKFLOW-ZYKLUS BEENDET")
    print(f"   Durchläufe: {cycle}/{MAX_CYCLES}")
    print("   Kein Endlosprozess.")
    print("============================================================")


if __name__ == "__main__":
    run_trader()
