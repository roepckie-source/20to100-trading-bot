# ============================================================
# 20to100 Trading Bot
# V7-S0 PAPER TRADING CONFIG
# ============================================================

import os


# ============================================================
# EXCHANGE
# ============================================================

# Nur öffentliche Marktdaten.
# KEINE API-Keys.
PAPER_EXCHANGE = os.getenv(
    "PAPER_EXCHANGE",
    "okx"
)


# ============================================================
# MARKETS
# ============================================================

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
]


# ============================================================
# TIMEFRAMES
# ============================================================

# Wir holen 5-Minuten-Daten.
DATA_TIMEFRAME = "5m"

# V7-S0 arbeitet auf 1h.
SIGNAL_TIMEFRAME = "1h"


# ============================================================
# LOOP
# ============================================================

# Wie oft neue Marktdaten abgefragt werden.
POLL_SECONDS = int(
    os.getenv(
        "POLL_SECONDS",
        "60"
    )
)


# ============================================================
# PAPER CAPITAL
# ============================================================

STARTING_CAPITAL = float(
    os.getenv(
        "PAPER_STARTING_CAPITAL",
        "20.0"
    )
)


# ============================================================
# V7-S0 RISK SETTINGS
# ============================================================

BASE_RISK_PER_TRADE = 0.01

FEE_RATE = 0.001

SLIPPAGE_RATE = 0.0005

ATR_STOP_MULTIPLIER = 3.0

TRAILING_ATR_MULTIPLIER = 3.0

ADX_MIN = 20.0


# ============================================================
# SURVIVAL PROTECTION
# ============================================================

MAX_DAILY_LOSS = 0.05

MAX_CONSECUTIVE_LOSSES = 3

LOSS_COOLDOWN_BARS = 24

GLOBAL_MAX_DRAWDOWN = 0.20


# ============================================================
# HARD SAFETY
# ============================================================

# MUSS False bleiben.
#
# Dieser Paper Trader darf keine echten Orders senden.
LIVE_TRADING = False


# ============================================================
# DATA
# ============================================================

# Anzahl 5m-Kerzen pro Abruf.
OHLCV_LIMIT = int(
    os.getenv(
        "OHLCV_LIMIT",
        "1000"
    )
)


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = "logs/paper"

STATE_FILE = (
    "logs/paper/paper_state.json"
)

TRADES_FILE = (
    "logs/paper/paper_trades.csv"
)

EQUITY_FILE = (
    "logs/paper/paper_equity.csv"
)

SIGNALS_FILE = (
    "logs/paper/paper_signals.csv"
)
