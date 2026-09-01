# ==========================================
# 20to100 Trading Bot
# Configuration
# ==========================================

# Trading pairs
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
]

# Timeframe
TIMEFRAME = "5m"

# Starting capital
STARTING_CAPITAL = 20.00

# Risk management
RISK_PER_TRADE = 0.015
MAX_DAILY_LOSS = 0.05

# Strategy parameters
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50

RSI_PERIOD = 14
RSI_MIN = 55
RSI_MAX = 70

ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5

VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.5

# Breakout
BREAKOUT_CANDLES = 6

# Trading costs
FEE_RATE = 0.001
MAX_SPREAD = 0.001

# Profit management
TAKE_PROFIT_R = 2.0
PARTIAL_EXIT_1_R = 1.0
PARTIAL_EXIT_2_R = 2.0

# Trailing stop
TRAILING_ATR_MULTIPLIER = 1.0

# Time stop
MAX_TRADE_MINUTES = 60

# Safety
MAX_CONSECUTIVE_LOSSES = 3
LOSS_COOLDOWN_MINUTES = 120
TRADE_COOLDOWN_MINUTES = 5
