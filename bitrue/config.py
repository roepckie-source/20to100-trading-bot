"""
Bitrue BTC/USDT 5m Strategy
Configuration

PAPER / BACKTEST ONLY
NO LIVE TRADING
"""

# ============================================================
# MARKET
# ============================================================

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"


# ============================================================
# CAPITAL
# ============================================================

STARTING_CAPITAL = 100.00

# Maximum percentage of current capital risked per trade
RISK_PER_TRADE = 0.01       # 1%

# Maximum daily loss
MAX_DAILY_LOSS = 0.05       # 5%


# ============================================================
# FEES / SLIPPAGE
# ============================================================

# Bitrue Futures taker fee used for conservative backtesting
TAKER_FEE = 0.0006          # 0.06%

# Conservative assumed slippage per execution
SLIPPAGE = 0.0002           # 0.02%


# ============================================================
# EMA TREND
# ============================================================

EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 55


# ============================================================
# MOMENTUM
# ============================================================

# Minimum 5m price movement required for entry
MIN_MOMENTUM_PCT = 0.05

# Momentum lookback candles
MOMENTUM_LOOKBACK = 3


# ============================================================
# ATR / VOLATILITY
# ============================================================

ATR_PERIOD = 14

# Minimum ATR relative to price
MIN_ATR_PCT = 0.03


# ============================================================
# VOLUME
# ============================================================

VOLUME_LOOKBACK = 20

# Current volume must be at least this fraction
# of average volume
MIN_VOLUME_RATIO = 1.00


# ============================================================
# TRADE MANAGEMENT
# ============================================================

STOP_LOSS_PCT = 0.30

TAKE_PROFIT_PCT = 0.60


# ============================================================
# POSITION LIMITS
# ============================================================

MIN_POSITION_USD = 5.00
MAX_POSITION_USD = 25.00


# ============================================================
# STRATEGY
# ============================================================

ALLOW_LONG = True
ALLOW_SHORT = True


# ============================================================
# SAFETY
# ============================================================

PAPER_ONLY = True
LIVE_TRADING = False
