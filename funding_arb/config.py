import os

NOTIONAL_USDT = float(os.getenv("NOTIONAL_USDT", "1000"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))

# ============================================================
# ABSOLUTE SAFETY SWITCHES
# ============================================================

PAPER_ONLY = True
LIVE_TRADING = False
REAL_ORDERS = False

# Conservative initial cost assumptions.
# We will replace these with verified account-specific fees.
DEFAULT_SPOT_FEE = 0.0010
DEFAULT_PERP_FEE = 0.0005
DEFAULT_SLIPPAGE = 0.0003

# Minimum estimated net edge per funding interval
MIN_NET_EDGE_PCT = float(
    os.getenv("MIN_NET_EDGE_PCT", "0.02")
)
