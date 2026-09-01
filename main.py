# ==========================================
# 20to100 Trading Bot
# Main Backtest Runner - Strategy v1.0
# ==========================================

from pathlib import Path

import pandas as pd

from data.data_manager import (
    fetch_ohlcv,
    save_data,
)

from backtest.engine import BacktestEngine

from backtest.metrics import (
    calculate_metrics,
    print_metrics,
)


# ==========================================
# Configuration
# ==========================================

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"

HISTORICAL_DAYS = 30

STARTING_CAPITAL = 20.00

FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

RISK_PER_TRADE = 0.015

ATR_STOP_MULTIPLIER = 1.5
TAKE_PROFIT_R = 2.0


# ==========================================
# Main
# ==========================================

def main():

    print()
    print("=" * 60)
    print("20→100 TRADING BOT")
    print("Strategy v1.0")
    print("=" * 60)

    print()
    print(f"Symbol:           {SYMBOL}")
    print(f"Timeframe:        {TIMEFRAME}")
    print(f"Starting capital: ${STARTING_CAPITAL:.2f}")
    print(f"Historical days:  {HISTORICAL_DAYS}")
    print()

    # --------------------------------------
    # Data file
    # --------------------------------------

    data_file = Path(
        "data",
        "BTC_USDT_5m.csv"
    )

    # --------------------------------------
    # Load existing data or download
    # --------------------------------------

    if data_file.exists():

        print(
            f"Using existing data: {data_file}"
        )

        df = pd.read_csv(
            data_file,
            index_col="timestamp",
            parse_dates=True,
        )

    else:

        print("No local data found.")
        print("Downloading historical data...")

        df = fetch_ohlcv(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            days=HISTORICAL_DAYS,
        )

        save_data(
            df,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
        )

    # --------------------------------------
    # Data overview
    # --------------------------------------

    print()
    print("-" * 60)
    print("DATA")
    print("-" * 60)

    print(f"Candles: {len(df)}")
    print(f"From:    {df.index.min()}")
    print(f"To:      {df.index.max()}")

    # --------------------------------------
    # Create engine
    # --------------------------------------

    engine = BacktestEngine(
        starting_balance=STARTING_CAPITAL,
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
        risk_per_trade=RISK_PER_TRADE,
        atr_stop_multiplier=ATR_STOP_MULTIPLIER,
        take_profit_r=TAKE_PROFIT_R,
    )

    # --------------------------------------
    # Run backtest
    # --------------------------------------

    print()
    print("-" * 60)
    print("RUNNING BACKTEST")
    print("-" * 60)

    result = engine.run(
        df=df,
        symbol=SYMBOL,
    )

    # --------------------------------------
    # Calculate metrics
    # --------------------------------------

    metrics = calculate_metrics(
        trades=result["trades"],
        equity_curve=result["equity_curve"],
        starting_balance=STARTING_CAPITAL,
    )

    # --------------------------------------
    # Print report
    # --------------------------------------

    print_metrics(metrics)

    # --------------------------------------
    # Save trades
    # --------------------------------------

    if result["trades"]:

        trades_df = pd.DataFrame(
            [
                {
                    "symbol": trade.symbol,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.final_exit_price,
                    "quantity": trade.initial_quantity,
                    "gross_profit": trade.gross_profit,
                    "fees": trade.fees,
                    "slippage_cost": trade.slippage_cost,
                    "net_profit": trade.net_profit,
                    "exit_reason": trade.exit_reason,
                    "r_multiple": trade.r_multiple,
                }
                for trade in result["trades"]
            ]
        )

        trades_df.to_csv(
            "logs/trades.csv",
            index=False,
        )

        print(
            "Trade log saved to logs/trades.csv"
        )

    # --------------------------------------
    # Save equity curve
    # --------------------------------------

    result["equity_curve"].to_csv(
        "logs/equity.csv",
        index=False,
    )

    print(
        "Equity curve saved to logs/equity.csv"
    )

    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
