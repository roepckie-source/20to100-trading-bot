# ==========================================
# 20to100 Trading Bot
# Main Backtest Runner
# ==========================================

from pathlib import Path

import pandas as pd

from config import (
    SYMBOLS,
    TIMEFRAME,
    STARTING_CAPITAL,
    HISTORICAL_DAYS,
)

from data.data_manager import (
    fetch_ohlcv,
    save_data,
)

from backtest.engine import (
    BacktestEngine,
)

from backtest.metrics import (
    calculate_metrics,
    print_metrics,
)

from strategy.signals import (
    diagnose_signals,
    print_signal_diagnostics,
)


def main():

    # ======================================
    # Create directories
    # ======================================

    Path("logs").mkdir(
        exist_ok=True
    )

    Path("data").mkdir(
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("20→100 TRADING BOT")
    print("Strategy v1.1")
    print("=" * 60)

    # ======================================
    # Test every configured market
    # ======================================

    for symbol in SYMBOLS:

        print()
        print("-" * 60)
        print(f"BACKTEST: {symbol}")
        print("-" * 60)

        # ----------------------------------
        # Data filename
        # ----------------------------------

        filename = (
            symbol.replace("/", "_")
            + "_"
            + TIMEFRAME
            + ".csv"
        )

        data_file = (
            Path("data") /
            filename
        )

        # ----------------------------------
        # Load or download data
        # ----------------------------------

        if data_file.exists():

            print(
                f"Using cached data: "
                f"{data_file}"
            )

            df = pd.read_csv(
                data_file,
                index_col="timestamp",
                parse_dates=True,
            )

        else:

            print(
                "No local data found."
            )

            print(
                "Downloading historical data..."
            )

            df = fetch_ohlcv(
                symbol=symbol,
                timeframe=TIMEFRAME,
                days=HISTORICAL_DAYS,
            )

            save_data(
                df,
                symbol=symbol,
                timeframe=TIMEFRAME,
            )

        # ----------------------------------
        # Data information
        # ----------------------------------

        print()
        print(
            f"Candles: {len(df)}"
        )

        print(
            f"From:    {df.index.min()}"
        )

        print(
            f"To:      {df.index.max()}"
        )

        # ==================================
        # SIGNAL DIAGNOSTICS
        # ==================================

        print()
        print(
            "Analyzing entry conditions..."
        )

        diagnostic_data = diagnose_signals(
            df
        )

        print_signal_diagnostics(
            diagnostic_data,
            symbol,
        )

        # ==================================
        # BACKTEST
        # ==================================

        engine = BacktestEngine(
            starting_balance=STARTING_CAPITAL
        )

        print()
        print(
            "Running strategy backtest..."
        )

        result = engine.run(
            df=df,
            symbol=symbol,
        )

        # ==================================
        # METRICS
        # ==================================

        metrics = calculate_metrics(
            trades=result["trades"],
            equity_curve=result[
                "equity_curve"
            ],
            starting_balance=(
                STARTING_CAPITAL
            ),
        )

        print_metrics(
            metrics
        )

        # ==================================
        # SAVE TRADE LOG
        # ==================================

        if result["trades"]:

            trades_df = pd.DataFrame(
                [
                    {
                        "symbol":
                            trade.symbol,

                        "entry_time":
                            trade.entry_time,

                        "exit_time":
                            trade.exit_time,

                        "entry_price":
                            trade.entry_price,

                        "exit_price":
                            trade.final_exit_price,

                        "quantity":
                            trade.initial_quantity,

                        "gross_profit":
                            trade.gross_profit,

                        "fees":
                            trade.fees,

                        "slippage_cost":
                            trade.slippage_cost,

                        "net_profit":
                            trade.net_profit,

                        "exit_reason":
                            trade.exit_reason,

                        "r_multiple":
                            trade.r_multiple,
                    }

                    for trade
                    in result["trades"]
                ]
            )

            trade_file = (
                Path("logs") /
                (
                    symbol.replace(
                        "/", "_"
                    )
                    + "_trades.csv"
                )
            )

            trades_df.to_csv(
                trade_file,
                index=False,
            )

            print(
                f"Trade log: {trade_file}"
            )

        else:

            print(
                "No trades generated."
            )

        # ==================================
        # SAVE EQUITY CURVE
        # ==================================

        equity_file = (
            Path("logs") /
            (
                symbol.replace(
                    "/", "_"
                )
                + "_equity.csv"
            )
        )

        result[
            "equity_curve"
        ].to_csv(
            equity_file,
            index=False,
        )

        print(
            f"Equity log: {equity_file}"
        )

    # ======================================
    # COMPLETE
    # ======================================

    print()
    print("=" * 60)
    print("ALL BACKTESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
