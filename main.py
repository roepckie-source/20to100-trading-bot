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

from strategy.indicators import (
    calculate_indicators,
)

from strategy.signals import (
    diagnose_signals,
    print_signal_diagnostics,
)

from backtest.engine import (
    BacktestEngine,
)

from backtest.metrics import (
    calculate_metrics,
    print_metrics,
)


def print_result_line(
    symbol,
    starting_capital,
    final_capital,
    trades,
):
    """
    Print compact result for one market.
    """

    profit_loss = (
        final_capital -
        starting_capital
    )

    return_pct = (
        profit_loss /
        starting_capital *
        100
    )

    progress_to_100 = (
        final_capital /
        100 *
        100
    )

    if final_capital >= 100:

        status = "🎯 TARGET REACHED"

    elif profit_loss > 0:

        status = "🟢 PROFITABLE"

    elif profit_loss < 0:

        status = "🔴 LOSS"

    else:

        status = "⚪ BREAK-EVEN"

    print()
    print("=" * 60)
    print(
        f"20→100 RESULT: {symbol}"
    )
    print("=" * 60)

    print(
        f"Starting capital:  ${starting_capital:.2f}"
    )

    print(
        f"Final capital:     ${final_capital:.2f}"
    )

    print(
        f"Profit/Loss:       "
        f"${profit_loss:+.2f}"
    )

    print(
        f"Return:            "
        f"{return_pct:+.2f}%"
    )

    print(
        f"Progress to $100:  "
        f"{progress_to_100:.2f}%"
    )

    print(
        f"Trades:            "
        f"{len(trades)}"
    )

    print(
        f"Status:             {status}"
    )

    print("=" * 60)


def print_portfolio_result(
    starting_capital,
    results,
):
    """
    Print combined portfolio result.

    Each market is treated as a separate
    strategy test. The portfolio figure is
    calculated from the average result so
    that BTC and ETH are compared on equal
    starting capital.
    """

    if not results:
        return

    # --------------------------------------
    # Calculate average final capital
    # --------------------------------------

    final_capitals = [
        result["final_capital"]
        for result in results
    ]

    average_final_capital = (
        sum(final_capitals)
        /
        len(final_capitals)
    )

    profit_loss = (
        average_final_capital -
        starting_capital
    )

    return_pct = (
        profit_loss /
        starting_capital *
        100
    )

    progress_to_100 = (
        average_final_capital /
        100 *
        100
    )

    total_trades = sum(
        result["trades"]
        for result in results
    )

    total_winners = sum(
        result["winners"]
        for result in results
    )

    total_losers = sum(
        result["losers"]
        for result in results
    )

    if average_final_capital >= 100:

        status = "🎯 TARGET REACHED"

    elif profit_loss > 0:

        status = "🟢 PROFITABLE"

    elif profit_loss < 0:

        status = "🔴 LOSS"

    else:

        status = "⚪ BREAK-EVEN"

    print()
    print("=" * 60)
    print("20→100 PORTFOLIO RESULT")
    print("=" * 60)

    print(
        f"Starting capital:  "
        f"${starting_capital:.2f}"
    )

    print(
        f"Average final:     "
        f"${average_final_capital:.2f}"
    )

    print(
        f"Total P/L:         "
        f"${profit_loss:+.2f}"
    )

    print(
        f"Total return:      "
        f"{return_pct:+.2f}%"
    )

    print(
        f"Progress to $100:  "
        f"{progress_to_100:.2f}%"
    )

    print(
        f"Total trades:      "
        f"{total_trades}"
    )

    print(
        f"Winners:           "
        f"{total_winners}"
    )

    print(
        f"Losers:            "
        f"{total_losers}"
    )

    print(
        f"Status:             {status}"
    )

    print("=" * 60)


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

    results = []

    # ======================================
    # Process every market
    # ======================================

    for symbol in SYMBOLS:

        print()
        print("-" * 60)
        print(
            f"BACKTEST: {symbol}"
        )
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
        # Load cached data
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

        # ----------------------------------
        # Download data
        # ----------------------------------

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
        # CALCULATE INDICATORS
        # ==================================

        print()
        print(
            "Calculating indicators..."
        )

        indicator_df = (
            calculate_indicators(df)
        )

        # ==================================
        # SIGNAL DIAGNOSTICS
        # ==================================

        print()
        print(
            "Analyzing entry conditions..."
        )

        diagnostic_data = (
            diagnose_signals(
                indicator_df
            )
        )

        print_signal_diagnostics(
            diagnostic_data,
            symbol,
        )

        # ==================================
        # BACKTEST
        # ==================================

        print()
        print(
            "Running strategy backtest..."
        )

        engine = BacktestEngine(
            starting_balance=(
                STARTING_CAPITAL
            )
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
        # Determine final capital
        # ==================================

        equity_curve = (
            result["equity_curve"]
        )

        if (
            not equity_curve.empty
            and
            "equity" in equity_curve.columns
        ):

            final_capital = float(
                equity_curve[
                    "equity"
                ].iloc[-1]
            )

        else:

            final_capital = float(
                result["balance"]
            )

        # ==================================
        # Trade statistics
        # ==================================

        trades = result["trades"]

        winners = sum(
            1
            for trade in trades
            if trade.net_profit > 0
        )

        losers = sum(
            1
            for trade in trades
            if trade.net_profit < 0
        )

        # ==================================
        # Result line
        # ==================================

        print_result_line(
            symbol=symbol,

            starting_capital=(
                STARTING_CAPITAL
            ),

            final_capital=final_capital,

            trades=trades,
        )

        # ==================================
        # Store portfolio result
        # ==================================

        results.append(
            {
                "symbol": symbol,

                "final_capital":
                    final_capital,

                "trades":
                    len(trades),

                "winners":
                    winners,

                "losers":
                    losers,
            }
        )

        # ==================================
        # SAVE TRADE LOG
        # ==================================

        if trades:

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

                    for trade in trades
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
    # PORTFOLIO RESULT
    # ======================================

    print_portfolio_result(
        starting_capital=(
            STARTING_CAPITAL
        ),
        results=results,
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
