# ==========================================
# 20to100 Trading Bot
# Main Backtest Runner
# Strategy v2.0
# ATR Stop Parameter Test
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


# ==========================================
# ATR VALUES TO TEST
# ==========================================

ATR_TEST_VALUES = [
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.2,
]


# ==========================================
# EXIT ANALYSIS
# ==========================================

def print_exit_analysis(
    trades,
    symbol,
):

    print()
    print("=" * 60)
    print(
        f"EXIT ANALYSIS: {symbol}"
    )
    print("=" * 60)

    if not trades:

        print(
            "No trades available."
        )

        print("=" * 60)

        return

    exit_reasons = [
        "STOP_LOSS",
        "EMA_EXIT",
        "RSI_EXIT",
        "TIME_STOP",
        "END_OF_TEST",
    ]

    total_net_profit = sum(
        trade.net_profit
        for trade in trades
    )

    print(
        f"Total trades:      "
        f"{len(trades)}"
    )

    print(
        f"Total net P/L:     "
        f"${total_net_profit:+.4f}"
    )

    print("-" * 60)

    print(
        f"{'Exit':<18}"
        f"{'Trades':>8}"
        f"{'Wins':>8}"
        f"{'Losses':>8}"
        f"{'Win %':>9}"
        f"{'Net P/L':>12}"
        f"{'Avg R':>10}"
    )

    print("-" * 60)

    for reason in exit_reasons:

        reason_trades = [
            trade
            for trade in trades
            if trade.exit_reason == reason
        ]

        count = len(
            reason_trades
        )

        if count == 0:

            print(
                f"{reason:<18}"
                f"{0:>8}"
                f"{0:>8}"
                f"{0:>8}"
                f"{0.0:>8.2f}%"
                f"{0.0:>12.4f}"
                f"{0.0:>10.3f}"
            )

            continue

        wins = sum(
            1
            for trade in reason_trades
            if trade.net_profit > 0
        )

        losses = sum(
            1
            for trade in reason_trades
            if trade.net_profit < 0
        )

        win_rate = (
            wins /
            count *
            100
        )

        net_profit = sum(
            trade.net_profit
            for trade in reason_trades
        )

        avg_r = sum(
            trade.r_multiple
            for trade in reason_trades
        ) / count

        print(
            f"{reason:<18}"
            f"{count:>8}"
            f"{wins:>8}"
            f"{losses:>8}"
            f"{win_rate:>8.2f}%"
            f"{net_profit:>12.4f}"
            f"{avg_r:>10.3f}"
        )

    print("=" * 60)


# ==========================================
# ATR STOP TEST
# ==========================================

def run_atr_test(
    df,
    symbol,
):

    results = []

    print()
    print("=" * 100)
    print(
        f"ATR STOP TEST: {symbol}"
    )
    print("=" * 100)

    print(
        f"{'ATR':>7}"
        f"{'Final':>12}"
        f"{'Return':>12}"
        f"{'Trades':>10}"
        f"{'Win %':>10}"
        f"{'PF':>10}"
        f"{'Expectancy':>14}"
        f"{'Max DD':>10}"
    )

    print("-" * 100)

    for atr_multiplier in ATR_TEST_VALUES:

        print()
        print(
            f"Testing ATR "
            f"{atr_multiplier:.1f}x..."
        )

        engine = BacktestEngine(
            starting_balance=(
                STARTING_CAPITAL
            ),

            atr_stop_multiplier=(
                atr_multiplier
            ),
        )

        result = engine.run(
            df=df,
            symbol=symbol,
        )

        trades = result[
            "trades"
        ]

        equity_curve = result[
            "equity_curve"
        ]

        # ----------------------------------
        # Final capital
        # ----------------------------------

        if (
            not equity_curve.empty
            and
            "equity"
            in equity_curve.columns
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

        # ----------------------------------
        # Return
        # ----------------------------------

        return_pct = (
            (
                final_capital -
                STARTING_CAPITAL
            )
            /
            STARTING_CAPITAL
            *
            100
        )

        # ----------------------------------
        # Winners
        # ----------------------------------

        winners = sum(
            1
            for trade in trades
            if trade.net_profit > 0
        )

        win_rate = (
            winners /
            len(trades) *
            100
            if trades
            else 0.0
        )

        # ----------------------------------
        # Profit factor
        # ----------------------------------

        gross_wins = sum(
            trade.net_profit
            for trade in trades
            if trade.net_profit > 0
        )

        gross_losses = abs(
            sum(
                trade.net_profit
                for trade in trades
                if trade.net_profit < 0
            )
        )

        if gross_losses > 0:

            profit_factor = (
                gross_wins /
                gross_losses
            )

        else:

            profit_factor = 0.0

        # ----------------------------------
        # Expectancy
        # ----------------------------------

        if trades:

            expectancy = (
                sum(
                    trade.net_profit
                    for trade in trades
                )
                /
                len(trades)
            )

        else:

            expectancy = 0.0

        # ----------------------------------
        # Max drawdown
        # ----------------------------------

        if (
            not equity_curve.empty
            and
            "equity"
            in equity_curve.columns
        ):

            equity = (
                equity_curve[
                    "equity"
                ]
                .astype(float)
            )

            peak = equity.cummax()

            drawdown = (
                (
                    equity -
                    peak
                )
                /
                peak
                *
                100
            )

            max_drawdown = abs(
                float(
                    drawdown.min()
                )
            )

        else:

            max_drawdown = 0.0

        # ----------------------------------
        # Print result
        # ----------------------------------

        print(
            f"{atr_multiplier:>7.1f}"
            f"{final_capital:>12.2f}"
            f"{return_pct:>11.2f}%"
            f"{len(trades):>10}"
            f"{win_rate:>9.2f}%"
            f"{profit_factor:>10.3f}"
            f"{expectancy:>14.4f}"
            f"{max_drawdown:>9.2f}%"
        )

        results.append(
            {
                "atr":
                    atr_multiplier,

                "final_capital":
                    final_capital,

                "return_pct":
                    return_pct,

                "trades":
                    len(trades),

                "win_rate":
                    win_rate,

                "profit_factor":
                    profit_factor,

                "expectancy":
                    expectancy,

                "max_drawdown":
                    max_drawdown,
            }
        )

    # ======================================
    # Best variant
    # ======================================

    print("-" * 100)

    profitable = [
        result
        for result in results
        if (
            result["profit_factor"] > 1
            and
            result["expectancy"] > 0
        )
    ]

    if profitable:

        best = max(
            profitable,
            key=lambda x:
                x["final_capital"],
        )

        print()
        print(
            "🟢 PROFITABLE ATR VARIANT FOUND"
        )

        print(
            f"Best ATR:          "
            f"{best['atr']:.1f}x"
        )

        print(
            f"Final capital:     "
            f"${best['final_capital']:.2f}"
        )

        print(
            f"Return:            "
            f"{best['return_pct']:+.2f}%"
        )

        print(
            f"Profit factor:     "
            f"{best['profit_factor']:.3f}"
        )

        print(
            f"Expectancy:        "
            f"${best['expectancy']:+.4f}"
        )

        print(
            f"Max drawdown:      "
            f"{best['max_drawdown']:.2f}%"
        )

    else:

        print()
        print(
            "🔴 NO PROFITABLE ATR VARIANT FOUND"
        )

        best = max(
            results,
            key=lambda x:
                x["final_capital"],
        )

        print(
            f"Best available ATR: "
            f"{best['atr']:.1f}x"
        )

        print(
            f"Final capital:      "
            f"${best['final_capital']:.2f}"
        )

        print(
            f"Return:             "
            f"{best['return_pct']:+.2f}%"
        )

    print("=" * 100)

    return results


# ==========================================
# MAIN
# ==========================================

def main():

    Path("logs").mkdir(
        exist_ok=True
    )

    Path("data").mkdir(
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("20→100 TRADING BOT")
    print("Strategy v2.0")
    print("ATR STOP PARAMETER TEST")
    print("=" * 60)

    all_results = []

    # ======================================
    # Markets
    # ======================================

    for symbol in SYMBOLS:

        print()
        print("-" * 60)
        print(
            f"BACKTEST: {symbol}"
        )
        print("-" * 60)

        filename = (
            symbol.replace(
                "/",
                "_",
            )
            + "_"
            + TIMEFRAME
            + ".csv"
        )

        data_file = (
            Path("data") /
            filename
        )

        # ==================================
        # DATA
        # ==================================

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
        # INDICATORS
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
        # ATR TEST
        # ==================================

        atr_results = run_atr_test(
            df=indicator_df,
            symbol=symbol,
        )

        all_results.append(
            {
                "symbol":
                    symbol,

                "results":
                    atr_results,
            }
        )

    # ======================================
    # FINAL SUMMARY
    # ======================================

    print()
    print("=" * 100)
    print("20→100 ATR TEST SUMMARY")
    print("=" * 100)

    for market in all_results:

        symbol = market[
            "symbol"
        ]

        results = market[
            "results"
        ]

        if not results:
            continue

        best = max(
            results,
            key=lambda x:
                x["final_capital"],
        )

        print()
        print(
            f"{symbol}"
        )

        print(
            f"Best ATR:          "
            f"{best['atr']:.1f}x"
        )

        print(
            f"Final capital:     "
            f"${best['final_capital']:.2f}"
        )

        print(
            f"Return:            "
            f"{best['return_pct']:+.2f}%"
        )

        print(
            f"Profit factor:     "
            f"{best['profit_factor']:.3f}"
        )

        print(
            f"Expectancy:        "
            f"${best['expectancy']:+.4f}"
        )

        print(
            f"Max drawdown:      "
            f"{best['max_drawdown']:.2f}%"
        )

    print()
    print("=" * 100)
    print(
        "ATR STOP TEST COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
