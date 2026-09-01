# ==========================================
# 20to100 Trading Bot
# Backtest Metrics - Strategy v1.0
# ==========================================

import math
from typing import Any

import pandas as pd


def calculate_metrics(
    trades: list,
    equity_curve: pd.DataFrame,
    starting_balance: float,
) -> dict[str, Any]:
    """
    Calculate the main performance statistics
    of a backtest.
    """

    if equity_curve.empty:
        return {
            "starting_balance": starting_balance,
            "ending_balance": starting_balance,
            "total_return_pct": 0.0,
            "number_of_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "longest_losing_streak": 0,
        }

    ending_balance = float(
        equity_curve["equity"].iloc[-1]
    )

    # --------------------------------------
    # Return
    # --------------------------------------

    total_return_pct = (
        (ending_balance / starting_balance) - 1
    ) * 100

    # --------------------------------------
    # Trade profits
    # --------------------------------------

    profits = [
        float(trade.net_profit)
        for trade in trades
    ]

    winning = [
        profit
        for profit in profits
        if profit > 0
    ]

    losing = [
        profit
        for profit in profits
        if profit < 0
    ]

    number_of_trades = len(profits)
    winning_trades = len(winning)
    losing_trades = len(losing)

    # --------------------------------------
    # Win rate
    # --------------------------------------

    if number_of_trades > 0:
        win_rate_pct = (
            winning_trades /
            number_of_trades
        ) * 100
    else:
        win_rate_pct = 0.0

    # --------------------------------------
    # Average win / loss
    # --------------------------------------

    average_win = (
        sum(winning) / len(winning)
        if winning
        else 0.0
    )

    average_loss = (
        sum(losing) / len(losing)
        if losing
        else 0.0
    )

    # --------------------------------------
    # Profit Factor
    # --------------------------------------

    gross_profit = sum(winning)
    gross_loss = abs(sum(losing))

    if gross_loss > 0:
        profit_factor = (
            gross_profit /
            gross_loss
        )
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    # --------------------------------------
    # Expectancy per trade
    # --------------------------------------

    if number_of_trades > 0:

        win_probability = (
            winning_trades /
            number_of_trades
        )

        loss_probability = (
            losing_trades /
            number_of_trades
        )

        expectancy = (
            win_probability * average_win
            +
            loss_probability * average_loss
        )

    else:
        expectancy = 0.0

    # --------------------------------------
    # Maximum drawdown
    # --------------------------------------

    equity = equity_curve["equity"]

    running_max = equity.cummax()

    drawdown = (
        equity -
        running_max
    ) / running_max

    max_drawdown_pct = (
        abs(drawdown.min()) * 100
    )

    # --------------------------------------
    # Largest win / loss
    # --------------------------------------

    largest_win = (
        max(winning)
        if winning
        else 0.0
    )

    largest_loss = (
        min(losing)
        if losing
        else 0.0
    )

    # --------------------------------------
    # Longest losing streak
    # --------------------------------------

    longest_losing_streak = 0
    current_losing_streak = 0

    for profit in profits:

        if profit < 0:

            current_losing_streak += 1

            longest_losing_streak = max(
                longest_losing_streak,
                current_losing_streak,
            )

        else:

            current_losing_streak = 0

    # --------------------------------------
    # R statistics
    # --------------------------------------

    r_values = [
        float(trade.r_multiple)
        for trade in trades
    ]

    average_r = (
        sum(r_values) / len(r_values)
        if r_values
        else 0.0
    )

    # --------------------------------------
    # Result
    # --------------------------------------

    return {
        "starting_balance": starting_balance,
        "ending_balance": ending_balance,
        "total_return_pct": total_return_pct,

        "number_of_trades": number_of_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate_pct,

        "average_win": average_win,
        "average_loss": average_loss,

        "gross_profit": gross_profit,
        "gross_loss": gross_loss,

        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "average_r": average_r,

        "max_drawdown_pct": max_drawdown_pct,

        "largest_win": largest_win,
        "largest_loss": largest_loss,

        "longest_losing_streak":
            longest_losing_streak,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """
    Print a readable backtest report.
    """

    print()
    print("=" * 60)
    print("20→100 BACKTEST REPORT")
    print("=" * 60)

    print(
        f"Starting balance: "
        f"${metrics['starting_balance']:.2f}"
    )

    print(
        f"Ending balance:   "
        f"${metrics['ending_balance']:.2f}"
    )

    print(
        f"Total return:     "
        f"{metrics['total_return_pct']:.2f}%"
    )

    print("-" * 60)

    print(
        f"Trades:           "
        f"{metrics['number_of_trades']}"
    )

    print(
        f"Winners:          "
        f"{metrics['winning_trades']}"
    )

    print(
        f"Losers:           "
        f"{metrics['losing_trades']}"
    )

    print(
        f"Win rate:         "
        f"{metrics['win_rate_pct']:.2f}%"
    )

    print("-" * 60)

    print(
        f"Average win:      "
        f"${metrics['average_win']:.4f}"
    )

    print(
        f"Average loss:     "
        f"${metrics['average_loss']:.4f}"
    )

    print(
        f"Profit factor:    "
        f"{metrics['profit_factor']:.3f}"
    )

    print(
        f"Expectancy:       "
        f"${metrics['expectancy']:.4f}"
    )

    print(
        f"Average R:        "
        f"{metrics['average_r']:.3f}"
    )

    print("-" * 60)

    print(
        f"Max drawdown:     "
        f"{metrics['max_drawdown_pct']:.2f}%"
    )

    print(
        f"Largest win:      "
        f"${metrics['largest_win']:.4f}"
    )

    print(
        f"Largest loss:     "
        f"${metrics['largest_loss']:.4f}"
    )

    print(
        f"Longest loss run: "
        f"{metrics['longest_losing_streak']}"
    )

    print("=" * 60)
    print()
