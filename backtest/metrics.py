# ==========================================
# 20to100 Trading Bot
# Backtest Metrics
# ==========================================

import math


def calculate_metrics(
    trades,
    equity_curve,
    starting_balance,
):

    ending_balance = (
        float(equity_curve["equity"].iloc[-1])
        if not equity_curve.empty
        else starting_balance
    )

    profits = [
        float(t.net_profit)
        for t in trades
    ]

    winners = [
        p for p in profits
        if p > 0
    ]

    losers = [
        p for p in profits
        if p < 0
    ]

    number_of_trades = len(profits)

    win_rate = (
        len(winners) /
        number_of_trades
        if number_of_trades
        else 0
    )

    average_win = (
        sum(winners) /
        len(winners)
        if winners
        else 0
    )

    average_loss = (
        sum(losers) /
        len(losers)
        if losers
        else 0
    )

    gross_profit = sum(winners)

    gross_loss = abs(
        sum(losers)
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    elif gross_profit > 0:

        profit_factor = math.inf

    else:

        profit_factor = 0.0

    expectancy = (
        win_rate * average_win
        +
        (
            1 - win_rate
        ) * average_loss
    )

    total_return_pct = (
        (
            ending_balance /
            starting_balance
        )
        - 1
    ) * 100

    equity = equity_curve["equity"]

    peak = equity.cummax()

    drawdown = (
        equity - peak
    ) / peak

    max_drawdown_pct = (
        abs(drawdown.min()) *
        100
    )

    longest_loss_streak = 0
    current_streak = 0

    for profit in profits:

        if profit < 0:

            current_streak += 1

            longest_loss_streak = max(
                longest_loss_streak,
                current_streak,
            )

        else:

            current_streak = 0

    average_r = (
        sum(
            t.r_multiple
            for t in trades
        )
        / number_of_trades
        if number_of_trades
        else 0
    )

    return {
        "starting_balance":
            starting_balance,

        "ending_balance":
            ending_balance,

        "total_return_pct":
            total_return_pct,

        "number_of_trades":
            number_of_trades,

        "winning_trades":
            len(winners),

        "losing_trades":
            len(losers),

        "win_rate_pct":
            win_rate * 100,

        "average_win":
            average_win,

        "average_loss":
            average_loss,

        "gross_profit":
            gross_profit,

        "gross_loss":
            gross_loss,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "average_r":
            average_r,

        "max_drawdown_pct":
            max_drawdown_pct,

        "largest_win":
            max(winners)
            if winners
            else 0,

        "largest_loss":
            min(losers)
            if losers
            else 0,

        "longest_losing_streak":
            longest_loss_streak,
    }


def print_metrics(metrics):

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
