# ==========================================
# 20to100 Trading Bot
# A/B/C Strategy Backtest
# ==========================================

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    SYMBOLS,
    TIMEFRAME,
    STARTING_CAPITAL,
    HISTORICAL_DAYS,
    FEE_RATE,
    SLIPPAGE_RATE,
    RISK_PER_TRADE,
    ATR_STOP_MULTIPLIER,
    PARTIAL_EXIT_1_R,
    PARTIAL_EXIT_1_SIZE,
    PARTIAL_EXIT_2_R,
    PARTIAL_EXIT_2_SIZE,
    TRAILING_ATR_MULTIPLIER,
    MAX_TRADE_MINUTES,
    MAX_DAILY_LOSS,
    MAX_CONSECUTIVE_LOSSES,
    LOSS_COOLDOWN_MINUTES,
    TRADE_COOLDOWN_MINUTES,
)

from data.data_manager import (
    fetch_ohlcv,
    save_data,
)

from strategy.indicators import (
    calculate_indicators,
)

from strategy.signals import (
    evaluate_conditions,
)

from risk.risk_manager import (
    RiskManager,
    RiskConfig,
)


# ==========================================
# STRATEGIES
# ==========================================

STRATEGIES = {

    "A_ORIGINAL": [
        "trend",
        "price_above_ema50",
        "rsi",
        "rsi_rising",
        "pullback",
        "confirmation",
        "volume",
    ],

    "B_TREND_CONFIRM_VOLUME": [
        "trend",
        "confirmation",
        "volume",
    ],

    "C_TREND_PULLBACK_CONFIRM_VOLUME": [
        "trend",
        "pullback",
        "confirmation",
        "volume",
    ],
}


# ==========================================
# POSITION
# ==========================================

@dataclass
class Position:

    strategy: str
    symbol: str

    entry_time: object

    entry_price: float
    quantity: float

    initial_stop: float
    current_stop: float

    risk_per_unit: float

    remaining_quantity: float
    highest_price: float

    realized_profit: float = 0.0
    total_fees: float = 0.0
    total_slippage_cost: float = 0.0

    partial_1_done: bool = False
    partial_2_done: bool = False


# ==========================================
# TRADE
# ==========================================

@dataclass
class Trade:

    strategy: str
    symbol: str

    entry_time: object
    exit_time: object

    entry_price: float
    final_exit_price: float

    initial_quantity: float

    gross_profit: float
    fees: float
    slippage_cost: float
    net_profit: float

    exit_reason: str
    r_multiple: float


# ==========================================
# ENGINE
# ==========================================

class StrategyBacktest:

    def __init__(
        self,
        strategy_name: str,
        conditions: list[str],
        starting_balance: float,
        symbol: str,
    ):

        self.strategy_name = strategy_name
        self.conditions = conditions
        self.symbol = symbol

        self.starting_balance = float(
            starting_balance
        )

        self.balance = float(
            starting_balance
        )

        self.position: Optional[
            Position
        ] = None

        self.trades = []

        self.equity_curve = []

        self.risk_manager = RiskManager(
            starting_balance,
            RiskConfig(
                risk_per_trade=RISK_PER_TRADE,
                max_daily_loss=MAX_DAILY_LOSS,
                max_consecutive_losses=(
                    MAX_CONSECUTIVE_LOSSES
                ),
            ),
        )

        self.cooldown_until = None
        self.loss_cooldown_until = None

        self.current_day = None

        self.signal_count = 0
        self.entry_count = 0

    # ======================================
    # DAY RESET
    # ======================================

    def update_day(
        self,
        timestamp,
    ):

        day = timestamp.date()

        if (
            self.current_day is None
            or day != self.current_day
        ):

            self.current_day = day

            self.risk_manager.reset_daily(
                self.balance
            )

    # ======================================
    # SIGNAL
    # ======================================

    def check_entry(
        self,
        df,
    ) -> bool:

        if len(df) < 60:
            return False

        row = df.iloc[-1]

        try:

            conditions = evaluate_conditions(
                df,
                spread=None,
            )

        except Exception:

            return False

        for condition in self.conditions:

            if not conditions.get(
                condition,
                False,
            ):

                return False

        return True

    # ======================================
    # BUY
    # ======================================

    def execute_buy(
        self,
        timestamp,
        close,
        atr,
    ):

        if self.position is not None:
            return False

        if pd.isna(atr) or atr <= 0:
            return False

        if self.balance <= 0:
            return False

        entry_price = (
            float(close)
            *
            (1 + SLIPPAGE_RATE)
        )

        stop_distance = (
            float(atr)
            *
            ATR_STOP_MULTIPLIER
        )

        if stop_distance <= 0:
            return False

        stop_price = (
            entry_price -
            stop_distance
        )

        if stop_price <= 0:
            return False

        # ----------------------------------
        # Risk based position size
        # ----------------------------------

        quantity = (
            self.risk_manager
            .calculate_position_size(
                balance=self.balance,
                entry_price=entry_price,
                stop_price=stop_price,
                fee_rate=FEE_RATE,
            )
        )

        if quantity <= 0:
            return False

        position_value = (
            quantity *
            entry_price
        )

        entry_fee = (
            position_value *
            FEE_RATE
        )

        total_cost = (
            position_value +
            entry_fee
        )

        # ----------------------------------
        # Safety adjustment
        # ----------------------------------

        if total_cost > self.balance:

            max_position_value = (
                self.balance /
                (1 + FEE_RATE)
            )

            quantity = (
                max_position_value /
                entry_price
            )

            position_value = (
                quantity *
                entry_price
            )

            entry_fee = (
                position_value *
                FEE_RATE
            )

            total_cost = (
                position_value +
                entry_fee
            )

        if quantity <= 0:
            return False

        if total_cost > self.balance:
            return False

        # ----------------------------------
        # Open position
        # ----------------------------------

        self.balance -= total_cost

        self.position = Position(

            strategy=self.strategy_name,

            symbol=self.symbol,

            entry_time=timestamp,

            entry_price=entry_price,

            quantity=quantity,

            initial_stop=stop_price,

            current_stop=stop_price,

            risk_per_unit=stop_distance,

            remaining_quantity=quantity,

            highest_price=entry_price,

            total_fees=entry_fee,
        )

        self.entry_count += 1

        return True

    # ======================================
    # SELL
    # ======================================

    def execute_sell(
        self,
        position,
        market_price,
        quantity,
    ):

        if quantity <= 0:
            return

        quantity = min(
            quantity,
            position.remaining_quantity,
        )

        if quantity <= 0:
            return

        exit_price = (
            float(market_price)
            *
            (1 - SLIPPAGE_RATE)
        )

        gross_value = (
            quantity *
            exit_price
        )

        exit_fee = (
            gross_value *
            FEE_RATE
        )

        self.balance += (
            gross_value -
            exit_fee
        )

        entry_value = (
            quantity *
            position.entry_price
        )

        entry_fee = (
            entry_value *
            FEE_RATE
        )

        gross_profit = (
            (
                exit_price -
                position.entry_price
            )
            *
            quantity
        )

        net_profit = (
            gross_profit
            -
            entry_fee
            -
            exit_fee
        )

        entry_slippage = (
            quantity
            *
            position.entry_price
            *
            SLIPPAGE_RATE
        )

        exit_slippage = (
            quantity
            *
            float(market_price)
            *
            SLIPPAGE_RATE
        )

        position.realized_profit += (
            net_profit
        )

        position.total_fees += (
            entry_fee +
            exit_fee
        )

        position.total_slippage_cost += (
            entry_slippage +
            exit_slippage
        )

        position.remaining_quantity -= (
            quantity
        )

    # ======================================
    # CLOSE
    # ======================================

    def close_position(
        self,
        market_price,
        timestamp,
        reason,
    ):

        if self.position is None:
            return

        position = self.position

        if position.remaining_quantity > 0:

            self.execute_sell(
                position,
                market_price,
                position.remaining_quantity,
            )

        initial_risk = (
            position.risk_per_unit
            *
            position.quantity
        )

        if initial_risk > 0:

            r_multiple = (
                position.realized_profit
                /
                initial_risk
            )

        else:

            r_multiple = 0.0

        gross_profit = (
            position.realized_profit
            +
            position.total_fees
        )

        trade = Trade(

            strategy=position.strategy,

            symbol=position.symbol,

            entry_time=position.entry_time,

            exit_time=timestamp,

            entry_price=position.entry_price,

            final_exit_price=market_price,

            initial_quantity=position.quantity,

            gross_profit=gross_profit,

            fees=position.total_fees,

            slippage_cost=(
                position.total_slippage_cost
            ),

            net_profit=(
                position.realized_profit
            ),

            exit_reason=reason,

            r_multiple=r_multiple,
        )

        self.trades.append(
            trade
        )

        self.risk_manager.record_trade(
            trade.net_profit
        )

        # ----------------------------------
        # Cooldown
        # ----------------------------------

        if (
            self.risk_manager
            .loss_streak_limit_reached()
        ):

            self.loss_cooldown_until = (
                timestamp
                +
                pd.Timedelta(
                    minutes=
                    LOSS_COOLDOWN_MINUTES
                )
            )

        else:

            self.cooldown_until = (
                timestamp
                +
                pd.Timedelta(
                    minutes=
                    TRADE_COOLDOWN_MINUTES
                )
            )

        self.position = None

    # ======================================
    # MANAGE POSITION
    # ======================================

    def manage_position(
        self,
        row,
    ):

        if self.position is None:
            return

        position = self.position

        timestamp = row.name

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        atr = row.get(
            "atr_14",
            None,
        )

        if atr is None or pd.isna(atr):
            atr = 0.0

        # ----------------------------------
        # Highest price
        # ----------------------------------

        if high > position.highest_price:

            position.highest_price = high

        # ----------------------------------
        # Initial stop
        # ----------------------------------

        if low <= position.current_stop:

            self.close_position(
                market_price=position.current_stop,
                timestamp=timestamp,
                reason="STOP_LOSS",
            )

            return

        # ==================================
        # R CALCULATION
        # ==================================

        risk_distance = (
            position.risk_per_unit
        )

        if risk_distance <= 0:
            return

        current_r = (
            (
                close -
                position.entry_price
            )
            /
            risk_distance
        )

        # ==================================
        # PARTIAL EXIT 1
        # ==================================

        if (
            not position.partial_1_done
            and
            current_r >= PARTIAL_EXIT_1_R
        ):

            quantity = (
                position.quantity
                *
                PARTIAL_EXIT_1_SIZE
            )

            self.execute_sell(
                position,
                close,
                quantity,
            )

            position.partial_1_done = True

            # Move stop to breakeven
            position.current_stop = max(
                position.current_stop,
                position.entry_price,
            )

        # ==================================
        # PARTIAL EXIT 2
        # ==================================

        if (
            not position.partial_2_done
            and
            current_r >= PARTIAL_EXIT_2_R
        ):

            quantity = (
                position.quantity
                *
                PARTIAL_EXIT_2_SIZE
            )

            self.execute_sell(
                position,
                close,
                quantity,
            )

            position.partial_2_done = True

        # ==================================
        # TRAILING STOP
        # ==================================

        if (
            position.partial_1_done
            and
            atr > 0
        ):

            trailing_stop = (
                position.highest_price
                -
                (
                    atr
                    *
                    TRAILING_ATR_MULTIPLIER
                )
            )

            if trailing_stop > position.current_stop:

                position.current_stop = (
                    trailing_stop
                )

        # ==================================
        # TAKE PROFIT
        # ==================================

        if current_r >= 3.0:

            self.close_position(
                market_price=close,
                timestamp=timestamp,
                reason="TAKE_PROFIT",
            )

            return

        # ==================================
        # TIME STOP
        # ==================================

        elapsed = (
            timestamp -
            position.entry_time
        )

        if (
            elapsed.total_seconds()
            >=
            MAX_TRADE_MINUTES * 60
        ):

            self.close_position(
                market_price=close,
                timestamp=timestamp,
                reason="TIME_STOP",
            )

            return

    # ======================================
    # RUN
    # ======================================

    def run(
        self,
        df,
    ):

        for i in range(
            60,
            len(df),
        ):

            history = df.iloc[
                : i + 1
            ]

            row = history.iloc[-1]

            timestamp = row.name

            self.update_day(
                timestamp
            )

            # ----------------------------------
            # Manage open position first
            # ----------------------------------

            if self.position is not None:

                self.manage_position(
                    row
                )

            # ----------------------------------
            # Entry checks
            # ----------------------------------

            if self.position is None:

                if (
                    self.cooldown_until
                    is not None
                    and
                    timestamp <
                    self.cooldown_until
                ):

                    pass

                elif (
                    self.loss_cooldown_until
                    is not None
                    and
                    timestamp <
                    self.loss_cooldown_until
                ):

                    pass

                elif (
                    self.risk_manager
                    .daily_loss_limit_reached(
                        self.balance
                    )
                ):

                    pass

                else:

                    if self.check_entry(
                        history
                    ):

                        self.signal_count += 1

                        atr = row.get(
                            "atr_14",
                            None,
                        )

                        self.execute_buy(
                            timestamp,
                            float(row["close"]),
                            atr,
                        )

            # ----------------------------------
            # Equity
            # ----------------------------------

            equity = self.balance

            if self.position is not None:

                equity += (
                    self.position
                    .remaining_quantity
                    *
                    float(row["close"])
                )

            self.equity_curve.append(
                {
                    "timestamp":
                        timestamp,

                    "equity":
                        equity,
                }
            )

        # ==================================
        # Close at end
        # ==================================

        if self.position is not None:

            last = df.iloc[-1]

            self.close_position(
                market_price=float(
                    last["close"]
                ),

                timestamp=last.name,

                reason="END_OF_TEST",
            )

        return self


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(
    engine,
):

    trades = engine.trades

    if not trades:

        return {
            "final":
                engine.balance,

            "return_pct":
                (
                    engine.balance /
                    engine.starting_balance
                    - 1
                ) * 100,

            "trades": 0,
            "wins": 0,
            "losses": 0,

            "win_rate": 0.0,

            "average_win": 0.0,
            "average_loss": 0.0,

            "profit_factor": 0.0,

            "expectancy": 0.0,

            "average_r": 0.0,

            "max_drawdown": 0.0,

            "fees": 0.0,
            "slippage": 0.0,
        }

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

    gross_profit = sum(
        winners
    )

    gross_loss = abs(
        sum(losers)
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = 0.0

    win_rate = (
        len(winners)
        /
        len(profits)
        *
        100
    )

    average_win = (
        sum(winners) /
        len(winners)
        if winners
        else 0.0
    )

    average_loss = (
        sum(losers) /
        len(losers)
        if losers
        else 0.0
    )

    expectancy = (
        sum(profits) /
        len(profits)
    )

    average_r = (
        sum(
            t.r_multiple
            for t in trades
        )
        /
        len(trades)
    )

    # ======================================
    # DRAW DOWN
    # ======================================

    equity = pd.DataFrame(
        engine.equity_curve
    )

    if equity.empty:

        max_drawdown = 0.0

    else:

        peak = (
            equity["equity"]
            .cummax()
        )

        drawdown = (
            (
                equity["equity"]
                /
                peak
            )
            - 1
        )

        max_drawdown = (
            abs(
                drawdown.min()
            )
            * 100
        )

    total_fees = sum(
        t.fees
        for t in trades
    )

    total_slippage = sum(
        t.slippage_cost
        for t in trades
    )

    return {

        "final":
            engine.balance,

        "return_pct":
            (
                engine.balance /
                engine.starting_balance
                - 1
            ) * 100,

        "trades":
            len(trades),

        "wins":
            len(winners),

        "losses":
            len(losers),

        "win_rate":
            win_rate,

        "average_win":
            average_win,

        "average_loss":
            average_loss,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "average_r":
            average_r,

        "max_drawdown":
            max_drawdown,

        "fees":
            total_fees,

        "slippage":
            total_slippage,
    }


# ==========================================
# PRINT STRATEGY RESULT
# ==========================================

def print_result(
    engine,
):

    metrics = calculate_metrics(
        engine
    )

    print()
    print("=" * 70)

    print(
        f"STRATEGY RESULT: "
        f"{engine.strategy_name}"
    )

    print("=" * 70)

    print(
        f"Starting capital:  "
        f"${engine.starting_balance:.2f}"
    )

    print(
        f"Final capital:     "
        f"${metrics['final']:.2f}"
    )

    print(
        f"Profit/Loss:       "
        f"${metrics['final'] - engine.starting_balance:+.2f}"
    )

    print(
        f"Return:            "
        f"{metrics['return_pct']:+.2f}%"
    )

    print("-" * 70)

    print(
        f"Signals:           "
        f"{engine.signal_count}"
    )

    print(
        f"Trades:            "
        f"{metrics['trades']}"
    )

    print(
        f"Winners:           "
        f"{metrics['wins']}"
    )

    print(
        f"Losers:            "
        f"{metrics['losses']}"
    )

    print(
        f"Win rate:          "
        f"{metrics['win_rate']:.2f}%"
    )

    print("-" * 70)

    print(
        f"Average win:       "
        f"${metrics['average_win']:+.4f}"
    )

    print(
        f"Average loss:      "
        f"${metrics['average_loss']:+.4f}"
    )

    print(
        f"Profit factor:     "
        f"{metrics['profit_factor']:.3f}"
    )

    print(
        f"Expectancy:        "
        f"${metrics['expectancy']:+.4f}"
    )

    print(
        f"Average R:         "
        f"{metrics['average_r']:+.3f}"
    )

    print(
        f"Max drawdown:      "
        f"{metrics['max_drawdown']:.2f}%"
    )

    print("-" * 70)

    print(
        f"Fees:              "
        f"${metrics['fees']:.4f}"
    )

    print(
        f"Slippage:          "
        f"${metrics['slippage']:.4f}"
    )

    print(
        f"Progress to $100:  "
        f"{metrics['final']:.2f}%"
    )

    status = (
        "🟢 PROFIT"
        if metrics["final"] >
        engine.starting_balance
        else
        "🔴 LOSS"
    )

    print(
        f"Status:            "
        f"{status}"
    )

    print("=" * 70)

    return metrics


# ==========================================
# SAVE TRADES
# ==========================================

def save_trades(
    engine,
):

    Path("logs").mkdir(
        exist_ok=True
    )

    if not engine.trades:
        return

    rows = []

    for trade in engine.trades:

        rows.append(
            {
                "strategy":
                    trade.strategy,

                "symbol":
                    trade.symbol,

                "entry_time":
                    trade.entry_time,

                "exit_time":
                    trade.exit_time,

                "entry_price":
                    trade.entry_price,

                "final_exit_price":
                    trade.final_exit_price,

                "quantity":
                    trade.initial_quantity,

                "gross_profit":
                    trade.gross_profit,

                "fees":
                    trade.fees,

                "slippage":
                    trade.slippage_cost,

                "net_profit":
                    trade.net_profit,

                "exit_reason":
                    trade.exit_reason,

                "r_multiple":
                    trade.r_multiple,
            }
        )

    filename = (
        engine.symbol.replace(
            "/",
            "_",
        )
        +
        "_"
        +
        engine.strategy_name
        +
        "_trades.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        Path("logs") /
        filename,
        index=False,
    )


# ==========================================
# SAVE EQUITY
# ==========================================

def save_equity(
    engine,
):

    Path("logs").mkdir(
        exist_ok=True
    )

    if not engine.equity_curve:
        return

    filename = (
        engine.symbol.replace(
            "/",
            "_",
        )
        +
        "_"
        +
        engine.strategy_name
        +
        "_equity.csv"
    )

    pd.DataFrame(
        engine.equity_curve
    ).to_csv(
        Path("logs") /
        filename,
        index=False,
    )


# ==========================================
# DATA LOADER
# ==========================================

def load_market_data(
    symbol,
):

    Path("data").mkdir(
        exist_ok=True
    )

    filename = (
        symbol.replace(
            "/",
            "_",
        )
        +
        "_"
        +
        TIMEFRAME
        +
        ".csv"
    )

    data_file = (
        Path("data") /
        filename
    )

    if data_file.exists():

        print(
            f"Using cached data: "
            f"{data_file}"
        )

        return pd.read_csv(
            data_file,
            index_col="timestamp",
            parse_dates=True,
        )

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

    return df


# ==========================================
# RUN STRATEGY
# ==========================================

def run_strategy(
    df,
    symbol,
    strategy_name,
    conditions,
):

    print()
    print("-" * 70)

    print(
        f"RUNNING: "
        f"{strategy_name}"
    )

    print("-" * 70)

    print(
        "Entry:"
    )

    print(
        " + ".join(
            conditions
        )
    )

    engine = StrategyBacktest(

        strategy_name=
            strategy_name,

        conditions=
            conditions,

        starting_balance=
            STARTING_CAPITAL,

        symbol=
            symbol,
    )

    engine.run(
        df
    )

    metrics = print_result(
        engine
    )

    save_trades(
        engine
    )

    save_equity(
        engine
    )

    return metrics


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
    print("=" * 70)

    print(
        "20→100 TRADING BOT"
    )

    print(
        "A/B/C STRATEGY BACKTEST"
    )

    print("=" * 70)

    print()
    print(
        f"Starting capital: "
        f"${STARTING_CAPITAL:.2f}"
    )

    print(
        f"Timeframe: "
        f"{TIMEFRAME}"
    )

    print(
        f"Historical days: "
        f"{HISTORICAL_DAYS}"
    )

    # ======================================
    # SUMMARY STORAGE
    # ======================================

    portfolio_results = []

    # ======================================
    # MARKET LOOP
    # ======================================

    for symbol in SYMBOLS:

        print()
        print("=" * 70)

        print(
            f"MARKET: {symbol}"
        )

        print("=" * 70)

        df = load_market_data(
            symbol
        )

        print(
            f"Candles: {len(df)}"
        )

        print(
            "Calculating indicators..."
        )

        df = calculate_indicators(
            df
        )

        market_results = []

        # ==================================
        # STRATEGY LOOP
        # ==================================

        for (
            strategy_name,
            conditions
        ) in STRATEGIES.items():

            metrics = run_strategy(
                df=df,
                symbol=symbol,
                strategy_name=
                    strategy_name,
                conditions=
                    conditions,
            )

            market_results.append(
                {
                    "symbol":
                        symbol,

                    "strategy":
                        strategy_name,

                    **metrics,
                }
            )

        portfolio_results.extend(
            market_results
        )

    # ======================================
    # COMPARISON
    # ======================================

    print()
    print("=" * 100)

    print(
        "20→100 STRATEGY COMPARISON"
    )

    print("=" * 100)

    print()

    print(
        f"{'Market':<12}"
        f"{'Strategy':<38}"
        f"{'Final':>10}"
        f"{'Return':>10}"
        f"{'Trades':>9}"
        f"{'Win %':>9}"
        f"{'PF':>8}"
        f"{'Expect':>10}"
        f"{'Max DD':>10}"
    )

    print("-" * 100)

    for result in portfolio_results:

        print(
            f"{result['symbol']:<12}"
            f"{result['strategy']:<38}"
            f"${result['final']:>8.2f}"
            f"{result['return_pct']:>9.2f}%"
            f"{result['trades']:>9}"
            f"{result['win_rate']:>8.2f}%"
            f"{result['profit_factor']:>8.3f}"
            f"${result['expectancy']:>9.4f}"
            f"{result['max_drawdown']:>9.2f}%"
        )

    print("-" * 100)

    # ======================================
    # BEST STRATEGY PER MARKET
    # ======================================

    print()
    print(
        "BEST STRATEGY PER MARKET"
    )

    print("-" * 100)

    for symbol in SYMBOLS:

        results = [
            r
            for r in portfolio_results
            if r["symbol"] == symbol
        ]

        if not results:
            continue

        best = max(
            results,
            key=lambda r:
                r["final"],
        )

        print()
        print(
            f"{symbol}"
        )

        print(
            f"Winner:            "
            f"{best['strategy']}"
        )

        print(
            f"Final capital:     "
            f"${best['final']:.2f}"
        )

        print(
            f"Return:            "
            f"{best['return_pct']:+.2f}%"
        )

        print(
            f"Trades:            "
            f"{best['trades']}"
        )

        print(
            f"Profit factor:     "
            f"{best['profit_factor']:.3f}"
        )

        print(
            f"Expectancy:        "
            f"${best['expectancy']:+.4f}"
        )

    # ======================================
    # OVERALL STRATEGY AVERAGE
    # ======================================

    print()
    print("=" * 100)

    print(
        "20→100 OVERALL STRATEGY SUMMARY"
    )

    print("=" * 100)

    for strategy_name in STRATEGIES:

        results = [
            r
            for r in portfolio_results
            if r["strategy"] ==
            strategy_name
        ]

        if not results:
            continue

        average_final = (
            sum(
                r["final"]
                for r in results
            )
            /
            len(results)
        )

        total_trades = sum(
            r["trades"]
            for r in results
        )

        average_return = (
            sum(
                r["return_pct"]
                for r in results
            )
            /
            len(results)
        )

        average_pf = (
            sum(
                r["profit_factor"]
                for r in results
            )
            /
            len(results)
        )

        average_expectancy = (
            sum(
                r["expectancy"]
                for r in results
            )
            /
            len(results)
        )

        print()
        print(
            strategy_name
        )

        print(
            f"Average final:     "
            f"${average_final:.2f}"
        )

        print(
            f"Average return:    "
            f"{average_return:+.2f}%"
        )

        print(
            f"Total trades:      "
            f"{total_trades}"
        )

        print(
            f"Average PF:        "
            f"{average_pf:.3f}"
        )

        print(
            f"Average expectancy:"
            f" ${average_expectancy:+.4f}"
        )

    # ======================================
    # FINAL WINNER
    # ======================================

    print()
    print("=" * 100)

    strategy_totals = {}

    for strategy_name in STRATEGIES:

        results = [
            r
            for r in portfolio_results
            if r["strategy"] ==
            strategy_name
        ]

        strategy_totals[
            strategy_name
        ] = sum(
            r["final"]
            for r in results
        )

    if strategy_totals:

        winner = max(
            strategy_totals,
            key=strategy_totals.get,
        )

        print(
            "🏆 CURRENT WINNER"
        )

        print()
        print(
            winner
        )

        print(
            f"Combined final capital: "
            f"${strategy_totals[winner]:.2f}"
        )

    print("=" * 100)

    print()
    print(
        "A/B/C BACKTEST COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
