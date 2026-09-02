# ==========================================
# 20to100 Trading Bot
# Strategy V3 Backtest
# Robustness Test
# ==========================================

from pathlib import Path

import pandas as pd

from strategy.strategy_v3 import (
    StrategyV3,
)

from backtest.v3_engine import (
    V3BacktestEngine,
)


# ==========================================
# CONFIG
# ==========================================

STARTING_CAPITAL = 20.0

FEE_RATE = 0.001

SLIPPAGE_RATE = 0.0005

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
]

DATA_DIR = Path("data")


# ==========================================
# V3 PARAMETER MATRIX
# ==========================================

PARAMETERS = [

    {
        "name": "V3_A",
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian": 20,
        "atr_stop": 2.0,
    },

    {
        "name": "V3_B",
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian": 20,
        "atr_stop": 2.5,
    },

    {
        "name": "V3_C",
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian": 20,
        "atr_stop": 3.0,
    },

    {
        "name": "V3_D",
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian": 50,
        "atr_stop": 2.5,
    },

    {
        "name": "V3_E",
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian": 50,
        "atr_stop": 3.0,
    },

    {
        "name": "V3_F",
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian": 50,
        "atr_stop": 3.5,
    },

    {
        "name": "V3_G",
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian": 100,
        "atr_stop": 3.0,
    },

    {
        "name": "V3_H",
        "ema_fast": 100,
        "ema_slow": 200,
        "donchian": 100,
        "atr_stop": 3.5,
    },

]


# ==========================================
# DATA LOADER
# ==========================================

def load_data(symbol):

    filename = (
        symbol.replace(
            "/",
            "_",
        )
        + "_5m.csv"
    )

    path = (
        DATA_DIR
        /
        filename
    )

    print(
        f"Loading {path}"
    )

    df = pd.read_csv(
        path
    )

    # ----------------------------------
    # Timestamp
    # ----------------------------------

    timestamp_column = None

    for column in [
        "timestamp",
        "datetime",
        "date",
        "time",
    ]:

        if column in df.columns:

            timestamp_column = (
                column
            )

            break

    if timestamp_column is None:

        raise ValueError(
            f"No timestamp column in {path}"
        )

    df[
        timestamp_column
    ] = pd.to_datetime(
        df[
            timestamp_column
        ],
        utc=True,
    )

    df = df.set_index(
        timestamp_column
    )

    # ----------------------------------
    # Normalize columns
    # ----------------------------------

    df.columns = [
        str(c).lower()
        for c in df.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    )

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df


# ==========================================
# RESAMPLE
# ==========================================

def resample_data(
    df,
    timeframe,
):

    if timeframe == "1h":

        rule = "1h"

    elif timeframe == "4h":

        rule = "4h"

    else:

        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    result = (
        df
        .resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )
        .dropna()
    )

    return result


# ==========================================
# SPLIT
# ==========================================

def split_data(
    df,
):

    split_index = int(
        len(df) * 0.70
    )

    train = df.iloc[
        :split_index
    ].copy()

    test = df.iloc[
        split_index:
    ].copy()

    return train, test


# ==========================================
# RUN ONE TEST
# ==========================================

def run_test(
    df,
    parameters,
    timeframe,
):

    strategy = StrategyV3(

        ema_fast=parameters[
            "ema_fast"
        ],

        ema_slow=parameters[
            "ema_slow"
        ],

        donchian_period=parameters[
            "donchian"
        ],

        atr_period=14,

        atr_stop_multiplier=parameters[
            "atr_stop"
        ],
    )

    engine = V3BacktestEngine(

        starting_balance=(
            STARTING_CAPITAL
        ),

        fee_rate=FEE_RATE,

        slippage_rate=(
            SLIPPAGE_RATE
        ),
    )

    return engine.run(
        df=df,
        strategy=strategy,
        timeframe_name=timeframe,
    )


# ==========================================
# PRINT RESULT
# ==========================================

def print_result(
    symbol,
    period,
    parameters,
    result,
):

    pf = result[
        "profit_factor"
    ]

    if pf == float("inf"):

        pf_text = "INF"

    else:

        pf_text = (
            f"{pf:.3f}"
        )

    print()

    print(
        f"{symbol:<10}"
        f"{period:<10}"
        f"{parameters['name']:<8}"
        f"Final ${result['final']:>8.2f}"
        f" Return {result['return_pct']:>8.2f}%"
        f" Trades {result['trades']:>5}"
        f" Win {result['win_rate']:>6.2f}%"
        f" PF {pf_text:>6}"
        f" DD {result['max_drawdown']:>8.2f}%"
    )


# ==========================================
# MAIN
# ==========================================

def main():

    print()

    print("=" * 110)

    print(
        "20→100 TRADING BOT"
    )

    print(
        "STRATEGY V3"
    )

    print(
        "ROBUST TREND-FOLLOWING TEST"
    )

    print("=" * 110)

    print()

    print(
        "Starting capital:"
        f" ${STARTING_CAPITAL:.2f}"
    )

    print(
        "Train/Test split:"
        " 70% / 30%"
    )

    print(
        "Timeframes:"
        " 1h / 4h"
    )

    print()

    all_results = []

    # ======================================
    # MARKET
    # ======================================

    for symbol in SYMBOLS:

        print()
        print(
            "=" * 110
        )

        print(
            f"MARKET: {symbol}"
        )

        print(
            "=" * 110
        )

        raw = load_data(
            symbol
        )

        print(
            f"5m candles: {len(raw):,}"
        )

        # ==================================
        # TIMEFRAME
        # ==================================

        for timeframe in [
            "1h",
            "4h",
        ]:

            print()
            print(
                "-" * 110
            )

            print(
                f"TIMEFRAME: {timeframe}"
            )

            print(
                "-" * 110
            )

            df = resample_data(
                raw,
                timeframe,
            )

            train, test = split_data(
                df
            )

            print(
                f"Total candles:"
                f" {len(df):,}"
            )

            print(
                f"Train candles:"
                f" {len(train):,}"
            )

            print(
                f"Test candles:"
                f" {len(test):,}"
            )

            # ==============================
            # PARAMETERS
            # ==============================

            for parameters in PARAMETERS:

                # --------------------------
                # TRAIN
                # --------------------------

                train_result = run_test(
                    train,
                    parameters,
                    timeframe,
                )

                print_result(
                    symbol,
                    "TRAIN",
                    parameters,
                    train_result,
                )

                # --------------------------
                # TEST
                # --------------------------

                test_result = run_test(
                    test,
                    parameters,
                    timeframe,
                )

                print_result(
                    symbol,
                    "OOS",
                    parameters,
                    test_result,
                )

                all_results.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": parameters[
                            "name"
                        ],
                        "train_final":
                            train_result[
                                "final"
                            ],
                        "train_return":
                            train_result[
                                "return_pct"
                            ],
                        "train_pf":
                            train_result[
                                "profit_factor"
                            ],
                        "train_dd":
                            train_result[
                                "max_drawdown"
                            ],
                        "oos_final":
                            test_result[
                                "final"
                            ],
                        "oos_return":
                            test_result[
                                "return_pct"
                            ],
                        "oos_pf":
                            test_result[
                                "profit_factor"
                            ],
                        "oos_dd":
                            test_result[
                                "max_drawdown"
                            ],
                        "oos_trades":
                            test_result[
                                "trades"
                            ],
                    }
                )

    # ======================================
    # SUMMARY
    # ======================================

    print()

    print(
        "=" * 110
    )

    print(
        "V3 OOS SUMMARY"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"{'Market':<10}"
        f"{'TF':<6}"
        f"{'Strategy':<10}"
        f"{'OOS Final':>12}"
        f"{'Return':>10}"
        f"{'PF':>8}"
        f"{'DD':>10}"
        f"{'Trades':>8}"
    )

    print(
        "-" * 110
    )

    for result in all_results:

        pf = result[
            "oos_pf"
        ]

        if pf == float("inf"):

            pf_text = "INF"

        else:

            pf_text = (
                f"{pf:.3f}"
            )

        print(
            f"{result['symbol']:<10}"
            f"{result['timeframe']:<6}"
            f"{result['strategy']:<10}"
            f"${result['oos_final']:>10.2f}"
            f"{result['oos_return']:>9.2f}%"
            f"{pf_text:>8}"
            f"{result['oos_dd']:>9.2f}%"
            f"{result['oos_trades']:>8}"
        )

    # ======================================
    # QUALIFIED STRATEGIES
    # ======================================

    print()

    print(
        "=" * 110
    )

    print(
        "QUALIFIED V3 CANDIDATES"
    )

    print(
        "=" * 110
    )

    qualified = [

        r
        for r in all_results

        if (
            r["train_pf"] > 1.0
            and
            r["oos_pf"] > 1.0
            and
            r["oos_return"] > 0
            and
            r["oos_trades"] >= 10
        )
    ]

    if not qualified:

        print()

        print(
            "❌ Keine Strategie erfüllt "
            "alle Robustheitskriterien."
        )

        print()

        print(
            "Das ist ein gültiges Ergebnis."
        )

        print(
            "Wir optimieren dann nicht blind,"
        )

        print(
            "sondern wechseln die Strategieidee."
        )

    else:

        print()

        for result in qualified:

            print(
                f"✅ {result['symbol']} "
                f"{result['timeframe']} "
                f"{result['strategy']} "
                f"| OOS "
                f"+{result['oos_return']:.2f}% "
                f"| PF "
                f"{result['oos_pf']:.3f}"
            )

    # ======================================
    # SAVE CSV
    # ======================================

    output = pd.DataFrame(
        all_results
    )

    Path("logs").mkdir(
        exist_ok=True
    )

    output.to_csv(
        "logs/v3_robustness_results.csv",
        index=False,
    )

    print()

    print(
        "Results saved:"
    )

    print(
        "logs/v3_robustness_results.csv"
    )

    print()

    print(
        "=" * 110
    )

    print(
        "V3 BACKTEST COMPLETE"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":

    main()
