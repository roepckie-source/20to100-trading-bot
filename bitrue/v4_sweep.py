"""
Bitrue V4 Strategy Sweep
========================

PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO REAL ORDERS

V4 basiert auf der V3-Struktur:

EMA 9 / 21 / 55
1H EMA20 / EMA50 Higher-Timeframe-Filter
Momentum
ATR-Regime
Volume Ratio
EMA Spread
30 Minuten Cooldown
70% TRAIN / 30% OOS

Der Sweep verändert gezielt:

- ATR-Regime
- Volume-Obergrenze
- Stop-Loss ATR-Multiplikator
- Take-Profit ATR-Multiplikator

Die Auswahl erfolgt NICHT nach TRAIN allein.
OOS wird für die Bewertung priorisiert.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "BTCUSDT_5m.csv"

STARTING_CAPITAL = 100.0

RISK_PER_TRADE = 0.01

MIN_POSITION_USD = 5.0
MAX_POSITION_USD = 25.0

TAKER_FEE = 0.0006
SLIPPAGE = 0.0002

EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 55

MOMENTUM_LOOKBACK = 3
MIN_MOMENTUM_PCT = 0.15

ATR_PERIOD = 14

MIN_EMA_SPREAD_PCT = 0.15

VOLUME_MIN = 1.20

COOLDOWN_MINUTES = 30

TRAIN_RATIO = 0.70


# ============================================================
# SWEEP PARAMETERS
# ============================================================

ATR_RANGES = [
    (0.150, 0.200),
    (0.175, 0.200),
    (0.175, 0.225),
    (0.180, 0.250),
]

VOLUME_MAX_VALUES = [
    3.0,
    4.0,
    5.0,
]

SL_MULTIPLIERS = [
    1.0,
    1.2,
    1.4,
    1.6,
]

TP_MULTIPLIERS = [
    2.0,
    2.5,
]


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):

    normalized = {
        str(c).lower().replace(" ", "").replace("_", ""): c
        for c in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if key in normalized:
            return normalized[key]

    return None


def load_data(filename):

    path = Path(filename)

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset fehlt: {filename}"
        )

    df = pd.read_csv(path)

    timestamp_col = find_column(
        df,
        [
            "timestamp",
            "time",
            "datetime",
            "open_time",
            "opentime",
        ],
    )

    if timestamp_col is None:

        raise ValueError(
            "Keine Timestamp-Spalte gefunden."
        )

    rename = {
        timestamp_col: "timestamp"
    }

    for target, candidates in {

        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close"],
        "volume": ["volume"],

    }.items():

        col = find_column(
            df,
            candidates
        )

        if col is None:

            raise ValueError(
                f"Spalte fehlt: {target}"
            )

        rename[col] = target

    df = df.rename(
        columns=rename
    )

    ts = df["timestamp"]

    if pd.api.types.is_numeric_dtype(ts):

        max_ts = ts.max()

        if max_ts > 10_000_000_000:

            df["timestamp"] = pd.to_datetime(
                ts,
                unit="ms",
                utc=True,
            )

        else:

            df["timestamp"] = pd.to_datetime(
                ts,
                unit="s",
                utc=True,
            )

    else:

        df["timestamp"] = pd.to_datetime(
            ts,
            utc=True,
        )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df
        .dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    data = df.copy()

    data["ema_fast"] = (
        data["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False,
        )
        .mean()
    )

    data["ema_medium"] = (
        data["close"]
        .ewm(
            span=EMA_MEDIUM,
            adjust=False,
        )
        .mean()
    )

    data["ema_slow"] = (
        data["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
        )
        .mean()
    )

    data["momentum_pct"] = (
        (
            data["close"]
            / data["close"]
            .shift(MOMENTUM_LOOKBACK)
            - 1.0
        )
        * 100.0
    )

    previous_close = (
        data["close"]
        .shift(1)
    )

    tr1 = (
        data["high"]
        - data["low"]
    )

    tr2 = (
        data["high"]
        - previous_close
    ).abs()

    tr3 = (
        data["low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    data["atr"] = (
        true_range
        .rolling(
            ATR_PERIOD
        )
        .mean()
    )

    data["atr_pct"] = (
        data["atr"]
        / data["close"]
        * 100.0
    )

    data["volume_avg"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"]
        / data["volume_avg"]
    )

    data["ema_spread_pct"] = (
        (
            data["ema_fast"]
            - data["ema_slow"]
        ).abs()
        / data["close"]
        * 100.0
    )

    # --------------------------------------------------------
    # 1H HIGHER TIMEFRAME
    # --------------------------------------------------------

    htf = (
        data[
            [
                "timestamp",
                "close",
            ]
        ]
        .set_index("timestamp")
        .resample("1h")
        .last()
        .dropna()
    )

    htf["ema20"] = (
        htf["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    htf["ema50"] = (
        htf["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    # IMPORTANT:
    # use previous completed 1H candle
    htf["ema20_prev"] = (
        htf["ema20"]
        .shift(1)
    )

    htf["ema50_prev"] = (
        htf["ema50"]
        .shift(1)
    )

    htf["htf_bullish"] = (
        htf["ema20_prev"]
        > htf["ema50_prev"]
    )

    htf["htf_bearish"] = (
        htf["ema20_prev"]
        < htf["ema50_prev"]
    )

    htf = htf[
        [
            "ema20_prev",
            "ema50_prev",
            "htf_bullish",
            "htf_bearish",
        ]
    ]

    data = pd.merge_asof(
        data.sort_values(
            "timestamp"
        ),
        htf.sort_index(),
        left_on="timestamp",
        right_index=True,
        direction="backward",
    )

    return data


# ============================================================
# POSITION SIZE
# ============================================================

def position_size(balance):

    size = (
        balance
        * RISK_PER_TRADE
    )

    size = max(
        MIN_POSITION_USD,
        size,
    )

    size = min(
        MAX_POSITION_USD,
        size,
    )

    return size


# ============================================================
# BACKTEST ONE VARIANT
# ============================================================

def run_variant(
    data,
    atr_min,
    atr_max,
    volume_max,
    sl_mult,
    tp_mult,
):

    split_index = int(
        len(data)
        * TRAIN_RATIO
    )

    balance = STARTING_CAPITAL

    trades = []

    current_position = None

    last_entry_time = None

    cooldown = pd.Timedelta(
        minutes=COOLDOWN_MINUTES
    )

    for i in range(
        len(data)
    ):

        row = data.iloc[i]

        timestamp = row["timestamp"]

        close = float(
            row["close"]
        )

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        atr = row["atr"]

        momentum = row[
            "momentum_pct"
        ]

        volume_ratio = row[
            "volume_ratio"
        ]

        ema_fast = row[
            "ema_fast"
        ]

        ema_medium = row[
            "ema_medium"
        ]

        ema_slow = row[
            "ema_slow"
        ]

        ema_spread = row[
            "ema_spread_pct"
        ]

        htf_bullish = bool(
            row.get(
                "htf_bullish",
                False,
            )
        )

        htf_bearish = bool(
            row.get(
                "htf_bearish",
                False,
            )
        )

        if any(
            pd.isna(x)
            for x in [
                atr,
                momentum,
                volume_ratio,
                ema_fast,
                ema_medium,
                ema_slow,
                ema_spread,
            ]
        ):

            continue

        # ====================================================
        # EXISTING POSITION
        # ====================================================

        if current_position is not None:

            p = current_position

            exit_price = None
            exit_reason = None

            # Conservative:
            # STOP checked before TP

            if p["side"] == "LONG":

                if low <= p["stop_loss"]:

                    exit_price = (
                        p["stop_loss"]
                    )

                    exit_reason = (
                        "STOP_LOSS"
                    )

                elif high >= p[
                    "take_profit"
                ]:

                    exit_price = (
                        p["take_profit"]
                    )

                    exit_reason = (
                        "TAKE_PROFIT"
                    )

            else:

                if high >= p[
                    "stop_loss"
                ]:

                    exit_price = (
                        p["stop_loss"]
                    )

                    exit_reason = (
                        "STOP_LOSS"
                    )

                elif low <= p[
                    "take_profit"
                ]:

                    exit_price = (
                        p["take_profit"]
                    )

                    exit_reason = (
                        "TAKE_PROFIT"
                    )

            if exit_price is not None:

                if p["side"] == "LONG":

                    entry_exec = (
                        p["entry_price"]
                        * (
                            1.0
                            + SLIPPAGE
                        )
                    )

                    exit_exec = (
                        exit_price
                        * (
                            1.0
                            - SLIPPAGE
                        )
                    )

                    price_return = (
                        exit_exec
                        - entry_exec
                    )

                else:

                    entry_exec = (
                        p["entry_price"]
                        * (
                            1.0
                            - SLIPPAGE
                        )
                    )

                    exit_exec = (
                        exit_price
                        * (
                            1.0
                            + SLIPPAGE
                        )
                    )

                    price_return = (
                        entry_exec
                        - exit_exec
                    )

                qty = (
                    p["position_size"]
                    / p["entry_price"]
                )

                gross_pnl = (
                    price_return
                    * qty
                )

                entry_notional = (
                    p["position_size"]
                )

                exit_notional = (
                    qty
                    * exit_price
                )

                fees = (
                    entry_notional
                    * TAKER_FEE
                    + exit_notional
                    * TAKER_FEE
                )

                slippage_cost = (
                    p["position_size"]
                    * SLIPPAGE
                    + exit_notional
                    * SLIPPAGE
                )

                net_pnl = (
                    gross_pnl
                    - fees
                    - slippage_cost
                )

                balance += net_pnl

                duration = (
                    timestamp
                    - p["entry_time"]
                )

                period = (
                    "TRAIN"
                    if p["entry_index"]
                    < split_index
                    else "OOS"
                )

                trades.append(
                    {
                        "entry_time":
                            p["entry_time"],

                        "exit_time":
                            timestamp,

                        "period":
                            period,

                        "side":
                            p["side"],

                        "entry_price":
                            p["entry_price"],

                        "exit_price":
                            exit_price,

                        "position_size":
                            p["position_size"],

                        "gross_pnl":
                            gross_pnl,

                        "fees":
                            fees,

                        "slippage_cost":
                            slippage_cost,

                        "net_pnl":
                            net_pnl,

                        "duration_minutes":
                            duration.total_seconds()
                            / 60.0,

                        "exit_reason":
                            exit_reason,

                        "atr_pct":
                            p["atr_pct"],

                        "momentum_pct":
                            p["momentum_pct"],

                        "volume_ratio":
                            p["volume_ratio"],

                        "ema_spread_pct":
                            p["ema_spread_pct"],
                    }
                )

                current_position = None

                continue

            # Still in position
            continue

        # ====================================================
        # ENTRY FILTERS
        # ====================================================

        if (
            last_entry_time is not None
            and timestamp
            - last_entry_time
            < cooldown
        ):

            continue

        if (
            momentum < MIN_MOMENTUM_PCT
            and momentum > -MIN_MOMENTUM_PCT
        ):

            continue

        if (
            float(atr)
            < atr_min
            or float(atr)
            > atr_max
        ):

            continue

        if (
            float(volume_ratio)
            < VOLUME_MIN
            or float(volume_ratio)
            > volume_max
        ):

            continue

        if (
            float(ema_spread)
            < MIN_EMA_SPREAD_PCT
        ):

            continue

        side = None

        # ====================================================
        # LONG
        # ====================================================

        if (
            momentum
            >= MIN_MOMENTUM_PCT
            and ema_fast
            > ema_medium
            > ema_slow
            and close > ema_fast
            and htf_bullish
        ):

            side = "LONG"

        # ====================================================
        # SHORT
        # ====================================================

        elif (
            momentum
            <= -MIN_MOMENTUM_PCT
            and ema_fast
            < ema_medium
            < ema_slow
            and close < ema_fast
            and htf_bearish
        ):

            side = "SHORT"

        if side is None:

            continue

        # ====================================================
        # POSITION
        # ====================================================

        size = position_size(
            balance
        )

        if size <= 0:

            continue

        atr_value = float(
            atr
        )

        if side == "LONG":

            stop_loss = (
                close
                - atr_value
                * sl_mult
            )

            take_profit = (
                close
                + atr_value
                * tp_mult
            )

        else:

            stop_loss = (
                close
                + atr_value
                * sl_mult
            )

            take_profit = (
                close
                - atr_value
                * tp_mult
            )

        current_position = {
            "side":
                side,

            "entry_time":
                timestamp,

            "entry_index":
                i,

            "entry_price":
                close,

            "position_size":
                size,

            "stop_loss":
                stop_loss,

            "take_profit":
                take_profit,

            "atr_pct":
                float(
                    row["atr_pct"]
                ),

            "momentum_pct":
                float(
                    momentum
                ),

            "volume_ratio":
                float(
                    volume_ratio
                ),

            "ema_spread_pct":
                float(
                    ema_spread
                ),
        }

        last_entry_time = (
            timestamp
        )

    trades_df = pd.DataFrame(
        trades
    )

    if trades_df.empty:

        return {
            "trades": 0,
            "train_trades": 0,
            "oos_trades": 0,
            "train_net": 0.0,
            "oos_net": 0.0,
            "total_net": 0.0,
            "oos_winrate": 0.0,
            "oos_profit_factor": 0.0,
            "oos_avg_trade": 0.0,
            "oos_max_drawdown": 0.0,
        }

    return calculate_stats(
        trades_df
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(
    trades_df
):

    total = len(
        trades_df
    )

    train = trades_df[
        trades_df["period"]
        == "TRAIN"
    ]

    oos = trades_df[
        trades_df["period"]
        == "OOS"
    ]

    def net(df):

        return (
            float(
                df["net_pnl"].sum()
            )
            if not df.empty
            else 0.0
        )

    def winrate(df):

        if df.empty:
            return 0.0

        return (
            (
                df["net_pnl"] > 0
            ).mean()
            * 100.0
        )

    def profit_factor(df):

        if df.empty:
            return 0.0

        wins = df.loc[
            df["net_pnl"] > 0,
            "net_pnl",
        ].sum()

        losses = -df.loc[
            df["net_pnl"] < 0,
            "net_pnl",
        ].sum()

        if losses <= 0:
            return 999.0

        return (
            wins
            / losses
        )

    def max_drawdown(df):

        if df.empty:
            return 0.0

        equity = (
            STARTING_CAPITAL
            + df["net_pnl"]
            .cumsum()
        )

        peak = (
            equity
            .cummax()
        )

        drawdown = (
            equity
            - peak
        )

        return float(
            drawdown.min()
        )

    return {
        "trades":
            total,

        "train_trades":
            len(train),

        "oos_trades":
            len(oos),

        "train_net":
            net(train),

        "oos_net":
            net(oos),

        "total_net":
            net(trades_df),

        "train_winrate":
            winrate(train),

        "oos_winrate":
            winrate(oos),

        "oos_profit_factor":
            profit_factor(oos),

        "oos_avg_trade":
            (
                float(
                    oos["net_pnl"]
                    .mean()
                )
                if not oos.empty
                else 0.0
            ),

        "oos_max_drawdown":
            max_drawdown(oos),
    }


# ============================================================
# SCORING
# ============================================================

def score_result(r):

    """
    OOS is deliberately weighted more heavily.

    We do NOT simply choose the highest return.

    Requirements for a meaningful candidate:

    - OOS trades >= 20
    - OOS net positive
    - OOS average trade positive
    - reasonable profit factor
    """

    score = 0.0

    if r["oos_trades"] < 20:

        return -999999.0

    # OOS net
    score += (
        r["oos_net"]
        * 100.0
    )

    # OOS average trade
    score += (
        r["oos_avg_trade"]
        * 5000.0
    )

    # Profit factor
    score += (
        min(
            r["oos_profit_factor"],
            3.0,
        )
        * 10.0
    )

    # OOS win rate
    score += (
        r["oos_winrate"]
        * 0.10
    )

    # Penalize drawdown
    score += (
        r["oos_max_drawdown"]
        * 2.0
    )

    # Penalize severe TRAIN/OOS divergence
    train = r["train_net"]
    oos = r["oos_net"]

    if train > 0 and oos < 0:

        score -= 20.0

    return score


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("BITRUE V4 AUTOMATIC STRATEGY SWEEP")
    print("=" * 80)
    print()
    print("PAPER ONLY")
    print("NO API")
    print("NO LIVE TRADING")
    print("NO REAL ORDERS")
    print()

    data = load_data(
        INPUT_FILE
    )

    print(
        f"Candles: {len(data):,}"
    )

    print(
        f"Start:   {data['timestamp'].iloc[0]}"
    )

    print(
        f"End:     {data['timestamp'].iloc[-1]}"
    )

    print()

    print(
        "Berechne Indikatoren..."
    )

    data = calculate_indicators(
        data
    )

    print(
        "Indikatoren fertig."
    )

    print()

    combinations = list(
        itertools.product(
            ATR_RANGES,
            VOLUME_MAX_VALUES,
            SL_MULTIPLIERS,
            TP_MULTIPLIERS,
        )
    )

    print(
        f"Anzahl Varianten: "
        f"{len(combinations)}"
    )

    print()

    results = []

    for number, params in enumerate(
        combinations,
        start=1,
    ):

        (
            atr_range,
            volume_max,
            sl_mult,
            tp_mult,
        ) = params

        atr_min, atr_max = (
            atr_range
        )

        result = run_variant(
            data,
            atr_min,
            atr_max,
            volume_max,
            sl_mult,
            tp_mult,
        )

        result.update(
            {
                "atr_min":
                    atr_min,

                "atr_max":
                    atr_max,

                "volume_max":
                    volume_max,

                "sl_mult":
                    sl_mult,

                "tp_mult":
                    tp_mult,
            }
        )

        result["score"] = (
            score_result(
                result
            )
        )

        results.append(
            result
        )

        if (
            number % 10 == 0
            or number
            == len(combinations)
        ):

            print(
                f"Progress: "
                f"{number}/"
                f"{len(combinations)}"
            )

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        "bitrue_v4_sweep_results.csv",
        index=False,
    )

    # ========================================================
    # CANDIDATES
    # ========================================================

    candidates = results_df[
        (results_df["oos_trades"] >= 20)
        & (
            results_df["oos_net"]
            > 0
        )
        & (
            results_df["oos_avg_trade"]
            > 0
        )
    ].copy()

    candidates = candidates.head(
        10
    )

    candidates.to_csv(
        "bitrue_v4_candidates.csv",
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    with open(
        "bitrue_v4_sweep_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "BITRUE V4 SWEEP REPORT\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"Varianten: "
            f"{len(results_df)}\n"
        )

        f.write(
            f"Datensatz: "
            f"{len(data):,} Candles\n\n"
        )

        f.write(
            "TOP 10 NACH SCORE\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        columns = [
            "score",
            "atr_min",
            "atr_max",
            "volume_max",
            "sl_mult",
            "tp_mult",
            "trades",
            "train_net",
            "oos_trades",
            "oos_net",
            "oos_winrate",
            "oos_profit_factor",
            "oos_avg_trade",
            "oos_max_drawdown",
        ]

        f.write(
            results_df[
                columns
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "POSITIVE OOS CANDIDATES\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        if candidates.empty:

            f.write(
                "KEINE Variante erfüllt "
                "alle positiven OOS-Kriterien.\n"
            )

        else:

            f.write(
                candidates[
                    columns
                ].to_string(
                    index=False
                )
            )

            f.write(
                "\n"
            )

        f.write(
            "\n\n"
        )

        f.write(
            "AUSWAHLREGEL:\n"
        )

        f.write(
            "OOS wird gegenüber TRAIN priorisiert.\n"
        )

        f.write(
            "Keine Auswahl ausschließlich nach TRAIN.\n"
        )

        f.write(
            "Mindestens 20 OOS-Trades für Kandidaten.\n"
        )

        f.write(
            "\nPAPER ONLY - NO LIVE TRADING\n"
        )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 80)
    print("TOP 10 V4 VARIANTEN")
    print("=" * 80)

    display_columns = [
        "score",
        "atr_min",
        "atr_max",
        "volume_max",
        "sl_mult",
        "tp_mult",
        "trades",
        "train_net",
        "oos_trades",
        "oos_net",
        "oos_winrate",
        "oos_profit_factor",
        "oos_avg_trade",
        "oos_max_drawdown",
    ]

    print(
        results_df[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("POSITIVE OOS KANDIDATEN")
    print("=" * 80)

    if candidates.empty:

        print(
            "Keine Variante ist im OOS positiv."
        )

    else:

        print(
            candidates[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()

    print(
        "Dateien erzeugt:"
    )

    print(
        "  bitrue_v4_sweep_results.csv"
    )

    print(
        "  bitrue_v4_candidates.csv"
    )

    print(
        "  bitrue_v4_sweep_report.txt"
    )

    print()

    print(
        "=" * 80
    )

    print(
        "V4 SWEEP ABGESCHLOSSEN"
    )

    print(
        "PAPER ONLY"
    )

    print(
        "NO LIVE TRADING"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
