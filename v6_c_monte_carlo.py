from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd

from strategy.strategy_v6 import calculate_indicators
from backtest.v6_engine import V6BacktestEngine

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
VARIANT = "V6_C"

STARTING_BALANCE = 20.0
RISK_PER_TRADE = 0.01
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
ATR_STOP_MULTIPLIER = 3.0
TRAILING_ATR_MULTIPLIER = 3.0
ADX_MIN = 20.0
MAX_DAILY_LOSS = 0.05
MAX_CONSECUTIVE_LOSSES = 3
LOSS_COOLDOWN_BARS = 24
GLOBAL_MAX_DRAWDOWN = 0.20

TRAIN_MONTHS = 12
OOS_MONTHS = 3
STEP_MONTHS = 3
SIMULATIONS = 10_000
RANDOM_SEED = 20260904

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = LOG_DIR / "v6_c_monte_carlo_window_results.csv"
SUMMARY_FILE = LOG_DIR / "v6_c_monte_carlo_summary.csv"
TRADE_FILE = LOG_DIR / "v6_c_monte_carlo_trades.csv"


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "time": "timestamp", "datetime": "timestamp", "date": "timestamp",
        "open_time": "timestamp", "vol": "volume"
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    return df


def load_symbol(symbol):
    path = DATA_DIR / (symbol.replace("/", "_") + "_5m.csv")
    if not path.exists():
        raise FileNotFoundError(f"Keine CSV-Datei für {symbol}: {path}")
    print(f"\nLOADING {symbol}\nFile: {path}")
    df = normalize_columns(pd.read_csv(path))
    print(f"5m Candles : {len(df):,}")
    hourly = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"])
    print(f"1h Candles  : {len(hourly):,}")
    return hourly


def month_start(ts):
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def add_months(ts, months):
    return month_start(ts) + pd.DateOffset(months=months)


def create_windows(data):
    start = month_start(data.index[0])
    end = month_start(data.index[-1])
    windows = []
    train_start = start
    while True:
        train_end = add_months(train_start, TRAIN_MONTHS)
        oos_end = add_months(train_end, OOS_MONTHS)
        if oos_end > end:
            break
        windows.append({
            "train_start": train_start, "train_end": train_end,
            "oos_start": train_end, "oos_end": oos_end
        })
        train_start = add_months(train_start, STEP_MONTHS)
    return windows


def run_real_window(data, window):
    oos = data[(data.index >= window["oos_start"]) &
               (data.index < window["oos_end"])].copy()
    if len(oos) < 300:
        return None

    engine = V6BacktestEngine(
        starting_balance=STARTING_BALANCE,
        risk_per_trade=RISK_PER_TRADE,
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
        atr_stop_multiplier=ATR_STOP_MULTIPLIER,
        trailing_atr_multiplier=TRAILING_ATR_MULTIPLIER,
        adx_min=ADX_MIN,
        max_daily_loss=MAX_DAILY_LOSS,
        max_consecutive_losses=MAX_CONSECUTIVE_LOSSES,
        loss_cooldown_bars=LOSS_COOLDOWN_BARS,
        global_max_drawdown=GLOBAL_MAX_DRAWDOWN,
        variant=VARIANT,
    )
    result = engine.run(oos)
    return result, list(engine.trades)


def simulate_sequence(r_values, rng):
    balance = STARTING_BALANCE
    peak = balance
    max_dd = 0.0
    streak = longest = 0

    for r in rng.permutation(r_values):
        balance += balance * RISK_PER_TRADE * float(r)
        balance = max(balance, 0.0)

        if r < 0:
            streak += 1
            longest = max(longest, streak)
        elif r > 0:
            streak = 0

        peak = max(peak, balance)
        dd = ((balance - peak) / peak * 100.0) if peak else -100.0
        max_dd = min(max_dd, dd)

    return {
        "final_balance": balance,
        "return_pct": (balance / STARTING_BALANCE - 1.0) * 100.0,
        "max_drawdown_pct": max_dd,
        "longest_loss_streak": longest,
    }


def monte_carlo(r_values, rng):
    finals = np.empty(SIMULATIONS)
    returns = np.empty(SIMULATIONS)
    dds = np.empty(SIMULATIONS)
    streaks = np.empty(SIMULATIONS)

    for i in range(SIMULATIONS):
        x = simulate_sequence(r_values, rng)
        finals[i] = x["final_balance"]
        returns[i] = x["return_pct"]
        dds[i] = x["max_drawdown_pct"]
        streaks[i] = x["longest_loss_streak"]

    return {
        "simulations": SIMULATIONS,
        "trades": len(r_values),
        "median_final_balance": np.median(finals),
        "mean_final_balance": np.mean(finals),
        "p05_final_balance": np.percentile(finals, 5),
        "p95_final_balance": np.percentile(finals, 95),
        "median_return_pct": np.median(returns),
        "mean_return_pct": np.mean(returns),
        "p05_return_pct": np.percentile(returns, 5),
        "p95_return_pct": np.percentile(returns, 95),
        "median_drawdown_pct": np.median(dds),
        "worst_drawdown_pct": np.min(dds),
        "median_longest_loss_streak": np.median(streaks),
        "max_longest_loss_streak": np.max(streaks),
        "probability_hit_100_pct": np.mean(finals >= 100.0) * 100.0,
        "probability_below_10_pct": np.mean(finals < 10.0) * 100.0,
    }


def main():
    print("=" * 90)
    print("V6_C MONTE-CARLO VALIDATION")
    print("=" * 90)
    print(f"Assets       : {', '.join(SYMBOLS)}")
    print(f"Risk/Trade   : {RISK_PER_TRADE * 100:.2f}%")
    print(f"Fee          : {FEE_RATE * 100:.2f}%")
    print(f"Slippage     : {SLIPPAGE_RATE * 100:.2f}%")
    print(f"Simulations  : {SIMULATIONS:,}")
    print(f"Seed         : {RANDOM_SEED}")

    rng = np.random.default_rng(RANDOM_SEED)
    rows, trade_rows = [], []

    for symbol in SYMBOLS:
        data = calculate_indicators(load_symbol(symbol))
        windows = create_windows(data)
        print(f"{symbol}: {len(windows)} Walk-Forward Fenster")

        for n, window in enumerate(windows, 1):
            print(f"  Fenster {n}/{len(windows)}: {window['oos_start'].date()} -> {window['oos_end'].date()}")
            out = run_real_window(data, window)
            if out is None:
                continue

            real, trades = out
            r = np.array([float(t.r_multiple) for t in trades], dtype=float)
            r = r[np.isfinite(r)]
            if len(r) == 0:
                continue

            mc = monte_carlo(r, rng)
            real_return = float(real.get("return_pct", 0.0))
            real_dd = float(real.get("max_drawdown_pct", 0.0))

            print(f"    Trades={len(r)} | Real={real_return:+.3f}% | MC Median={mc['median_return_pct']:+.3f}% | P($100)={mc['probability_hit_100_pct']:.2f}%")

            rows.append({
                "symbol": symbol, "window": n,
                "oos_start": window["oos_start"],
                "oos_end": window["oos_end"],
                "real_return_pct": real_return,
                "real_max_drawdown_pct": real_dd,
                **mc
            })

            for j, t in enumerate(trades, 1):
                trade_rows.append({
                    "symbol": symbol, "window": n, "trade_number": j,
                    **asdict(t)
                })

    if not rows:
        raise RuntimeError("Keine V6_C Monte-Carlo-Ergebnisse erzeugt.")

    details = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trade_rows)
    details.to_csv(DETAIL_FILE, index=False)
    trades_df.to_csv(TRADE_FILE, index=False)

    summary = pd.DataFrame([{
        "variant": VARIANT,
        "assets": details["symbol"].nunique(),
        "windows": len(details),
        "simulations_per_window": SIMULATIONS,
        "total_real_trades": int(details["trades"].sum()),
        "positive_real_window_pct": (details["real_return_pct"] > 0).mean() * 100.0,
        "real_median_return_pct": details["real_return_pct"].median(),
        "mc_median_final_balance": details["median_final_balance"].median(),
        "mc_p05_final_balance": details["p05_final_balance"].median(),
        "mc_p95_final_balance": details["p95_final_balance"].median(),
        "mc_median_return_pct": details["median_return_pct"].median(),
        "mc_p05_return_pct": details["p05_return_pct"].median(),
        "mc_p95_return_pct": details["p95_return_pct"].median(),
        "mc_median_drawdown_pct": details["median_drawdown_pct"].median(),
        "mc_worst_drawdown_pct": details["worst_drawdown_pct"].min(),
        "median_probability_hit_100_pct": details["probability_hit_100_pct"].median(),
        "median_probability_below_10_pct": details["probability_below_10_pct"].median(),
        "median_longest_loss_streak": details["median_longest_loss_streak"].median(),
        "max_longest_loss_streak": details["max_longest_loss_streak"].max(),
    }])
    summary.to_csv(SUMMARY_FILE, index=False)

    print()
    print("=" * 90)
    print("V6_C MONTE-CARLO SUMMARY")
    print("=" * 90)
    for k, v in summary.iloc[0].items():
        print(f"{k:32}: {v}")
    print("=" * 90)
    print(f"Details : {DETAIL_FILE}")
    print(f"Trades  : {TRADE_FILE}")
    print(f"Summary : {SUMMARY_FILE}")
    print("V6_C Monte-Carlo abgeschlossen.")


if __name__ == "__main__":
    main()
