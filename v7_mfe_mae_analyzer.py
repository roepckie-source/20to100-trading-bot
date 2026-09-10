from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TRADES_FILE = Path("v7_walk_forward_oos_trades.csv")

DATA_DIR = Path("data")

OUTPUT_FILE = Path(
    "v7_walk_forward_oos_trades_mfe_mae.csv"
)

SUMMARY_FILE = Path(
    "v7_mfe_mae_summary.csv"
)


# ============================================================
# LOAD 5M MARKET DATA
# ============================================================

def load_5m(asset):
    """
    Load 5-minute OHLCV data for one asset.
    """

    path = DATA_DIR / asset

    if not path.exists():
        raise FileNotFoundError(
            f"Missing market data: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns "
            f"{sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
    )

    df = df.set_index("timestamp")

    return df


# ============================================================
# CALCULATE MFE / MAE
# ============================================================

def calc_trade_mfe_mae(
    trade,
    market,
):
    """
    Calculate MFE / MAE for one long trade.

    MFE = Maximum Favorable Excursion
    MAE = Maximum Adverse Excursion

    The calculation uses the 5-minute market data
    between entry and exit.
    """

    entry = float(
        trade["entry_price"]
    )

    entry_time = pd.Timestamp(
        trade["entry_time"]
    )

    exit_time = pd.Timestamp(
        trade["exit_time"]
    )

    # --------------------------------------------------------
    # MARKET WINDOW
    # --------------------------------------------------------

    window = market.loc[
        (market.index >= entry_time)
        &
        (market.index <= exit_time)
    ]

    if window.empty:

        return pd.Series(
            {
                "mfe_pct": np.nan,
                "mae_pct": np.nan,
                "mfe_r": np.nan,
                "mae_r": np.nan,
                "bars_observed": 0,
                "mfe_time": "",
                "mae_time": "",
            }
        )

    # --------------------------------------------------------
    # LONG POSITION
    # --------------------------------------------------------

    max_high = float(
        window["high"].max()
    )

    min_low = float(
        window["low"].min()
    )

    # --------------------------------------------------------
    # MFE / MAE %
    # --------------------------------------------------------

    mfe_pct = (
        (max_high / entry) - 1.0
    ) * 100.0

    mae_pct = (
        (min_low / entry) - 1.0
    ) * 100.0

    # --------------------------------------------------------
    # INITIAL PRICE RISK
    # --------------------------------------------------------

    initial_stop = float(
        trade["initial_stop_price"]
    )

    price_risk = abs(
        entry - initial_stop
    )

    # --------------------------------------------------------
    # MFE / MAE IN R
    # --------------------------------------------------------

    if price_risk > 0:

        mfe_r = (
            max_high - entry
        ) / price_risk

        mae_r = (
            min_low - entry
        ) / price_risk

    else:

        mfe_r = np.nan
        mae_r = np.nan

    # --------------------------------------------------------
    # TIMES
    # --------------------------------------------------------

    mfe_time = window[
        "high"
    ].idxmax()

    mae_time = window[
        "low"
    ].idxmin()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return pd.Series(
        {
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "bars_observed": len(window),
            "mfe_time": mfe_time.isoformat(),
            "mae_time": mae_time.isoformat(),
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V7-S0 MFE / MAE TRADE DIAGNOSTICS")
    print("=" * 70)

    print(
        "No strategy changes."
    )

    print(
        "No parameter optimization."
    )

    print(
        "No live trading."
    )

    print(
        "No real orders."
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD TRADE DATA
    # --------------------------------------------------------

    if not TRADES_FILE.exists():

        raise FileNotFoundError(
            f"Missing trade file: "
            f"{TRADES_FILE}"
        )

    trades = pd.read_csv(
        TRADES_FILE
    )

    print()
    print(
        f"Loaded trades: {len(trades)}"
    )

    # --------------------------------------------------------
    # USE $100 RUN ONLY
    # --------------------------------------------------------
    #
    # The strategy logic is identical for all
    # starting capitals.
    #
    # Therefore $100 is sufficient for diagnostics
    # and avoids analysing the same trades four times.
    #

    trades = trades[
        trades["starting_capital"] == 100.0
    ].copy()

    if trades.empty:

        raise RuntimeError(
            "No $100 trades found."
        )

    print(
        f"$100 trades: {len(trades)}"
    )

    # --------------------------------------------------------
    # DATETIME CONVERSION
    # --------------------------------------------------------

    trades["signal_time"] = pd.to_datetime(
        trades["signal_time"],
        utc=True,
    )

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"],
        utc=True,
    )

    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"],
        utc=True,
    )

    # --------------------------------------------------------
    # MARKET DATA CACHE
    # --------------------------------------------------------

    market_cache = {}

    enriched = []

    # --------------------------------------------------------
    # PROCESS EACH TRADE
    # --------------------------------------------------------

    for trade_number, (_, trade) in enumerate(
        trades.iterrows(),
        start=1,
    ):

        asset = trade["asset"]

        print(
            f"Analysing trade "
            f"{trade_number}/{len(trades)} "
            f"{asset}"
        )

        # ----------------------------------------------------
        # LOAD MARKET DATA ONCE PER ASSET
        # ----------------------------------------------------

        if asset not in market_cache:

            market_cache[asset] = (
                load_5m(asset)
            )

        market = market_cache[asset]

        # ----------------------------------------------------
        # MFE / MAE
        # ----------------------------------------------------

        metrics = calc_trade_mfe_mae(
            trade,
            market,
        )

        row = trade.copy()

        for key, value in metrics.items():

            row[key] = value

        enriched.append(row)

    # --------------------------------------------------------
    # CREATE RESULT DATAFRAME
    # --------------------------------------------------------

    out = pd.DataFrame(
        enriched
    )

    # --------------------------------------------------------
    # DIAGNOSTIC FLAGS
    # --------------------------------------------------------

    out["hit_1R_before_exit"] = (
        out["mfe_r"] >= 1.0
    )

    out["hit_2R_before_exit"] = (
        out["mfe_r"] >= 2.0
    )

    out["adverse_0_5R_before_exit"] = (
        out["mae_r"] <= -0.5
    )

    out["adverse_1R_before_exit"] = (
        out["mae_r"] <= -1.0
    )

    # --------------------------------------------------------
    # ADD DIAGNOSTIC CLASSIFICATION
    # --------------------------------------------------------

    def classify_trade(row):

        engine_result = str(
            row.get(
                "result_class",
                ""
            )
        ).upper()

        if "WIN" in engine_result:
            return "WIN"

        if "LOSS" in engine_result:
            return "LOSS"

        net_profit = float(
            row.get(
                "net_profit",
                0.0
            )
        )

        if net_profit > 0:
            return "WIN"

        if net_profit < 0:
            return "LOSS"

        return "FLAT"

    out["diagnostic_result"] = (
        out.apply(
            classify_trade,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # SAVE DETAILED TRADE FILE
    # --------------------------------------------------------

    out.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        out
        .groupby(
            [
                "asset",
                "diagnostic_result",
            ],
            dropna=False,
        )
        .agg(
            trades=(
                "r_multiple",
                "size",
            ),

            avg_engine_R=(
                "r_multiple",
                "mean",
            ),

            median_engine_R=(
                "r_multiple",
                "median",
            ),

            avg_MFE_R=(
                "mfe_r",
                "mean",
            ),

            median_MFE_R=(
                "mfe_r",
                "median",
            ),

            avg_MAE_R=(
                "mae_r",
                "mean",
            ),

            median_MAE_R=(
                "mae_r",
                "median",
            ),

            hit_1R_pct=(
                "hit_1R_before_exit",
                "mean",
            ),

            hit_2R_pct=(
                "hit_2R_before_exit",
                "mean",
            ),

            adverse_0_5R_pct=(
                "adverse_0_5R_before_exit",
                "mean",
            ),

            adverse_1R_pct=(
                "adverse_1R_before_exit",
                "mean",
            ),

            avg_hold_bars=(
                "held_bars",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # CONVERT RATIOS TO PERCENT
    # --------------------------------------------------------

    summary["hit_1R_pct"] *= 100.0
    summary["hit_2R_pct"] *= 100.0
    summary["adverse_0_5R_pct"] *= 100.0
    summary["adverse_1R_pct"] *= 100.0

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("MFE / MAE SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # LOSS ANALYSIS
    # ========================================================

    losses = out[
        out["diagnostic_result"] == "LOSS"
    ].copy()

    wins = out[
        out["diagnostic_result"] == "WIN"
    ].copy()

    print()
    print("=" * 70)
    print("LOSS ANALYSIS")
    print("=" * 70)

    print(
        f"Total losses: {len(losses)}"
    )

    print(
        f"Total wins:   {len(wins)}"
    )

    if not losses.empty:

        print()
        print(
            "Losses that reached +1R "
            "before exit:"
        )

        print(
            f"{losses['hit_1R_before_exit'].mean() * 100:.2f}%"
        )

        print()
        print(
            "Losses that reached +2R "
            "before exit:"
        )

        print(
            f"{losses['hit_2R_before_exit'].mean() * 100:.2f}%"
        )

        print()
        print(
            "Losses that first reached "
            "-0.5R or worse:"
        )

        print(
            f"{losses['adverse_0_5R_before_exit'].mean() * 100:.2f}%"
        )

        print()
        print(
            "Losses that reached "
            "-1R:"
        )

        print(
            f"{losses['adverse_1R_before_exit'].mean() * 100:.2f}%"
        )

    # ========================================================
    # WIN ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("WIN ANALYSIS")
    print("=" * 70)

    if not wins.empty:

        print(
            f"Winning trades reaching +1R: "
            f"{wins['hit_1R_before_exit'].mean() * 100:.2f}%"
        )

        print(
            f"Winning trades reaching +2R: "
            f"{wins['hit_2R_before_exit'].mean() * 100:.2f}%"
        )

        print(
            f"Winning trades experiencing -0.5R MAE: "
            f"{wins['adverse_0_5R_before_exit'].mean() * 100:.2f}%"
        )

    # ========================================================
    # ASSET ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("ASSET ANALYSIS")
    print("=" * 70)

    for asset in sorted(
        out["asset"].unique()
    ):

        asset_df = out[
            out["asset"] == asset
        ]

        asset_losses = asset_df[
            asset_df["diagnostic_result"]
            == "LOSS"
        ]

        asset_wins = asset_df[
            asset_df["diagnostic_result"]
            == "WIN"
        ]

        print()
        print(
            f"--- {asset} ---"
        )

        print(
            f"Trades: {len(asset_df)}"
        )

        print(
            f"Wins:   {len(asset_wins)}"
        )

        print(
            f"Losses: {len(asset_losses)}"
        )

        print(
            f"Average MFE: "
            f"{asset_df['mfe_r'].mean():.3f} R"
        )

        print(
            f"Average MAE: "
            f"{asset_df['mae_r'].mean():.3f} R"
        )

        if not asset_losses.empty:

            print(
                f"Losses reaching +1R: "
                f"{asset_losses['hit_1R_before_exit'].mean() * 100:.2f}%"
            )

            print(
                f"Losses reaching +2R: "
                f"{asset_losses['hit_2R_before_exit'].mean() * 100:.2f}%"
            )

    # ========================================================
    # EXIT REASONS
    # ========================================================

    if "exit_reason" in out.columns:

        print()
        print("=" * 70)
        print("EXIT REASONS")
        print("=" * 70)

        print(
            out["exit_reason"]
            .value_counts()
            .to_string()
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)

    print(
        f"Detailed trades:"
        f" {OUTPUT_FILE}"
    )

    print(
        f"Summary:"
        f" {SUMMARY_FILE}"
    )

    print()
    print("=" * 70)
    print("V7-S0 MFE / MAE ANALYSIS COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
