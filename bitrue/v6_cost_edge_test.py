"""
Bitrue V6 - Cost & Edge Test

PURPOSE
-------
Test whether the V5 strategy has enough gross edge to survive
realistic trading costs.

IMPORTANT
---------
PAPER / BACKTEST ONLY
NO LIVE TRADING
NO API KEYS
NO WALLET
NO REAL ORDERS

Input:
    bitrue_v5_long_only_trades.csv
    bitrue_v5_short_only_trades.csv
    bitrue_v5_both_trades.csv

V6 does NOT optimize strategy parameters.

It tests:
    - gross edge before costs
    - fee sensitivity
    - slippage sensitivity
    - combined cost sensitivity
    - break-even fee
    - break-even slippage
    - cost burden
"""

from pathlib import Path
import pandas as pd


STARTING_CAPITAL = 100.0

# Current assumed Bitrue costs
CURRENT_FEE = 0.0006       # 0.06% per side
CURRENT_SLIPPAGE = 0.0002  # 0.02% per side

FILES = {
    "LONG_ONLY": Path("bitrue_v5_long_only_trades.csv"),
    "SHORT_ONLY": Path("bitrue_v5_short_only_trades.csv"),
    "BOTH": Path("bitrue_v5_both_trades.csv"),
}


def load_oos(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "period",
        "position_size",
        "gross_pnl",
        "fees",
        "net_pnl",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )

    df = df[df["period"] == "OOS"].copy()

    if df.empty:
        raise ValueError(
            f"{path} contains no OOS trades."
        )

    return df


def basic_stats(df: pd.DataFrame) -> dict:
    trades = len(df)

    gross = df["gross_pnl"].sum()
    fees = df["fees"].sum()
    net = df["net_pnl"].sum()

    total_position = df["position_size"].sum()

    avg_gross = df["gross_pnl"].mean()
    avg_net = df["net_pnl"].mean()

    gross_edge_pct = (
        gross / total_position * 100
        if total_position > 0
        else 0.0
    )

    current_fee_rate = (
        fees / (2.0 * total_position)
        if total_position > 0
        else 0.0
    )

    # Fee per side at which total OOS P&L reaches zero
    breakeven_fee = (
        gross / (2.0 * total_position)
        if total_position > 0
        else 0.0
    )

    # Same concept for slippage
    breakeven_slippage = breakeven_fee

    return {
        "trades": trades,
        "gross": gross,
        "fees": fees,
        "net": net,
        "avg_gross": avg_gross,
        "avg_net": avg_net,
        "total_position": total_position,
        "gross_edge_pct": gross_edge_pct,
        "current_fee_rate": current_fee_rate,
        "breakeven_fee": breakeven_fee,
        "breakeven_slippage": breakeven_slippage,
    }


def net_with_costs(
    df: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
) -> float:

    total_position = df["position_size"].sum()
    gross = df["gross_pnl"].sum()

    total_cost_rate = (
        fee_rate + slippage_rate
    )

    round_trip_cost = (
        2.0
        * total_position
        * total_cost_rate
    )

    return gross - round_trip_cost


def print_basic(name: str, stats: dict) -> None:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Trades:                 {stats['trades']}")
    print(
        f"Gross P&L:              "
        f"${stats['gross']:.6f}"
    )
    print(
        f"Current fees:           "
        f"${stats['fees']:.6f}"
    )
    print(
        f"Current net P&L:        "
        f"${stats['net']:.6f}"
    )
    print(
        f"Average gross/trade:    "
        f"${stats['avg_gross']:.6f}"
    )
    print(
        f"Average net/trade:      "
        f"${stats['avg_net']:.6f}"
    )
    print(
        f"Gross edge:             "
        f"{stats['gross_edge_pct']:.6f}%"
    )
    print(
        f"Observed fee/side:      "
        f"{stats['current_fee_rate'] * 100:.6f}%"
    )
    print(
        f"Break-even fee/side:    "
        f"{stats['breakeven_fee'] * 100:.6f}%"
    )


def fee_sensitivity(
    name: str,
    df: pd.DataFrame,
    fee_levels: list[float],
) -> list[dict]:

    rows = []

    for fee in fee_levels:

        net = net_with_costs(
            df=df,
            fee_rate=fee,
            slippage_rate=0.0,
        )

        rows.append(
            {
                "strategy": name,
                "fee_per_side_pct": fee * 100,
                "slippage_per_side_pct": 0.0,
                "net_pnl": net,
            }
        )

    return rows


def slippage_sensitivity(
    name: str,
    df: pd.DataFrame,
    slippage_levels: list[float],
) -> list[dict]:

    rows = []

    for slippage in slippage_levels:

        net = net_with_costs(
            df=df,
            fee_rate=0.0,
            slippage_rate=slippage,
        )

        rows.append(
            {
                "strategy": name,
                "fee_per_side_pct": 0.0,
                "slippage_per_side_pct": slippage * 100,
                "net_pnl": net,
            }
        )

    return rows


def combined_sensitivity(
    name: str,
    df: pd.DataFrame,
    fee_levels: list[float],
    slippage_levels: list[float],
) -> list[dict]:

    rows = []

    for fee in fee_levels:

        for slippage in slippage_levels:

            net = net_with_costs(
                df=df,
                fee_rate=fee,
                slippage_rate=slippage,
            )

            rows.append(
                {
                    "strategy": name,
                    "fee_per_side_pct": fee * 100,
                    "slippage_per_side_pct": slippage * 100,
                    "net_pnl": net,
                }
            )

    return rows


def main() -> None:

    print("=" * 70)
    print("BITRUE V6 - COST & EDGE TEST")
    print("=" * 70)
    print()
    print("PAPER / BACKTEST ONLY")
    print("NO LIVE TRADING")
    print("NO API KEYS")
    print("NO WALLET")
    print("NO REAL ORDERS")
    print()

    all_results = []
    fee_results = []
    slippage_results = []
    combined_results = []

    for name, path in FILES.items():

        df = load_oos(path)

        stats = basic_stats(df)

        print_basic(name, stats)

        all_results.append(
            {
                "strategy": name,
                **stats,
            }
        )

        # Fee sensitivity
        fee_levels = [
            0.0000,
            0.0001,
            0.0002,
            0.0003,
            0.0004,
            0.0005,
            0.0006,
        ]

        fee_results.extend(
            fee_sensitivity(
                name,
                df,
                fee_levels,
            )
        )

        # Slippage sensitivity
        slippage_levels = [
            0.0000,
            0.00005,
            0.00010,
            0.00015,
            0.00020,
            0.00025,
        ]

        slippage_results.extend(
            slippage_sensitivity(
                name,
                df,
                slippage_levels,
            )
        )

        # Combined sensitivity
        combined_results.extend(
            combined_sensitivity(
                name,
                df,
                fee_levels,
                slippage_levels,
            )
        )

    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------

    pd.DataFrame(all_results).to_csv(
        "bitrue_v6_cost_edge_summary.csv",
        index=False,
    )

    pd.DataFrame(fee_results).to_csv(
        "bitrue_v6_fee_sensitivity.csv",
        index=False,
    )

    pd.DataFrame(slippage_results).to_csv(
        "bitrue_v6_slippage_sensitivity.csv",
        index=False,
    )

    pd.DataFrame(combined_results).to_csv(
        "bitrue_v6_combined_cost_sensitivity.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Current cost scenario
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("CURRENT COST SCENARIO")
    print("=" * 70)

    for name, path in FILES.items():

        df = load_oos(path)

        gross = df["gross_pnl"].sum()

        net = net_with_costs(
            df,
            CURRENT_FEE,
            CURRENT_SLIPPAGE,
        )

        print()
        print(name)
        print(
            f"Gross:                    ${gross:.6f}"
        )
        print(
            f"Fee:                      "
            f"{CURRENT_FEE * 100:.4f}% / side"
        )
        print(
            f"Slippage:                 "
            f"{CURRENT_SLIPPAGE * 100:.4f}% / side"
        )
        print(
            f"Net after both costs:     "
            f"${net:.6f}"
        )

    # ----------------------------------------------------------
    # Break-even summary
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("BREAK-EVEN ANALYSIS")
    print("=" * 70)

    for name, path in FILES.items():

        df = load_oos(path)

        stats = basic_stats(df)

        print()
        print(name)
        print(
            f"Gross edge/trade:         "
            f"${stats['avg_gross']:.6f}"
        )
        print(
            f"Gross edge/notional:      "
            f"{stats['gross_edge_pct']:.6f}%"
        )
        print(
            f"Break-even fee/side:      "
            f"{stats['breakeven_fee'] * 100:.6f}%"
        )
        print(
            f"Current fee/side:         "
            f"{CURRENT_FEE * 100:.6f}%"
        )

        if stats["breakeven_fee"] > CURRENT_FEE:
            print("Fee margin:               POSITIVE")
        else:
            print("Fee margin:               NEGATIVE")

    # ----------------------------------------------------------
    # Final safety / interpretation
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("V6 INTERPRETATION")
    print("=" * 70)

    print()
    print(
        "V6 does NOT optimize parameters."
    )
    print(
        "V6 only tests whether the existing V5 edge "
        "survives trading costs."
    )

    print()
    print(
        "Important:"
    )
    print(
        "- Historical source is Binance BTCUSDT Futures."
    )
    print(
        "- Execution venue assumption remains Bitrue."
    )
    print(
        "- No live trading is enabled."
    )
    print(
        "- No API keys are used."
    )
    print(
        "- No wallet is used."
    )
    print(
        "- No real orders are placed."
    )

    print()
    print("=" * 70)
    print("V6 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
