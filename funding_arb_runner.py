from funding_arb.scanner import (
    scan_once,
    format_rows,
)

from funding_arb.config import (
    PAPER_ONLY,
    LIVE_TRADING,
    REAL_ORDERS,
)


def main():

    print("=" * 70)
    print("BTC FUNDING / BASIS ARBITRAGE")
    print("PAPER SCANNER")
    print("=" * 70)

    print(f"PAPER_ONLY={PAPER_ONLY}")
    print(f"LIVE_TRADING={LIVE_TRADING}")
    print(f"REAL_ORDERS={REAL_ORDERS}")

    print()

    # Absolute safety check
    assert PAPER_ONLY
    assert not LIVE_TRADING
    assert not REAL_ORDERS

    rows = scan_once()

    print(format_rows(rows))


if __name__ == "__main__":
    main()
