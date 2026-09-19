from dataclasses import asdict

from .config import (
    DEFAULT_SPOT_FEE,
    DEFAULT_PERP_FEE,
    DEFAULT_SLIPPAGE,
    MIN_NET_EDGE_PCT,
)

from .exchanges import FETCHERS


def scan_once():

    rows = []

    for name, fetcher in FETCHERS.items():

        snapshot = fetcher()

        row = asdict(snapshot)

        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        if (
            snapshot.error
            or snapshot.funding_rate is None
        ):

            row["status"] = "ERROR"
            row["net_edge_pct"] = None

            rows.append(row)

            continue

        # ----------------------------------------------------
        # FUNDING
        # ----------------------------------------------------

        funding_pct = (
            abs(snapshot.funding_rate)
            * 100
        )

        # ----------------------------------------------------
        # COST MODEL
        # ----------------------------------------------------

        round_trip_cost_pct = (

            2
            * (
                DEFAULT_SPOT_FEE
                + DEFAULT_PERP_FEE
                + DEFAULT_SLIPPAGE
            )
            * 100

        )

        # ----------------------------------------------------
        # NET EDGE
        # ----------------------------------------------------

        net_edge_pct = (
            funding_pct
            - round_trip_cost_pct
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if net_edge_pct >= MIN_NET_EDGE_PCT:

            status = "WATCH"

        else:

            status = "NO_EDGE"

        # ----------------------------------------------------
        # HEDGE DIRECTION
        # ----------------------------------------------------

        if snapshot.funding_rate > 0:

            direction = (
                "SPOT_LONG_PERP_SHORT"
            )

        else:

            direction = (
                "SPOT_SHORT_PERP_LONG"
            )

        row.update(

            {

                "funding_income_pct_per_interval":
                    funding_pct,

                "round_trip_cost_pct":
                    round_trip_cost_pct,

                "net_edge_pct":
                    net_edge_pct,

                "status":
                    status,

                "direction":
                    direction,

            }

        )

        rows.append(row)

    return rows


def format_rows(rows):

    output = []

    for row in rows:

        if row["status"] == "ERROR":

            output.append(

                f'{row["exchange"]:7} '
                f'ERROR: {row["error"]}'

            )

            continue

        spot_mid = (
            row["spot_bid"]
            + row["spot_ask"]
        ) / 2

        perp_mid = (
            row["perp_bid"]
            + row["perp_ask"]
        ) / 2

        output.append(

            f'{row["exchange"]:7} '

            f'spot={spot_mid:.2f} '

            f'perp={perp_mid:.2f} '

            f'basis={row["basis_pct"]:+.4f}% '

            f'funding='
            f'{row["funding_rate"] * 100:+.5f}% '

            f'net='
            f'{row["net_edge_pct"]:+.4f}% '

            f'{row["status"]}'

        )

    return "\n".join(output)
