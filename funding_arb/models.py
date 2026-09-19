from dataclasses import dataclass


@dataclass
class MarketSnapshot:
    exchange: str

    spot_bid: float | None = None
    spot_ask: float | None = None

    perp_bid: float | None = None
    perp_ask: float | None = None

    funding_rate: float | None = None
    funding_interval_hours: float | None = None
    next_funding_ms: int | None = None

    timestamp_ms: int | None = None
    error: str | None = None

    @property
    def spot_mid(self):
        if self.spot_bid and self.spot_ask:
            return (self.spot_bid + self.spot_ask) / 2

        return None

    @property
    def perp_mid(self):
        if self.perp_bid and self.perp_ask:
            return (self.perp_bid + self.perp_ask) / 2

        return None

    @property
    def basis_pct(self):
        if self.spot_mid and self.perp_mid:
            return (
                self.perp_mid / self.spot_mid - 1
            ) * 100

        return None
