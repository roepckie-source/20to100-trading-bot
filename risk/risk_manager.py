# ==========================================
# 20to100 Trading Bot
# Risk Manager
# ==========================================

from dataclasses import dataclass


@dataclass
class RiskConfig:

    risk_per_trade: float = 0.015
    max_daily_loss: float = 0.05
    max_consecutive_losses: int = 3


class RiskManager:

    def __init__(
        self,
        starting_balance: float,
        config: RiskConfig | None = None,
    ):

        self.config = (
            config or RiskConfig()
        )

        self.daily_start_balance = (
            starting_balance
        )

        self.consecutive_losses = 0

    # --------------------------------------
    # Maximum risk
    # --------------------------------------

    def max_risk_amount(
        self,
        balance: float,
    ) -> float:

        return (
            balance *
            self.config.risk_per_trade
        )

    # --------------------------------------
    # Daily loss
    # --------------------------------------

    def daily_loss_limit_amount(self) -> float:

        return (
            self.daily_start_balance *
            self.config.max_daily_loss
        )

    def daily_loss_limit_reached(
        self,
        balance: float,
    ) -> bool:

        loss = (
            self.daily_start_balance -
            balance
        )

        return (
            loss >=
            self.daily_loss_limit_amount()
        )

    # --------------------------------------
    # Trade result
    # --------------------------------------

    def record_trade(
        self,
        profit: float,
    ) -> None:

        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    # --------------------------------------
    # Loss streak
    # --------------------------------------

    def loss_streak_limit_reached(self) -> bool:

        return (
            self.consecutive_losses >=
            self.config.max_consecutive_losses
        )

    # --------------------------------------
    # New day
    # --------------------------------------

    def reset_daily(
        self,
        balance: float,
    ) -> None:

        self.daily_start_balance = balance
        self.consecutive_losses = 0

    # --------------------------------------
    # Position size
    # --------------------------------------

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_price: float,
    ) -> float:

        if balance <= 0:
            return 0.0

        if entry_price <= 0:
            return 0.0

        if stop_price >= entry_price:
            return 0.0

        stop_distance = (
            entry_price -
            stop_price
        )

        if stop_distance <= 0:
            return 0.0

        risk_amount = (
            self.max_risk_amount(
                balance
            )
        )

        quantity = (
            risk_amount /
            stop_distance
        )

        # Never exceed available capital.
        max_quantity = (
            balance /
            entry_price
        )

        return min(
            quantity,
            max_quantity,
        )
