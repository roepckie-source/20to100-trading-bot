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
            float(starting_balance)
        )

        self.consecutive_losses = 0

    # ======================================
    # Maximum risk
    # ======================================

    def max_risk_amount(
        self,
        balance: float,
    ) -> float:

        return (
            balance *
            self.config.risk_per_trade
        )

    # ======================================
    # Daily loss
    # ======================================

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

    # ======================================
    # Trade result
    # ======================================

    def record_trade(
        self,
        profit: float,
    ) -> None:

        if profit < 0:

            self.consecutive_losses += 1

        else:

            self.consecutive_losses = 0

    # ======================================
    # Loss streak
    # ======================================

    def loss_streak_limit_reached(
        self,
    ) -> bool:

        return (
            self.consecutive_losses >=
            self.config.max_consecutive_losses
        )

    # ======================================
    # New day
    # ======================================

    def reset_daily(
        self,
        balance: float,
    ) -> None:

        self.daily_start_balance = (
            float(balance)
        )

        self.consecutive_losses = 0

    # ======================================
    # Position size
    # ======================================

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_price: float,
        fee_rate: float = 0.001,
    ) -> float:

        """
        Calculate quantity while respecting BOTH:

        1. Maximum risk per trade
        2. Available capital including entry fee
        """

        if balance <= 0:
            return 0.0

        if entry_price <= 0:
            return 0.0

        if stop_price >= entry_price:
            return 0.0

        # ----------------------------------
        # Stop distance
        # ----------------------------------

        stop_distance = (
            entry_price -
            stop_price
        )

        if stop_distance <= 0:
            return 0.0

        # ----------------------------------
        # Maximum monetary risk
        # ----------------------------------

        risk_amount = (
            balance *
            self.config.risk_per_trade
        )

        # ----------------------------------
        # Quantity based on risk
        # ----------------------------------

        risk_quantity = (
            risk_amount /
            stop_distance
        )

        # ----------------------------------
        # Maximum position value
        #
        # Position + fee must fit inside
        # available balance.
        # ----------------------------------

        max_position_value = (
            balance /
            (1 + fee_rate)
        )

        capital_quantity = (
            max_position_value /
            entry_price
        )

        # ----------------------------------
        # Use the smaller quantity.
        # ----------------------------------

        quantity = min(
            risk_quantity,
            capital_quantity,
        )

        return max(
            quantity,
            0.0,
        )
