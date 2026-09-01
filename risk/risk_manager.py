# ==========================================
# 20to100 Trading Bot
# Risk Manager - Strategy v1.0
# ==========================================

from dataclasses import dataclass


@dataclass
class RiskConfig:
    """
    Central risk-management configuration.
    """

    risk_per_trade: float = 0.015
    max_daily_loss: float = 0.05
    max_consecutive_losses: int = 3
    loss_cooldown_minutes: int = 120


class RiskManager:
    """
    Controls position risk and trading restrictions.
    """

    def __init__(
        self,
        starting_balance: float,
        config: RiskConfig | None = None
    ):
        self.starting_balance = starting_balance
        self.config = config or RiskConfig()

        self.daily_start_balance = starting_balance
        self.consecutive_losses = 0

    # --------------------------------------
    # Maximum risk per trade
    # --------------------------------------

    def max_risk_amount(
        self,
        current_balance: float
    ) -> float:
        """
        Maximum amount we are allowed to lose
        on a single trade.
        """

        return (
            current_balance *
            self.config.risk_per_trade
        )

    # --------------------------------------
    # Maximum daily loss
    # --------------------------------------

    def max_daily_loss_amount(self) -> float:
        """
        Maximum allowed loss for the current day.
        """

        return (
            self.daily_start_balance *
            self.config.max_daily_loss
        )

    # --------------------------------------
    # Check daily loss limit
    # --------------------------------------

    def daily_loss_limit_reached(
        self,
        current_balance: float
    ) -> bool:
        """
        Returns True if the daily loss limit
        has been reached.
        """

        loss = (
            self.daily_start_balance -
            current_balance
        )

        return loss >= self.max_daily_loss_amount()

    # --------------------------------------
    # Record trade result
    # --------------------------------------

    def record_trade(self, profit: float) -> None:
        """
        Update consecutive loss counter.
        """

        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    # --------------------------------------
    # Loss streak check
    # --------------------------------------

    def loss_streak_limit_reached(self) -> bool:
        """
        Returns True after the configured number
        of consecutive losing trades.
        """

        return (
            self.consecutive_losses >=
            self.config.max_consecutive_losses
        )

    # --------------------------------------
    # Position sizing
    # --------------------------------------

    def calculate_position_size(
        self,
        current_balance: float,
        entry_price: float,
        stop_price: float
    ) -> float:
        """
        Calculate position size from account risk.

        The result is limited to the available balance.
        No leverage is used.
        """

        if current_balance <= 0:
            return 0.0

        if entry_price <= 0:
            return 0.0

        if stop_price >= entry_price:
            return 0.0

        stop_distance = (
            entry_price -
            stop_price
        )

        risk_amount = self.max_risk_amount(
            current_balance
        )

        if stop_distance <= 0:
            return 0.0

        # Amount of asset we can buy while
        # respecting the maximum risk.
        quantity = (
            risk_amount /
            stop_distance
        )

        # Never use more capital than available.
        max_quantity = (
            current_balance /
            entry_price
        )

        quantity = min(
            quantity,
            max_quantity
        )

        return quantity

    # --------------------------------------
    # Position value
    # --------------------------------------

    def position_value(
        self,
        quantity: float,
        entry_price: float
    ) -> float:
        """
        Calculate position value.
        """

        return quantity * entry_price

    # --------------------------------------
    # Actual monetary risk
    # --------------------------------------

    def actual_risk(
        self,
        quantity: float,
        entry_price: float,
        stop_price: float
    ) -> float:
        """
        Calculate the actual monetary risk
        of a position.
        """

        if quantity <= 0:
            return 0.0

        if stop_price >= entry_price:
            return 0.0

        return (
            quantity *
            (entry_price - stop_price)
        )

    # --------------------------------------
    # Reset daily statistics
    # --------------------------------------

    def reset_daily(
        self,
        current_balance: float
    ) -> None:
        """
        Start a new trading day.
        """

        self.daily_start_balance = current_balance
        self.consecutive_losses = 0
