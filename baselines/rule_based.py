import numpy as np


class RuleBasedPricer:
    """
    A deterministic pricing policy based on inventory thresholds.

    Strategy:
    - Inventory > 70% remaining → cut price by 15% (clear stock)
    - Inventory < 30% remaining → raise price by 10% (scarcity premium)
    - Otherwise → hold at the reference price

    obs format: [inventory_ratio, days_remaining, last_price_norm, demand_sensitivity]
    price_levels: the list of available price buckets
    """

    def __init__(
        self,
        price_levels: np.ndarray,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
        discount_factor: float = 0.85,
        premium_factor: float = 1.10,
    ):
        self.price_levels = price_levels
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.discount_factor = discount_factor
        self.premium_factor = premium_factor
        self._mid_price = price_levels[len(price_levels) // 2]

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Matches the SB3 predict() signature."""
        obs = np.atleast_2d(obs)
        actions = []
        for o in obs:
            inventory_ratio = o[1]
            if inventory_ratio > self.high_threshold:
                target = self._mid_price * self.discount_factor
            elif inventory_ratio < self.low_threshold:
                target = self._mid_price * self.premium_factor
            else:
                target = self._mid_price

            # Find the closest price bucket
            action = int(np.argmin(np.abs(self.price_levels - target)))
            actions.append(action)

        return np.array(actions), None