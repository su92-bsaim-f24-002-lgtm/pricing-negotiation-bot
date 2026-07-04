import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.buyer_simulator import simulate_sales
from environment.demand_model import (
    MarketParams,
    sample_market_params,
)


class PricingEnv(gym.Env):
    """
    Custom Gymnasium environment for dynamic pricing.

    Observation:
        [
            remaining_days_ratio,
            inventory_ratio,
            normalized_reference_price,
            normalized_cost,
        ]

    Action:
        Discrete price bucket.

    Reward:
        Scaled daily profit.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        self.rng = np.random.default_rng(
            config.get("seed", 42)
        )

        self.market: MarketParams | None = None

        self.day = 0
        self.inventory = 0

        self.min_price = config["min_price"]
        self.max_price = config["max_price"]
        self.n_price_buckets = config["n_price_buckets"]

        self.max_inventory = config["max_inventory"]
        self.episode_days = config["episode_days"]

        self.action_space = spaces.Discrete(
            self.n_price_buckets
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    def _action_to_price(self, action: int) -> float:
        """
        Convert a discrete action into a selling price.
        """

        if self.n_price_buckets == 1:
            return self.min_price

        return self.min_price + (
            action
            * (self.max_price - self.min_price)
            / (self.n_price_buckets - 1)
        )

    def _get_observation(self) -> np.ndarray:
        """
        Return the normalized observation.
        """

        assert self.market is not None

        remaining_days = (
            self.episode_days - self.day
        ) / self.episode_days

        inventory_ratio = (
            self.inventory / self.max_inventory
        )

        normalized_reference_price = (
            (self.market.reference_price - self.min_price)
            / (self.max_price - self.min_price)
        )

        normalized_cost = (
            self.market.cost_per_unit
            / self.max_price
        )

        return np.array(
            [
                remaining_days,
                inventory_ratio,
                normalized_reference_price,
                normalized_cost,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """
        Reset the environment.
        """

        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.market = sample_market_params(
            self.rng
        )

        self.day = 0
        self.inventory = self.max_inventory

        observation = self._get_observation()

        info = {}

        return observation, info

    def step(self, action: int):
        """
        Execute one pricing decision.
        """

        assert self.market is not None

        price = self._action_to_price(action)

        units_sold = simulate_sales(
            price=price,
            inventory=self.inventory,
            params=self.market,
            rng=self.rng,
        )

        revenue = price * units_sold

        production_cost = (
            self.market.cost_per_unit
            * units_sold
        )

        self.inventory -= units_sold

        holding_cost = (
            self.inventory
            * self.market.holding_cost
        )

        profit = (
            revenue
            - production_cost
            - holding_cost
        )

        reward = (
            profit / self.max_inventory
        )

        self.day += 1

        terminated = (
            self.day >= self.episode_days
            or self.inventory <= 0
        )

        truncated = False

        observation = self._get_observation()

        info = {
            "price": price,
            "units_sold": units_sold,
            "inventory": self.inventory,
            "revenue": revenue,
            "profit": profit,
            "holding_cost": holding_cost,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        """
        Print the current environment state.
        """

        assert self.market is not None

        print(
            f"Day: {self.day}/{self.episode_days}"
        )
        print(
            f"Inventory: {self.inventory}/{self.max_inventory}"
        )
        print(
            f"Reference Price: {self.market.reference_price:.2f}"
        )
        print(
            f"Cost Per Unit: {self.market.cost_per_unit:.2f}"
        )