import numpy as np

from environment.demand_model import (
    MarketParams,
    sigmoid_demand,
)


def simulate_sales(
    price: float,
    inventory: int,
    params: MarketParams,
    rng: np.random.Generator,
) -> int:
    """
    Simulate the number of units sold for one day.

    Args:
        price: Selling price chosen by the agent.
        inventory: Current inventory available.
        params: Market parameters.
        rng: NumPy random number generator.

    Returns:
        Number of units sold.
    """

    expected_demand = sigmoid_demand(price, params)

    sampled_demand = rng.poisson(expected_demand)

    units_sold = min(sampled_demand, inventory)

    return int(units_sold)