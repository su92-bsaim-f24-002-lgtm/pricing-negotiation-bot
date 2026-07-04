import numpy as np
from dataclasses import dataclass


@dataclass
class MarketParams:
    """
    Stores all market parameters for a single market episode.
    """

    max_demand: float
    elasticity_k: float
    reference_price: float
    cost_per_unit: float
    holding_cost: float


def sigmoid_demand(price: float, params: MarketParams) -> float:
    """
    Calculate the expected customer demand for a given price using
    a sigmoid (logistic) demand curve.

    Behaviour:
    - Lower price  -> Higher demand
    - Higher price -> Lower demand

    Args:
        price: Selling price chosen by the agent.
        params: Market parameters for the current episode.

    Returns:
        Expected demand as a floating-point number.
    """

    exponent = params.elasticity_k * (
        price - params.reference_price
    )

    demand = params.max_demand / (
        1 + np.exp(exponent)
    )

    return demand


def sample_market_params(
    rng: np.random.Generator,
) -> MarketParams:
    """
    Generate random market parameters for a new episode.

    Randomising the market each episode prevents the RL agent
    from memorising one market and encourages generalisation.

    Args:
        rng: NumPy random number generator.

    Returns:
        A MarketParams object containing randomly generated
        market characteristics.
    """

    return MarketParams(
        max_demand=rng.uniform(20, 80),
        elasticity_k=rng.uniform(0.05, 0.20),
        reference_price=rng.uniform(15, 35),
        cost_per_unit=rng.uniform(5, 15),
        holding_cost=rng.uniform(0.1, 1.0),
    )