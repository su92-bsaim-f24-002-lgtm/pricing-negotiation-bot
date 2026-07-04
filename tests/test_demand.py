import numpy as np
import pytest

from environment.demand_model import (
    sigmoid_demand,
    MarketParams,
    sample_market_params,
)


@pytest.fixture
def params():
    return MarketParams(
        max_demand=50,
        elasticity_k=0.10,
        reference_price=25.0,
        cost_per_unit=10.0,
        holding_cost=0.5,
    )


def test_reference_price_gives_half_demand(params):
    """
    At the reference price, demand should be exactly half of max demand.
    """

    demand = sigmoid_demand(
        params.reference_price,
        params,
    )

    assert np.isclose(
        demand,
        params.max_demand / 2,
        atol=1e-6,
    )


def test_demand_decreases_as_price_increases(params):
    """
    Lower prices should always generate higher demand.
    """

    low = sigmoid_demand(10, params)
    high = sigmoid_demand(40, params)

    assert low > high


def test_demand_is_bounded(params):
    """
    Demand should always remain between 0 and max_demand.
    """

    for price in np.linspace(0, 100, 30):

        demand = sigmoid_demand(price, params)

        assert 0 <= demand <= params.max_demand


@pytest.mark.parametrize(
    "elasticity",
    [0.05, 0.10, 0.15, 0.20],
)
def test_monotonic_for_all_elasticities(elasticity):

    p = MarketParams(
        max_demand=50,
        elasticity_k=elasticity,
        reference_price=25,
        cost_per_unit=10,
        holding_cost=0.5,
    )

    prices = np.linspace(5, 50, 25)

    demands = [
        sigmoid_demand(price, p)
        for price in prices
    ]

    assert np.all(np.diff(demands) <= 0)


def test_sample_market_params_ranges():
    """
    Randomly generated market parameters should stay within
    configured ranges.
    """

    rng = np.random.default_rng(42)

    for _ in range(100):

        p = sample_market_params(rng)

        assert 20 <= p.max_demand <= 80
        assert 0.05 <= p.elasticity_k <= 0.20
        assert 15 <= p.reference_price <= 35
        assert 5 <= p.cost_per_unit <= 15
        assert 0.1 <= p.holding_cost <= 1.0


def test_high_price_produces_lower_demand(params):
    """
    Extremely high prices should produce very small demand.
    """

    demand = sigmoid_demand(1000, params)

    assert demand < 1


def test_low_price_produces_high_demand(params):
    """
    Extremely low prices should approach maximum demand.
    """

    demand = sigmoid_demand(-1000, params)

    assert demand > params.max_demand * 0.99