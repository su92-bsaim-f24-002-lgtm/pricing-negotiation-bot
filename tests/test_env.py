import yaml
import numpy as np
import pytest

from stable_baselines3.common.env_checker import check_env

from environment.pricing_env import PricingEnv


@pytest.fixture
def env():
    """
    Create a fresh environment for every test.
    """

    with open("config/env_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = 42

    environment = PricingEnv(config)

    yield environment

    environment.close()


def test_sb3_env_checker(env):
    """
    Verify compatibility with Stable-Baselines3.
    """
    check_env(env, warn=True)


def test_reset_returns_valid_observation(env):
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_observation_bounds(env):
    obs, _ = env.reset()

    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_action_space(env):
    assert env.action_space.n == env.n_price_buckets


def test_step_returns_correct_types(env):
    env.reset()

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_inventory_never_negative(env):
    env.reset()

    done = False

    while not done:

        action = env.action_space.sample()

        _, _, terminated, truncated, info = env.step(action)

        assert info["inventory"] >= 0

        done = terminated or truncated


def test_price_within_limits(env):
    env.reset()

    for action in range(env.n_price_buckets):

        price = env._action_to_price(action)

        assert env.min_price <= price <= env.max_price


def test_episode_terminates(env):
    env.reset()

    steps = 0

    done = False

    while not done:

        action = env.action_space.sample()

        _, _, terminated, truncated, _ = env.step(action)

        steps += 1

        done = terminated or truncated

    assert steps <= env.episode_days


def test_reward_is_finite(env):
    env.reset()

    action = env.action_space.sample()

    _, reward, _, _, _ = env.step(action)

    assert np.isfinite(reward)