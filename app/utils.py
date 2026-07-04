import streamlit as st
import numpy as np
import pandas as pd
import yaml

from environment.pricing_env import PricingEnv
from agent import PricingAgent


@st.cache_resource
def load_agent():
    """
    Load the trained PPO model once and cache it.
    """
    return PricingAgent.load("outputs/models/ppo_default_500k")


def create_env():
    with open("config/env_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # New random seed for every episode
    config["seed"] = np.random.randint(0, 2**31 - 1)

    return PricingEnv(config)


def run_episode(policy) -> pd.DataFrame:
    """
    Run one complete episode and return the results as a DataFrame.
    """

    env = create_env()

    obs, _ = env.reset()

    rows = []

    cumulative_reward = 0.0

    step = 0

    while True:

        action, _ = policy.predict(obs, deterministic=True)

        action = int(np.asarray(action).item())

        obs, reward, terminated, truncated, info = env.step(action)

        cumulative_reward += reward

        rows.append(
            {
                "step": step + 1,
                "price": info.get("price", 0),
                "units_sold": info.get("units_sold", 0),
                "inventory": info.get("inventory", 0),
                "profit": info.get("profit", 0),
                "reward": reward,
                "cumulative_reward": cumulative_reward,
            }
        )

        step += 1

        if terminated or truncated:
            break

    return pd.DataFrame(rows)