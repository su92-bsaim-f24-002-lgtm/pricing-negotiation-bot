"""
Evaluate trained PPO agent against Rule-Based and Random baselines.

Outputs:
- Prints comparison table
- Saves outputs/results.csv
- Saves outputs/plots/policy_comparison.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from agent import PricingAgent
from baselines.random_policy import RandomPricer
from baselines.rule_based import RuleBasedPricer
from environment.pricing_env import PricingEnv


# ===========================
# Configuration
# ===========================

N_EPISODES = 1000

MODEL_PATH = "outputs/models/ppo_best"

CONFIG_PATH = "config/env_config.yaml"


# ===========================
# Environment
# ===========================

with open(CONFIG_PATH, "r") as f:
    env_config = yaml.safe_load(f)

env = PricingEnv(env_config)


# ===========================
# Load PPO
# ===========================

ppo_agent = PricingAgent.load(MODEL_PATH)


# ===========================
# Baselines
# ===========================

price_levels = np.linspace(
    env.min_price,
    env.max_price,
    env.n_price_buckets,
)

rule_agent = RuleBasedPricer(price_levels)
random_agent = RandomPricer(env.n_price_buckets)


# ===========================
# Evaluation Function
# ===========================

def evaluate(agent, name, episodes=N_EPISODES):

    rewards = []
    lengths = []

    for episode in range(episodes):

        obs, info = env.reset()

        done = False

        total_reward = 0.0
        steps = 0

        while not done:

            action, _ = agent.predict(
                obs,
                deterministic=True,
            )

            if isinstance(action, np.ndarray):
                if action.ndim == 0:
                    action = int(action)
                else:
                    action = int(action[0])
            else:
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "Policy": name,
        "Mean Reward": np.mean(rewards),
        "Std Reward": np.std(rewards),
        "Min Reward": np.min(rewards),
        "Max Reward": np.max(rewards),
        "Avg Episode Length": np.mean(lengths),
    }


# ===========================
# Run Evaluations
# ===========================

results = []

print("\nEvaluating PPO...")
results.append(evaluate(ppo_agent, "PPO"))

print("Evaluating Rule-Based...")
results.append(evaluate(rule_agent, "Rule-Based"))

print("Evaluating Random...")
results.append(evaluate(random_agent, "Random"))


# ===========================
# Results Table
# ===========================

df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print(df)
print("=" * 70)


# ===========================
# Save CSV
# ===========================

Path("outputs").mkdir(exist_ok=True)

df.to_csv(
    "outputs/results.csv",
    index=False,
)

print("\nSaved outputs/results.csv")


# ===========================
# Plot
# ===========================

Path("outputs/plots").mkdir(
    parents=True,
    exist_ok=True,
)

plt.figure(figsize=(8, 5))

plt.bar(
    df["Policy"],
    df["Mean Reward"],
)

plt.ylabel("Mean Reward")

plt.title("Policy Comparison")

plt.tight_layout()

plt.savefig(
    "outputs/plots/policy_comparison.png",
    dpi=300,
)

plt.close()

print("Saved outputs/plots/policy_comparison.png")


print("\nEvaluation Complete.")