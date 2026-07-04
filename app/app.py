import streamlit as st
import pandas as pd
import numpy as np
import yaml

from utils import load_agent, run_episode
from charts import (
    reward_curve,
    price_inventory_chart,
    policy_comparison_bar,
)

from baselines.rule_based import RuleBasedPricer
from baselines.random_policy import RandomPricer
from environment.pricing_env import PricingEnv


# -----------------------------------------------------
# Page configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="Pricing Negotiation Bot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Autonomous Price Negotiation Bot")
st.caption(
    "A reinforcement learning agent that learns optimal pricing through market simulation."
)

# -----------------------------------------------------
# Load environment configuration
# -----------------------------------------------------

with open("config/env_config.yaml", "r") as f:
    env_config = yaml.safe_load(f)

env = PricingEnv(env_config)

price_levels = np.linspace(
    env.min_price,
    env.max_price,
    env.n_price_buckets,
)

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

with st.sidebar:

    st.header("Controls")

    policy_choice = st.selectbox(
        "Policy",
        [
            "PPO agent (RL)",
            "Rule-based",
            "Random",
        ],
    )

    n_episodes = st.slider(
        "Episodes",
        min_value=1,
        max_value=100,
        value=10,
    )

    run_btn = st.button(
        "Run Episode",
        type="primary",
        use_container_width=True,
    )

# -----------------------------------------------------
# Top Metrics
# -----------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

# -----------------------------------------------------
# Run Simulation
# -----------------------------------------------------

if run_btn:

    agent = load_agent()

    rule = RuleBasedPricer(price_levels)

    random_policy = RandomPricer(env.n_price_buckets)

    policy_map = {
        "PPO agent (RL)": agent,
        "Rule-based": rule,
        "Random": random_policy,
    }

    selected_policy = policy_map[policy_choice]

    with st.spinner(f"Running {n_episodes} episodes..."):

        episodes = [
            run_episode(selected_policy)
            for _ in range(n_episodes)
        ]

    last_episode = episodes[-1]

    combined = pd.concat(
        episodes,
        ignore_index=True,
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    episode_returns = [
        df["profit"].sum()
        for df in episodes
    ]

    episode_lengths = [
        len(df)
        for df in episodes
    ]

    inventory_sold = (
        env.max_inventory
        - last_episode["inventory"].iloc[-1]
    )

    col1.metric(
        "💰 Mean Episode Profit",
        f"${np.mean(episode_returns):.2f}",
    )

    col2.metric(
        "💵 Mean Selling Price",
        f"${combined['price'].mean():.2f}",
    )

    col3.metric(
        "📦 Inventory Sold",
        inventory_sold,
    )

    col4.metric(
        "📅 Avg Episode Length",
        f"{np.mean(episode_lengths):.1f} Days",
    )

    # -------------------------------------------------
    # Episode Summary
    # -------------------------------------------------

    st.markdown("---")

    st.subheader("📈 Episode Summary")

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "💵 Total Profit",
        f"${last_episode['profit'].sum():.2f}",
    )

    s2.metric(
        "📦 Units Sold",
        int(last_episode["units_sold"].sum()),
    )

    s3.metric(
        "📉 Remaining Inventory",
        int(last_episode["inventory"].iloc[-1]),
    )

    # -------------------------------------------------
    # Charts
    # -------------------------------------------------

    st.plotly_chart(
        price_inventory_chart(last_episode),
        use_container_width=True,
    )

    st.plotly_chart(
        reward_curve(last_episode),
        use_container_width=True,
    )

    # -------------------------------------------------
    # Episode Data
    # -------------------------------------------------

    with st.expander("📄 Episode Data"):

        st.dataframe(
            last_episode.style.format(
                {
                    "price": "${:.2f}",
                    "profit": "${:.2f}",
                    "reward": "{:.2f}",
                    "cumulative_reward": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    # -------------------------------------------------
    # Policy Leaderboard
    # -------------------------------------------------

    st.markdown("---")

    st.subheader("🏆 Policy Leaderboard")

    results_df = pd.read_csv(
        "outputs/results.csv"
    )

    leaderboard = (
        results_df
        .sort_values(
            "Mean Reward",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    leaderboard.index += 1

    best_policy = leaderboard.iloc[0]["Policy"]

    st.success(
        f"🏆 Best Performing Policy: **{best_policy}**"
    )

    results = {
        row["Policy"]: row["Mean Reward"]
        for _, row in leaderboard.iterrows()
    }

    st.plotly_chart(
        policy_comparison_bar(results),
        use_container_width=True,
    )

    st.dataframe(
        leaderboard.style.format(
            {
                "Mean Reward": "{:.2f}",
                "Std Reward": "{:.2f}",
                "Min Reward": "{:.2f}",
                "Max Reward": "{:.2f}",
                "Avg Episode Length": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

else:

    col1.metric("💰 Mean Episode Profit", "—")
    col2.metric("💵 Mean Selling Price", "—")
    col3.metric("📦 Inventory Sold", "—")
    col4.metric("📅 Avg Episode Length", "—")

    st.info(
        "Select a policy from the sidebar and click **Run Episode**."
    )