import streamlit as st
import pandas as pd
import numpy as np
import yaml
from styles import load_css

from utils import load_agent, run_episode
from sidebar import render_sidebar
from metrics import render_metrics
from sections import (
    render_charts,
    render_episode_table,
    render_leaderboard,
)

from baselines.rule_based import RuleBasedPricer
from baselines.random_policy import RandomPricer
from environment.pricing_env import PricingEnv


# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="AI Pricing Negotiation Bot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

# -----------------------------------------------------
# Header
# -----------------------------------------------------

st.markdown("""
# 💰 AI Pricing Negotiation Dashboard

### Reinforcement Learning for Dynamic Pricing

An autonomous pricing agent trained using **Proximal Policy Optimization (PPO)** that learns optimal pricing strategies through interaction with a simulated marketplace.
""")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.info("🤖 PPO Agent")

with c2:
    st.success("🏆 500K Timesteps")

with c3:
    st.warning("📈 Dynamic Pricing")

with c4:
    st.error("⚡ Stable-Baselines3")

st.divider()

# -----------------------------------------------------
# Environment
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

policy_choice, n_episodes, run_btn = render_sidebar()


dashboard_tab, analysis_tab, leaderboard_tab, about_tab = st.tabs(
    [
        "📊 Dashboard",
        "📈 Analysis",
        "🏆 Leaderboard",
        "ℹ️ About",
    ]
)

# -----------------------------------------------------
# Empty KPI cards before running
# -----------------------------------------------------

if not run_btn:

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("💰 Average Profit", "—")
    c2.metric("💵 Average Price", "—")
    c3.metric("📦 Units Sold", "—")
    c4.metric("📅 Episode Length", "—")

    st.info(
        "Select a policy in the sidebar and click **Run Simulation**."
    )

    st.stop()

# -----------------------------------------------------
# Load policies
# -----------------------------------------------------

agent = load_agent()

rule = RuleBasedPricer(price_levels)

random_policy = RandomPricer(env.n_price_buckets)

policy_map = {
    "PPO agent (RL)": agent,
    "Rule-based": rule,
    "Random": random_policy,
}

selected_policy = policy_map[policy_choice]

# -----------------------------------------------------
# Run Episodes
# -----------------------------------------------------

with st.spinner(f"Running {n_episodes} episode(s)..."):

    episodes = [
        run_episode(selected_policy)
        for _ in range(n_episodes)
    ]

combined = pd.concat(
    episodes,
    ignore_index=True,
)

last_episode = episodes[-1]

# -----------------------------------------------------
# Metrics
# -----------------------------------------------------

# =====================================================
# Dashboard Tab
# =====================================================

with dashboard_tab:

    render_metrics(
        episodes,
        combined,
        last_episode,
        env,
    )

# =====================================================
# Analysis Tab
# =====================================================

with analysis_tab:

    render_charts(last_episode)

    st.divider()

    render_episode_table(last_episode)

# =====================================================
# Leaderboard Tab
# =====================================================

with leaderboard_tab:

    render_leaderboard()

# =====================================================
# About Tab
# =====================================================

with about_tab:

    st.header("📖 About This Project")

    st.markdown(
        """
This project demonstrates how **Reinforcement Learning (RL)** can be applied to
dynamic pricing problems.

Instead of using fixed pricing rules, a **PPO (Proximal Policy Optimization)**
agent learns an optimal pricing strategy by interacting with a simulated market
environment.

The objective is to maximize long-term profit while balancing customer demand,
inventory levels, and pricing decisions.
"""
    )

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:

        st.subheader("🎯 Project Objective")

        st.markdown("""
- Learn optimal pricing automatically
- Compare RL with traditional pricing methods
- Simulate realistic customer demand
- Evaluate pricing performance over multiple episodes
""")

        st.subheader("⚙️ System Architecture")

        st.code(
            """
Environment
      │
      ▼
PPO Agent
      │
      ▼
Price Decision
      │
      ▼
Customer Demand
      │
      ▼
Reward (Profit)
      │
      ▼
Policy Update
""",
            language="text",
        )

    with col2:

        st.subheader("🛠 Tech Stack")

        st.success("🐍 Python")
        st.success("🤖 Stable-Baselines3")
        st.success("🎮 Gymnasium")
        st.success("📈 Plotly")
        st.success("⚡ Streamlit")
        st.success("📊 Pandas")
        st.success("🔬 NumPy")

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.info(
        """
**Abdullah Waheed**

BS Artificial Intelligence

Portfolio Project

2026
"""
    )

st.divider()

st.markdown(
"""
---
Built with ❤️ by **Abdullah Waheed**

Python • Streamlit • PPO • Stable-Baselines3 • Plotly • Gymnasium
"""
)