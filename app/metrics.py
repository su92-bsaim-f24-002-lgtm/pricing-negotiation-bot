import streamlit as st
import numpy as np


def render_metrics(
    episodes,
    combined,
    last_episode,
    env,
):
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

    st.subheader("📊 Performance Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💰 Average Profit",
        f"${np.mean(episode_returns):,.2f}",
        delta=f"${np.max(episode_returns)-np.min(episode_returns):,.2f}",
    )

    c2.metric(
        "💵 Average Price",
        f"${combined['price'].mean():.2f}",
    )

    c3.metric(
        "📦 Units Sold",
        f"{inventory_sold}",
    )

    c4.metric(
        "📅 Episode Length",
        f"{np.mean(episode_lengths):.1f} Days",
    )

    st.divider()