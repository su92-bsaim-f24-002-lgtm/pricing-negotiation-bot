from turtle import left, right

import streamlit as st
import pandas as pd

from charts import (
    reward_curve,
    price_inventory_chart,
    policy_comparison_bar,
)


def render_charts(last_episode):

    st.subheader("📊 Performance Charts")

    left, right = st.columns([1, 1], gap="large")

    fig1 = price_inventory_chart(last_episode)
    fig1.update_layout(height=380)

    fig2 = reward_curve(last_episode)
    fig2.update_layout(height=380)

    with left:
        st.plotly_chart(
            fig1,
            use_container_width=True,
            theme=None,
        )

    with right:
        st.plotly_chart(
            fig2,
            use_container_width=True,
            theme=None,
        )


def render_episode_table(last_episode):
    """Render episode dataframe."""

    with st.expander("📄 Episode Data", expanded=False):

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


def render_leaderboard():
    """Render policy comparison leaderboard."""

    st.divider()

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

    winner = leaderboard.iloc[0]["Policy"]

    st.success(
        f"🏆 Best Performing Policy: **{winner}**"
    )

    results = {
        row["Policy"]: row["Mean Reward"]
        for _, row in leaderboard.iterrows()
    }

    fig = policy_comparison_bar(results)
    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
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