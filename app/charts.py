import plotly.graph_objects as go
import pandas as pd
import numpy as np


def reward_curve(episode_df: pd.DataFrame) -> go.Figure:
    """Cumulative reward over an episode's steps."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=episode_df["step"],
            y=episode_df["cumulative_reward"],
            mode="lines",
            line=dict(color="#534AB7", width=2),
            name="Cumulative reward",
        )
    )

    fig.update_layout(
        title="Episode reward over time",
        xaxis_title="Day",
        yaxis_title="Cumulative profit (£)",
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig


def price_inventory_chart(episode_df: pd.DataFrame) -> go.Figure:
    """Price chosen and inventory level on dual axes."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=episode_df["step"],
            y=episode_df["price"],
            name="Price (£)",
            line=dict(color="#534AB7"),
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Bar(
            x=episode_df["step"],
            y=episode_df["inventory"],
            name="Inventory",
            marker_color="#E1F5EE",
            yaxis="y2",
            opacity=0.6,
        )
    )

    fig.update_layout(
        title="Price strategy vs inventory level",
        yaxis=dict(title="Price (£)"),
        yaxis2=dict(
            title="Inventory",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h"),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def policy_comparison_bar(results: dict[str, float]) -> go.Figure:
    """Bar chart comparing mean profit across policies."""

    fig = go.Figure(
        go.Bar(
            x=list(results.keys()),
            y=list(results.values()),
            marker_color=[
                "#534AB7",
                "#1D9E75",
                "#B4B2A9",
            ],
        )
    )

    fig.update_layout(
        title="Mean episode profit by policy",
        yaxis_title="Mean profit (£)",
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig