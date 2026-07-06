import streamlit as st


def render_sidebar():
    """Render the dashboard sidebar."""

    with st.sidebar:

        st.title("⚙️ Controls")

        policy = st.selectbox(
            "Select Policy",
            [
                "PPO agent (RL)",
                "Rule-based",
                "Random",
            ],
        )

        episodes = st.slider(
            "Episodes",
            min_value=1,
            max_value=100,
            value=10,
        )

        run = st.button(
            "▶ Run Simulation",
            type="primary",
            use_container_width=True,
        )

        st.divider()

        st.subheader("🤖 Model")

        st.success("PPO (500k Timesteps)")

        st.divider()

        st.subheader("📚 Project")

        st.markdown(
            """
**Algorithm**

- PPO

**Environment**

- Gymnasium

**Framework**

- Stable-Baselines3

**Language**

- Python

**Training**

- 500,000 Timesteps

**Evaluation**

- PPO vs Rule-Based vs Random
"""
        )

    return policy, episodes, run


st.markdown("---")

st.success("✅ Project Status")

st.progress(100)

st.caption("Training Complete")

st.markdown("---")

st.caption("Version 1.0")