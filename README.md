# 💰 AI Pricing Negotiation Bot

An intelligent **Reinforcement Learning** system that learns optimal pricing strategies using **Proximal Policy Optimization (PPO)**. The agent interacts with a simulated marketplace and continuously improves its pricing decisions to maximize long-term profit while balancing customer demand and inventory.

---

## 📌 Project Overview

Traditional pricing strategies often rely on fixed rules or manually designed heuristics. This project demonstrates how **Reinforcement Learning** can autonomously discover profitable pricing policies by learning directly from interactions with a simulated environment.

The project compares a trained **PPO agent** against:

- 📏 Rule-Based Pricing
- 🎲 Random Pricing

and visualizes their performance through an interactive Streamlit dashboard.

---

## 🚀 Features

- 🤖 PPO Reinforcement Learning Agent
- 🛒 Custom Gymnasium Pricing Environment
- 📈 Dynamic Pricing Simulation
- 📦 Inventory Management
- 👥 Simulated Customer Demand
- 📊 Interactive Streamlit Dashboard
- 📉 Performance Visualization with Plotly
- 🏆 Policy Comparison Leaderboard
- 📋 Episode Analytics
- 📁 Clean Modular Architecture

---

## 🧠 Reinforcement Learning Workflow

```text
                 Customer Demand
                        ▲
                        │
                        │
Environment ───► PPO Agent ───► Price Decision
      ▲                              │
      │                              ▼
      └──────── Reward (Profit) ◄────┘
```

The agent repeatedly interacts with the environment, receives rewards based on generated profit, and updates its policy to improve future pricing decisions.

---

## 📂 Project Structure

```text
pricing-negotiation-bot/
│
├── agent/
│   ├── pricing_agent.py
│   ├── callbacks.py
│   └── policy_config.py
│
├── app/
│   ├── app.py
│   ├── charts.py
│   ├── metrics.py
│   ├── sections.py
│   ├── sidebar.py
│   ├── styles.py
│   └── utils.py
│
├── baselines/
│   ├── random_policy.py
│   └── rule_based.py
│
├── config/
│
├── environment/
│   ├── buyer_simulator.py
│   ├── demand_model.py
│   └── pricing_env.py
│
├── outputs/
│   ├── models/
│   ├── plots/
│   └── results.csv
│
├── tests/
│
├── train.py
├── evaluate.py
└── README.md
```

---

## 📊 Dashboard

The Streamlit dashboard includes:

- 📈 Performance Charts
- 💰 Profit Metrics
- 📦 Inventory Tracking
- 📉 Reward Curves
- 📄 Episode Data
- 🏆 Policy Leaderboard
- ℹ️ Project Overview

---

## 🛠 Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Reinforcement Learning | Stable-Baselines3 |
| Algorithm | PPO |
| Environment | Gymnasium |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Data Analysis | Pandas |
| Numerical Computing | NumPy |
| Configuration | YAML |
| Experiment Tracking | Weights & Biases |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/pricing-negotiation-bot.git

cd pricing-negotiation-bot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Agent

```bash
python train.py
```

---

## 📈 Evaluate the Agent

```bash
python evaluate.py
```

---

## 🖥 Launch the Dashboard

```bash
streamlit run app/app.py
```

---

## 📊 Results

The trained PPO agent consistently outperforms baseline pricing strategies by learning to:

- Maximize cumulative profit
- Adapt prices dynamically
- Manage inventory efficiently
- Balance demand and pricing decisions

Performance comparisons are available in the interactive dashboard.

---

## 🎯 Future Improvements

- Multi-product pricing
- Competitor-aware pricing
- Demand forecasting with LSTMs
- Multi-agent reinforcement learning
- Live API integration
- Cloud deployment

---

## 👨‍💻 Author

**Abdullah Waheed**

BS Artificial Intelligence

Portfolio Project • 2026

---

## 📜 License

This project is intended for educational and portfolio purposes.

---

⭐ If you found this project useful, consider giving it a star!