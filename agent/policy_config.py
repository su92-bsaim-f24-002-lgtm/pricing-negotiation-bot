"""
Named PPO hyperparameter configurations.
"""

CONFIGS = {
    "default": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 0,
    },
    "high_lr": {
        "policy": "MlpPolicy",
        "learning_rate": 1e-3,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.3,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 0,
    },
    "small_network": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 32,
        "n_epochs": 10,
        "gamma": 0.95,
        "gae_lambda": 0.9,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": [32, 32]
        },
        "verbose": 0,
    },
} 