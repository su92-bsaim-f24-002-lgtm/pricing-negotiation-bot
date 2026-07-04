from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env

from agent.callbacks import WandbCallback, WandbEvalCallback
from agent.policy_config import CONFIGS
from environment.pricing_env import PricingEnv


class PricingAgent:
    """
    Wrapper around Stable-Baselines3 PPO for the pricing environment.

    Handles environment creation, callback setup, training,
    saving, and loading.
    """

    def __init__(
        self,
        config_name="default",
        n_envs=4,
        seed=42,
        use_wandb=True,
        learning_rate=None,
        clip_range=None,
        ent_coef=None,
        n_steps=None,
    ):
        self.config = CONFIGS[config_name].copy()

        # Override hyperparameters from W&B Sweep
        if learning_rate is not None:
            self.config["learning_rate"] = learning_rate

        if clip_range is not None:
            self.config["clip_range"] = clip_range

        if ent_coef is not None:
            self.config["ent_coef"] = ent_coef

        if n_steps is not None:
            self.config["n_steps"] = n_steps

        self.n_envs = n_envs
        self.seed = seed
        self.use_wandb = use_wandb

        # Load environment configuration
        with open("config/env_config.yaml", "r") as f:
            env_config = yaml.safe_load(f)

        env_config["seed"] = seed

        # Training environment
        self.env = make_vec_env(
            PricingEnv,
            n_envs=n_envs,
            seed=seed,
            env_kwargs={"config": env_config},
        )

        # Evaluation environment
        eval_config = env_config.copy()
        eval_config["seed"] = seed + 100

        self.eval_env = make_vec_env(
            PricingEnv,
            n_envs=1,
            seed=seed + 100,
            env_kwargs={"config": eval_config},
        )

        # PPO model
        self.model = PPO(
            env=self.env,
            **self.config,
            seed=seed,
            tensorboard_log="outputs/logs",
        )

    def train(self, total_timesteps: int, eval_freq: int = 10_000):
        callbacks = []

        # Training metrics
        if self.use_wandb:
            callbacks.append(WandbCallback())

        # Evaluation metrics + best model saving
        eval_callback = WandbEvalCallback(
            self.eval_env,
            best_model_save_path="outputs/models/",
            log_path="outputs/logs/",
            eval_freq=max(eval_freq // self.n_envs, 1),
            n_eval_episodes=100,
            deterministic=True,
            verbose=1,
        )

        callbacks.append(eval_callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )

    def predict(self, obs, deterministic: bool = True):
        """
        Match the Stable-Baselines3 predict() interface.
        """
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}.zip")

    @classmethod
    def load(cls, path: str, config_name="default"):
        instance = cls.__new__(cls)
        instance.model = PPO.load(path)
        return instance 