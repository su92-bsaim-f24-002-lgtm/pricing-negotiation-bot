import numpy as np
import wandb

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class WandbCallback(BaseCallback):
    """
    Logs training metrics to Weights & Biases after each rollout.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_prices = []

    def _on_step(self) -> bool:
        # Collect information from all parallel environments
        for info in self.locals.get("infos", []):
            if "price" in info:
                self._episode_prices.append(info["price"])

            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

        return True

    def _on_rollout_end(self) -> None:
        if self._episode_rewards:
            wandb.log({
                "train/mean_episode_reward": float(np.mean(self._episode_rewards)),
                "train/mean_price_chosen": float(np.mean(self._episode_prices)) if self._episode_prices else 0.0,
                "train/timestep": self.num_timesteps,
            })

            self._episode_rewards.clear()
            self._episode_prices.clear()


class WandbEvalCallback(EvalCallback):
    """
    Evaluation callback that logs evaluation metrics to W&B.
    """

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Log after each evaluation
        if self.n_calls % self.eval_freq == 0:
            wandb.log({
                "eval/mean_reward": float(self.last_mean_reward),
                "eval/timestep": self.num_timesteps,
            })

        return result