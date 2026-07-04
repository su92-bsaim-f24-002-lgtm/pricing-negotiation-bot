import numpy as np


class RandomPricer:
    """Samples a random price each step. Acts as the lower bound."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._rng = np.random.default_rng()

    def predict(self, obs: np.ndarray, deterministic: bool = False):
        obs = np.atleast_2d(obs)
        actions = self._rng.integers(0, self.n_actions, size=len(obs))
        return actions, None