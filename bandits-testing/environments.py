"""
Data generating processes for bandit testing.

These environments simulate different reward distributions for testing
bandit algorithms. They're simpler than the research repo versions since
we don't need full statistics tracking - that's handled by TensorZero.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BanditEnvironment(ABC):
    """Base class for bandit reward environments."""

    def __init__(self, K: int, seed: Optional[int] = None):
        """
        Initialize the bandit environment.

        Args:
            K: Number of arms
            seed: Random seed for reproducibility
        """
        self.K = K
        self.rng = np.random.RandomState(seed)
        self._setup_arms()

    @abstractmethod
    def _setup_arms(self):
        """Set up true parameters for each arm (implemented by subclasses)."""
        pass

    @abstractmethod
    def sample_reward(self, arm: int) -> float:
        """Generate a reward from the specified arm."""
        pass

    @property
    @abstractmethod
    def true_means(self) -> np.ndarray:
        """True mean rewards for each arm."""
        pass

    @property
    def best_arm(self) -> int:
        """Index of the true best arm."""
        return int(np.argmax(self.true_means))

    @property
    def best_mean(self) -> float:
        """True mean of the best arm."""
        return float(self.true_means[self.best_arm])


class BernoulliEnvironment(BanditEnvironment):
    """Bernoulli bandit environment."""

    def __init__(self, K: int, difficulty: str = "medium", seed: Optional[int] = None):
        """
        Initialize Bernoulli environment.

        Args:
            K: Number of arms
            difficulty: "easy", "medium", or "hard" - controls mean separation
            seed: Random seed
        """
        self.difficulty = difficulty
        super().__init__(K, seed)

    def _setup_arms(self):
        """Set up Bernoulli probabilities."""
        if self.difficulty == "easy":
            # Large gaps between arms
            self._true_probs = self.rng.uniform(0.1, 0.9, self.K)
            self._true_probs[0] = 0.9  # Clear best arm
            gaps = [0.15, 0.12, 0.10, 0.08, 0.06]
            for i in range(1, min(6, self.K)):
                if i - 1 < len(gaps):
                    self._true_probs[i] = max(0.1, self._true_probs[0] - gaps[i - 1])
        elif self.difficulty == "medium":
            # Moderate gaps
            self._true_probs = self.rng.uniform(0.3, 0.7, self.K)
            self._true_probs[0] = 0.7  # Best arm
            gaps = [0.08, 0.06, 0.05, 0.04, 0.03]
            for i in range(1, min(6, self.K)):
                if i - 1 < len(gaps):
                    self._true_probs[i] = max(0.2, self._true_probs[0] - gaps[i - 1])
        else:  # hard
            # Small gaps
            self._true_probs = self.rng.uniform(0.45, 0.55, self.K)
            self._true_probs[0] = 0.55  # Best arm
            gaps = [0.03, 0.025, 0.02, 0.015, 0.01]
            for i in range(1, min(6, self.K)):
                if i - 1 < len(gaps):
                    self._true_probs[i] = max(0.4, self._true_probs[0] - gaps[i - 1])

    def sample_reward(self, arm: int) -> float:
        return float(self.rng.binomial(1, self._true_probs[arm]))

    @property
    def true_means(self) -> np.ndarray:
        return self._true_probs.copy()


class BetaEnvironment(BanditEnvironment):
    """Beta-distributed bandit environment (bounded rewards in [0,1])."""

    def __init__(
        self,
        K: int,
        difficulty: str = "medium",
        concentration: float = 5.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Beta environment.

        Args:
            K: Number of arms
            difficulty: "easy", "medium", or "hard" - controls mean separation
            concentration: Controls variance of Beta distributions (higher = lower variance)
            seed: Random seed
        """
        self.difficulty = difficulty
        self.concentration = concentration
        super().__init__(K, seed)

    def _setup_arms(self):
        """Set up Beta distribution parameters."""
        if self.difficulty == "easy":
            means = self.rng.uniform(0.1, 0.9, self.K)
            means[0] = 0.9
            gaps = [0.15, 0.12, 0.10, 0.08, 0.06]
            for i in range(1, min(6, self.K)):
                if i - 1 < len(gaps):
                    means[i] = max(0.1, means[0] - gaps[i - 1])
        elif self.difficulty == "medium":
            means = self.rng.uniform(0.3, 0.7, self.K)
            means[0] = 0.7
            gaps = [0.08, 0.06, 0.05, 0.04, 0.03]
            for i in range(1, min(6, self.K)):
                if i - 1 < len(gaps):
                    means[i] = max(0.2, means[0] - gaps[i - 1])
        else:  # hard
            means = self.rng.uniform(0.45, 0.55, self.K)
            means[0] = 0.55
            gaps = [0.03, 0.025, 0.02, 0.015, 0.01]
            for i in range(1, min(6, self.K)):
                if i - 1 < len(gaps):
                    means[i] = max(0.4, means[0] - gaps[i - 1])

        # Convert means to Beta parameters
        self._alpha = means * self.concentration
        self._beta = (1 - means) * self.concentration
        self._true_means = means.copy()

    def sample_reward(self, arm: int) -> float:
        return float(self.rng.beta(self._alpha[arm], self._beta[arm]))

    @property
    def true_means(self) -> np.ndarray:
        return self._true_means.copy()


class GaussianEnvironment(BanditEnvironment):
    """Gaussian bandit environment."""

    def __init__(
        self, K: int, difficulty: str = "medium", variance: float = 1.0, seed: Optional[int] = None
    ):
        """
        Initialize Gaussian environment.

        Args:
            K: Number of arms
            difficulty: "easy", "medium", or "hard" - controls mean separation
            variance: Variance for all arms
            seed: Random seed
        """
        self.difficulty = difficulty
        self.variance = variance
        super().__init__(K, seed)

    def _setup_arms(self):
        """Set up Gaussian parameters."""
        std = np.sqrt(self.variance)

        if self.difficulty == "easy":
            self._means = self.rng.uniform(-2 * std, 2 * std, self.K)
            self._means[0] = 3 * std
            gaps = [1.5 * std, 1.2 * std, 1.0 * std, 0.8 * std, 0.6 * std]
            for i in range(1, min(6, self.K)):
                if i - 1 < len(gaps):
                    self._means[i] = self._means[0] - gaps[i - 1]
        elif self.difficulty == "medium":
            self._means = self.rng.uniform(-std, std, self.K)
            self._means[0] = 2 * std
            gaps = [0.8 * std, 0.6 * std, 0.5 * std, 0.4 * std, 0.3 * std]
            for i in range(1, min(6, self.K)):
                if i - 1 < len(gaps):
                    self._means[i] = self._means[0] - gaps[i - 1]
        else:  # hard
            self._means = self.rng.uniform(-0.5 * std, 0.5 * std, self.K)
            self._means[0] = std
            gaps = [0.3 * std, 0.25 * std, 0.2 * std, 0.15 * std, 0.1 * std]
            for i in range(1, min(6, self.K)):
                if i - 1 < len(gaps):
                    self._means[i] = self._means[0] - gaps[i - 1]

        self._variances = np.full(self.K, self.variance)

    def sample_reward(self, arm: int) -> float:
        return float(self.rng.normal(self._means[arm], np.sqrt(self._variances[arm])))

    @property
    def true_means(self) -> np.ndarray:
        return self._means.copy()


def create_environment(
    env_type: str, K: int, difficulty: str = "medium", seed: Optional[int] = None, **kwargs
) -> BanditEnvironment:
    """
    Factory function to create bandit environments.

    Args:
        env_type: "bernoulli", "beta", or "gaussian"
        K: Number of arms
        difficulty: "easy", "medium", "hard"
        seed: Random seed
        **kwargs: Additional environment-specific parameters

    Returns:
        BanditEnvironment instance
    """
    env_type = env_type.lower()

    if env_type == "bernoulli":
        return BernoulliEnvironment(K, difficulty, seed)
    elif env_type == "beta":
        concentration = kwargs.get("concentration", 5.0)
        return BetaEnvironment(K, difficulty, concentration, seed)
    elif env_type == "gaussian":
        variance = kwargs.get("variance", 1.0)
        return GaussianEnvironment(K, difficulty, variance, seed)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
