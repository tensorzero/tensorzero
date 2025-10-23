"""
Naive bandit baselines for comparison.

These implement simple uniform sampling with statistical stopping rules
based on confidence intervals (Wald intervals with empirical variance).
"""

from typing import Optional, Tuple

import numpy as np
from scipy import stats


class NaiveBandit:
    """
    Base class for naive uniform sampling bandits.

    Uses uniform sampling and classical Wald confidence intervals
    with empirical variance estimates.
    """

    def __init__(
        self,
        K: int,
        delta: float,
        epsilon: float = 0.0,
        min_pulls_per_arm: int = 10,
        bonferroni: bool = False,
    ):
        """
        Initialize naive bandit.

        Args:
            K: Number of arms
            delta: Confidence level (error probability)
            epsilon: Best arm tolerance
            min_pulls_per_arm: Minimum pulls per arm before stopping
            bonferroni: Whether to use Bonferroni correction
        """
        self.K = K
        self.delta = delta
        self.epsilon = epsilon
        self.min_pulls_per_arm = min_pulls_per_arm
        self.bonferroni = bonferroni

        # State
        self.pull_counts = np.zeros(K, dtype=int)
        self.sum_rewards = np.zeros(K)
        self.sum_squared_rewards = np.zeros(K)

    def select_arm(self) -> int:
        """Select an arm uniformly at random."""
        return np.random.randint(0, self.K)

    def update(self, arm: int, reward: float):
        """Update statistics with new observation."""
        self.pull_counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.sum_squared_rewards[arm] += reward**2

    def get_mean_rewards(self) -> np.ndarray:
        """Get current empirical mean rewards."""
        means = np.zeros(self.K)
        for i in range(self.K):
            if self.pull_counts[i] > 0:
                means[i] = self.sum_rewards[i] / self.pull_counts[i]
            else:
                means[i] = np.nan
        return means

    def get_variances(self) -> np.ndarray:
        """Get current empirical variance estimates."""
        variances = np.zeros(self.K)
        for i in range(self.K):
            if self.pull_counts[i] > 1:
                mean = self.sum_rewards[i] / self.pull_counts[i]
                # Unbiased variance estimate
                numerator = self.sum_squared_rewards[i] - self.pull_counts[i] * mean**2
                variances[i] = numerator / (self.pull_counts[i] - 1)
                variances[i] = max(variances[i], 1e-12)  # Ridge regularization
            else:
                variances[i] = 1.0  # Default for insufficient data
        return variances

    def compute_confidence_interval(self, arm: int) -> Tuple[float, float]:
        """Compute Wald confidence interval for an arm."""
        if self.pull_counts[arm] == 0:
            return float("-inf"), float("inf")

        mean = self.sum_rewards[arm] / self.pull_counts[arm]
        variance = self.get_variances()[arm]
        n = self.pull_counts[arm]

        # Determine confidence level
        conf_delta = self.delta
        if self.bonferroni:
            conf_delta = self.delta / self.K

        # Critical value for two-sided confidence interval
        z_critical = stats.norm.ppf(1 - conf_delta / 2)

        standard_error = np.sqrt(variance / n)
        margin = z_critical * standard_error

        return mean - margin, mean + margin

    def check_stopping(self) -> Tuple[bool, Optional[int]]:
        """
        Check if we should stop based on confidence intervals.

        Returns:
            (should_stop, recommended_arm)
        """
        # Need minimum samples
        if np.any(self.pull_counts < self.min_pulls_per_arm):
            return False, None

        means = self.get_mean_rewards()
        leader_arm = np.argmax(means)

        # Get confidence intervals
        lcb_leader, ucb_leader = self.compute_confidence_interval(leader_arm)

        # Check stopping condition: LCB_leader >= UCB_j - epsilon for all j
        for j in range(self.K):
            if j == leader_arm:
                continue
            lcb_j, ucb_j = self.compute_confidence_interval(j)
            if lcb_leader < ucb_j - self.epsilon:
                return False, leader_arm

        return True, leader_arm

    def get_total_pulls(self) -> int:
        """Get total number of pulls across all arms."""
        return int(np.sum(self.pull_counts))


class UniformNaiveNoBonferroni(NaiveBandit):
    """Naive bandit without Bonferroni correction."""

    def __init__(self, K: int, delta: float, epsilon: float = 0.0, min_pulls_per_arm: int = 10):
        super().__init__(K, delta, epsilon, min_pulls_per_arm, bonferroni=False)


class UniformNaiveBonferroni(NaiveBandit):
    """Naive bandit with Bonferroni correction."""

    def __init__(self, K: int, delta: float, epsilon: float = 0.0, min_pulls_per_arm: int = 10):
        super().__init__(K, delta, epsilon, min_pulls_per_arm, bonferroni=True)
