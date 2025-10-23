"""
Experiment runner that orchestrates bandit testing.

This module runs experiments comparing track-and-stop (subgaussian) bandit
with naive uniform baselines, tracking cumulative regret over time.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from environments import BanditEnvironment
from naive_bandits import UniformNaiveBonferroni, UniformNaiveNoBonferroni


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    bandit_type: str
    env_type: str
    difficulty: str
    K: int
    total_pulls: List[int]  # Time steps
    cumulative_regrets: List[float]  # Cumulative regret at each time step
    stopped: bool
    stopping_time: Optional[int]
    final_arm: Optional[int]
    true_best_arm: int
    true_means: np.ndarray
    seed: int


class NaiveExperimentRunner:
    """
    Runs experiments for naive uniform baseline bandits.

    These don't interact with TensorZero - they're pure Python simulations
    for baseline comparison.
    """

    def __init__(
        self,
        env: BanditEnvironment,
        bandit_type: str,
        delta: float = 0.05,
        epsilon: float = 0.0,
        min_pulls_per_arm: int = 10,
        max_time_steps: int = 10000,
    ):
        """
        Initialize experiment runner.

        Args:
            env: Bandit environment
            bandit_type: "uniform_naive_no_bonferroni" or "uniform_naive_bonferroni"
            delta: Confidence level
            epsilon: Best arm tolerance
            min_pulls_per_arm: Minimum pulls per arm before stopping
            max_time_steps: Maximum number of arm pulls
        """
        self.env = env
        self.bandit_type = bandit_type
        self.delta = delta
        self.epsilon = epsilon
        self.min_pulls_per_arm = min_pulls_per_arm
        self.max_time_steps = max_time_steps

        # Create bandit
        if bandit_type == "uniform_naive_no_bonferroni":
            self.bandit = UniformNaiveNoBonferroni(env.K, delta, epsilon, min_pulls_per_arm)
        elif bandit_type == "uniform_naive_bonferroni":
            self.bandit = UniformNaiveBonferroni(env.K, delta, epsilon, min_pulls_per_arm)
        else:
            raise ValueError(f"Unknown bandit type: {bandit_type}")

    def run(self, seed: int) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            seed: Random seed for this run

        Returns:
            ExperimentResult with cumulative regret trajectory
        """
        np.random.seed(seed)

        # Track metrics over time
        total_pulls = []
        cumulative_regrets = []

        # Run experiment
        stopped = False
        stopping_time = None
        final_arm = None

        for t in range(1, self.max_time_steps + 1):
            # If stopped, use the recommended arm; otherwise select and pull
            if stopped:
                # After stopping, continue accumulating regret for the final arm
                arm = final_arm
            else:
                # Select and pull arm
                arm = self.bandit.select_arm()
                reward = self.env.sample_reward(arm)
                # Update bandit
                self.bandit.update(arm, reward)

            # Compute cumulative regret
            regret_per_pull = self.env.best_mean - self.env.true_means[arm]
            if t == 1:
                cumulative_regret = regret_per_pull
            else:
                cumulative_regret = cumulative_regrets[-1] + regret_per_pull

            total_pulls.append(t)
            cumulative_regrets.append(cumulative_regret)

            # Check stopping condition (only if not already stopped)
            if not stopped:
                should_stop, recommended_arm = self.bandit.check_stopping()
                if should_stop:
                    stopped = True
                    stopping_time = t
                    final_arm = recommended_arm

        # If didn't stop, use empirical best
        if not stopped:
            means = self.bandit.get_mean_rewards()
            final_arm = int(np.argmax(means))

        return ExperimentResult(
            bandit_type=self.bandit_type,
            env_type=getattr(self.env, "difficulty", "unknown"),
            difficulty=getattr(self.env, "difficulty", "unknown"),
            K=self.env.K,
            total_pulls=total_pulls,
            cumulative_regrets=cumulative_regrets,
            stopped=stopped,
            stopping_time=stopping_time,
            final_arm=final_arm,
            true_best_arm=self.env.best_arm,
            true_means=self.env.true_means,
            seed=seed,
        )


class TensorZeroExperimentRunner:
    """
    Runs experiments for TensorZero's track-and-stop (subgaussian) bandit.

    This will interact with a running TensorZero gateway via the Python client.
    For now, this is a placeholder that simulates the interface.

    TODO: Implement actual TensorZero integration once gateway is running.
    """

    def __init__(
        self,
        env: BanditEnvironment,
        function_name: str = "test_function",
        metric_name: str = "test_metric",
        max_time_steps: int = 10000,
    ):
        """
        Initialize TensorZero experiment runner.

        Args:
            env: Bandit environment
            function_name: TensorZero function name
            metric_name: TensorZero metric name
            max_time_steps: Maximum number of arm pulls
        """
        self.env = env
        self.function_name = function_name
        self.metric_name = metric_name
        self.max_time_steps = max_time_steps

        # TODO: Initialize TensorZero client
        # from tensorzero import TensorZeroClient
        # self.client = TensorZeroClient(...)

    def run(self, seed: int) -> ExperimentResult:
        """
        Run a single experiment with TensorZero track-and-stop.

        Args:
            seed: Random seed for this run

        Returns:
            ExperimentResult with cumulative regret trajectory
        """
        raise NotImplementedError(
            "TensorZero integration not yet implemented. "
            "This requires a running TensorZero gateway with track-and-stop "
            "experimentation configured, and proper ClickHouse/Postgres setup."
        )


def run_experiment_batch(
    env_type: str,
    K: int,
    difficulty: str,
    bandit_types: List[str],
    n_runs: int = 10,
    delta: float = 0.05,
    epsilon: float = 0.0,
    min_pulls_per_arm: int = 10,
    max_time_steps: int = 10000,
    base_seed: int = 42,
    **env_kwargs,
) -> Dict[str, List[ExperimentResult]]:
    """
    Run a batch of experiments for multiple bandit types.

    Args:
        env_type: "bernoulli", "beta", or "gaussian"
        K: Number of arms
        difficulty: "easy", "medium", or "hard"
        bandit_types: List of bandit types to test
        n_runs: Number of independent runs per bandit type
        delta: Confidence level
        epsilon: Best arm tolerance
        min_pulls_per_arm: Minimum pulls per arm before stopping
        max_time_steps: Maximum arm pulls per run
        base_seed: Base random seed
        **env_kwargs: Additional environment parameters

    Returns:
        Dictionary mapping bandit_type to list of ExperimentResults
    """
    from environments import create_environment

    results = {bt: [] for bt in bandit_types}

    for bandit_type in bandit_types:
        print(f"\nRunning {n_runs} experiments for {bandit_type}...")

        for run_idx in range(n_runs):
            seed = base_seed + run_idx

            # Create fresh environment for this run
            env = create_environment(env_type, K, difficulty, seed, **env_kwargs)

            # Run experiment
            if bandit_type == "subgaussian":
                # TensorZero track-and-stop (not yet implemented)
                print(f"  Run {run_idx + 1}/{n_runs}: Skipping subgaussian (not implemented)")
                continue
            else:
                # Naive baseline
                runner = NaiveExperimentRunner(
                    env, bandit_type, delta, epsilon, min_pulls_per_arm, max_time_steps
                )
                result = runner.run(seed)
                results[bandit_type].append(result)

                stop_info = (
                    f"stopped at t={result.stopping_time}" if result.stopped else "did not stop"
                )
                final_regret = result.cumulative_regrets[-1]
                print(f"  Run {run_idx + 1}/{n_runs}: {stop_info}, final regret={final_regret:.2f}")

    return results
