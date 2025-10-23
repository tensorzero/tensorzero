"""
TensorZero experiment runner for track-and-stop bandits.

This module integrates with a running TensorZero gateway to test the
track-and-stop implementation against naive baselines.
"""

import asyncio
from typing import Optional
from uuid import UUID

import numpy as np
from tensorzero import AsyncTensorZeroGateway, Text

from environments import BanditEnvironment
from naive_bandits_runner import ExperimentResult


class TensorZeroExperimentRunner:
    """
    Runs experiments using TensorZero's track-and-stop implementation.

    This requires:
    - A running TensorZero gateway with track-and-stop experimentation
    - ClickHouse for storing telemetry
    - Postgres for episode-to-variant mapping
    """

    def __init__(
        self,
        env: BanditEnvironment,
        config_file: str,
        function_name: str = "test_function",
        metric_name: str = "test_metric",
        max_time_steps: int = 10000,
        clickhouse_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
    ):
        """
        Initialize TensorZero experiment runner.

        Args:
            env: Bandit environment
            config_file: Path to TensorZero config file
            function_name: TensorZero function name
            metric_name: TensorZero metric name
            max_time_steps: Maximum number of arm pulls
            clickhouse_url: ClickHouse connection URL
            postgres_url: PostgreSQL connection URL
        """
        self.env = env
        self.config_file = config_file
        self.function_name = function_name
        self.metric_name = metric_name
        self.max_time_steps = max_time_steps
        self.clickhouse_url = clickhouse_url
        self.postgres_url = postgres_url
        self.client = None

    async def setup(self):
        """Initialize the TensorZero gateway."""
        # build_embedded returns Awaitable when async_setup=True (default)
        gateway = AsyncTensorZeroGateway.build_embedded(
            config_file=self.config_file,
            clickhouse_url=self.clickhouse_url,
            postgres_url=self.postgres_url,
        )
        self.client = await gateway

    async def run(self, seed: int, episode_id: Optional[UUID] = None) -> ExperimentResult:
        """
        Run a single experiment with TensorZero track-and-stop.

        Args:
            seed: Random seed for this run
            episode_id: Optional episode ID for grouping inferences

        Returns:
            ExperimentResult with cumulative regret trajectory
        """
        if self.client is None:
            raise RuntimeError("Must call setup() before run()")

        np.random.seed(seed)

        # Use episode_id for consistency across inferences
        if episode_id is None:
            from uuid_utils import uuid7

            episode_id = uuid7()

        # Track metrics over time
        total_pulls = []
        cumulative_regrets = []
        variant_to_arm = {}  # Map variant names to arm indices

        # Track which variant was selected (for final recommendation)
        last_variant = None

        for t in range(1, self.max_time_steps + 1):
            # Call TensorZero inference - it selects the variant
            # Use a fixed input for cache hits (only first call costs money, rest are cached)
            response = await self.client.inference(
                function_name=self.function_name,
                input={"messages": [{"role": "user", "content": [Text(type="text", text="test")]}]},
                episode_id=episode_id,
                cache_options={"enabled": "on"},
            )

            # Extract variant name
            variant_name = response.variant_name

            # Map variant to arm index (e.g., "variant_0" -> 0)
            if variant_name not in variant_to_arm:
                # Parse "variant_N" to get arm index
                try:
                    arm_idx = int(variant_name.split("_")[1])
                    variant_to_arm[variant_name] = arm_idx
                except (IndexError, ValueError):
                    raise ValueError(
                        f"Unexpected variant name format: {variant_name}. "
                        "Expected format: variant_N"
                    )

            arm = variant_to_arm[variant_name]
            last_variant = variant_name

            # Sample reward from environment
            reward = self.env.sample_reward(arm)

            # Submit feedback to TensorZero
            await self.client.feedback(
                metric_name=self.metric_name,
                value=float(reward),
                inference_id=response.inference_id,
            )

            # Compute cumulative regret
            regret_per_pull = self.env.best_mean - self.env.true_means[arm]
            if t == 1:
                cumulative_regret = regret_per_pull
            else:
                cumulative_regret = cumulative_regrets[-1] + regret_per_pull

            total_pulls.append(t)
            cumulative_regrets.append(cumulative_regret)

            # Small delay to avoid overwhelming the gateway
            # (in real scenarios, inferences wouldn't happen this rapidly)
            await asyncio.sleep(0.001)

        # Determine final arm from last variant
        final_arm = variant_to_arm.get(last_variant, 0) if last_variant else 0

        # TODO: Detect if track-and-stop actually stopped
        # For now, we assume it doesn't stop within max_time_steps
        stopped = False
        stopping_time = None

        return ExperimentResult(
            bandit_type="subgaussian",
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

    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            await self.client.close()


async def run_tensorzero_experiment_batch(
    env_type: str,
    K: int,
    difficulty: str,
    config_file: str,
    n_runs: int = 10,
    max_time_steps: int = 10000,
    base_seed: int = 42,
    **env_kwargs,
):
    """
    Run a batch of TensorZero experiments.

    Args:
        env_type: "bernoulli", "beta", or "gaussian"
        K: Number of arms
        difficulty: "easy", "medium", or "hard"
        config_file: Path to TensorZero config
        n_runs: Number of independent runs
        max_time_steps: Maximum arm pulls per run
        base_seed: Base random seed
        **env_kwargs: Additional environment parameters

    Returns:
        List of ExperimentResults
    """
    import os

    from environments import create_environment

    results = []

    # Get database URLs from environment
    clickhouse_url = os.getenv("TENSORZERO_CLICKHOUSE_URL")
    postgres_url = os.getenv("TENSORZERO_POSTGRES_URL")

    # Create runner (reuse same gateway for all runs)
    env = create_environment(env_type, K, difficulty, base_seed, **env_kwargs)
    runner = TensorZeroExperimentRunner(
        env,
        config_file,
        max_time_steps=max_time_steps,
        clickhouse_url=clickhouse_url,
        postgres_url=postgres_url,
    )

    await runner.setup()

    try:
        for run_idx in range(n_runs):
            seed = base_seed + run_idx

            # Create fresh environment for this run
            env = create_environment(env_type, K, difficulty, seed, **env_kwargs)
            runner.env = env

            print(f"  Run {run_idx + 1}/{n_runs}: Running TensorZero experiment...")
            result = await runner.run(seed)
            results.append(result)

            stop_info = f"stopped at t={result.stopping_time}" if result.stopped else "did not stop"
            final_regret = result.cumulative_regrets[-1]
            print(f"  Run {run_idx + 1}/{n_runs}: {stop_info}, final regret={final_regret:.2f}")

    finally:
        await runner.cleanup()

    return results
