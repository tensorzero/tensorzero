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
        batch_size: int = 100,
        batch_wait_time: float = 1.5,
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
            batch_size: Number of arm pulls per batch
            batch_wait_time: Time to wait between batches for track-and-stop update (seconds)
            clickhouse_url: ClickHouse connection URL
            postgres_url: PostgreSQL connection URL
        """
        self.env = env
        self.config_file = config_file
        self.function_name = function_name
        self.metric_name = metric_name
        self.max_time_steps = max_time_steps
        self.batch_size = batch_size
        self.batch_wait_time = batch_wait_time
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
        Run a single experiment with TensorZero track-and-stop using batch-based arm pulls.

        Args:
            seed: Random seed for this run
            episode_id: Optional episode ID for grouping inferences

        Returns:
            ExperimentResult with cumulative regret trajectory
        """
        import time

        if self.client is None:
            raise RuntimeError("Must call setup() before run()")

        np.random.seed(seed)

        # Track metrics over time
        total_pulls = []
        cumulative_regrets = []
        variant_to_arm = {}  # Map variant names to arm indices

        # Track which variant was selected (for final recommendation)
        last_variant = None
        t = 0
        batch_num = 0

        while t < self.max_time_steps:
            batch_num += 1
            batch_start = time.time()

            # Determine batch size (might be smaller for last batch)
            current_batch_size = min(self.batch_size, self.max_time_steps - t)

            # Run inference calls in parallel for the batch
            # Don't pass episode_id so each inference gets a fresh variant assignment
            inference_tasks = []
            for _ in range(current_batch_size):
                task = self.client.inference(
                    function_name=self.function_name,
                    input={
                        "messages": [{"role": "user", "content": [Text(type="text", text="test")]}]
                    },
                    cache_options={"enabled": "on"},
                )
                inference_tasks.append(task)

            # Wait for all inferences to complete
            inference_start = time.time()
            responses = await asyncio.gather(*inference_tasks)
            inference_duration = time.time() - inference_start

            # Process responses and submit feedback in parallel
            feedback_tasks = []
            batch_regrets = []
            batch_arms = []  # Track which arms were selected in this batch

            for response in responses:
                t += 1

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
                batch_arms.append(arm)

                # Sample reward from environment
                reward = self.env.sample_reward(arm)

                # Submit feedback (don't await yet)
                feedback_task = self.client.feedback(
                    metric_name=self.metric_name,
                    value=float(reward),
                    inference_id=response.inference_id,
                )
                feedback_tasks.append(feedback_task)

                # Compute cumulative regret
                regret_per_pull = self.env.best_mean - self.env.true_means[arm]
                if t == 1:
                    cumulative_regret = regret_per_pull
                else:
                    cumulative_regret = cumulative_regrets[-1] + regret_per_pull

                total_pulls.append(t)
                cumulative_regrets.append(cumulative_regret)
                batch_regrets.append(cumulative_regret)

            # Wait for all feedback to be submitted
            feedback_start = time.time()
            await asyncio.gather(*feedback_tasks)
            feedback_duration = time.time() - feedback_start

            # Calculate batch timing
            batch_end = time.time()
            batch_duration = batch_end - batch_start
            final_regret = cumulative_regrets[-1] if cumulative_regrets else 0.0

            # Calculate arm selection distribution for this batch
            from collections import Counter

            arm_counts = Counter(batch_arms)
            arm_dist = {i: arm_counts.get(i, 0) for i in range(self.env.K)}

            print(
                f"    Batch {batch_num}: t={t}/{self.max_time_steps}, "
                f"size={current_batch_size}, duration={batch_duration:.2f}s "
                f"(inference={inference_duration:.2f}s, feedback={feedback_duration:.2f}s), "
                f"cumulative_regret={final_regret:.2f}"
            )
            print(f"      Arm distribution: {arm_dist}")
            print(f"      True means: {self.env.true_means}")
            print(f"      Best arm: {self.env.best_arm} (mean={self.env.best_mean:.3f})")

            # Wait for track-and-stop background task to update probabilities
            # The update_period_s in config is 1, so we wait a bit longer to ensure it runs
            await asyncio.sleep(self.batch_wait_time)

        # Determine final arm from last variant
        final_arm = variant_to_arm.get(last_variant, 0) if last_variant else 0

        # We don't detect stopping in TensorZero (just track cumulative regret)
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
    batch_size: int = 100,
    batch_wait_time: float = 1.5,
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
        batch_size: Number of arm pulls per batch
        batch_wait_time: Time to wait between batches for track-and-stop update (seconds)
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

    print(f"  Using ClickHouse URL: {clickhouse_url}")
    print(f"  Using Postgres URL: {postgres_url}")

    # Create runner (reuse same gateway for all runs)
    env = create_environment(env_type, K, difficulty, base_seed, **env_kwargs)
    runner = TensorZeroExperimentRunner(
        env,
        config_file,
        max_time_steps=max_time_steps,
        batch_size=batch_size,
        batch_wait_time=batch_wait_time,
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
