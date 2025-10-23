#!/usr/bin/env python3
"""
Quick test script to verify the framework is working.

Runs a small experiment to ensure everything is set up correctly.
"""

from pathlib import Path

from experiment_runner import run_experiment_batch
from plotting import plot_cumulative_regret_trajectories


def main():
    print("Running quick test of bandits framework...")
    print("=" * 60)

    # Run a small experiment
    results = run_experiment_batch(
        env_type="bernoulli",
        K=3,
        difficulty="medium",
        bandit_types=["uniform_naive_no_bonferroni", "uniform_naive_bonferroni"],
        n_runs=3,
        delta=0.05,
        epsilon=0.0,
        min_pulls_per_arm=10,
        max_time_steps=5000,
        base_seed=42,
    )

    print("\n" + "=" * 60)
    print("Quick test complete!")
    print("=" * 60)

    # Print summary
    for bandit_type, runs in results.items():
        if runs:
            final_regrets = [run.cumulative_regrets[-1] for run in runs]
            stopped = sum(1 for run in runs if run.stopped)
            print(f"\n{bandit_type}:")
            print(f"  Runs completed: {len(runs)}")
            print(f"  Stopped: {stopped}/{len(runs)}")
            print(f"  Mean final regret: {sum(final_regrets) / len(final_regrets):.2f}")

    # Create a quick plot
    output_dir = Path("results/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "quick_test_plot.png"

    print(f"\nCreating plot at {plot_path}...")
    plot_cumulative_regret_trajectories(
        results, title="Quick Test - Bernoulli Medium (K=3)", save_path=plot_path
    )

    print("\n" + "=" * 60)
    print("Framework is working correctly!")
    print(f"Check the plot at: {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
