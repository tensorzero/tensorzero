#!/usr/bin/env python3
"""
Run full comparison of track-and-stop vs naive baselines.

This script runs experiments with all three bandit types:
- uniform_naive_no_bonferroni
- uniform_naive_bonferroni
- subgaussian (TensorZero track-and-stop)
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from naive_bandits_runner import run_experiment_batch
from plotting import create_comparison_plots, plot_grid_comparison
from tensorzero_runner import run_tensorzero_experiment_batch


async def main():
    parser = argparse.ArgumentParser(
        description="Run full bandit comparison with TensorZero integration"
    )
    parser.add_argument(
        "--env-types",
        nargs="+",
        default=["bernoulli"],
        choices=["bernoulli", "beta", "gaussian"],
        help="Environment types to test",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["medium"],
        choices=["easy", "medium", "hard"],
        help="Difficulty levels to test",
    )
    parser.add_argument("--K", type=int, default=5, help="Number of arms")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of independent runs per configuration",
    )
    parser.add_argument(
        "--max-time-steps", type=int, default=10000, help="Maximum time steps per run"
    )
    parser.add_argument("--delta", type=float, default=0.05, help="Confidence level for stopping")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Best arm tolerance")
    parser.add_argument(
        "--min-pulls-per-arm",
        type=int,
        default=10,
        help="Minimum pulls per arm before stopping",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--max-plot-time",
        type=int,
        default=None,
        help="Maximum time to show in plots (truncates trajectories)",
    )
    parser.add_argument(
        "--skip-tensorzero",
        action="store_true",
        help="Skip TensorZero experiments (only run naive baselines)",
    )
    parser.add_argument(
        "--only-tensorzero",
        action="store_true",
        help="Only run TensorZero experiments (skip naive baselines)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config/tensorzero.toml",
        help="TensorZero config file path",
    )
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    # Validate K matches config
    if args.K != 6:
        print(f"Warning: K={args.K} but config has 6 variants. Using K=6 to match config.")
        args.K = 6

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which bandit types to run
    if args.only_tensorzero:
        bandit_types = ["subgaussian"]
    elif args.skip_tensorzero:
        bandit_types = ["uniform_naive_no_bonferroni", "uniform_naive_bonferroni"]
    else:
        bandit_types = [
            "uniform_naive_no_bonferroni",
            "uniform_naive_bonferroni",
            "subgaussian",
        ]

    # Save configuration
    config = {
        "env_types": args.env_types,
        "difficulties": args.difficulties,
        "K": args.K,
        "n_runs": args.n_runs,
        "max_time_steps": args.max_time_steps,
        "delta": args.delta,
        "epsilon": args.epsilon,
        "min_pulls_per_arm": args.min_pulls_per_arm,
        "bandit_types": bandit_types,
        "base_seed": args.base_seed,
        "config_file": args.config_file,
        "timestamp": timestamp,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")

    # Run experiments
    all_results = {}
    grid_results = {}

    for env_type in args.env_types:
        for difficulty in args.difficulties:
            exp_name = f"{env_type}_{difficulty}_K{args.K}"
            print(f"\n{'=' * 60}")
            print(f"Running experiments: {exp_name}")
            print(f"{'=' * 60}")

            # Environment-specific parameters
            env_kwargs = {}
            if env_type == "beta":
                env_kwargs["concentration"] = 5.0
            elif env_type == "gaussian":
                env_kwargs["variance"] = 1.0

            results = {}

            # Run naive baselines (if not skipped)
            if not args.only_tensorzero:
                naive_types = [bt for bt in bandit_types if bt.startswith("uniform_naive")]
                if naive_types:
                    print("\nRunning naive baseline experiments...")
                    naive_results = run_experiment_batch(
                        env_type=env_type,
                        K=args.K,
                        difficulty=difficulty,
                        bandit_types=naive_types,
                        n_runs=args.n_runs,
                        delta=args.delta,
                        epsilon=args.epsilon,
                        min_pulls_per_arm=args.min_pulls_per_arm,
                        max_time_steps=args.max_time_steps,
                        base_seed=args.base_seed,
                        **env_kwargs,
                    )
                    results.update(naive_results)

            # Run TensorZero track-and-stop (if not skipped)
            if "subgaussian" in bandit_types:
                print("\nRunning TensorZero track-and-stop experiments...")
                tz_results = await run_tensorzero_experiment_batch(
                    env_type=env_type,
                    K=args.K,
                    difficulty=difficulty,
                    config_file=args.config_file,
                    n_runs=args.n_runs,
                    max_time_steps=args.max_time_steps,
                    base_seed=args.base_seed,
                    **env_kwargs,
                )
                results["subgaussian"] = tz_results

            all_results[exp_name] = results
            grid_results[(env_type, difficulty)] = results

    # Create plots
    print(f"\n{'=' * 60}")
    print("Creating plots...")
    print(f"{'=' * 60}")

    plots_dir = output_dir / "plots"
    create_comparison_plots(all_results, plots_dir, args.max_plot_time)

    # Create grid plot
    if len(args.env_types) > 1 or len(args.difficulties) > 1:
        grid_path = plots_dir / "cumulative_regret_grid.png"
        plot_grid_comparison(
            grid_results,
            args.env_types,
            args.difficulties,
            grid_path,
            args.max_plot_time,
        )

    # Save summary statistics
    summary = {}
    for exp_name, results in all_results.items():
        summary[exp_name] = {}
        for bandit_type, runs in results.items():
            if not runs:
                continue

            final_regrets = [run.cumulative_regrets[-1] for run in runs]
            stopping_times = [run.stopping_time for run in runs if run.stopped]
            stopped_fraction = len(stopping_times) / len(runs) if runs else 0

            mean_regret = sum(final_regrets) / len(final_regrets)
            squared_diffs = sum((r - mean_regret) ** 2 for r in final_regrets)
            variance = squared_diffs / len(final_regrets)
            std_regret = variance**0.5

            summary[exp_name][bandit_type] = {
                "n_runs": len(runs),
                "mean_final_regret": float(mean_regret),
                "std_final_regret": float(std_regret),
                "stopped_fraction": stopped_fraction,
                "mean_stopping_time": (
                    float(sum(stopping_times) / len(stopping_times)) if stopping_times else None
                ),
            }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary statistics saved to {summary_path}")

    print(f"\n{'=' * 60}")
    print("Experiments complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
