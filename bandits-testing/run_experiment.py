#!/usr/bin/env python3
"""
Run bandit experiments using configurations from experiment_config.py.

Usage:
    python run_experiment.py quick_test
    python run_experiment.py naive_only
    python run_experiment.py full_comparison
    python run_experiment.py tensorzero_only
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from experiment_config import (
    BASE_SEED,
    get_bandit_config,
    get_environment_params,
    get_experiment_config,
    list_experiments,
)
from naive_bandits_runner import run_experiment_batch
from plotting import create_comparison_plots, plot_grid_comparison
from tensorzero_runner import run_tensorzero_experiment_batch


async def main():
    parser = argparse.ArgumentParser(
        description="Run bandit experiments from config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available experiment sets:
{chr(10).join(f"  {name}" for name in list_experiments())}

Use 'python run_experiment.py <experiment_name>' to run.
""",
    )
    parser.add_argument(
        "experiment",
        choices=list_experiments(),
        help="Name of experiment set to run (from experiment_config.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: results/run_<timestamp>)",
    )
    parser.add_argument(
        "--max-plot-time",
        type=int,
        default=None,
        help="Maximum time to show in plots (truncates trajectories)",
    )

    args = parser.parse_args()

    # Load experiment configuration
    try:
        config = get_experiment_config(args.experiment)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nRunning experiment: {args.experiment}")
    print(f"Description: {config['description']}")
    print("=" * 60)

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from experiment_config import RESULTS_BASE_DIR, TIMESTAMP_FORMAT

        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        output_dir = Path(RESULTS_BASE_DIR) / f"run_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_with_metadata = {
        "experiment_name": args.experiment,
        "experiment_config": config,
        "base_seed": BASE_SEED,
        "timestamp": datetime.now().isoformat(),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_with_metadata, f, indent=2)
    print(f"\nConfiguration saved to {config_path}")

    # Determine which bandits to run
    algorithms = config["algorithms"]
    naive_algorithms = [a for a in algorithms if a.startswith("uniform_naive")]
    has_tensorzero = "subgaussian" in algorithms

    # Run experiments
    all_results = {}
    grid_results = {}

    for env_type in config["environments"]:
        for difficulty in config["difficulties"]:
            for K in config["K_values"]:
                exp_name = f"{env_type}_{difficulty}_K{K}"
                print(f"\n{'=' * 60}")
                print(f"Running experiments: {exp_name}")
                print(f"{'=' * 60}")

                # Get environment-specific parameters
                env_kwargs = get_environment_params(env_type)

                results = {}

                # Run naive baselines
                if naive_algorithms:
                    print("\nRunning naive baseline experiments...")
                    naive_results = run_experiment_batch(
                        env_type=env_type,
                        K=K,
                        difficulty=difficulty,
                        bandit_types=naive_algorithms,
                        n_runs=config["n_runs"],
                        delta=get_bandit_config(naive_algorithms[0])["delta"],
                        epsilon=get_bandit_config(naive_algorithms[0])["epsilon"],
                        min_pulls_per_arm=get_bandit_config(naive_algorithms[0])[
                            "min_pulls_per_arm"
                        ],
                        max_time_steps=config["max_time_steps"],
                        base_seed=BASE_SEED,
                        **env_kwargs,
                    )
                    results.update(naive_results)

                # Run TensorZero track-and-stop
                if has_tensorzero:
                    print("\nRunning TensorZero track-and-stop experiments...")
                    tz_config = get_bandit_config("subgaussian")
                    tz_results = await run_tensorzero_experiment_batch(
                        env_type=env_type,
                        K=K,
                        difficulty=difficulty,
                        config_file=tz_config["config_file"],
                        n_runs=config["n_runs"],
                        max_time_steps=config["max_time_steps"],
                        base_seed=BASE_SEED,
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

    # Create grid plot if multiple environments/difficulties
    if len(config["environments"]) > 1 or len(config["difficulties"]) > 1:
        grid_path = plots_dir / "cumulative_regret_grid.png"
        plot_grid_comparison(
            grid_results,
            config["environments"],
            config["difficulties"],
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
