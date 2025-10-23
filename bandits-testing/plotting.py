"""
Plotting utilities for bandit experiment results.

Creates cumulative regret trajectory plots similar to those in the research repo.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from experiment_runner import ExperimentResult


def plot_cumulative_regret_trajectories(
    results: Dict[str, List[ExperimentResult]],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    max_time: Optional[int] = None,
):
    """
    Plot cumulative regret trajectories for multiple bandit types.

    Args:
        results: Dictionary mapping bandit_type to list of ExperimentResults
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size (width, height)
        max_time: Maximum time to plot (truncates trajectories)
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "uniform_naive_no_bonferroni": "#1f77b4",  # Blue
        "uniform_naive_bonferroni": "#ff7f0e",  # Orange
        "subgaussian": "#2ca02c",  # Green
    }

    labels = {
        "uniform_naive_no_bonferroni": "Uniform Naive (No Bonferroni)",
        "uniform_naive_bonferroni": "Uniform Naive (Bonferroni)",
        "subgaussian": "Track-and-Stop (Subgaussian)",
    }

    for bandit_type, runs in results.items():
        if not runs:
            continue

        color = colors.get(bandit_type, None)
        label = labels.get(bandit_type, bandit_type)

        # Collect all trajectories
        all_times = []
        all_regrets = []

        for run in runs:
            times = np.array(run.total_pulls)
            regrets = np.array(run.cumulative_regrets)

            # Apply max_time cutoff if specified
            if max_time is not None:
                mask = times <= max_time
                times = times[mask]
                regrets = regrets[mask]

            if len(times) > 0:
                all_times.append(times)
                all_regrets.append(regrets)

        if not all_times:
            continue

        # Find common time range for interpolation
        min_time = min(t[0] for t in all_times)
        max_time_actual = max(t[-1] for t in all_times)
        if max_time is not None:
            max_time_actual = min(max_time_actual, max_time)

        # Create common time points
        common_times = np.linspace(min_time, max_time_actual, 200)

        # Interpolate each trajectory
        interpolated_regrets = []
        for times, regrets in zip(all_times, all_regrets):
            if len(times) > 1:
                interp_regrets = np.interp(common_times, times, regrets)
                interpolated_regrets.append(interp_regrets)

        if not interpolated_regrets:
            continue

        # Compute mean and std
        mean_regrets = np.mean(interpolated_regrets, axis=0)
        std_regrets = np.std(interpolated_regrets, axis=0)

        # Plot mean line
        ax.plot(common_times, mean_regrets, color=color, linewidth=2, label=label)

        # Add shaded region for std
        lower = np.maximum(0, mean_regrets - std_regrets)
        upper = mean_regrets + std_regrets
        ax.fill_between(common_times, lower, upper, color=color, alpha=0.2)

    ax.set_xlabel("Total Arm Pulls", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig, ax


def create_comparison_plots(
    all_results: Dict[str, Dict[str, List[ExperimentResult]]],
    output_dir: Path,
    max_time: Optional[int] = None,
):
    """
    Create comparison plots for multiple experiment configurations.

    Args:
        all_results: Nested dict: {env_config -> {bandit_type -> [results]}}
        output_dir: Directory to save plots
        max_time: Maximum time to plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for env_config, results in all_results.items():
        # Create plot for this environment configuration
        save_path = output_dir / f"cumulative_regret_{env_config}.png"

        plot_cumulative_regret_trajectories(
            results,
            title=f"Cumulative Regret - {env_config}",
            save_path=save_path,
            max_time=max_time,
        )

    print(f"\nAll plots saved to {output_dir}")


def plot_grid_comparison(
    all_results: Dict[tuple, Dict[str, List[ExperimentResult]]],
    env_types: List[str],
    difficulties: List[str],
    save_path: Optional[Path] = None,
    max_time: Optional[int] = None,
):
    """
    Create a grid of plots comparing different environments and difficulties.

    Args:
        all_results: Dict with keys (env_type, difficulty) mapping to {bandit_type -> [results]}
        env_types: List of environment types
        difficulties: List of difficulty levels
        save_path: Path to save figure
        max_time: Maximum time to plot
    """
    n_env = len(env_types)
    n_diff = len(difficulties)

    fig, axes = plt.subplots(n_diff, n_env, figsize=(5 * n_env, 4 * n_diff))

    if n_diff == 1:
        axes = axes.reshape(1, -1)
    if n_env == 1:
        axes = axes.reshape(-1, 1)

    colors = {
        "uniform_naive_no_bonferroni": "#1f77b4",
        "uniform_naive_bonferroni": "#ff7f0e",
        "subgaussian": "#2ca02c",
    }

    labels = {
        "uniform_naive_no_bonferroni": "Uniform Naive\n(No Bonferroni)",
        "uniform_naive_bonferroni": "Uniform Naive\n(Bonferroni)",
        "subgaussian": "Track-and-Stop\n(Subgaussian)",
    }

    for row, difficulty in enumerate(difficulties):
        for col, env_type in enumerate(env_types):
            ax = axes[row, col]
            key = (env_type, difficulty)

            if key not in all_results:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{env_type.title()} - {difficulty.title()}")
                continue

            results = all_results[key]

            for bandit_type, runs in results.items():
                if not runs:
                    continue

                color = colors.get(bandit_type)
                label = labels.get(bandit_type, bandit_type)

                # Process trajectories
                all_times = []
                all_regrets = []

                for run in runs:
                    times = np.array(run.total_pulls)
                    regrets = np.array(run.cumulative_regrets)

                    if max_time is not None:
                        mask = times <= max_time
                        times = times[mask]
                        regrets = regrets[mask]

                    if len(times) > 0:
                        all_times.append(times)
                        all_regrets.append(regrets)

                if not all_times:
                    continue

                # Interpolate
                min_time = min(t[0] for t in all_times)
                max_time_actual = max(t[-1] for t in all_times)
                if max_time is not None:
                    max_time_actual = min(max_time_actual, max_time)

                common_times = np.linspace(min_time, max_time_actual, 100)

                interpolated_regrets = []
                for times, regrets in zip(all_times, all_regrets):
                    if len(times) > 1:
                        interp = np.interp(common_times, times, regrets)
                        interpolated_regrets.append(interp)

                if not interpolated_regrets:
                    continue

                mean_regrets = np.mean(interpolated_regrets, axis=0)
                std_regrets = np.std(interpolated_regrets, axis=0)

                ax.plot(common_times, mean_regrets, color=color, linewidth=1.5, label=label)
                lower = np.maximum(0, mean_regrets - std_regrets)
                upper = mean_regrets + std_regrets
                ax.fill_between(common_times, lower, upper, color=color, alpha=0.2)

            # Labels
            if row == n_diff - 1:
                ax.set_xlabel("Total Arm Pulls", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{difficulty.title()}\nCumulative Regret", fontsize=9)

            # Title
            if row == 0:
                ax.set_title(env_type.title(), fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved grid plot to {save_path}")

    return fig
