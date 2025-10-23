"""
Configuration for bandit testing experiments.

This module defines experimental parameters for comparing TensorZero's
track-and-stop implementation with naive baseline approaches.
"""

from typing import Any, Dict, List

# =============================================================================
# Environment Configuration
# =============================================================================

# Available environment types
ENVIRONMENT_TYPES = ["bernoulli", "beta", "gaussian"]

# Difficulty levels
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Number of arms
K_VALUES = [6]  # Fixed to match config file with 6 variants

# Environment-specific parameters
ENVIRONMENT_PARAMS = {
    "beta": {"concentration": 5.0},
    "gaussian": {"variance": 1.0},
}

# =============================================================================
# Bandit Algorithm Configuration
# =============================================================================

# Available bandit algorithms
BANDIT_ALGORITHMS = [
    "uniform_naive_no_bonferroni",
    "uniform_naive_bonferroni",
    "subgaussian",  # TensorZero track-and-stop
]

# Default bandit parameters
BANDIT_CONFIGS = {
    "uniform_naive_no_bonferroni": {
        "delta": 0.05,
        "epsilon": 0.0,
        "min_pulls_per_arm": 10,
    },
    "uniform_naive_bonferroni": {
        "delta": 0.05,
        "epsilon": 0.0,
        "min_pulls_per_arm": 10,
    },
    "subgaussian": {
        "config_file": "config/tensorzero.toml",
    },
}

# =============================================================================
# Experimental Design Configuration
# =============================================================================

# Number of independent runs per configuration
N_INDEPENDENT_RUNS = 2

# Maximum time steps per run
MAX_TIME_STEPS = 200

# Batch size for arm pulls (check stopping at batch boundaries)
BATCH_SIZE = 100

# TensorZero-specific: Time to wait between batches for track-and-stop update (seconds)
# The update_period_s in config is 1, so we wait a bit longer to ensure it runs
BATCH_WAIT_TIME = 1.5

# Random seed base (each run gets base_seed + run_idx)
BASE_SEED = 42

# =============================================================================
# Output Configuration
# =============================================================================

# Results directory
RESULTS_BASE_DIR = "results"

# Timestamp format
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# =============================================================================
# Experiment Sets Definition
# =============================================================================

# Define specific experiment configurations
EXPERIMENT_SETS = {
    "quick_test": {
        "description": "Quick test with single environment",
        "environments": ["bernoulli"],
        "difficulties": ["medium"],
        "K_values": K_VALUES,
        "algorithms": ["uniform_naive_no_bonferroni", "uniform_naive_bonferroni"],
        "n_runs": 3,
        "max_time_steps": MAX_TIME_STEPS,
    },
    "naive_only": {
        "description": "Compare only naive baseline algorithms",
        "environments": ["bernoulli", "beta", "gaussian"],
        "difficulties": ["easy", "medium", "hard"],
        "K_values": K_VALUES,
        "algorithms": ["uniform_naive_no_bonferroni", "uniform_naive_bonferroni"],
        "n_runs": N_INDEPENDENT_RUNS,
        "max_time_steps": MAX_TIME_STEPS,
    },
    "full_comparison": {
        "description": "Full comparison including TensorZero track-and-stop",
        "environments": ["bernoulli", "beta", "gaussian"],
        "difficulties": ["easy", "medium", "hard"],
        "K_values": K_VALUES,
        "algorithms": BANDIT_ALGORITHMS,
        "n_runs": N_INDEPENDENT_RUNS,
        "max_time_steps": MAX_TIME_STEPS,
    },
    "tensorzero_only": {
        "description": "Test only TensorZero track-and-stop",
        "environments": ["bernoulli"],
        "difficulties": ["medium"],
        "K_values": K_VALUES,
        "algorithms": ["subgaussian"],
        "n_runs": N_INDEPENDENT_RUNS,
        "max_time_steps": MAX_TIME_STEPS,
    },
}

# =============================================================================
# Utility Functions
# =============================================================================


def get_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """Get configuration for a specific experiment set."""
    if experiment_name not in EXPERIMENT_SETS:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. Available: {list(EXPERIMENT_SETS.keys())}"
        )
    return EXPERIMENT_SETS[experiment_name].copy()


def get_bandit_config(algorithm: str) -> Dict[str, Any]:
    """Get configuration for a specific bandit algorithm."""
    if algorithm not in BANDIT_CONFIGS:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Available: {list(BANDIT_CONFIGS.keys())}"
        )
    return BANDIT_CONFIGS[algorithm].copy()


def get_environment_params(env_type: str) -> Dict[str, Any]:
    """Get parameters for a specific environment type."""
    return ENVIRONMENT_PARAMS.get(env_type, {}).copy()


def list_experiments() -> List[str]:
    """List all available experiment sets."""
    return list(EXPERIMENT_SETS.keys())


def validate_config():
    """Validate the configuration parameters."""
    # Check that all referenced algorithms exist
    for exp_name, exp_config in EXPERIMENT_SETS.items():
        for algo in exp_config["algorithms"]:
            if algo not in BANDIT_CONFIGS:
                raise ValueError(f"Experiment '{exp_name}' references unknown algorithm '{algo}'")

        for env in exp_config["environments"]:
            if env not in ENVIRONMENT_TYPES:
                raise ValueError(f"Experiment '{exp_name}' references unknown environment '{env}'")

    print("Configuration validation passed!")


if __name__ == "__main__":
    validate_config()
    print("\nAvailable experiment sets:")
    for name, config in EXPERIMENT_SETS.items():
        print(f"  {name}: {config['description']}")
