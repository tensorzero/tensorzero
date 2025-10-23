# Bandits Testing Framework

This directory contains a testing framework for comparing TensorZero's track-and-stop bandit implementation with naive uniform baseline approaches.

## Overview

The framework tests bandit algorithms on synthetic data generating processes:
- **Bernoulli**: Binary rewards with different success probabilities
- **Beta**: Continuous rewards in [0,1] with Beta distributions
- **Gaussian**: Continuous rewards with Gaussian distributions

It compares three bandit types:
- **`uniform_naive_no_bonferroni`**: Uniform sampling with Wald confidence intervals (no Bonferroni correction)
- **`uniform_naive_bonferroni`**: Uniform sampling with Bonferroni-corrected confidence intervals
- **`subgaussian`**: TensorZero's track-and-stop implementation (TODO: not yet integrated)

## Setup

Install dependencies:

```bash
pip install -r bandits-testing/requirements.txt
```

## Running Experiments

### Basic Usage

Run experiments with default settings:

```bash
python bandits-testing/run_experiments.py
```

This will:
1. Run 10 independent trials for each configuration
2. Test Bernoulli, Beta, and Gaussian environments at easy/medium/hard difficulty
3. Use K=5 arms
4. Generate cumulative regret trajectory plots
5. Save results to `bandits-testing/results/run_<timestamp>/`

### Custom Configuration

```bash
python bandits-testing/run_experiments.py \
  --env-types bernoulli gaussian \
  --difficulties medium hard \
  --K 6 \
  --n-runs 20 \
  --max-time-steps 15000 \
  --delta 0.05 \
  --epsilon 0.0 \
  --output-dir bandits-testing/results/my_experiment
```

### Arguments

- `--env-types`: Environment types to test (choices: bernoulli, beta, gaussian)
- `--difficulties`: Difficulty levels (choices: easy, medium, hard)
- `--K`: Number of arms (default: 5)
- `--n-runs`: Number of independent runs per configuration (default: 10)
- `--max-time-steps`: Maximum time steps per run (default: 10000)
- `--delta`: Confidence level for stopping (default: 0.05)
- `--epsilon`: Best arm tolerance (default: 0.0)
- `--min-pulls-per-arm`: Minimum pulls per arm before stopping (default: 10)
- `--output-dir`: Output directory for results (default: bandits-testing/results)
- `--max-plot-time`: Maximum time to show in plots (truncates trajectories)
- `--bandit-types`: Which bandit types to test (default: both naive baselines)
- `--base-seed`: Base random seed (default: 42)

## Output

Results are saved in timestamped directories under `results/`:

```
results/run_20250123_143022/
├── config.json                          # Experiment configuration
├── summary.json                         # Summary statistics
└── plots/
    ├── cumulative_regret_bernoulli_medium_K5.png
    ├── cumulative_regret_beta_medium_K5.png
    ├── cumulative_regret_gaussian_medium_K5.png
    └── cumulative_regret_grid.png       # Grid of all comparisons
```

### Plots

- **Individual plots**: One plot per (environment, difficulty, K) combination showing mean cumulative regret trajectories ± 1 std deviation
- **Grid plot**: Matrix view showing all environment × difficulty combinations

Plots compare the naive uniform baselines (and eventually the track-and-stop implementation).

## Code Structure

- `environments.py`: Data generating processes (Bernoulli, Beta, Gaussian)
- `naive_bandits.py`: Naive uniform baseline implementations
- `experiment_runner.py`: Orchestrates experiments and tracks cumulative regret
- `plotting.py`: Creates cumulative regret trajectory plots
- `run_experiments.py`: Main entry point script
- `config/tensorzero.toml`: TensorZero configuration (for future integration)

## Future Work: TensorZero Integration

The `subgaussian` bandit type is not yet implemented. To integrate with TensorZero:

1. Start a TensorZero gateway with track-and-stop experimentation configured
2. Set up local ClickHouse and Postgres instances
3. Implement `TensorZeroExperimentRunner.run()` in `experiment_runner.py` to:
   - Call TensorZero inference API to select variants (arms)
   - Submit feedback with reward values from the environment
   - Track cumulative regret over time
   - Query the gateway to detect when stopping occurs

The current framework provides the naive baselines for comparison once the TensorZero integration is complete.

## Comparison with Research Repo

This framework is adapted from `../research/bandits3` but simplified for integration with TensorZero:

- **Removed**: Complex optimization logic, KDE estimates, multi-client simulation
- **Kept**: Core environments, naive baselines, cumulative regret tracking, plotting utilities
- **Added**: Structure for TensorZero client integration (placeholder for now)

The plotting style and experiment structure match the research repo's `cumulative_regret_trajectories` plots.
