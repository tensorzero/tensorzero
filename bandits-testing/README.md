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

### Prerequisites

1. Install Docker and Docker Compose
2. Install Python 3.10+
3. Install the Python dependencies with `pip install -r requirements.txt`
4. Generate an API key for OpenAI (`OPENAI_API_KEY`)

### Environment Setup

1. Create a `.env` file with the required environment variables (see `.env.example` for an example):

```bash
cp .env.example .env
# Edit .env and add your actual OPENAI_API_KEY
```

2. Launch the required services (ClickHouse and PostgreSQL):

```bash
docker compose up -d
```

This will automatically run the PostgreSQL migrations needed for track-and-stop experimentation.

## Running Experiments

### Recommended: Config-Based Experiments

The recommended way to run experiments is using predefined configurations in `experiment_config.py`:

```bash
# Quick test with single environment (naive baselines only)
python run_experiment.py quick_test

# Compare naive baselines across all environments
python run_experiment.py naive_only

# Full comparison including TensorZero track-and-stop
python run_experiment.py full_comparison

# Test only TensorZero track-and-stop
python run_experiment.py tensorzero_only
```

**Advantages of config-based approach:**
- Reproducible experiments with version-controlled parameters
- Easy to define new experiment sets in `experiment_config.py`
- Cleaner command-line interface
- Matches the pattern used in `../research/bandits3`

To create custom experiments, edit `experiment_config.py` and add new entries to `EXPERIMENT_SETS`.

### Alternative: Legacy Command-Line Scripts

The older command-line scripts are still available but less recommended:

```bash
# Run only naive baselines (no TensorZero)
python run_experiments.py --env-types bernoulli --difficulties medium

# Run full comparison with TensorZero
python run_full_comparison.py --env-types bernoulli --difficulties medium
```

These scripts accept many command-line arguments. See `python run_experiments.py --help` for details.

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

- `experiment_config.py`: **Experiment configuration** - define experiment sets here
- `run_experiment.py`: **Main runner script** - uses configurations from experiment_config.py
- `environments.py`: Data generating processes (Bernoulli, Beta, Gaussian)
- `naive_bandits.py`: Naive uniform baseline implementations
- `naive_bandits_runner.py`: Orchestrates experiments for naive bandits
- `tensorzero_runner.py`: TensorZero integration for track-and-stop experiments
- `plotting.py`: Creates cumulative regret trajectory plots
- `run_experiments.py`: Legacy script for naive baselines (command-line args)
- `run_full_comparison.py`: Legacy script for full comparison (command-line args)
- `quick_test.py`: Quick verification script for naive baselines
- `config/tensorzero.toml`: TensorZero config with track-and-stop experimentation

## Running Full Comparison with TensorZero

To compare all three bandit types (including TensorZero's track-and-stop):

**Note on costs**: The framework uses `gpt-4o-mini` with caching enabled. Since all variants use the same model/provider and we send the same message for every inference, only the **very first inference** will actually hit the OpenAI API. All subsequent inferences (regardless of which variant is selected) are cache hits and cost nothing. Total cost per experiment run is negligible (~$0.0001 for a single API call).

### Run Full Comparison

First, ensure your environment variables are loaded. You can use [direnv](https://direnv.net/) to automatically load the `.env` file, or manually export them:

```bash
export $(cat .env | xargs)
```

Then run the comparison:

```bash
# Run all three bandit types
python run_full_comparison.py \
  --env-types bernoulli \
  --difficulties medium \
  --K 6 \
  --n-runs 10

# Run only naive baselines (skip TensorZero)
python run_full_comparison.py --skip-tensorzero

# Run only TensorZero (skip naive baselines)
python run_full_comparison.py --only-tensorzero
```

The script will:
1. Run naive baseline experiments (pure Python)
2. Start embedded TensorZero gateway with track-and-stop config
3. Run track-and-stop experiments via TensorZero Python client
4. Generate comparison plots showing all three approaches

### How TensorZero Integration Works

The `tensorzero_runner.py` module:
- Starts an embedded TensorZero gateway with track-and-stop experimentation
- Makes inference calls which trigger variant selection via track-and-stop
- Samples rewards from the environment for the selected variant
- Submits feedback to TensorZero
- Tracks cumulative regret over time
- Background task in gateway updates sampling probabilities asynchronously

## Comparison with Research Repo

This framework is adapted from `../research/bandits3` but simplified for integration with TensorZero:

- **Removed**: Complex optimization logic, KDE estimates, multi-client simulation
- **Kept**: Core environments, naive baselines, cumulative regret tracking, plotting utilities
- **Added**: Structure for TensorZero client integration (placeholder for now)

The plotting style and experiment structure match the research repo's `cumulative_regret_trajectories` plots.
