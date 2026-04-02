# Autopilot Benchmarks

Benchmark harness for TensorZero Autopilot. Runs LLM tasks through a TensorZero gateway, connects to Autopilot for optimization, and measures improvement over iterations.

## How It Works

1. **Baseline**: Generate a TensorZero config from an [llmgym](https://github.com/tensorzero/llmgym) environment, start a gateway, and run episodes to establish baseline metrics.
2. **Autopilot iteration**: Create an Autopilot session, let it analyze the data and propose config changes (new prompts, models, parameters), apply those changes, restart the gateway.
3. **Evaluate**: Run episodes again with the updated config. Record train and test metrics separately.

```
eval_config.yaml
  -> cli.py
    -> orchestrator.py (per environment)
      |-- config_generator.py  -> generates T0 config from llmgym env
      |-- gateway_process.py   -> manages gateway binary as subprocess
      |-- runner.py            -> runs concurrent episodes via llmgym
      |-- session.py           -> polls Autopilot, auto-approves, handles Q&A
      |-- config_applier.py    -> applies Autopilot's edits via Rust CLI
      '-- recorder.py          -> writes results as JSON files
```

## Prerequisites

- Docker and Docker Compose
- API keys (see below)

## Quick Start

```bash
cp .env.example .env
# Fill in your API keys

# Build the eval container (first time only, takes ~5 min for Rust compilation)
docker compose -f docker/docker-compose.yml --env-file .env build

# Run a benchmark
docker compose -f docker/docker-compose.yml --env-file .env run --rm eval \
  run --config configs/ner.yaml --verbose
```

Results are written to `output/<env_name>/<timestamp>/`.

## Available Benchmarks

| Config                   | Environment          | Type                     | Metric      | Required Keys              |
| ------------------------ | -------------------- | ------------------------ | ----------- | -------------------------- |
| `ner.yaml`               | ner_conllpp_v0       | NER extraction           | exact_match | OPENAI, ANTHROPIC          |
| `21_questions.yaml`      | 21_questions_v0      | 21 Questions game        | solved      | OPENAI, ANTHROPIC          |
| `babyai.yaml`            | babyai_pickup_v0     | BabyAI grid world        | success     | OPENAI, ANTHROPIC          |
| `tau_bench_airline.yaml` | tau_bench_airline_v0 | Airline customer service | success     | OPENAI, ANTHROPIC          |
| `tau_bench_retail.yaml`  | tau_bench_retail_v0  | Retail customer service  | success     | OPENAI, ANTHROPIC          |
| `lawbench.yaml`          | lawbench@1.0         | Legal QA                 | reward      | OPENAI, ANTHROPIC, DAYTONA |
| `medagentbench.yaml`     | medagentbench@1.0    | Medical QA               | reward      | OPENAI, ANTHROPIC, DAYTONA |
| `replicationbench.yaml`  | replicationbench@1.0 | ML replication           | reward      | OPENAI, ANTHROPIC, DAYTONA |
| `terminal_bench.yaml`    | terminal-bench@2.0   | Terminal commands        | reward      | OPENAI, ANTHROPIC, DAYTONA |

All benchmarks also require `TENSORZERO_AUTOPILOT_API_KEY`.
Please visit our website to [get access](https://www.tensorzero.com).
Harbor-based benchmarks (`lawbench`, `medagentbench`, `replicationbench`, `terminal-bench`) additionally require `DAYTONA_API_KEY` for sandboxed code execution.

## Configuration

Each YAML config specifies:

```yaml
autopilot_target:
  kind: prod # Autopilot environment (prod)
  api_key_env: "TENSORZERO_AUTOPILOT_API_KEY"

interlocutor:
  config_file: "interlocutor_config/tensorzero.toml" # LLM that answers Autopilot's questions

infra:
  gateway_binary_path: "/usr/local/bin/gateway"
  gateway_port: 3000

environments:
  - name: "ner_conllpp_v0"
    function_name: "ner_conllpp_v0::extract_entities"
    metric_name: "exact_match"
    initial_model: "openai::gpt-5-mini"
    num_iterations: 3 # Number of Autopilot optimization rounds
    episodes_per_iteration: 100 # Episodes per rollout
    episode_concurrency: 10 # Parallel episodes
    autopilot_max_turns: 70 # Max Autopilot conversation turns
    available_models: # Models Autopilot can experiment with
      - "anthropic::claude-haiku-4-5"
      - "openai::gpt-5-mini"
      - "google_ai_studio_gemini::gemini-3-flash-preview"
```

### CLI Options

```bash
autopilot-benchmark run \
  --config configs/ner.yaml \     # Config file (required)
  --env ner_conllpp_v0 \          # Run only this env (optional)
  --work-dir /app/output \        # Output directory
  --num-iterations 1 \            # Override iteration count
  --episodes 10 \                 # Override episode count
  --seed 42 \                     # RNG seed for reproducibility
  --verbose                       # Debug logging
```

## Output Artifacts

Each run writes to `output/<env>/<timestamp>/`:

```
output/ner_conllpp_v0/20260320T153000Z/
  config/tensorzero.toml              # Generated T0 config (updated each iteration)
  gateway/runtime.json                # Train gateway DB name and URL
  gateway/gateway.stdout.log          # Gateway logs
  gateway/test/runtime.json           # Test gateway DB name and URL
  autopilot/iteration_001/
    session_result.json               # Autopilot session outcome
    config_writes.raw.json            # Raw Autopilot config edits
    config_writes.flattened.json      # Flattened applied edits
  rollouts/iteration_001/post_autopilot/test/
    failed_episodes.jsonl             # Failed episode details
    episode_timings.jsonl             # Per-episode timing breakdown
  results/
    run.json                          # Run metadata and status
    iterations.json                   # Per-iteration metrics (train + test)
```

## Snapshots

Snapshots cache the baseline state so subsequent runs can skip the expensive baseline rollout.

```bash
# Create a snapshot
docker compose -f docker/docker-compose.yml --env-file .env run --rm eval \
  snapshot --config configs/ner.yaml --env ner_conllpp_v0 \
  --work-dir /app/output --snapshot-dir /app/snapshots --verbose

# Run from a snapshot (jumps straight to Autopilot iterations)
docker compose -f docker/docker-compose.yml --env-file .env run --rm eval \
  run --config configs/ner.yaml --env ner_conllpp_v0 \
  --snapshot /app/snapshots/ner_conllpp_v0/<timestamp>/ \
  --work-dir /app/output --verbose
```

## Adding a New Benchmark

1. Find an environment in [llmgym](https://github.com/tensorzero/llmgym) or create one.
2. Create a new YAML config following the pattern in `configs/`.
3. Set `function_name` to `<env_name>::<function>` and `metric_name` to the llmgym metric.
4. Run it: `docker compose -f docker/docker-compose.yml --env-file .env run --rm eval run --config configs/your_env.yaml --verbose`
