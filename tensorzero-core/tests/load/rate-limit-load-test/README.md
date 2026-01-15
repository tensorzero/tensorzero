# Rate Limiting Load Tests

This directory contains a configurable load testing binary for TensorZero's rate limiting functionality using the `rlt` crate. The test supports two modes:

1. **Direct mode**: Every request hits the PostgreSQL database directly (original behavior)
2. **Pooled mode**: Uses in-memory token pool with adaptive pre-borrowing to reduce database contention

## Prerequisites

Run all commands below from the root of the repository.

1. **Start PostgreSQL**: The test uses the same PostgreSQL instance as the e2e tests:

   ```bash
   docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up --wait
   ```

2. **Environment Setup**: Set the database URL (optional, defaults to localhost):
   ```bash
   export TENSORZERO_POSTGRES_URL="postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests"
   ```

## Running the Load Test

### Direct Mode (DB every request)

```bash
cargo test-rate-limit-load --mode direct -r 1000 --requests-per-iteration 1 --contention-keys 0 -c 5
```

### Pooled Mode (in-memory token pool)

```bash
cargo test-rate-limit-load --mode pooled -r 1000 --requests-per-iteration 1 --contention-keys 0 -c 5
```

### Comparing Performance

Run both modes to compare database contention and latency:

```bash
# Direct mode baseline
cargo test-rate-limit-load --mode direct -r 5000 -c 10

# Pooled mode (should show fewer DB queries and lower latency)
cargo test-rate-limit-load --mode pooled -r 5000 -c 10
```

### All CLI Options

```bash
cargo test-rate-limit-load --help
```

On a recent MacBook Pro, direct mode achieves p99 latency of <5ms. Pooled mode should show significantly lower latency under high concurrency due to reduced database contention.
