# Rate Limiting Load Tests

This directory contains a configurable load testing binary for TensorZero's rate limiting functionality using the `rlt` crate. The test focuses on database-level performance of the PostgreSQL-based rate limiting implementation.

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

```bash
cargo test-rate-limit-load -r 1000 --requests-per-iteration 1 --contention-keys 0 -c 5
# To see all CLI args
cargo test-rate-limit-load --help
```

On a recent MacBook Pro, the first command achieves p99 latency of <5ms.
