# Feedback Load Tests

This directory contains a configurable load testing binary for TensorZero's feedback functionality using the `rlt` crate.

## Prerequisites

Run all commands below from the root of the repository.

1. **Start ClickHouse**: The test uses the same ClickHouse instance as the e2e tests:
   ```bash
   docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up --wait
   ```
   You'll also want to set the `TENSORZERO_CLICKHOUSE_URL` environment variable to point to the ClickHouse instance.

## Running the Load Test

```bash
cargo test-feedback-load -r 1000 --num-distinct-inferences 1000
# To see all CLI args
cargo test-feedback-load --help
```
