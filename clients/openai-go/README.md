# `openai-go` Client Compatibility Tests

This directory contains tests for TensorZero's OpenAI API compatibility using the official OpenAI Go client.

## Setup

1. Make sure you have Go installed on your system
2. Make sure TensorZero is running locally on port 3000 with the E2E test fixtures
   - From the root of the repository, run `docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up --force-recreate --build`
   - In a separate terminal, run `cargo run-e2e`

## Testing

```bash
cd tests
go test -v
```
