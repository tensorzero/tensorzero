# `openai-node` Client Compatibility Tests

This directory contains tests for OpenAI compatibility using the official OpenAI Node.js client.

## Setup

1. Install the dependencies: `pnpm install`
2. Make sure TensorZero is running locally on port 3000 with the E2E test fixtures
   - From the root of the repository, run `docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up --force-recreate --build`
   - In a separate terminal, run `cargo run-e2e`

## Testing

```bash
pnpm typecheck
pnpm test
```
