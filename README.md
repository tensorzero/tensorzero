# TensorZero

## Local Development

- Install `pre-commit` [(→)](https://pre-commit.com/#installation) and run `pre-commit install`
- Install `cargo-deny` [(→)](https://github.com/EmbarkStudios/cargo-deny): `cargo install --locked cargo-deny`

To run locally with Docker:

```sh
docker run --name tensorzero-gateway \
  -v "./config:/app/config" \
  --env-file example.env \
  -p 3000:3000 \
  tensorzero/gateway
```

To run locally without Docker:

```sh
cargo run --release --bin gateway -- path/to/tensorzero.toml
```

## Testing

We use [cargo-nextest](https://nexte.st/book/introduction.html) for faster and more robust testing.

- Run unit tests: `cargo test-unit`
- Run E2E tests: `cargo test-e2e`
  - See `gateway/tests/e2e/README.md` for more details on running E2E tests (e.g. environment setup).
- Run all tests: `cargo test-all`

## ClickHouse Cloud

- Set up ClickHouse Cloud [(→)](https://clickhouse.com/cloud)
- Set the environment variable:
  ```
  CLICKHOUSE_URL="https://USERNAME:PASSWORD@XXXXXXXX.XXXXXXXX.XXX.clickhouse.cloud:8443"
  ```
