# Load Testing

- Install `vegeta` [→](https://github.com/tsenart/vegeta).
- Launch the ClickHouse instance:

  ```
  docker compose -f gateway/tests/load/docker-compose.yml up -d --build --force-recreate --remove-orphans
  ```

- Launch the mock inference provider:

  ```
  cargo run --release --bin mock-inference-provider
  ```

- Launch the gateway:

  ```
  cargo run --release --bin gateway gateway/tests/load/tensorzero.toml
  ```

- Then, you can run a load test with `sh path/to/test/run.sh`.
