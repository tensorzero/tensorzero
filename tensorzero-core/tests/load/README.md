# Load Testing

- Install `vegeta` [→](https://github.com/tsenart/vegeta).
- Launch the ClickHouse instance:

  ```
  docker compose -f tensorzero-core/tests/load/docker-compose.yml up -d --build --force-recreate --remove-orphans
  ```

- Launch the mock inference provider:

  ```
  cargo run --profile performance --bin mock-inference-provider
  ```

- Launch the gateway.
  - With observability:

    ```
    cargo run --profile performance --bin gateway tensorzero-core/tests/load/tensorzero.toml
    ```

  - Without observability:

    ```
    cargo run --profile performance --bin gateway tensorzero-core/tests/load/tensorzero-without-observability.toml
    ```

- Then, you can run a load test with `sh path/to/test/run.sh`.
