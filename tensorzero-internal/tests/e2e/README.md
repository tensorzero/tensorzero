# E2E Tests

# Running E2E Tests

- Spin up the ClickHouse container using Docker Compose:

  ```sh
  docker compose -f gateway/tests/e2e/docker-compose.yml up -d --build --force-recreate --remove-orphans  --wait
  ```

- Set the `TENSORZERO_CLICKHOUSE_URL` environment variable to the URL of the ClickHouse container (e.g. `TENSORZERO_CLICKHOUSE_URL=http://localhost:8123`).

- Spin up the gateway: `cargo run-e2e` or `cargo watch-e2e`

- Run the tests: `cargo test-e2e`

  - If you want to run the tests against a different gateway address, you can set the `GATEWAY_URL` environment variable to the URL of the gateway:

    ```sh
    GATEWAY_URL="http://localhost:1234" cargo test-e2e
    ```
