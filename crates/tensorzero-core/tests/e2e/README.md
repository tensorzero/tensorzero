# E2E Tests

## Running E2E Tests

- Spin up the ClickHouse container using Docker Compose.

  For local development, set these variables to avoid requiring cloud fixture credentials and to skip downloading large fixtures:

  ```sh
  export TENSORZERO_DOWNLOAD_FIXTURES_WITHOUT_CREDENTIALS=1
  export TENSORZERO_SKIP_LARGE_FIXTURES=1
  docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up -d --build --force-recreate --remove-orphans --wait
  ```

- Set the `TENSORZERO_CLICKHOUSE_URL` environment variable to the URL of the ClickHouse container:

  ```sh
  export TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests
  ```

  The ClickHouse Docker container uses:
  - user: `chuser`
  - password: `chpassword`
  - database: `tensorzero_e2e_tests`

- Spin up the gateway: `cargo run-e2e` or `cargo watch-e2e`

- Run the tests: `cargo test-e2e`
  - If you want to run the tests against a different gateway address, you can set the `GATEWAY_URL` environment variable to the URL of the gateway:

    ```sh
    GATEWAY_URL="http://localhost:1234" cargo test-e2e
    ```

## Notes

- Tests involving gateway relay should go in `gateway/tests/relay` instead of `tensorzero-core`.
