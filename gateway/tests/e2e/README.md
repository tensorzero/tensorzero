# E2E Tests

# Running E2E Tests

Before running the tests, you should spin up the gateway and ClickHouse containers using Docker Compose:

```sh
docker compose -f gateway/tests/e2e/docker-compose.yml up -d --build --force-recreate --remove-orphans  --wait
```

Then, you can run the tests using Cargo:

```sh
cargo test --features e2e_tests
```

If you want to run the tests against a different gateway address, you can set the `GATEWAY_URL` environment variable to the URL of the gateway:

```sh
GATEWAY_URL="http://localhost:1234" cargo test --features e2e_tests
```
