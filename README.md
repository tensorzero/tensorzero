# TensorZero

## Standing up a test environment

You can deploy TensorZero using Docker Compose.

```bash
docker compose up --build -d
```

This will deploy with the test config file at `api/tests/e2e/tensorzero.test.toml`.

You can also deploy with a custom config file by setting the `TENSORZERO_CONFIG_PATH` environment variable to the path of your config file.
