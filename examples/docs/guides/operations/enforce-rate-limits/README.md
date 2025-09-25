docker run -e TENSORZERO_CLICKHOUSE_URL=${TENSORZERO_CLICKHOUSE_URL:?} -e TENSORZERO_POSTGRES_URL=${TENSORZERO_POSTGRES_URL:?} tensorzero/gateway --run-migrations-only

docker run -e TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@clickhouse:8123/tensorzero -e TENSORZERO_POSTGRES_URL=postgres://postgres:postgres@postgres:5432/tensorzero tensorzero/gateway --run-migrations-only

docker compose run --rm gateway --run-migrations-only
