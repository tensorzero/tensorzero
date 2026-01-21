# Postgres Development

## Local Development

For developing Postgres features (writing queries, testing locally):

```bash
# 1. Set environment variables
export TENSORZERO_POSTGRES_URL="postgres://postgres:postgres@localhost:5432/postgres_migration_dev"
export TENSORZERO_INTERNAL_FLAG_ENABLE_POSTGRES_READ=1

# 2. Reset Postgres and load fixture data
./ui/fixtures/reset-dev-postgres.sh
```

### Rebuilding SQLx Cache

When you edit SQL queries, rebuild the sqlx cache:

```bash
DATABASE_URL="postgres://postgres:postgres@localhost:5432/postgres_migration_dev" cargo sqlx prepare
```

We build sqlx cache within each crate (see https://github.com/tensorzero/tensorzero/pull/5622).

## Running E2E Tests

### With New Migrations (Fast Workflow)

When you add new migration files, run migrations via `cargo` instead of rebuilding the Docker container:

```bash
# 1. (Optional) Remove existing Postgres container if iterating on migrations
docker compose -f tensorzero-core/tests/e2e/docker-compose.yml down -v postgres

# 2. Start Postgres and ClickHouse
docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up -d postgres clickhouse

# 3. Wait for Postgres to be ready
until docker compose -f tensorzero-core/tests/e2e/docker-compose.yml exec postgres pg_isready -U postgres; do sleep 1; done

# 4. Run migrations via cargo (uses your local code)
TENSORZERO_POSTGRES_URL=postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests cargo migrate-postgres

# 5. Load Postgres fixtures
TENSORZERO_POSTGRES_URL=postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests ./ui/fixtures/load_fixtures_postgres.sh

# 6. Run Postgres E2E tests
TENSORZERO_POSTGRES_URL=postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests \
  cargo test --package tensorzero-core --features e2e_tests postgres::inference_count_queries
```

### Without New Migrations (Full Docker Compose)

When using existing migrations (no local changes), use the standard docker-compose workflow:

```bash
# 1. Start all services (uses published gateway image for migrations)
docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up -d

# 2. Wait for fixtures to load
docker compose -f tensorzero-core/tests/e2e/docker-compose.yml ps  # Check health status

# 3. Run tests
TENSORZERO_POSTGRES_URL=postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests \
TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests \
  cargo test --package tensorzero-core --features e2e_tests
```

### With New Migrations (Slow Alternative)

If you need the full docker-compose orchestration with new migrations:

```bash
# 1. Rebuild gateway image with new migrations (~5 min)
docker compose -f tensorzero-core/tests/e2e/docker-compose.yml build gateway-postgres-migrations

# 2. Start all services
docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up -d
```

## CI Behavior

CI sets `TENSORZERO_GATEWAY_TAG=sha-${{ github.sha }}`, which uses the freshly-built image from the PR's commit. The container rebuild is only slow for local development.
