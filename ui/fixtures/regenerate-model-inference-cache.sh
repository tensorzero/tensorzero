#!/bin/bash

set -euxo pipefail

cd "$(dirname "$0")"/../

docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml down
docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml rm -f
TENSORZERO_INTERNAL_MOCK_PROVIDER_API=http://mock-provider-api:3030 TENSORZERO_SKIP_LARGE_FIXTURES=1 VITE_TENSORZERO_FORCE_CACHE_ON=1 docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml up --force-recreate -d
docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml wait fixtures
# Wipe the ModelInferenceCache table to ensure that we regenerate everything
docker run --add-host=host.docker.internal:host-gateway clickhouse/clickhouse-server clickhouse-client --host host.docker.internal --user chuser --password chpassword --database tensorzero_ui_fixtures 'TRUNCATE TABLE ModelInferenceCache SYNC'
# Don't use any retries, since this will pollute the model inference cache with duplicate entries.
export TENSORZERO_PLAYWRIGHT_RETRIES=0
export TENSORZERO_PLAYWRIGHT_NO_WEBSERVER=1
export TENSORZERO_GATEWAY_URL=http://localhost:3000
export TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero_ui_fixtures
pnpm test-e2e -j 1 --grep-invert "@credentials" --max-failures 1
# Remove the existing file if it exists (it may have restricted permissions from R2 download)
rm -f ./fixtures/model_inference_cache_e2e.jsonl
docker run --add-host=host.docker.internal:host-gateway clickhouse/clickhouse-server clickhouse-client --host host.docker.internal --user chuser --password chpassword --database tensorzero_ui_fixtures 'SELECT * FROM ModelInferenceCache ORDER BY long_cache_key ASC FORMAT JSONEachRow' > ./fixtures/model_inference_cache_e2e.jsonl
