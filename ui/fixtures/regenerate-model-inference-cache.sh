#!/bin/bash

set -euxo pipefail

cd "$(dirname "$0")"/../

docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml down
docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml rm -f
OPENAI_BASE_URL=http://mock-inference-provider:3030/openai/ FIREWORKS_BASE_URL=http://mock-inference-provider:3030/fireworks/ FIREWORKS_ACCOUNT_ID=fake_fireworks_account TENSORZERO_SKIP_LARGE_FIXTURES=1 docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml up --force-recreate -d
docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml wait fixtures
TENSORZERO_PLAYWRIGHT_NO_WEBSERVER=1 TENSORZERO_GATEWAY_URL=http://localhost:3000 TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero_ui_fixtures pnpm test-e2e
clickhouse-client --user chuser --password chpassword --database tensorzero_ui_fixtures 'SELECT * FROM ModelInferenceCache ORDER BY long_cache_key ASC FORMAT JSONEachRow' > ./fixtures/model_inference_cache_examples.jsonl
docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml down
docker compose -f ./fixtures/docker-compose.e2e.yml -f ./fixtures/docker-compose.ui.yml rm
