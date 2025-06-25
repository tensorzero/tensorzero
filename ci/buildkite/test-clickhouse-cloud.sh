#!/bin/bash
set -euxo pipefail
echo "CLICKHOUSE_ID=${CLICKHOUSE_ID}"
# Get the url of a ClickHouse cloud instance, based on the $CLICKHOUSE_ID that we assigned in '.buildkite/hooks/post-checkout'
# We also use this id as part of the concurrent group key in '.buildkite/pipeline.yml', so we're guaranteed to be the only job trying to use this instance.
# This allows us to distribute our tests across multiple ClickHouse cloud instances (since running too many parallel tests will overload any single instance).
export CLICKHOUSE_API_KEY=$(buildkite-agent secret get clickhouse_api_key)
export CLICKHOUSE_KEY_SECRET=$(buildkite-agent secret get clickhouse_key_secret)
export TENSORZERO_CLICKHOUSE_SERVICE=$(curl --user "$CLICKHOUSE_API_KEY:$CLICKHOUSE_KEY_SECRET" https://api.clickhouse.cloud/v1/organizations/b55f1935-803f-4931-90b3-4d26089004d4/services | jq ".result[] | select(.name == \"dev-tensorzero-e2e-tests-instance-${CLICKHOUSE_ID}\") | .id")
echo $TENSORZERO_CLICKHOUSE_URL | buildkite-agent redactor add
curl "$TENSORZERO_CLICKHOUSE_URL" --data-binary 'SHOW DATABASES'
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  -s -- -y
. "$HOME/.cargo/env"
curl -LsSf https://astral.sh/uv/0.6.17/install.sh | sh
source $HOME/.local/bin/env
curl -LsSf https://get.nexte.st/latest/linux | tar zxf - -C ~/.cargo/bin
uv run ./ui/fixtures/download-fixtures.py
./ci/delete-clickhouse-dbs.sh
cargo build-e2e
cargo run-e2e > e2e_logs.txt 2>&1 &
    count=0
    max_attempts=30
    while ! curl -s -f http://localhost:3000/health >/dev/null 2>&1; do
        echo "Waiting for gateway to be healthy..."
        sleep 1
        count=$((count + 1))
        if [ $count -ge $max_attempts ]; then
        echo "Gateway failed to become healthy after $max_attempts attempts"
        exit 1
        fi
    done
    export GATEWAY_PID=$!
cargo test-e2e-no-creds -- --skip test_concurrent_clickhouse_migrations
cat e2e_logs.txt