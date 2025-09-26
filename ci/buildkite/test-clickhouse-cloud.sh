#!/bin/bash
set -euo pipefail
echo "CLICKHOUSE_ID=${CLICKHOUSE_ID}"

set +x
# Get the url of a ClickHouse cloud instance, based on the $CLICKHOUSE_ID that we assigned in '.buildkite/hooks/post-checkout'
# We also use this id as part of the concurrent group key in '.buildkite/pipeline.yml', so we're guaranteed to be the only job trying to use this instance.
# This allows us to distribute our tests across multiple ClickHouse cloud instances (since running too many parallel tests will overload any single instance).
export CLICKHOUSE_API_KEY=$(buildkite-agent secret get clickhouse_api_key)
export CLICKHOUSE_KEY_SECRET=$(buildkite-agent secret get clickhouse_key_secret)
export CLICKHOUSE_USERNAME=$(buildkite-agent secret get clickhouse_username)
export CLICKHOUSE_PASSWORD=$(buildkite-agent secret get clickhouse_password)
# We concatenate our clickhouse instance prefix, along with our chosen clickhouse id (e.g. 'dev-tensorzero-e2e-tests-instance-' and '0'), to form the instance name
# Then, we look up the instance url for this name, and add basic-auth credentials to the url to get our full TENSORZERO_CLICKHOUSE_URL
# The 'export' statements go on separate lines to prevent the return code from the $() command from being ignored
CURL_OUTPUT=$(curl --user "$CLICKHOUSE_API_KEY:$CLICKHOUSE_KEY_SECRET" https://api.clickhouse.cloud/v1/organizations/b55f1935-803f-4931-90b3-4d26089004d4/services)
echo "ClickHouse API response: $CURL_OUTPUT"
TENSORZERO_CLICKHOUSE_URL=$(echo "$CURL_OUTPUT" | jq -r ".result[] | select(.name == \"${CLICKHOUSE_PREFIX}${CLICKHOUSE_ID}\") | .endpoints[] | select(.protocol == \"https\") | \"https://$CLICKHOUSE_USERNAME:$CLICKHOUSE_PASSWORD@\" + .host + \":\" + (.port | tostring)")
export TENSORZERO_CLICKHOUSE_URL
echo $TENSORZERO_CLICKHOUSE_URL
CLICKHOUSE_HOST=$(echo $TENSORZERO_CLICKHOUSE_URL | sed 's|https://[^@]*@||' | sed 's|:[0-9]*||')
export CLICKHOUSE_HOST

# Generate unique database name with random suffix for isolation
RANDOM_SUFFIX=$(openssl rand -hex 4)
export TENSORZERO_E2E_TESTS_DATABASE="tensorzero_e2e_tests_${RANDOM_SUFFIX}"
echo "Using database name: $TENSORZERO_E2E_TESTS_DATABASE"

# Set up cleanup function to run on exit
cleanup_database() {
    if [ -n "${TENSORZERO_E2E_TESTS_DATABASE:-}" ] && [ -n "${TENSORZERO_CLICKHOUSE_URL:-}" ]; then
        echo "Cleaning up database: $TENSORZERO_E2E_TESTS_DATABASE"
        curl -X POST "${TENSORZERO_CLICKHOUSE_URL%/}/?param_target=$TENSORZERO_E2E_TESTS_DATABASE" \
            --data-binary "DROP DATABASE IF EXISTS {target:Identifier}" || true
        echo "Cleanup completed for database: $TENSORZERO_E2E_TESTS_DATABASE"
    fi
    cat e2e_logs.txt || echo "e2e logs don't exist"
}

# Register cleanup function to run on script exit (success or failure)
trap cleanup_database EXIT

set -x


# Install ClickHouse client and load fixtures into remote ClickHouse
# Install prerequisite packages
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
# Download the ClickHouse GPG key and store it in the keyring
curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
# Get the system architecture
ARCH=$(dpkg --print-architecture)
# Add the ClickHouse repository to apt sources
echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb stable main" | sudo tee /etc/apt/sources.list.d/clickhouse.list
# Update apt package lists
sudo apt-get update
sudo apt-get install -y clickhouse-client

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
    max_attempts=90
    while ! curl -s -f http://localhost:3000/health >/dev/null 2>&1; do
        echo "Waiting for gateway to be healthy..."
        sleep 1
        count=$((count + 1))
        if [ $count -ge $max_attempts ]; then
        echo "Gateway failed to become healthy after $max_attempts attempts"
        cat e2e_logs.txt
        exit 1
        fi
    done
    export GATEWAY_PID=$!

export CLICKHOUSE_USER="$CLICKHOUSE_USERNAME"
export CLICKHOUSE_PASSWORD="$CLICKHOUSE_PASSWORD"
cd ui/fixtures
./load_fixtures.sh $TENSORZERO_E2E_TESTS_DATABASE
cd ../..
sleep 2

cargo test-e2e-no-creds --no-fail-fast -- --skip test_concurrent_clickhouse_migrations
cat e2e_logs.txt
