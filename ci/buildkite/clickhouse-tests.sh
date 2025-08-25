# Set BUILDKITE_ANALYTICS_TOKEN
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get CLICKHOUSE_TESTS_ANALYTICS_ACCESS_TOKEN)

if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi

# Check for required TENSORZERO_CLICKHOUSE_VERSION environment variable
if [ -z "$TENSORZERO_CLICKHOUSE_VERSION" ]; then
    echo "Error: TENSORZERO_CLICKHOUSE_VERSION environment variable is required"
    exit 1
fi

# ------------------------------------------------------------------------------
# Set the short commit hash
# ------------------------------------------------------------------------------
SHORT_HASH=${BUILDKITE_COMMIT:0:7}


# Install cargo-binstall
curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# TODO: handle replication (for merge queue checks)
# if env var REPLICATED="1" then set TENSORZERO_CLICKHOUSE_CLUSTER_NAME=tensorzero_e2e_tests_cluster in the environment of the script

# Get the fixtures
buildkite-agent artifact download fixtures.tar.gz ui/fixtures
tar -xzvf ui/fixtures/fixtures.tar.gz

# ------------------------------------------------------------------------------
# Docker: download & load gateway container
# ------------------------------------------------------------------------------
buildkite-agent artifact download gateway-container.tar .
docker load < gateway-container.tar

# ------------------------------------------------------------------------------
# Docker: download & load mock-inference-provider container
# ------------------------------------------------------------------------------
buildkite-agent artifact download mock-inference-provider-container.tar .
docker load < mock-inference-provider-container.tar

# ------------------------------------------------------------------------------
# Setup Rust
# ------------------------------------------------------------------------------
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version

cargo build-e2e

# Install Python
# apt update && apt install -y python3.13-dev

# Install cargo-nextest
# TODO: install the pre-built package instead of building from source
cargo binstall -y cargo-nextest --secure

export TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero_e2e_tests"
export TENSORZERO_SKIP_LARGE_FIXTURES=1
export TENSORZERO_GATEWAY_TAG=sha-$SHORT_HASH
# Default version was "24.12-alpine" - now required as environment variable
export TENSORZERO_MOCK_INFERENCE_PROVIDER_TAG=sha-$SHORT_HASH
# TODO: handle replication
# We'll need to set the cluster name environment variable

# Launch non-replicated ClickHouse containers for E2E tensorzero_e2e_tests
docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up clickhouse gateway fixtures mock-inference-provider --wait

CLICKHOUSE_VERSION=$(curl -s "http://localhost:8123/query?user=chuser&password=chpassword" --data-binary "SELECT version()")
echo "Requested clickhouse version: $TENSORZERO_CLICKHOUSE_VERSION"
echo "Detected clickhouse version: $CLICKHOUSE_VERSION"

# Launch the gateway for E2E tests
cargo run-e2e > e2e_logs.txt 2>&1 &
count=0
max_attempts=20
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

cargo test-e2e-no-creds

# Upload the test JUnit XML files
curl -X POST \
  -H "Authorization: Token token=$BUILDKITE_ANALYTICS_TOKEN" \
  -F "format=junit" \
  -F "data=@target/nextest/clickhouse/junit.xml" \
  -F "run_env[CI]=buildkite" \
  -F "run_env[key]=$BUILDKITE_BUILD_ID" \
  -F "run_env[number]=$BUILDKITE_BUILD_NUMBER" \
  -F "run_env[job_id]=$BUILDKITE_JOB_ID" \
  -F "run_env[branch]=$BUILDKITE_BRANCH" \
  -F "run_env[commit_sha]=$BUILDKITE_COMMIT" \
  -F "run_env[message]=$BUILDKITE_MESSAGE" \
  -F "run_env[url]=$BUILDKITE_BUILD_URL" \
  https://analytics-api.buildkite.com/v1/uploads
