# Set BUILDKITE_ANALYTICS_TOKEN
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get CI_UNIT_BUILDKITE_ANALYTICS_TOKEN)

if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi

# Install cargo-binstall
curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# Install Python
apt update && apt install -y python3.13-dev

# Install cargo-nextest
# TODO: install the pre-built package instead of building from source
cargo binstall -y cargo-nextest --secure

# Run unit tests using cargo-nextest
cargo test-unit --profile ci-unit

# Upload the test JUnit XML files
curl -X POST \
  -H "Authorization: Token token=$BUILDKITE_ANALYTICS_TOKEN" \
  -F "format=junit" \
  -F "data=@target/nextest/ci-unit/junit.xml" \
  -F "run_env[CI]=buildkite" \
  -F "run_env[key]=$BUILDKITE_BUILD_ID" \
  -F "run_env[number]=$BUILDKITE_BUILD_NUMBER" \
  -F "run_env[job_id]=$BUILDKITE_JOB_ID" \
  -F "run_env[branch]=$BUILDKITE_BRANCH" \
  -F "run_env[commit_sha]=$BUILDKITE_COMMIT" \
  -F "run_env[message]=$BUILDKITE_MESSAGE" \
  -F "run_env[url]=$BUILDKITE_BUILD_URL" \
  https://analytics-api.buildkite.com/v1/uploads
