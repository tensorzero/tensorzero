# Set BUILDKITE_ANALYTICS_TOKEN
echo $BUILDKITE_ANALYTICS_TOKEN
echo "AAA"
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get CI_UNIT_BUILDKITE_ANALYTICS_TOKEN)
echo $BUILDKITE_ANALYTICS_TOKEN
echo "XXX"
echo $(buildkite-agent secret get CI_UNIT_BUILDKITE_ANALYTICS_TOKEN)
echo "YYY"
exit 1

# Install cargo-binstall
curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# Install Python
apt update && apt install -y python3.13-dev

# Install cargo-nextest
# TODO: install the pre-built package instead of building from source
cargo binstall -y cargo-nextest --secure

# Install the BuildKite test collector
cargo install buildkite-test-collector

# Run unit tests using cargo-nextest
cargo test-unit --profile ci-unit
