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
cargo test-unit --profile ci-unit | buildkite-test-collector
