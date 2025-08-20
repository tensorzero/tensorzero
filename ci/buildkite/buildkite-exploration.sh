# Install cargo-binstall
curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
# Install uv
echo "\n\n"
echo $PATH
echo "\n\n"
curl -LsSf https://astral.sh/uv/0.6.17/install.sh | sh
echo "\n\n"
echo $PATH
echo "\n\n"
export PATH=$PATH:$HOME/.local/bin
echo "\n\n"
echo $PATH
echo "\n\n"
# Install Python
uv python install
# Install cargo-nextest
# TODO: install the pre-built package instead of building from source
cargo binstall -y cargo-nextest --secure
# Run unit tests using cargo-nextest
cargo test-unit
