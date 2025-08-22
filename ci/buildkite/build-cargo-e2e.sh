#!/bin/bash

set -euxo pipefail


# ------------------------------------------------------------------------------
# Setup Rust
# ------------------------------------------------------------------------------
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version

cargo build-e2e

tar -czvf target.e2e.tar.gz target
