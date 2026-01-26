#!/bin/bash
set -euxo pipefail

# Install `uv`
curl -LsSf https://astral.sh/uv/0.6.17/install.sh | sh
source $HOME/.local/bin/env
# Download the fixtures
uv run ./ui/fixtures/download-large-fixtures.py
uv run ./ui/fixtures/download-small-fixtures.py
# Zip the fixtures
tar -czvf fixtures.tar.gz ./ui/fixtures/large-fixtures ./ui/fixtures/small-fixtures/*.jsonl
