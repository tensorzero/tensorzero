#!/bin/bash
set -euxo pipefail

export R2_ACCESS_KEY_ID=$(buildkite-agent secret get R2_ACCESS_KEY_ID)
if [ -z "$R2_ACCESS_KEY_ID" ]; then
    echo "Error: R2_ACCESS_KEY_ID is not set"
    exit 1
fi

export R2_SECRET_ACCESS_KEY=$(buildkite-agent secret get R2_SECRET_ACCESS_KEY)
if [ -z "$R2_SECRET_ACCESS_KEY" ]; then
    echo "Error: R2_SECRET_ACCESS_KEY is not set"
    exit 1
fi

# Install `uv`
curl -LsSf https://astral.sh/uv/0.6.17/install.sh | sh
source $HOME/.local/bin/env
# Download the fixtures
uv run ./ui/fixtures/download-large-fixtures.py
uv run ./ui/fixtures/download-small-fixtures.py
# Zip the fixtures
tar -czvf fixtures.tar.gz ./ui/fixtures/large-fixtures ./ui/fixtures/small-fixtures/*.jsonl
