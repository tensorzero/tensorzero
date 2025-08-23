#!/bin/bash

# Install `uv`
curl -LsSf https://astral.sh/uv/0.6.17/install.sh | sh
source $HOME/.local/bin/env
# Download the fixtures
uv run ./ui/fixtures/download-fixtures.py
# Zip the fixtures
tar -czvf fixtures.tar.gz ./ui/fixtures/s3-fixtures
