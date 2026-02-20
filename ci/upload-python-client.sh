#!/usr/bin/env bash
set -euxo pipefail

cd "$(dirname "$0")"
curl -LsSf https://astral.sh/uv/0.9.27/install.sh | sh
cd ../clients/python
uv sync
uv run maturin upload -v --repository $1 --non-interactive --skip-existing ../../wheels-*/*
