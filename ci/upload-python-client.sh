#!/usr/bin/env bash
set -euxo pipefail

cd "$(dirname "$0")"
curl -LsSf https://astral.sh/uv/0.6.4/install.sh | sh
cd ../clients/python
uv run maturin upload -v --repository $1 --non-interactive --skip-existing ../../wheels-*/*
