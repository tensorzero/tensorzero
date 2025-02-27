#!/bin/bash

set -euxo pipefail
uv run maturin develop --uv --features e2e_tests
uv run pytest -n auto $@
