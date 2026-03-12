#!/usr/bin/env bash
# Run this script once after cloning the repo to configure your local dev environment.
set -euo pipefail

echo "Installing pre-commit hooks..."
uvx pre-commit install

echo "Done! Pre-commit hooks are now active."
