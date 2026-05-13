#!/usr/bin/env bash

set -euo pipefail

JUPYTEXT="uvx jupytext@1.17.1"
NB_CLEAN="uvx nb-clean@4.0.1"
RUFF="uvx ruff@0.14.0"

###############################################################################

compile_notebook_to_script () {
  local nb="$1"
  [[ $nb != *.ipynb ]] && { echo "Error: File must be a .ipynb file" >&2; exit 1; }

  echo "🔄 Compiling notebook to script: $nb"
  local target="${nb%.ipynb}_nb.py"

  # Clean the notebook first
  $NB_CLEAN clean "$nb"

  # Compile the notebook to Python script
  $JUPYTEXT --to py:percent --opt notebook_metadata_filter=-all --opt cell_metadata_filter=-all \
            --output "$target" "$nb"

  # Run ruff on the generated script
  $RUFF format "$target"
}

###############################################################################

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 path/to/notebook.ipynb" >&2
  exit 1
fi

compile_notebook_to_script "$1"
