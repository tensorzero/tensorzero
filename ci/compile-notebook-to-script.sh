#!/usr/bin/env bash

set -euo pipefail

JUPYTEXT="uvx jupytext@1.17.1"      # single authoritative version

###############################################################################

compile_notebook_to_script () {
  local nb="$1"
  [[ $nb != *.ipynb ]] && { echo "Error: File must be a .ipynb file" >&2; exit 1; }

  echo "🔄 Compiling notebook to script: $nb"
  local target="${nb%.ipynb}_nb.py"

  # Compile the notebook to Python script
  $JUPYTEXT --to py:percent --opt notebook_metadata_filter=-all --opt cell_metadata_filter=-all \
            --output "$target" "$nb"
}

###############################################################################

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 path/to/notebook.ipynb" >&2
  exit 1
fi

compile_notebook_to_script "$1"
