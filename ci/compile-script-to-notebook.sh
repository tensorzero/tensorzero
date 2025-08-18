#!/usr/bin/env bash

set -euo pipefail

JUPYTEXT="uvx jupytext@1.17.1"
NB_CLEAN="uvx nb-clean@4.0.1"

###############################################################################

compile_script_to_notebook () {
  local script="$1"
  [[ $script != *_nb.py ]] && { echo "Error: File must end with _nb.py" >&2; exit 1; }

  echo "🔄 Compiling script: $script"
  local target="${script%_nb.py}.ipynb"

  # Compile the Python script to notebook and clean it with nb-clean
  $JUPYTEXT --to ipynb --update --set-formats "ipynb,py:percent" \
            --opt notebook_metadata_filter=-all --opt cell_metadata_filter=-all \
            --output "$target" "$script"

  # Clean the notebook
  $NB_CLEAN clean "$target"
}

###############################################################################

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 path/to/script_nb.py" >&2
  exit 1
fi

compile_script_to_notebook "$1"
