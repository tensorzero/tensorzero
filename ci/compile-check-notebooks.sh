#!/usr/bin/env bash

# compile_notebooks.sh
#
# Command:   [files…]
# Default is to compile every notebook in $RECIPES_DIR if no paths are given.
#
# Behaviour:
#   • CI (no args) ➜ compile every notebook in $RECIPES_DIR
#   • pre‑commit (paths passed) ➜ operate only on touched files
#   • No files are added to the index; any changes stay unstaged.

set -euo pipefail
shopt -s nullglob

RECIPES_DIR="recipes"
JUPYTEXT="uvx jupytext@1.17.1"      # single authoritative version

###############################################################################

compile_notebooks () {
  local targets=("$@")
  if [[ ${#targets[@]} -eq 0 ]]; then
    mapfile -t targets < <(git ls-files "${RECIPES_DIR}/**/*_nb.py")
  fi

  echo "🔄 Compiling ${#targets[@]} notebook(s)…"
  local failed=()
  for nb in "${targets[@]}"; do
    [[ $nb != *_nb.py ]] && continue
    # Copy the notebook to a temporary file
    local target="${nb%_nb.py}.ipynb"
    local tmp_out="/tmp/compiled_notebook.ipynb"
    cp "$target" "$tmp_out"

    # Compile the Python script to a temporary notebook file and clean it with nb-clean
    $JUPYTEXT --to ipynb --update --set-formats "ipynb,py:percent" \
              --opt notebook_metadata_filter=-all --opt cell_metadata_filter=-all \
              --output "$tmp_out" "$nb"
    uvx nb-clean clean $tmp_out

    # Fail if the generated notebook doesn't match the current version of the notebook
    if ! diff -q "$target" "$tmp_out" >/dev/null; then
      failed+=("${target} does not match ${nb}")
    fi
  done

  if [[ ${#failed[@]} -gt 0 ]]; then
    echo ""
    echo "The following notebooks don't match the source script:"
    echo ""
    for out in "${failed[@]}"; do
      echo "✖ $out" >&2
    done
    echo ""
    echo "Please use one of the following options to fix the issue:"
    echo " • Compile the notebook to a script: ci/compile-notebook-to-script.sh path/to/notebook.ipynb"
    echo " • Compile the script to a notebook: ci/compile-script-to-notebook.sh path/to/script_nb.py"
    exit 1
  fi
}

###############################################################################

compile_notebooks "$@"
