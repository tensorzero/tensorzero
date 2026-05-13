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

JUPYTEXT="uvx jupytext@1.17.1"
NB_CLEAN="uvx nb-clean@4.0.1"
RUFF="uvx ruff@0.14.0"

###############################################################################

compile_notebooks () {
  local failed_nb=()
  local failed_scripts=()

  # Check if changed scripts match the notebooks

  # Handle pre-commit
  local source_scripts=("$@")
  if [[ ${#source_scripts[@]} -eq 0 ]]; then
    mapfile -t source_scripts < <(git ls-files "${RECIPES_DIR}/**/*_nb.py")
  fi

  echo "🔄 Checking modified scripts..."

  for source_script in "${source_scripts[@]}"; do
    [[ $source_script != *_nb.py ]] && continue
    local target_nb="${source_script%_nb.py}.ipynb"
    local tmp_nb="$(mktemp).ipynb"
    cp "$target_nb" "$tmp_nb"

    # Compile the Python script to a temporary notebook file and clean it with nb-clean
    $JUPYTEXT --to ipynb --update --set-formats "ipynb,py:percent" \
              --opt notebook_metadata_filter=-all --opt cell_metadata_filter=-all \
              --output "$tmp_nb" "$source_script"

    $NB_CLEAN clean "$tmp_nb"

    # Fail if the generated notebook doesn't match the current version of the notebook
    if ! diff -q "$target_nb" "$tmp_nb" >/dev/null; then
      failed_scripts+=("${source_script}")
      failed_nb+=("${target_nb}")
    fi
  done

  # Check if changed notebooks match the scripts

  # Handle pre-commit
  local source_nbs=("$@")
  if [[ ${#source_nbs[@]} -eq 0 ]]; then
    mapfile -t source_nbs < <(git ls-files "${RECIPES_DIR}/**/*.ipynb")
  fi

  echo
  echo "🔄 Checking modified notebooks..."

  for source_nb in "${source_nbs[@]}"; do
    [[ $source_nb != *.ipynb ]] && continue
    local target_script="${source_nb/.ipynb/_nb.py}"

    # Skip if the script has already failed
    if [[ ${#failed_scripts[@]} -gt 0 ]]; then
      for failed_script in "${failed_scripts[@]}"; do
        if [[ "$target_script" == "$failed_script" ]]; then
          continue 2
        fi
      done
    fi

    local tmp_script="$(mktemp)_nb.py"
    cp "$target_script" "$tmp_script"

    # Clean the notebook first
    $NB_CLEAN clean "$source_nb"

    # Compile the notebook to Python script
    $JUPYTEXT --to py:percent --opt notebook_metadata_filter=-all --opt cell_metadata_filter=-all \
              --output "$tmp_script" "$source_nb"

    # Run ruff on the generated script
    $RUFF format "$tmp_script"

    # Fail if the generated notebook doesn't match the current version of the notebook
    if ! diff -q "$target_script" "$tmp_script" >/dev/null; then
      failed_scripts+=("${target_script}")
      failed_nb+=("${source_nb}")
    fi
  done

  # Check if changed notebooks match the scripts
  if [[ ${#failed_nb[@]} -gt 0 ]]; then
    echo ""
    echo "The following notebooks don't match the source script:"
    echo ""
    for i in "${!failed_nb[@]}"; do
      echo "✖ ${failed_nb[$i]} <> ${failed_scripts[$i]}" >&2
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
