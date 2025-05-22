#!/usr/bin/env bash
# compile_notebooks.sh
#
# Command:   [files…]
# Default is to compile every notebook in $RECIPES_DIR if no paths are given.
#
# Behaviour:
#   • CI (no args)  ➜ compile every notebook in $RECIPES_DIR
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
    local out="${nb%_nb.py}.ipynb"
    $JUPYTEXT --to ipynb --update --set-formats "ipynb,py:percent" \
              --opt notebook_metadata_filter=-all --opt cell_metadata_filter=-all \
              --output "$out" "$nb"
    # Run nb-clean on the generated notebook
    uvx nb-clean clean $out
    # ─── fail if the generated notebook isn't in sync ──────────────────────────
    if ! git diff --quiet -- "$out"; then
      failed+=("$out")
    fi
  done

  if [[ ${#failed[@]} -gt 0 ]]; then
    for out in "${failed[@]}"; do
      echo "✖ $out is not up to date; please run $0 to recompile notebooks" >&2
    done
    exit 1
  fi
}

###############################################################################
compile_notebooks "$@"
