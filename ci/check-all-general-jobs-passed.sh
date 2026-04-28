#!/usr/bin/env bash
set -euo pipefail

# This script checks that all jobs in the current GitHub Actions workflow run
# have passed, except for jobs in the ALLOWED_FAIL and ALLOWED_SKIP lists.
#
# It queries the GitHub API to discover all jobs dynamically, rather than
# relying solely on the `needs` list in the workflow YAML.
#
# Required environment variables:
#   GH_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_EVENT_NAME

# Jobs that are allowed to fail (their failure does not block merging).
# These jobs show up in the PR but are intentionally non-blocking.
ALLOWED_FAIL=(
  "cargo-deny"
)

# Jobs that are allowed to be skipped even in the merge queue.
# In PR CI, any job may be skipped (e.g., due to path filters or fork conditions).
# In the merge queue, only jobs in this list may be skipped — all others must run.
ALLOWED_SKIP=(
  "build-fixtures-container"
  "build-gateway-container"
  "build-gateway-e2e-container"
  "build-live-tests-container"
  "build-mock-provider-api-container"
  "build-provider-proxy-container"
  "build-ui-container"
  "build-windows"
  "cargo-deny"
  "check-docker-compose-commit"
  "check-docker-compose-released"
  "check-docs-broken-links"
  "check-if-edited-then-edit"
  "check-node-bindings"
  "check-production-deployment-docker-compose"
  "check-python-client-build"
  "check-python-schemas"
  "check-version-consistency"
  "clickhouse-tests"
  "client-tests"
  "db-only-boot-e2e"
  "inference-cache-tests"
  "Endpoint tests"
  "k3d"
  "lint-rust"
  "live-tests"
  "live-tests-config-in-database"
  "minikube"
  "mock-optimization-tests"
  "mocked-batch-tests"
  "Postgres tests"
  "rust-build"
  "rust-test"
  "ui-tests"
  "ui-tests-e2e"
  "validate"
  "validate-node"
  "validate-python"
)

SELF_JOB="check-all-general-jobs-passed"

# Check if a job name matches an entry in a list.
# Handles exact matches, matrix jobs ("entry (...)"), and reusable workflow jobs ("entry / ...").
matches_list() {
  local job_name="$1"
  shift
  for entry in "$@"; do
    if [[ "$job_name" == "$entry" || "$job_name" == "$entry ("* || "$job_name" == "$entry / "* ]]; then
      return 0
    fi
  done
  return 1
}

main() {
  echo "Fetching jobs for run ${GITHUB_RUN_ID} attempt ${GITHUB_RUN_ATTEMPT}..."
  echo ""

  local jobs
  jobs=$(gh api --paginate \
    "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/attempts/${GITHUB_RUN_ATTEMPT}/jobs" \
    --jq '.jobs[] | {name: .name, conclusion: .conclusion, status: .status}')

  echo "=== All Job Results ==="
  echo "$jobs" | jq -r '"\(.conclusion // .status)\t\(.name)"' | sort | column -t -s $'\t'
  echo ""

  local failed=0

  while IFS= read -r job; do
    local name conclusion
    name=$(echo "$job" | jq -r '.name')
    conclusion=$(echo "$job" | jq -r '.conclusion // .status')

    # Skip self
    if matches_list "$name" "$SELF_JOB"; then
      continue
    fi

    # Allowed-fail jobs: any conclusion is OK
    if matches_list "$name" "${ALLOWED_FAIL[@]}"; then
      if [[ "$conclusion" != "success" ]]; then
        echo "ALLOWED_FAIL: ${name} (${conclusion})"
      fi
      continue
    fi

    # Successful jobs are always fine
    if [[ "$conclusion" == "success" ]]; then
      continue
    fi

    # Skipped jobs
    if [[ "$conclusion" == "skipped" ]]; then
      if [[ "$GITHUB_EVENT_NAME" != "merge_group" ]]; then
        # In PR CI, any job may be skipped
        continue
      fi
      if matches_list "$name" "${ALLOWED_SKIP[@]}"; then
        echo "ALLOWED_SKIP: ${name}"
        continue
      fi
      echo "FAIL: ${name} was skipped in merge queue (not in ALLOWED_SKIP)"
      failed=1
      continue
    fi

    # Any other conclusion (failure, cancelled, null/in_progress) is a failure
    echo "FAIL: ${name} (${conclusion})"
    failed=1

  done < <(echo "$jobs" | jq -c '.')

  echo ""
  if [[ $failed -ne 0 ]]; then
    echo "ERROR: Some jobs did not pass. See above for details."
    exit 1
  fi

  echo "All jobs passed!"
}

main
