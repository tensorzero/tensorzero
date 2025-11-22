#!/usr/bin/env bash
# Check that the Helm values schema is in sync with values.yaml
# This script verifies that the generated schema matches what's committed
# It runs after the helm-values-schema hook generates/updates the schema

set -euo pipefail

SCHEMA_JSON="examples/production-deployment-k8s-helm/values.schema.json"

# Check if the schema file exists
if [ ! -f "$SCHEMA_JSON" ]; then
  echo "Error: $SCHEMA_JSON not found"
  exit 1
fi

# Check if there are any uncommitted changes to the schema file
# This means the generated schema doesn't match what's committed
if ! git diff --quiet HEAD -- "$SCHEMA_JSON" 2>/dev/null; then
  echo "Error: Generated helm chart values schema does not match committed version"
  echo "The schema file has uncommitted changes. Please commit the updated schema."
  echo ""
  echo "Diff:"
  git diff HEAD -- "$SCHEMA_JSON" || true
  exit 1
fi

echo "✅ Schema is in sync"
exit 0

