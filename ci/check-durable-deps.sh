#!/bin/bash
set -euo pipefail

echo "Checking tensorzero/durable git dependencies..."

# Find all Cargo.toml files (excluding node_modules and target)
CARGO_TOMLS=$(find . -name "Cargo.toml" -not -path "*/node_modules/*" -not -path "*/target/*")

# Temporary directory for cloning repos
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Repository to check
REPO_NAME="tensorzero/durable"
REPO_URL="https://github.com/tensorzero/durable"

# Clone repo once (reused for all checks)
repo_dir="$TEMP_DIR/durable"
echo "Cloning $REPO_NAME..."
git clone --quiet "$REPO_URL" "$repo_dir"

# Track if we found any issues
FOUND_ISSUES=0

# Check each Cargo.toml
for toml in $CARGO_TOMLS; do
    # Read the file line by line
    while IFS= read -r line; do
        # Check if this line contains a git dependency to durable
        if echo "$line" | grep -q "git.*$REPO_NAME"; then
            echo "Found $REPO_NAME dependency in $toml"
            echo "  Line: $line"

            # Check if it has a rev or tag (not branch)
            if echo "$line" | grep -qE 'branch[[:space:]]*='; then
                echo "  ❌ ERROR: Dependency on $REPO_NAME uses 'branch' instead of 'rev' or 'tag'"
                echo "     Dependencies must be pinned to a specific commit (rev) or tag that is in main's history"
                FOUND_ISSUES=1
                continue
            fi

            if ! echo "$line" | grep -qE '(rev|tag)[[:space:]]*='; then
                echo "  ❌ ERROR: Dependency on $REPO_NAME does not have 'rev' or 'tag' specified"
                FOUND_ISSUES=1
                continue
            fi

            # Extract the rev or tag value
            REF=""
            REF_TYPE=""
            if echo "$line" | grep -qE 'rev[[:space:]]*='; then
                REF=$(echo "$line" | sed -E 's/.*rev[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/')
                REF_TYPE="rev"
            elif echo "$line" | grep -qE 'tag[[:space:]]*='; then
                REF=$(echo "$line" | sed -E 's/.*tag[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/')
                REF_TYPE="tag"
            fi

            if [ -z "$REF" ]; then
                echo "  ❌ ERROR: Could not extract $REF_TYPE value from line"
                FOUND_ISSUES=1
                continue
            fi

            echo "  Found $REF_TYPE: $REF"

            # For tags, we need to check refs/tags/TAG
            if [ "$REF_TYPE" = "tag" ]; then
                REF_TO_CHECK="refs/tags/$REF"
            else
                REF_TO_CHECK="$REF"
            fi

            # Check if the ref exists first
            if ! git -C "$repo_dir" rev-parse --verify "$REF_TO_CHECK^{commit}" &>/dev/null; then
                echo "  ❌ ERROR: $REF_TYPE '$REF' does not exist in $REPO_NAME"
                FOUND_ISSUES=1
                continue
            fi

            # Check if this ref is an ancestor of origin/main (i.e., in main's history)
            if ! git -C "$repo_dir" merge-base --is-ancestor "$REF_TO_CHECK" origin/main 2>/dev/null; then
                echo "  ❌ ERROR: $REF_TYPE '$REF' is not in $REPO_NAME's main branch history"
                echo "     This means the commit is not merged into main yet"
                FOUND_ISSUES=1
            else
                echo "  ✓ $REF_TYPE '$REF' is in main's history"
            fi
        fi
    done < "$toml"
done

echo ""
if [ $FOUND_ISSUES -eq 0 ]; then
    echo "✅ All $REPO_NAME dependencies are properly pinned to refs in main's history!"
    exit 0
else
    echo "❌ Found issues with $REPO_NAME dependencies"
    exit 1
fi
