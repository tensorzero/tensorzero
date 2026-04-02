#!/usr/bin/env bash
# Seed cache test data by sending real inferences to the gateway.
# Usage: ./scripts/seed-cache-test-data.sh [GATEWAY_URL] [COUNT]
#
# Requires the gateway to be running with the cache_test config loaded.

set -euo pipefail

GATEWAY="${1:-http://localhost:3000}"
COUNT="${2:-10}"

echo "Gateway: $GATEWAY"
echo "Inferences per variant: $COUNT"
echo ""

send_inference() {
    local func="$1" variant="$2" msg="$3"
    curl -sf -X POST "${GATEWAY}/inference" \
        -H "Content-Type: application/json" \
        -d "{
            \"function_name\": \"${func}\",
            \"variant_name\": \"${variant}\",
            \"input\": {
                \"messages\": [{
                    \"role\": \"user\",
                    \"content\": \"${msg}\"
                }]
            }
        }" | jq -r '.inference_id // empty' 2>/dev/null || echo "FAILED"
}

# ── Scenario 1: Anthropic + tools, no breakpoints ───────────────────────────
echo "=== cache_test / anthropic-with-tools (Anthropic + tools, no breakpoints) ==="
for i in $(seq 1 "$COUNT"); do
    id=$(send_inference "cache_test" "anthropic-with-tools" "What is the temperature in city ${i}? Also check humidity.")
    echo "  [$i/$COUNT] $id"
done
echo ""

# ── Scenario 2: Anthropic + tools, WITH breakpoints ─────────────────────────
echo "=== cache_test / anthropic-with-tools-cached (Anthropic + tools, WITH breakpoints) ==="
for i in $(seq 1 "$COUNT"); do
    id=$(send_inference "cache_test" "anthropic-with-tools-cached" "What is the temperature in city ${i}? Also check humidity.")
    echo "  [$i/$COUNT] $id"
done
echo ""

# ── Scenario 3: OpenAI + tools ──────────────────────────────────────────────
echo "=== cache_test / openai-with-tools (OpenAI + tools, automatic caching) ==="
for i in $(seq 1 "$COUNT"); do
    id=$(send_inference "cache_test" "openai-with-tools" "What is the temperature in city ${i}? Also check humidity.")
    echo "  [$i/$COUNT] $id"
done
echo ""

# ── Scenario 4: Anthropic, no tools, no breakpoints ─────────────────────────
echo "=== cache_test_no_tools / anthropic-no-tools (Anthropic, no tools, no breakpoints) ==="
for i in $(seq 1 "$COUNT"); do
    id=$(send_inference "cache_test_no_tools" "anthropic-no-tools" "Explain the architecture of microservice ${i} in detail.")
    echo "  [$i/$COUNT] $id"
done
echo ""

# ── Scenario 5: Anthropic, no tools, WITH breakpoints ───────────────────────
echo "=== cache_test_no_tools / anthropic-no-tools-cached (Anthropic, no tools, WITH breakpoints) ==="
for i in $(seq 1 "$COUNT"); do
    id=$(send_inference "cache_test_no_tools" "anthropic-no-tools-cached" "Explain the architecture of microservice ${i} in detail.")
    echo "  [$i/$COUNT] $id"
done
echo ""

echo "Done! Now test the skills:"
echo "  /prompt-audit cache_test"
echo "  /prompt-audit cache_test_no_tools"
echo "  /cache-status"
