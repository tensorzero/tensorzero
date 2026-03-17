#!/usr/bin/env bash
# Test cache token tracking across providers and modes.
#
# Usage: ./scripts/test-cache-tokens.sh [OPTIONS]
#
# Provider flags (can combine multiple):
#   --anthropic     Test Anthropic (claude)
#   --openai        Test OpenAI (gpt-4o-mini)
#   --gemini        Test Google AI Studio Gemini (gemini-2.5-flash-lite)
#   --xai           Test xAI (grok)
#   --all           Test all providers (default if no provider flag given)
#
# Backend flags:
#   --clickhouse    Only query ClickHouse
#   --postgres      Only query Postgres
#   (no flag)       Query whichever is available
#
# Examples:
#   ./scripts/test-cache-tokens.sh                        # all providers, auto-detect backend
#   ./scripts/test-cache-tokens.sh --anthropic --openai   # just Anthropic + OpenAI
#   ./scripts/test-cache-tokens.sh --gemini --clickhouse  # Gemini only, ClickHouse backend
#
# Prerequisites:
#   - Gateway running on localhost:3000
#   - Relevant API keys set (ANTHROPIC_API_KEY, OPENAI_API_KEY,
#     GOOGLE_AI_STUDIO_API_KEY, XAI_API_KEY)
set -euo pipefail

# ── Parse flags ──────────────────────────────────────────
BACKEND="auto"
DO_ANTHROPIC=false
DO_OPENAI=false
DO_GEMINI=false
DO_XAI=false
ANY_PROVIDER=false

for arg in "$@"; do
  case "$arg" in
    --clickhouse) BACKEND="clickhouse" ;;
    --postgres)   BACKEND="postgres" ;;
    --anthropic)  DO_ANTHROPIC=true; ANY_PROVIDER=true ;;
    --openai)     DO_OPENAI=true;    ANY_PROVIDER=true ;;
    --gemini)     DO_GEMINI=true;    ANY_PROVIDER=true ;;
    --xai)        DO_XAI=true;       ANY_PROVIDER=true ;;
    --all)        DO_ANTHROPIC=true; DO_OPENAI=true; DO_GEMINI=true; DO_XAI=true; ANY_PROVIDER=true ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# Default: all providers
if [ "$ANY_PROVIDER" = false ]; then
  DO_ANTHROPIC=true
  DO_OPENAI=true
  DO_GEMINI=true
  DO_XAI=true
fi

GATEWAY="http://localhost:3000"
CH="http://localhost:8123/?user=chuser&password=chpassword&database=tensorzero_e2e_tests"

ch_query() {
  curl -s "$CH" --data-binary "$1"
}

# Generate a long padding string (~2000 tokens) to exceed Anthropic's 1024 token cache threshold.
PADDING=$(python3 -c "print(' '.join(['word' + str(i) for i in range(2000)]))")

inference() {
  local label="$1"
  local variant="$2"
  local stream="$3"
  local msg="$4"

  echo ""
  echo "--- $label ---"
  if [ "$stream" = "true" ]; then
    RESP=$(curl -s "$GATEWAY/inference" \
      -H "Content-Type: application/json" \
      -d "{
        \"function_name\": \"basic_test\",
        \"variant_name\": \"$variant\",
        \"stream\": true,
        \"input\": {
          \"system\": {\"assistant_name\": \"CacheBot $PADDING\"},
          \"messages\": [{\"role\": \"user\", \"content\": \"$msg\"}]
        }
      }")
    # Extract last data chunk with usage
    LAST=$(echo "$RESP" | grep "^data:" | tail -1 | sed 's/^data: //')
    INF_ID=$(echo "$RESP" | grep "^data:" | head -1 | sed 's/^data: //' | jq -r '.inference_id // empty')
    echo "  inference_id: $INF_ID"
    echo "  usage: $(echo "$LAST" | jq -c '.usage // empty' 2>/dev/null)"
  else
    RESP=$(curl -s "$GATEWAY/inference" \
      -H "Content-Type: application/json" \
      -d "{
        \"function_name\": \"basic_test\",
        \"variant_name\": \"$variant\",
        \"input\": {
          \"system\": {\"assistant_name\": \"CacheBot $PADDING\"},
          \"messages\": [{\"role\": \"user\", \"content\": \"$msg\"}]
        }
      }")
    INF_ID=$(echo "$RESP" | jq -r '.inference_id // empty')
    echo "  inference_id: $INF_ID"
    echo "  usage: $(echo "$RESP" | jq -c '.usage // empty')"
  fi
  echo "$INF_ID"
}

echo "============================================"
echo "  Cache Token Tracking Test Suite"
echo "============================================"
echo "Providers: anthropic=$DO_ANTHROPIC openai=$DO_OPENAI gemini=$DO_GEMINI xai=$DO_XAI"

STEP=0

# ──────────────────────────────────────────────
# Anthropic
# ──────────────────────────────────────────────
if [ "$DO_ANTHROPIC" = true ]; then
  STEP=$((STEP+1))
  echo ""
  echo "=== $STEP. Anthropic Non-Streaming (cache write, then cache read) ==="
  ID=$(inference "Anthropic non-stream #1 (expect cache WRITE)" "anthropic" "false" "Say hi in 3 words.")
  sleep 2
  ID=$(inference "Anthropic non-stream #2 (expect cache READ)" "anthropic" "false" "Say hi in 3 words.")

  STEP=$((STEP+1))
  echo ""
  echo "=== $STEP. Anthropic Streaming (cache read from previous) ==="
  ID=$(inference "Anthropic stream #1 (expect cache READ)" "anthropic" "true" "Say hi in 3 words.")
  sleep 2
  ID=$(inference "Anthropic stream #2 (expect cache READ)" "anthropic" "true" "Say hi in 3 words.")

  STEP=$((STEP+1))
  echo ""
  echo "=== $STEP. Anthropic with different user message (system cached, user not) ==="
  ID=$(inference "Anthropic different msg (expect cache READ for system)" "anthropic" "false" "Tell me a joke.")
fi

# ──────────────────────────────────────────────
# OpenAI
# ──────────────────────────────────────────────
if [ "$DO_OPENAI" = true ]; then
  STEP=$((STEP+1))
  echo ""
  echo "=== $STEP. OpenAI Non-Streaming (cache write, then cache read) ==="
  ID=$(inference "OpenAI non-stream #1 (first call)" "openai" "false" "Say hi in 3 words.")
  sleep 2
  ID=$(inference "OpenAI non-stream #2 (expect cache READ)" "openai" "false" "Say hi in 3 words.")

  STEP=$((STEP+1))
  echo ""
  echo "=== $STEP. OpenAI Streaming ==="
  ID=$(inference "OpenAI stream #1" "openai" "true" "Say hi in 3 words.")
  sleep 2
  ID=$(inference "OpenAI stream #2 (expect cache READ)" "openai" "true" "Say hi in 3 words.")
fi

# ──────────────────────────────────────────────
# Google AI Studio Gemini
# Note: Gemini's automatic caching requires a minimum prompt size (~32K tokens
# for most models). With our ~2000 token padding, we won't see cached tokens,
# but we verify the field flows through correctly (should be null/absent).
# ──────────────────────────────────────────────
if [ "$DO_GEMINI" = true ]; then
  STEP=$((STEP+1))
  echo ""
  echo "=== $STEP. Google AI Studio Gemini Non-Streaming ==="
  ID=$(inference "Gemini non-stream #1 (first call)" "google-ai-studio-gemini-flash-lite" "false" "Say hi in 3 words.")
  sleep 2
  ID=$(inference "Gemini non-stream #2 (second call)" "google-ai-studio-gemini-flash-lite" "false" "Say hi in 3 words.")

  STEP=$((STEP+1))
  echo ""
  echo "=== $STEP. Google AI Studio Gemini Streaming ==="
  ID=$(inference "Gemini stream #1" "google-ai-studio-gemini-flash-lite" "true" "Say hi in 3 words.")
  sleep 2
  ID=$(inference "Gemini stream #2" "google-ai-studio-gemini-flash-lite" "true" "Say hi in 3 words.")
fi

# ──────────────────────────────────────────────
# xAI
# Note: xAI uses OpenAI-compatible prompt_tokens_details.cached_tokens.
# Automatic prefix caching behavior depends on xAI's server-side implementation.
# ──────────────────────────────────────────────
if [ "$DO_XAI" = true ]; then
  STEP=$((STEP+1))
  echo ""
  echo "=== $STEP. xAI Non-Streaming ==="
  ID=$(inference "xAI non-stream #1 (first call)" "xai" "false" "Say hi in 3 words.")
  sleep 2
  ID=$(inference "xAI non-stream #2 (second call)" "xai" "false" "Say hi in 3 words.")

  STEP=$((STEP+1))
  echo ""
  echo "=== $STEP. xAI Streaming ==="
  ID=$(inference "xAI stream #1" "xai" "true" "Say hi in 3 words.")
  sleep 2
  ID=$(inference "xAI stream #2" "xai" "true" "Say hi in 3 words.")
fi

# ──────────────────────────────────────────────
# Wait for trailing writes, then query the database
# ──────────────────────────────────────────────
echo ""
echo "Waiting for database writes..."
sleep 3

PG_URL="postgres://postgres:postgres@localhost:5432/tensorzero-e2e-tests"
PSQL="${PSQL:-$(command -v psql 2>/dev/null || echo /opt/homebrew/Cellar/postgresql@16/16.13/bin/psql)}"

# Detect which backend to query
case "$BACKEND" in
  clickhouse) HAS_CH=true; HAS_PG=false ;;
  postgres)   HAS_CH=false; HAS_PG=true ;;
  *)
    HAS_CH=false; HAS_PG=false
    curl -s "$CH" --data-binary "SELECT 1" &>/dev/null && HAS_CH=true
    "$PSQL" "$PG_URL" -c "SELECT 1" &>/dev/null 2>&1 && HAS_PG=true
    ;;
esac

if [ "$HAS_CH" = true ]; then
  echo ""
  echo "============================================"
  echo "  ClickHouse Results: ModelInference"
  echo "============================================"
  ch_query "SELECT
    id,
    input_tokens,
    output_tokens,
    cache_read_input_tokens,
    cache_write_input_tokens,
    model_name
  FROM ModelInference
  ORDER BY timestamp DESC
  LIMIT 30
  FORMAT PrettyCompact"

  echo ""
  echo "============================================"
  echo "  ClickHouse Results: ModelProviderStatistics"
  echo "============================================"
  ch_query "SELECT
    model_name,
    model_provider_name,
    minute,
    sumMerge(total_cache_read_input_tokens) as cache_read,
    sumMerge(total_cache_write_input_tokens) as cache_write
  FROM ModelProviderStatistics
  GROUP BY model_name, model_provider_name, minute
  HAVING cache_read > 0 OR cache_write > 0
  ORDER BY minute DESC
  LIMIT 20
  FORMAT PrettyCompact"
fi

if [ "$HAS_PG" = true ]; then
  echo ""
  echo "============================================"
  echo "  Postgres Results: model_inferences"
  echo "============================================"
  "$PSQL" "$PG_URL" -c "SELECT id, input_tokens, output_tokens, cache_read_input_tokens, cache_write_input_tokens, model_name FROM tensorzero.model_inferences ORDER BY created_at DESC LIMIT 30;"

  echo ""
  echo "============================================"
  echo "  Postgres Results: model_provider_statistics"
  echo "============================================"
  "$PSQL" "$PG_URL" -c "SELECT model_name, model_provider_name, minute, total_cache_read_input_tokens as cache_read, total_cache_write_input_tokens as cache_write FROM tensorzero.model_provider_statistics WHERE total_cache_read_input_tokens > 0 OR total_cache_write_input_tokens > 0 ORDER BY minute DESC LIMIT 20;"
fi

if [ "$HAS_CH" = false ] && [ "$HAS_PG" = false ]; then
  echo "ERROR: Neither ClickHouse nor Postgres is reachable."
fi

echo ""
echo "=== Done ==="
