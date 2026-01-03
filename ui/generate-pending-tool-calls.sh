#!/bin/bash
set -euo pipefail

# Generate pending tool calls for testing the approval UI
# Usage: ./generate-pending-tool-calls.sh [max_tools] [interval_seconds]
#
# Creates one session with initial context, then adds pending tool calls
# that are never authorized - useful for testing the pending approval workflow.

MAX_TOOLS="${1:-10}"
INTERVAL="${2:-10}"
API_URL="${TENSORZERO_AUTOPILOT_API_URL:-http://localhost:4444}"
API_KEY="${TENSORZERO_AUTOPILOT_API_KEY:-}"
POSTGRES_URL="${TENSORZERO_AUTOPILOT_POSTGRES_URL:-postgres://postgres:postgres@localhost:5433/autopilot_api}"

# Try to load API key from fixtures if not set
if [ -z "$API_KEY" ]; then
  FIXTURES_KEY_FILE="$HOME/Developer/tensorzero/autopilot/e2e_tests/fixtures/api_key.env"
  if [ -f "$FIXTURES_KEY_FILE" ]; then
    source "$FIXTURES_KEY_FILE"
    API_KEY="$TENSORZERO_AUTOPILOT_API_KEY"
  fi
fi

if [ -z "$API_KEY" ]; then
  echo "Error: TENSORZERO_AUTOPILOT_API_KEY not set and could not load from fixtures"
  exit 1
fi

# Generate a UUIDv7 (time-ordered UUID)
generate_uuid() {
  npx uuid v7
}

insert_event() {
  local event_id="$1"
  local session_id="$2"
  local payload="$3"
  psql "$POSTGRES_URL" -q -c "INSERT INTO autopilot.events (id, payload, session_id) VALUES ('$event_id', \$json\$$payload\$json\$::jsonb, '$session_id')"
}

# Status updates (randomly added between tool calls)
STATUS_UPDATES=(
  "Analyzing the dataset..."
  "Preparing inference request..."
  "Collecting feedback data..."
  "Processing datapoints..."
  "Thinking about the next step..."
  "Evaluating results..."
)

# Tool call definitions (actual autopilot tools)
# Each entry: "tool_name|arguments_json"
TOOL_CALLS=(
  'inference|{"function_name": "extract_entities", "input": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Extract entities from this document about machine learning."}]}]}}'
  'inference|{"model_name": "openai::gpt-4o", "input": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Summarize this article about AI safety."}]}]}}'
  'feedback|{"inference_id": "019b7bb4-0000-0000-0000-000000000001", "metric_name": "comment", "value": "This output looks correct and well-structured."}'
  'feedback|{"episode_id": "019b7bb4-0000-0000-0000-000000000002", "metric_name": "accuracy", "value": 0.95}'
  'list_datapoints|{"dataset_name": "training_data", "limit": 10}'
  'list_datapoints|{"dataset_name": "evaluation_set", "limit": 25, "offset": 0}'
  'get_datapoints|{"dataset_name": "training_data", "ids": ["019b7bb4-0000-0000-0000-000000000003"]}'
  'create_datapoints|{"dataset_name": "training_data", "datapoints": [{"input": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Sample input"}]}]}, "output": "Sample output"}]}'
  'delete_datapoints|{"ids": ["019b7bb4-0000-0000-0000-000000000004", "019b7bb4-0000-0000-0000-000000000005"]}'
  'update_datapoints|{"dataset_name": "training_data", "datapoints": [{"id": "019b7bb4-0000-0000-0000-000000000006", "output": "Updated output value"}]}'
  'get_latest_feedback_by_metric|{"target_id": "019b7bb4-0000-0000-0000-000000000007", "metric_names": ["accuracy", "quality", "relevance"]}'
)

# Hardcoded deployment_id (temporary - will be removed soon)
DEPLOYMENT_ID="019b7bb4-bd08-76ec-875e-4d27d5eb3864"

# Create initial session with context
echo "Creating session at $API_URL..."
RESPONSE=$(curl -s -X POST "$API_URL/v1/sessions/00000000-0000-0000-0000-000000000000/events" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"deployment_id\": \"$DEPLOYMENT_ID\",
    \"tensorzero_version\": \"2025.12.6\",
    \"payload\": {
      \"type\": \"message\",
      \"role\": \"user\",
      \"content\": [{\"type\": \"text\", \"text\": \"Hello! I need help analyzing some data and running experiments. Can you assist me?\"}]
    }
  }")

SESSION_ID=$(echo "$RESPONSE" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
INITIAL_EVENT_ID=$(echo "$RESPONSE" | grep -o '"event_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$SESSION_ID" ]; then
  echo "Failed to create session: $RESPONSE"
  exit 1
fi

# Add assistant response
ASSISTANT_EVENT_ID=$(generate_uuid)
ASSISTANT_PAYLOAD='{"type": "message", "role": "assistant", "content": [{"type": "text", "text": "I would be happy to help you with data analysis and experiments! Let me start by examining your datasets and preparing some inference requests. I will need to use several tools to accomplish this."}]}'
insert_event "$ASSISTANT_EVENT_ID" "$SESSION_ID" "$ASSISTANT_PAYLOAD"

echo ""
echo "=========================================="
echo "Session created: $SESSION_ID"
echo "Initial event:   $INITIAL_EVENT_ID"
echo "Assistant event: $ASSISTANT_EVENT_ID"
echo ""
echo "UI URL: http://localhost:5173/autopilot/sessions/$SESSION_ID"
echo ""
echo "Adding up to $MAX_TOOLS pending tool calls every ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

TOOL_COUNT=0
TOOL_IDX=0
HALF_INTERVAL=$((INTERVAL / 2))

while [ $TOOL_COUNT -lt $MAX_TOOLS ]; do
  # For the first tool call, wait the full interval
  # For subsequent tool calls, we've already waited half after the status
  if [ $TOOL_COUNT -eq 0 ]; then
    sleep "$INTERVAL"
  else
    sleep "$HALF_INTERVAL"
  fi

  # Add a pending tool call
  TOOL_ENTRY="${TOOL_CALLS[$TOOL_IDX]}"
  TOOL_NAME="${TOOL_ENTRY%%|*}"
  TOOL_ARGS="${TOOL_ENTRY#*|}"

  TOOL_EVENT_ID=$(generate_uuid)
  # Escape the JSON arguments for embedding in the payload
  ESCAPED_ARGS=$(echo "$TOOL_ARGS" | sed 's/"/\\"/g')
  TOOL_PAYLOAD="{\"type\": \"tool_call\", \"id\": \"call_pending_$TOOL_COUNT\", \"name\": \"$TOOL_NAME\", \"arguments\": \"$ESCAPED_ARGS\"}"

  insert_event "$TOOL_EVENT_ID" "$SESSION_ID" "$TOOL_PAYLOAD"
  echo "[tool #$((TOOL_COUNT + 1))] $TOOL_EVENT_ID - $TOOL_NAME (PENDING)"

  TOOL_COUNT=$((TOOL_COUNT + 1))
  TOOL_IDX=$(( (TOOL_IDX + 1) % ${#TOOL_CALLS[@]} ))

  # Add a status update halfway to the next tool call (after the first tool call)
  if [ $TOOL_COUNT -lt $MAX_TOOLS ]; then
    sleep "$HALF_INTERVAL"
    STATUS_IDX=$((RANDOM % ${#STATUS_UPDATES[@]}))
    STATUS="${STATUS_UPDATES[$STATUS_IDX]}"
    STATUS_EVENT_ID=$(generate_uuid)
    STATUS_PAYLOAD="{\"type\": \"status_update\", \"status_update\": {\"type\": \"text\", \"text\": \"$STATUS\"}}"
    insert_event "$STATUS_EVENT_ID" "$SESSION_ID" "$STATUS_PAYLOAD"
    echo "[status] $STATUS_EVENT_ID - $STATUS"
  fi
done

echo ""
echo "=========================================="
echo "Done! Added $TOOL_COUNT pending tool calls to session $SESSION_ID"
echo "UI URL: http://localhost:5173/autopilot/sessions/$SESSION_ID"
echo "=========================================="
