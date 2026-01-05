#!/bin/bash
set -euo pipefail

# Generate streaming test data for a single autopilot session
# Usage: ./generate-autopilot-data-stream.sh [max_events] [interval_seconds]
#
# Creates one session and adds events every N seconds, simulating a live conversation.

MAX_EVENTS="${1:-100}"
INTERVAL="${2:-5}"
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

# Event templates
USER_MESSAGES=(
  "Can you help me with this task?"
  "What do you think about this approach?"
  "Can you explain that in more detail?"
  "Let me clarify what I meant..."
  "That sounds good. What is the next step?"
  "I have a follow-up question about that."
  "Can you show me an example?"
  "How would this work in production?"
  "What are the potential issues with this?"
  "Thanks! Can you also check the tests?"
)

ASSISTANT_MESSAGES=(
  "I will help you with that. Let me analyze the situation first."
  "Based on my analysis, here is what I recommend..."
  "Let me break this down step by step for you."
  "I understand. Here is a more detailed explanation..."
  "Great question! The next step would be to..."
  "To answer your follow-up: the key consideration is..."
  "Here is a concrete example that illustrates the concept..."
  "In production, you would want to consider these factors..."
  "The potential issues to watch out for include..."
  "I will examine the test files and provide feedback."
)

STATUS_UPDATES=(
  "Analyzing the codebase..."
  "Searching for relevant files..."
  "Reading file contents..."
  "Processing the results..."
  "Generating response..."
  "Checking for edge cases..."
  "Validating the approach..."
  "Compiling findings..."
)

TOOL_NAMES=("read_file" "search_code" "run_tests" "list_files" "execute_command")

# Create initial session
# Hardcoded deployment_id (temporary - will be removed soon)
DEPLOYMENT_ID="019b7bb4-bd08-76ec-875e-4d27d5eb3864"

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
      \"content\": [{\"type\": \"text\", \"text\": \"[Event #0] Hello! I need help with a complex coding task. Can you assist me?\"}]
    }
  }")

SESSION_ID=$(echo "$RESPONSE" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
INITIAL_EVENT_ID=$(echo "$RESPONSE" | grep -o '"event_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$SESSION_ID" ]; then
  echo "Failed to create session: $RESPONSE"
  exit 1
fi

echo ""
echo "=========================================="
echo "Session created: $SESSION_ID"
echo "Initial event:   $INITIAL_EVENT_ID"
echo "Adding up to $MAX_EVENTS events every ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

EVENT_COUNT=1
TURN=0

while [ $EVENT_COUNT -lt $MAX_EVENTS ]; do
  sleep "$INTERVAL"

  # Rotate through different event types
  EVENT_TYPE=$((EVENT_COUNT % 5))

  case $EVENT_TYPE in
    0)
      # User message (via API)
      MSG_IDX=$((TURN % ${#USER_MESSAGES[@]}))
      USER_MSG="[Event #$EVENT_COUNT] ${USER_MESSAGES[$MSG_IDX]}"

      RESP=$(curl -s -X POST "$API_URL/v1/sessions/$SESSION_ID/events" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{
          \"deployment_id\": \"$DEPLOYMENT_ID\",
          \"tensorzero_version\": \"2025.1.0\",
          \"payload\": {
            \"type\": \"message\",
            \"role\": \"user\",
            \"content\": [{\"type\": \"text\", \"text\": \"$USER_MSG\"}]
          }
        }")
      EVENT_UUID=$(echo "$RESP" | grep -o '"event_id":"[^"]*"' | cut -d'"' -f4)

      echo "[$EVENT_COUNT] $EVENT_UUID - User message: $USER_MSG"
      TURN=$((TURN + 1))
      ;;

    1)
      # Status update (via DB)
      EVENT_ID=$(generate_uuid)
      STATUS_IDX=$((EVENT_COUNT % ${#STATUS_UPDATES[@]}))
      STATUS="[Event #$EVENT_COUNT] ${STATUS_UPDATES[$STATUS_IDX]}"
      PAYLOAD="{\"type\": \"status_update\", \"status_update\": {\"type\": \"text\", \"text\": \"$STATUS\"}}"
      insert_event "$EVENT_ID" "$SESSION_ID" "$PAYLOAD"
      echo "[$EVENT_COUNT] $EVENT_ID - Status update: $STATUS"
      ;;

    2)
      # Tool call (via DB)
      EVENT_ID=$(generate_uuid)
      TOOL_IDX=$((EVENT_COUNT % ${#TOOL_NAMES[@]}))
      TOOL_NAME="${TOOL_NAMES[$TOOL_IDX]}"
      PAYLOAD="{\"type\": \"tool_call\", \"id\": \"call_$EVENT_COUNT\", \"name\": \"$TOOL_NAME\", \"arguments\": \"{\\\"path\\\": \\\"src/module_$EVENT_COUNT.rs\\\", \\\"event_number\\\": $EVENT_COUNT}\"}"
      insert_event "$EVENT_ID" "$SESSION_ID" "$PAYLOAD"
      echo "[$EVENT_COUNT] $EVENT_ID - Tool call: $TOOL_NAME"
      LAST_TOOL_CALL_ID="$EVENT_ID"
      LAST_TOOL_NAME="$TOOL_NAME"
      LAST_TOOL_EVENT_NUM="$EVENT_COUNT"
      ;;

    3)
      # Tool result (via API if we have a pending tool call)
      if [ -n "${LAST_TOOL_CALL_ID:-}" ]; then
        RESULT_TEXT="[Event #$EVENT_COUNT] Result for tool call #${LAST_TOOL_EVENT_NUM:-?}: Found $((RANDOM % 50 + 1)) matches in the codebase."
        RESP=$(curl -s -X POST "$API_URL/v1/sessions/$SESSION_ID/events" \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer $API_KEY" \
          -d "{
            \"deployment_id\": \"$DEPLOYMENT_ID\",
            \"tensorzero_version\": \"2025.1.0\",
            \"payload\": {
              \"type\": \"tool_result\",
              \"tool_call_event_id\": \"$LAST_TOOL_CALL_ID\",
              \"outcome\": {
                \"type\": \"success\",
                \"id\": \"call_$EVENT_COUNT\",
                \"name\": \"$LAST_TOOL_NAME\",
                \"result\": \"$RESULT_TEXT\"
              }
            }
          }")
        EVENT_UUID=$(echo "$RESP" | grep -o '"event_id":"[^"]*"' | cut -d'"' -f4)
        echo "[$EVENT_COUNT] $EVENT_UUID - Tool result for: $LAST_TOOL_NAME"
        unset LAST_TOOL_CALL_ID
      else
        # Fall back to status update
        EVENT_ID=$(generate_uuid)
        PAYLOAD="{\"type\": \"status_update\", \"status_update\": {\"type\": \"text\", \"text\": \"[Event #$EVENT_COUNT] Continuing analysis...\"}}"
        insert_event "$EVENT_ID" "$SESSION_ID" "$PAYLOAD"
        echo "[$EVENT_COUNT] $EVENT_ID - Status update: Continuing analysis..."
      fi
      ;;

    4)
      # Assistant message (via DB)
      EVENT_ID=$(generate_uuid)
      MSG_IDX=$((TURN % ${#ASSISTANT_MESSAGES[@]}))
      ASSISTANT_MSG="[Event #$EVENT_COUNT] ${ASSISTANT_MESSAGES[$MSG_IDX]}"
      PAYLOAD="{\"type\": \"message\", \"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": \"$ASSISTANT_MSG\"}]}"
      insert_event "$EVENT_ID" "$SESSION_ID" "$PAYLOAD"
      echo "[$EVENT_COUNT] $EVENT_ID - Assistant message: ${ASSISTANT_MSG:0:60}..."
      ;;
  esac

  EVENT_COUNT=$((EVENT_COUNT + 1))
done

echo ""
echo "=========================================="
echo "Done! Added $EVENT_COUNT events to session $SESSION_ID"
echo "=========================================="
