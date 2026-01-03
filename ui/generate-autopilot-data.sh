#!/bin/bash
set -euo pipefail

# Generate test data for autopilot sessions with full conversations
# Usage: ./generate-autopilot-data.sh [num_sessions]
#
# Creates sessions with multiple event types:
# - User messages (via API)
# - Assistant messages (via direct DB insert)
# - Tool calls (via direct DB insert)
# - Tool results (via API)

NUM_SESSIONS="${1:-10}"
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
  echo "Set it via environment variable or ensure fixtures/api_key.env exists"
  exit 1
fi

# Generate a UUIDv7 (time-ordered UUID)
generate_uuid() {
  npx uuid v7
}

# Insert an event directly into the database using dollar-quoting to avoid escaping issues
insert_event() {
  local event_id="$1"
  local session_id="$2"
  local payload="$3"

  # Use dollar-quoting ($$...$$) to avoid single-quote escaping issues in PostgreSQL
  psql "$POSTGRES_URL" -q -c "INSERT INTO autopilot.events (id, payload, session_id) VALUES ('$event_id', \$json\$$payload\$json\$::jsonb, '$session_id')"
}

echo "Creating $NUM_SESSIONS autopilot sessions at $API_URL..."
echo "Using Postgres: $POSTGRES_URL"
echo ""

# Messages without apostrophes to avoid JSON/SQL escaping complexity
USER_MESSAGES=(
  "Can you help me analyze this data and create a summary report?"
  "I need to refactor the authentication module. Where should I start?"
  "What is the best approach to implement caching for our API endpoints?"
  "Can you review this pull request and suggest improvements?"
  "Help me debug this failing test - it works locally but fails in CI."
  "I want to add real-time notifications. What technologies would you recommend?"
  "Can you explain how the payment processing flow works in our system?"
  "I need to optimize our database queries - they are getting slow."
  "Help me write documentation for the new API endpoints."
  "What security vulnerabilities should I check for in this code?"
)

ASSISTANT_RESPONSES=(
  "I will analyze the data and create a comprehensive summary for you. Let me start by examining the structure and key metrics."
  "For the authentication refactor, I recommend starting with the token validation logic. Let me search for the relevant files."
  "For API caching, I would suggest using Redis with a TTL-based invalidation strategy. Let me outline the implementation steps."
  "I will review the PR thoroughly. Let me fetch the diff and analyze the changes."
  "Let me investigate the CI failure. I will check the environment differences and test configuration."
  "For real-time notifications, WebSockets or Server-Sent Events would work well. Let me compare the options."
  "I will trace through the payment flow and document each step. Let me find the entry point."
  "I will profile the slow queries and suggest optimizations. Let me start with the most frequent ones."
  "I will help you write clear, comprehensive API documentation. Let me gather the endpoint specifications."
  "I will perform a security audit focusing on OWASP top 10 vulnerabilities. Let me scan the codebase."
)

TOOL_NAMES=("read_file" "search_code" "run_tests" "fetch_url" "execute_command")

for i in $(seq 1 "$NUM_SESSIONS"); do
  # Hardcoded deployment_id (temporary - will be removed soon)
  DEPLOYMENT_ID="019b7bb4-bd08-76ec-875e-4d27d5eb3864"
  MSG_INDEX=$(( (i - 1) % ${#USER_MESSAGES[@]} ))
  USER_MSG="${USER_MESSAGES[$MSG_INDEX]}"
  ASSISTANT_MSG="${ASSISTANT_RESPONSES[$MSG_INDEX]}"

  # 1. Create session with initial user message (via API)
  RESPONSE=$(curl -s -X POST "$API_URL/v1/sessions/00000000-0000-0000-0000-000000000000/events" \
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

  SESSION_ID=$(echo "$RESPONSE" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)

  if [ -z "$SESSION_ID" ]; then
    echo "Failed to create session $i: $RESPONSE"
    continue
  fi

  echo "Created session $i: $SESSION_ID"

  # 2. Insert assistant response (via DB)
  ASSISTANT_EVENT_ID=$(generate_uuid)
  ASSISTANT_PAYLOAD="{\"type\": \"message\", \"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": \"$ASSISTANT_MSG\"}]}"
  insert_event "$ASSISTANT_EVENT_ID" "$SESSION_ID" "$ASSISTANT_PAYLOAD"

  # 3. Insert a tool call (via DB) - for some sessions
  if [ $((i % 2)) -eq 0 ]; then
    TOOL_CALL_EVENT_ID=$(generate_uuid)
    TOOL_NAME="${TOOL_NAMES[$((i % ${#TOOL_NAMES[@]}))]}"
    TOOL_CALL_PAYLOAD="{\"type\": \"tool_call\", \"id\": \"call_$i\", \"name\": \"$TOOL_NAME\", \"arguments\": \"{\\\"path\\\": \\\"src/main.rs\\\"}\"}"
    insert_event "$TOOL_CALL_EVENT_ID" "$SESSION_ID" "$TOOL_CALL_PAYLOAD"

    # 4. Add tool result (via API)
    curl -s -X POST "$API_URL/v1/sessions/$SESSION_ID/events" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $API_KEY" \
      -d "{
        \"deployment_id\": \"$DEPLOYMENT_ID\",
        \"tensorzero_version\": \"2025.1.0\",
        \"payload\": {
          \"type\": \"tool_result\",
          \"tool_call_event_id\": \"$TOOL_CALL_EVENT_ID\",
          \"outcome\": {
            \"type\": \"success\",
            \"id\": \"call_$i\",
            \"name\": \"$TOOL_NAME\",
            \"result\": \"Tool executed successfully. Found 42 relevant items.\"
          }
        }
      }" > /dev/null 2>&1 || true

    # 5. Insert follow-up assistant message (via DB)
    FOLLOWUP_EVENT_ID=$(generate_uuid)
    FOLLOWUP_PAYLOAD="{\"type\": \"message\", \"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": \"Based on the tool results, I can see the relevant information. Here is my analysis...\"}]}"
    insert_event "$FOLLOWUP_EVENT_ID" "$SESSION_ID" "$FOLLOWUP_PAYLOAD"
  fi

  # 6. Add another user turn for some sessions
  if [ $((i % 3)) -eq 0 ]; then
    curl -s -X POST "$API_URL/v1/sessions/$SESSION_ID/events" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $API_KEY" \
      -d "{
        \"deployment_id\": \"$DEPLOYMENT_ID\",
        \"tensorzero_version\": \"2025.1.0\",
        \"payload\": {
          \"type\": \"message\",
          \"role\": \"user\",
          \"content\": [{\"type\": \"text\", \"text\": \"Thanks! Can you also check the related test files?\"}]
        }
      }" > /dev/null 2>&1 || true

    # Insert final assistant response
    FINAL_EVENT_ID=$(generate_uuid)
    FINAL_PAYLOAD="{\"type\": \"message\", \"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": \"Of course! I will examine the test files now and provide a comprehensive review.\"}]}"
    insert_event "$FINAL_EVENT_ID" "$SESSION_ID" "$FINAL_PAYLOAD"
  fi
done

echo ""
echo "Done! Created $NUM_SESSIONS sessions with full conversations."
