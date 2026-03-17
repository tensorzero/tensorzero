#!/bin/bash

# Set the base URL for the TensorZero Gateway
GATEWAY_URL="http://localhost:3000"
DATASET_NAME="email_application"

# Create a temporary file to store the JSON data
TEMP_FILE=$(mktemp)

# Function to make the JSON data more readable in the script
cat << 'EOF' > "$TEMP_FILE"
{
  "datapoints": [
    {
      "type": "json",
      "function_name": "extract_recipient",
      "input": {
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "Please send this to Alice at alice@example.com"
              }
            ]
          }
        ]
      }
    },
    {
      "type": "json",
      "function_name": "extract_recipient",
      "input": {
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "Please send this to Bob at bob@example.com"
              }
            ]
          }
        ]
      },
      "output": {
        "name": "Bob",
        "email": "bob@example.com"
      }
    },
    {
      "type": "chat",
      "function_name": "draft_email",
      "input": {
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "Please draft an email to Bob at bob@example.com"
              }
            ]
          }
        ]
      },
      "tags": {
        "customer_id": "123"
      }
    }
  ]
}
EOF

# 1. Insert datapoints
echo -e "POST ${GATEWAY_URL}/v1/datasets/${DATASET_NAME}/datapoints\n"
BULK_INSERT_RESPONSE=$(curl -s -X POST \
  "${GATEWAY_URL}/v1/datasets/${DATASET_NAME}/datapoints" \
  -H "Content-Type: application/json" \
  -d @"$TEMP_FILE")

echo "$BULK_INSERT_RESPONSE"

# Check if bulk insert was successful by verifying the response has an "ids" field
if ! echo "$BULK_INSERT_RESPONSE" | jq -e '.ids | type == "array"' >/dev/null; then
    echo "Error: Bulk insert failed - invalid response format"
    rm "$TEMP_FILE"
    exit 1
fi

# Extract the first datapoint ID from the response
FIRST_DATAPOINT_ID=$(echo "$BULK_INSERT_RESPONSE" | jq -r '.ids[0]')

# Check if we got a valid ID
if [ -z "$FIRST_DATAPOINT_ID" ] || [ "$FIRST_DATAPOINT_ID" = "null" ]; then
    echo "Error: Could not get valid datapoint ID"
    rm "$TEMP_FILE"
    exit 1
fi

# 2. Get a single datapoint
echo -e "\nPOST ${GATEWAY_URL}/v1/datasets/${DATASET_NAME}/get_datapoints\n"
GET_RESPONSE=$(curl -s -X POST \
  "${GATEWAY_URL}/v1/datasets/${DATASET_NAME}/get_datapoints" \
  -H "Content-Type: application/json" \
  -d "{\"ids\": [\"${FIRST_DATAPOINT_ID}\"]}")

echo "$GET_RESPONSE"

# 3. Delete the first datapoint
echo -e "\nDELETE ${GATEWAY_URL}/v1/datasets/${DATASET_NAME}/datapoints\n"
DELETE_RESPONSE=$(curl -s -X DELETE \
  "${GATEWAY_URL}/v1/datasets/${DATASET_NAME}/datapoints" \
  -H "Content-Type: application/json" \
  -d "{\"ids\": [\"${FIRST_DATAPOINT_ID}\"]}")

echo "$DELETE_RESPONSE"

# 4. List all datapoints
echo -e "\nPOST ${GATEWAY_URL}/v1/datasets/${DATASET_NAME}/list_datapoints\n"
LIST_RESPONSE=$(curl -s -X POST \
  "${GATEWAY_URL}/v1/datasets/${DATASET_NAME}/list_datapoints" \
  -H "Content-Type: application/json" \
  -d '{}')

echo "$LIST_RESPONSE"

# 5. Clean up - delete remaining datapoints
REMAINING_IDS=$(echo "$LIST_RESPONSE" | jq -c '[.datapoints[].id]')
if [ "$REMAINING_IDS" != "[]" ] && [ -n "$REMAINING_IDS" ]; then
  curl -s -X DELETE \
    "${GATEWAY_URL}/v1/datasets/${DATASET_NAME}/datapoints" \
    -H "Content-Type: application/json" \
    -d "{\"ids\": ${REMAINING_IDS}}"
fi

# Clean up temporary file
rm "$TEMP_FILE"
