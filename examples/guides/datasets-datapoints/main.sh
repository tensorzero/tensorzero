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
echo -e "POST ${GATEWAY_URL}/datasets/${DATASET_NAME}/datapoints\n"
BULK_INSERT_RESPONSE=$(curl -s -X POST \
  "${GATEWAY_URL}/datasets/${DATASET_NAME}/datapoints" \
  -H "Content-Type: application/json" \
  -d @"$TEMP_FILE")

echo "$BULK_INSERT_RESPONSE"

# Check if bulk insert was successful by verifying it's a valid JSON array
if ! echo "$BULK_INSERT_RESPONSE" | jq -e 'type == "array"' >/dev/null; then
    echo "Error: Bulk insert failed - invalid response format"
    rm "$TEMP_FILE"
    exit 1
fi

# Extract the first datapoint ID from the response
FIRST_DATAPOINT_ID=$(echo "$BULK_INSERT_RESPONSE" | jq -r '.[0]')

# Check if we got a valid ID
if [ -z "$FIRST_DATAPOINT_ID" ] || [ "$FIRST_DATAPOINT_ID" = "null" ]; then
    echo "Error: Could not get valid datapoint ID"
    rm "$TEMP_FILE"
    exit 1
fi

# 2. Get a single datapoint
echo -e "\nGET ${GATEWAY_URL}/datasets/${DATASET_NAME}/datapoints/${FIRST_DATAPOINT_ID}\n"
GET_RESPONSE=$(curl -s -X GET \
  "${GATEWAY_URL}/datasets/${DATASET_NAME}/datapoints/${FIRST_DATAPOINT_ID}")

echo "$GET_RESPONSE"

# 3. Delete the first datapoint
echo -e "\nDELETE ${GATEWAY_URL}/datasets/${DATASET_NAME}/datapoints/${FIRST_DATAPOINT_ID}"
curl -s -X DELETE \
  "${GATEWAY_URL}/datasets/${DATASET_NAME}/datapoints/${FIRST_DATAPOINT_ID}"

# 4. List all datapoints
echo -e "\nGET ${GATEWAY_URL}/datasets/${DATASET_NAME}/datapoints\n"
LIST_RESPONSE=$(curl -s -X GET \
  "${GATEWAY_URL}/datasets/${DATASET_NAME}/datapoints")

echo "$LIST_RESPONSE"

# 5. Clean up - delete remaining datapoints
echo "$LIST_RESPONSE" | jq -r '.[].id' | while read -r datapoint_id; do
  curl -s -X DELETE \
    "${GATEWAY_URL}/datasets/${DATASET_NAME}/datapoints/${datapoint_id}"
done

# Clean up temporary file
rm "$TEMP_FILE"
