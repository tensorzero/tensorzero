#!/bin/bash

: "${TENSORZERO_API_KEY:?Environment variable TENSORZERO_API_KEY must be set.}"

# Good request

curl -X POST http://localhost:3000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TENSORZERO_API_KEY}" \
  -d '{
    "model": "tensorzero::model_name::openai::gpt-5-mini",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a fun fact."
      }
    ]
  }'

echo ""

# Bad request

curl -X POST http://localhost:3000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-t0-evilevilevil-hackerhackerhackerhackerhackerhackerhackerhacker" \
  -d '{
    "model": "tensorzero::model_name::openai::gpt-5-mini",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a fun fact."
      }
    ]
  }'
