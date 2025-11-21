#!/bin/bash

# Retrieve the Helicone API key from the environment variable
if [ -z "$HELICONE_API_KEY" ]; then
  echo "HELICONE_API_KEY is not set"
  exit 1
fi

# Test our `helicone_gpt_4o_mini` model
curl http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "helicone_gpt_4o_mini",
    "input": {"messages": [{"role": "user", "content": "Who is the CEO of OpenAI?"}]},
    "extra_headers": [{
      "model_name": "helicone_gpt_4o_mini",
      "provider_name": "helicone",
      "name": "Helicone-Auth",
      "value": "Bearer '"$HELICONE_API_KEY"'"
    }]
  }'

echo "\n"

# Test our `helicone_grok_3` model
curl http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "helicone_grok_3",
    "input": {"messages": [{"role": "user", "content": "Who is the CEO of xAI?"}]},
    "extra_headers": [{
      "model_name": "helicone_grok_3",
      "provider_name": "helicone",
      "name": "Helicone-Auth",
      "value": "Bearer '"$HELICONE_API_KEY"'"
    }]
  }'
