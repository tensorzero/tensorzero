## Setup

- Install LiteLLM: `pip install 'litellm[proxy]'`
- Run the LiteLLM Proxy:
  ```
  litellm --config gateway/tests/load/simple-litellm/config.yaml
  ```
- Make a sanity check request to the LiteLLM Proxy:
  ```
  curl --location 'http://localhost:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
        "model": "gpt-3.5-turbo",
        "messages": [
           {
                "role": "user",
                "content": "Is Santa real?"
            }
        ]
    }'
  ```
- Run the load test:
  ```
  sh gateway/tests/load/simple-litellm/run.sh
  ```
