# Guide: Anthropic

See the [Getting Started with Anthropic](https://www.tensorzero.com/docs/gateway/guides/providers/anthropic) guide for more information.

## Examples

### Python Client

The TensorZero Python client can use a built-in embedded (in-memory) gateway or a standalone HTTP gateway.
We'll use the embedded gateway for this example.

1. Install the Python library: `pip install tensorzero`
2. Make an inference request: `python main.py`

### OpenAI Python SDK

The OpenAI Python SDK can make inference requests to the TensorZero Gateway with the `tensorzero::model_name::` prefix. It can use a built-in embedded (in-memory) gateway or a standalone HTTP gateway.
We'll use the embedded gateway for this example.

1. Install the Python libraries: `pip install openai tensorzero`
2. Make an inference request: `python openai_python_sdk.py`

### OpenAI Node SDK

The OpenAI Node SDK can make inference requests to the TensorZero Gateway using the `tensorzero::model_name::` prefix.

1. Launch a standalone TensorZero Gateway: `docker compose up`
2. Install the OpenAI Node SDK: `npm install`
3. Make an inference request: `node openai_node_sdk.js`

### HTTP

For any other language or platform, you can make inference requests to the TensorZero Gateway using its HTTP API.

1. Launch a standalone TensorZero Gateway: `docker compose up`
2. Make an inference request:

```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "anthropic::claude-3-5-haiku-20241022",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is the capital of Japan?"
        }
      ]
    }
  }'
```
