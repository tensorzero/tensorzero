# Guide: How to use Anthropic models with TensorZero

See the [How to use Anthropic models with TensorZero](https://www.tensorzero.com/docs/gateway/guides/providers/anthropic) in the TensorZero documentation for more information.

## Examples

### Python Client

Our Python client can use a built-in embedded gateway or a standalone HTTP gateway.
This example uses the embedded gateway.

1. Install the Python library: `pip install tensorzero`
2. Make an inference request: `python main.py`

### OpenAI Python SDK

The OpenAI Python SDK can use a built-in embedded gateway or a standalone HTTP gateway. It requires the `tensorzero::model_name::` prefix in the model field.
This example uses the embedded gateway.

1. Install the Python libraries: `pip install openai tensorzero`
2. Make an inference request: `python openai_python_sdk.py`

### OpenAI Node SDK

The OpenAI Node SDK requires a standalone HTTP gateway and the `tensorzero::model_name::` prefix in the model field.

1. Launch a standalone TensorZero Gateway: `docker compose up`
2. Install the OpenAI Node SDK: `npm install`
3. Make an inference request: `node openai_node_sdk.js`

### HTTP

For any other language or platform, you can make inference requests to a standalone gateway using its HTTP API.

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
