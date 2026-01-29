# Guide: Streaming Inference

This directory contains the code for the **[Streaming Inference](https://www.tensorzero.com/docs/gateway/guides/streaming-inference)** guide.

This directory contains the code for streaming inference.

## Running the Example

1. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..." # Replace with your OpenAI API key
```

2. Launch the TensorZero Gateway:

```bash
docker compose up
```

3. Run the example:

<details>
<summary><b>HTTP</b></summary>

Run the following command to make a streaming inference request to the TensorZero Gateway:

```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "chatbot",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "Share an extensive list of fun facts about Japan."
        }
      ]
    },
    "stream": true
  }'
```

</details>

<details>
<summary><b>Python</b></summary>

a. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

b. Run the example:

```bash
uv run run.py
```

</details>
