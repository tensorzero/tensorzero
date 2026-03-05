# Guide: Metrics & Feedback

This directory contains the code for the **[Metrics & Feedback](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback)** guide.

## Running the Example

1. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..." # Replace with your OpenAI API key
```

2. Launch the TensorZero Gateway, the TensorZero UI, and a local ClickHouse database:

```bash
docker compose up
```

3. Run the example:

<details>
<summary><b>Python</b></summary>

a. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

b. Run the example:

```bash
uv run main.py
```

</details>

<details>
<summary><b>HTTP</b></summary>

```bash
curl -X POST http://localhost:3000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "haiku_rating",
    "inference_id": "00000000-0000-0000-0000-000000000000",
    "value": true
  }';
```

</details>
