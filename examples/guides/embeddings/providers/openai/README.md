# Guide: How to use OpenAI embedding models with TensorZero

## Running the Example

1. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..." # Replace with your OpenAI API key
```

2. Launch the TensorZero Gateway:

```bash
docker compose up
```

3. Run the example (in a separate terminal):

<details open>
<summary><b>Python (OpenAI SDK)</b></summary>

a. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

b. Run the example:

```bash
uv run openai_sdk.py
```

</details>
