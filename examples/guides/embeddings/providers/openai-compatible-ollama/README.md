# Guide: How to use OpenAI-compatible embedding models (e.g. Ollama) with TensorZero

## Running the Example

1. Launch the Ollama server:

```bash
ollama serve
```

2. Download an embedding model:

```bash
ollama pull nomic-embed-text
```

3. Run the example:

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
