# Guide: How to use Azure embedding models with TensorZero

## Running the Example

1. Set the `AZURE_API_KEY` environment variable:

```bash
export AZURE_API_KEY="..." # Replace with your Azure OpenAI API key
```

2. Set your Azure `endpoint` in `config/tensorzero.toml`.

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
