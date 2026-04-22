# Code Example: How to create a prompt template

This folder contains the code for the [Guides » Gateway » Create a prompt template](https://www.tensorzero.com/docs/gateway/create-a-prompt-template) page in the documentation.

## Running the Example

1. Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..." # Replace with your OpenAI API key
```

2. Launch the TensorZero Gateway and a local Postgres database:

```bash
docker compose up
```

3. Run the example:

Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

Run the example:

```bash
uv run openai_sdk.py
```
