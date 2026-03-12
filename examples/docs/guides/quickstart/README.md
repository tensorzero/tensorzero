# TensorZero Quickstart

This directory contains the code for the **[TensorZero Quick Start](https://www.tensorzero.com/docs/quickstart)** guide.

## Running the Example

Before running the example, set the `OPENAI_API_KEY` environment variable to your OpenAI API key.

### Python

1. Launch the TensorZero Gateway, the TensorZero UI, and a development ClickHouse database: `docker compose up`
2. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv): `uv sync`
3. Run the example: `python before.py` and `python after.py`

### Node (JavaScript/TypeScript)

1. Launch the TensorZero Gateway, the TensorZero UI, and a development ClickHouse database: `docker compose up`
2. Install the dependencies: `npm install`
3. Run the example: `npm start`
