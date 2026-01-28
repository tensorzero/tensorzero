# Example: Tool Use (Function Calling)

This directory contains the code for the **[Tool Use (Function Calling)](https://www.tensorzero.com/docs/gateway/guides/tool-use)** guide.

## Getting Started

### TensorZero

We provide a simple TensorZero configuration with a `weather_chatbot` chat function that has access to a `get_temperature` tool.

### Prerequisites

1. Install Docker.
2. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Set the `OPENAI_API_KEY` environment variable in your shell (not in your `.env` file).
2. Launch the TensorZero Gateway with `docker compose up`.

### Running the Example

#### Python (TensorZero Client)

1. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv): `uv sync`
2. Run the example with `python main_tensorzero.py`.

#### Python (OpenAI SDK)

1. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv): `uv sync`
2. Run the example with `python main_openai.py`.

### Node / TypeScript (OpenAI SDK)

1. Install the dependencies with `npm install`.
2. Run the example with `npx tsx main_openai.ts`.

### HTTP (cURL)

1. Run the example with `bash main_curl.sh`.
