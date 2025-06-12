# Example: Integrating Helicone with TensorZero

This example demonstrates how to integrate Helicone with TensorZero.

## Getting Started

### TensorZero

We provide a simple TensorZero configuration with two custom models: `helicone_gpt_4o_mini` and `helicone_grok_3`.

### Prerequisites

1. Install Docker.
2. Generate an API key for OpenAI (`OPENAI_API_KEY`).
3. Generate an API key for xAI (`XAI_API_KEY`).
4. Generate an API key for Helicone (`HELICONE_API_KEY`).

### Setup

1. Set the `HELICONE_API_KEY` environment variable in your shell (not in your `.env` file).
2. Create a `.env` file in the root of the repository and set the `OPENAI_API_KEY` and `XAI_API_KEY` environment variables (see `.env.example` for reference).
3. Launch the TensorZero Gateway with `docker compose up`.

### Running the Example

#### Python (TensorZero Client)

1. Install the dependencies with `pip install -r requirements.txt`.
2. Run the example with `python main_tensorzero.py`.

#### Python (OpenAI SDK)

1. Install the dependencies with `pip install -r requirements.txt`.
2. Run the example with `python main_openai.py`.

### Node / TypeScript (OpenAI SDK)

1. Install the dependencies with `npm install`.
2. Run the example with `npx tsx main_openai.ts`.

### HTTP (cURL)

1. Run the example with `bash main_curl.sh`.
