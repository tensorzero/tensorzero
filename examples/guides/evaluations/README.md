# Guide: TensorZero Evals

This directory contains the code for the TensorZero Evals guide. [TODO, DO NOT MERGE: link]

docker build -t tensorzero/ui:latest -f ui/Dockerfile .

docker build -t tensorzero/evaluations:latest -f ui/Dockerfile .

TODO

## Getting Started

### TensorZero

### Prerequisites

1. Install Docker.
2. Install Python 3.10+.
3. Install the Python dependencies with `pip install -r requirements.txt`.
4. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Create a `.env` file with the `OPENAI_API_KEY` environment variable (see `.env.example` for an example).
2. Run `docker compose up` to launch the TensorZero Gateway, the TensorZero UI, and a development ClickHouse database.
3. Run the `main.py` script to generate 100 haikus.
4. TODO...

### Evaluations

#### TensorZero UI

#### CLI
