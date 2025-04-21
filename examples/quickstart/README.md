# TensorZero Quickstart

This directory contains the code for the **[TensorZero Quick Start](https://www.tensorzero.com/docs/quickstart)** guide.

## Running the Example

### Python

1. Launch the TensorZero Gateway, the TensorZero UI, and a development ClickHouse database: `docker compose up`
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -e .
   ```
4. Create a `.env` file and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   You can use `.env.example` as a template
5. Run the example: `python before.py` and `python after.py`

> These scripts will automatically load the `.env` file using `python-dotenv`

### Node (JavaScript/TypeScript)

1. Launch the TensorZero Gateway, the TensorZero UI, and a development ClickHouse database: `docker compose up`
2. Install the dependencies: `npm install`
3. Create a `.env` file with your OpenAI API key (you can copy from `.env.example`)
4. Run the example: `npm start`

> The Node script also loads the `.env` file automatically using `dotenv`
