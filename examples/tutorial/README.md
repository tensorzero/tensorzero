# TensorZero Tutorial

This directory contains the code for the **[TensorZero Tutorial](https://www.tensorzero.com/docs/gateway/tutorial)**.

## Running an Example

1. Set the `OPENAI_API_KEY` environment variable.
   For the second example (Email Copilot), you will also need to set the `ANTHROPIC_API_KEY` environment variable.

2. Launch the TensorZero Gateway and ClickHouse database:

```bash
docker compose up
```

3. Install the dependencies:

```bash
uv venv
uv pip sync requirements.txt
```

or

```bash
# We recommend using Python 3.10+ and a virtual environment
pip install -r requirements.txt
```

4. Run the example:

```bash
python run.py # or run_async.py or run_openai.py
```
