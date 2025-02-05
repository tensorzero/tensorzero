# TensorZero Quickstart

This directory contains the code for the **[TensorZero Quick Start](https://www.tensorzero.com/docs/quickstart)** guide.

## Running the Example

1. Launch the TensorZero Gateway and ClickHouse database:

```bash
docker compose up
```

2. Install the dependencies:

```bash
uv venv
uv pip sync requirements.txt
```

or

```bash
# We recommend using Python 3.10+ and a virtual environment
pip install -r requirements.txt
```

3. Run the example:

```bash
python before.py
python after.py # or after_async.py or after_openai.py
```
