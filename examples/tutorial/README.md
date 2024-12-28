# TensorZero Tutorial

This directory contains the code for the **[TensorZero Tutorial](https://www.tensorzero.com/docs/gateway/tutorial)**.

## Running an Example

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
python run.py
```
