# TensorZero README example

This directory contains the code used for the absolutely minimal example in the README at the repository root.

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
python sync_client_example.py
python async_client_example.py
python openai_client_example.py
```
