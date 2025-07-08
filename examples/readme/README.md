# TensorZero `README` Examples

This directory contains the code used for the absolutely minimal examples in the README at the repository root.

## Running the Example

1. Launch the TensorZero Gateway and ClickHouse database:

```bash
docker compose up
```

2. Install the dependencies:

```bash
# We recommend using Python 3.10+ and a virtual environment
pip install -r requirements.txt
```

3. Run the examples:

```bash
python tensorzero_sync_client.py
python tensorzero_async_client.py
python openai_client.py
```
