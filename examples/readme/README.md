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

3. Run the example:

```bash
python sync_client_example.py
python async_client_example.py
```
