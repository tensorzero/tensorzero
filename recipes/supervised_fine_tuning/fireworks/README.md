# TensorZero Recipe: Supervised Fine-Tuning with Fireworks

The `fireworks.ipynb` notebook provides a step-by-step recipe to perform supervised fine-tuning with Fireworks based on data collected by the TensorZero Gateway. Be sure that `CLICKHOUSE_URL` and `FIREWORKS_API_KEY` are set in your notebook.

## Setup

### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
uv venv  # Create a new virtual environment
uv pip sync requirements.txt  # Install the dependencies
```

### Using `pip`

We recommend using Python 3.10+ and a virtual environment.

```bash
pip install -r requirements.txt
```
