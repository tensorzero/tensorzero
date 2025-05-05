# TensorZero Recipe: DPO (Preference Fine-tuning) with OpenAI

The `openai.ipynb` notebook provides a step-by-step recipe to perform **Direct Preference Optimization (DPO) &mdash; also known as Preference Fine-tuning &mdash; of OpenAI models** based on data collected by the TensorZero Gateway.

Set `TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero` and `OPENAI_API_KEY` in the shell your notebook will run in.

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
