# TensorZero Recipe: Automated Prompt Engineering for LLM

The mipro.ipynb notebook provides a step-by-step recipe to perform automated prompt engineering for OpenAI models based on data collected by the TensorZero Gateway.
Set the OPENAI_API_KEY in the shell your notebook will run in.

## Setup

### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
uv venv  # Create a new virtual environment
source .venv/bin/activate
uv pip sync requirements.txt  # Install the dependencies
```

### Using `pip`

We recommend using Python 3.10+ and a virtual environment.

```bash
pip install -r requirements.txt
```
