# TensorZero Recipe: GEPA Optimization

The `gepa.ipynb` notebook provides a step-by-step recipe to optimize prompts using GEPA (Genetic-Pareto prompt optimization) based on data collected by the TensorZero Gateway.

GEPA evolves prompts through an iterative process of evaluation, analysis, and mutation to improve performance on your defined metrics.

Set `TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/tensorzero` and `OPENAI_API_KEY` in the shell your notebook will run in.

## Setup

### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
uv venv  # Create a new virtual environment
uv sync  # Install the dependencies
```

### Using `pip`

We recommend using Python 3.10+ and a virtual environment.

```bash
pip install -r requirements.txt
```

## Expected Results

Running this notebook on the NER example (`examples/data-extraction-ner`) produces significant improvements in both exact match and Jaccard similarity metrics.

![Metrics by Variant](visualization.svg)

GEPA optimization improved gpt-4o-mini from 9.4% to 62.4% exact match (+53pp) and gpt-4o from 47.6% to 74.8% exact match (+27pp), demonstrating that prompt evolution can substantially boost performance without changing the underlying model.
