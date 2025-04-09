# Guide: TensorZero Evaluations

This directory contains the code for the **[TensorZero Evaluations Guide](https://www.tensorzero.com/docs/evaluations/guide)**.

## Getting Started

### TensorZero

We provide a configuration file (`./config/tensorzero.toml`) that specifies:

- A `write_haiku` function that generates a haiku, with `gpt_4o` and `gpt_4o_mini` variants.
- A `haiku_eval` evaluation, with evaluators for exact match and LLM judges.

### Prerequisites

1. Install Docker.
2. Install Python 3.10+.
3. Install the Python dependencies with `pip install -r requirements.txt`.
4. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Create a `.env` file with the `OPENAI_API_KEY` environment variable (see `.env.example` for an example).
2. Run `docker compose up` to launch the TensorZero Gateway, the TensorZero UI, and a development ClickHouse database.
3. Run the `main.py` script to generate 100 haikus.

### Evaluations

#### Create a Dataset

Let's generate a dataset composed of our 100 haikus.

1. Open the UI, navigate to "Datasets", and select "Build Dataset" (`http://localhost:4000/datasets/builder`).
2. Create a new dataset for the `write_haiku` called `haiku_dataset`.
3. Select "Datasets" in the sidebar, with "Inference" as the dataset output.

> [^TIP]
>
> You can also create datasets programmatically with arbitrary data using the gateway. See the **[TensorZero Evaluations Configuration Reference](https://www.tensorzero.com/docs/evaluations/configuration-reference)**.

#### Run an Evaluation &mdash; CLI

Let's evaluate our `gpt_4o` variant using the TensorZero Evaluations CLI tool.

1. Launch an evaluation with the CLI:

```bash
docker compose run --rm evaluations \
    --evaluation-name haiku_eval \
    --dataset-name haiku_dataset \
    --variant-name gpt_4o \
    --concurrency 5
```

#### Evaluate a Dataset &mdash; UI

Let's evaluate our `gpt_4o_mini` variant using the TensorZero Evaluations UI, and compare the results.

1. Navigate to "Evaluations" and select "New Run" (`http://localhost:4000/evaluations`).
2. Launch an evaluation with the `gpt_4o_mini` variant.
3. Select the previous evaluation run in the dropdown to compare the results.
