# Example: Multimodal (Vision) Finetuning

This example demonstrates how to finetune a vision-language model (VLM) using TensorZero.

To keep things simple, we'll tackle a naive task: classifying arXiv papers based on an image of the paper's first page, without providing any instructions about the taxonomy of the categories.
Before fine-tuning, unsurprisingly, the model completely fails at this task due to the lack of context (e.g. with outputs such as `{"category": "Academic Paper"}`).
After fine-tuning, the model achieves high accuracy on a held-out test set (e.g. with outputs such as `{"category": "cs.CV"}`).

## Getting Started

### TensorZero

We provide a simple configuration in `config/tensorzero.toml`.
The configuration specifies a straightforward JSON function `classify_document` with a single variant that uses GPT-4o Mini and a boolean metric `correct_classification`.

### Prerequisites

1. Install Docker.
2. Install Python 3.10+.
3. Generate an OpenAI API key.

### Dataset

We generate a dataset of 20 papers for each of arXiv's computer science (`cs.*`) categories.
The dataset includes an image of the paper's first page and the correct category.

To download a pre-generated dataset, run the following commands:

```bash
wget https://assets.tensorzero.com/examples/multimodal-vision-finetuning.tar.gz
tar -xvf multimodal-vision-finetuning.tar.gz -C .
```

To generate the dataset from scratch, see `data/generate_dataset.ipynb`.

### Setup

1. Set the `OPENAI_API_KEY` environment variable to your OpenAI API key.
2. Run `docker compose up` to start TensorZero.
3. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv): `uv sync`
4. Run the notebook: `main.ipynb`

### Fine-tuning

After running the notebook with the `baseline` variant, you can fine-tune a model using the TensorZero UI.

1. Visit `http://localhost:4000`.
2. Go to `Supervised Finetuning`.
3. Start a fine-tuning job using `demonstration` as the metric.

> [!TIP]
>
> Most models don't support multi-modal fine-tuning.
> We recommend using `gpt-4o-2024-08-06` for this example.

After completion, create a new variant in your `config/tensorzero.toml` file.
It should look like this:

```toml
[functions.classify_document.variants.finetuned]
type = "chat_completion"
model = "openai::ft:gpt-4o-2024-08-06:tensorzero::xxxxxxx"
json_mode = "strict"
retries = { num_retries = 2, max_delay_s = 10 }
```

Restart the gateway to apply the configuration changes and re-run the notebook with the `finetuned` variant.
You'll notice that the model now achieves high accuracy on the held-out test set.
