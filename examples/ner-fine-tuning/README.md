# Example: Improving Data Extraction (NER) by Fine-Tuning a Llama 3 Model

## Background

Named Entity Recognition (NER) is the process of identifying and categorizing named entities in text into predefined categories such as person, organization, location, and date. NER is a fundamental task in natural language processing (NLP) and is widely used in various applications such as information extraction, question answering, and machine translation.
Here, we present a stylized example of an NER system that uses TensorZero JSON functions to decode named entities from text. [^1]

Each example in the dataset includes a short segment of text and instructs the model to produce a JSON of named entities in the input.
**We'll show that an optimized Llama 3.1 8B model can be trained to outperform GPT-4o on this task using a small amount of training data, and served by Fireworks AI at a fraction of the cost and latency.**

## Sample Data

### Input

```
The former Wimbledon champion said the immediate future of Australia 's Davis Cup coach Tony Roche could also be determined by events in Split .
```

### Output

```
{
  "person": ["Tony Roche"],
  "organization": [],
  "location": ["Australia", "Split"],
  "miscellaneous": ["Wimbledon", "Davis Cup"]
}
```

</details>

## Setup

### TensorZero

We provide a TensorZero configuration file (`config/tensorzero.toml`) to get you started.
The configuration includes a JSON function `extract_entities` with variants for GPT-4o (OpenAI) and Llama 3.1 8B (Fireworks AI).
This function uses the output schema in `config/functions/extract_entities/output_schema.json`.

### Prerequisites

1. Install Docker.
2. Install Python 3.10+.
3. Install the Python dependencies with `pip install -r requirements.txt`.
4. Create API keys for OpenAI (`OPENAI_API_KEY`) and Fireworks AI (`FIREWORKS_ACCOUNT_ID` and `FIREWORKS_API_KEY`).
5. Create a `.env` file with these environment variables (see `.env.example` for an example).
6. Run `docker compose up` to launch the TensorZero Gateway, the TensorZero UI (`http://localhost:4000/`), and a test ClickHouse database.
7. Set `CLICKHOUSE_URL=http://localhost:8123/tensorzero` in the shell your Jupyter notebook will run in.
8. Run the `ner-fine-tuning.ipynb` notebook.

## Running the Example

The notebook will first attempt to solve the NER task using the `extract_entities` TensorZero JSON function and randomly sample either GPT-4o or vanilla Llama 3.1 8B to do it with.
After this is done, we evaluate the output using both an exact match metric and Jaccard similarity.
We provide feedback in each of these metrics to TensorZero to learn from the results.

Afterwards we run an evaluation on a subset of the test set (and use the same set for each variant) to get a clear picture of the performance of each variant.
This inference is performed with a variant specified and `dryrun` set to `true` to avoid storing the data and contaminating the training set.

## Improving the NER System

At this point, your ClickHouse database will include inferences in a structured format along with feedback on how they went.
You can now use TensorZero recipes to learn from this experience to produce better variants of the NER system.
You might notice that the best performing LLM is GPT-4o from OpenAI (not surprising!).

You can run a fine-tuning recipes by opening the UI (`http://localhost:4000/`) and clicking on the `Supervised Fine-Tuning` tab.

<details>
<summary>
<b>Fine-Tuning Programatically</b>
</summary>

Alternatively, you can run a fine-tuning recipe programatically using the Jupyter notebook in `recipes/supervised_fine_tuning/metrics/fireworks/`.

</details>

Once you complete the fine-tuning recipe, you'll see additional configuration blocks that you can add to your `tensorzero.toml` file.

At the conclusion of that notebook you should see a few blocks to add to `tensorzero.toml` to update the system to use the new model and the corresponding variant.

You'll see that a fine-tuned Llama-3.1 8B model &mdash; even with a small amount of data &mdash; outperforms GPT-4o on this task.

> [!TIP]
>
> Restart the TensorZero Gateway when you update the `tensorzero.toml` configuration file by killing the running container and running `docker compose up` again.

## Experimenting with Improved Variants

Once you've generated one or more improved variants (and, critically, given them some positive weight), you can re-run the test set evaluation in the `ner-fine-tuning.ipynb` notebook to see how the new variants perform.

From a single fine-tune we see the Llama-3.1 8B model greatly outperform GPT-4o on this task with just 100-200 examples!

[^1]: We build off of the [CoNLL++ dataset](https://arxiv.org/abs/1909.01441v1) and [work](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora) from Predibase for the problem setting.
