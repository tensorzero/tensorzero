# Example: Improving Data Extraction (NER) by Fine-Tuning a Llama 3 Model

## Background

Named Entity Recognition (NER) is the process of identifying and categorizing named entities in text into predefined categories such as person, organization, and location, and date. NER is a fundamental task in natural language processing (NLP) and is widely used in various applications such as information extraction, question answering, and machine translation.

Once upon a time, this was done using rule-based systems or special-purpose models. In light of progress in foundation models, most would use an LLM to address this task today, especially given recent advancements in structured decoding and JSON mode offerings from most inference providers.

Here, we present a stylized example of an NER system that uses TensorZero JSON functions to decode named entities from text.[^1]
Each example in the dataset includes a short segment of text and instructs the model to produce a JSON of named entities in the input.

<details>
<summary>
<b>Sample Data</b>
</summary>

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

In our problem setting, we consider any output that fails to validate against the schema to be incorrect.

We'll show that an optimized Llama 3.1 8B model can be trained to outperform GPT-4o on this task using a small amount of training data, and served by Fireworks at a fraction of the cost and latency.

## Setup

### Prerequisites

1. Install Docker.
2. Install Python 3.10+.
3. Install the Python dependencies: `pip install -r requirements.txt`

### TensorZero

We've written TensorZero configuration files to accomplish this example and have provided them in the `config` directory.
See `tensorzero.toml` for the main configuration details.
We provide the output schema to TensorZero at `config/functions/extract_entities/output_schema.json`.

1. Create a `.env` file with the following environment variables:

   ```
   FIREWORKS_ACCOUNT_ID="xxxxx-xxxxxx"
   FIREWORKS_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

2. Run the following command to start the TensorZero Gateway, the TensorZero UI (`http://localhost:4000/`), and a test ClickHouse database:

   ```bash
   docker compose up
   ```

3. Set `CLICKHOUSE_URL=http://localhost:8123/tensorzero` in the shell your Jupyter notebook will run in.

## Running the Example

You can run the example in the `ner-fine-tuning.ipynb` notebook.

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

> [!TIP]
>
> Restart the TensorZero Gateway when you update the `tensorzero.toml` configuration file.

At the conclusion of that notebook you should see a few blocks to add to `tensorzero.toml` to update the system to use the new model and the corresponding variant.

You'll see that a fine-tuned Llama-3.1 8B model &mdash; even with a small amount of data &mdash; outperforms GPT-4o on this task.

## Experimenting with Improved Variants

Once you've generated one or more improved variants (and, critically, given them some positive weight), you should restart the TensorZero Gateway with the new configuration:

```bash
docker compose up
```

You can then re-run the test set evaluation in the `ner-fine-tuning.ipynb` notebook to see how the new variants perform.

From a single fine-tune we see the Llama-3.1 8B model greatly outperform GPT-4o on this task with just 100-200 examples!

[^1]: We build off of the [CoNLL++ dataset](https://arxiv.org/abs/1909.01441v1) and [work](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora) from Predibase for the problem setting.
