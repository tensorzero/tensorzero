# Example: Named Entity Recognition using JSON functions

## Background

Named Entity Recognition (NER) is the process of identifying and categorizing named entities in text into predefined categories such as person, organization, location, and date. NER is a fundamental task in natural language processing (NLP) and is widely used in various applications such as information extraction, question answering, and machine translation.

This was formerly done using rule-based systems or special-purpose models. In light of progress in foundation models, most would use an LLM to address this task today, especially given recent advancements in structured decoding and JSON mode offerings from most inference providers.

Here, we present a stylized example of a NER system that uses TensorZero JSON functions to decode named entities from text.
We build off of the [CoNLL++ dataset](https://arxiv.org/abs/1909.01441v1) and [work](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora) from Predibase for the problem setting.
Each example in the dataset includes a short segment of text and instructs the model to produce a JSON of named entities in the input.
We provide the output schema to TensorZero at `config/functions/extract_entities/output_schema.json`.
In our problem setting, we consider any output that fails to validate against the schema to be incorrect.

## Setup

We've written TensorZero configuration files to accomplish this example and have provided them in the `config` directory.
See `tensorzero.toml` for the main configuration details.

To get started, create a `.env` file with your OpenAI API key (`OPENAI_API_KEY`), Anthropic API key (`ANTHROPIC_API_KEY`), Mistral API key (`MISTRAL_API_KEY`), and Fireworks API key (`FIREWORKS_API_KEY`) and run the following command. Docker Compose will launch the TensorZero gateway and a test ClickHouse database.

```bash
docker compose up -d --wait
```

**Note:** if you do not have one of these keys, you can simply comment out the models, model providers, and variants which use those providers in `tensorzero.toml`.

## Running the Example

You can run the example in the `conll.ipynb` notebook.
Make sure to install the dependencies in the `requirements.txt` file.
It should not require any changes to run and will automatically connect to the TensorZero gateway you started.

The notebook will first attempt to solve the NER task using the `extract_entities` JSON function and randomly sample various LLMs to do it with. After this is done, we evaluate the output using both an exact match metric and Jaccard similarity. We provide feedback in each of these metrics to TensorZero to learn from the results.

Afterwards we run an evaluation on a subset of the test set (and use the same set for each variant) to get a clear picture of the performance of each variant. This inference is performed with a variant specified and `dryrun` set to `true` to avoid storing the data and contaminating the training set.

## Improving the NER System

At this point, your ClickHouse database will include inferences in a structured format along with feedback on how they went.
You can now use TensorZero recipes to learn from this experience to produce better variants of the NER system.
You might notice that the best performing LLM out of those considered is GPT-4o from OpenAI (not surprising!).

However, we offer a recipe in `recipes/supervised_fine_tuning/fireworks/` that can be used with very small amounts of data to fine-tune a Llama-3.1 8B model to achieve superior performance to GPT-4o at a fraction of the cost and latency!
At the conclusion of that notebook you should see a few blocks to add to `tensorzero.toml` to update the system to use the new model and the corresponding variant.

You can also easily experiment with other recipes,models, prompts you think might be better, or combinations thereof by editing the configuration.

## Experimenting with Improved Variants

Once you've generated one or more improved variants (and, critically, given them some positive weight), you should restart the TensorZero gateway with the new configuration:

```bash
docker compose restart gateway
```

You can then re-run the test set evaluation in the `conll.ipynb` notebook to see how the new variants perform.

From a single fine-tune we see the Llama-3.1 8B model greatly outperform GPT-4o on this task with ~100-200 examples!
