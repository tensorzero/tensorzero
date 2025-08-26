# Example: Optimizing Data Extraction (NER) with SFT + DPO in TensorZero

## Background

Named Entity Recognition (NER) is the process of identifying and categorizig named entities in text into predefined categories such as person, organization, location, and date. NER is a fundamental task in natural language processing (NLP) and is widely used in various applications such as information extraction, question answering, and machine translation.

#### TODO: Metrics goes here

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

## Getting Started

### TensorZero

We provide a TensorZero configuration file (`config/tensorzero.toml`) to get you started.
The configuration includes a JSON function `extract_entities` with variants for vanilla GPT-4.1 (OpenAI) and GPT-4.1 Mini (OpenAI).
This function uses the output schema in `config/functions/extract_entities/output_schema.json`.

### Prerequisites

1. Install Docker.
2. Install Python 3.10+.
3. Install the Python dependencies with `pip install -r requirements.txt`.
4. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Create a `.env` file with the `OPENAI_API_KEY` environment variable (see `.env.example` for an example).

## Running the Example
The notebook will first attempt to solve the NER task using the `extract_entities` TensorZero JSON function. Under the hood, the TensorZero Gateway will randomly sample either GPT-4.1 or GPT-4.1 Mini each inference.

After completing this process, we evaluate the outputs using exact atch and Jaccard similarity and provide feedback for these metrics to the TensorZero Gateway.

Finally, we run an evaluation on a subset of the validation set to get a clear picture of the performance of each variant.

## Improving the NER System
At this point, your ClickHouse database will include inferences in a structured format along with feedback on how they went. You can now use TensorZero recipes to learn from this experience to produce better variants of the NER system.

In this example, we show how NER can be further refined using a combination of **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimazation (DPO)** with TensorZero.

- **SFT** teaches the model to replicate gold-standard outputs exactly.
- **DPO** bulids on top of the SFT model, aligning it with human or programmatic preference via pairwise comparisons between a preferred and a less-preferred output.

Our goal is to compare **GPT-4.1 Mini** (baseline), **GpT-4.1 Mini + SFT**, and **GPT-4.1 Mini + SFT + DPO**, and show how TensorZero recipe make it easy to combine these methods in a single workflow.

This workflow is broken into two stages:

1. Fine-tune GPT-4.1 Mini with SFT
- Use the TensoZero UI(**Supervised Fine-Tuning** tab) to fine-tune GPT-4.1 Mini. You can run it by opening the UI(`http://localhost:4000/`) clicking on the `Supervised Fine-Tuning` tab. Let's run fine-tuning on GPT-4.1 Mini with OpenAI using the `exact_match` metric. (Fine-tuning can take some time)
- Once the job finishes, add the new fine-tuned model ID to the `tensorzero.toml` file. We only need the `model_name` for creating a new variant.

```toml
[functions.extract_entities.variants.gpt_4_1_mini_sft]
type = "chat_completion"
model = "openai::ft:gpt-4.1-mini-2025-04-14:xxxxxxxx::xxxxxxxx"  # TODO: Replace with your model ID
system_template = "functions/extract_entities/initial_prompt/system_template.minijinja"
json_mode = "strict"
```
**Let's restart the TensorZero Gateway to apply the new configuration.** You can do this by killing the running container and re-running `docker compose up`.

To see the new variant performance, re-run the `ner-sft-dpo.ipynb` notebook.

2. Fine-tune GPT-4.1-mini-sft with DPO
- Launch DPO fine-tuning via `recipes/dpo/openai` notebook.
- Add the resulting DPO model ID to the `tensorzero.toml` file.

```toml
[functions.extract_entities.variants.gpt_4_1_mini_sft_dpo]
type = "chat_completion"
model = "openai::ft:gpt-4.1-mini-2025-04-14:xxxxxxxx::xxxxxxxx"  # TODO: Replace with your model ID
system_template = "functions/extract_entities/initial_prompt/system_template.minijinja"
json_mode = "strict"
```

**Let's restart the TensorZero Gateway to apply the new configuration**, just like we did earlier and re-run the `ner-sft-dpo.ipynb` notebook to see the new variant performance.

**You'll see that each fine-tune improves the performance with just a few hundred examples.**
