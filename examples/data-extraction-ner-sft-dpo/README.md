# Example: Improving Data Extraction (NER) with SFT + DPO in TensorZero

## Background

Named Entity Recognition (NER) is the process of identifying and categorizig named entities in text into predefined categories such as person, organization, location, and date. NER is a fundamental task in natural language processing (NLP) and is widely used in various applications such as information extraction, question answering, and machine translation.

In this example, we extend the original NER optimazation example to show how **Supervised Fine-Tuning (SFT)** can be further refined using **Direct Preference Optimazation (DPO)** with TensorZero.

- **SFT** teaches the model to replicate gold-standard outputs exactly.
- **DPO** bulids on top of the SFT model, aligning it with human or programmatic preference via pairwise comparisons between a preferred and a less-preferred output.

Our goal is to compare **GPT-40 Mini** (baseline), **GpT-40 Mini + SFT**, and **GPT-40 Mini + SFT + DPO**, and show how TensorZero recipe make it easy to combine these methods in a single workflow.

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
The configuration includes a JSON function `extract_entities` with variants for vanilla GPT-4o (OpenAI) and GPT-4o Mini (OpenAI).
This function uses the output schema in `config/functions/extract_entities/output_schema.json`.

### Prerequisites

1. Install Docker.
2. Install Python 3.10+.
3. Install the Python dependencies with `pip install -r requirements.txt`.
4. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Create a `.env` file with the `OPENAI_API_KEY` environment variable (see `.env.example` for an example).

## Running the Example
This workflow is broken into three stages:

1. Run baseline NER with GPT-40 and GPT-40 Mini
- Launch the TensorZero Gateway
```
docker compose up
```
- Open and run the `ner-sft-dpo.ipynb` notebook to perform NER inference and collect evaluation metrics (Exat Match, Jaccard Similarity).

2. Fine-tune GPT-40 Mini with SFT
- Use the TensoZero UI(**Supervised Fine-Tuning** tab) to fine-tune GPT-40 Mini. You can run it by opening the UI(`http://localhost:4000/`) clicking on the `Supervised Fine-Tuning` tab. Let's run fine-tuning on GPT-4o Mini with OpenAI using the `exact_match` metric. (Fine-tuning can take some time)
- Once the job finishes, upload `tensorzero.toml` with your new model ID. We only need the `model_name` for creating a new variant.

```toml
[functions.extract_entities.variants.gpt_4o_mini_fine_tuned]
type = "chat_completion"
model = "openai::ft:gpt-4.1-mini-2025-04-14:xxxxxxxx::xxxxxxxx"  # TODO: Replace with your model ID
system_template = "functions/extract_entities/initial_prompt/system_template.minijinja"
json_mode = "strict"
```
