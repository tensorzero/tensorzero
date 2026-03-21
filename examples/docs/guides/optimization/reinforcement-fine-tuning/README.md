# Code Example: How to fine-tune your LLM with Reinforcement Fine-Tuning (RFT)

This folder contains the code for the [Guides > Optimization > Reinforcement Fine-Tuning](https://www.tensorzero.com/docs/optimization/reinforcement-fine-tuning-rft/) page in the documentation.

This example uses OpenAI, which is currently the only supported provider for RFT.

## Prerequisites

1. Set the `OPENAI_API_KEY` environment variable

## Running the Example

1. Start the required services:

   ```bash
   docker compose up -d
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Run the example:

   ```bash
   uv run python main.py
   ```

The script will:

1. Create datapoints from the NER dataset[^1] in a TensorZero dataset
2. Define an LLM judge grader to evaluate NER quality
3. Launch an RFT job with OpenAI
4. Poll until the job completes
5. Print the configuration needed to use the fine-tuned model

[^1]: We build off of the [CoNLL++ dataset](https://arxiv.org/abs/1909.01441v1) for the problem setting.
