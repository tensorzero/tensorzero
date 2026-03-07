# Code Example: How to fine-tune your LLM with Reinforcement Fine-Tuning (RFT)

This folder contains the code for the [Guides > Optimization > Reinforcement Fine-Tuning](https://www.tensorzero.com/docs/optimization/reinforcement-fine-tuning-rft/) page in the documentation.

This example uses OpenAI, which is currently the only supported provider for RFT.

## Prerequisites

1. Set the `OPENAI_API_KEY` environment variable
2. Set the `TENSORZERO_CLICKHOUSE_URL` environment variable (e.g., `http://chuser:chpassword@localhost:8123/tensorzero`)

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

1. Run inferences on the NER dataset
2. Submit metric feedback for tracking
3. Define an LLM judge grader to evaluate NER quality
4. Launch an RFT job with OpenAI
5. Poll until the job completes
6. Print the configuration needed to use the fine-tuned model

[^1]: We build off of the [CoNLL++ dataset](https://arxiv.org/abs/1909.01441v1) for the problem setting.
