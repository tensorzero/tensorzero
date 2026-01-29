# Example: LLMs Learn to Navigate Mazes from Experience (BabyAI Benchmark)

## Background: BabyAI

[BabyAI](https://github.com/mila-iqia/babyai) is a grid world environment designed to test the sample efficiency of grounded language acquisition.
Each task is described in natural language (e.g. `put the red ball next to the blue ball`).
To complete a task, the agent must execute a sequence of actions given partial observations of the environment.
The set of actions are `go forward`, `turn right`, `turn left`, `pick up`, `drop`, and `toggle`.
An example observation is `you carry a yellow ball\n a wall 2 steps right\n a red ball 1 step forward`.

<p align="center">
  <img src=img/babyai.png width="400" height="400" alt="BabyAI">
</p>

We use the [BALROG agentic LLM benchmark](https://github.com/balrog-ai/BALROG) implementation of the BabyAI environment to demonstrate how you can use TensorZero to develop an LLM application to solve such tasks.

## Getting Started

### TensorZero

We provide a TensorZero configuration file to tackle BALROG's BabyAI benchmark.
Our setup implements the function `act` with multiple variants (e.g. baseline, reasoning, history).

### Prerequisites

1. Install `cmake` (e.g. `brew install cmake` or `sudo apt-get install cmake`)
2. Install Docker
3. Install Python 3.10+
4. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv): `uv sync`
5. Generate an API key for OpenAI (`OPENAI_API_KEY`)

### Setup

1. Set the `OPENAI_API_KEY` environment variable
2. Run `docker compose up` to launch the TensorZero Gateway, the TensorZero UI, and a development ClickHouse database
3. Run the `babyai.ipynb` Jupyter notebook

Here are our results showing the success rate, episode return, episode length, and input tokens.
We find that the history_and_reasoning variant perfoms best and that using our fine-tuning recipe can improve its performance.

## Running the Example

The notebook will evaluate the performance of multiple variants that use the `gpt-4o-mini` model.
You'll notice that adding history to the prompt improves performance.

Later, you can use the a fine-tuning recipe to improve the performance of the `history_and_reasoning` variant.
The simplest way to fine-tune a model is to use the TensorZero UI (available at `http://localhost:4000`).
The fine-tuned variant will achieve materially higher success rate and episode return.
