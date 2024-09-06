# Example: GSM8K with DSPy

TensorZero provides a number of pre-built optimization recipes covering common LLM engineering workflows.
But you can also easily create your own recipes and workflows!

This example shows how to optimize a TensorZero function using an arbitrary tool — namely, [DSPy](https://github.com/stanfordnlp/dspy).

## Background

The GSM8K [dataset](https://github.com/openai/grade-school-math), introduced in a [paper](https://arxiv.org/abs/2110.14168) from OpenAI, is a collection of ~8,000 grade school math word problems and their solutions.
It has lately seen extensive use as a simple benchmark for evaluating LLMs.
We include an example of a very simple implementation of a TensorZero function for solving GSM8K in this example.
Since this benchmark is relatively easy, we have configured this example to use the Llama 3.1 8B model (zero-shot) served on Together's API.

After running the example and generating some data, we show how to query a dataset of inferences from the ClickHouse database in order to optimize the prompt using a DSPy teleprompter.

## Setup

### TensorZero

We've written TensorZero configuration files to accomplish this example and have provided them in the `config` directory.
See `tensorzero.toml` for the main configuration details.

To get started, create a `.env` file with your Together API key (`TOGETHER_API_KEY`) and run the following command. Docker Compose will launch the TensorZero gateway and a test ClickHouse database.

```bash
docker compose up
```

### Python Environment

#### Using [`uv`](https://github.com/astral-sh/uv) (Recommended)

```bash
uv venv  # Create a new virtual environment
uv pip sync requirements.txt  # Install the dependencies
```

#### Using `pip`

We recommend using Python 3.10+ and a virtual environment.

```bash
pip install -r requirements.txt
```

## Running the Example

You can run the example in the `gsm8k_dspy.ipynb` notebook.
Make sure to install the dependencies in the `requirements.txt` file.
It should not require any changes to run and will automatically connect to the TensorZero Gateway you started.

Llama 3.1 8B with a very basic zero-shot prompt should score around 60% on this dataset out of the box.

## Improving the GSM8K Solver

At this point, your ClickHouse database will include inferences in a structured format along with feedback on how they went.
You can now use TensorZero recipes or DSPy itself to learn from this experience to produce better variants of the GSM8K solver.
Each recipe should print some additional elements to add to the `tensorzero.toml` file.

If you follow along further in the notebook, we use DSPy to generate an prompt using in-context learning that you can also evaluate.

You can also easily experiment with other models, prompts you think might be better, or combinations thereof by editing the configuration.

## Experimenting with Improved Variants

Once you've generated one or more improved variants (and, critically, given them some positive weight), you should restart the TensorZero gateway with the new configuration:

```bash
docker compose up
```

You can then re-run the test evaluation cell in the `gsm8k_dspy.ipynb` notebook to see how the new variants perform.
If you use the DSPy code in the notebook, you should see an improvement in performance from ~60% to ~80%!
