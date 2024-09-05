# Example: GSM8K with DSPy

## Background

The GSM8K [dataset](https://github.com/openai/grade-school-math), introduced in a [paper](https://arxiv.org/abs/2110.14168) from OpenAI, is a collection of ~8,000 grade school math word problems and their solutions.
It has lately seen extensive use as a simple / easy benchmark for evaluating LLMs.
We include an example of a very simple implementation of a TensorZero function for solving GSM8K in this example.
Since this benchmark is relatively easy, we have configured this example to use the Llama 3.1 8B model (zero-shot)served on Together's API.

After running the example and generating some data, we show how to query a dataset of inferences from the ClickHouse database in order to optimize the prompt using a DSPy teleprompter.

## Setup

We've written TensorZero configuration files to accomplish this example and have provided them in the `config` directory.
See `tensorzero.toml` for the main configuration details.

To get started, create a `.env` file with your Together API key (`TOGETHER_API_KEY`) and run the following command. Docker Compose will launch the TensorZero gateway and a test ClickHouse database.

```bash
docker compose up -d --wait
```

## Running the Example

You can run the example in the `gsm8k_dspy.ipynb` notebook.
Make sure to install the dependencies in the `requirements.txt` file.
It should not require any changes to run and will automatically connect to the TensorZero gateway you started.

Llama 3.1 8B with a very basic zero-shot promptshould score around 30% on this dataset out of the box.

## Improving the GSM8K Solver

At this point, your ClickHouse database will include inferences in a structured format along with feedback on how they went.
You can now use TensorZero recipes or DSPy itself to learn from this experience to produce better variants of the GSM8K solver.
Each recipe should print some additional elements to add to the `tensorzero.toml` file.

You can also easily experiment with other models, prompts you think might be better, or combinations thereof by editing the configuration.

## Experimenting with Improved Variants

Once you've generated one or more improved variants (and, critically, given them some positive weight), you should restart the TensorZero gateway with the new configuration:

```bash
docker compose up
```

You can then re-run the test evaluation cell in the `gsm8k_dspy.ipynb` notebook to see how the new variants perform.
