# Example: Writing Haikus to Satisfy a Judge with Hidden Preferences

## Background

Many LLM applications produce content that depends on the preferences of an individual or group.
For example, LLMs can generate advertising copy to drive purchasing decisions or draft emails as a starting point for a human to edit.

In these cases, the developer might have a rough idea of what preferences humans broadly have but may not know the specific preferences of a particular individual or group.
Over time, if feedback signals can be collected, the system can learn to produce content that satisfies preferences.

Here we present a stylized example of such a system.
We've built a system that writes haikus about a variety of topics to satisfy an LLM judge with hidden preferences.
The judge accepts or rejects haikus, and the haiku writer needs to learn to write haikus that satisfy the judge's preferences.
The judge's preferences are hidden from the haiku writer and do not change over time.

We use a [list of common nouns](https://www.desiquintans.com/nounlist) as the set of topics for the haikus.

## Setup

## TensorZero

We've written TensorZero configuration files to accomplish this example and have provided them in the `config` directory.
See `tensorzero.toml` for the main configuration details.
You can also find the judge's preferences in `config/functions/judge_haiku/judge_prompt/system_template.minijinja`.

To get started, create a `.env` file with your OpenAI API key (`OPENAI_API_KEY`) and run the following command.
Docker Compose will launch the TensorZero Gateway and a test ClickHouse database.
Set `TENSORZERO_CLICKHOUSE_URL=http://localhost:8123/tensorzero` in the shell your notebook will run in.

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

You can run the example in the `haiku.ipynb` notebook.
Make sure to install the dependencies in the `requirements.txt` file.
It should not require any changes to run and will automatically connect to the TensorZero gateway you started.

After the haikus are judged and the feedback has been posted to the TensorZero gateway, you can see the average score for each haiku variant in the last cell of the notebook.
If this is the first time you've run the example, you should expect a single variant with a fairly low score (typically ~15%).
This is because the haiku writer has no knowledge of the critic's idiosyncratic preferences.

## Improving the Haiku Writer

At this point, your ClickHouse database will include inferences in a structured format along with feedback on how they went.
You can now use TensorZero recipes to learn from this experience to produce better variants of the haiku writer.
We recommend starting with supervised fine-tuning of a custom OpenAI model using the notebook in `recipes/supervised_fine_tuning/metrics/openai.ipynb`.
Each recipe should print some additional elements to add to the `tensorzero.toml` file.

You can also easily experiment with other models, prompts you think might be better, or combinations thereof by editing the configuration.

## Experimenting with Improved Variants

Once you've generated one or more improved variants (and, critically, given them some positive weight), you should restart the TensorZero gateway with the new configuration.
You can then re-run the haiku generation cell in the `haiku.ipynb` notebook to see how the new variants perform.

From a single fine-tune we typically see a relative improvement of ~50% in the haiku score.
Not bad, given that we only had a few dozen good examples and binary labels!
