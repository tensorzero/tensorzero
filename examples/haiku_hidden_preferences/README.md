# Example: Writing haikus to satisfy a judge with hidden preferences

## Background

Many real-world applications involve producing some content that a human responds to. For example, advertisements are designed to be shown to humans and influence them to purchase a product. Alternately, an email draft may be pre-written for a human as a starting point for their approval or editing.

In these cases, the content creator has a rough idea of what preferences humans broadly have but may not know the specific preferences of a particular individual or group.
Over time, if feedback can be collected from the user, the system can learn to produce content that satisfies the user's preferences.

In this example, we'll give a stylized example of such a system. For simplicity,
we have built a system that writes haikus about a variety of topics to satisfy an LLM judge with hidden preferences.
The judge will accept or reject haikus, and the haiku writer will learn to write haikus that satisfy the judge's preferences over time.
The judge's preferences are hidden from the haiku writer and will not change during this example.

We use a list of common nouns provided by Desi Quintans [here](https://www.desiquintans.com/nounlist) as the set of topics for the haikus.

## Setup

We've written TensorZero configuration files to accomplish this example and have provided them in the `config` directory. See `tensorzero.toml` for the main configuration details. If you're curious what the judge's preferences are, you can look at `config/functions/judge_haiku/judge_prompt/system_template.minijinja`.

To start the TensorZero gateway locally with this configuration, run `docker compose up -d`.

## Running the example

You can run the example in the `haiku.ipynb` notebook. It should not require any changes to run and will automatically connect to the TensorZero gateway you started. The cell that actually writes and judges the haikus will take a few minutes to run.

After the haikus are judged and the feedback has been posted to the TensorZero gateway, you can see the average score for each haiku variant in the last cell of the notebook.
If this is the first time you've run the example, you should expect a single variant with a fairly low score (around 17%).
This is because the haiku writer has no knowledge of the critic's idiosyncratice preferences and is writing haikus with no guidance.

## Improving the haiku writer

At this point, your ClickHouse database will include inferences in a structured format along with feedback on how they went.
You can now use TensorZero recipes to learn from this experience to produce better variants of the haiku writer.
We recommend starting with supervised fine-tuning of a custom OpenAI model using the notebook in `recipes/supervised_fine_tuning/openai.ipynb`.
Each recipe should print some additional elements to add to the `tensorzero.toml` file.

You can also easily experiment with other model types, prompts you think might be better, or combinations thereof by editing the `tensorzero.toml` file and writing minijinja templates directly.

## Experimenting with improved variants

Once you've generated one or more improved variants (and, critically, given them some positive weight), you should restart the TensorZero gateway with the new configuration using `docker compose restart gateway`.
You can then run the haiku generation cell in the `haiku.ipynb` notebook again to see how the new variants perform.

From a single fine-tune we typically see an improvement from 17% to 28% or so. Not bad, given that we only had a few dozen good examples and binary labels!
