# Example: Writing haikus to satisfy a judge with hidden preferences

## Background

Many real-world applications involve producing some content that a human responds to. For example, advertisements are designed to be shown to humans and influence them to purchase a product. Alternately, an email draft may be pre-written for a human as a starting point for their approval or editing.

In these cases, the content creator has a rough idea of what preferences humans broadly have but may not know the specific preferences of a particular individual.
Over time, if feedback can be collected from the user, the system can learn to produce content that satisfies the user's preferences.

In this example, we'll give a stylized example of such a system. For simplicity,
we have built a system that writes haikus to satisfy a critic with hidden preferences.
The critic will accept or reject haikus, and the haiku writer will learn to write haikus that satisfy the critic's preferences over time.

We use a list of common nouns provided by Desi Quintans [here](https://www.desiquintans.com/nounlist) as the set of topics for the haikus.

## Setup

We've written TensorZero configuration files to accomplish this example and have provided them in the `config` directory. See `tensorzero.toml` for the main configuration details. If you're curious what the judge's preferences are, you can look at `config/functions/judge_haiku/judge_prompt/system_template.minijinja`.

To start the TensorZero gateway locally with this configuration, run `cargo run --bin gateway -- config/tensorzero.toml`.
