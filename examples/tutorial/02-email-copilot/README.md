# Tutorial — Part II — Email Copilot

This directory contains the code for the "Email Copilot" tutorial.
You can find the full tutorial [here](https://www.tensorzero.com/docs/gateway/tutorial/).

## Setup

To get started, create a `.env` file with your OpenAI API key (`OPENAI_API_KEY`) and your Anthropic API key (`ANTHROPIC_API_KEY`).
Then, run the following command.
Docker Compose will launch the TensorZero Gateway and a test ClickHouse database.

```bash
docker compose up
```

> Did you get an error mentioning "port is already allocated"? You should kill the containers or services using that port.

## Running the Example

In the tutorial, we discuss how to create an LLM-powered email copilot using TensorZero.
The email copilot is defined by the `tensorzero.toml` configuration file (located at `./config/tensorzero.toml`).
The `./config` directory also contains the schemas and templates used by the email copilot.

You can run the full example using the `email-copilot.ipynb` notebook.
Make sure to install the `tensorzero` Python package and other dependencies (see `../requirements.txt`).
