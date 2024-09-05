# Tutorial

This directory contains the code for the tutorial.
You can find the full tutorial [here](https://www.tensorzero.com/docs/gateway/tutorial/).

## Setup

First, set the `OPENAI_API_KEY` environment variable.
For the second example (Email Copilot), you will also need to set the `ANTHROPIC_API_KEY` environment variable.

Then, execute the following commands to run the examples.
Docker Compose will launch a ClickHouse database, launch the TensorZero Gateway, and run the requests using a Python script.

## Examples

```bash
docker compose up --build --force-recreate simple-chatbot
```

```bash
docker compose up --build --force-recreate email-copilot
```

```bash
docker compose up --build --force-recreate weather-rag
```

```bash
docker compose up --build --force-recreate email-data-extraction
```

You can explore the scripts and configuration files to understand how the the examples work.
Feel free to make changes and explore!
