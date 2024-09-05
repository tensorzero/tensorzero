# Tutorial

This directory contains the code for the tutorial.
You can find the full tutorial [here](https://www.tensorzero.com/docs/gateway/tutorial/).

## Setup

First, set the `OPENAI_API_KEY` environment variable.
For the second example (Email Copilot), you will also need to set the `ANTHROPIC_API_KEY` environment variable.

Then, execute the following commands to run the examples.
We use Docker Compose to simplify the setup for the tutorial.
It will launch a ClickHouse database, launch the TensorZero Gateway with the corresponding configuration, and run the API calls using a Python script.

## Examples

You can find the script and configuration files for each example in its corresponding sub-directory.
Feel free to make changes and explore!

### Part I — Simple Chatbot

```bash
docker compose up --build --force-recreate simple-chatbot
```

### Part II — Email Copilot

```bash
docker compose up --build --force-recreate email-copilot
```

### Part III — Weather RAG

```bash
docker compose up --build --force-recreate weather-rag
```

### Part IV — Email Data Extraction

```bash
docker compose up --build --force-recreate email-data-extraction
```
