# Tutorial — Part III — Weather RAG

This directory contains the code for the "Weather RAG" tutorial.
You can find the full tutorial [here](https://www.tensorzero.com/docs/gateway/tutorial/).

## Setup

To get started, create a `.env` file with your OpenAI API key (`OPENAI_API_KEY`) and run the following command.
Docker Compose will launch the TensorZero Gateway and a test ClickHouse database.

```bash
docker compose up
```

> Did you get an error mentioning "port is already allocated"? You should kill the containers or services using that port.

## Running the Example

In the tutorial, we discuss how to create a simple weather RAG system using TensorZero.
The weather RAG is defined by the `tensorzero.toml` configuration file (located at `./config/tensorzero.toml`).
The `./config` directory also contains the schemas and templates used by the weather RAG system.

You can run the full example using the `weather-rag.ipynb` notebook.
Make sure to install the `tensorzero` Python package and other dependencies (see `../requirements.txt`).
