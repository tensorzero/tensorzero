# Tutorial — Part I — Simple Chatbot

This directory contains the code for the "Simple Chatbot" tutorial.
You can find the full tutorial [here](https://www.tensorzero.com/docs/gateway/tutorial/).

## Setup

To get started, create a `.env` file with your OpenAI API key (`OPENAI_API_KEY`) and run the following command.
Docker Compose will launch the TensorZero Gateway and a test ClickHouse database.

```bash
docker compose up
```

> Did you get an error mentioning "port is already allocated"? You should kill the containers or services using that port.

## Running the Example

In the tutorial, we discuss how to create a simple chatbot using TensorZero.
The chatbot is defined by the `tensorzero.toml` configuration file (located at `./config/tensorzero.toml`).

You can run the full example using the `simple-chatbot.ipynb` notebook.
Make sure to install the `tensorzero` Python package and other dependencies (see `../requirements.txt`).
