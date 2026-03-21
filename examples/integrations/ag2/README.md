# Example: Integrating AG2 with TensorZero

## Overview

This example demonstrates how to integrate [AG2](https://ag2.ai/) (formerly AutoGen) with TensorZero to create a simple chatbot that can call external APIs.

AG2 connects to TensorZero via the OpenAI-compatible endpoint. TensorZero manages model routing and observability, while AG2 handles the multi-agent conversation flow and tool execution.

## Getting Started

### TensorZero

We provide a simple TensorZero configuration. AG2 manages the tool definitions and sends them to the model via TensorZero's OpenAI-compatible proxy.

### Prerequisites

1. Install Python 3.10+.
2. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv): `uv sync`
3. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Set the `OPENAI_API_KEY` environment variable.
2. Launch the TensorZero Gateway: `docker compose up`
3. Run the chatbot using `python main.py`. The script is interactive.

## Sample Run

```
user (to assistant):

What is the weather in New York City?

--------------------------------------------------------------------------------
assistant (to user):

***** Suggested tool call: temperature_api *****
Arguments:
{"location":"New York City"}
************************************************

--------------------------------------------------------------------------------

>>>>>>>> EXECUTE FUNCTION temperature_api? (Y/n):
user (to assistant):

***** Response from calling tool: temperature_api *****
{"location": "New York City", "temperature": 25, "unit": "C"}
********************************************************

--------------------------------------------------------------------------------
assistant (to user):

The current temperature in New York City is 25 degrees Celsius. If you need more details about the weather, feel free to ask!
```
