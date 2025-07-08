# Example: Integrating LangGraph with TensorZero

## Overview

This example demonstrates how to integrate LangGraph with TensorZero to create a simple chatbot that can call external APIs.

## Getting Started

### TensorZero

We provide a simple TensorZero configuration with a function `chatbot` that uses GPT-4o Mini and has access to a tool called `temperature_api`.

### Prerequisites

1. Install Python 3.10+.
2. Install the Python dependencies with `pip install -r requirements.txt`.
3. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Set the `OPENAI_API_KEY` environment variable.
2. Run the chatbot using `python main.py`. The script is interactive.

## Sample Run

```
                +-----------+
                | __start__ |
                +-----------+
                      *
                      *
                      *
                 +---------+
                 | chatbot |
                 +---------+
               ...          ...
              .                .
            ..                  ..
+-----------------+           +---------+
| temperature_api |           | __end__ |
+-----------------+           +---------+

[User]
My name is Gabriel.

[Assistant]
Nice to meet you, Gabriel! How can I assist you today?

[User]
What is my name?

[Assistant]
Your name is Gabriel.

[User]
What is the weather in NYC?

[Tool Call: temperature_api]
{"location":"New York City"}

[Tool Result: temperature_api]
25

[Assistant]
The current temperature in New York City is 25°C. If you need more weather information, feel free to ask!
```
