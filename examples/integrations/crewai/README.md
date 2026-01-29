# Example: Integrating Crew AI with TensorZero

## Overview

This example demonstrates how to integrate Crew AI with TensorZero to create a simple chatbot that can call external APIs.

## Getting Started

### TensorZero

We provide a blank TensorZero configuration in `example/tensorzero/tensorzero.toml`.
You can define custom functions, metrics, and more as needed.
For this example, we simply call the model `openai::gpt-4o-mini` with the default function.

### Prerequisites

1. Open the `example/` directory: `cd example/`
2. Install Python 3.10+ and the `crewai` CLI utility.
3. Install the Python dependencies. We recommend using [`uv`](https://github.com/astral-sh/uv): `uv sync`
4. Generate an API key for OpenAI (`OPENAI_API_KEY`).

### Setup

1. Set the `OPENAI_API_KEY` environment variable.
2. Run the chatbot using `crewai run`.

## Sample Run

```
Running the Crew
╭────────────────────────────────────────────── Crew Execution Started ──────────────────────────────────────────────╮
│                                                                                                                    │
│  Crew Execution Started                                                                                            │
│  Name: crew                                                                                                        │
│  ID: 24a156f1-3021-4f35-b585-6a12a05b5fd1                                                                          │
│                                                                                                                    │
│                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

# Agent: Weather Analyst
## Task: Report the weather for the current day based on the location provided.



# Agent: Weather Analyst
## Thought: I need to get the current temperature for New York City to report the weather for today.
## Using tool: Temperature Tool
## Tool Input:
"{\"location\": \"New York City\"}"
## Tool Output:
{"location": "New York City", "temperature": 70, "unit": "F"}


# Agent: Weather Analyst
## Final Answer:
The current weather in New York City is 70°F.


🚀 Crew: crew
└── 📋 Task: 727ab50d-b22b-4c83-9459-fe8bc2794e84
    Assigned to: Weather Analyst

    Status: ✅ Completed
    └── 🔧 Used Temperature Tool (1)
╭───────────────────────────────────────────────── Task Completion ──────────────────────────────────────────────────╮
│                                                                                                                    │
│  Task Completed                                                                                                    │
│  Name: 727ab50d-b22b-4c83-9459-fe8bc2794e84                                                                        │
│  Agent: Weather Analyst                                                                                            │
│                                                                                                                    │
│                                                                                                                    │
│                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────────────────────────── Crew Completion ──────────────────────────────────────────────────╮
│                                                                                                                    │
│  Crew Execution Completed                                                                                          │
│  Name: crew                                                                                                        │
│  ID: 24a156f1-3021-4f35-b585-6a12a05b5fd1                                                                          │
│                                                                                                                    │
│                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
