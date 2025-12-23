# Example: TensorZero + MCP (Model Context Protocol)

This example shows how to use an MCP (Model Context Protocol) server with TensorZero.

We'll use [`mcp-clickhouse`](https://github.com/ClickHouse/mcp-clickhouse) to build a chatbot that can answer questions about the contents of your ClickHouse database.

## Example

```
[User]
Inspect the schemas and tell me how many inferences I have?

[Tool Call: list_tables]
{"database":"tensorzero"}

[Tool Result]
... redacted for brevity ...

[Tool Call: run_select_query]
{"query":"SELECT count(DISTINCT inference_id) AS total_inferences FROM tensorzero.ModelInference"}
... redacted for brevity ...

[Tool Result]
{"total_inferences": 90}

[Assistant]
You have a total of 90 inferences recorded in the tensorzero.ModelInference table. Let me know if you need inference counts from other related tables or more details.
```

> [!WARNING]
>
> This example is for educational purposes only.
> The agent is likely to hallucinate and make mistakes without additional context and optimization.

## Getting Started

### TensorZero

We provide a simple configuration in `config/tensorzero.toml`.
The configuration specifies a straightforward chat function `clickhouse_copilot` with a single variant that uses GPT 4.1 Mini.

### MCP Server

We provide a sample configuration for the MCP server in `config/mcp-clickhouse.toml`.

### Prerequisites

1. Install Docker.
2. Install Python 3.10+.
3. Generate an OpenAI API key.

### Setup

1. Set the `OPENAI_API_KEY` environment variable to your OpenAI API key.
2. Run `docker compose up` to start TensorZero.
3. Install the Python dependencies: `pip install -r requirements.txt`
4. Run the script: `python main.py`
