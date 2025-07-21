# Pure OpenAI Agents SDK - Simple Agentic RAG

This is a pure implementation of the simple agentic RAG example using only the OpenAI Agents SDK, without TensorZero integration. This serves as the target experience for our TensorZero + Agents SDK integration.

## Comparison with TensorZero Version

| Aspect | TensorZero Version | Pure Agents SDK | Target Integration |
|--------|-------------------|-----------------|-------------------|
| **Tool Definition** | JSON schemas in config | `@function_tool` decorators | Auto-converted from TensorZero config |
| **Agent Loop** | Manual tool execution | Built-in `Runner.run()` | Built-in `Runner.run()` |
| **System Prompt** | Template file | Direct string | Template automatically applied |
| **Observability** | Automatic ClickHouse logging | Manual/none | Automatic TensorZero logging |
| **Episode Management** | Automatic | Manual session handling | Automatic via TensorZero |
| **Configuration** | `tensorzero.toml` | Python code | `tensorzero.toml` (auto-detected) |

## Getting Started

### Prerequisites

1. Install the OpenAI Agents SDK: `pip install openai-agents`
2. Set your OpenAI API key: `export OPENAI_API_KEY=your_key_here`

### Usage

```python
import asyncio
from main import create_rag_agent

async def main():
    agent = create_rag_agent()
    result = await Runner.run(
        agent,
        "What is a common dish in the hometown of the scientist that won the Nobel Prize for the discovery of the positron?"
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

## Files

- `main.py` - Pure Agents SDK implementation
- `tools.py` - Wikipedia tools implemented as function_tools
- `pyproject.toml` - Dependencies
- `test_comparison.py` - Compare with TensorZero version

This implementation shows what the developer experience should be like with our TensorZero integration - simple, clean, and leveraging the best of both frameworks.
