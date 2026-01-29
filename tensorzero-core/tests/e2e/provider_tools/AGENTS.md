# Provider Tools E2E Tests

This directory contains E2E tests for provider tools (e.g., `web_search`, `bash`) across different providers.

## Tool Types

- **Server-side tools** (e.g., `web_search`): The provider executes the tool and returns results inline. Response includes `Unknown` content blocks with tool results.
- **Client-side tools** (e.g., `bash`): The provider returns a `ToolCall` for the client to execute. Response includes `ToolCall` content blocks.

## Test Coverage Checklist

When adding provider tools support for a new provider, implement tests for at least one tool with full coverage:

### Required: Full test suite for primary tool (e.g., `web_search`)

**Static Provider Tools (configured in model config):**

- `test_{provider}_provider_tools_{tool}_nonstreaming` - Multi-turn conversation
- `test_{provider}_provider_tools_{tool}_streaming` - Multi-turn conversation with ClickHouse persistence check

**Dynamic Provider Tools (passed via request):**

- `test_{provider}_dynamic_provider_tools_{tool}_nonstreaming` - Include:
  - Scope filtering: Pass both `Unscoped` tool and wrongly-scoped "garbage" tool
  - Multi-turn round-trip test
- `test_{provider}_dynamic_provider_tools_{tool}_streaming` - Include scope filtering
- `test_{provider}_dynamic_provider_tools_positive_scope_matching` - Pass ONLY a correctly-scoped `ModelProvider` tool (no `Unscoped` fallback) to verify scope matching works

### Optional: Basic tests for additional tools (e.g., `bash`)

For other tools, a streaming/nonstreaming pair of static tests is sufficient:

- `test_{provider}_provider_tools_{tool}_nonstreaming`
- `test_{provider}_provider_tools_{tool}_streaming`

### Key Assertions

- Streaming tests: Capture `inference_id` from chunks, verify ClickHouse `raw_response` contains expected content (e.g., `server_tool_use` for Anthropic, `web_search_call` for OpenAI)
- Multi-turn: Convert assistant response to input format and make follow-up inference
- Scope filtering: Include `ProviderToolScope::ModelProvider` with `model_name: "garbage"` to verify filtering

### Naming Convention

```
test_{provider}_provider_tools_{tool}_{streaming|nonstreaming}
test_{provider}_dynamic_provider_tools_{tool}_{streaming|nonstreaming}
test_{provider}_dynamic_provider_tools_positive_scope_matching
```
