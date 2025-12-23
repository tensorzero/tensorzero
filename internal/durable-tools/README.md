# durable-tools

A Rust library for defining and executing tools in a durable execution environment, backed by the [`durable`](https://github.com/tensorzero/durable) crate.

## Overview

This crate provides abstractions for building AI agent tools with durable execution guarantees. It supports two types of tools:

- **`TaskTool`**: Durable tools that run as full durable tasks. They can call other tools, checkpoint progress, and spawn subtasks.
- **`SimpleTool`**: Lightweight tools that run inside a `TaskTool`'s `step()` checkpoint. Simpler to implement but cannot call other tools.

## Public Interface

### Core Traits

All tools implement the `ToolMetadata` trait for metadata, plus either `TaskTool` or `SimpleTool` for execution.

#### `ToolMetadata`

Provides the tool's name, description, parameter schema, and LLM parameter type:

```rust
pub trait ToolMetadata: Send + Sync + 'static {
    fn name() -> Cow<'static, str>;
    fn description() -> Cow<'static, str>;
    fn parameters_schema() -> ToolResult<Schema>;

    type LlmParams: Serialize + DeserializeOwned + Send + 'static;
}
```

#### `TaskTool`

For complex, durable operations that may need to call other tools or checkpoint progress:

```rust
#[async_trait]
pub trait TaskTool: ToolMetadata {
    type SideInfo: SideInfo;
    type Output: Serialize + DeserializeOwned + Send + 'static;

    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output>;
}
```

#### `SimpleTool`

For simple, stateless operations like API calls or database queries:

```rust
#[async_trait]
pub trait SimpleTool: ToolMetadata {
    type SideInfo: SideInfo;
    type Output: Serialize + DeserializeOwned + Send + 'static;

    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: Self::SideInfo,
        ctx: SimpleToolContext<'_>,
        idempotency_key: &str,
    ) -> ToolResult<Self::Output>;
}
```

### Main Types

| Type                       | Description                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `ToolExecutor`             | High-level orchestrator for registering and spawning tools                               |
| `ToolExecutorBuilder`      | Builder for configuring `ToolExecutor`                                                   |
| `ToolContext`              | Context passed to `TaskTool::execute()` with checkpointing and tool-calling capabilities |
| `SimpleToolContext`        | Simplified context passed to `SimpleTool::execute()` with database and inference access  |
| `ToolRegistry`             | Registry of tools for lookup and OpenAI function schema generation                       |
| `ToolAppState`             | Application state passed to all tools (pool, registry, inference client)                 |
| `DurableClient`            | Type alias for `Durable<ToolAppState>`                                                   |
| `ToolError` / `ToolResult` | Error types for tool execution                                                           |
| `SideInfo`                 | Marker trait for side information types (hidden from LLM)                                |
| `InferenceClient`          | Trait for TensorZero inference backends                                                  |
| `InferenceError`           | Error type for inference operations                                                      |

### Context Management

Tools can be wrapped with context management strategies to handle large outputs that might overwhelm LLM context windows. The `ctx` parameter namespace is reserved for context management parameters (filtering, pagination, etc.) and should not be used by inner tools.

### Re-exports

The crate re-exports commonly needed types:

- `async_trait` - For implementing tool traits
- `schemars` - For parameter schema generation
- `SpawnOptions`, `SpawnResult`, `TaskHandle`, `WorkerOptions` - From `durable`
- `http_gateway_client`, `embedded_gateway_client` - Inference client constructors
- TensorZero types: `ClientInferenceParams`, `InferenceParams`, `InferenceResponse`, `Input`, `InputMessage`, `InputMessageContent`, `Role`, `TensorZeroError`

## Usage Example

```rust
use durable_tools::{
    SimpleTool, TaskTool, ToolContext, SimpleToolContext, ToolMetadata,
    ToolExecutor, ToolResult, async_trait, WorkerOptions,
    http_gateway_client,
};
use schemars::{schema_for, JsonSchema, Schema};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use uuid::Uuid;

// Define a SimpleTool
#[derive(Serialize, Deserialize, JsonSchema)]
struct SearchParams { query: String }

#[derive(Serialize, Deserialize)]
struct SearchResult { results: Vec<String> }

#[derive(Default)]
struct SearchTool;

impl ToolMetadata for SearchTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("search")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Search the web")
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(SearchParams))
    }

    type LlmParams = SearchParams;
}

#[async_trait]
impl SimpleTool for SearchTool {
    type SideInfo = ();
    type Output = SearchResult;

    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        // Implementation...
        Ok(SearchResult { results: vec![] })
    }
}

// Define a TaskTool that calls the SimpleTool
#[derive(Serialize, Deserialize, JsonSchema)]
struct ResearchParams { topic: String }

#[derive(Serialize, Deserialize)]
struct ResearchResult { summary: String }

struct ResearchTool;

impl ToolMetadata for ResearchTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("research")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Research a topic")
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(ResearchParams))
    }

    type LlmParams = ResearchParams;
}

#[async_trait]
impl TaskTool for ResearchTool {
    type SideInfo = ();
    type Output = ResearchResult;

    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        // Call another tool
        let _search = ctx
            .call_tool("search", serde_json::json!({"query": llm_params.topic}))
            .await?;

        // Use a checkpointed step
        let summary = ctx
            .step("summarize", (), |(), _state| async {
                Ok("Summary of results".to_string())
            })
            .await?;

        Ok(ResearchResult { summary })
    }
}

// Setup and run
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let inference_client = http_gateway_client(url::Url::parse("http://localhost:3000")?)?;
    let executor = ToolExecutor::builder()
        .database_url(std::env::var("DATABASE_URL")?.into())
        .queue_name("tools")
        .inference_client(inference_client)
        .build()
        .await?;

    // Create the queue (required before spawning)
    executor.durable().create_queue(None).await?;

    // Register tools
    executor.register_simple_tool::<SearchTool>().await;
    executor.register_task_tool::<ResearchTool>().await;

    // Spawn a tool execution
    let episode_id = Uuid::now_v7();
    executor.spawn_tool::<ResearchTool>(
        ResearchParams { topic: "rust".into() },
        (),
        episode_id,
    ).await?;

    // Start a worker to process tasks
    let worker = executor.start_worker(WorkerOptions::default()).await;
    // ... worker processes tasks until shutdown
    worker.shutdown().await;

    Ok(())
}
```

## Running Tests

### Unit Tests (no database required)

```bash
cargo test --lib
```

### Integration Tests (requires Postgres)

The integration tests use `#[sqlx::test]` with the durable migrator to automatically set up the database schema.

```bash
DATABASE_URL="postgres://postgres:postgres@localhost:5433/test" cargo test --test integration
```

### All Tests

```bash
DATABASE_URL="postgres://postgres:postgres@localhost:5433/test" cargo test
```

## Test Coverage

**Unit tests** (34 tests in `src/tests.rs`):

- Registry: registration, lookup, listing, `is_durable()`, `to_tensorzero_tools()`
- Type erasure wrappers: metadata exposure, timeout defaults
- Builder: default values, method chaining
- Error conversions: `ToolError` <-> `TaskError`

**Integration tests** (5 tests in `tests/integration.rs`):

- `execute_erased` serialization/deserialization
- Tool registration and listing
- Task spawning via typed API and by name
