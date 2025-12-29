#![recursion_limit = "512"]
//! Durable tool execution framework.
//!
//! This crate provides abstractions for defining and executing tools in a durable
//! execution environment backed by the `durable` crate.
//!
//! # Lightweight Spawning
//!
//! For consumers who only need to spawn tasks without the full tool framework,
//! use the `durable-tools-spawn` crate directly or the re-exported types:
//!
//! ```ignore
//! use durable_tools::spawn::{SpawnClient, TaskToolParams};
//! ```
//!
//! # Tool Types
//!
//! - **`TaskTool`**: Durable tools that run as full durable tasks. Can call other
//!   tools, checkpoint progress, and spawn subtasks.
//!
//! - **`SimpleTool`**: Lightweight tools that run inside a `TaskTool`'s `step()`.
//!   Cannot call other tools but have access to the database pool.
//!
//! # Side Information
//!
//! Both tool types support "side information" - parameters provided at spawn time
//! but hidden from the LLM. Set `type SideInfo = ()` for tools that don't need it.
//!
//! # Example
//!
//! ```ignore
//! use durable_tools::{
//!     TaskTool, SimpleTool, ToolContext, SimpleToolContext, ToolMetadata,
//!     ToolExecutor, ToolResult, async_trait, WorkerOptions,
//!     http_gateway_client,
//! };
//! use schemars::{schema_for, JsonSchema, Schema};
//! use serde::{Deserialize, Serialize};
//! use std::borrow::Cow;
//! use uuid::Uuid;
//!
//! // Define a SimpleTool
//! #[derive(Serialize, Deserialize, JsonSchema)]
//! struct SearchParams {
//!     query: String,
//! }
//!
//! #[derive(Serialize, Deserialize)]
//! struct SearchResult {
//!     results: Vec<String>,
//! }
//!
//! #[derive(Default)]
//! struct SearchTool;
//!
//! impl ToolMetadata for SearchTool {
//!     fn name() -> Cow<'static, str> {
//!         Cow::Borrowed("search")
//!     }
//!
//!     fn description() -> Cow<'static, str> {
//!         Cow::Borrowed("Search the web")
//!     }
//!
//!     fn parameters_schema() -> ToolResult<Schema> {
//!         Ok(schema_for!(SearchParams))
//!     }
//!
//!     type LlmParams = SearchParams;
//!     type SideInfo = ();
//!     type Output = SearchResult;
//! }
//!
//! #[async_trait]
//! impl SimpleTool for SearchTool {
//!     async fn execute(
//!         llm_params: <Self as ToolMetadata>::LlmParams,
//!         _side_info: <Self as ToolMetadata>::SideInfo,
//!         ctx: SimpleToolContext<'_>,
//!         idempotency_key: &str,
//!     ) -> ToolResult<<Self as ToolMetadata>::Output> {
//!         // Implementation...
//!         Ok(SearchResult { results: vec![] })
//!     }
//! }
//!
//! // Define a TaskTool that uses the SimpleTool
//! #[derive(Serialize, Deserialize, JsonSchema)]
//! struct ResearchParams {
//!     topic: String,
//! }
//!
//! #[derive(Serialize, Deserialize)]
//! struct ResearchResult {
//!     summary: String,
//! }
//!
//! struct ResearchTool;
//!
//! impl ToolMetadata for ResearchTool {
//!     fn name() -> Cow<'static, str> {
//!         Cow::Borrowed("research")
//!     }
//!
//!     fn description() -> Cow<'static, str> {
//!         Cow::Borrowed("Research a topic")
//!     }
//!
//!     fn parameters_schema() -> ToolResult<Schema> {
//!         Ok(schema_for!(ResearchParams))
//!     }
//!
//!     type LlmParams = ResearchParams;
//!     type SideInfo = ();
//!     type Output = ResearchResult;
//! }
//!
//! #[async_trait]
//! impl TaskTool for ResearchTool {
//!     async fn execute(
//!         llm_params: <Self as ToolMetadata>::LlmParams,
//!         _side_info: <Self as ToolMetadata>::SideInfo,
//!         ctx: &mut ToolContext<'_>,
//!     ) -> ToolResult<<Self as ToolMetadata>::Output> {
//!         // Call the search tool
//!         let _search = ctx
//!             .call_tool("search", serde_json::json!({"query": llm_params.topic}))
//!             .await?;
//!
//!         // Use a checkpointed step
//!         let summary = ctx
//!             .step("summarize", (), |(), _state| async {
//!                 Ok("Summary of search results".to_string())
//!             })
//!             .await?;
//!
//!         Ok(ResearchResult { summary })
//!     }
//! }
//!
//! // Setup and run
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create inference client for TensorZero
//!     let inference_client = http_gateway_client(
//!         url::Url::parse("http://localhost:3000")?
//!     )?;
//!
//!     let executor = ToolExecutor::builder()
//!         .database_url(std::env::var("DATABASE_URL")?.into())
//!         .queue_name("tools")
//!         .inference_client(inference_client)
//!         .build()
//!         .await?;
//!
//!     executor.register_simple_tool::<SearchTool>().await;
//!     executor.register_task_tool::<ResearchTool>().await;
//!
//!     let episode_id = Uuid::now_v7();
//!     executor.spawn_tool::<ResearchTool>(
//!         ResearchParams { topic: "rust".into() },
//!         (),  // No side info
//!         episode_id,
//!     ).await?;
//!
//!     let worker = executor.start_worker(WorkerOptions::default()).await;
//!     // ... worker processes tasks until shutdown
//!     worker.shutdown().await;
//!     Ok(())
//! }
//! ```

mod context;
mod error;
mod executor;
pub mod inference;
mod registry;
mod simple_tool;
mod task_tool;
mod tool_metadata;

#[cfg(test)]
mod tests;

// Re-export spawn crate types for lightweight spawning
pub mod spawn {
    //! Lightweight spawning types re-exported from `durable-tools-spawn`.
    pub use durable_tools_spawn::{
        SpawnClient, SpawnClientBuilder, SpawnError, SpawnResult, TaskToolParams,
    };
}

// Re-export main types
pub use context::{DurableClient, SimpleToolContext, ToolAppState, ToolContext};
pub use durable_tools_spawn::TaskToolParams;
pub use error::{ToolError, ToolResult};
pub use executor::{ToolExecutor, ToolExecutorBuilder};
pub use registry::{ErasedSimpleTool, ErasedTaskToolWrapper, ErasedTool, ToolRegistry};
pub use simple_tool::SimpleTool;
pub use task_tool::{TaskTool, TaskToolAdapter};
pub use tool_metadata::{SideInfo, ToolMetadata};

// Re-export inference trait and helpers
pub use inference::{
    EmbeddedInferenceClient, InferenceClient, InferenceError, embedded_gateway_client, from_client,
    http_gateway_client,
};

// Re-export autopilot types for use by tools
pub use inference::{
    CreateEventRequest, CreateEventResponse, EventPayload, ListEventsParams, ListEventsResponse,
    ListSessionsParams, ListSessionsResponse, ToolOutcome,
};

// Re-export TensorZero inference types for convenience
pub use tensorzero::{
    Client, ClientInferenceParams, DynamicToolParams, InferenceParams, InferenceResponse, Input,
    InputMessage, InputMessageContent, Role, TensorZeroError, Tool,
};

// Re-export async_trait for convenience
pub use async_trait::async_trait;

// Re-export durable types that tools may need
pub use durable::{SpawnOptions, SpawnResult, TaskHandle, Worker, WorkerOptions};

// Re-export schemars for parameter schemas
pub use schemars;
