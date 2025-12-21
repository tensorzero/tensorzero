#![recursion_limit = "512"]
//! Durable tool execution framework.
//!
//! This crate provides abstractions for defining and executing tools in a durable
//! execution environment backed by the `durable` crate.
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
//!     TaskTool, SimpleTool, ToolContext, SimpleToolContext,
//!     ToolExecutor, ToolResult, async_trait, WorkerOptions,
//!     http_gateway_client,
//! };
//! use schemars::{schema_for, schema::RootSchema, JsonSchema};
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
//! #[async_trait]
//! impl SimpleTool for SearchTool {
//!     const NAME: &'static str = "search";
//!
//!     fn description() -> Cow<'static, str> {
//!         Cow::Borrowed("Search the web")
//!     }
//!
//!     fn parameters_schema() -> RootSchema {
//!         schema_for!(SearchParams)
//!     }
//!
//!     type LlmParams = SearchParams;
//!     type SideInfo = ();
//!     type Output = SearchResult;
//!
//!     async fn execute(
//!         llm_params: Self::LlmParams,
//!         _side_info: Self::SideInfo,
//!         ctx: SimpleToolContext<'_>,
//!         idempotency_key: &str,
//!     ) -> ToolResult<Self::Output> {
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
//! #[async_trait]
//! impl TaskTool for ResearchTool {
//!     const NAME: &'static str = "research";
//!
//!     fn description() -> Cow<'static, str> {
//!         Cow::Borrowed("Research a topic")
//!     }
//!
//!     fn parameters_schema() -> RootSchema {
//!         schema_for!(ResearchParams)
//!     }
//!
//!     type LlmParams = ResearchParams;
//!     type SideInfo = ();
//!     type Output = ResearchResult;
//!
//!     async fn execute(
//!         llm_params: Self::LlmParams,
//!         _side_info: Self::SideInfo,
//!         ctx: &mut ToolContext<'_>,
//!     ) -> ToolResult<Self::Output> {
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

#[cfg(test)]
mod tests;

// Re-export main types
pub use context::{DurableClient, SimpleToolContext, StateExtension, ToolAppState, ToolContext};
pub use error::{ToolError, ToolResult};
pub use executor::{ToolExecutor, ToolExecutorBuilder};
pub use registry::{ErasedSimpleTool, ErasedTaskToolWrapper, ErasedTool, ToolRegistry};
pub use simple_tool::SimpleTool;
pub use task_tool::{SideInfo, TaskTool, TaskToolAdapter, TaskToolParams};

// Re-export inference trait and helpers
pub use inference::{
    InferenceClient, InferenceError, embedded_gateway_client, http_gateway_client,
};

// Re-export TensorZero inference types for convenience
pub use tensorzero::{
    ClientInferenceParams, InferenceParams, InferenceResponse, Input, InputMessage,
    InputMessageContent, Role, TensorZeroError,
};

// Re-export async_trait for convenience
pub use async_trait::async_trait;

// Re-export durable types that tools may need
pub use durable::{SpawnOptions, SpawnResult, TaskHandle, WorkerOptions};

// Re-export schemars for parameter schemas
pub use schemars;
