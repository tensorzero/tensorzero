//! Trait for abstracting durable tool context operations.
//!
//! This trait allows consumers to work with a type-erased `ToolContext<S>`
//! without depending on the concrete generic parameter `S`.

use async_trait::async_trait;
use durable::SpawnOptions;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::ToolHandle;
use crate::tool_error::ToolResult;

/// Trait abstracting durable tool context operations.
///
/// This allows consumers (such as `ts-executor-pool`) to be generic-free
/// so that they can be downcast in deno ops. Callers provide an implementation
/// that wraps their concrete `ToolContext<S>`.
///
/// Note that `inference` is deliberately not part of this trait — that method
/// requires the full `ClientInferenceParams` type which lives in `tensorzero-core`
/// and would pull a heavy dependency into consumers that don't need it.
/// Concrete `ToolContext` implementors expose `inference` as an inherent method.
#[async_trait]
pub trait ToolContextHelper: Send + Sync {
    fn episode_id(&self) -> Uuid;
    async fn join_tool(&mut self, handle: ToolHandle) -> ToolResult<JsonValue>;
    async fn uuid7(&mut self) -> ToolResult<Uuid>;
    async fn spawn_tool(
        &mut self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        options: SpawnOptions,
    ) -> ToolResult<ToolHandle>;
    async fn heartbeat(&mut self) -> ToolResult<()>;
}
