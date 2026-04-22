//! Trait for abstracting durable tool context operations.
//!
//! This trait allows consumers to work with a type-erased `ToolContext<S>`
//! without depending on the concrete generic parameter `S`.

use async_trait::async_trait;
use durable::SpawnOptions;
use serde_json::Value as JsonValue;
use tensorzero_types::ToolError;
use tensorzero_types::ToolHandle;
use tensorzero_types::tool_error::ToolResult;
use tensorzero_types::tool_failure::NonControlToolError;
use uuid::Uuid;

use super::ClientInferenceParams;
use crate::endpoints::inference::InferenceResponse;

/// Trait abstracting durable tool context operations.
///
/// This allows `ExposedTools` (in `ts-executor-pool`) to be generic-free
/// so that it can be downcast in deno ops. Callers provide an implementation
/// that wraps their concrete `ToolContext<S>`.
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
    async fn inference(&mut self, params: ClientInferenceParams) -> ToolResult<InferenceResponse>;
    async fn heartbeat(&mut self) -> ToolResult<()>;
}

/// Run a checkpointed inference call via [`ToolContextHelper`].
///
/// Validates that `episode_id` is `None` (it will be set from the context),
/// then delegates to [`ToolContextHelper::inference`].
pub async fn checkpointed_inference(
    ctx: &mut dyn ToolContextHelper,
    mut params: ClientInferenceParams,
) -> ToolResult<InferenceResponse> {
    if params.episode_id.is_some() {
        return Err(ToolError::NonControl(NonControlToolError::Internal {
            message: "episode_id must be None when using checkpointed_inference".to_string(),
        }));
    }
    params.episode_id = Some(ctx.episode_id());
    ctx.inference(params).await
}
