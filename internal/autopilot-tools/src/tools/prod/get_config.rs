//! Tool for retrieving config snapshots.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero::GetConfigResponse;

use autopilot_client::AutopilotSideInfo;

/// Parameters for the get_config tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetConfigToolParams;

/// Tool for retrieving config snapshots.
#[derive(Default)]
pub struct GetConfigTool;

impl ToolMetadata for GetConfigTool {
    type SideInfo = AutopilotSideInfo;
    type Output = GetConfigResponse;
    type LlmParams = GetConfigToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("get_config")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Get the live config snapshot or a historical snapshot by hash.")
    }
}

#[async_trait]
impl SimpleTool for GetConfigTool {
    async fn execute(
        _llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .get_config_snapshot(side_info.config_snapshot_hash)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
