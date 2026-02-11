//! Tool for retrieving config snapshots.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::GetConfigResponse;

use autopilot_client::AutopilotSideInfo;

/// Parameters for the get_config tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetConfigToolParams {}

/// Tool for retrieving config snapshots.
#[derive(Default)]
pub struct GetConfigTool;

impl ToolMetadata for GetConfigTool {
    type SideInfo = AutopilotSideInfo;
    type Output = GetConfigResponse;
    type LlmParams = GetConfigToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("get_config")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Get the live config snapshot or a historical snapshot by hash.")
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Get the config snapshot. No parameters required.",
            "properties": {},
            "additionalProperties": false
        });

        serde_json::from_value(schema).map_err(|e| {
            NonControlToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }
}

#[async_trait]
impl SimpleTool for GetConfigTool {
    async fn execute(
        &self,
        _llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .get_config_snapshot(Some(side_info.config_snapshot_hash))
            .await
            .map_err(|e| AutopilotToolError::client_error("get_config", e).into())
    }
}
