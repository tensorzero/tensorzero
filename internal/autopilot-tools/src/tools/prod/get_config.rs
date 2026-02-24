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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetConfigToolParams {}

/// Tool for retrieving config snapshots.
#[derive(Default)]
pub struct GetConfigTool;

impl ToolMetadata for GetConfigTool {
    type SideInfo = AutopilotSideInfo;
    type Output = GetConfigResponse;
    type LlmParams = GetConfigToolParams;

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GET_CONFIG_TOOL_PARAMS
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GET_CONFIG_RESPONSE
    }

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
