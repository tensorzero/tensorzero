//! Tool for writing config snapshots.

use std::borrow::Cow;
use std::collections::HashMap;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero::{WriteConfigRequest, WriteConfigResponse};
use tensorzero_core::config::UninitializedConfig;

use crate::types::AutopilotToolSideInfo;

/// Parameters for the write_config tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct WriteConfigToolParams {
    /// The config to write as a JSON object.
    pub config: Value,
    /// Templates that should be stored with the config.
    #[serde(default)]
    pub extra_templates: HashMap<String, String>,
    /// User-defined tags for categorizing this config snapshot.
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

/// Tool for writing config snapshots.
#[derive(Default)]
pub struct WriteConfigTool;

impl ToolMetadata for WriteConfigTool {
    type SideInfo = AutopilotToolSideInfo;
    type Output = WriteConfigResponse;
    type LlmParams = WriteConfigToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("write_config")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Write a config snapshot to storage and return its hash. \
             Autopilot tags are automatically merged into the provided tags.",
        )
    }
}

fn build_autopilot_tags(side_info: &AutopilotToolSideInfo) -> HashMap<String, String> {
    let mut tags = HashMap::new();
    tags.insert(
        "autopilot_session_id".to_string(),
        side_info.session_id.to_string(),
    );
    tags.insert(
        "autopilot_tool_call_id".to_string(),
        side_info.tool_call_id.to_string(),
    );
    tags.insert(
        "autopilot_tool_call_event_id".to_string(),
        side_info.tool_call_event_id.to_string(),
    );
    tags
}

#[async_trait]
impl SimpleTool for WriteConfigTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let config: UninitializedConfig =
            serde_json::from_value(llm_params.config).map_err(|e| ToolError::Validation {
                message: format!("Invalid `config`: {e}"),
            })?;

        let mut tags = build_autopilot_tags(&side_info);
        tags.extend(llm_params.tags);

        let request = WriteConfigRequest {
            config,
            extra_templates: llm_params.extra_templates,
            tags,
        };

        ctx.client()
            .write_config(request)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
