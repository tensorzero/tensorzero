//! Tool for creating datapoints in a dataset.

use std::borrow::Cow;
use std::collections::HashMap;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use tensorzero::{CreateDatapointRequest, CreateDatapointsResponse};

use crate::types::AutopilotToolSideInfo;

/// Parameters for the create_datapoints tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CreateDatapointsToolParams {
    /// The name of the dataset to create datapoints in.
    pub dataset_name: String,
    /// The datapoints to create. Can be Chat or Json type.
    pub datapoints: Vec<CreateDatapointRequest>,
}

/// Tool for creating datapoints in a dataset.
///
/// This tool creates new datapoints and automatically tags them with
/// autopilot session metadata for tracking.
#[derive(Default)]
pub struct CreateDatapointsTool;

impl ToolMetadata for CreateDatapointsTool {
    type SideInfo = AutopilotToolSideInfo;
    type Output = CreateDatapointsResponse;
    type LlmParams = CreateDatapointsToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("create_datapoints")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Create datapoints in a dataset. Datapoints can be Chat or Json type. \
             Autopilot tags are automatically added for tracking.",
        )
    }

    fn parameters_schema() -> Schema {
        schema_for!(CreateDatapointsToolParams)
    }
}

/// Build autopilot tags from side info.
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

/// Merge autopilot tags into existing tags, preserving user-provided tags.
fn merge_tags(
    existing: Option<HashMap<String, String>>,
    autopilot_tags: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut merged = autopilot_tags.clone();
    if let Some(existing) = existing {
        // User-provided tags take precedence over autopilot tags
        merged.extend(existing);
    }
    merged
}

/// Add autopilot tags to a datapoint request.
fn add_tags_to_datapoint(
    datapoint: CreateDatapointRequest,
    autopilot_tags: &HashMap<String, String>,
) -> CreateDatapointRequest {
    match datapoint {
        CreateDatapointRequest::Chat(mut chat) => {
            chat.tags = Some(merge_tags(chat.tags, autopilot_tags));
            CreateDatapointRequest::Chat(chat)
        }
        CreateDatapointRequest::Json(mut json) => {
            json.tags = Some(merge_tags(json.tags, autopilot_tags));
            CreateDatapointRequest::Json(json)
        }
    }
}

#[async_trait]
impl SimpleTool for CreateDatapointsTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let autopilot_tags = build_autopilot_tags(&side_info);

        // Add autopilot tags to each datapoint
        let datapoints: Vec<CreateDatapointRequest> = llm_params
            .datapoints
            .into_iter()
            .map(|dp| add_tags_to_datapoint(dp, &autopilot_tags))
            .collect();

        ctx.client()
            .create_datapoints(llm_params.dataset_name, datapoints)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
