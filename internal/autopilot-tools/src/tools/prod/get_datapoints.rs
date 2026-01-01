//! Tool for getting specific datapoints by ID.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::GetDatapointsResponse;
use uuid::Uuid;

use crate::types::AutopilotToolSideInfo;

/// Parameters for the get_datapoints tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetDatapointsToolParams {
    /// The name of the dataset (optional, but recommended for performance).
    #[serde(default)]
    pub dataset_name: Option<String>,
    /// The IDs of the datapoints to retrieve.
    pub ids: Vec<Uuid>,
}

/// Tool for getting specific datapoints by their IDs.
///
/// Providing the dataset name improves query performance.
#[derive(Default)]
pub struct GetDatapointsTool;

impl ToolMetadata for GetDatapointsTool {
    type SideInfo = AutopilotToolSideInfo;
    type Output = GetDatapointsResponse;
    type LlmParams = GetDatapointsToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("get_datapoints")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Get specific datapoints by their IDs. \
             Optionally provide dataset_name for better query performance.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Get specific datapoints by their IDs.",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "The name of the dataset (optional, but recommended for performance)."
                },
                "ids": {
                    "type": "array",
                    "items": { "type": "string", "format": "uuid" },
                    "description": "The IDs of the datapoints to retrieve."
                }
            },
            "required": ["ids"]
        });

        serde_json::from_value(schema).map_err(|e| ToolError::SchemaGeneration(e.into()))
    }
}

#[async_trait]
impl SimpleTool for GetDatapointsTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .get_datapoints(llm_params.dataset_name, llm_params.ids)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
