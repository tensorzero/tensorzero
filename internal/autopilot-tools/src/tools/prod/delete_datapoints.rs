//! Tool for deleting datapoints from a dataset.

use std::borrow::Cow;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::DeleteDatapointsResponse;
use uuid::Uuid;

/// Parameters for the delete_datapoints tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DeleteDatapointsToolParams {
    /// The name of the dataset containing the datapoints.
    pub dataset_name: String,
    /// The IDs of the datapoints to delete.
    pub ids: Vec<Uuid>,
}

/// Tool for deleting datapoints from a dataset.
///
/// Performs a soft delete - datapoints are marked as stale but not truly removed.
#[derive(Default)]
pub struct DeleteDatapointsTool;

impl ToolMetadata for DeleteDatapointsTool {
    type SideInfo = AutopilotSideInfo;
    type Output = DeleteDatapointsResponse;
    type LlmParams = DeleteDatapointsToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("delete_datapoints")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Delete datapoints from a dataset by their IDs. \
             This is a soft delete - datapoints are marked as stale but not truly removed.",
        )
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Delete datapoints from a dataset by their IDs.",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "The name of the dataset containing the datapoints."
                },
                "ids": {
                    "type": "array",
                    "items": { "type": "string", "format": "uuid" },
                    "description": "The IDs of the datapoints to delete."
                }
            },
            "required": ["dataset_name", "ids"],
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
impl SimpleTool for DeleteDatapointsTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .delete_datapoints(llm_params.dataset_name, llm_params.ids)
            .await
            .map_err(|e| AutopilotToolError::client_error("delete_datapoints", e).into())
    }
}
