//! Tool for updating existing datapoints.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{InnerToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::{UpdateDatapointRequest, UpdateDatapointsResponse};

use autopilot_client::AutopilotSideInfo;

/// Parameters for the update_datapoints tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct UpdateDatapointsToolParams {
    /// The name of the dataset containing the datapoints.
    pub dataset_name: String,
    /// The datapoints to update. Can be Chat or Json type.
    pub datapoints: Vec<UpdateDatapointRequest>,
}

/// Tool for updating existing datapoints.
///
/// Can update input, output, tags, and metadata fields.
/// Updates create new versions of datapoints (immutable history).
#[derive(Default)]
pub struct UpdateDatapointsTool;

impl ToolMetadata for UpdateDatapointsTool {
    type SideInfo = AutopilotSideInfo;
    type Output = UpdateDatapointsResponse;
    type LlmParams = UpdateDatapointsToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("update_datapoints")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Update existing datapoints in a dataset. \
             Can modify input, output, tags, and metadata. \
             Returns new IDs for the updated datapoints (versions are immutable).",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Update existing datapoints in a dataset.",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "The name of the dataset containing the datapoints."
                },
                "datapoints": {
                    "type": "array",
                    "description": "The datapoints to update.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "format": "uuid",
                                "description": "The ID of the datapoint to update."
                            },
                            "input": {
                                "type": "object",
                                "description": "New input data (optional)."
                            },
                            "output": {
                                "description": "New output data (optional)."
                            },
                            "tags": {
                                "type": "object",
                                "additionalProperties": { "type": "string" },
                                "description": "New tags (optional)."
                            }
                        },
                        "required": ["id"]
                    }
                }
            },
            "required": ["dataset_name", "datapoints"]
        });

        serde_json::from_value(schema).map_err(|e| {
            InnerToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }
}

#[async_trait]
impl SimpleTool for UpdateDatapointsTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .update_datapoints(llm_params.dataset_name, llm_params.datapoints)
            .await
            .map_err(|e| AutopilotToolError::client_error("update_datapoints", e).into())
    }
}
