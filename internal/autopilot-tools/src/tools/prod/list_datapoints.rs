//! Tool for listing datapoints in a dataset.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{InnerToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::{GetDatapointsResponse, ListDatapointsRequest};

use autopilot_client::AutopilotSideInfo;

/// Parameters for the list_datapoints tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ListDatapointsToolParams {
    /// The name of the dataset to list datapoints from.
    pub dataset_name: String,
    /// Request parameters for listing datapoints (filtering, pagination, ordering).
    #[serde(flatten)]
    pub request: ListDatapointsRequest,
}

/// Tool for listing datapoints in a dataset.
///
/// Supports filtering by function name, tags, time ranges, and pagination.
#[derive(Default)]
pub struct ListDatapointsTool;

impl ToolMetadata for ListDatapointsTool {
    type SideInfo = AutopilotSideInfo;
    type Output = GetDatapointsResponse;
    type LlmParams = ListDatapointsToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("list_datapoints")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "List datapoints in a dataset with optional filtering and pagination. \
             Can filter by function name, tags, time ranges, and order results.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "List datapoints in a dataset with filtering and pagination.",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "The name of the dataset to list datapoints from."
                },
                "function_name": {
                    "type": "string",
                    "description": "Filter by function name (optional)."
                },
                "tags": {
                    "type": "object",
                    "additionalProperties": { "type": "string" },
                    "description": "Filter by tags (optional)."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of datapoints to return (default: 100)."
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of datapoints to skip (for pagination)."
                },
                "order_by": {
                    "type": "string",
                    "enum": ["created_at_asc", "created_at_desc"],
                    "description": "Sort order (default: created_at_desc)."
                }
            },
            "required": ["dataset_name"]
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
impl SimpleTool for ListDatapointsTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .list_datapoints(llm_params.dataset_name, llm_params.request)
            .await
            .map_err(|e| AutopilotToolError::client_error("list_datapoints", e).into())
    }
}
