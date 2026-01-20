//! Tool for listing datapoints in a dataset.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

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

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("list_datapoints")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "List datapoints in a dataset with optional filtering and pagination. \
             Can filter by function name, tags, time ranges, and order results.",
        )
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
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
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of datapoints to return (default: 20)."
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of datapoints to skip (for pagination)."
                },
                "filter": {
                    "description": "Optional filter to apply when querying datapoints. Supports filtering by tags, time, and logical combinations (AND/OR/NOT).",
                    "oneOf": [
                        {
                            "type": "object",
                            "description": "Filter by tag key-value pair.",
                            "properties": {
                                "type": { "const": "tag" },
                                "key": { "type": "string", "description": "Tag key." },
                                "value": { "type": "string", "description": "Tag value." },
                                "comparison_operator": {
                                    "type": "string",
                                    "enum": ["=", "!="],
                                    "description": "Comparison operator."
                                }
                            },
                            "required": ["type", "key", "value", "comparison_operator"]
                        },
                        {
                            "type": "object",
                            "description": "Filter by timestamp.",
                            "properties": {
                                "type": { "const": "time" },
                                "time": { "type": "string", "format": "date-time", "description": "Timestamp to compare against (ISO 8601 format)." },
                                "comparison_operator": {
                                    "type": "string",
                                    "enum": ["<", "<=", "=", ">", ">=", "!="],
                                    "description": "Comparison operator."
                                }
                            },
                            "required": ["type", "time", "comparison_operator"]
                        },
                        {
                            "type": "object",
                            "description": "Logical AND of multiple filters.",
                            "properties": {
                                "type": { "const": "and" },
                                "children": { "type": "array", "description": "Array of filters to AND together.", "items": { "type": "object" } }
                            },
                            "required": ["type", "children"]
                        },
                        {
                            "type": "object",
                            "description": "Logical OR of multiple filters.",
                            "properties": {
                                "type": { "const": "or" },
                                "children": { "type": "array", "description": "Array of filters to OR together.", "items": { "type": "object" } }
                            },
                            "required": ["type", "children"]
                        },
                        {
                            "type": "object",
                            "description": "Logical NOT of a filter.",
                            "properties": {
                                "type": { "const": "not" },
                                "child": { "type": "object", "description": "Filter to negate." }
                            },
                            "required": ["type", "child"]
                        }
                    ]
                },
                "order_by": {
                    "type": "array",
                    "description": "Optional ordering criteria for the results.",
                    "items": {
                        "type": "object",
                        "description": "A single ordering criterion.",
                        "properties": {
                            "by": {
                                "type": "string",
                                "enum": ["timestamp", "search_relevance"],
                                "description": "The property to order by. 'timestamp' orders by creation time, 'search_relevance' orders by search query relevance (requires search_query_experimental)."
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["ascending", "descending"],
                                "description": "The ordering direction."
                            }
                        },
                        "required": ["by", "direction"]
                    }
                },
                "search_query_experimental": {
                    "type": "string",
                    "description": "EXPERIMENTAL: Text query for case-insensitive substring search over input and output. Requires exact substring match. May be slow without other filters."
                }
            },
            "required": ["dataset_name"]
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
impl SimpleTool for ListDatapointsTool {
    async fn execute(
        &self,
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
