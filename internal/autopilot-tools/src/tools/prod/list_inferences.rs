//! Tool for listing inferences with filtering and pagination.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::Schema;
use serde::{Deserialize, Serialize};
use tensorzero::{GetInferencesResponse, ListInferencesRequest};

use autopilot_client::AutopilotSideInfo;

/// Parameters for the list_inferences tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ListInferencesToolParams {
    /// Request parameters for listing inferences (filtering, pagination, ordering).
    #[serde(flatten)]
    pub request: ListInferencesRequest,
}

/// Tool for listing inferences with filtering and pagination.
///
/// Supports filtering by function name, variant name, episode ID, tags, metrics,
/// time ranges, and pagination with offset or cursor-based navigation.
#[derive(Default)]
pub struct ListInferencesTool;

impl ToolMetadata for ListInferencesTool {
    type SideInfo = AutopilotSideInfo;
    type Output = GetInferencesResponse;
    type LlmParams = ListInferencesToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("list_inferences")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "List inferences with optional filtering and pagination. \
             Can filter by function name, variant name, episode ID, tags, metrics, \
             time ranges, and order results. Supports both offset and cursor-based pagination.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "List inferences with filtering and pagination.",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Filter by function name (optional)."
                },
                "variant_name": {
                    "type": "string",
                    "description": "Filter by variant name (optional)."
                },
                "episode_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Filter by episode ID (optional)."
                },
                "output_source": {
                    "type": "string",
                    "enum": ["none", "inference", "demonstration"],
                    "default": "inference",
                    "description": "Source of the inference output. 'inference' returns the original output, 'demonstration' returns manually-curated output if available, 'none' returns no output."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of inferences to return (default: 20)."
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of inferences to skip (for pagination). Cannot be used with 'before' or 'after'."
                },
                "before": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Inference ID to paginate before (exclusive). Returns inferences earlier in time. Cannot be used with 'after' or 'offset'."
                },
                "after": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Inference ID to paginate after (exclusive). Returns inferences later in time. Cannot be used with 'before' or 'offset'."
                },
                "filters": {
                    "description": "Optional filter to apply when querying inferences. Supports filtering by metrics, tags, time, and logical combinations (AND/OR/NOT).",
                    "oneOf": [
                        {
                            "type": "object",
                            "description": "Filter by float metric value.",
                            "properties": {
                                "type": { "const": "float_metric" },
                                "metric_name": { "type": "string", "description": "Name of the metric to filter by." },
                                "value": { "type": "number", "description": "Value to compare against." },
                                "comparison_operator": {
                                    "type": "string",
                                    "enum": ["<", "<=", "=", ">", ">=", "!="],
                                    "description": "Comparison operator."
                                }
                            },
                            "required": ["type", "metric_name", "value", "comparison_operator"]
                        },
                        {
                            "type": "object",
                            "description": "Filter by boolean metric value.",
                            "properties": {
                                "type": { "const": "boolean_metric" },
                                "metric_name": { "type": "string", "description": "Name of the metric to filter by." },
                                "value": { "type": "boolean", "description": "Value to compare against." }
                            },
                            "required": ["type", "metric_name", "value"]
                        },
                        {
                            "type": "object",
                            "description": "Filter by whether an inference has a demonstration.",
                            "properties": {
                                "type": { "const": "demonstration_feedback" },
                                "has_demonstration": { "type": "boolean", "description": "Whether the inference has a demonstration." }
                            },
                            "required": ["type", "has_demonstration"]
                        },
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
                    "description": "Optional ordering criteria for the results. Supports multiple sort criteria.",
                    "items": {
                        "type": "object",
                        "description": "A single ordering criterion.",
                        "properties": {
                            "by": {
                                "type": "string",
                                "enum": ["timestamp", "metric", "search_relevance"],
                                "description": "The property to order by. 'timestamp' orders by creation time, 'metric' orders by a metric value (requires 'name'), 'search_relevance' orders by search query relevance (requires search_query_experimental)."
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the metric to order by. Required when 'by' is 'metric'."
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
            }
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
impl SimpleTool for ListInferencesTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .list_inferences(llm_params.request)
            .await
            .map_err(|e| AutopilotToolError::client_error("list_inferences", e).into())
    }
}
