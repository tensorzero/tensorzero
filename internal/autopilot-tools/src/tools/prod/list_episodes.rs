//! Tool for listing episodes with pagination and filtering.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::{InferenceFilter, ListEpisodesRequest, ListEpisodesResponse};
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;

/// Parameters for the list_episodes tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ListEpisodesToolParams {
    /// Maximum number of episodes to return (max 100).
    pub limit: u32,
    /// Return episodes before this episode_id (for pagination).
    pub before: Option<Uuid>,
    /// Return episodes after this episode_id (for pagination).
    pub after: Option<Uuid>,
    /// Filter to episodes containing inferences for this function.
    pub function_name: Option<String>,
    /// Filter to episodes containing inferences matching these criteria.
    /// Supports boolean_metric, float_metric, tag, time, and logical combinators.
    pub filters: Option<InferenceFilter>,
}

/// Tool for listing episodes with pagination and filtering.
///
/// Returns episodes ordered by episode_id in descending order (most recent first).
/// Each episode includes its inference count, time range, and last inference ID.
/// Episodes are returned if they contain at least one inference matching the filter criteria.
#[derive(Default)]
pub struct ListEpisodesTool;

impl ToolMetadata for ListEpisodesTool {
    type SideInfo = AutopilotSideInfo;
    type Output = ListEpisodesResponse;
    type LlmParams = ListEpisodesToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("list_episodes")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "List episodes with pagination and optional filtering. \
             Returns episodes ordered by episode_id descending (most recent first). \
             Each episode includes inference count, start/end time, and last inference ID. \
             Filter by function_name and/or inference filters (metrics, tags, time). \
             Episodes are returned if they have at least one matching inference.",
        )
    }

    fn strict(&self) -> bool {
        false // Filter children are recursive arbitrary objects
    }

    // NOTE: This schema must be kept in sync with the tool params.
    // We manually construct it for OpenAI structured outputs compatibility.
    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "List episodes with pagination and optional filtering.",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of episodes to return (max 100).",
                    "minimum": 1,
                    "maximum": 100
                },
                "before": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Return episodes before this episode_id (exclusive). For paginating to older episodes."
                },
                "after": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Return episodes after this episode_id (exclusive). For paginating to newer episodes."
                },
                "function_name": {
                    "type": "string",
                    "description": "Filter to episodes containing inferences for this function."
                },
                "filters": {
                    "description": "Optional filter to apply when querying episodes. Episodes are returned if they have at least one inference matching the filter. Supports filtering by metrics, tags, time, and logical combinations (AND/OR/NOT).",
                    "anyOf": [
                        {
                            "type": "object",
                            "description": "Filter by boolean metric value.",
                            "properties": {
                                "type": { "const": "boolean_metric" },
                                "metric_name": { "type": "string", "description": "Name of the metric to filter by." },
                                "value": { "type": "boolean", "description": "Value to compare against." }
                            },
                            "required": ["type", "metric_name", "value"],
                            "additionalProperties": false
                        },
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
                            "required": ["type", "metric_name", "value", "comparison_operator"],
                            "additionalProperties": false
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
                            "required": ["type", "key", "value", "comparison_operator"],
                            "additionalProperties": false
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
                            "required": ["type", "time", "comparison_operator"],
                            "additionalProperties": false
                        },
                        {
                            "type": "object",
                            "description": "Logical AND of multiple filters.",
                            "properties": {
                                "type": { "const": "and" },
                                "children": { "type": "array", "description": "Array of filters to AND together.", "items": { "type": "object" } }
                            },
                            "required": ["type", "children"],
                            "additionalProperties": false
                        },
                        {
                            "type": "object",
                            "description": "Logical OR of multiple filters.",
                            "properties": {
                                "type": { "const": "or" },
                                "children": { "type": "array", "description": "Array of filters to OR together.", "items": { "type": "object" } }
                            },
                            "required": ["type", "children"],
                            "additionalProperties": false
                        },
                        {
                            "type": "object",
                            "description": "Logical NOT of a filter.",
                            "properties": {
                                "type": { "const": "not" },
                                "child": { "type": "object", "description": "Filter to negate." }
                            },
                            "required": ["type", "child"],
                            "additionalProperties": false
                        }
                    ]
                }
            },
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
impl SimpleTool for ListEpisodesTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let request = ListEpisodesRequest {
            limit: llm_params.limit,
            before: llm_params.before,
            after: llm_params.after,
            function_name: llm_params.function_name,
            filters: llm_params.filters,
        };
        ctx.client()
            .list_episodes(request)
            .await
            .map_err(|e| AutopilotToolError::client_error("list_episodes", e).into())
    }
}
