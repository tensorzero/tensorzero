//! Tool for listing inferences with filtering and pagination.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::{GetInferencesResponse, ListInferencesRequest};

use crate::types::AutopilotToolSideInfo;

/// Parameters for the list_inferences tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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
    type SideInfo = AutopilotToolSideInfo;
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
                "tags": {
                    "type": "object",
                    "additionalProperties": { "type": "string" },
                    "description": "Filter by tags (optional)."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of inferences to return (default: 100)."
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of inferences to skip (for pagination)."
                },
                "order_by": {
                    "type": "string",
                    "enum": ["created_at_asc", "created_at_desc"],
                    "description": "Sort order (default: created_at_desc)."
                }
            }
        });

        serde_json::from_value(schema).map_err(|e| ToolError::SchemaGeneration(e.into()))
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
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
