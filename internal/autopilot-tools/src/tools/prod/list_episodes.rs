//! Tool for listing episodes with pagination.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::{ListEpisodesParams, ListEpisodesResponse};
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
}

/// Tool for listing episodes with pagination.
///
/// Returns episode metadata including episode ID, inference count,
/// start/end times, and last inference ID. Use this to discover
/// episodes and paginate through them.
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
            "List episodes with pagination. Returns episode metadata including episode ID, \
             inference count, start/end times, and last inference ID.",
        )
    }

    // NOTE: This schema must be kept in sync with `ListEpisodesToolParams` fields.
    // We manually construct it for OpenAI structured outputs compatibility (using anyOf for optional fields).
    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "List episodes with pagination.",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of episodes to return (max 100)."
                },
                "before": {
                    "anyOf": [
                        { "type": "string", "format": "uuid" },
                        { "type": "null" }
                    ],
                    "description": "Return episodes before this episode_id (for pagination)."
                },
                "after": {
                    "anyOf": [
                        { "type": "string", "format": "uuid" },
                        { "type": "null" }
                    ],
                    "description": "Return episodes after this episode_id (for pagination)."
                }
            },
            "required": ["limit"],
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
        let params = ListEpisodesParams {
            limit: llm_params.limit,
            before: llm_params.before,
            after: llm_params.after,
        };
        ctx.client()
            .list_episodes(params)
            .await
            .map_err(|e| AutopilotToolError::client_error("list_episodes", e).into())
    }
}
