//! Tool for listing inferences with filtering and pagination.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
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
        Ok(schema_for!(ListInferencesToolParams))
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
