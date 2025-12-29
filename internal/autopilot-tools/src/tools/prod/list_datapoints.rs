//! Tool for listing datapoints in a dataset.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use tensorzero::{GetDatapointsResponse, ListDatapointsRequest};

use crate::types::AutopilotToolSideInfo;

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
    type SideInfo = AutopilotToolSideInfo;
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
        Ok(schema_for!(ListDatapointsToolParams))
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
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
