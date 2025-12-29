//! Tool for deleting datapoints from a dataset.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use tensorzero::DeleteDatapointsResponse;
use uuid::Uuid;

use crate::types::AutopilotToolSideInfo;

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
    type SideInfo = AutopilotToolSideInfo;
    type Output = DeleteDatapointsResponse;
    type LlmParams = DeleteDatapointsToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("delete_datapoints")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Delete datapoints from a dataset by their IDs. \
             This is a soft delete - datapoints are marked as stale but not truly removed.",
        )
    }

    fn parameters_schema() -> Schema {
        schema_for!(DeleteDatapointsToolParams)
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
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
