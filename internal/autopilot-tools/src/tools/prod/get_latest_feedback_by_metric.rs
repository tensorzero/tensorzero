//! Tool for getting the latest feedback ID for each metric for a target.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
use uuid::Uuid;

use crate::types::AutopilotToolSideInfo;

/// Parameters for the get_latest_feedback_by_metric tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetLatestFeedbackByMetricToolParams {
    /// The target ID (inference ID) to get feedback for.
    pub target_id: Uuid,
}

/// Tool for getting the latest feedback ID for each metric for a target.
///
/// This tool returns a map from metric name to the latest feedback ID for that metric.
#[derive(Default)]
pub struct GetLatestFeedbackByMetricTool;

impl ToolMetadata for GetLatestFeedbackByMetricTool {
    type SideInfo = AutopilotToolSideInfo;
    type Output = LatestFeedbackIdByMetricResponse;
    type LlmParams = GetLatestFeedbackByMetricToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("get_latest_feedback_by_metric")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Get the latest feedback ID for each metric for a given target (inference). \
             Returns a map from metric name to feedback ID.",
        )
    }
}

#[async_trait]
impl SimpleTool for GetLatestFeedbackByMetricTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .get_latest_feedback_id_by_metric(llm_params.target_id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
