//! Feedback tool for calling TensorZero feedback endpoint.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero::{FeedbackParams, FeedbackResponse};
use uuid::Uuid;

use crate::AutopilotToolSideInfo;

/// Parameters for the feedback tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FeedbackToolParams {
    /// The episode ID to provide feedback for. Exactly one of episode_id or inference_id must be set.
    #[serde(default)]
    pub episode_id: Option<Uuid>,
    /// The inference ID to provide feedback for. Exactly one of episode_id or inference_id must be set.
    #[serde(default)]
    pub inference_id: Option<Uuid>,
    /// The name of the metric to provide feedback for.
    /// Use "comment" for free-text comments, "demonstration" for demonstration feedback,
    /// or a configured metric name for float/boolean feedback.
    pub metric_name: String,
    /// The value of the feedback. Type depends on metric_name:
    /// - "comment": string
    /// - "demonstration": string or array of content blocks
    /// - float metric: number
    /// - boolean metric: boolean
    pub value: Value,
    /// If true, the feedback will not be stored (useful for testing).
    #[serde(default)]
    pub dryrun: Option<bool>,
}

/// Tool for calling TensorZero feedback endpoint.
///
/// This tool allows autopilot to submit feedback for inferences or episodes.
/// Feedback can be comments, demonstrations, or metric values (float or boolean).
#[derive(Default)]
pub struct FeedbackTool;

impl ToolMetadata for FeedbackTool {
    type SideInfo = AutopilotToolSideInfo;
    type Output = FeedbackResponse;
    type LlmParams = FeedbackToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("feedback")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Submit feedback for a TensorZero inference or episode. \
             Use metric_name='comment' for free-text comments, 'demonstration' for demonstrations, \
             or a configured metric name for float/boolean feedback values.",
        )
    }
}

#[async_trait]
impl SimpleTool for FeedbackTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let params = FeedbackParams {
            episode_id: llm_params.episode_id,
            inference_id: llm_params.inference_id,
            metric_name: llm_params.metric_name,
            value: llm_params.value,
            internal: true, // Always internal for autopilot
            tags: side_info.to_tags(),
            dryrun: llm_params.dryrun,
        };

        ctx.client()
            .feedback(params)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
