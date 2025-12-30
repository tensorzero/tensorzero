//! Feedback tool for calling TensorZero feedback endpoint.

use std::borrow::Cow;
use std::collections::HashMap;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero::{FeedbackParams, FeedbackResponse};
use uuid::Uuid;

use crate::types::AutopilotToolSideInfo;

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
    /// Optional tags to add to the feedback.
    #[serde(default)]
    pub tags: HashMap<String, String>,
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

/// Build autopilot tags from side info.
fn build_autopilot_tags(side_info: &AutopilotToolSideInfo) -> HashMap<String, String> {
    let mut tags = HashMap::new();
    tags.insert(
        "autopilot_session_id".to_string(),
        side_info.session_id.to_string(),
    );
    tags.insert(
        "autopilot_tool_call_id".to_string(),
        side_info.tool_call_id.to_string(),
    );
    tags.insert(
        "autopilot_tool_call_event_id".to_string(),
        side_info.tool_call_event_id.to_string(),
    );
    tags
}

/// Merge autopilot tags into existing tags, preserving user-provided tags.
fn merge_tags(
    existing: HashMap<String, String>,
    autopilot_tags: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut merged = autopilot_tags.clone();
    // User-provided tags take precedence over autopilot tags
    merged.extend(existing);
    merged
}

#[async_trait]
impl SimpleTool for FeedbackTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        // Build autopilot tags and merge with user-provided tags
        let autopilot_tags = build_autopilot_tags(&side_info);
        let tags = merge_tags(llm_params.tags, &autopilot_tags);

        let params = FeedbackParams {
            episode_id: llm_params.episode_id,
            inference_id: llm_params.inference_id,
            metric_name: llm_params.metric_name,
            value: llm_params.value,
            internal: true, // Always internal for autopilot
            tags,
            dryrun: llm_params.dryrun,
        };

        ctx.client()
            .feedback(params)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
