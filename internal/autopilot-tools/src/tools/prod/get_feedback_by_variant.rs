//! Tool for getting feedback statistics by variant for a function and metric.
//!
//! TODO: The `GetFeedbackByVariantToolParams` type is defined here temporarily because
//! there is no HTTP endpoint for this operation yet. Once an HTTP endpoint is added,
//! this should be replaced with the wire types from tensorzero-core (re-exported
//! through the SDK).

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_core::db::feedback::FeedbackByVariant;

use crate::types::AutopilotToolSideInfo;

/// Parameters for the get_feedback_by_variant tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetFeedbackByVariantToolParams {
    /// The name of the metric to query.
    pub metric_name: String,
    /// The name of the function to query.
    pub function_name: String,
    /// Optional filter for specific variants. If not provided, all variants are included.
    #[serde(default)]
    pub variant_names: Option<Vec<String>>,
}

/// Tool for getting feedback statistics by variant for a function and metric.
///
/// Returns mean, variance, and count for each variant. This is useful for
/// analyzing variant performance.
///
/// Note: This tool only works in embedded mode (no HTTP endpoint available).
#[derive(Default)]
pub struct GetFeedbackByVariantTool;

impl ToolMetadata for GetFeedbackByVariantTool {
    type SideInfo = AutopilotToolSideInfo;
    type Output = Vec<FeedbackByVariant>;
    type LlmParams = GetFeedbackByVariantToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("get_feedback_by_variant")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Get feedback statistics (mean, variance, count) by variant for a function and metric. \
             Returns statistics for each variant that has feedback data. \
             Optionally filter by specific variant names.",
        )
    }
}

#[async_trait]
impl SimpleTool for GetFeedbackByVariantTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .get_feedback_by_variant(
                llm_params.metric_name,
                llm_params.function_name,
                llm_params.variant_names,
            )
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
