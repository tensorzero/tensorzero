//! Tool for getting the latest feedback ID for each metric for a target.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;

/// Parameters for the get_latest_feedback_by_metric tool (visible to LLM).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
    type SideInfo = AutopilotSideInfo;
    type Output = LatestFeedbackIdByMetricResponse;
    type LlmParams = GetLatestFeedbackByMetricToolParams;

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GET_LATEST_FEEDBACK_BY_METRIC_TOOL_PARAMS
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::LATEST_FEEDBACK_ID_BY_METRIC_RESPONSE
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("get_latest_feedback_by_metric")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Get the latest feedback ID for each metric for a given target (inference). \
             Returns a map from metric name to feedback ID.",
        )
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Get the latest feedback ID for each metric for a target inference.",
            "properties": {
                "target_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "The target ID (inference ID) to get feedback for."
                }
            },
            "required": ["target_id"],
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
            .map_err(|e| {
                AutopilotToolError::client_error("get_latest_feedback_by_metric", e).into()
            })
    }
}
