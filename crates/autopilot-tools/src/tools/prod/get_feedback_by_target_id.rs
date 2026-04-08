//! Tool for getting feedback by target ID (inference or episode).

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero_core::endpoints::feedback::internal::GetFeedbackByTargetIdResponse;
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;

/// Parameters for the get_feedback_by_target_id tool (visible to LLM).
#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[ts(export)]
pub struct GetFeedbackByTargetIdToolParams {
    /// The target ID (inference or episode) to get feedback for.
    pub target_id: Uuid,
    /// Maximum number of feedback entries to return.
    #[serde(default)]
    pub limit: Option<u32>,
}

/// Tool for getting all feedback for a target (inference or episode).
///
/// This tool returns a list of feedback entries (boolean metrics, float metrics,
/// comments, and demonstrations) for a given target.
#[derive(Default)]
pub struct GetFeedbackByTargetIdTool;

impl ToolMetadata for GetFeedbackByTargetIdTool {
    type SideInfo = AutopilotSideInfo;
    type Output = GetFeedbackByTargetIdResponse;
    type LlmParams = GetFeedbackByTargetIdToolParams;

    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GET_FEEDBACK_BY_TARGET_ID_TOOL_PARAMS
    }

    fn llm_params_ts_bundle_type_name() -> String {
        "GetFeedbackByTargetIdToolParams".to_string()
    }

    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GET_FEEDBACK_BY_TARGET_ID_RESPONSE
    }

    fn output_ts_bundle_type_name() -> String {
        "GetFeedbackByTargetIdResponse".to_string()
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("get_feedback_by_target_id")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Get all feedback for a given target (inference or episode). \
             Returns a list of feedback entries including boolean metrics, float metrics, \
             comments, and demonstrations.",
        )
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Get all feedback for a target (inference or episode).",
            "properties": {
                "target_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "The target ID (inference or episode) to get feedback for."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of feedback entries to return.",
                    "minimum": 1
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
impl SimpleTool for GetFeedbackByTargetIdTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .get_feedback_by_target_id(llm_params.target_id, None, None, llm_params.limit)
            .await
            .map_err(|e| AutopilotToolError::client_error("get_feedback_by_target_id", e).into())
    }
}
