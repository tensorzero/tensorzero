//! Auto-reject tool for handling unknown tool calls.
//!
//! This tool is called when the autopilot client receives a tool call for a tool
//! that doesn't exist in the TensorZero deployment. It sends a `NotAvailable`
//! authorization event to the autopilot API.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, TaskTool, ToolContext, ToolMetadata, ToolResult};
use schemars::Schema;
use serde::{Deserialize, Serialize};

use crate::error::AutopilotToolError;
use autopilot_client::{
    AutopilotSideInfo, EventPayload, EventPayloadToolCallAuthorization,
    ToolCallAuthorizationStatus, ToolCallDecisionSource,
};
use schemars::JsonSchema;
use tensorzero_core::endpoints::internal::autopilot::CreateEventGatewayRequest;

/// Parameters for the auto-reject tool (not visible to LLM - internal use only).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct AutoRejectToolCallParams {}

/// Built-in tool to send `NotAvailable` authorization for unknown tool calls.
///
/// This tool is spawned by the autopilot client when it receives a tool call
/// for a tool that doesn't exist in the deployment. It sends a `NotAvailable`
/// event to the autopilot API so the session can continue without hanging.
///
/// Reserved name: `__auto_reject_tool_call__`
#[derive(Default)]
pub struct AutoRejectToolCallTool;

impl ToolMetadata for AutoRejectToolCallTool {
    type SideInfo = AutopilotSideInfo;
    type Output = ();
    type LlmParams = AutoRejectToolCallParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("__auto_reject_tool_call__")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Internal tool for rejecting unknown tool calls. Not intended for direct use.",
        )
    }

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::AUTO_REJECT_TOOL_CALL_PARAMS
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::UNIT
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Internal tool for auto-rejecting unknown tool calls.",
            "properties": {},
            "required": [],
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
impl TaskTool for AutoRejectToolCallTool {
    async fn execute(
        &self,
        _llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        // Send ToolCallAuthorization::NotAvailable to autopilot API
        ctx.client()
            .create_autopilot_event(
                side_info.session_id,
                CreateEventGatewayRequest {
                    payload: EventPayload::ToolCallAuthorization(
                        EventPayloadToolCallAuthorization {
                            source: ToolCallDecisionSource::Automatic,
                            tool_call_event_id: side_info.tool_call_event_id,
                            status: ToolCallAuthorizationStatus::NotAvailable,
                        },
                    ),
                    previous_user_message_event_id: None,
                },
            )
            .await
            .map_err(|e| AutopilotToolError::client_error("auto_reject_tool_call", e))?;

        Ok(())
    }
}
