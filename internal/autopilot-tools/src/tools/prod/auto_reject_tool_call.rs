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
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
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

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("__auto_reject_tool_call__")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Internal tool for rejecting unknown tool calls. Not intended for direct use.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Internal tool for auto-rejecting unknown tool calls.",
            "properties": {},
            "required": []
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
