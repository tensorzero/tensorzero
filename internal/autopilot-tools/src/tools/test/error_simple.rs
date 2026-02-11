//! Error simple tool that always returns an error.

use std::borrow::Cow;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the error simple tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct ErrorSimpleParams {
    /// The error message to return.
    pub error_message: String,
}

/// Error simple tool that always returns an error.
/// Useful for testing SimpleTool error propagation.
#[derive(Default)]
pub struct ErrorSimpleTool;

impl ToolMetadata for ErrorSimpleTool {
    type SideInfo = AutopilotSideInfo;
    type Output = ();
    type LlmParams = ErrorSimpleParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("error_simple")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Always returns an error with the specified message. A SimpleTool for testing error propagation.",
        )
    }
}

#[async_trait]
impl SimpleTool for ErrorSimpleTool {
    async fn execute(
        &self,
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        Err(AutopilotToolError::test_error(llm_params.error_message).into())
    }
}
