//! Failing tool that always returns an error.

use std::borrow::Cow;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{TaskTool, ToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the failing tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct FailingToolParams {
    /// The error message to return.
    pub error_message: String,
}

/// Failing tool that always returns an error.
/// Useful for testing error propagation to LLM.
#[derive(Default)]
pub struct FailingTool;

impl ToolMetadata for FailingTool {
    type SideInfo = AutopilotSideInfo;
    type Output = ();
    type LlmParams = FailingToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("failing")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Always returns an error with the specified message. Used for testing error propagation.",
        )
    }
}

#[async_trait]
impl TaskTool for FailingTool {
    async fn execute(
        &self,
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        Err(AutopilotToolError::test_error(llm_params.error_message).into())
    }
}
