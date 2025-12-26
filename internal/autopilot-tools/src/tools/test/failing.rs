//! Failing tool that always returns an error.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{TaskTool, ToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};

/// Parameters for the failing tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FailingToolParams {
    /// The error message to return.
    pub error_message: String,
}

/// Failing tool that always returns an error.
/// Useful for testing error propagation to LLM.
#[derive(Default)]
pub struct FailingTool;

impl ToolMetadata for FailingTool {
    type SideInfo = ();
    type Output = ();

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("failing")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Always returns an error with the specified message. Used for testing error propagation.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(FailingToolParams))
    }

    type LlmParams = FailingToolParams;
}

#[async_trait]
impl TaskTool for FailingTool {

    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        Err(ToolError::Validation {
            message: llm_params.error_message,
        })
    }
}
