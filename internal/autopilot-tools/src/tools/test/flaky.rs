//! Flaky tool that fails deterministically based on attempt number.

use std::borrow::Cow;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{TaskTool, ToolContext, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};

use crate::error::AutopilotToolError;
use crate::fix_strict_tool_schema::fix_strict_tool_schema;

/// Parameters for the flaky tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct FlakyToolParams {
    /// Fail when attempt_number % fail_on_attempt == 0.
    pub fail_on_attempt: u32,
    /// The current attempt number (caller increments this).
    pub attempt_number: u32,
    /// Message to return on success.
    pub message: String,
}

/// Output from the flaky tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlakyToolOutput {
    /// The message.
    pub message: String,
    /// The attempt number that succeeded.
    pub attempt_number: u32,
}

/// Flaky tool that fails deterministically based on attempt number.
/// Useful for testing retry logic and deterministic failure scenarios.
#[derive(Default)]
pub struct FlakyTool;

impl ToolMetadata for FlakyTool {
    type SideInfo = AutopilotSideInfo;
    type Output = FlakyToolOutput;
    type LlmParams = FlakyToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("flaky")
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        Ok(fix_strict_tool_schema(schema_for!(FlakyToolParams)))
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Fails when attempt_number % fail_on_attempt == 0. Used for testing deterministic failures.",
        )
    }
}

#[async_trait]
impl TaskTool for FlakyTool {
    type ExtraState = ();
    async fn execute(
        &self,
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        if llm_params.fail_on_attempt > 0
            && llm_params.attempt_number % llm_params.fail_on_attempt == 0
        {
            return Err(AutopilotToolError::test_error(format!(
                "Deterministic failure on attempt {} (fail_on_attempt={})",
                llm_params.attempt_number, llm_params.fail_on_attempt
            ))
            .into());
        }

        Ok(FlakyToolOutput {
            message: llm_params.message,
            attempt_number: llm_params.attempt_number,
        })
    }
}
