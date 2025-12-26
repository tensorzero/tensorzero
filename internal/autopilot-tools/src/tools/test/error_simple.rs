//! Error simple tool that always returns an error.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};

/// Parameters for the error simple tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ErrorSimpleParams {
    /// The error message to return.
    pub error_message: String,
}

/// Error simple tool that always returns an error.
/// Useful for testing SimpleTool error propagation.
#[derive(Default)]
pub struct ErrorSimpleTool;

impl ToolMetadata for ErrorSimpleTool {
    type SideInfo = ();
    type Output = ();
    type LlmParams = ErrorSimpleParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("error_simple")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Always returns an error with the specified message. A SimpleTool for testing error propagation.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(ErrorSimpleParams))
    }
}

#[async_trait]
impl SimpleTool for ErrorSimpleTool {
    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        Err(ToolError::Validation {
            message: llm_params.error_message,
        })
    }
}
