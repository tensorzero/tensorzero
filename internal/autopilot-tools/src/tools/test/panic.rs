//! Panic tool for testing crash recovery.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{TaskTool, ToolContext, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};

/// Parameters for the panic tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PanicToolParams {
    /// The panic message.
    pub panic_message: String,
}

/// Panic tool that panics with the given message.
/// Useful for testing crash recovery and worker stability after panics.
#[derive(Default)]
pub struct PanicTool;

impl ToolMetadata for PanicTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("panic")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Panics with the given message. Used for testing crash recovery.")
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(PanicToolParams))
    }

    type LlmParams = PanicToolParams;
}

#[async_trait]
impl TaskTool for PanicTool {
    type SideInfo = ();
    type Output = ();

    #[expect(
        clippy::panic,
        reason = "This tool is specifically for testing panic handling"
    )]
    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        std::panic::panic_any(llm_params.panic_message);
    }
}
