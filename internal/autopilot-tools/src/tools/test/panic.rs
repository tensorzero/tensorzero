//! Panic tool for testing crash recovery.

use std::borrow::Cow;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{TaskTool, ToolContext, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the panic tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct PanicToolParams {
    /// The panic message.
    pub panic_message: String,
}

/// Panic tool that panics with the given message.
/// Useful for testing crash recovery and worker stability after panics.
#[derive(Default)]
pub struct PanicTool;

impl ToolMetadata for PanicTool {
    type SideInfo = AutopilotSideInfo;
    type Output = ();
    type LlmParams = PanicToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("panic")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Panics with the given message. Used for testing crash recovery.")
    }
}

#[async_trait]
impl TaskTool for PanicTool {
    type ExtraState = ();
    #[expect(
        clippy::panic,
        reason = "This tool is specifically for testing panic handling"
    )]
    async fn execute(
        &self,
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        std::panic::panic_any(llm_params.panic_message);
    }
}
