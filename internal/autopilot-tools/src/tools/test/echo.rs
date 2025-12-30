//! Simple echo tool for testing the autopilot worker infrastructure.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{TaskTool, ToolContext, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the echo tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EchoParams {
    /// The message to echo back.
    pub message: String,
}

/// Output from the echo tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoOutput {
    /// The echoed message.
    pub echoed: String,
    /// The task ID that processed this request.
    pub task_id: String,
}

/// Simple echo tool for testing infrastructure.
#[derive(Default)]
pub struct EchoTool;

impl ToolMetadata for EchoTool {
    type SideInfo = ();
    type Output = EchoOutput;
    type LlmParams = EchoParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("echo")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Echoes back the input message. Used for testing the autopilot worker.")
    }
}

#[async_trait]
impl TaskTool for EchoTool {
    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        Ok(EchoOutput {
            echoed: llm_params.message,
            task_id: ctx.task_id().to_string(),
        })
    }
}
