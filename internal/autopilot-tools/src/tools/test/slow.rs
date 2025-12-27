//! Slow tool for testing timeout behavior with configurable delay.

use std::borrow::Cow;
use std::time::Instant;

use async_trait::async_trait;
use durable_tools::{TaskTool, ToolContext, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};

/// Parameters for the slow tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SlowToolParams {
    /// How long to sleep before returning (in milliseconds).
    pub delay_ms: u64,
    /// Message to echo back.
    pub message: String,
}

/// Output from the slow tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowToolOutput {
    /// The echoed message.
    pub echoed: String,
    /// The actual delay in milliseconds.
    pub actual_delay_ms: u64,
}

/// Slow tool that sleeps for a configurable duration before returning.
/// Useful for testing timeout behavior.
#[derive(Default)]
pub struct SlowTool;

impl ToolMetadata for SlowTool {
    type SideInfo = ();
    type Output = SlowToolOutput;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("slow")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Sleeps for the specified duration before returning. Used for testing timeout behavior.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(SlowToolParams))
    }

    type LlmParams = SlowToolParams;
}

#[async_trait]
impl TaskTool for SlowTool {
    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        let start = Instant::now();
        tokio::time::sleep(tokio::time::Duration::from_millis(llm_params.delay_ms)).await;
        let actual_delay_ms = start.elapsed().as_millis() as u64;

        Ok(SlowToolOutput {
            echoed: llm_params.message,
            actual_delay_ms,
        })
    }
}
