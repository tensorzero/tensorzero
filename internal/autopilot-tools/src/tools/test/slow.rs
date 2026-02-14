//! Slow tool for testing timeout behavior with configurable delay.

use std::borrow::Cow;
use std::time::Instant;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{TaskTool, ToolContext, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};

use crate::fix_strict_tool_schema::fix_strict_tool_schema;

/// Parameters for the slow tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
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
    type SideInfo = AutopilotSideInfo;
    type Output = SlowToolOutput;
    type LlmParams = SlowToolParams;

    fn parameters_schema(&self) -> ToolResult<Schema> {
        Ok(fix_strict_tool_schema(schema_for!(SlowToolParams)))
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("slow")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Sleeps for the specified duration before returning. Used for testing timeout behavior.",
        )
    }
}

#[async_trait]
impl TaskTool for SlowTool {
    type ExtraState = ();
    async fn execute(
        &self,
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
