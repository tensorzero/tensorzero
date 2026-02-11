//! Slow simple tool for testing timeout behavior with configurable delay.

use std::borrow::Cow;
use std::time::Instant;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};

use crate::fix_strict_tool_schema::fix_strict_tool_schema;
use serde::{Deserialize, Serialize};

/// Parameters for the slow simple tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct SlowSimpleParams {
    /// How long to sleep before returning (in milliseconds).
    pub delay_ms: u64,
    /// Message to echo back.
    pub message: String,
}

/// Output from the slow simple tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowSimpleOutput {
    /// The echoed message.
    pub echoed: String,
    /// The actual delay in milliseconds.
    pub actual_delay_ms: u64,
}

/// Slow simple tool that sleeps for a configurable duration before returning.
/// Useful for testing SimpleTool timeout behavior.
#[derive(Default)]
pub struct SlowSimpleTool;

impl ToolMetadata for SlowSimpleTool {
    type SideInfo = AutopilotSideInfo;
    type Output = SlowSimpleOutput;
    type LlmParams = SlowSimpleParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("slow_simple")
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        Ok(fix_strict_tool_schema(schema_for!(SlowSimpleParams)))
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Sleeps for the specified duration before returning. A SimpleTool for testing timeout behavior.",
        )
    }
}

#[async_trait]
impl SimpleTool for SlowSimpleTool {
    async fn execute(
        &self,
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        let start = Instant::now();
        tokio::time::sleep(tokio::time::Duration::from_millis(llm_params.delay_ms)).await;
        let actual_delay_ms = start.elapsed().as_millis() as u64;

        Ok(SlowSimpleOutput {
            echoed: llm_params.message,
            actual_delay_ms,
        })
    }
}
