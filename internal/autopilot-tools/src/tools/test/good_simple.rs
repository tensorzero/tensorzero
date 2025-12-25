//! Good simple tool that always succeeds.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};

/// Parameters for the good simple tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GoodSimpleParams {
    /// The message to echo back.
    pub message: String,
}

/// Output from the good simple tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoodSimpleOutput {
    /// The echoed message.
    pub echoed: String,
}

/// Good simple tool that always succeeds.
/// Useful for testing basic SimpleTool functionality.
#[derive(Default)]
pub struct GoodSimpleTool;

impl ToolMetadata for GoodSimpleTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("good_simple")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Echoes back the input message. A SimpleTool for testing basic success cases.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(GoodSimpleParams))
    }

    type LlmParams = GoodSimpleParams;
}

#[async_trait]
impl SimpleTool for GoodSimpleTool {
    type SideInfo = ();
    type Output = GoodSimpleOutput;

    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        Ok(GoodSimpleOutput {
            echoed: llm_params.message,
        })
    }
}
