//! Good simple tool that always succeeds.

use std::borrow::Cow;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the good simple tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
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
    type SideInfo = AutopilotSideInfo;
    type Output = GoodSimpleOutput;
    type LlmParams = GoodSimpleParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("good_simple")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Echoes back the input message. A SimpleTool for testing basic success cases.",
        )
    }

    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::UNIT
    }

    fn llm_params_ts_bundle_type_name() -> String {
        "void".to_string()
    }

    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::UNIT
    }

    fn output_ts_bundle_type_name() -> String {
        "void".to_string()
    }
}

#[async_trait]
impl SimpleTool for GoodSimpleTool {
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
