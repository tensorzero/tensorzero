//! Inference tool for calling TensorZero inference endpoint.

use std::borrow::Cow;
use std::collections::HashMap;

use async_trait::async_trait;
use durable_tools::{SideInfo, SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero::{
    ActionInput, ClientInferenceParams, DynamicToolParams, InferenceParams, InferenceResponse,
    Input,
};
use tensorzero_core::config::snapshot::SnapshotHash;
use uuid::Uuid;

/// Parameters for the inference tool (visible to LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InferenceToolParams {
    /// The function name to call. Exactly one of function_name or model_name required.
    #[serde(default)]
    pub function_name: Option<String>,
    /// Model name shorthand (e.g., "openai::gpt-4"). Alternative to function_name.
    #[serde(default)]
    pub model_name: Option<String>,
    /// The input for inference.
    pub input: Input,
    /// Inference parameters (temperature, max_tokens, etc.).
    #[serde(default)]
    pub params: InferenceParams,
    /// Pin a specific variant (optional, normally let API select).
    #[serde(default)]
    pub variant_name: Option<String>,
    /// Dynamic tool parameters (allowed_tools, additional_tools, tool_choice, parallel_tool_calls).
    #[serde(flatten, default)]
    pub dynamic_tool_params: DynamicToolParams,
    /// Output schema override (for JSON functions).
    #[serde(default)]
    pub output_schema: Option<Value>,
}

/// Side information for the inference tool (hidden from LLM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceToolSideInfo {
    /// Episode ID to use for the inference (links to autopilot session).
    pub episode_id: Uuid,
    /// Session ID for tagging.
    pub session_id: Uuid,
    /// Tool call ID for tagging.
    pub tool_call_id: Uuid,
    /// Tool call event ID for tagging.
    pub tool_call_event_id: Uuid,
    /// Optional config snapshot hash - if provided, uses action endpoint with historical config.
    #[serde(default)]
    pub config_snapshot_hash: Option<String>,
}

impl SideInfo for InferenceToolSideInfo {}

/// Tool for calling TensorZero inference endpoint.
///
/// This tool allows autopilot to make inference calls, optionally using
/// a historical config snapshot for reproducibility.
#[derive(Default)]
pub struct InferenceTool;

impl ToolMetadata for InferenceTool {
    type SideInfo = InferenceToolSideInfo;
    type Output = InferenceResponse;
    type LlmParams = InferenceToolParams;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("inference")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Call TensorZero inference endpoint. Optionally use a config snapshot hash to use historical configuration.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(InferenceToolParams))
    }
}

#[async_trait]
impl SimpleTool for InferenceTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        // Build tags with useful metadata
        let mut tags = HashMap::new();
        tags.insert(
            "autopilot_session_id".to_string(),
            side_info.session_id.to_string(),
        );
        tags.insert(
            "autopilot_tool_call_id".to_string(),
            side_info.tool_call_id.to_string(),
        );
        tags.insert(
            "autopilot_tool_call_event_id".to_string(),
            side_info.tool_call_event_id.to_string(),
        );

        let client_params = ClientInferenceParams {
            function_name: llm_params.function_name,
            model_name: llm_params.model_name,
            input: llm_params.input,
            episode_id: Some(side_info.episode_id),
            params: llm_params.params,
            variant_name: llm_params.variant_name,
            dryrun: Some(false), // Always store
            tags,
            dynamic_tool_params: llm_params.dynamic_tool_params,
            output_schema: llm_params.output_schema,
            stream: Some(false), // Never stream
            internal: true,      // Skip tag validation
            ..Default::default()
        };

        let response = if let Some(hash) = side_info.config_snapshot_hash {
            let snapshot_hash: SnapshotHash =
                hash.parse()
                    .map_err(|_: std::convert::Infallible| ToolError::Validation {
                        message: "Invalid snapshot hash".to_string(),
                    })?;
            ctx.client()
                .action(
                    snapshot_hash,
                    ActionInput::Inference(Box::new(client_params)),
                )
                .await
                .map_err(|e| ToolError::ExecutionFailed(e.into()))?
        } else {
            ctx.client()
                .inference(client_params)
                .await
                .map_err(|e| ToolError::ExecutionFailed(e.into()))?
        };

        Ok(response)
    }
}
