//! Inference tool for calling TensorZero inference endpoint.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero::{
    ActionInput, ClientInferenceParams, DynamicToolParams, InferenceParams, InferenceResponse,
    Input,
};
use tensorzero_core::config::snapshot::SnapshotHash;

use autopilot_client::AutopilotSideInfo;

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

/// Tool for calling TensorZero inference endpoint.
///
/// This tool allows autopilot to make inference calls, optionally using
/// a historical config snapshot for reproducibility.
#[derive(Default)]
pub struct InferenceTool;

impl ToolMetadata for InferenceTool {
    type SideInfo = AutopilotSideInfo;
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
        let schema = serde_json::json!({
            "type": "object",
            "description": "Call TensorZero inference endpoint to get an LLM response.",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "The function name to call (e.g., 'my_chat_function'). Either function_name or model_name is required."
                },
                "model_name": {
                    "type": "string",
                    "description": "Model shorthand as alternative to function_name (e.g., 'openai::gpt-4o', 'anthropic::claude-sonnet-4-20250514')"
                },
                "input": {
                    "type": "object",
                    "description": "The input for inference",
                    "properties": {
                        "system": {
                            "description": "System prompt (string or array of content blocks)",
                            "oneOf": [
                                { "type": "string" },
                                { "type": "array", "items": { "type": "object" } }
                            ]
                        },
                        "messages": {
                            "type": "array",
                            "description": "Conversation messages",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": { "type": "string", "enum": ["user", "assistant"] },
                                    "content": {
                                        "description": "Message content (string or array of content blocks)",
                                        "oneOf": [
                                            { "type": "string" },
                                            { "type": "array", "items": { "type": "object" } }
                                        ]
                                    }
                                },
                                "required": ["role", "content"]
                            }
                        }
                    },
                    "required": ["messages"]
                },
                "params": {
                    "type": "object",
                    "description": "Inference parameters",
                    "properties": {
                        "chat_completion": {
                            "type": "object",
                            "properties": {
                                "temperature": { "type": "number", "description": "Sampling temperature (0.0-2.0)" },
                                "max_tokens": { "type": "integer", "description": "Maximum tokens to generate" },
                                "seed": { "type": "integer", "description": "Random seed for reproducibility" }
                            }
                        }
                    }
                },
                "variant_name": {
                    "type": "string",
                    "description": "Pin a specific variant (optional, normally let API select)"
                },
                "output_schema": {
                    "type": "object",
                    "description": "Output schema override for JSON functions (optional)"
                }
            },
            "required": ["input"]
        });

        serde_json::from_value(schema)
            .map_err(|e| NonControlToolError::SchemaGeneration(e.into()).into())
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
        let client_params = ClientInferenceParams {
            function_name: llm_params.function_name,
            model_name: llm_params.model_name,
            input: llm_params.input,
            episode_id: None,
            params: llm_params.params,
            variant_name: llm_params.variant_name,
            dryrun: Some(false), // Always store
            tags: side_info.to_tags(),
            dynamic_tool_params: llm_params.dynamic_tool_params,
            output_schema: llm_params.output_schema,
            stream: Some(false), // Never stream
            internal: true,      // Skip tag validation
            ..Default::default()
        };

        let response = if let Some(hash) = side_info.config_snapshot_hash {
            let snapshot_hash: SnapshotHash =
                hash.parse().map_err(|_: std::convert::Infallible| {
                    NonControlToolError::Validation {
                        message: "Invalid snapshot hash".to_string(),
                    }
                })?;
            ctx.client()
                .action(
                    snapshot_hash,
                    ActionInput::Inference(Box::new(client_params)),
                )
                .await
                .map_err(|e| NonControlToolError::ExecutionFailed(e.into()))?
        } else {
            ctx.client()
                .inference(client_params)
                .await
                .map_err(|e| NonControlToolError::ExecutionFailed(e.into()))?
        };

        Ok(response)
    }
}
