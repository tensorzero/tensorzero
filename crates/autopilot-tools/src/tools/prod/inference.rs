//! Inference tool for calling TensorZero inference endpoint.

use std::{borrow::Cow, time::Duration};

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use durable_tools::{ActionInput, ActionResponse};
use schemars::Schema;
use tensorzero::{ClientInferenceParams, InferenceResponse};
use tensorzero_core::config::snapshot::SnapshotHash;
pub use tensorzero_core::endpoints::inference::InferenceToolParams;

use autopilot_client::AutopilotSideInfo;

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

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::INFERENCE_TOOL_PARAMS
    }

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle_type_name() -> String {
        "InferenceToolParams".to_string()
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::INFERENCE_RESPONSE
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle_type_name() -> String {
        "InferenceResponse".to_string()
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("inference")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Call TensorZero inference endpoint. Optionally use a config snapshot hash to use historical configuration.",
        )
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
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
                            "anyOf": [
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
                                        "anyOf": [
                                            { "type": "string" },
                                            { "type": "array", "items": { "type": "object" } }
                                        ]
                                    }
                                },
                                "required": ["role", "content"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["messages"],
                    "additionalProperties": false
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
                            },
                            "additionalProperties": false
                        }
                    },
                    "additionalProperties": false
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
            "required": ["input"],
            "additionalProperties": false
        });

        serde_json::from_value(schema).map_err(|e| {
            NonControlToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }

    fn strict(&self) -> bool {
        false // We need an arbitrary object for 'output_schema'
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5 * 60)
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

        let snapshot_hash: SnapshotHash = side_info
            .config_snapshot_hash
            .parse()
            .map_err(|_| AutopilotToolError::validation("Invalid snapshot hash"))?;
        let response = ctx
            .client()
            .action(
                snapshot_hash,
                ActionInput::Inference(Box::new(client_params)),
                ctx.heartbeater().clone(),
            )
            .await
            .map_err(|e| AutopilotToolError::client_error("inference", e))?;

        match response {
            ActionResponse::Inference(inference_response) => Ok(inference_response),
            _ => Err(AutopilotToolError::validation(
                "Unexpected response type from action endpoint",
            )
            .into()),
        }
    }
}
