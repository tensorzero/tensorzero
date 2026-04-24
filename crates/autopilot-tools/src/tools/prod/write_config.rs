//! Tool for writing config snapshots.

use std::borrow::Cow;
use std::collections::HashMap;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero::WriteConfigRequest;
use tensorzero_core::config::UninitializedConfig;

use autopilot_client::AutopilotSideInfo;

// Re-export EditPayload types from config-applier
pub use config_applier::{
    EditPayload, UpsertEvaluationPayload, UpsertEvaluatorPayload, UpsertExperimentationPayload,
    UpsertVariantPayload,
};

/// Output of the `write_config` tool (visible to LLM).
///
/// Tagged with `status` so the LLM sees an explicit success/error discriminant
/// rather than needing to distinguish a bare hash from a thrown exception.
/// Only LLM-actionable failures (e.g. malformed `config` that fails local
/// deserialization) land here as `Error`. Infrastructure failures from the
/// underlying client call (gateway 5xx, network, DB) propagate as `ToolError`
/// so the worker's failure-oriented handling and observability still fire.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum WriteConfigToolOutput {
    /// The config snapshot was written successfully.
    Ok {
        /// Hash identifying the newly-written config snapshot.
        hash: String,
    },
    /// The write failed in a way the LLM can reason about and recover from.
    Error {
        /// Stable machine-readable error category.
        code: WriteConfigErrorCode,
        /// Human-readable error message.
        message: String,
    },
}

/// Stable error category for `WriteConfigToolOutput::Error`.
///
/// Only covers cases the LLM can act on. Transport/server failures are
/// surfaced as `ToolError` instead and never reach this enum.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
pub enum WriteConfigErrorCode {
    /// Input parameters were invalid (e.g. malformed config JSON).
    Validation,
}

/// Parameters for the write_config tool (visible to LLM).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct WriteConfigToolParams {
    /// The config to write as a JSON object.
    pub config: Value,
    /// Templates that should be stored with the config.
    #[serde(default)]
    pub extra_templates: HashMap<String, String>,
    /// We could have consolidated an array of server-side edits into one client-side edit, so this type contains a Vec
    /// Unset means an older API. This should always be set and we should make it mandatory once upstream merges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit: Option<Vec<EditPayload>>,
}

/// Tool for writing config snapshots.
#[derive(Default)]
pub struct WriteConfigTool;

impl ToolMetadata for WriteConfigTool {
    type SideInfo = AutopilotSideInfo;
    type Output = WriteConfigToolOutput;
    type LlmParams = WriteConfigToolParams;
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::WRITE_CONFIG_TOOL_PARAMS
    }
    fn llm_params_ts_bundle_type_name() -> String {
        "WriteConfigToolParams".to_string()
    }
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::WRITE_CONFIG_TOOL_OUTPUT
    }
    fn output_ts_bundle_type_name() -> String {
        "WriteConfigToolOutput".to_string()
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("write_config")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Write a config snapshot to storage. Returns `{status: \"ok\", hash}` on success \
             or `{status: \"error\", code, message}` when the input is invalid and should be retried \
             with fixes. Autopilot tags are automatically merged into the provided tags.",
        )
    }

    fn strict(&self) -> bool {
        false // Config objects have arbitrary nested structures (tools, gateway)
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Write a config snapshot to storage.",
            "properties": {
                "config": {
                    "type": "object",
                    "description": "The TensorZero config to write. Contains functions, metrics, tools, and gateway settings.",
                    "properties": {
                        "functions": {
                            "type": "object",
                            "description": "Map of function names to function configurations.",
                            "additionalProperties": {
                                "type": "object",
                                "description": "A function configuration with type (chat or json) and variants.",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["chat", "json"],
                                        "description": "Function type: 'chat' for conversational, 'json' for structured output."
                                    },
                                    "variants": {
                                        "type": "object",
                                        "description": "Map of variant names to variant configurations."
                                    }
                                },
                                "additionalProperties": false
                            }
                        },
                        "metrics": {
                            "type": "object",
                            "description": "Map of metric names to metric configurations.",
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["float", "boolean", "comment", "demonstration"],
                                        "description": "Metric type."
                                    },
                                    "optimize": {
                                        "type": "string",
                                        "enum": ["max", "min"],
                                        "description": "Optimization direction for float metrics."
                                    },
                                    "level": {
                                        "type": "string",
                                        "enum": ["inference", "episode"],
                                        "description": "Whether metric applies to individual inferences or episodes."
                                    }
                                },
                                "additionalProperties": false
                            }
                        },
                        "tools": {
                            "type": "object",
                            "description": "Map of tool names to tool configurations."
                        },
                        "gateway": {
                            "type": "object",
                            "description": "Gateway configuration settings."
                        }
                    },
                    "additionalProperties": false
                },
                "extra_templates": {
                    "type": "object",
                    "description": "Map of template paths to template content strings. Used for storing template files with the config.",
                    "additionalProperties": { "type": "string" }
                }
            },
            "required": ["config"],
            "additionalProperties": false
        });

        serde_json::from_value(schema).map_err(|e| {
            NonControlToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }
}

#[async_trait]
impl SimpleTool for WriteConfigTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let config: UninitializedConfig = match serde_json::from_value(llm_params.config) {
            Ok(config) => config,
            Err(e) => {
                return Ok(WriteConfigToolOutput::Error {
                    code: WriteConfigErrorCode::Validation,
                    message: format!("Invalid `config`: {e}"),
                });
            }
        };

        let request = WriteConfigRequest {
            config,
            extra_templates: llm_params.extra_templates,
            tags: side_info.to_tags(),
        };

        let response = ctx
            .client()
            .write_config(request)
            .await
            .map_err(|e| AutopilotToolError::client_error("write_config", e))?;

        Ok(WriteConfigToolOutput::Ok {
            hash: response.hash,
        })
    }
}
