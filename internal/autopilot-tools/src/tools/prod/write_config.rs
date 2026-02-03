//! Tool for writing config snapshots.

use std::borrow::Cow;
use std::collections::HashMap;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero::{WriteConfigRequest, WriteConfigResponse};
use tensorzero_core::config::UninitializedConfig;

use autopilot_client::AutopilotSideInfo;

// Re-export EditPayload types from tensorzero-config-writer
pub use config_writer::{
    EditPayload, UpsertEvaluationPayload, UpsertEvaluatorPayload, UpsertExperimentationPayload,
    UpsertVariantPayload,
};

/// Parameters for the write_config tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct WriteConfigToolParams {
    /// The config to write as a JSON object.
    pub config: Value,
    /// Templates that should be stored with the config.
    #[serde(default)]
    pub extra_templates: HashMap<String, String>,
    /// We could have consolidated an array of server-side edits into one client-side edit, so this type contains a Vec
    /// Unset means an older API. This should always be set and we should make it mandatory once upstream merges.
    pub edit: Option<Vec<EditPayload>>,
}

/// Tool for writing config snapshots.
#[derive(Default)]
pub struct WriteConfigTool;

impl ToolMetadata for WriteConfigTool {
    type SideInfo = AutopilotSideInfo;
    type Output = WriteConfigResponse;
    type LlmParams = WriteConfigToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("write_config")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Write a config snapshot to storage and return its hash. \
             Autopilot tags are automatically merged into the provided tags.",
        )
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
                                }
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
                                }
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
                    }
                },
                "extra_templates": {
                    "type": "object",
                    "description": "Map of template paths to template content strings. Used for storing template files with the config.",
                    "additionalProperties": { "type": "string" }
                }
            },
            "required": ["config"]
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
        let config: UninitializedConfig = serde_json::from_value(llm_params.config)
            .map_err(|e| AutopilotToolError::validation(format!("Invalid `config`: {e}")))?;

        let request = WriteConfigRequest {
            config,
            extra_templates: llm_params.extra_templates,
            tags: side_info.to_tags(),
        };

        ctx.client()
            .write_config(request)
            .await
            .map_err(|e| AutopilotToolError::client_error("write_config", e).into())
    }
}
