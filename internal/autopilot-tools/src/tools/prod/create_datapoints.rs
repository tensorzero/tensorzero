//! Tool for creating datapoints in a dataset.

use std::borrow::Cow;
use std::collections::HashMap;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::{CreateDatapointRequest, CreateDatapointsResponse};

use autopilot_client::AutopilotSideInfo;

/// Parameters for the create_datapoints tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CreateDatapointsToolParams {
    /// The name of the dataset to create datapoints in.
    pub dataset_name: String,
    /// The datapoints to create. Can be Chat or Json type.
    pub datapoints: Vec<CreateDatapointRequest>,
}

/// Tool for creating datapoints in a dataset.
///
/// This tool creates new datapoints and automatically tags them with
/// autopilot session metadata for tracking.
#[derive(Default)]
pub struct CreateDatapointsTool;

impl ToolMetadata for CreateDatapointsTool {
    type SideInfo = AutopilotSideInfo;
    type Output = CreateDatapointsResponse;
    type LlmParams = CreateDatapointsToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("create_datapoints")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Create datapoints in a dataset. Datapoints can be Chat or Json type. \
             Autopilot tags are automatically added for tracking.",
        )
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Create datapoints in a dataset.",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "The name of the dataset to create datapoints in."
                },
                "datapoints": {
                    "type": "array",
                    "description": "The datapoints to create. Each can be Chat or Json type.",
                    "items": {
                        "type": "object",
                        "description": "A datapoint. Use 'chat' type with 'input' containing messages, or 'json' type with structured input/output.",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["chat", "json"],
                                "description": "The datapoint type."
                            },
                            "function_name": {
                                "type": "string",
                                "description": "The function name this datapoint is for."
                            },
                            "input": {
                                "type": "object",
                                "description": "The input data. For Chat: {system?, messages}. For Json: any JSON object."
                            },
                            "output": {
                                "description": "Expected output. For 'chat': array of content blocks like [{\"type\": \"text\", \"text\": \"...\"}]. For 'json': object with 'raw' field containing JSON string, e.g. {\"raw\": \"{\\\"key\\\": \\\"value\\\"}\"}"
                            },
                            "output_schema": {
                                "type": "object",
                                "description": "JSON Schema for validating the output. Only used for 'json' type datapoints."
                            },
                            "tags": {
                                "type": "object",
                                "additionalProperties": { "type": "string" },
                                "description": "Optional tags for the datapoint."
                            }
                        },
                        "required": ["type", "function_name", "input"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["dataset_name", "datapoints"],
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

/// Merge autopilot tags into existing tags, preserving user-provided tags.
fn merge_tags(
    existing: Option<HashMap<String, String>>,
    autopilot_tags: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut merged = autopilot_tags.clone();
    if let Some(existing) = existing {
        // User-provided tags take precedence over autopilot tags
        merged.extend(existing);
    }
    merged
}

/// Add autopilot tags to a datapoint request.
fn add_tags_to_datapoint(
    datapoint: CreateDatapointRequest,
    autopilot_tags: &HashMap<String, String>,
) -> CreateDatapointRequest {
    match datapoint {
        CreateDatapointRequest::Chat(mut chat) => {
            chat.tags = Some(merge_tags(chat.tags, autopilot_tags));
            CreateDatapointRequest::Chat(chat)
        }
        CreateDatapointRequest::Json(mut json) => {
            json.tags = Some(merge_tags(json.tags, autopilot_tags));
            CreateDatapointRequest::Json(json)
        }
    }
}

#[async_trait]
impl SimpleTool for CreateDatapointsTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let autopilot_tags = side_info.to_tags();

        // Add autopilot tags to each datapoint
        let datapoints: Vec<CreateDatapointRequest> = llm_params
            .datapoints
            .into_iter()
            .map(|dp| add_tags_to_datapoint(dp, &autopilot_tags))
            .collect();

        ctx.client()
            .create_datapoints(llm_params.dataset_name, datapoints)
            .await
            .map_err(|e| AutopilotToolError::client_error("create_datapoints", e).into())
    }
}
