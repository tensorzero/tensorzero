//! Tool for creating datapoints from existing inferences.

use std::borrow::Cow;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::{CreateDatapointsFromInferenceRequestParams, CreateDatapointsResponse};

/// Parameters for the create_datapoints_from_inferences tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CreateDatapointsFromInferencesToolParams {
    /// The name of the dataset to create datapoints in.
    pub dataset_name: String,
    /// Parameters specifying which inferences to create datapoints from.
    /// Can be either specific inference IDs or a query to find inferences.
    pub params: CreateDatapointsFromInferenceRequestParams,
}

/// Tool for creating datapoints from existing inferences.
///
/// This tool creates new datapoints based on existing inferences.
/// Note: Tags are inherited from the source inferences.
#[derive(Default)]
pub struct CreateDatapointsFromInferencesTool;

impl ToolMetadata for CreateDatapointsFromInferencesTool {
    type SideInfo = AutopilotSideInfo;
    type Output = CreateDatapointsResponse;
    type LlmParams = CreateDatapointsFromInferencesToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("create_datapoints_from_inferences")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Create datapoints in a dataset from existing inferences. \
             Specify either specific inference IDs or a query to find inferences. \
             Tags are inherited from the source inferences.",
        )
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Create datapoints from existing inferences.",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "The name of the dataset to create datapoints in."
                },
                "params": {
                    "description": "Parameters specifying which inferences to use. Use 'inference_ids' mode to specify exact IDs, or 'inference_query' mode to query by function name.",
                    "anyOf": [
                        {
                            "type": "object",
                            "description": "Create datapoints from specific inference IDs.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["inference_ids"],
                                    "description": "Mode selector - must be 'inference_ids' for this variant."
                                },
                                "inference_ids": {
                                    "type": "array",
                                    "items": { "type": "string", "format": "uuid" },
                                    "description": "The inference IDs to create datapoints from."
                                },
                                "output_source": {
                                    "type": "string",
                                    "enum": ["none", "inference", "demonstration"],
                                    "description": "Source for datapoint output: 'none' (input-only), 'inference' (original output, default), or 'demonstration' (use demonstration feedback)."
                                }
                            },
                            "required": ["type", "inference_ids"],
                            "additionalProperties": false
                        },
                        {
                            "type": "object",
                            "description": "Create datapoints from inferences matching a query.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["inference_query"],
                                    "description": "Mode selector - must be 'inference_query' for this variant."
                                },
                                "function_name": {
                                    "type": "string",
                                    "description": "Filter inferences by function name."
                                },
                                "variant_name": {
                                    "type": "string",
                                    "description": "Filter inferences by variant name."
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of inferences to use.",
                                },
                                "output_source": {
                                    "type": "string",
                                    "enum": ["none", "inference", "demonstration"],
                                    "description": "Source for datapoint output: 'none' (input-only), 'inference' (original output, default), or 'demonstration' (use demonstration feedback)."
                                }
                            },
                            "required": ["type"],
                            "additionalProperties": false
                        }
                    ]
                }
            },
            "required": ["dataset_name", "params"],
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
impl SimpleTool for CreateDatapointsFromInferencesTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .create_datapoints_from_inferences(llm_params.dataset_name, llm_params.params)
            .await
            .map_err(|e| {
                AutopilotToolError::client_error("create_datapoints_from_inferences", e).into()
            })
    }
}
