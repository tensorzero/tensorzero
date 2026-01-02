//! Tool for creating datapoints from existing inferences.

use std::borrow::Cow;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
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

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("create_datapoints_from_inferences")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Create datapoints in a dataset from existing inferences. \
             Specify either specific inference IDs or a query to find inferences. \
             Tags are inherited from the source inferences.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Create datapoints from existing inferences.",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "The name of the dataset to create datapoints in."
                },
                "params": {
                    "type": "object",
                    "description": "Parameters specifying which inferences to use.",
                    "properties": {
                        "inference_ids": {
                            "type": "array",
                            "items": { "type": "string", "format": "uuid" },
                            "description": "Specific inference IDs to create datapoints from (use this OR query parameters)."
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Filter inferences by function name (for query mode)."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of inferences to use (for query mode)."
                        }
                    }
                }
            },
            "required": ["dataset_name", "params"]
        });

        serde_json::from_value(schema).map_err(|e| ToolError::SchemaGeneration(e.into()))
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
            .map_err(|e| ToolError::ExecutionFailed(e.into()))
    }
}
