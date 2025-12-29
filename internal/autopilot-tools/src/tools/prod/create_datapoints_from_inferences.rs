//! Tool for creating datapoints from existing inferences.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{SimpleTool, SimpleToolContext, ToolError, ToolMetadata, ToolResult};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use tensorzero::{CreateDatapointsFromInferenceRequestParams, CreateDatapointsResponse};

use crate::types::AutopilotToolSideInfo;

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
    type SideInfo = AutopilotToolSideInfo;
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

    fn parameters_schema() -> Schema {
        schema_for!(CreateDatapointsFromInferencesToolParams)
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
