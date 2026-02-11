//! Tool for listing available datasets.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero::{ListDatasetsRequest, ListDatasetsResponse};

use autopilot_client::AutopilotSideInfo;

/// Parameters for the list_datasets tool (visible to LLM).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ListDatasetsToolParams {
    /// Request parameters for listing datasets (filtering, pagination).
    #[serde(flatten)]
    pub request: ListDatasetsRequest,
}

/// Tool for listing available datasets.
///
/// Returns metadata about datasets including name, datapoint count, and last updated timestamp.
/// Use this to discover what datasets are available before using list_datapoints to explore
/// the datapoints within a specific dataset.
#[derive(Default)]
pub struct ListDatasetsTool;

impl ToolMetadata for ListDatasetsTool {
    type SideInfo = AutopilotSideInfo;
    type Output = ListDatasetsResponse;
    type LlmParams = ListDatasetsToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("list_datasets")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "List available datasets with metadata (name, datapoint count, last updated). \
             Use this to discover datasets before exploring their datapoints with list_datapoints.",
        )
    }

    // NOTE: This schema must be kept in sync with `ListDatasetsRequest` fields.
    // We manually construct it for OpenAI structured outputs compatibility (using anyOf for optional fields).
    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "List available datasets with optional filtering and pagination.",
            "properties": {
                "function_name": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ],
                    "description": "Filter by function name. Only datasets with datapoints for this function will be returned."
                },
                "limit": {
                    "anyOf": [
                        { "type": "integer" },
                        { "type": "null" }
                    ],
                    "description": "Maximum number of datasets to return."
                },
                "offset": {
                    "anyOf": [
                        { "type": "integer" },
                        { "type": "null" }
                    ],
                    "description": "Number of datasets to skip for pagination."
                }
            },
            "required": [],
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
impl SimpleTool for ListDatasetsTool {
    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .list_datasets(llm_params.request)
            .await
            .map_err(|e| AutopilotToolError::client_error("list_datasets", e).into())
    }
}
