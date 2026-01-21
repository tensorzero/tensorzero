//! Tool for getting specific inferences by their IDs.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::Schema;
use tensorzero::{GetInferencesRequest, GetInferencesResponse};

use autopilot_client::AutopilotSideInfo;

/// Tool for getting specific inferences by their IDs.
///
/// Retrieves the full inference data for the given IDs, including input,
/// output, and metadata. Use this when you have known inference IDs to fetch.
#[derive(Default)]
pub struct GetInferencesTool;

impl ToolMetadata for GetInferencesTool {
    type SideInfo = AutopilotSideInfo;
    type Output = GetInferencesResponse;
    type LlmParams = GetInferencesRequest;

    fn name() -> Cow<'static, str> {
        Cow::Borrowed("get_inferences")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed(
            "Retrieves specific inferences by their IDs. \
             Use this when you have known inference IDs to fetch. \
             Returns the full inference data including input, output, and metadata.",
        )
    }

    fn parameters_schema() -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Get specific inferences by their IDs.",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": { "type": "string", "format": "uuid" },
                    "description": "List of inference IDs to retrieve. Required."
                },
                "function_name": {
                    "type": "string",
                    "description": "Optional function name to filter by. Including this improves query performance."
                },
                "output_source": {
                    "type": "string",
                    "enum": ["none", "inference", "demonstration"],
                    "default": "inference",
                    "description": "Source for the output field: 'none' (no output), 'inference' (original inference output), or 'demonstration' (demonstration feedback if available)."
                }
            },
            "required": ["ids"]
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
impl SimpleTool for GetInferencesTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .get_inferences(llm_params)
            .await
            .map_err(|e| AutopilotToolError::client_error("get_inferences", e).into())
    }
}
