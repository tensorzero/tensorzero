//! Tool for getting variant-level usage and cost statistics for a function.

use std::borrow::Cow;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, SimpleTool, SimpleToolContext, ToolMetadata, ToolResult};

use crate::error::AutopilotToolError;
use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use tensorzero_core::db::variant_statistics::GetVariantStatisticsResponse;

use autopilot_client::AutopilotSideInfo;

/// Parameters for the get_variant_statistics tool (visible to LLM).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetVariantStatisticsToolParams {
    /// The name of the function to query statistics for.
    pub function_name: String,
    /// Optional filter for specific variants. If not provided, all variants are included.
    #[serde(default)]
    pub variant_names: Option<Vec<String>>,
    /// Optional lower bound on the time window (inclusive, RFC 3339 format).
    #[serde(default)]
    pub after: Option<String>,
}

/// Tool for getting variant-level usage and cost statistics for a function.
///
/// Returns inference count, token usage, cost, and latency quantiles
/// (ClickHouse only) grouped by variant.
#[derive(Default)]
pub struct GetVariantStatisticsTool;

impl ToolMetadata for GetVariantStatisticsTool {
    type SideInfo = AutopilotSideInfo;
    type Output = GetVariantStatisticsResponse;
    type LlmParams = GetVariantStatisticsToolParams;

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GET_VARIANT_STATISTICS_TOOL_PARAMS
    }

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle_type_name() -> String {
        "GetVariantStatisticsToolParams".to_string()
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GET_VARIANT_STATISTICS_RESPONSE
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle_type_name() -> String {
        "GetVariantStatisticsResponse".to_string()
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("get_variant_statistics")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Get aggregated variant statistics (inference count, token usage, cost, and optionally latency quantiles) \
             for a function. Returns statistics grouped by variant name. \
             Optionally filter by specific variant names and a lower time bound.",
        )
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = serde_json::json!({
            "type": "object",
            "description": "Get variant-level usage and cost statistics for a function.",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "The name of the function to query statistics for."
                },
                "variant_names": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional filter for specific variants. If not provided, all variants are included."
                },
                "after": {
                    "type": "string",
                    "description": "Optional lower bound on the time window (inclusive, RFC 3339 format, e.g. '2025-01-01T00:00:00Z')."
                }
            },
            "required": ["function_name"],
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
impl SimpleTool for GetVariantStatisticsTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: <Self as ToolMetadata>::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        ctx.client()
            .get_variant_statistics(
                llm_params.function_name,
                llm_params.variant_names,
                llm_params.after,
            )
            .await
            .map_err(|e| AutopilotToolError::client_error("get_variant_statistics", e).into())
    }
}
