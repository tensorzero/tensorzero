use rmcp::schemars;
use uuid::Uuid;

use tensorzero_core::db::inferences::InferenceOutputSource;
use tensorzero_core::endpoints::stored_inferences::v1::types::GetInferencesRequest;

/// Parameters for the `get_inferences` MCP tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetInferencesParams {
    /// List of inference IDs to retrieve.
    #[schemars(description = "List of inference IDs to retrieve")]
    pub ids: Vec<Uuid>,

    /// Optional function name filter (improves query performance).
    #[schemars(description = "Optional function name filter (improves query performance)")]
    pub function_name: Option<String>,

    /// Source of the inference output: "inference" (default), "demonstration", or "none".
    #[schemars(description = "Output source: 'inference' (default), 'demonstration', or 'none'")]
    pub output_source: Option<InferenceOutputSource>,
}

impl From<GetInferencesParams> for GetInferencesRequest {
    fn from(params: GetInferencesParams) -> Self {
        Self {
            ids: params.ids,
            function_name: params.function_name,
            output_source: params.output_source.unwrap_or_default(),
        }
    }
}
