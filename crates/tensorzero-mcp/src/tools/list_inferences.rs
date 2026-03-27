use rmcp::schemars;
use uuid::Uuid;

use tensorzero_core::db::inferences::InferenceOutputSource;
use tensorzero_core::endpoints::stored_inferences::v1::types::{
    InferenceFilter, ListInferencesRequest, OrderBy,
};

/// Parameters for the `list_inferences` MCP tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ListInferencesParams {
    /// Filter by function name.
    #[schemars(description = "Filter inferences by function name")]
    pub function_name: Option<String>,

    /// Filter by variant name.
    #[schemars(description = "Filter inferences by variant name")]
    pub variant_name: Option<String>,

    /// Filter by episode ID.
    #[schemars(description = "Filter inferences by episode ID")]
    pub episode_id: Option<Uuid>,

    /// Source of the inference output: "inference" (default), "demonstration", or "none".
    #[schemars(description = "Output source: 'inference' (default), 'demonstration', or 'none'")]
    pub output_source: Option<InferenceOutputSource>,

    /// Maximum number of inferences to return (default: 20).
    #[schemars(description = "Maximum number of inferences to return (default: 20)")]
    pub limit: Option<u32>,

    /// Number of inferences to skip.
    #[schemars(description = "Number of inferences to skip")]
    pub offset: Option<u32>,

    /// Paginate before this inference ID (exclusive, earlier in time).
    #[schemars(description = "Paginate before this inference ID (exclusive)")]
    pub before: Option<Uuid>,

    /// Paginate after this inference ID (exclusive, later in time).
    #[schemars(description = "Paginate after this inference ID (exclusive)")]
    pub after: Option<Uuid>,

    /// Filter expression for metrics, tags, time, and logical combinations.
    #[schemars(description = "Filter expression (metrics, tags, time, AND/OR/NOT)")]
    pub filters: Option<InferenceFilter>,

    /// Ordering criteria.
    #[schemars(description = "Sort criteria (e.g., by timestamp or metric)")]
    pub order_by: Option<Vec<OrderBy>>,
}

impl From<ListInferencesParams> for ListInferencesRequest {
    #[expect(deprecated)]
    fn from(params: ListInferencesParams) -> Self {
        Self {
            function_name: params.function_name,
            variant_name: params.variant_name,
            episode_id: params.episode_id,
            output_source: params.output_source.unwrap_or_default(),
            limit: params.limit,
            offset: params.offset,
            before: params.before,
            after: params.after,
            filters: params.filters,
            filter: None,
            order_by: params.order_by,
            search_query_experimental: None,
        }
    }
}
