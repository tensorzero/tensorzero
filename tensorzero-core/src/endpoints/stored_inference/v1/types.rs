use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::clickhouse::query_builder::{InferenceFilter, OrderBy};
use crate::db::inferences::InferenceOutputSource;
use crate::stored_inference::StoredInference;

/// Request to list inferences with pagination and filters.
/// Used by the `POST /v1/inferences/list_inferences` endpoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct ListInferencesRequest {
    /// Optional function name to filter inferences by.
    /// If provided, only inferences from this function will be returned.
    pub function_name: Option<String>,

    /// Optional variant name to filter inferences by.
    /// If provided, only inferences from this variant will be returned.
    pub variant_name: Option<String>,

    /// Optional episode ID to filter inferences by.
    /// If provided, only inferences from this episode will be returned.
    pub episode_id: Option<Uuid>,

    /// Source of the inference output. Determines whether to return the original
    /// inference output or demonstration feedback (manually-curated output) if available.
    pub output_source: InferenceOutputSource,

    /// The maximum number of inferences to return.
    /// Defaults to 20.
    pub page_size: Option<u32>,

    /// The number of inferences to skip before starting to return results.
    /// Defaults to 0.
    pub offset: Option<u32>,

    /// Optional filter to apply when querying inferences.
    /// Supports filtering by metrics, tags, time, and logical combinations (AND/OR/NOT).
    pub filter: Option<InferenceFilter>,

    /// Optional ordering criteria for the results.
    /// Supports multiple sort criteria (e.g., sort by timestamp then by metric).
    pub order_by: Option<Vec<OrderBy>>,
}

/// Request to get specific inferences by their IDs.
/// Used by the `POST /v1/inferences/get_inferences` endpoint.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GetInferencesRequest {
    /// The IDs of the inferences to retrieve. Required.
    pub ids: Vec<Uuid>,
}

/// Response containing the requested inferences.
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GetInferencesResponse {
    /// The retrieved inferences.
    pub inferences: Vec<StoredInference>,
}
