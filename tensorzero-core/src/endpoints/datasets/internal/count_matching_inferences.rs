//! Endpoint for counting inferences matching dataset query parameters.
//!
//! This endpoint is used by the dataset builder to preview how many inferences
//! would be selected for a dataset based on the filter criteria.

use axum::extract::State;
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::datasets::{DatasetQueries, DatasetQueryParams};
use crate::endpoints::datasets::internal::types::FilterInferencesForDatasetBuilderRequest;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

/// Response containing the count of matching inferences
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct CountMatchingInferencesResponse {
    /// The count of inferences matching the query parameters
    pub count: u32,
}

/// Counts inferences matching the provided query parameters
pub async fn count_matching_inferences(
    clickhouse: &impl DatasetQueries,
    params: DatasetQueryParams,
) -> Result<CountMatchingInferencesResponse, Error> {
    let count = clickhouse.count_rows_for_dataset(&params).await?;

    Ok(CountMatchingInferencesResponse { count })
}

/// HTTP handler for the count matching inferences endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "count_matching_inferences_handler",
    skip_all,
    fields(
        inference_type = ?request.inference_type,
        function_name = ?request.function_name,
    )
)]
pub async fn count_matching_inferences_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<FilterInferencesForDatasetBuilderRequest>,
) -> Result<Json<CountMatchingInferencesResponse>, Error> {
    let params = DatasetQueryParams {
        inference_type: request.inference_type,
        function_name: request.function_name,
        dataset_name: None, // Not needed for counting
        variant_name: request.variant_name,
        extra_where: None,
        extra_params: None,
        metric_filter: request.metric_filter,
        output_source: request.output_source,
        limit: None,
        offset: None,
    };
    let response = count_matching_inferences(&app_state.clickhouse_connection_info, params).await?;

    Ok(Json(response))
}
