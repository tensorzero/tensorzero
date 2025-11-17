use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use crate::db::inferences::{GetInferenceBoundsParams, InferenceQueries};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

use super::types::GetInferenceBoundsResponse;

/// Query parameters for the inference bounds endpoint.
/// Used by the `GET /internal/inferences/bounds` endpoint.
#[derive(Debug, Deserialize)]
pub struct GetInferenceBoundsQueryParams {
    /// Optional function name to filter inferences by.
    pub function_name: Option<String>,

    /// Optional variant name to filter inferences by.
    pub variant_name: Option<String>,

    /// Optional episode ID to filter inferences by.
    pub episode_id: Option<Uuid>,
}

/// Handler for the GET `/internal/inferences/bounds` endpoint.
/// Returns the bounds (latest and earliest IDs) and count of inferences matching the filter criteria.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "inferences.v1.get_bounds", skip(app_state))]
pub async fn get_inference_bounds_handler(
    State(app_state): AppState,
    Query(query_params): Query<GetInferenceBoundsQueryParams>,
) -> Result<Json<GetInferenceBoundsResponse>, Error> {
    let params = GetInferenceBoundsParams {
        function_name: query_params.function_name,
        variant_name: query_params.variant_name,
        episode_id: query_params.episode_id,
    };

    let bounds = app_state
        .clickhouse_connection_info
        .get_inference_bounds(params)
        .await?;

    // Convert InferenceBounds to GetInferenceBoundsResponse
    let response = GetInferenceBoundsResponse {
        latest_id: bounds.latest_id,
        earliest_id: bounds.earliest_id,
        count: bounds.count,
    };

    Ok(Json(response))
}
