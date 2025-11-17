use axum::extract::{Query, State};
use axum::Json;
use tracing::instrument;

use crate::db::inferences::{GetInferenceBoundsParams, InferenceQueries};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

use super::types::{GetInferenceBoundsQueryParams, GetInferenceBoundsResponse};

/// Handler for the GET `/internal/v1/inferences/bounds` endpoint.
/// Returns the bounds (min/max IDs) and count of inferences matching the filter criteria.
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
        first_id: bounds.first_id,
        last_id: bounds.last_id,
        count: bounds.count,
    };

    Ok(Json(response))
}
