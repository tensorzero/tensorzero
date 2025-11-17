use axum::extract::{Query, State};
use axum::Json;
use tracing::instrument;

use crate::db::clickhouse::inference_queries::{get_inference_bounds, GetInferenceBoundsParams};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

use super::types::{GetInferenceBoundsParams as QueryParams, GetInferenceBoundsResponse};

/// Handler for the GET `/internal/v1/inferences/bounds` endpoint.
/// Returns the bounds (min/max IDs) and count of inferences matching the filter criteria.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "inferences.v1.get_bounds", skip(app_state))]
pub async fn get_inference_bounds_handler(
    State(app_state): AppState,
    Query(query_params): Query<QueryParams>,
) -> Result<Json<GetInferenceBoundsResponse>, Error> {
    let params = GetInferenceBoundsParams {
        function_name: query_params.function_name,
        variant_name: query_params.variant_name,
        episode_id: query_params.episode_id,
        limit: query_params.limit,
        before: query_params.before,
        after: query_params.after,
    };

    let response = get_inference_bounds(&app_state.clickhouse_connection_info, params).await?;

    Ok(Json(response))
}
