//! Variant statistics endpoint.
//!
//! Provides aggregated usage and cost statistics per variant for a function.

use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;

use crate::db::variant_statistics::{
    GetVariantStatisticsParams, GetVariantStatisticsResponse, VariantStatisticsQueries,
};
use crate::error::Error;
use crate::utils::gateway::AppState;

/// Handler for `GET /internal/variant_statistics`
///
/// Returns aggregated variant statistics for a function, optionally filtered
/// by variant names and a lower time bound.
#[instrument(name = "variant_statistics", skip_all)]
pub async fn get_variant_statistics_handler(
    State(app_state): AppState,
    Query(params): Query<GetVariantStatisticsParams>,
) -> Result<Json<GetVariantStatisticsResponse>, Error> {
    let database = app_state.get_delegating_database();

    let quantiles = database
        .get_variant_statistics_quantiles()
        .map(|q| q.to_vec());
    let data = database.get_variant_statistics(&params).await?;

    Ok(Json(GetVariantStatisticsResponse { quantiles, data }))
}
