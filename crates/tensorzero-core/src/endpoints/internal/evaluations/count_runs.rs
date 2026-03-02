//! Handler for counting total evaluation runs.

use axum::Json;
use axum::extract::State;
use tracing::instrument;

use super::types::EvaluationRunStatsResponse;
use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/evaluations/runs/count`
///
/// Returns the total count of unique evaluation runs across all functions.
#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/internal/evaluations/runs/count",
    responses(
        (status = 200, description = "Evaluation runs count", body = EvaluationRunStatsResponse),
        (status = 400, description = "Bad request"),
    ),
    tag = "Internal"
))]
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.count_evaluation_runs", skip_all)]
pub async fn count_evaluation_runs_handler(
    State(app_state): AppState,
) -> Result<Json<EvaluationRunStatsResponse>, Error> {
    let database = app_state.get_delegating_database();
    let count = database.count_total_evaluation_runs().await?;

    Ok(Json(EvaluationRunStatsResponse { count }))
}
