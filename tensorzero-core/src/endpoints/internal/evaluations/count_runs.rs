//! Handler for counting total evaluation runs.

use axum::Json;
use axum::extract::State;
use tracing::instrument;

use super::types::EvaluationRunStatsResponse;
use crate::db::delegating_connection::DelegatingDatabaseConnection;
use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/evaluations/runs/count`
///
/// Returns the total count of unique evaluation runs across all functions.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "evaluations.count_evaluation_runs", skip_all)]
pub async fn count_evaluation_runs_handler(
    State(app_state): AppState,
) -> Result<Json<EvaluationRunStatsResponse>, Error> {
    let database = DelegatingDatabaseConnection::new(
        app_state.clickhouse_connection_info.clone(),
        app_state.postgres_connection_info.clone(),
    );
    let count = database.count_total_evaluation_runs().await?;

    Ok(Json(EvaluationRunStatsResponse { count }))
}
