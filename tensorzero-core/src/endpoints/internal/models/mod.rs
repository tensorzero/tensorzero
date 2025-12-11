//! Model statistics endpoints.
//!
//! These endpoints provide model-related statistics for the UI.

pub mod types;

use axum::Json;
use axum::extract::State;
use tracing::instrument;

use crate::db::SelectQueries;
use crate::endpoints::internal::models::types::CountModelsResponse;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Handler for `GET /internal/models/count`
///
/// Returns the count of distinct models that have been used.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "models.count", skip_all)]
pub async fn count_models_handler(
    State(app_state): AppState,
) -> Result<Json<CountModelsResponse>, Error> {
    let count = app_state
        .clickhouse_connection_info
        .count_distinct_models_used()
        .await?;

    Ok(Json(CountModelsResponse { model_count: count }))
}
