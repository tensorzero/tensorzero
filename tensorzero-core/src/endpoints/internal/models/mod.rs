//! Model statistics endpoints.
//!
//! These endpoints provide model-related statistics for the UI.

pub mod types;

pub use types::{CountModelsResponse, GetModelLatencyResponse, GetModelUsageResponse};

use axum::Json;
use axum::extract::{Query, State};
use tracing::instrument;

use crate::db::SelectQueries;
use crate::endpoints::internal::models::types::{
    GetModelLatencyQueryParams, GetModelUsageQueryParams,
};
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

/// Handler for `GET /internal/models/usage`
///
/// Returns model usage timeseries data (tokens, counts over time).
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "models.usage", skip_all)]
pub async fn get_model_usage_handler(
    State(app_state): AppState,
    Query(params): Query<GetModelUsageQueryParams>,
) -> Result<Json<GetModelUsageResponse>, Error> {
    let data = app_state
        .clickhouse_connection_info
        .get_model_usage_timeseries(params.time_window, params.max_periods)
        .await?;

    Ok(Json(GetModelUsageResponse { data }))
}

/// Handler for `GET /internal/models/latency`
///
/// Returns model latency quantile distributions.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "models.latency", skip_all)]
pub async fn get_model_latency_handler(
    State(app_state): AppState,
    Query(params): Query<GetModelLatencyQueryParams>,
) -> Result<Json<GetModelLatencyResponse>, Error> {
    let data = app_state
        .clickhouse_connection_info
        .get_model_latency_quantiles(params.time_window)
        .await?;

    Ok(Json(GetModelLatencyResponse { data }))
}
