//! Model statistics endpoints.
//!
//! These endpoints provide model-related statistics for the UI.

pub mod types;

pub use types::{
    CountModelsResponse, GetCacheStatisticsResponse, GetModelLatencyResponse,
    GetModelUsageResponse, GetVariantUsageResponse,
};

use axum::Json;
use axum::extract::{Path, Query, State};
use tracing::instrument;

use crate::db::model_inferences::ModelInferenceQueries;
use crate::endpoints::internal::models::types::{
    GetCacheStatisticsQueryParams, GetModelLatencyQueryParams, GetModelUsageQueryParams,
    GetVariantUsageQueryParams,
};
use crate::error::Error;
use crate::utils::gateway::{AppState, SwappableAppStateData};

/// Handler for `GET /internal/models/count`
///
/// Returns the count of distinct models that have been used.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "models.count", skip_all)]
pub async fn count_models_handler(
    State(app_state): AppState,
) -> Result<Json<CountModelsResponse>, Error> {
    let database = app_state.get_delegating_database();

    let count = database.count_distinct_models_used().await?;

    Ok(Json(CountModelsResponse { model_count: count }))
}

/// Handler for `GET /internal/models/usage`
///
/// Returns model usage timeseries data (tokens, counts over time).
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "models.usage", skip_all)]
pub async fn get_model_usage_handler(
    State(app_state): AppState,
    Query(params): Query<GetModelUsageQueryParams>,
) -> Result<Json<GetModelUsageResponse>, Error> {
    let database = app_state.get_delegating_database();

    let data = database
        .get_model_usage_timeseries(params.time_window, params.max_periods)
        .await?;

    Ok(Json(GetModelUsageResponse { data }))
}

/// Handler for `GET /internal/models/latency`
///
/// Returns model latency quantile distributions.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "models.latency", skip_all)]
pub async fn get_model_latency_handler(
    State(app_state): AppState,
    Query(params): Query<GetModelLatencyQueryParams>,
) -> Result<Json<GetModelLatencyResponse>, Error> {
    let database = app_state.get_delegating_database();

    let quantiles = database
        .get_model_latency_quantile_function_inputs()
        .to_vec();
    let data = database
        .get_model_latency_quantiles(params.time_window)
        .await?;

    Ok(Json(GetModelLatencyResponse { quantiles, data }))
}

/// Handler for `GET /internal/models/cache_statistics`
///
/// Returns cache statistics timeseries data grouped by model and provider.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "models.cache_statistics", skip_all)]
pub async fn get_cache_statistics_handler(
    State(app_state): AppState,
    Query(params): Query<GetCacheStatisticsQueryParams>,
) -> Result<Json<GetCacheStatisticsResponse>, Error> {
    let database = app_state.get_delegating_database();

    let data = database
        .get_cache_statistics_timeseries(
            params.time_window,
            params.max_periods,
            params.model_name.as_deref(),
            params.model_provider_name.as_deref(),
        )
        .await?;

    Ok(Json(GetCacheStatisticsResponse { data }))
}

/// Handler for `GET /internal/functions/{function_name}/variant_usage`
///
/// Returns variant usage timeseries data (tokens, costs, counts over time) for a function.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "variants.usage", skip_all, fields(function_name = %function_name))]
pub async fn get_variant_usage_handler(
    State(app_state): AppState,
    Path(function_name): Path<String>,
    Query(params): Query<GetVariantUsageQueryParams>,
) -> Result<Json<GetVariantUsageResponse>, Error> {
    let database = app_state.get_delegating_database();

    let data = database
        .get_variant_usage_timeseries(&function_name, params.time_window, params.max_periods)
        .await?;

    Ok(Json(GetVariantUsageResponse { data }))
}
