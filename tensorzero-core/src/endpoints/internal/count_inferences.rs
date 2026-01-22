//! Count inferences endpoint for getting the count of inferences matching query parameters.

use axum::extract::State;
use axum::{Json, debug_handler};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::clickhouse::query_builder::{DemonstrationFeedbackFilter, InferenceFilter};
use crate::db::inferences::{CountInferencesParams, InferenceOutputSource, InferenceQueries};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Request to count inferences matching the given parameters.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct CountInferencesRequest {
    /// Optional function name to filter inferences by.
    pub function_name: Option<String>,

    /// Optional variant name to filter inferences by.
    pub variant_name: Option<String>,

    /// Optional episode ID to filter inferences by.
    pub episode_id: Option<Uuid>,

    /// Source of the inference output. When set to "demonstration", only inferences
    /// with demonstration feedback will be counted.
    #[serde(default)]
    pub output_source: InferenceOutputSource,

    /// Optional filter to apply when counting inferences.
    pub filters: Option<InferenceFilter>,

    /// Experimental: search query to filter inferences by.
    pub search_query_experimental: Option<String>,
}

/// Response containing the count of matching inferences.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CountInferencesResponse {
    /// The count of inferences matching the query parameters.
    pub count: u64,
}

/// Counts inferences matching the given parameters.
pub async fn count_inferences(
    clickhouse: &impl InferenceQueries,
    config: &crate::config::Config,
    request: &CountInferencesRequest,
) -> Result<CountInferencesResponse, Error> {
    // If output_source is Demonstration, add a has_demonstration filter
    let effective_filters = match request.output_source {
        InferenceOutputSource::Demonstration => {
            let demo_filter = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
                has_demonstration: true,
            });

            match &request.filters {
                Some(existing_filters) => Some(InferenceFilter::And {
                    children: vec![existing_filters.clone(), demo_filter],
                }),
                None => Some(demo_filter),
            }
        }
        _ => request.filters.clone(),
    };

    let params = CountInferencesParams {
        function_name: request.function_name.as_deref(),
        variant_name: request.variant_name.as_deref(),
        episode_id: request.episode_id.as_ref(),
        filters: effective_filters.as_ref(),
        search_query_experimental: request.search_query_experimental.as_deref(),
    };

    let count = clickhouse.count_inferences(config, &params).await?;

    Ok(CountInferencesResponse { count })
}

/// HTTP handler for the count inferences endpoint.
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "count_inferences_handler",
    skip_all,
    fields(
        function_name = ?request.function_name,
        variant_name = ?request.variant_name,
        output_source = ?request.output_source,
    )
)]
pub async fn count_inferences_handler(
    State(app_state): AppState,
    Json(request): Json<CountInferencesRequest>,
) -> Result<Json<CountInferencesResponse>, Error> {
    let response = count_inferences(
        &app_state.clickhouse_connection_info,
        &app_state.config,
        &request,
    )
    .await?;

    Ok(Json(response))
}
