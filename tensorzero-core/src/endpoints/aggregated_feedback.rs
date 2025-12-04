use axum::extract::{Path, Query, State};
use axum::{debug_handler, Json};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::feedback::{AggregatedFeedbackByVariant, FeedbackQueries};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the aggregated feedback endpoint
#[derive(Debug, Deserialize)]
pub struct GetAggregatedFeedbackQueryParams {
    /// Optional variant name filter. If not provided, returns stats for all variants.
    pub variant_name: Option<String>,
    /// Optional metric name filter. If not provided, groups results by metric_name.
    pub metric_name: Option<String>,
}

/// Response containing aggregated feedback statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct GetAggregatedFeedbackResponse {
    pub feedback: Vec<AggregatedFeedbackByVariant>,
}

/// HTTP handler for the aggregated feedback endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "internal.get_aggregated_feedback",
    skip(app_state, params),
    fields(function_name = %function_name)
)]
pub async fn get_aggregated_feedback_handler(
    State(app_state): AppState,
    Path(function_name): Path<String>,
    Query(params): Query<GetAggregatedFeedbackQueryParams>,
) -> Result<Json<GetAggregatedFeedbackResponse>, Error> {
    let feedback = get_aggregated_feedback(
        &app_state.clickhouse_connection_info,
        &function_name,
        params.variant_name.as_deref(),
        params.metric_name.as_deref(),
    )
    .await?;

    Ok(Json(GetAggregatedFeedbackResponse { feedback }))
}

/// Core business logic for getting aggregated feedback statistics
pub async fn get_aggregated_feedback(
    clickhouse: &impl FeedbackQueries,
    function_name: &str,
    variant_name: Option<&str>,
    metric_name: Option<&str>,
) -> Result<Vec<AggregatedFeedbackByVariant>, Error> {
    clickhouse
        .get_aggregated_feedback_by_variant(function_name, variant_name, metric_name)
        .await
}
