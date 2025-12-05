use axum::extract::{Path, Query, State};
use axum::{debug_handler, Json};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::feedback::{FeedbackQueries, MetricFeedbackRow};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData};

#[derive(Debug, Deserialize, Serialize)]
pub struct GetFeedbackQueryParams {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetFeedbackResponse {
    pub feedback: Vec<MetricFeedbackRow>,
}

/// Handler for the GET `/internal/feedback/{metric_name}` endpoint.
/// Returns the latest feedback for a specific metric, grouped by target_id.
#[debug_handler(state = AppStateData)]
#[instrument(name = "feedback.get_feedback", skip(app_state, params))]
pub async fn get_feedback_handler(
    State(app_state): AppState,
    Path(metric_name): Path<String>,
    Query(params): Query<GetFeedbackQueryParams>,
) -> Result<Json<GetFeedbackResponse>, Error> {
    let feedback = get_feedback(app_state, metric_name, params).await?;
    Ok(Json(feedback))
}

/// Core logic for getting feedback by metric name.
/// This is separate from the handler to allow direct calls from the embedded client.
pub async fn get_feedback(
    AppStateData {
        config,
        clickhouse_connection_info,
        ..
    }: AppStateData,
    metric_name: String,
    params: GetFeedbackQueryParams,
) -> Result<GetFeedbackResponse, Error> {
    // Look up the metric in config - returns 404 if not found
    let metric_config = config.get_metric_or_err(&metric_name)?;

    let limit = params.limit.unwrap_or(100).min(1000);
    let offset = params.offset.unwrap_or(0);

    let feedback = clickhouse_connection_info
        .get_feedback_by_metric(&metric_name, metric_config.r#type, limit, offset)
        .await?;

    Ok(GetFeedbackResponse { feedback })
}
