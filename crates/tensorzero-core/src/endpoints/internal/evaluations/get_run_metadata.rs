//! Handler for getting evaluation run metadata (function_name, function_type, metrics)
//! from the database, without requiring an evaluation config.

use std::collections::HashMap;

use axum::Json;
use axum::extract::{Query, State};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::config::MetricConfigOptimize;
use crate::db::evaluation_queries::EvaluationQueries;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, SwappableAppStateData};

/// Query parameters for getting evaluation run metadata.
#[derive(Debug, Deserialize)]
pub struct GetRunMetadataParams {
    pub evaluation_run_ids: String,
}

/// Metric metadata from an evaluation run.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct RunMetricMetadata {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluator_name: Option<String>,
    /// `boolean` or `float`
    pub value_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimize: Option<MetricConfigOptimize>,
}

/// Metadata for a single evaluation run.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export, optional_fields)]
pub struct EvaluationRunMetadata {
    pub evaluation_name: String,
    pub function_name: String,
    /// `chat` or `json`
    pub function_type: String,
    pub metrics: Vec<RunMetricMetadata>,
}

/// Response containing evaluation run metadata, keyed by run ID.
#[derive(ts_rs::TS, Debug, Serialize, Deserialize)]
#[ts(export)]
pub struct GetEvaluationRunMetadataResponse {
    pub metadata: HashMap<Uuid, EvaluationRunMetadata>,
}

/// Handler for `GET /internal/evaluations/run_metadata`
///
/// Returns metadata for one or more evaluation runs, including the evaluation name,
/// function name, function type, and metric definitions.
#[axum::debug_handler(state = SwappableAppStateData)]
#[instrument(name = "evaluations.get_run_metadata", skip_all)]
pub async fn get_run_metadata_handler(
    State(app_state): AppState,
    Query(params): Query<GetRunMetadataParams>,
) -> Result<Json<GetEvaluationRunMetadataResponse>, Error> {
    let evaluation_run_ids: Vec<Uuid> = params
        .evaluation_run_ids
        .split(',')
        .map(|s| {
            s.trim().parse::<Uuid>().map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Invalid evaluation run ID `{s}`: {e}"),
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if evaluation_run_ids.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "`evaluation_run_ids` must not be empty".to_string(),
        }));
    }

    let database = app_state.get_delegating_database();

    let results = database
        .get_inference_evaluation_run_metadata(&evaluation_run_ids)
        .await?;

    let metadata = results
        .into_iter()
        .map(|(run_id, m)| {
            let metrics = m
                .metrics
                .into_iter()
                .map(|metric| RunMetricMetadata {
                    name: metric.name,
                    evaluator_name: metric.evaluator_name,
                    value_type: metric.value_type,
                    optimize: metric.optimize,
                })
                .collect();
            (
                run_id,
                EvaluationRunMetadata {
                    evaluation_name: m.evaluation_name,
                    function_name: m.function_name,
                    function_type: m.function_type.as_str().to_string(),
                    metrics,
                },
            )
        })
        .collect();

    Ok(Json(GetEvaluationRunMetadataResponse { metadata }))
}
