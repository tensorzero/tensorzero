//! Request and response types for evaluation endpoints.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// =============================================================================
// Count Evaluation Runs
// =============================================================================

/// Response containing the count of evaluation runs.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EvaluationRunStatsResponse {
    /// The total count of evaluation runs.
    pub count: u64,
}

// =============================================================================
// List Evaluation Runs
// =============================================================================

/// Query parameters for listing evaluation runs.
#[derive(Debug, Deserialize)]
pub struct ListEvaluationRunsParams {
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

fn default_limit() -> u32 {
    100
}

/// Response containing a list of evaluation runs.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListEvaluationRunsResponse {
    pub runs: Vec<EvaluationRunInfo>,
}

/// Information about a single evaluation run.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct EvaluationRunInfo {
    pub evaluation_run_id: Uuid,
    pub evaluation_name: String,
    pub dataset_name: String,
    pub function_name: String,
    pub variant_name: String,
    pub last_inference_timestamp: DateTime<Utc>,
}

// =============================================================================
// Count Datapoints for Evaluation
// =============================================================================

/// Query parameters for counting datapoints across evaluation runs.
#[derive(Debug, Deserialize)]
pub struct CountDatapointsParams {
    pub function_name: String,
    /// Comma-separated list of evaluation run IDs
    pub evaluation_run_ids: String,
}

/// Response containing the count of unique datapoints.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct DatapointStatsResponse {
    pub count: u64,
}

// =============================================================================
// Search Evaluation Runs
// =============================================================================

/// Query parameters for searching evaluation runs.
#[derive(Debug, Deserialize)]
pub struct SearchEvaluationRunsParams {
    pub evaluation_name: String,
    pub function_name: String,
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

/// Response containing search results for evaluation runs.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct SearchEvaluationRunsResponse {
    pub results: Vec<SearchEvaluationRunResult>,
}

/// A single search result for an evaluation run.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct SearchEvaluationRunResult {
    pub evaluation_run_id: Uuid,
    pub variant_name: String,
}

// =============================================================================
// Get Evaluation Statistics
// =============================================================================

/// Query parameters for getting evaluation statistics.
#[derive(Debug, Deserialize)]
pub struct GetEvaluationStatisticsParams {
    pub function_name: String,
    /// Function type: "chat" or "json"
    pub function_type: String,
    /// Comma-separated list of metric names
    pub metric_names: String,
    /// Comma-separated list of evaluation run IDs
    pub evaluation_run_ids: String,
}

/// Response containing evaluation statistics.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetEvaluationStatisticsResponse {
    pub statistics: Vec<EvaluationStatistics>,
}

/// Statistics for a single evaluation run and metric.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct EvaluationStatistics {
    pub evaluation_run_id: Uuid,
    pub metric_name: String,
    pub datapoint_count: u32,
    pub mean_metric: f64,
    pub ci_lower: Option<f64>,
    pub ci_upper: Option<f64>,
}
