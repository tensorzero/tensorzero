//! Request and response types for evaluation endpoints.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// =============================================================================
// Count Evaluation Runs
// =============================================================================

/// Response containing the count of evaluation runs.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
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
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct ListEvaluationRunsResponse {
    pub runs: Vec<EvaluationRunInfo>,
}

/// Information about a single evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
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
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct DatapointStatsResponse {
    pub count: u64,
}
