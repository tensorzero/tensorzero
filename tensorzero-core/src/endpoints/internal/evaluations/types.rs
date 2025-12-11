//! Request and response types for evaluation endpoints.

use serde::{Deserialize, Serialize};

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
