//! Request and response types for workflow evaluation endpoints.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// =============================================================================
// Get Workflow Evaluation Projects
// =============================================================================

/// Response containing a list of workflow evaluation projects.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetWorkflowEvaluationProjectsResponse {
    pub projects: Vec<WorkflowEvaluationProject>,
}

/// Information about a single workflow evaluation project.
#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct WorkflowEvaluationProject {
    pub name: String,
    pub count: u32,
    pub last_updated: DateTime<Utc>,
}

// =============================================================================
// Get Workflow Evaluation Project Count
// =============================================================================

/// Response containing the count of workflow evaluation projects.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetWorkflowEvaluationProjectCountResponse {
    pub count: u32,
}

// =============================================================================
// Search Workflow Evaluation Runs
// =============================================================================

/// Response containing a list of workflow evaluation runs from search.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct SearchWorkflowEvaluationRunsResponse {
    pub runs: Vec<WorkflowEvaluationRun>,
}

/// Information about a single workflow evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct WorkflowEvaluationRun {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub id: Uuid,
    pub variant_pins: HashMap<String, String>,
    pub tags: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,
    pub timestamp: DateTime<Utc>,
}

// =============================================================================
// List Workflow Evaluation Runs
// =============================================================================

/// Response containing a list of workflow evaluation runs with episode counts.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct ListWorkflowEvaluationRunsResponse {
    pub runs: Vec<WorkflowEvaluationRunWithEpisodeCount>,
}

/// Information about a single workflow evaluation run with episode count.
#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct WorkflowEvaluationRunWithEpisodeCount {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub id: Uuid,
    pub variant_pins: HashMap<String, String>,
    pub tags: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,
    pub num_episodes: u32,
    pub timestamp: DateTime<Utc>,
}

// =============================================================================
// Count Workflow Evaluation Runs
// =============================================================================

/// Response containing the count of workflow evaluation runs.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct CountWorkflowEvaluationRunsResponse {
    pub count: u32,
}

// =============================================================================
// Get Workflow Evaluation Runs
// =============================================================================

/// Response containing a list of workflow evaluation runs by IDs.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetWorkflowEvaluationRunsResponse {
    pub runs: Vec<WorkflowEvaluationRun>,
}

// =============================================================================
// Get Workflow Evaluation Run Statistics
// =============================================================================

/// Statistics for a single metric within a workflow evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export, optional_fields)]
pub struct WorkflowEvaluationRunStatistics {
    pub metric_name: String,
    pub count: u32,
    pub avg_metric: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdev: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_lower: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_upper: Option<f64>,
}

/// Response containing statistics for a workflow evaluation run grouped by metric.
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct GetWorkflowEvaluationRunStatisticsResponse {
    pub statistics: Vec<WorkflowEvaluationRunStatistics>,
}
