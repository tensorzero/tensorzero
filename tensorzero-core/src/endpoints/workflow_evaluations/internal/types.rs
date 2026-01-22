//! Request and response types for workflow evaluation endpoints.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// =============================================================================
// Get Workflow Evaluation Projects
// =============================================================================

/// Response containing a list of workflow evaluation projects.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetWorkflowEvaluationProjectsResponse {
    pub projects: Vec<WorkflowEvaluationProject>,
}

/// Information about a single workflow evaluation project.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct WorkflowEvaluationProject {
    pub name: String,
    pub count: u32,
    pub last_updated: DateTime<Utc>,
}

// =============================================================================
// Get Workflow Evaluation Project Count
// =============================================================================

/// Response containing the count of workflow evaluation projects.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetWorkflowEvaluationProjectCountResponse {
    pub count: u32,
}

// =============================================================================
// Search Workflow Evaluation Runs
// =============================================================================

/// Response containing a list of workflow evaluation runs from search.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct SearchWorkflowEvaluationRunsResponse {
    pub runs: Vec<WorkflowEvaluationRun>,
}

/// Information about a single workflow evaluation run.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListWorkflowEvaluationRunsResponse {
    pub runs: Vec<WorkflowEvaluationRunWithEpisodeCount>,
}

/// Information about a single workflow evaluation run with episode count.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CountWorkflowEvaluationRunsResponse {
    pub count: u32,
}

// =============================================================================
// Get Workflow Evaluation Runs
// =============================================================================

/// Response containing a list of workflow evaluation runs by IDs.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetWorkflowEvaluationRunsResponse {
    pub runs: Vec<WorkflowEvaluationRun>,
}

// =============================================================================
// Get Workflow Evaluation Run Statistics
// =============================================================================

/// Statistics for a single metric within a workflow evaluation run.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetWorkflowEvaluationRunStatisticsResponse {
    pub statistics: Vec<WorkflowEvaluationRunStatistics>,
}

// =============================================================================
// List Workflow Evaluation Run Episodes By Task Name
// =============================================================================

/// Response containing lists of workflow evaluation run episodes grouped by task name.
///
/// Each inner Vec contains all episodes that share the same task_name (or NULL task_name).
/// Episodes with NULL task_name are grouped individually.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListWorkflowEvaluationRunEpisodesByTaskNameResponse {
    pub episodes: Vec<Vec<crate::db::workflow_evaluation_queries::GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow>>,
}

// =============================================================================
// Count Workflow Evaluation Run Episode Groups
// =============================================================================

/// Response containing the count of distinct episodes by task_name.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CountWorkflowEvaluationRunEpisodesByTaskNameResponse {
    pub count: u32,
}

// =============================================================================
// Get Workflow Evaluation Run Episodes with Feedback
// =============================================================================

/// Response containing a list of workflow evaluation run episodes with feedback.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GetWorkflowEvaluationRunEpisodesWithFeedbackResponse {
    pub episodes: Vec<WorkflowEvaluationRunEpisodeWithFeedback>,
}

/// Information about a single workflow evaluation run episode with feedback.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct WorkflowEvaluationRunEpisodeWithFeedback {
    pub episode_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub run_id: Uuid,
    pub tags: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_name: Option<String>,
    /// The feedback metric names, sorted alphabetically.
    pub feedback_metric_names: Vec<String>,
    /// The feedback values, corresponding to the metric names.
    pub feedback_values: Vec<String>,
}

// =============================================================================
// Count Workflow Evaluation Run Episodes
// =============================================================================

/// Response containing the count of episodes for a workflow evaluation run.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CountWorkflowEvaluationRunEpisodesResponse {
    pub count: u32,
}
