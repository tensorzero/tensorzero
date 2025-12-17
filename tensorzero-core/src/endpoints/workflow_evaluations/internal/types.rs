//! Request and response types for workflow evaluation endpoints.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
