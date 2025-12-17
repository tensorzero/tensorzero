//! Request and response types for workflow evaluation endpoints.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// =============================================================================
// Get Workflow Evaluation Projects
// =============================================================================

/// Query parameters for getting workflow evaluation projects.
#[derive(Debug, Deserialize)]
pub struct GetWorkflowEvaluationProjectsParams {
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

fn default_limit() -> u32 {
    100
}

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
