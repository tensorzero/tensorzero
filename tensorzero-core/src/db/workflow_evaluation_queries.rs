//! Workflow evaluation types and trait definitions.

use async_trait::async_trait;

use chrono::{DateTime, Utc};
#[cfg(test)]
use mockall::automock;
use serde::Deserialize;

use crate::error::Error;

/// Database struct for deserializing workflow evaluation project info from ClickHouse.
#[derive(Debug, Deserialize)]
pub struct WorkflowEvaluationProjectRow {
    pub name: String,
    pub count: u32,
    pub last_updated: DateTime<Utc>,
}

/// Trait for workflow evaluation-related queries.
#[async_trait]
#[cfg_attr(test, automock)]
pub trait WorkflowEvaluationQueries {
    /// Lists workflow evaluation projects with pagination.
    async fn list_workflow_evaluation_projects(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<WorkflowEvaluationProjectRow>, Error>;
}
