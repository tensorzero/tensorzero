//! Workflow evaluation types and trait definitions.

use std::collections::HashMap;

use async_trait::async_trait;

use chrono::{DateTime, Utc};
#[cfg(test)]
use mockall::automock;
use serde::Deserialize;
use uuid::Uuid;

use crate::error::Error;

/// Database struct for deserializing workflow evaluation project info from ClickHouse.
#[derive(Debug, Deserialize)]
pub struct WorkflowEvaluationProjectRow {
    pub name: String,
    pub count: u32,
    pub last_updated: DateTime<Utc>,
}

/// Database struct for deserializing workflow evaluation run info from ClickHouse.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkflowEvaluationRunRow {
    pub name: Option<String>,
    pub id: Uuid,
    pub variant_pins: HashMap<String, String>,
    pub tags: HashMap<String, String>,
    pub project_name: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Database struct for deserializing workflow evaluation run with episode count from ClickHouse.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkflowEvaluationRunWithEpisodeCountRow {
    pub name: Option<String>,
    pub id: Uuid,
    pub variant_pins: HashMap<String, String>,
    pub tags: HashMap<String, String>,
    pub project_name: Option<String>,
    pub num_episodes: u32,
    pub timestamp: DateTime<Utc>,
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

    /// Counts workflow evaluation projects.
    async fn count_workflow_evaluation_projects(&self) -> Result<u32, Error>;

    /// Searches workflow evaluation runs by project_name and/or search_query.
    /// Returns runs matching the search criteria with pagination.
    async fn search_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        project_name: Option<&str>,
        search_query: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error>;

    /// Lists workflow evaluation runs with episode counts, optionally filtered by run_id or project_name.
    async fn list_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        run_id: Option<Uuid>,
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunWithEpisodeCountRow>, Error>;

    /// Counts workflow evaluation runs.
    async fn count_workflow_evaluation_runs(&self) -> Result<u32, Error>;

    /// Gets workflow evaluation runs by their IDs.
    /// Returns run info for the specified run IDs, optionally filtered by project_name.
    async fn get_workflow_evaluation_runs(
        &self,
        run_ids: &[Uuid],
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error>;
}
