//! Workflow evaluation types and trait definitions.

use std::collections::HashMap;

use async_trait::async_trait;

use chrono::{DateTime, Utc};
#[cfg(test)]
use mockall::automock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[cfg(feature = "ts-bindings")]
use ts_rs::TS;

use crate::config::snapshot::SnapshotHash;
use crate::error::Error;

/// Info returned when looking up a workflow evaluation run by episode ID.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkflowEvaluationRunInfo {
    pub variant_pins: HashMap<String, String>,
    pub tags: HashMap<String, String>,
}

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

/// Internal database struct for deserializing raw statistics from ClickHouse.
/// This is used before computing confidence intervals in Rust.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkflowEvaluationRunStatisticsRaw {
    pub metric_name: String,
    pub count: u32,
    pub avg_metric: f64,
    pub stdev: Option<f64>,
    /// Whether this metric is a boolean metric (for Wilson CI) or float metric (for Wald CI)
    pub is_boolean: bool,
}

/// Database struct for workflow evaluation run statistics by metric name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkflowEvaluationRunStatisticsRow {
    pub metric_name: String,
    pub count: u32,
    pub avg_metric: f64,
    pub stdev: Option<f64>,
    pub ci_lower: Option<f64>,
    pub ci_upper: Option<f64>,
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

    /// Gets statistics for workflow evaluation run feedback grouped by metric name.
    /// Returns aggregated statistics (count, mean, stdev, confidence intervals) for each metric.
    async fn get_workflow_evaluation_run_statistics(
        &self,
        run_id: Uuid,
        metric_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunStatisticsRow>, Error>;

    /// Lists workflow evaluation run episodes grouped by task_name for given run IDs.
    ///
    /// Returns episodes grouped by task_name. Episodes with NULL task_name are grouped
    /// individually using a generated key based on their episode_id.
    ///
    /// # Parameters
    /// - `run_ids`: List of run IDs to filter episodes by
    /// - `limit`: Maximum number of groups to return
    /// - `offset`: Number of groups to skip
    ///
    /// # Returns
    /// A vector of episode rows ordered by group's last_updated DESC, then by group_key, then by episode_id.
    async fn list_workflow_evaluation_run_episodes_by_task_name(
        &self,
        run_ids: &[Uuid],
        limit: u32,
        offset: u32,
    ) -> Result<Vec<GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow>, Error>;

    /// Counts the number of distinct task_name groups for the given run IDs.
    ///
    /// Episodes with NULL task_name are counted as individual groups.
    async fn count_workflow_evaluation_run_episodes_by_task_name(
        &self,
        run_ids: &[Uuid],
    ) -> Result<u32, Error>;

    /// Gets workflow evaluation run episodes with their feedback for a specific run.
    /// Returns episodes with feedback metric names and values, sorted by episode_id descending.
    async fn get_workflow_evaluation_run_episodes_with_feedback(
        &self,
        run_id: Uuid,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<WorkflowEvaluationRunEpisodeWithFeedbackRow>, Error>;

    /// Counts the total number of episodes for a specific workflow evaluation run.
    async fn count_workflow_evaluation_run_episodes(&self, run_id: Uuid) -> Result<u32, Error>;

    /// Inserts a new workflow evaluation run.
    ///
    /// Note: The table is named `DynamicEvaluationRun` for historical reasons,
    /// but this feature is now called "Workflow Evaluations".
    async fn insert_workflow_evaluation_run(
        &self,
        run_id: Uuid,
        variant_pins: &HashMap<String, String>,
        tags: &HashMap<String, String>,
        project_name: Option<&str>,
        run_display_name: Option<&str>,
        snapshot_hash: &SnapshotHash,
    ) -> Result<(), Error>;

    /// Inserts a new workflow evaluation run episode.
    ///
    /// This copies variant_pins from the parent run and merges the provided tags
    /// on top of the run's tags.
    ///
    /// Note: The table is named `DynamicEvaluationRunEpisode` for historical reasons,
    /// but this feature is now called "Workflow Evaluations".
    async fn insert_workflow_evaluation_run_episode(
        &self,
        run_id: Uuid,
        episode_id: Uuid,
        task_name: Option<&str>,
        tags: &HashMap<String, String>,
        snapshot_hash: &SnapshotHash,
    ) -> Result<(), Error>;

    /// Looks up a workflow evaluation run by episode ID.
    ///
    /// Returns the variant_pins and tags for the run associated with the given episode.
    async fn get_workflow_evaluation_run_by_episode_id(
        &self,
        episode_id: Uuid,
    ) -> Result<Option<WorkflowEvaluationRunInfo>, Error>;
}

/// A single workflow evaluation run episode with its associated feedback.
#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct WorkflowEvaluationRunEpisodeWithFeedbackRow {
    /// The episode ID
    pub episode_id: Uuid,
    /// When the episode started (RFC 3339 format)
    pub timestamp: DateTime<Utc>,
    /// The run ID this episode belongs to
    pub run_id: Uuid,
    /// Tags associated with this episode
    pub tags: HashMap<String, String>,
    /// The task name (datapoint_name). NULL for episodes without a task name.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub task_name: Option<String>,
    /// Metric names for feedback, sorted alphabetically
    pub feedback_metric_names: Vec<String>,
    /// Feedback values corresponding to feedback_metric_names
    pub feedback_values: Vec<String>,
}

/// A workflow evaluation run episode with feedback, including its group key.
///
/// The group_key is either the task_name or a generated key for episodes with NULL task_name.
#[cfg_attr(feature = "ts-bindings", derive(TS))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow {
    /// The grouping key - either task_name or 'NULL_EPISODE_{episode_id_uint}'
    pub group_key: String,
    /// The episode ID
    pub episode_id: Uuid,
    /// When the episode started (RFC 3339 format)
    pub timestamp: DateTime<Utc>,
    /// The run ID this episode belongs to
    pub run_id: Uuid,
    /// Tags associated with this episode
    pub tags: HashMap<String, String>,
    /// The task name (datapoint_name). NULL for episodes without a task name.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub task_name: Option<String>,
    /// Metric names for feedback, sorted alphabetically
    pub feedback_metric_names: Vec<String>,
    /// Feedback values corresponding to feedback_metric_names
    pub feedback_values: Vec<String>,
}
