//! Evaluation statistics types and trait definitions.

use async_trait::async_trait;

use chrono::{DateTime, Utc};
#[cfg(test)]
use mockall::automock;
use serde::Deserialize;
use uuid::Uuid;

use crate::error::Error;
use crate::function::FunctionConfigType;

/// Database struct for deserializing evaluation run info from ClickHouse.
#[derive(Debug, Deserialize)]
pub struct EvaluationRunInfoRow {
    pub evaluation_run_id: Uuid,
    pub evaluation_name: String,
    pub function_name: String,
    pub variant_name: String,
    pub dataset_name: String,
    pub last_inference_timestamp: DateTime<Utc>,
}

/// Database struct for deserializing evaluation run search results from ClickHouse.
#[derive(Debug, Deserialize)]
pub struct EvaluationRunSearchResult {
    pub evaluation_run_id: Uuid,
    pub variant_name: String,
}

/// Database struct for deserializing evaluation run info by IDs from ClickHouse.
/// This is a simpler struct than `EvaluationRunInfoRow` - used when querying by specific run IDs.
#[derive(Debug, Deserialize)]
pub struct EvaluationRunInfoByIdRow {
    pub evaluation_run_id: Uuid,
    pub variant_name: String,
    pub most_recent_inference_date: DateTime<Utc>,
}

/// Database struct for deserializing evaluation statistics from ClickHouse.
/// Contains aggregated metrics for an evaluation run.
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct EvaluationStatisticsRow {
    pub evaluation_run_id: Uuid,
    pub metric_name: String,
    pub datapoint_count: u32,
    pub mean_metric: f64,
    pub ci_lower: Option<f64>,
    pub ci_upper: Option<f64>,
}

/// Trait for evaluation-related queries.
#[async_trait]
#[cfg_attr(test, automock)]
pub trait EvaluationQueries {
    /// Counts the total number of unique evaluation runs across all functions.
    async fn count_total_evaluation_runs(&self) -> Result<u64, Error>;

    /// Lists evaluation runs with pagination.
    async fn list_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationRunInfoRow>, Error>;

    /// Counts unique datapoints across the specified evaluation runs.
    async fn count_datapoints_for_evaluation(
        &self,
        function_name: &str,
        evaluation_run_ids: &[Uuid],
    ) -> Result<u64, Error>;

    /// Searches evaluation runs by ID or variant name.
    async fn search_evaluation_runs(
        &self,
        evaluation_name: &str,
        function_name: &str,
        query: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationRunSearchResult>, Error>;

    /// Gets evaluation run info for specific evaluation run IDs and function name.
    async fn get_evaluation_run_infos(
        &self,
        evaluation_run_ids: &[Uuid],
        function_name: &str,
    ) -> Result<Vec<EvaluationRunInfoByIdRow>, Error>;

    /// Gets evaluation run info for inferences associated with a specific datapoint.
    async fn get_evaluation_run_infos_for_datapoint(
        &self,
        datapoint_id: &Uuid,
        function_name: &str,
        function_type: FunctionConfigType,
    ) -> Result<Vec<EvaluationRunInfoByIdRow>, Error>;

    /// Gets evaluation statistics (aggregated metrics) for specified evaluation runs.
    ///
    /// For each evaluation run and metric, returns:
    /// - datapoint count
    /// - mean metric value
    /// - confidence interval bounds (Wald CI for float metrics, Wilson CI for boolean metrics)
    async fn get_evaluation_statistics(
        &self,
        function_name: &str,
        function_type: FunctionConfigType,
        metric_names: &[String],
        evaluation_run_ids: &[Uuid],
    ) -> Result<Vec<EvaluationStatisticsRow>, Error>;
}
