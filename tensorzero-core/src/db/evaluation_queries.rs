//! Evaluation statistics types and trait definitions.

use async_trait::async_trait;

use chrono::{DateTime, Utc};
#[cfg(test)]
use mockall::automock;
use serde::Deserialize;
use uuid::Uuid;

use crate::error::Error;

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
}
