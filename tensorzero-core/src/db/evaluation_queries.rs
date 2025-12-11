//! Evaluation statistics types and trait definitions.

use async_trait::async_trait;

use crate::error::Error;

/// Trait for evaluation-related queries.
#[async_trait]
pub trait EvaluationQueries {
    /// Counts the total number of unique evaluation runs across all functions.
    async fn count_total_evaluation_runs(&self) -> Result<u64, Error>;
}
