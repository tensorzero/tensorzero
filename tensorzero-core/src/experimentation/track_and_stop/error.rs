use super::{
    check_stopping::CheckStoppingError,
    estimate_optimal_probabilities::EstimateOptimalProbabilitiesError,
};
use crate::error::Error;
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum TrackAndStopError {
    #[error("Error checking stopping conditions: {0}")]
    CheckStopping(#[from] CheckStoppingError),
    #[error("Error estimating optimal probabilities: {0}")]
    OptimalProbs(#[from] EstimateOptimalProbabilitiesError),
    #[error("Database error: {0}")]
    Database(#[from] Error),
    #[error("Task join error: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),
    #[error("Multiple feedback entries for variant '{variant_name}': found {num_entries} entries")]
    MultipleEntriesForVariant {
        variant_name: String,
        num_entries: usize,
    },
    #[error("Negative probability {probability} for variant '{variant_name}'")]
    NegativeProbability {
        variant_name: String,
        probability: f64,
    },
    #[error("No nursery or bandit arms detected. This should not happen.")]
    NoArmsDetected,
}
