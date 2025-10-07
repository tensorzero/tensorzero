use super::{
    check_stopping::CheckStoppingError,
    estimate_optimal_probabilities::EstimateOptimalProbabilitiesError,
};
use crate::error::Error;
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub(super) enum TrackAndStopError {
    #[error("Error checking stopping conditions: {0}")]
    CheckStopping(#[from] CheckStoppingError),
    #[error("Error estimating optimal probabilities: {0}")]
    OptimalProbs(#[from] EstimateOptimalProbabilitiesError),
    #[error("Database error: {0}")]
    Database(#[from] Error),
    #[error("Task join error: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),
}
