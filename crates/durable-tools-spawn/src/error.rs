//! Error types for spawning operations.

use thiserror::Error;

/// Error type for spawn operations.
#[derive(Debug, Error)]
pub enum SpawnError {
    /// Missing required configuration.
    #[error("Missing configuration: {0}")]
    MissingConfig(&'static str),

    /// Database operation failed.
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// JSON serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Durable framework error.
    #[error("Durable error: {0}")]
    Durable(#[from] durable::DurableError),
}

/// Result type alias for spawn operations.
pub type SpawnResult<T> = Result<T, SpawnError>;
