use durable::{ControlFlow, TaskError};
use thiserror::Error;

/// Error type for tool execution.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Tool was not found in the registry.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Parameter validation or parsing failed.
    #[error("Parameter error: {0}")]
    InvalidParams(String),

    /// Tool execution failed with an error.
    #[error("Execution failed: {0}")]
    ExecutionFailed(#[from] anyhow::Error),

    /// JSON serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Database operation failed.
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Durable control flow signal (suspend, cancelled).
    #[error("Control flow: {0:?}")]
    Control(ControlFlow),
}

/// Result type alias for tool operations.
pub type ToolResult<T> = Result<T, ToolError>;

impl From<TaskError> for ToolError {
    fn from(err: TaskError) -> Self {
        match err {
            TaskError::Control(cf) => ToolError::Control(cf),
            TaskError::Failed(e) => ToolError::ExecutionFailed(e),
        }
    }
}

impl From<ToolError> for TaskError {
    fn from(err: ToolError) -> Self {
        match err {
            ToolError::Control(cf) => TaskError::Control(cf),
            ToolError::ToolNotFound(msg) => TaskError::Failed(anyhow::anyhow!(msg)),
            ToolError::InvalidParams(msg) => TaskError::Failed(anyhow::anyhow!(msg)),
            ToolError::ExecutionFailed(e) => TaskError::Failed(e),
            ToolError::Serialization(e) => TaskError::Failed(e.into()),
            ToolError::Database(e) => TaskError::Failed(e.into()),
        }
    }
}
