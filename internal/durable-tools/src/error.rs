use durable::{ControlFlow, DurableError, TaskError};
use thiserror::Error;

/// Error type for tool execution.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Tool was not found in the registry.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Tool with this name is already registered.
    #[error("Tool '{0}' is already registered. Each tool name must be unique.")]
    DuplicateToolName(String),

    /// Parameter validation or parsing failed.
    #[error("Parameter error: {0}")]
    InvalidParams(String),

    /// Tool execution failed with an error.
    #[error("Execution failed: {0}")]
    ExecutionFailed(#[from] anyhow::Error),

    /// Schema generation failed.
    #[error("Schema generation failed: {0}")]
    SchemaGeneration(anyhow::Error),

    /// JSON serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Database operation failed.
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Durable control flow signal (suspend, cancelled).
    #[error("Control flow: {0:?}")]
    Control(ControlFlow),

    /// Operation timed out.
    #[error("Timed out waiting for '{step_name}'")]
    Timeout {
        /// The name of the step or event that timed out.
        step_name: String,
    },

    /// A validation error occurred.
    #[error("{message}")]
    Validation {
        /// A description of the validation error.
        message: String,
    },
}

/// Result type alias for tool operations.
pub type ToolResult<T> = Result<T, ToolError>;

impl From<TaskError> for ToolError {
    fn from(err: TaskError) -> Self {
        match err {
            TaskError::Control(cf) => ToolError::Control(cf),
            TaskError::Database(e) => ToolError::Database(e),
            TaskError::Serialization(e) => ToolError::Serialization(e),
            TaskError::Timeout { step_name } => ToolError::Timeout { step_name },
            TaskError::Validation { message } => ToolError::Validation { message },
            TaskError::TaskInternal(e) => ToolError::ExecutionFailed(e),
            TaskError::ChildFailed { step_name, message } => ToolError::ExecutionFailed(
                anyhow::anyhow!("child task failed at '{step_name}': {message}"),
            ),
            TaskError::ChildCancelled { step_name } => ToolError::ExecutionFailed(anyhow::anyhow!(
                "child task was cancelled at '{step_name}'"
            )),
        }
    }
}

impl From<ToolError> for TaskError {
    fn from(err: ToolError) -> Self {
        match err {
            ToolError::Control(cf) => TaskError::Control(cf),
            ToolError::Database(e) => TaskError::Database(e),
            ToolError::Serialization(e) => TaskError::Serialization(e),
            ToolError::Timeout { step_name } => TaskError::Timeout { step_name },
            ToolError::Validation { message } => TaskError::Validation { message },
            ToolError::ToolNotFound(msg) => TaskError::TaskInternal(anyhow::anyhow!(msg)),
            ToolError::DuplicateToolName(msg) => TaskError::TaskInternal(anyhow::anyhow!(msg)),
            ToolError::InvalidParams(msg) => TaskError::TaskInternal(anyhow::anyhow!(msg)),
            ToolError::ExecutionFailed(e) => TaskError::TaskInternal(e),
            ToolError::SchemaGeneration(e) => TaskError::TaskInternal(e),
        }
    }
}

impl From<DurableError> for ToolError {
    fn from(err: DurableError) -> Self {
        match err {
            DurableError::Database(e) => ToolError::Database(e),
            DurableError::Serialization(e) => ToolError::Serialization(e),
            DurableError::TaskNotRegistered { task_name } => ToolError::ToolNotFound(task_name),
            DurableError::InvalidConfiguration { reason } => ToolError::Validation {
                message: format!("Durable configuration invalid: {reason}"),
            },
            DurableError::TaskAlreadyRegistered { task_name } => {
                ToolError::DuplicateToolName(task_name)
            }
            DurableError::ReservedHeaderPrefix { key } => ToolError::Validation {
                message: format!(
                    "header key '{key}' uses reserved prefix 'durable::'. \
                     User headers cannot start with 'durable::'."
                ),
            },
            DurableError::InvalidEventName { reason } => ToolError::Validation {
                message: format!("invalid event name: {reason}"),
            },
        }
    }
}
