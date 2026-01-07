use durable::{ControlFlow, DurableError, TaskError};
use thiserror::Error;

/// Non-control errors that can be caught and handled.
#[derive(Debug, Error)]
pub enum NonControlToolError {
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
    ExecutionFailed(anyhow::Error),

    /// Schema generation failed.
    #[error("Schema generation failed: {0}")]
    SchemaGeneration(anyhow::Error),

    /// JSON serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(serde_json::Error),

    /// Database operation failed.
    #[error("Database error: {0}")]
    Database(sqlx::Error),

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

impl From<anyhow::Error> for NonControlToolError {
    fn from(err: anyhow::Error) -> Self {
        NonControlToolError::ExecutionFailed(err)
    }
}

impl From<serde_json::Error> for NonControlToolError {
    fn from(err: serde_json::Error) -> Self {
        NonControlToolError::Serialization(err)
    }
}

impl From<sqlx::Error> for NonControlToolError {
    fn from(err: sqlx::Error) -> Self {
        NonControlToolError::Database(err)
    }
}

/// Error type for tool execution, separating control signals from handleable errors.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Durable control flow signal (suspend, cancelled).
    #[error("Control flow: {0:?}")]
    Control(ControlFlow),

    /// Non-control errors that can be caught and handled.
    #[error(transparent)]
    NonControl(#[from] NonControlToolError),
}

// Convenience From impls for ToolError that delegate through NonControlToolError
impl From<serde_json::Error> for ToolError {
    fn from(err: serde_json::Error) -> Self {
        NonControlToolError::Serialization(err).into()
    }
}

impl From<sqlx::Error> for ToolError {
    fn from(err: sqlx::Error) -> Self {
        NonControlToolError::Database(err).into()
    }
}

impl From<anyhow::Error> for ToolError {
    fn from(err: anyhow::Error) -> Self {
        NonControlToolError::ExecutionFailed(err).into()
    }
}

/// Result type alias for tool operations.
pub type ToolResult<T> = Result<T, ToolError>;

/// Extension trait for ergonomic control error propagation.
pub trait ToolResultExt<T> {
    /// Propagates control errors, returns inner result for non-control cases.
    ///
    /// # Usage
    ///
    /// ```ignore
    /// let inner_result = some_tool_call().await.propagate_control()?;
    /// match inner_result {
    ///     Ok(value) => { /* use value */ },
    ///     Err(non_control_err) => { /* handle error */ },
    /// }
    /// ```
    fn propagate_control(self) -> Result<Result<T, NonControlToolError>, ToolError>;
}

impl<T> ToolResultExt<T> for ToolResult<T> {
    fn propagate_control(self) -> Result<Result<T, NonControlToolError>, ToolError> {
        match self {
            Ok(v) => Ok(Ok(v)),
            Err(ToolError::Control(cf)) => Err(ToolError::Control(cf)),
            Err(ToolError::NonControl(e)) => Ok(Err(e)),
        }
    }
}

impl From<TaskError> for ToolError {
    fn from(err: TaskError) -> Self {
        match err {
            TaskError::Control(cf) => ToolError::Control(cf),
            TaskError::Database(e) => NonControlToolError::Database(e).into(),
            TaskError::Serialization(e) => NonControlToolError::Serialization(e).into(),
            TaskError::Timeout { step_name } => NonControlToolError::Timeout { step_name }.into(),
            TaskError::Validation { message } => NonControlToolError::Validation { message }.into(),
            TaskError::TaskInternal(e) => NonControlToolError::ExecutionFailed(e).into(),
            TaskError::ChildFailed { step_name, message } => NonControlToolError::ExecutionFailed(
                anyhow::anyhow!("child task failed at '{step_name}': {message}"),
            )
            .into(),
            TaskError::User {
                message,
                error_data,
            } => NonControlToolError::ExecutionFailed(anyhow::anyhow!(
                "user error: {message} {error_data:?}"
            ))
            .into(),
            TaskError::ChildCancelled { step_name } => NonControlToolError::ExecutionFailed(
                anyhow::anyhow!("child task was cancelled at '{step_name}'"),
            )
            .into(),
        }
    }
}

impl From<ToolError> for TaskError {
    fn from(err: ToolError) -> Self {
        match err {
            ToolError::Control(cf) => TaskError::Control(cf),
            ToolError::NonControl(e) => e.into(),
        }
    }
}

impl From<NonControlToolError> for TaskError {
    fn from(err: NonControlToolError) -> Self {
        match err {
            NonControlToolError::Database(e) => TaskError::Database(e),
            NonControlToolError::Serialization(e) => TaskError::Serialization(e),
            NonControlToolError::Timeout { step_name } => TaskError::Timeout { step_name },
            NonControlToolError::Validation { message } => TaskError::Validation { message },
            NonControlToolError::ToolNotFound(msg) => TaskError::TaskInternal(anyhow::anyhow!(msg)),
            NonControlToolError::DuplicateToolName(msg) => {
                TaskError::TaskInternal(anyhow::anyhow!(msg))
            }
            NonControlToolError::InvalidParams(msg) => {
                TaskError::TaskInternal(anyhow::anyhow!(msg))
            }
            NonControlToolError::ExecutionFailed(e) => TaskError::TaskInternal(e),
            NonControlToolError::SchemaGeneration(e) => TaskError::TaskInternal(e),
        }
    }
}

impl From<DurableError> for ToolError {
    fn from(err: DurableError) -> Self {
        match err {
            DurableError::Database(e) => NonControlToolError::Database(e).into(),
            DurableError::Serialization(e) => NonControlToolError::Serialization(e).into(),
            DurableError::TaskNotRegistered { task_name } => {
                NonControlToolError::ToolNotFound(task_name).into()
            }
            DurableError::InvalidConfiguration { reason } => NonControlToolError::Validation {
                message: format!("Durable configuration invalid: {reason}"),
            }
            .into(),
            DurableError::TaskAlreadyRegistered { task_name } => {
                NonControlToolError::DuplicateToolName(task_name).into()
            }
            DurableError::ReservedHeaderPrefix { key } => NonControlToolError::Validation {
                message: format!(
                    "header key '{key}' uses reserved prefix 'durable::'. \
                     User headers cannot start with 'durable::'."
                ),
            }
            .into(),
            DurableError::InvalidEventName { reason } => NonControlToolError::Validation {
                message: format!("invalid event name: {reason}"),
            }
            .into(),
        }
    }
}
