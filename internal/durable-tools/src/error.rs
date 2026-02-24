use durable::{ControlFlow, DurableError, TaskError};
use tensorzero_core::error::IMPOSSIBLE_ERROR_MESSAGE;
use thiserror::Error;

pub use tensorzero_types::tool_failure::NonControlToolError;

/// Error type for tool execution.
///
/// This enum separates control flow signals and non-serializable errors
/// from serializable errors that can be persisted and returned in API responses.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Durable control flow signal (suspend, cancelled).
    /// This is internal and should never be serialized.
    #[error("Control flow: {0:?}")]
    Control(ControlFlow),

    /// Database operation failed.
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// A serializable tool error.
    #[error(transparent)]
    NonControl(#[from] NonControlToolError),
}

impl From<serde_json::Error> for ToolError {
    fn from(err: serde_json::Error) -> Self {
        ToolError::NonControl(NonControlToolError::Serialization {
            message: err.to_string(),
        })
    }
}

/// Result type alias for tool operations.
pub type ToolResult<T> = Result<T, ToolError>;

/// Extension trait for ergonomic control error propagation.
pub trait ToolResultExt<T> {
    /// Propagates control, database, and serialization errors.
    /// Returns inner result for non-control cases that can be handled.
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
            Err(ToolError::Database(e)) => Err(ToolError::Database(e)),
            Err(ToolError::NonControl(e)) => Ok(Err(e)),
        }
    }
}

impl From<TaskError> for ToolError {
    fn from(err: TaskError) -> Self {
        match err {
            TaskError::Control(cf) => ToolError::Control(cf),
            TaskError::Database(e) => ToolError::Database(e),
            TaskError::Serialization(e) => {
                ToolError::NonControl(NonControlToolError::Serialization {
                    message: e.to_string(),
                })
            }
            TaskError::Timeout { step_name } => {
                ToolError::NonControl(NonControlToolError::Timeout { step_name })
            }
            TaskError::User {
                message,
                error_data,
            } => ToolError::NonControl(NonControlToolError::User {
                message,
                error_data,
            }),
            TaskError::Validation { message } => {
                ToolError::NonControl(NonControlToolError::Validation { message })
            }
            TaskError::ChildFailed { step_name, message } => {
                ToolError::NonControl(NonControlToolError::ChildFailed { step_name, message })
            }
            TaskError::ChildCancelled { step_name } => {
                ToolError::NonControl(NonControlToolError::ChildCancelled { step_name })
            }
            TaskError::Step { base_name, error } => {
                ToolError::NonControl(NonControlToolError::Step {
                    base_name,
                    error: error.to_string(),
                })
            }
            TaskError::TaskPanicked { message } => {
                ToolError::NonControl(NonControlToolError::TaskPanicked { message })
            }
            TaskError::SubtaskSpawnFailed { name, error } => {
                ToolError::NonControl(NonControlToolError::SubtaskSpawnFailed {
                    name,
                    error: error.to_string(),
                })
            }
            TaskError::EmitEventFailed { event_name, error } => {
                ToolError::NonControl(NonControlToolError::EmitEventFailed {
                    event_name,
                    error: error.to_string(),
                })
            }
        }
    }
}

impl From<ToolError> for TaskError {
    fn from(err: ToolError) -> Self {
        match err {
            ToolError::Control(cf) => TaskError::Control(cf),
            ToolError::Database(e) => TaskError::Database(e),
            ToolError::NonControl(e) => non_control_tool_error_to_task_error(e),
        }
    }
}

/// Convert a `NonControlToolError` to a `TaskError`.
///
/// This is a standalone function rather than a `From` impl because
/// `NonControlToolError` is defined in `tensorzero-types` and `TaskError`
/// in `durable`, so the orphan rule prevents an impl here.
pub fn non_control_tool_error_to_task_error(err: NonControlToolError) -> TaskError {
    match err {
        NonControlToolError::Timeout { step_name } => TaskError::Timeout { step_name },
        NonControlToolError::User {
            message,
            error_data,
        } => TaskError::User {
            message,
            error_data,
        },
        NonControlToolError::Validation { message } => TaskError::Validation { message },
        NonControlToolError::ChildFailed { step_name, message } => {
            TaskError::ChildFailed { step_name, message }
        }
        NonControlToolError::ChildCancelled { step_name } => {
            TaskError::ChildCancelled { step_name }
        }
        // For all other variants, use the Serialize impl to generate error_data
        other => {
            let error_data = serde_json::to_value(&other).unwrap_or_else(|e| {
                serde_json::json!({
                    "kind": "error_serialization_failed",
                    "serialization_error": e.to_string(),
                })
            });
            let message = other.to_string();
            TaskError::User {
                message,
                error_data,
            }
        }
    }
}

impl From<ToolError> for tensorzero_types::ToolFailure {
    fn from(e: ToolError) -> Self {
        use tensorzero_types::ToolFailure;
        match e {
            ToolError::Control(cf) => ToolFailure::Control {
                message: format!(
                    "Unexpected control flow signal: {cf:?}. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            },
            ToolError::Database(db_error) => ToolFailure::Database {
                message: db_error.to_string(),
            },
            ToolError::NonControl(non_control_error) => ToolFailure::Tool {
                error: non_control_error,
            },
        }
    }
}

impl From<DurableError> for ToolError {
    fn from(err: DurableError) -> Self {
        match err {
            DurableError::Database(e) => ToolError::Database(e),
            DurableError::Serialization(e) => {
                ToolError::NonControl(NonControlToolError::Serialization {
                    message: e.to_string(),
                })
            }
            DurableError::TaskNotRegistered { task_name } => {
                ToolError::NonControl(NonControlToolError::ToolNotFound { name: task_name })
            }
            DurableError::TaskAlreadyRegistered { task_name } => {
                ToolError::NonControl(NonControlToolError::DuplicateToolName { name: task_name })
            }
            DurableError::InvalidConfiguration { reason } => {
                ToolError::NonControl(NonControlToolError::InvalidConfiguration { reason })
            }
            DurableError::ReservedHeaderPrefix { key } => {
                ToolError::NonControl(NonControlToolError::ReservedHeaderPrefix { key })
            }
            DurableError::InvalidEventName { reason } => {
                ToolError::NonControl(NonControlToolError::InvalidEventName { reason })
            }
            DurableError::InvalidTaskParams { task_name, message } => {
                ToolError::NonControl(NonControlToolError::InvalidParams {
                    message: format!("invalid task parameters for `{task_name}`: {message}"),
                })
            }
            DurableError::InvalidState { state } => {
                ToolError::NonControl(NonControlToolError::Internal {
                    message: format!("invalid task state: {state}"),
                })
            }
            DurableError::InvalidState { state } => {
                ToolError::NonControl(NonControlToolError::Internal {
                    message: format!("invalid task state: {state}"),
                })
            }
        }
    }
}
