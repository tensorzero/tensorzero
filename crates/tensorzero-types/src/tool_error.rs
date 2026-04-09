use durable::{ControlFlow, DurableError, NonControlTaskError, TaskError};
use serde::de::Error as _;
use thiserror::Error;

use crate::tool_failure::{NonControlToolError, ToolFailure};

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

// ---------------------------------------------------------------------------
// Conversions between durable TaskError/DurableError and ToolError
// ---------------------------------------------------------------------------

impl From<TaskError> for ToolError {
    fn from(err: TaskError) -> Self {
        match err {
            TaskError::Control(cf) => ToolError::Control(cf),
            TaskError::NonControl(e) => non_control_task_error_to_tool_error(e),
        }
    }
}

fn non_control_task_error_to_tool_error(err: NonControlTaskError) -> ToolError {
    match err {
        NonControlTaskError::Database(e) => ToolError::Database(e),
        NonControlTaskError::Serialization(e) => {
            ToolError::NonControl(NonControlToolError::Serialization {
                message: e.to_string(),
            })
        }
        NonControlTaskError::Timeout { step_name } => {
            ToolError::NonControl(NonControlToolError::Timeout { step_name })
        }
        NonControlTaskError::Step { base_name, error } => {
            ToolError::NonControl(NonControlToolError::Step {
                base_name,
                error: error.to_string(),
            })
        }
        NonControlTaskError::User {
            message,
            error_data,
        } => ToolError::NonControl(NonControlToolError::User {
            message,
            error_data,
        }),
        NonControlTaskError::Validation { message } => {
            ToolError::NonControl(NonControlToolError::Validation { message })
        }
        NonControlTaskError::TaskPanicked { message } => {
            ToolError::NonControl(NonControlToolError::TaskPanicked { message })
        }
        NonControlTaskError::ChildFailed { step_name, message } => {
            ToolError::NonControl(NonControlToolError::ChildFailed { step_name, message })
        }
        NonControlTaskError::ChildCancelled { step_name } => {
            ToolError::NonControl(NonControlToolError::ChildCancelled { step_name })
        }
        NonControlTaskError::SubtaskSpawnFailed { name, error } => {
            ToolError::NonControl(NonControlToolError::SubtaskSpawnFailed {
                name,
                error: error.to_string(),
            })
        }
        NonControlTaskError::EmitEventFailed { event_name, error } => {
            ToolError::NonControl(NonControlToolError::EmitEventFailed {
                event_name,
                error: error.to_string(),
            })
        }
    }
}

impl From<ToolError> for TaskError {
    fn from(err: ToolError) -> Self {
        match err {
            ToolError::Control(cf) => TaskError::Control(cf),
            ToolError::Database(e) => NonControlTaskError::Database(e).into(),
            ToolError::NonControl(e) => non_control_tool_error_to_task_error(e),
        }
    }
}

/// Convert a `NonControlToolError` to a `TaskError`.
pub fn non_control_tool_error_to_task_error(err: NonControlToolError) -> TaskError {
    match err {
        NonControlToolError::Step { base_name, error } => NonControlTaskError::Step {
            base_name,
            error: anyhow::anyhow!(error),
        }
        .into(),
        NonControlToolError::Timeout { step_name } => {
            NonControlTaskError::Timeout { step_name }.into()
        }
        NonControlToolError::User {
            message,
            error_data,
        } => NonControlTaskError::User {
            message,
            error_data,
        }
        .into(),
        NonControlToolError::Validation { message } => {
            NonControlTaskError::Validation { message }.into()
        }
        NonControlToolError::ChildFailed { step_name, message } => {
            NonControlTaskError::ChildFailed { step_name, message }.into()
        }
        NonControlToolError::ChildCancelled { step_name } => {
            NonControlTaskError::ChildCancelled { step_name }.into()
        }
        NonControlToolError::Serialization { message } => {
            TaskError::from(serde_json::Error::custom(message))
        }
        NonControlToolError::TaskPanicked { message } => {
            NonControlTaskError::TaskPanicked { message }.into()
        }
        // For remaining variants, use the Serialize impl to generate error_data
        other @ (NonControlToolError::ToolNotFound { .. }
        | NonControlToolError::DuplicateToolName { .. }
        | NonControlToolError::InvalidParams { .. }
        | NonControlToolError::SchemaGeneration { .. }
        | NonControlToolError::Internal { .. }
        | NonControlToolError::InvalidConfiguration { .. }
        | NonControlToolError::ReservedHeaderPrefix { .. }
        | NonControlToolError::InvalidEventName { .. }
        | NonControlToolError::SubtaskSpawnFailed { .. }
        | NonControlToolError::EmitEventFailed { .. }) => {
            let error_data = serde_json::to_value(&other).unwrap_or_else(|e| {
                serde_json::json!({
                    "kind": "error_serialization_failed",
                    "serialization_error": e.to_string(),
                })
            });
            let message = other.to_string();
            NonControlTaskError::User {
                message,
                error_data,
            }
            .into()
        }
    }
}

impl From<ToolError> for ToolFailure {
    fn from(e: ToolError) -> Self {
        match e {
            ToolError::Control(cf) => ToolFailure::Control {
                message: format!(
                    "Unexpected control flow signal: {cf:?}. \
                     This should never happen, please file a bug report at \
                     https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports"
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
            DurableError::InvalidScheduleName { name, reason } => {
                ToolError::NonControl(NonControlToolError::Internal {
                    message: format!("invalid schedule name `{name}`: {reason}"),
                })
            }
            DurableError::ScheduleNotFound {
                schedule_name,
                queue_name,
            } => ToolError::NonControl(NonControlToolError::Internal {
                message: format!("schedule `{schedule_name}` not found in queue `{queue_name}`"),
            }),
        }
    }
}
