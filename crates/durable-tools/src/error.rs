use durable::{ControlFlow, DurableError, NonControlTaskError, TaskError};
use serde::de::Error as _;
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
///
/// This is a standalone function rather than a `From` impl because
/// `NonControlToolError` is defined in `tensorzero-types` and `TaskError`
/// in `durable`, so the orphan rule prevents an impl here.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The critical invariant: TaskError → ToolError → TaskError must preserve retryability.
    ///
    /// This is the round-trip that happens in `TaskToolAdapter::run()`:
    /// 1. `TaskContext::step()` produces a `TaskError`
    /// 2. `ToolContext::step()` converts it to `ToolError` (via `From<TaskError>`)
    /// 3. `TaskToolAdapter::run()` converts it back to `TaskError` (via `From<ToolError>`)
    ///
    /// If retryability is lost in this round-trip, the durable worker won't retry
    /// transient failures that should be retried.
    fn round_trip(original: TaskError) -> TaskError {
        let tool_error: ToolError = original.into();
        tool_error.into()
    }

    use durable::NonControlTaskError;

    #[test]
    fn step_error_retryable_after_round_trip() {
        let err = TaskError::NonControl(NonControlTaskError::Step {
            base_name: "my_step".to_string(),
            error: anyhow::anyhow!("transient failure"),
        });
        assert!(
            err.retryable(),
            "Step should be retryable before round-trip"
        );

        let back = round_trip(TaskError::NonControl(NonControlTaskError::Step {
            base_name: "my_step".to_string(),
            error: anyhow::anyhow!("transient failure"),
        }));
        assert!(
            back.retryable(),
            "Step must remain retryable after round-trip through ToolError, got: {back}"
        );
    }

    #[test]
    fn timeout_retryable_after_round_trip() {
        let back = round_trip(TaskError::NonControl(NonControlTaskError::Timeout {
            step_name: "wait_for_event".to_string(),
        }));
        assert!(back.retryable());
    }

    #[test]
    fn user_error_not_retryable_after_round_trip() {
        let back = round_trip(TaskError::NonControl(NonControlTaskError::User {
            message: "bad input".to_string(),
            error_data: serde_json::json!({"message": "bad input"}),
        }));
        assert!(!back.retryable());
    }

    #[test]
    fn validation_not_retryable_after_round_trip() {
        let back = round_trip(TaskError::NonControl(NonControlTaskError::Validation {
            message: "invalid".to_string(),
        }));
        assert!(!back.retryable());
    }

    #[test]
    fn serialization_not_retryable_after_round_trip() {
        let back = round_trip(TaskError::NonControl(NonControlTaskError::Serialization(
            serde_json::Error::custom("bad json"),
        )));
        assert!(!back.retryable());
    }

    #[test]
    fn task_panicked_not_retryable_after_round_trip() {
        let back = round_trip(TaskError::NonControl(NonControlTaskError::TaskPanicked {
            message: "panic".to_string(),
        }));
        assert!(!back.retryable());
    }

    #[test]
    fn child_failed_not_retryable_after_round_trip() {
        let back = round_trip(TaskError::NonControl(NonControlTaskError::ChildFailed {
            step_name: "child".to_string(),
            message: "failed".to_string(),
        }));
        assert!(!back.retryable());
    }

    #[test]
    fn child_cancelled_not_retryable_after_round_trip() {
        let back = round_trip(TaskError::NonControl(NonControlTaskError::ChildCancelled {
            step_name: "child".to_string(),
        }));
        assert!(!back.retryable());
    }

    #[test]
    fn step_error_preserves_message_after_round_trip() {
        let back = round_trip(TaskError::NonControl(NonControlTaskError::Step {
            base_name: "fetch_data".to_string(),
            error: anyhow::anyhow!("connection refused"),
        }));
        let msg = format!("{back}");
        assert!(msg.contains("fetch_data"), "expected base_name in: {msg}");
        assert!(
            msg.contains("connection refused"),
            "expected error message in: {msg}"
        );
        assert!(back.retryable());
    }

    #[test]
    fn serialization_error_preserves_message_after_round_trip() {
        let back = round_trip(TaskError::NonControl(NonControlTaskError::Serialization(
            serde_json::Error::custom("bad json"),
        )));
        let msg = format!("{back}");
        assert!(msg.contains("bad json"), "expected error message in: {msg}");
        assert!(!back.retryable());
    }
}
