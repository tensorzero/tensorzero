pub use tensorzero_types::tool_failure::NonControlToolError;

// Re-export the core error types from tensorzero-types.
pub use tensorzero_types::tool_error::{
    ToolError, ToolResult, ToolResultExt, non_control_tool_error_to_task_error,
};

#[cfg(test)]
mod tests {
    use super::*;

    use durable::{NonControlTaskError, TaskError};
    use serde::de::Error as _;

    /// The critical invariant: TaskError -> ToolError -> TaskError must preserve retryability.
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
