//! Error types for the TypeScript executor pool.

use durable_tools::{NonControlToolError, TensorZeroClientError, ToolError};

/// Errors that can occur during TypeScript execution.
#[derive(Debug, thiserror::Error)]
pub enum TsError {
    /// General execution failure (e.g., too many consecutive JS errors).
    #[error("TS execution failed: {message}")]
    Execution {
        /// Description of what went wrong.
        message: String,
    },

    /// TensorZero inference call failed.
    #[error("T0 inference failed: {source}")]
    Inference {
        /// The underlying client error.
        source: TensorZeroClientError,
    },

    /// JavaScript runtime error during code execution.
    #[error("JS runtime error: {message}")]
    JsRuntime {
        /// The error message from deno_core.
        message: String,
    },

    /// The loop exceeded the maximum number of iterations without
    /// the model calling `FINAL()`.
    #[error("Max iterations ({max}) exceeded without FINAL()")]
    MaxIterations {
        /// The configured maximum iteration count.
        max: usize,
    },

    /// TypeScript type check failed after too many consecutive errors.
    #[error("TypeScript type check failed: {message}")]
    TypeCheck {
        /// Description of the type check failure.
        message: String,
    },

    /// The Deno worker thread pool has shut down.
    #[error("Thread pool shutdown")]
    PoolShutdown,

    /// Invalid configuration parameter.
    #[error("Invalid config: {message}")]
    InvalidConfig {
        /// Description of the invalid configuration.
        message: String,
    },
}

impl From<ToolError> for TsError {
    fn from(err: ToolError) -> Self {
        TsError::Execution {
            message: err.to_string(),
        }
    }
}

/// Format a non-control tool error as a string for LLM consumption.
pub fn format_non_control_tool_error_for_llm(error: &NonControlToolError) -> String {
    match error {
        // A top-level ChildFailed is produced by our `join_tool` call.
        // The step_name is not useful to LLM, so we just return the underlying message.
        NonControlToolError::ChildFailed {
            step_name: _,
            message,
        } => message.clone(),
        err => err.to_string(),
    }
}
