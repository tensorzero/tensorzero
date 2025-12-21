//! Error types for the autopilot worker.

use durable_tools::ToolError;
use thiserror::Error;

/// Errors that can occur during autopilot tool execution.
#[derive(Debug, Error)]
pub enum AutopilotToolError {
    /// Error from the underlying durable tool infrastructure.
    #[error("tool error: {0}")]
    Tool(#[from] ToolError),

    /// Error from the autopilot client when sending results.
    #[error("autopilot client error: {0}")]
    AutopilotClient(#[from] autopilot_client::AutopilotError),

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Generic error with message.
    #[error("{0}")]
    Other(String),
}

impl AutopilotToolError {
    /// Create an error from a string message.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

/// Result type for autopilot tool operations.
pub type AutopilotToolResult<T> = Result<T, AutopilotToolError>;
