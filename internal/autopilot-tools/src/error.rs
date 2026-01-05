//! Error types for autopilot tools.
//!
//! This module defines structured error types for all autopilot tool operations.
//! Errors are serialized to JSON for persistence in durable's fail_run table
//! and can be deserialized back for programmatic error handling.

use durable_tools::ToolError;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use thiserror::Error;

/// Error type for autopilot tools.
///
/// All autopilot tool errors are converted to `ToolError::User` for persistence.
/// The JSON representation uses `#[serde(tag = "kind")]` for easy discrimination:
///
/// ```json
/// { "kind": "ClientError", "operation": "inference", "message": "connection timeout" }
/// ```
#[derive(Debug, Error, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum AutopilotToolError {
    /// TensorZero client operation failed.
    #[error("{operation} failed: {message}")]
    ClientError {
        /// The operation that failed (e.g., "inference", "feedback", "create_datapoints").
        operation: String,
        /// The error message from the client.
        message: String,
    },

    /// Validation error for tool parameters or state.
    #[error("{message}")]
    Validation {
        /// Description of the validation error.
        message: String,
    },

    /// Test tool error (for e2e testing).
    #[error("{message}")]
    TestError {
        /// The test error message.
        message: String,
    },
}

impl AutopilotToolError {
    /// Create a client error for a specific operation.
    pub fn client_error(operation: impl Into<String>, error: impl std::fmt::Display) -> Self {
        Self::ClientError {
            operation: operation.into(),
            message: error.to_string(),
        }
    }

    /// Create a validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a test error.
    pub fn test_error(message: impl Into<String>) -> Self {
        Self::TestError {
            message: message.into(),
        }
    }
}

impl From<AutopilotToolError> for ToolError {
    fn from(e: AutopilotToolError) -> ToolError {
        ToolError::User {
            message: e.to_string(),
            error_data: serde_json::to_value(&e).unwrap_or(JsonValue::Null),
        }
    }
}

/// Result type alias for autopilot tools.
pub type AutopilotToolResult<T> = Result<T, AutopilotToolError>;
