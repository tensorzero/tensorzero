//! Error types for TypeScript tool execution.

use thiserror::Error;

/// Errors that can occur during TypeScript tool execution.
#[derive(Debug, Error)]
pub enum TypeScriptToolError {
    /// Failed to transpile TypeScript to JavaScript.
    #[error("TypeScript transpilation failed: {0}")]
    Transpile(String),

    /// Failed during JavaScript execution.
    #[error("Tool execution failed: {0}")]
    Execution(String),

    /// Failed to serialize or deserialize data.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid tool definition.
    #[error("Invalid tool definition: {0}")]
    InvalidTool(String),

    /// Schema generation failed.
    #[error("Schema generation failed: {0}")]
    SchemaGeneration(String),

    /// Worker pool was shut down.
    #[error("Worker pool is shut down")]
    PoolShutdown,

    /// Worker thread panicked.
    #[error("Worker thread panicked")]
    WorkerPanicked,

    /// Channel receive failed.
    #[error("Channel receive failed: {0}")]
    ChannelReceive(String),

    /// TypeScript type checking failed.
    #[error("TypeScript type checking failed:\n{0}")]
    TypeCheck(String),
}

impl From<serde_json::Error> for TypeScriptToolError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}
