//! Error types for the Autopilot client.

use thiserror::Error;
use uuid::Uuid;

use crate::types::ApproveAllToolCallsResponse;

/// Errors that can occur when using the Autopilot client.
#[derive(Error, Debug)]
pub enum AutopilotError {
    /// HTTP error returned by the API.
    #[error("HTTP error {status_code}: {message}")]
    Http { status_code: u16, message: String },

    /// Error making the HTTP request.
    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),

    /// Error parsing JSON response.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Error with SSE streaming.
    #[error("SSE error: {0}")]
    Sse(String),

    /// Invalid URL.
    #[error("Invalid URL: {0}")]
    InvalidUrl(#[from] url::ParseError),

    /// Error spawning a tool task.
    #[error("Spawn error: {0}")]
    Spawn(#[from] durable_tools_spawn::SpawnError),

    /// Database error.
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Missing required configuration.
    #[error("Missing required configuration: {0}")]
    MissingConfig(&'static str),

    /// Tool call not found.
    #[error("Tool call not found: {0}")]
    ToolCallNotFound(Uuid),

    /// Some tool call spawns failed after approval.
    /// Contains the successful response and the errors that occurred.
    #[error("Failed to spawn {count} of {total} approved tool calls", count = .errors.len(), total = .response.tool_call_event_ids.len())]
    PartialSpawnFailure {
        response: ApproveAllToolCallsResponse,
        errors: Vec<(Uuid, Box<AutopilotError>)>,
    },

    /// Internal error - indicates a bug in the client.
    #[error("Internal error: {0}")]
    Internal(String),
}
