//! Error types for the Autopilot client.

use thiserror::Error;

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

    /// Missing required configuration.
    #[error("Missing required configuration: {0}")]
    MissingConfig(&'static str),
}
