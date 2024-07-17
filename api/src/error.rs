use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Serialize;
use serde_json::json;

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
enum Error {
    #[allow(dead_code)]
    Example,
}

impl Error {
    /// Defines the error level for logging this error
    fn error_level(&self) -> tracing::Level {
        match self {
            Error::Example => tracing::Level::ERROR,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            Error::Example => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Log the error using the `tracing` library
    fn log(&self) {
        match self.error_level() {
            tracing::Level::ERROR => tracing::error!(error = self.to_string()),
            tracing::Level::WARN => tracing::warn!(error = self.to_string()),
            tracing::Level::INFO => tracing::info!(error = self.to_string()),
            tracing::Level::DEBUG => tracing::debug!(error = self.to_string()),
            tracing::Level::TRACE => tracing::trace!(error = self.to_string()),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Example => write!(f, "Example"),
        }
    }
}

impl std::error::Error for Error {}

impl IntoResponse for Error {
    /// Log the error and convert it into an Axum response
    fn into_response(self) -> Response {
        self.log();
        let body = json!({"error": self.to_string()});
        (self.status_code(), Json(body)).into_response()
    }
}
