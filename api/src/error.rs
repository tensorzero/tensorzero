use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

#[derive(Debug, PartialEq)]
pub enum Error {
    AnthropicClient {
        message: String,
        status_code: StatusCode,
    },
    AnthropicServer {
        message: String,
    },
    InferenceClient {
        message: String,
    },
    InvalidMessage {
        message: String,
    },
    InvalidRequest {
        message: String,
    },
    InvalidTool {
        message: String,
    },
}

impl Error {
    /// Defines the error level for logging this error
    fn level(&self) -> tracing::Level {
        match self {
            Error::AnthropicServer { .. }
            | Error::InferenceClient { .. }
            | Error::InvalidMessage { .. }
            | Error::InvalidRequest { .. }
            | Error::InvalidTool { .. } => tracing::Level::ERROR,
            Error::AnthropicClient { .. } => tracing::Level::WARN,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            Error::AnthropicServer { .. }
            | Error::InferenceClient { .. }
            | Error::InvalidMessage { .. }
            | Error::InvalidRequest { .. }
            | Error::InvalidTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::AnthropicClient { status_code, .. } => *status_code,
        }
    }

    /// Log the error using the `tracing` library
    fn log(&self) {
        match self.level() {
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
            Error::AnthropicServer { message }
            | Error::InferenceClient { message }
            | Error::InvalidMessage { message }
            | Error::InvalidRequest { message }
            | Error::InvalidTool { message }
            | Error::AnthropicClient { message, .. } => write!(f, "{}", message),
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
