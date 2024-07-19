use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

#[derive(Debug, PartialEq)]
pub enum Error {
    Inference {
        message: String,
    },
    InvalidFunctionVariants {
        message: String,
    },
    InvalidInputSchema {
        messages: Vec<String>,
    },
    JsonRequest {
        message: String,
    },
    UnknownFunction {
        name: String,
    },
    UnknownVariant {
        name: String,
    },

    // TODO: clean up merge
    #[allow(dead_code)] // TODO: remove
    AnthropicClient {
        message: String,
        status_code: StatusCode,
    },
    #[allow(dead_code)] // TODO: remove
    AnthropicServer {
        message: String,
    },
    #[allow(dead_code)] // TODO: remove
    InferenceClient {
        message: String,
    },
    InvalidMessage {
        message: String,
    },
    #[allow(dead_code)] // TODO: remove
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
            Error::AnthropicClient { .. } => tracing::Level::WARN,
            Error::AnthropicServer { .. } => tracing::Level::ERROR,
            Error::Inference { .. } => tracing::Level::ERROR,
            Error::InferenceClient { .. } => tracing::Level::ERROR,
            Error::InvalidFunctionVariants { .. } => tracing::Level::ERROR,
            Error::InvalidInputSchema { .. } => tracing::Level::WARN,
            Error::InvalidMessage { .. } => tracing::Level::WARN,
            Error::InvalidRequest { .. } => tracing::Level::ERROR,
            Error::InvalidTool { .. } => tracing::Level::ERROR,
            Error::JsonRequest { .. } => tracing::Level::WARN,
            Error::UnknownFunction { .. } => tracing::Level::WARN,
            Error::UnknownVariant { .. } => tracing::Level::WARN,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            Error::Inference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidFunctionVariants { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidInputSchema { .. } => StatusCode::BAD_REQUEST,
            Error::JsonRequest { .. } => StatusCode::BAD_REQUEST,
            Error::UnknownFunction { .. } => StatusCode::NOT_FOUND,
            Error::UnknownVariant { .. } => StatusCode::NOT_FOUND,
            Error::AnthropicServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InferenceClient { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidRequest { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::AnthropicClient { status_code, .. } => *status_code,
            Error::InvalidMessage { .. } => StatusCode::BAD_REQUEST,
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
            Error::AnthropicClient { message, .. } => write!(f, "{}", message),
            Error::AnthropicServer { message } => write!(f, "{}", message),
            Error::Inference { message } => write!(f, "{}", message),
            Error::InferenceClient { message } => write!(f, "{}", message),
            Error::InvalidFunctionVariants { message } => write!(f, "{}", message),
            Error::InvalidInputSchema { messages } => {
                write!(
                    f,
                    "The parameter 'input' does not fit the schema for this Function:\n\n{}",
                    messages.join("\n")
                )
            }
            Error::InvalidMessage { message } => write!(f, "{}", message),
            Error::InvalidRequest { message } => write!(f, "{}", message),
            Error::InvalidTool { message } => write!(f, "{}", message),
            Error::JsonRequest { message } => write!(f, "{}", message),
            Error::UnknownFunction { name } => write!(f, "Unknown function: {}", name),
            Error::UnknownVariant { name } => write!(f, "Unknown variant: {}", name),
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
