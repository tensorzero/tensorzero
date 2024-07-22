use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

#[derive(Debug, PartialEq)]
pub enum Error {
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
    FireworksClient {
        message: String,
        status_code: StatusCode,
    },
    #[allow(dead_code)] // TODO: remove
    FireworksServer {
        message: String,
    },
    #[allow(dead_code)] // TODO: remove
    InferenceClient {
        message: String,
    },
    InvalidBaseUrl {
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
    InvalidProviderConfig {
        message: String,
    },
    OpenAIClient {
        message: String,
        status_code: StatusCode,
    },
    OpenAIServer {
        message: String,
    },
}

impl Error {
    /// Defines the error level for logging this error
    fn level(&self) -> tracing::Level {
        match self {
            Error::AnthropicServer { .. } => tracing::Level::ERROR,
            Error::FireworksServer { .. } => tracing::Level::ERROR,
            Error::InferenceClient { .. } => tracing::Level::ERROR,
            Error::InvalidRequest { .. } => tracing::Level::ERROR,
            Error::InvalidBaseUrl { .. } => tracing::Level::ERROR,
            Error::InvalidTool { .. } => tracing::Level::ERROR,
            Error::InvalidProviderConfig { .. } => tracing::Level::ERROR,
            Error::OpenAIServer { .. } => tracing::Level::ERROR,
            Error::AnthropicClient { .. } => tracing::Level::WARN,
            Error::FireworksClient { .. } => tracing::Level::WARN,
            Error::InvalidMessage { .. } => tracing::Level::WARN,
            Error::OpenAIClient { .. } => tracing::Level::WARN,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            Error::AnthropicServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::FireworksServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InferenceClient { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidBaseUrl { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidRequest { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidProviderConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OpenAIServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::AnthropicClient { status_code, .. } => *status_code,
            Error::OpenAIClient { status_code, .. } => *status_code,
            Error::FireworksClient { status_code, .. } => *status_code,
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
            Error::InferenceClient { message } => write!(f, "{}", message),
            Error::InvalidBaseUrl { message } => write!(f, "{}", message),
            Error::InvalidMessage { message } => write!(f, "{}", message),
            Error::InvalidRequest { message } => write!(f, "{}", message),
            Error::InvalidTool { message } => write!(f, "{}", message),
            Error::InvalidProviderConfig { message } => write!(f, "{}", message),
            Error::AnthropicServer { message } => {
                write!(f, "Error from Anthropic servers: {}", message)
            }
            Error::AnthropicClient { message, .. } => {
                write!(f, "Error from Anthropic client: {}", message)
            }
            Error::FireworksServer { message } => {
                write!(f, "Error from Fireworks servers: {}", message)
            }
            Error::FireworksClient { message, .. } => {
                write!(f, "Error from Fireworks client: {}", message)
            }
            Error::OpenAIServer { message } => write!(f, "Error from OpenAI servers: {}", message),
            Error::OpenAIClient { message, .. } => {
                write!(f, "Error from OpenAI client: {}", message)
            }
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
