use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

#[derive(Debug, PartialEq)]
pub enum Error {
    ApiKeyMissing {
        provider_name: String,
    },
    AnthropicClient {
        message: String,
        status_code: StatusCode,
    },
    AnthropicServer {
        message: String,
    },
    ClickHouseWrite {
        message: String,
    },
    FireworksClient {
        message: String,
        status_code: StatusCode,
    },
    FireworksServer {
        message: String,
    },
    InferenceClient {
        message: String,
    },
    Inference {
        message: String,
    },
    InvalidBaseUrl {
        message: String,
    },
    InvalidFunctionVariants {
        message: String,
    },
    InvalidInputSchema {
        messages: Vec<String>,
    },
    InvalidMessage {
        message: String,
    },
    InvalidProviderConfig {
        message: String,
    },
    InvalidRequest {
        message: String,
    },
    InvalidTemplatePath,
    InvalidTool {
        message: String,
    },
    JsonRequest {
        message: String,
    },
    MiniJinjaTemplateMissing {
        template_name: String,
    },
    MiniJinjaTemplateRender {
        template_name: String,
        message: String,
    },
    ModelNotFound {
        model: String,
    },
    ModelProvidersExhausted {
        provider_errors: Vec<Error>,
    },
    OpenAIClient {
        message: String,
        status_code: StatusCode,
    },
    OpenAIServer {
        message: String,
    },
    OutputParsing {
        raw_output: String,
        message: String,
    },
    OutputValidation {
        raw_output: String,
        message: String,
    },
    ProviderNotFound {
        provider_name: String,
    },
    Serialization {
        message: String,
    },
    TypeConversion {
        message: String,
    },
    UnknownFunction {
        name: String,
    },
    UnknownVariant {
        name: String,
    },
    UnknownMetric {
        name: String,
    },
}

impl Error {
    /// Defines the error level for logging this error
    fn level(&self) -> tracing::Level {
        match self {
            Error::ApiKeyMissing { .. } => tracing::Level::ERROR,
            Error::AnthropicClient { .. } => tracing::Level::WARN,
            Error::AnthropicServer { .. } => tracing::Level::ERROR,
            Error::ClickHouseWrite { .. } => tracing::Level::ERROR,
            Error::FireworksClient { .. } => tracing::Level::WARN,
            Error::FireworksServer { .. } => tracing::Level::ERROR,
            Error::Inference { .. } => tracing::Level::ERROR,
            Error::InferenceClient { .. } => tracing::Level::ERROR,
            Error::InvalidBaseUrl { .. } => tracing::Level::ERROR,
            Error::InvalidFunctionVariants { .. } => tracing::Level::ERROR,
            Error::InvalidInputSchema { .. } => tracing::Level::ERROR,
            Error::InvalidMessage { .. } => tracing::Level::WARN,
            Error::InvalidProviderConfig { .. } => tracing::Level::ERROR,
            Error::InvalidRequest { .. } => tracing::Level::ERROR,
            Error::InvalidTemplatePath => tracing::Level::ERROR,
            Error::InvalidTool { .. } => tracing::Level::ERROR,
            Error::JsonRequest { .. } => tracing::Level::WARN,
            Error::MiniJinjaTemplateMissing { .. } => tracing::Level::ERROR,
            Error::MiniJinjaTemplateRender { .. } => tracing::Level::ERROR,
            Error::ModelNotFound { .. } => tracing::Level::ERROR,
            Error::ModelProvidersExhausted { .. } => tracing::Level::ERROR,
            Error::OpenAIClient { .. } => tracing::Level::WARN,
            Error::OpenAIServer { .. } => tracing::Level::ERROR,
            Error::OutputParsing { .. } => tracing::Level::WARN,
            Error::OutputValidation { .. } => tracing::Level::WARN,
            Error::ProviderNotFound { .. } => tracing::Level::ERROR,
            Error::Serialization { .. } => tracing::Level::ERROR,
            Error::TypeConversion { .. } => tracing::Level::ERROR,
            Error::UnknownFunction { .. } => tracing::Level::WARN,
            Error::UnknownVariant { .. } => tracing::Level::WARN,
            Error::UnknownMetric { .. } => tracing::Level::WARN,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            Error::ApiKeyMissing { .. } => StatusCode::BAD_REQUEST,
            Error::AnthropicClient { status_code, .. } => *status_code,
            Error::AnthropicServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ClickHouseWrite { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::FireworksServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::FireworksClient { status_code, .. } => *status_code,
            Error::Inference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InferenceClient { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidBaseUrl { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidFunctionVariants { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidInputSchema { .. } => StatusCode::BAD_REQUEST,
            Error::InvalidMessage { .. } => StatusCode::BAD_REQUEST,
            Error::InvalidProviderConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            Error::InvalidTemplatePath => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::JsonRequest { .. } => StatusCode::BAD_REQUEST,
            Error::MiniJinjaTemplateMissing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::MiniJinjaTemplateRender { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ModelNotFound { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ModelProvidersExhausted { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OpenAIClient { status_code, .. } => *status_code,
            Error::OpenAIServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OutputParsing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OutputValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ProviderNotFound { .. } => StatusCode::NOT_FOUND,
            Error::Serialization { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::TypeConversion { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::UnknownFunction { .. } => StatusCode::NOT_FOUND,
            Error::UnknownVariant { .. } => StatusCode::NOT_FOUND,
            Error::UnknownMetric { .. } => StatusCode::NOT_FOUND,
        }
    }

    /// Log the error using the `tracing` library
    pub fn log(&self) {
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
            Error::ApiKeyMissing { provider_name } => {
                write!(f, "API key missing for provider: {}", provider_name)
            }
            Error::AnthropicClient { message, .. } => {
                write!(f, "Error from Anthropic client: {}", message)
            }
            Error::AnthropicServer { message } => {
                write!(f, "Error from Anthropic servers: {}", message)
            }
            Error::ClickHouseWrite { message } => {
                write!(f, "Error writing to ClickHouse: {}", message)
            }
            Error::FireworksClient { message, .. } => {
                write!(f, "Error from Fireworks client: {}", message)
            }
            Error::FireworksServer { message } => {
                write!(f, "Error from Fireworks servers: {}", message)
            }
            Error::Inference { message } => write!(f, "{}", message),
            Error::InferenceClient { message } => write!(f, "{}", message),
            Error::InvalidBaseUrl { message } => write!(f, "{}", message),
            Error::InvalidFunctionVariants { message } => write!(f, "{}", message),
            Error::InvalidInputSchema { messages } => {
                write!(
                    f,
                    "The parameter 'input' does not fit the schema for this Function:\n\n{}",
                    messages.join("\n")
                )
            }
            Error::InvalidMessage { message } => write!(f, "{}", message),
            Error::InvalidProviderConfig { message } => write!(f, "{}", message),
            Error::InvalidRequest { message } => write!(f, "{}", message),
            Error::InvalidTemplatePath => {
                write!(f, "Template path failed to convert to Rust string")
            }
            Error::InvalidTool { message } => write!(f, "{}", message),
            Error::JsonRequest { message } => write!(f, "{}", message),
            Error::MiniJinjaTemplateMissing { template_name } => {
                write!(f, "Template not found: {}", template_name)
            }
            Error::MiniJinjaTemplateRender {
                template_name,
                message,
            } => {
                write!(f, "Error rendering template {}: {}", template_name, message)
            }
            Error::ModelNotFound { model } => {
                write!(f, "Model not found: {}", model)
            }
            Error::ModelProvidersExhausted { provider_errors } => {
                write!(
                    f,
                    "All model providers failed to infer with errors: {}",
                    provider_errors
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Error::OpenAIClient { message, .. } => {
                write!(f, "Error from OpenAI client: {}", message)
            }
            Error::OpenAIServer { message } => write!(f, "Error from OpenAI servers: {}", message),
            Error::OutputParsing {
                raw_output,
                message,
            } => {
                write!(
                    f,
                    "Error parsing output as JSON: {}: {}",
                    message, raw_output
                )
            }
            Error::OutputValidation {
                raw_output,
                message,
            } => {
                write!(f, "Error validating output: {}: {}", raw_output, message)
            }
            Error::ProviderNotFound { provider_name } => {
                write!(f, "Provider not found: {}", provider_name)
            }
            Error::Serialization { message } => write!(f, "{}", message),
            Error::TypeConversion { message } => write!(f, "{}", message),
            Error::UnknownFunction { name } => write!(f, "Unknown function: {}", name),
            Error::UnknownVariant { name } => write!(f, "Unknown variant: {}", name),
            Error::UnknownMetric { name } => write!(f, "Unknown metric: {}", name),
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
