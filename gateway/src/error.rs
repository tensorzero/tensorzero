use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::{json, Value};

#[derive(Debug, PartialEq)]
pub enum Error {
    AllVariantsFailed {
        errors: Vec<Error>,
    },
    ApiKeyMissing {
        provider_name: String,
    },
    AppState {
        message: String,
    },
    AnthropicClient {
        message: String,
        status_code: StatusCode,
    },
    AnthropicServer {
        message: String,
    },
    AWSBedrockClient {
        message: String,
        status_code: StatusCode,
    },
    AWSBedrockServer {
        message: String,
    },
    AzureClient {
        message: String,
        status_code: StatusCode,
    },
    AzureServer {
        message: String,
    },
    ChannelWrite {
        message: String,
    },
    ClickHouseWrite {
        message: String,
    },
    Config {
        message: String,
    },
    FireworksClient {
        message: String,
        status_code: StatusCode,
    },
    FireworksServer {
        message: String,
    },
    GCPCredentials {
        message: String,
    },
    GCPVertexClient {
        message: String,
        status_code: StatusCode,
    },
    GCPVertexServer {
        message: String,
    },
    InferenceClient {
        message: String,
    },
    Inference {
        message: String,
    },
    InputValidation {
        source: Box<Error>,
    },
    InvalidBaseUrl {
        message: String,
    },
    InvalidFunctionVariants {
        message: String,
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
    JsonSchema {
        message: String,
    },
    JsonSchemaValidation {
        messages: Vec<String>,
        data: Value,
        schema: Value,
    },
    MiniJinjaEnvironment {
        message: String,
    },
    MiniJinjaTemplate {
        template_name: String,
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
    Observability {
        message: String,
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
        source: Box<Error>,
    },
    ProviderNotFound {
        provider_name: String,
    },
    Serialization {
        message: String,
    },
    TogetherClient {
        message: String,
        status_code: StatusCode,
    },
    TogetherServer {
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
            Error::AllVariantsFailed { .. } => tracing::Level::ERROR,
            Error::ApiKeyMissing { .. } => tracing::Level::ERROR,
            Error::AppState { .. } => tracing::Level::ERROR,
            Error::AnthropicClient { .. } => tracing::Level::WARN,
            Error::AnthropicServer { .. } => tracing::Level::ERROR,
            Error::AWSBedrockClient { .. } => tracing::Level::WARN,
            Error::AWSBedrockServer { .. } => tracing::Level::ERROR,
            Error::AzureClient { .. } => tracing::Level::WARN,
            Error::AzureServer { .. } => tracing::Level::ERROR,
            Error::ChannelWrite { .. } => tracing::Level::ERROR,
            Error::ClickHouseWrite { .. } => tracing::Level::ERROR,
            Error::Config { .. } => tracing::Level::ERROR,
            Error::FireworksClient { .. } => tracing::Level::WARN,
            Error::FireworksServer { .. } => tracing::Level::ERROR,
            Error::GCPCredentials { .. } => tracing::Level::ERROR,
            Error::GCPVertexClient { .. } => tracing::Level::WARN,
            Error::GCPVertexServer { .. } => tracing::Level::ERROR,
            Error::Inference { .. } => tracing::Level::ERROR,
            Error::InferenceClient { .. } => tracing::Level::ERROR,
            Error::InputValidation { .. } => tracing::Level::WARN,
            Error::InvalidBaseUrl { .. } => tracing::Level::ERROR,
            Error::InvalidFunctionVariants { .. } => tracing::Level::ERROR,
            Error::InvalidMessage { .. } => tracing::Level::WARN,
            Error::InvalidProviderConfig { .. } => tracing::Level::ERROR,
            Error::InvalidRequest { .. } => tracing::Level::ERROR,
            Error::InvalidTemplatePath => tracing::Level::ERROR,
            Error::InvalidTool { .. } => tracing::Level::ERROR,
            Error::JsonRequest { .. } => tracing::Level::WARN,
            Error::JsonSchema { .. } => tracing::Level::ERROR,
            Error::JsonSchemaValidation { .. } => tracing::Level::ERROR,
            Error::MiniJinjaEnvironment { .. } => tracing::Level::ERROR,
            Error::MiniJinjaTemplate { .. } => tracing::Level::ERROR,
            Error::MiniJinjaTemplateMissing { .. } => tracing::Level::ERROR,
            Error::MiniJinjaTemplateRender { .. } => tracing::Level::ERROR,
            Error::ModelNotFound { .. } => tracing::Level::ERROR,
            Error::ModelProvidersExhausted { .. } => tracing::Level::ERROR,
            Error::Observability { .. } => tracing::Level::ERROR,
            Error::OpenAIClient { .. } => tracing::Level::WARN,
            Error::OpenAIServer { .. } => tracing::Level::ERROR,
            Error::OutputParsing { .. } => tracing::Level::WARN,
            Error::OutputValidation { .. } => tracing::Level::WARN,
            Error::ProviderNotFound { .. } => tracing::Level::ERROR,
            Error::Serialization { .. } => tracing::Level::ERROR,
            Error::TogetherClient { .. } => tracing::Level::WARN,
            Error::TogetherServer { .. } => tracing::Level::ERROR,
            Error::TypeConversion { .. } => tracing::Level::ERROR,
            Error::UnknownFunction { .. } => tracing::Level::WARN,
            Error::UnknownVariant { .. } => tracing::Level::WARN,
            Error::UnknownMetric { .. } => tracing::Level::WARN,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            Error::AllVariantsFailed { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ApiKeyMissing { .. } => StatusCode::BAD_REQUEST,
            Error::AppState { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::AnthropicClient { status_code, .. } => *status_code,
            Error::AnthropicServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::AWSBedrockClient { status_code, .. } => *status_code,
            Error::AWSBedrockServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::AzureClient { status_code, .. } => *status_code,
            Error::AzureServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ChannelWrite { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ClickHouseWrite { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::Config { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::FireworksServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::FireworksClient { status_code, .. } => *status_code,
            Error::GCPCredentials { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::GCPVertexClient { status_code, .. } => *status_code,
            Error::GCPVertexServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::Inference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InferenceClient { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InputValidation { .. } => StatusCode::BAD_REQUEST,
            Error::InvalidBaseUrl { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidFunctionVariants { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidMessage { .. } => StatusCode::BAD_REQUEST,
            Error::InvalidProviderConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            Error::InvalidTemplatePath => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::JsonRequest { .. } => StatusCode::BAD_REQUEST,
            Error::JsonSchema { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::JsonSchemaValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::MiniJinjaEnvironment { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::MiniJinjaTemplate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::MiniJinjaTemplateMissing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::MiniJinjaTemplateRender { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ModelNotFound { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ModelProvidersExhausted { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::Observability { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OpenAIClient { status_code, .. } => *status_code,
            Error::OpenAIServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OutputParsing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OutputValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ProviderNotFound { .. } => StatusCode::NOT_FOUND,
            Error::Serialization { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::TogetherClient { status_code, .. } => *status_code,
            Error::TogetherServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::TypeConversion { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::UnknownFunction { .. } => StatusCode::NOT_FOUND,
            Error::UnknownVariant { .. } => StatusCode::NOT_FOUND,
            Error::UnknownMetric { .. } => StatusCode::NOT_FOUND,
        }
    }

    /// Log the error using the `tracing` library
    pub fn log(&self) {
        match self.level() {
            tracing::Level::ERROR => tracing::error!("{self}"),
            tracing::Level::WARN => tracing::warn!("{self}"),
            tracing::Level::INFO => tracing::info!("{self}"),
            tracing::Level::DEBUG => tracing::debug!("{self}"),
            tracing::Level::TRACE => tracing::trace!("{self}"),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::AllVariantsFailed { errors } => {
                write!(
                    f,
                    "All variants failed with errors: {}",
                    errors
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Error::ApiKeyMissing { provider_name } => {
                write!(f, "API key missing for provider: {}", provider_name)
            }
            Error::AppState { message } => {
                write!(f, "Error initializing AppState: {}", message)
            }
            Error::AnthropicClient { message, .. } => {
                write!(f, "Error from Anthropic client: {}", message)
            }
            Error::AnthropicServer { message } => {
                write!(f, "Error from Anthropic servers: {}", message)
            }
            Error::AWSBedrockClient { message, .. } => {
                write!(f, "Error from AWS Bedrock client: {}", message)
            }
            Error::AWSBedrockServer { message } => {
                write!(f, "Error from AWS Bedrock servers: {}", message)
            }
            Error::AzureClient { message, .. } => {
                write!(f, "Error from Azure client: {}", message)
            }
            Error::AzureServer { message } => {
                write!(f, "Error from Azure servers: {}", message)
            }
            Error::ChannelWrite { message } => {
                write!(f, "Error writing to channel: {}", message)
            }
            Error::ClickHouseWrite { message } => {
                write!(f, "Error writing to ClickHouse: {}", message)
            }
            Error::Config { message } => {
                write!(f, "Error in TensorZero config: {}", message)
            }
            Error::FireworksClient { message, .. } => {
                write!(f, "Error from Fireworks client: {}", message)
            }
            Error::FireworksServer { message } => {
                write!(f, "Error from Fireworks servers: {}", message)
            }
            Error::GCPCredentials { message } => {
                write!(f, "Error in acquiring GCP credentials: {}", message)
            }
            Error::GCPVertexClient { message, .. } => {
                write!(f, "Error from GCP Vertex client: {}", message)
            }
            Error::GCPVertexServer { message } => {
                write!(f, "Error from GCP Vertex servers: {}", message)
            }
            Error::Inference { message } => write!(f, "{}", message),
            Error::InferenceClient { message } => write!(f, "{}", message),
            Error::InputValidation { source } => {
                write!(f, "Input validation failed with messages: {}", source)
            }
            Error::InvalidBaseUrl { message } => write!(f, "{}", message),
            Error::InvalidFunctionVariants { message } => write!(f, "{}", message),
            Error::InvalidMessage { message } => write!(f, "{}", message),
            Error::InvalidProviderConfig { message } => write!(f, "{}", message),
            Error::InvalidRequest { message } => write!(f, "{}", message),
            Error::InvalidTemplatePath => {
                write!(f, "Template path failed to convert to Rust string")
            }
            Error::InvalidTool { message } => write!(f, "{}", message),
            Error::JsonRequest { message } => write!(f, "{}", message),
            Error::JsonSchema { message } => write!(f, "{}", message),
            Error::JsonSchemaValidation {
                messages,
                data,
                schema,
            } => {
                write!(
                    f,
                    "JSON Schema validation failed for Function:\n\n{}",
                    messages.join("\n")
                )?;
                write!(
                    f,
                    "Data: {}",
                    serde_json::to_string(data).map_err(|_| std::fmt::Error)?
                )?;
                write!(
                    f,
                    "Schema: {}",
                    serde_json::to_string(schema).map_err(|_| std::fmt::Error)?
                )
            }
            Error::MiniJinjaEnvironment { message } => {
                write!(f, "Error initializing MiniJinja environment: {}", message)
            }
            Error::MiniJinjaTemplate {
                template_name,
                message,
            } => {
                write!(f, "Error rendering template {}: {}", template_name, message)
            }
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
            Error::Observability { message } => {
                write!(f, "{}", message)
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
                    "Error parsing output as JSON with message: {}: {}",
                    message, raw_output
                )
            }
            Error::OutputValidation { source } => {
                write!(f, "Output validation failed with messages: {}", source)
            }
            Error::ProviderNotFound { provider_name } => {
                write!(f, "Provider not found: {}", provider_name)
            }
            Error::Serialization { message } => write!(f, "{}", message),
            Error::TypeConversion { message } => write!(f, "{}", message),
            Error::TogetherClient { message, .. } => {
                write!(f, "Error from Together client: {}", message)
            }
            Error::TogetherServer { message } => {
                write!(f, "Error from Together servers: {}", message)
            }
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

pub trait ResultExt<T> {
    fn ok_or_log(self) -> Option<T>;
}

impl<T> ResultExt<T> for Result<T, Error> {
    fn ok_or_log(self) -> Option<T> {
        match self {
            Ok(value) => Some(value),
            Err(error) => {
                error.log();
                None
            }
        }
    }
}
