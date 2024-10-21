use std::collections::HashMap;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::{json, Value};

#[derive(Debug, PartialEq)]
pub enum Error {
    AllVariantsFailed {
        errors: HashMap<String, Error>,
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
    BadCredentialsPreInference {
        provider_name: String,
    },
    ChannelWrite {
        message: String,
    },
    ClickHouseMigration {
        id: String,
        message: String,
    },
    ClickHouseQuery {
        message: String,
    },
    Config {
        message: String,
    },
    DynamicJsonSchema {
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
    GoogleAIStudioClient {
        message: String,
        status_code: StatusCode,
    },
    GoogleAIStudioServer {
        message: String,
    },
    Inference {
        message: String,
    },
    InferenceClient {
        message: String,
    },
    InferenceTimeout {
        variant_name: String,
    },
    InputValidation {
        source: Box<Error>,
    },
    InvalidBaseUrl {
        message: String,
    },
    InvalidCandidate {
        variant_name: String,
        message: String,
    },
    InvalidEpisodeId {
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
        data: Box<Value>,
        schema: Box<Value>,
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
    MistralClient {
        message: String,
        status_code: StatusCode,
    },
    MistralServer {
        message: String,
    },
    ModelProvidersExhausted {
        provider_errors: HashMap<String, Error>,
    },
    ModelValidation {
        message: String,
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
        message: String,
        raw_output: String,
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
    ToolNotFound {
        name: String,
    },
    ToolNotLoaded {
        name: String,
    },
    TypeConversion {
        message: String,
    },
    UnexpectedDynamicCredentials {
        provider_name: String,
    },
    UnknownCandidate {
        name: String,
    },
    UnknownFunction {
        name: String,
    },
    UnknownModel {
        name: String,
    },
    UnknownTool {
        name: String,
    },
    UnknownVariant {
        name: String,
    },
    UnknownMetric {
        name: String,
    },
    VLLMClient {
        message: String,
        status_code: StatusCode,
    },
    VLLMServer {
        message: String,
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
            Error::BadCredentialsPreInference { .. } => tracing::Level::ERROR,
            Error::ChannelWrite { .. } => tracing::Level::ERROR,
            Error::ClickHouseMigration { .. } => tracing::Level::ERROR,
            Error::ClickHouseQuery { .. } => tracing::Level::ERROR,
            Error::Config { .. } => tracing::Level::ERROR,
            Error::DynamicJsonSchema { .. } => tracing::Level::WARN,
            Error::FireworksClient { .. } => tracing::Level::WARN,
            Error::FireworksServer { .. } => tracing::Level::ERROR,
            Error::GCPCredentials { .. } => tracing::Level::ERROR,
            Error::GCPVertexClient { .. } => tracing::Level::WARN,
            Error::GCPVertexServer { .. } => tracing::Level::ERROR,
            Error::GoogleAIStudioClient { .. } => tracing::Level::WARN,
            Error::GoogleAIStudioServer { .. } => tracing::Level::ERROR,
            Error::Inference { .. } => tracing::Level::ERROR,
            Error::InferenceClient { .. } => tracing::Level::ERROR,
            Error::InferenceTimeout { .. } => tracing::Level::WARN,
            Error::InputValidation { .. } => tracing::Level::WARN,
            Error::InvalidBaseUrl { .. } => tracing::Level::ERROR,
            Error::InvalidCandidate { .. } => tracing::Level::ERROR,
            Error::InvalidEpisodeId { .. } => tracing::Level::WARN,
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
            Error::MistralClient { .. } => tracing::Level::WARN,
            Error::MistralServer { .. } => tracing::Level::ERROR,
            Error::ModelProvidersExhausted { .. } => tracing::Level::ERROR,
            Error::ModelValidation { .. } => tracing::Level::ERROR,
            Error::Observability { .. } => tracing::Level::ERROR,
            Error::OpenAIClient { .. } => tracing::Level::WARN,
            Error::OpenAIServer { .. } => tracing::Level::ERROR,
            Error::OutputParsing { .. } => tracing::Level::WARN,
            Error::OutputValidation { .. } => tracing::Level::WARN,
            Error::ProviderNotFound { .. } => tracing::Level::ERROR,
            Error::Serialization { .. } => tracing::Level::ERROR,
            Error::TogetherClient { .. } => tracing::Level::WARN,
            Error::TogetherServer { .. } => tracing::Level::ERROR,
            Error::ToolNotFound { .. } => tracing::Level::WARN,
            Error::ToolNotLoaded { .. } => tracing::Level::ERROR,
            Error::TypeConversion { .. } => tracing::Level::ERROR,
            Error::UnexpectedDynamicCredentials { .. } => tracing::Level::WARN,
            Error::UnknownCandidate { .. } => tracing::Level::ERROR,
            Error::UnknownFunction { .. } => tracing::Level::WARN,
            Error::UnknownModel { .. } => tracing::Level::ERROR,
            Error::UnknownTool { .. } => tracing::Level::ERROR,
            Error::UnknownVariant { .. } => tracing::Level::WARN,
            Error::UnknownMetric { .. } => tracing::Level::WARN,
            Error::VLLMClient { .. } => tracing::Level::WARN,
            Error::VLLMServer { .. } => tracing::Level::ERROR,
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
            Error::BadCredentialsPreInference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ChannelWrite { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ClickHouseMigration { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ClickHouseQuery { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::Config { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::DynamicJsonSchema { .. } => StatusCode::BAD_REQUEST,
            Error::FireworksServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::FireworksClient { status_code, .. } => *status_code,
            Error::GCPCredentials { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::GCPVertexClient { status_code, .. } => *status_code,
            Error::GCPVertexServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::GoogleAIStudioClient { status_code, .. } => *status_code,
            Error::GoogleAIStudioServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::Inference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InferenceClient { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InferenceTimeout { .. } => StatusCode::REQUEST_TIMEOUT,
            Error::InvalidEpisodeId { .. } => StatusCode::BAD_REQUEST,
            Error::InputValidation { .. } => StatusCode::BAD_REQUEST,
            Error::InvalidBaseUrl { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidCandidate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidFunctionVariants { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidMessage { .. } => StatusCode::BAD_REQUEST,
            Error::InvalidProviderConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            Error::InvalidTemplatePath => StatusCode::INTERNAL_SERVER_ERROR,
            Error::InvalidTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::JsonRequest { .. } => StatusCode::BAD_REQUEST,
            Error::JsonSchema { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::JsonSchemaValidation { .. } => StatusCode::BAD_REQUEST,
            Error::MiniJinjaEnvironment { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::MiniJinjaTemplate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::MiniJinjaTemplateMissing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::MiniJinjaTemplateRender { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::MistralClient { status_code, .. } => *status_code,
            Error::MistralServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ModelProvidersExhausted { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ModelValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::Observability { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OpenAIClient { status_code, .. } => *status_code,
            Error::OpenAIServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OutputParsing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::OutputValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ProviderNotFound { .. } => StatusCode::NOT_FOUND,
            Error::Serialization { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::TogetherClient { status_code, .. } => *status_code,
            Error::TogetherServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::ToolNotFound { .. } => StatusCode::BAD_REQUEST,
            Error::ToolNotLoaded { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::TypeConversion { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::UnknownCandidate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::UnexpectedDynamicCredentials { .. } => StatusCode::BAD_REQUEST,
            Error::UnknownFunction { .. } => StatusCode::NOT_FOUND,
            Error::UnknownModel { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::UnknownTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::UnknownVariant { .. } => StatusCode::NOT_FOUND,
            Error::UnknownMetric { .. } => StatusCode::NOT_FOUND,
            Error::VLLMClient { status_code, .. } => *status_code,
            Error::VLLMServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
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
                        .map(|(variant_name, error)| format!("{}: {}", variant_name, error))
                        .collect::<Vec<_>>()
                        .join("\n")
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
                write!(f, "Error from Anthropic server: {}", message)
            }
            Error::AWSBedrockClient { message, .. } => {
                write!(f, "Error from AWS Bedrock client: {}", message)
            }
            Error::AWSBedrockServer { message } => {
                write!(f, "Error from AWS Bedrock server: {}", message)
            }
            Error::AzureClient { message, .. } => {
                write!(f, "Error from Azure client: {}", message)
            }
            Error::AzureServer { message } => {
                write!(f, "Error from Azure server: {}", message)
            }
            Error::BadCredentialsPreInference { provider_name } => {
                write!(
                    f,
                    "Bad credentials at inference time for provider: {}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new",
                    provider_name
                )
            }
            Error::ChannelWrite { message } => {
                write!(f, "Error writing to channel: {}", message)
            }
            Error::ClickHouseMigration { id, message } => {
                write!(f, "Error running ClickHouse migration {}: {}", id, message)
            }
            Error::ClickHouseQuery { message } => {
                write!(f, "Failed to run ClickHouse query: {}", message)
            }
            Error::Config { message } => {
                write!(f, "{}", message)
            }
            Error::DynamicJsonSchema { message } => {
                write!(
                    f,
                    "Error in compiling client-provided JSON schema: {}",
                    message
                )
            }
            Error::FireworksClient { message, .. } => {
                write!(f, "Error from Fireworks client: {}", message)
            }
            Error::FireworksServer { message } => {
                write!(f, "Error from Fireworks server: {}", message)
            }
            Error::GCPCredentials { message } => {
                write!(f, "Error in acquiring GCP credentials: {}", message)
            }
            Error::GCPVertexClient { message, .. } => {
                write!(f, "Error from GCP Vertex client: {}", message)
            }
            Error::GCPVertexServer { message } => {
                write!(f, "Error from GCP Vertex server: {}", message)
            }
            Error::GoogleAIStudioClient { message, .. } => {
                write!(f, "Error from Google AI Studio client: {}", message)
            }
            Error::GoogleAIStudioServer { message } => {
                write!(f, "Error from Google AI Studio server: {}", message)
            }
            Error::Inference { message } => write!(f, "{}", message),
            Error::InferenceClient { message } => write!(f, "{}", message),
            Error::InferenceTimeout { variant_name } => {
                write!(f, "Inference timed out for variant: {}", variant_name)
            }
            Error::InputValidation { source } => {
                write!(f, "Input validation failed with messages: {}", source)
            }
            Error::InvalidBaseUrl { message } => write!(f, "{}", message),
            Error::InvalidCandidate {
                variant_name,
                message,
            } => {
                write!(
                    f,
                    "Invalid candidate variant as a component of variant {}: {}",
                    variant_name, message
                )
            }
            Error::InvalidFunctionVariants { message } => write!(f, "{}", message),
            Error::InvalidEpisodeId { message } => write!(f, "Invalid Episode ID: {}", message),
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
                    "\nData: {}",
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
            Error::MistralClient { message, .. } => {
                write!(f, "Error from Mistral client: {}", message)
            }
            Error::MistralServer { message } => {
                write!(f, "Error from Mistral server: {}", message)
            }
            Error::ModelProvidersExhausted { provider_errors } => {
                write!(
                    f,
                    "All model providers failed to infer with errors: {}",
                    provider_errors
                        .iter()
                        .map(|(provider_name, error)| format!("{}: {}", provider_name, error))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Error::ModelValidation { message } => {
                write!(f, "Failed to validate model: {}", message)
            }
            Error::Observability { message } => {
                write!(f, "{}", message)
            }
            Error::OpenAIClient { message, .. } => {
                write!(f, "Error from OpenAI client: {}", message)
            }
            Error::OpenAIServer { message } => write!(f, "Error from OpenAI server: {}", message),
            Error::OutputParsing {
                raw_output,
                message,
            } => {
                write!(
                    f,
                    "Error parsing output as JSON with message: {message}: {raw_output}"
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
                write!(f, "Error from Together server: {}", message)
            }
            Error::ToolNotFound { name } => write!(f, "Tool not found: {}", name),
            Error::ToolNotLoaded { name } => write!(f, "Tool not loaded: {}", name),
            Error::UnexpectedDynamicCredentials { provider_name } => {
                write!(
                    f,
                    "Unexpected dynamic credentials for model provider: {}. Please enable the `dynamic_credentials` flag in config if appropriate.",
                    provider_name
                )
            }
            Error::UnknownCandidate { name } => write!(f, "Unknown candidate variant: {}", name),
            Error::UnknownFunction { name } => write!(f, "Unknown function: {}", name),
            Error::UnknownModel { name } => write!(f, "Unknown model: {}", name),
            Error::UnknownTool { name } => write!(f, "Unknown tool: {}", name),
            Error::UnknownVariant { name } => write!(f, "Unknown variant: {}", name),
            Error::UnknownMetric { name } => write!(f, "Unknown metric: {}", name),
            Error::VLLMClient { message, .. } => {
                write!(f, "Error from vLLM client: {}", message)
            }
            Error::VLLMServer { message } => write!(f, "Error from vLLM server: {}", message),
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
