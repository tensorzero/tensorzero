use std::collections::HashMap;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::{json, Value};

#[derive(Debug, PartialEq)]
// As long as the struct member is private, we force people to use the `new` method and log the error.
pub struct Error(ErrorDetails);

impl Error {
    pub fn new(details: ErrorDetails) -> Self {
        details.log();
        Error(details)
    }

    pub fn status_code(&self) -> StatusCode {
        self.0.status_code()
    }

    pub fn get_details(&self) -> &ErrorDetails {
        &self.0
    }

    pub fn get_owned_details(self) -> ErrorDetails {
        self.0
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl From<ErrorDetails> for Error {
    fn from(details: ErrorDetails) -> Self {
        Error::new(details)
    }
}

#[derive(Debug, PartialEq)]
pub enum ErrorDetails {
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
    ClickHouseConnection {
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
    InvalidOpenAICompatibleRequest {
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

impl ErrorDetails {
    /// Defines the error level for logging this error
    fn level(&self) -> tracing::Level {
        match self {
            ErrorDetails::AllVariantsFailed { .. } => tracing::Level::ERROR,
            ErrorDetails::ApiKeyMissing { .. } => tracing::Level::ERROR,
            ErrorDetails::AppState { .. } => tracing::Level::ERROR,
            ErrorDetails::AnthropicClient { .. } => tracing::Level::WARN,
            ErrorDetails::AnthropicServer { .. } => tracing::Level::ERROR,
            ErrorDetails::AWSBedrockClient { .. } => tracing::Level::WARN,
            ErrorDetails::AWSBedrockServer { .. } => tracing::Level::ERROR,
            ErrorDetails::AzureClient { .. } => tracing::Level::WARN,
            ErrorDetails::AzureServer { .. } => tracing::Level::ERROR,
            ErrorDetails::BadCredentialsPreInference { .. } => tracing::Level::ERROR,
            ErrorDetails::ChannelWrite { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseConnection { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseMigration { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseQuery { .. } => tracing::Level::ERROR,
            ErrorDetails::Config { .. } => tracing::Level::ERROR,
            ErrorDetails::DynamicJsonSchema { .. } => tracing::Level::WARN,
            ErrorDetails::FireworksClient { .. } => tracing::Level::WARN,
            ErrorDetails::FireworksServer { .. } => tracing::Level::ERROR,
            ErrorDetails::GCPCredentials { .. } => tracing::Level::ERROR,
            ErrorDetails::GCPVertexClient { .. } => tracing::Level::WARN,
            ErrorDetails::GCPVertexServer { .. } => tracing::Level::ERROR,
            ErrorDetails::GoogleAIStudioClient { .. } => tracing::Level::WARN,
            ErrorDetails::GoogleAIStudioServer { .. } => tracing::Level::ERROR,
            ErrorDetails::Inference { .. } => tracing::Level::ERROR,
            ErrorDetails::InferenceClient { .. } => tracing::Level::ERROR,
            ErrorDetails::InferenceTimeout { .. } => tracing::Level::WARN,
            ErrorDetails::InputValidation { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidBaseUrl { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidCandidate { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidEpisodeId { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidFunctionVariants { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidMessage { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidOpenAICompatibleRequest { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidProviderConfig { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidRequest { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidTemplatePath => tracing::Level::ERROR,
            ErrorDetails::InvalidTool { .. } => tracing::Level::ERROR,
            ErrorDetails::JsonRequest { .. } => tracing::Level::WARN,
            ErrorDetails::JsonSchema { .. } => tracing::Level::ERROR,
            ErrorDetails::JsonSchemaValidation { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaEnvironment { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaTemplate { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaTemplateMissing { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaTemplateRender { .. } => tracing::Level::ERROR,
            ErrorDetails::MistralClient { .. } => tracing::Level::WARN,
            ErrorDetails::MistralServer { .. } => tracing::Level::ERROR,
            ErrorDetails::ModelProvidersExhausted { .. } => tracing::Level::ERROR,
            ErrorDetails::ModelValidation { .. } => tracing::Level::ERROR,
            ErrorDetails::Observability { .. } => tracing::Level::ERROR,
            ErrorDetails::OpenAIClient { .. } => tracing::Level::WARN,
            ErrorDetails::OpenAIServer { .. } => tracing::Level::ERROR,
            ErrorDetails::OutputParsing { .. } => tracing::Level::WARN,
            ErrorDetails::OutputValidation { .. } => tracing::Level::WARN,
            ErrorDetails::ProviderNotFound { .. } => tracing::Level::ERROR,
            ErrorDetails::Serialization { .. } => tracing::Level::ERROR,
            ErrorDetails::TogetherClient { .. } => tracing::Level::WARN,
            ErrorDetails::TogetherServer { .. } => tracing::Level::ERROR,
            ErrorDetails::ToolNotFound { .. } => tracing::Level::WARN,
            ErrorDetails::ToolNotLoaded { .. } => tracing::Level::ERROR,
            ErrorDetails::TypeConversion { .. } => tracing::Level::ERROR,
            ErrorDetails::UnexpectedDynamicCredentials { .. } => tracing::Level::WARN,
            ErrorDetails::UnknownCandidate { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownFunction { .. } => tracing::Level::WARN,
            ErrorDetails::UnknownModel { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownTool { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownVariant { .. } => tracing::Level::WARN,
            ErrorDetails::UnknownMetric { .. } => tracing::Level::WARN,
            ErrorDetails::VLLMClient { .. } => tracing::Level::WARN,
            ErrorDetails::VLLMServer { .. } => tracing::Level::ERROR,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            ErrorDetails::AllVariantsFailed { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ApiKeyMissing { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::AppState { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::AnthropicClient { status_code, .. } => *status_code,
            ErrorDetails::AnthropicServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::AWSBedrockClient { status_code, .. } => *status_code,
            ErrorDetails::AWSBedrockServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::AzureClient { status_code, .. } => *status_code,
            ErrorDetails::AzureServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::BadCredentialsPreInference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ChannelWrite { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseConnection { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseMigration { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseQuery { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::Config { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::DynamicJsonSchema { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::FireworksServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::FireworksClient { status_code, .. } => *status_code,
            ErrorDetails::GCPCredentials { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::GCPVertexClient { status_code, .. } => *status_code,
            ErrorDetails::GCPVertexServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::GoogleAIStudioClient { status_code, .. } => *status_code,
            ErrorDetails::GoogleAIStudioServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::Inference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InferenceClient { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InferenceTimeout { .. } => StatusCode::REQUEST_TIMEOUT,
            ErrorDetails::InvalidEpisodeId { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InputValidation { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidBaseUrl { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidCandidate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidFunctionVariants { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidMessage { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidOpenAICompatibleRequest { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidProviderConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidTemplatePath => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::JsonRequest { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::JsonSchema { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::JsonSchemaValidation { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::MiniJinjaEnvironment { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::MiniJinjaTemplate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::MiniJinjaTemplateMissing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::MiniJinjaTemplateRender { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::MistralClient { status_code, .. } => *status_code,
            ErrorDetails::MistralServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ModelProvidersExhausted { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ModelValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::Observability { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::OpenAIClient { status_code, .. } => *status_code,
            ErrorDetails::OpenAIServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::OutputParsing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::OutputValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ProviderNotFound { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::Serialization { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::TogetherClient { status_code, .. } => *status_code,
            ErrorDetails::TogetherServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ToolNotFound { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::ToolNotLoaded { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::TypeConversion { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownCandidate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnexpectedDynamicCredentials { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::UnknownFunction { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::UnknownModel { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownVariant { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::UnknownMetric { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::VLLMClient { status_code, .. } => *status_code,
            ErrorDetails::VLLMServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
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

impl std::fmt::Display for ErrorDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorDetails::AllVariantsFailed { errors } => {
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
            ErrorDetails::ApiKeyMissing { provider_name } => {
                write!(f, "API key missing for provider: {}", provider_name)
            }
            ErrorDetails::AppState { message } => {
                write!(f, "Error initializing AppState: {}", message)
            }
            ErrorDetails::AnthropicClient { message, .. } => {
                write!(f, "Error from Anthropic client: {}", message)
            }
            ErrorDetails::AnthropicServer { message } => {
                write!(f, "Error from Anthropic server: {}", message)
            }
            ErrorDetails::AWSBedrockClient { message, .. } => {
                write!(f, "Error from AWS Bedrock client: {}", message)
            }
            ErrorDetails::AWSBedrockServer { message } => {
                write!(f, "Error from AWS Bedrock server: {}", message)
            }
            ErrorDetails::AzureClient { message, .. } => {
                write!(f, "Error from Azure client: {}", message)
            }
            ErrorDetails::AzureServer { message } => {
                write!(f, "Error from Azure server: {}", message)
            }
            ErrorDetails::BadCredentialsPreInference { provider_name } => {
                write!(
                    f,
                    "Bad credentials at inference time for provider: {}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new",
                    provider_name
                )
            }
            ErrorDetails::ChannelWrite { message } => {
                write!(f, "Error writing to channel: {}", message)
            }
            ErrorDetails::ClickHouseConnection { message } => {
                write!(f, "Error connecting to ClickHouse: {}", message)
            }
            ErrorDetails::ClickHouseMigration { id, message } => {
                write!(f, "Error running ClickHouse migration {}: {}", id, message)
            }
            ErrorDetails::ClickHouseQuery { message } => {
                write!(f, "Failed to run ClickHouse query: {}", message)
            }
            ErrorDetails::Config { message } => {
                write!(f, "{}", message)
            }
            ErrorDetails::DynamicJsonSchema { message } => {
                write!(
                    f,
                    "Error in compiling client-provided JSON schema: {}",
                    message
                )
            }
            ErrorDetails::FireworksClient { message, .. } => {
                write!(f, "Error from Fireworks client: {}", message)
            }
            ErrorDetails::FireworksServer { message } => {
                write!(f, "Error from Fireworks server: {}", message)
            }
            ErrorDetails::GCPCredentials { message } => {
                write!(f, "Error in acquiring GCP credentials: {}", message)
            }
            ErrorDetails::GCPVertexClient { message, .. } => {
                write!(f, "Error from GCP Vertex client: {}", message)
            }
            ErrorDetails::GCPVertexServer { message } => {
                write!(f, "Error from GCP Vertex server: {}", message)
            }
            ErrorDetails::GoogleAIStudioClient { message, .. } => {
                write!(f, "Error from Google AI Studio client: {}", message)
            }
            ErrorDetails::GoogleAIStudioServer { message } => {
                write!(f, "Error from Google AI Studio server: {}", message)
            }
            ErrorDetails::Inference { message } => write!(f, "{}", message),
            ErrorDetails::InferenceClient { message } => write!(f, "{}", message),
            ErrorDetails::InferenceTimeout { variant_name } => {
                write!(f, "Inference timed out for variant: {}", variant_name)
            }
            ErrorDetails::InputValidation { source } => {
                write!(f, "Input validation failed with messages: {}", source)
            }
            ErrorDetails::InvalidBaseUrl { message } => write!(f, "{}", message),
            ErrorDetails::InvalidCandidate {
                variant_name,
                message,
            } => {
                write!(
                    f,
                    "Invalid candidate variant as a component of variant {}: {}",
                    variant_name, message
                )
            }
            ErrorDetails::InvalidFunctionVariants { message } => write!(f, "{}", message),
            ErrorDetails::InvalidEpisodeId { message } => {
                write!(f, "Invalid Episode ID: {}", message)
            }
            ErrorDetails::InvalidMessage { message } => write!(f, "{}", message),
            ErrorDetails::InvalidOpenAICompatibleRequest { message } => write!(
                f,
                "Invalid request to OpenAI-compatible endpoint: {}",
                message
            ),
            ErrorDetails::InvalidProviderConfig { message } => write!(f, "{}", message),
            ErrorDetails::InvalidRequest { message } => write!(f, "{}", message),
            ErrorDetails::InvalidTemplatePath => {
                write!(f, "Template path failed to convert to Rust string")
            }
            ErrorDetails::InvalidTool { message } => write!(f, "{}", message),
            ErrorDetails::JsonRequest { message } => write!(f, "{}", message),
            ErrorDetails::JsonSchema { message } => write!(f, "{}", message),
            ErrorDetails::JsonSchemaValidation {
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
            ErrorDetails::MiniJinjaEnvironment { message } => {
                write!(f, "Error initializing MiniJinja environment: {}", message)
            }
            ErrorDetails::MiniJinjaTemplate {
                template_name,
                message,
            } => {
                write!(f, "Error rendering template {}: {}", template_name, message)
            }
            ErrorDetails::MiniJinjaTemplateMissing { template_name } => {
                write!(f, "Template not found: {}", template_name)
            }
            ErrorDetails::MiniJinjaTemplateRender {
                template_name,
                message,
            } => {
                write!(f, "Error rendering template {}: {}", template_name, message)
            }
            ErrorDetails::MistralClient { message, .. } => {
                write!(f, "Error from Mistral client: {}", message)
            }
            ErrorDetails::MistralServer { message } => {
                write!(f, "Error from Mistral server: {}", message)
            }
            ErrorDetails::ModelProvidersExhausted { provider_errors } => {
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
            ErrorDetails::ModelValidation { message } => {
                write!(f, "Failed to validate model: {}", message)
            }
            ErrorDetails::Observability { message } => {
                write!(f, "{}", message)
            }
            ErrorDetails::OpenAIClient { message, .. } => {
                write!(f, "Error from OpenAI client: {}", message)
            }
            ErrorDetails::OpenAIServer { message } => {
                write!(f, "Error from OpenAI server: {}", message)
            }
            ErrorDetails::OutputParsing {
                raw_output,
                message,
            } => {
                write!(
                    f,
                    "Error parsing output as JSON with message: {message}: {raw_output}"
                )
            }
            ErrorDetails::OutputValidation { source } => {
                write!(f, "Output validation failed with messages: {}", source)
            }
            ErrorDetails::ProviderNotFound { provider_name } => {
                write!(f, "Provider not found: {}", provider_name)
            }
            ErrorDetails::Serialization { message } => write!(f, "{}", message),
            ErrorDetails::TypeConversion { message } => write!(f, "{}", message),
            ErrorDetails::TogetherClient { message, .. } => {
                write!(f, "Error from Together client: {}", message)
            }
            ErrorDetails::TogetherServer { message } => {
                write!(f, "Error from Together server: {}", message)
            }
            ErrorDetails::ToolNotFound { name } => write!(f, "Tool not found: {}", name),
            ErrorDetails::ToolNotLoaded { name } => write!(f, "Tool not loaded: {}", name),
            ErrorDetails::UnexpectedDynamicCredentials { provider_name } => {
                write!(
                    f,
                    "Unexpected dynamic credentials for model provider: {}. Please enable the `dynamic_credentials` flag in config if appropriate.",
                    provider_name
                )
            }
            ErrorDetails::UnknownCandidate { name } => {
                write!(f, "Unknown candidate variant: {}", name)
            }
            ErrorDetails::UnknownFunction { name } => write!(f, "Unknown function: {}", name),
            ErrorDetails::UnknownModel { name } => write!(f, "Unknown model: {}", name),
            ErrorDetails::UnknownTool { name } => write!(f, "Unknown tool: {}", name),
            ErrorDetails::UnknownVariant { name } => write!(f, "Unknown variant: {}", name),
            ErrorDetails::UnknownMetric { name } => write!(f, "Unknown metric: {}", name),
            ErrorDetails::VLLMClient { message, .. } => {
                write!(f, "Error from vLLM client: {}", message)
            }
            ErrorDetails::VLLMServer { message } => {
                write!(f, "Error from vLLM server: {}", message)
            }
        }
    }
}

impl std::error::Error for Error {}

impl IntoResponse for Error {
    /// Log the error and convert it into an Axum response
    fn into_response(self) -> Response {
        let body = json!({"error": self.to_string()});
        (self.status_code(), Json(body)).into_response()
    }
}
