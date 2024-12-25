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

    pub fn new_without_logging(details: ErrorDetails) -> Self {
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
    BadCredentialsPreInference {
        provider_name: String,
    },
    BatchInputValidation {
        index: usize,
        message: String,
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
    GCPCredentials {
        message: String,
    },
    InferenceServer {
        message: String,
        provider_type: String,
    },
    InferenceClient {
        message: String,
        status_code: Option<StatusCode>,
        provider_type: String,
    },
    Inference {
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
    InvalidDiclConfig {
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
    ModelProvidersExhausted {
        provider_errors: HashMap<String, Error>,
    },
    ModelValidation {
        message: String,
    },
    Observability {
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
    ToolNotFound {
        name: String,
    },
    ToolNotLoaded {
        name: String,
    },
    TypeConversion {
        message: String,
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
    UnsupportedModelProviderForBatchInference {
        provider_type: String,
    },
    UnsupportedVariantForBatchInference {
        variant_name: Option<String>,
    },
}

impl ErrorDetails {
    /// Defines the error level for logging this error
    fn level(&self) -> tracing::Level {
        match self {
            ErrorDetails::AllVariantsFailed { .. } => tracing::Level::ERROR,
            ErrorDetails::ApiKeyMissing { .. } => tracing::Level::ERROR,
            ErrorDetails::AppState { .. } => tracing::Level::ERROR,
            ErrorDetails::BadCredentialsPreInference { .. } => tracing::Level::ERROR,
            ErrorDetails::BatchInputValidation { .. } => tracing::Level::WARN,
            ErrorDetails::ChannelWrite { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseConnection { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseMigration { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseQuery { .. } => tracing::Level::ERROR,
            ErrorDetails::Config { .. } => tracing::Level::ERROR,
            ErrorDetails::DynamicJsonSchema { .. } => tracing::Level::WARN,
            ErrorDetails::GCPCredentials { .. } => tracing::Level::ERROR,
            ErrorDetails::Inference { .. } => tracing::Level::ERROR,
            ErrorDetails::InferenceClient { .. } => tracing::Level::ERROR,
            ErrorDetails::InferenceServer { .. } => tracing::Level::ERROR,
            ErrorDetails::InferenceTimeout { .. } => tracing::Level::WARN,
            ErrorDetails::InputValidation { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidBaseUrl { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidCandidate { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidDiclConfig { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidEpisodeId { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidFunctionVariants { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidMessage { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidOpenAICompatibleRequest { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidProviderConfig { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidRequest { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidTemplatePath => tracing::Level::ERROR,
            ErrorDetails::InvalidTool { .. } => tracing::Level::ERROR,
            ErrorDetails::JsonRequest { .. } => tracing::Level::WARN,
            ErrorDetails::JsonSchema { .. } => tracing::Level::ERROR,
            ErrorDetails::JsonSchemaValidation { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaEnvironment { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaTemplate { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaTemplateMissing { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaTemplateRender { .. } => tracing::Level::ERROR,
            ErrorDetails::ModelProvidersExhausted { .. } => tracing::Level::ERROR,
            ErrorDetails::ModelValidation { .. } => tracing::Level::ERROR,
            ErrorDetails::Observability { .. } => tracing::Level::ERROR,
            ErrorDetails::OutputParsing { .. } => tracing::Level::WARN,
            ErrorDetails::OutputValidation { .. } => tracing::Level::WARN,
            ErrorDetails::ProviderNotFound { .. } => tracing::Level::ERROR,
            ErrorDetails::Serialization { .. } => tracing::Level::ERROR,
            ErrorDetails::ToolNotFound { .. } => tracing::Level::WARN,
            ErrorDetails::ToolNotLoaded { .. } => tracing::Level::ERROR,
            ErrorDetails::TypeConversion { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownCandidate { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownFunction { .. } => tracing::Level::WARN,
            ErrorDetails::UnknownModel { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownTool { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownVariant { .. } => tracing::Level::WARN,
            ErrorDetails::UnknownMetric { .. } => tracing::Level::WARN,
            ErrorDetails::UnsupportedModelProviderForBatchInference { .. } => tracing::Level::WARN,
            ErrorDetails::UnsupportedVariantForBatchInference { .. } => tracing::Level::WARN,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            ErrorDetails::AllVariantsFailed { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ApiKeyMissing { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::AppState { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::BadCredentialsPreInference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::BatchInputValidation { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::ChannelWrite { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseConnection { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseMigration { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseQuery { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::Config { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::DynamicJsonSchema { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::GCPCredentials { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::Inference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InferenceClient { status_code, .. } => {
                status_code.unwrap_or_else(|| StatusCode::INTERNAL_SERVER_ERROR)
            }
            ErrorDetails::InferenceServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InferenceTimeout { .. } => StatusCode::REQUEST_TIMEOUT,
            ErrorDetails::InvalidEpisodeId { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InputValidation { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidBaseUrl { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidCandidate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidDiclConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
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
            ErrorDetails::ModelProvidersExhausted { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ModelValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::Observability { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::OutputParsing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::OutputValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ProviderNotFound { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::Serialization { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ToolNotFound { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::ToolNotLoaded { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::TypeConversion { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownCandidate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownFunction { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::UnknownModel { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownVariant { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::UnknownMetric { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::UnsupportedModelProviderForBatchInference { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            ErrorDetails::UnsupportedVariantForBatchInference { .. } => StatusCode::BAD_REQUEST,
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
            ErrorDetails::BadCredentialsPreInference { provider_name } => {
                write!(
                    f,
                    "Bad credentials at inference time for provider: {}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new",
                    provider_name
                )
            }
            ErrorDetails::BatchInputValidation { index, message } => {
                write!(f, "Input at index {} failed validation: {}", index, message,)
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
            ErrorDetails::GCPCredentials { message } => {
                write!(f, "Error in acquiring GCP credentials: {}", message)
            }
            ErrorDetails::Inference { message } => write!(f, "{}", message),
            ErrorDetails::InferenceClient {
                message,
                provider_type,
                ..
            } => write!(f, "Error from {} client: {}", provider_type, message),
            ErrorDetails::InferenceServer {
                message,
                provider_type,
                ..
            } => write!(f, "Error from {} server: {}", provider_type, message),
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
            ErrorDetails::InvalidDiclConfig { message } => {
                write!(f, "Invalid dynamic in-context learning config: {}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new", message)
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
            ErrorDetails::ToolNotFound { name } => write!(f, "Tool not found: {}", name),
            ErrorDetails::ToolNotLoaded { name } => write!(f, "Tool not loaded: {}", name),
            ErrorDetails::UnknownCandidate { name } => {
                write!(f, "Unknown candidate variant: {}", name)
            }
            ErrorDetails::UnknownFunction { name } => write!(f, "Unknown function: {}", name),
            ErrorDetails::UnknownModel { name } => write!(f, "Unknown model: {}", name),
            ErrorDetails::UnknownTool { name } => write!(f, "Unknown tool: {}", name),
            ErrorDetails::UnknownVariant { name } => write!(f, "Unknown variant: {}", name),
            ErrorDetails::UnknownMetric { name } => write!(f, "Unknown metric: {}", name),
            ErrorDetails::UnsupportedModelProviderForBatchInference { provider_type } => {
                write!(
                    f,
                    "Unsupported model provider for batch inference: {}",
                    provider_type
                )
            }
            ErrorDetails::UnsupportedVariantForBatchInference { variant_name } => {
                match variant_name {
                    Some(variant_name) => write!(
                        f,
                        "Unsupported variant for batch inference: {}",
                        variant_name
                    ),
                    None => write!(f, "Unsupported variant for batch inference"),
                }
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
