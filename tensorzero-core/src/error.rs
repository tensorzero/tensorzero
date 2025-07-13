use std::collections::HashMap;
use std::time::Duration;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::{json, Value};
use std::fmt::{Debug, Display};
use tokio::sync::OnceCell;
use url::Url;
use uuid::Uuid;

use crate::inference::types::storage::StoragePath;
use crate::inference::types::Thought;

/// Controls whether to include raw request/response details in error output
///
/// When true:
/// - Raw request/response details are logged for inference provider errors
/// - Raw details are included in error response bodies
/// - Most commonly affects errors from provider API requests/responses
///
/// WARNING: Setting this to true will expose potentially sensitive request/response
/// data in logs and error responses. Use with caution.
static DEBUG: OnceCell<bool> =
    if cfg!(feature = "e2e_tests") || cfg!(feature = "optimization_tests") {
        OnceCell::const_new_with(true)
    } else {
        OnceCell::const_new()
    };

pub fn set_debug(debug: bool) -> Result<(), Error> {
    // We already initialized `DEBUG`, so do nothing
    if cfg!(feature = "e2e_tests") {
        return Ok(());
    }
    DEBUG.set(debug).map_err(|_| {
        Error::new(ErrorDetails::Config {
            message: "Failed to set debug mode".to_string(),
        })
    })
}

pub fn warn_discarded_thought_block(provider_type: &str, thought: &Thought) {
    if *DEBUG.get().unwrap_or(&false) {
        tracing::warn!("Provider type `{provider_type}` does not support input thought blocks, discarding: {thought}");
    } else {
        tracing::warn!(
            "Provider type `{provider_type}` does not support input thought blocks, discarding"
        );
    }
}

pub fn warn_discarded_unknown_chunk(provider_type: &str, part: &str) {
    if *DEBUG.get().unwrap_or(&false) {
        tracing::warn!("Discarding unknown chunk in {provider_type} response: {part}");
    } else {
        tracing::warn!("Discarding unknown chunk in {provider_type} response");
    }
}

pub const IMPOSSIBLE_ERROR_MESSAGE: &str = "This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports";

/// Chooses between a `Debug` or `Display` representation based on the gateway-level `DEBUG` flag.
pub struct DisplayOrDebugGateway<T: Debug + Display> {
    val: T,
}

impl<T: Debug + Display> DisplayOrDebugGateway<T> {
    pub fn new(val: T) -> Self {
        Self { val }
    }
}

impl<T: Debug + Display> Display for DisplayOrDebugGateway<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *DEBUG.get().unwrap_or(&false) {
            write!(f, "{:?}", self.val)
        } else {
            write!(f, "{}", self.val)
        }
    }
}

#[derive(Debug, PartialEq)]
// As long as the struct member is private, we force people to use the `new` method and log the error.
// We box `ErrorDetails` per the `clippy::result_large_err` lint
pub struct Error(Box<ErrorDetails>);

impl Error {
    #[must_use]
    pub fn new(details: ErrorDetails) -> Self {
        details.log();
        Error(Box::new(details))
    }

    #[must_use]
    pub fn new_without_logging(details: ErrorDetails) -> Self {
        Error(Box::new(details))
    }

    #[must_use]
    pub fn status_code(&self) -> StatusCode {
        self.0.status_code()
    }

    #[must_use]
    pub fn get_details(&self) -> &ErrorDetails {
        &self.0
    }

    #[must_use]
    pub fn get_owned_details(self) -> ErrorDetails {
        *self.0
    }

    pub fn log(&self) {
        self.0.log();
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
    InvalidInferenceTarget {
        message: String,
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
    BatchNotFound {
        id: Uuid,
    },
    BadImageFetch {
        url: Url,
        message: String,
    },
    Cache {
        message: String,
    },
    ChannelWrite {
        message: String,
    },
    ClickHouseConnection {
        message: String,
    },
    ClickHouseDeserialization {
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
    ObjectStoreUnconfigured {
        block_type: String,
    },
    DatapointNotFound {
        dataset_name: String,
        datapoint_id: Uuid,
    },
    DuplicateTool {
        name: String,
    },
    DynamicJsonSchema {
        message: String,
    },
    FileRead {
        message: String,
        file_path: String,
    },
    GCPCredentials {
        message: String,
    },
    Inference {
        message: String,
    },
    InferenceClient {
        message: String,
        status_code: Option<StatusCode>,
        provider_type: String,
        raw_request: Option<String>,
        raw_response: Option<String>,
    },
    InferenceNotFound {
        inference_id: Uuid,
    },
    InferenceServer {
        message: String,
        provider_type: String,
        raw_request: Option<String>,
        raw_response: Option<String>,
    },
    InvalidClientMode {
        mode: String,
        message: String,
    },
    InvalidEncodedJobHandle,
    InvalidJobHandle {
        message: String,
    },
    InvalidInferenceOutputSource {
        source: String,
    },
    ObjectStoreWrite {
        message: String,
        path: StoragePath,
    },
    InternalError {
        message: String,
    },
    InferenceTimeout {
        variant_name: String,
    },
    VariantTimeout {
        variant_name: Option<String>,
        timeout: Duration,
        streaming: bool,
    },
    ModelTimeout {
        model_name: String,
        timeout: Duration,
        streaming: bool,
    },
    ModelProviderTimeout {
        provider_name: String,
        timeout: Duration,
        streaming: bool,
    },
    InputValidation {
        source: Box<Error>,
    },
    InvalidBatchParams {
        message: String,
    },
    InvalidBaseUrl {
        message: String,
    },
    InvalidCandidate {
        variant_name: String,
        message: String,
    },
    InvalidDatasetName {
        dataset_name: String,
    },
    InvalidDiclConfig {
        message: String,
    },
    InvalidDynamicEvaluationRun {
        episode_id: Uuid,
    },
    InvalidTensorzeroUuid {
        kind: String,
        message: String,
    },
    InvalidFunctionVariants {
        message: String,
    },
    InvalidMetricName {
        metric_name: String,
    },
    InvalidMessage {
        message: String,
    },
    InvalidModel {
        model_name: String,
    },
    InvalidModelProvider {
        model_name: String,
        provider_name: String,
    },
    InvalidOpenAICompatibleRequest {
        message: String,
    },
    InvalidProviderConfig {
        message: String,
    },
    InvalidRenderedStoredInference {
        message: String,
    },
    InvalidRequest {
        message: String,
    },
    InvalidTemplatePath,
    InvalidTool {
        message: String,
    },
    InvalidVariantForOptimization {
        function_name: String,
        variant_name: String,
    },
    InvalidValFraction {
        val_fraction: f64,
    },
    InvalidUuid {
        raw_uuid: String,
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
    MissingFunctionInVariants {
        function_name: String,
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
    MissingBatchInferenceResponse {
        inference_id: Option<Uuid>,
    },
    MissingFileExtension {
        file_name: String,
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
    OptimizationResponse {
        message: String,
        provider_type: String,
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
    ExtraBodyReplacement {
        message: String,
        pointer: String,
    },
    StreamError {
        source: Box<Error>,
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
    UnknownEvaluation {
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
    UnsupportedVariantForStreamingInference {
        variant_type: String,
        issue_link: Option<String>,
    },
    UnsupportedVariantForFunctionType {
        function_name: String,
        variant_name: String,
        function_type: String,
        variant_type: String,
    },
    UnsupportedContentBlockType {
        content_block_type: String,
        provider_type: String,
    },
    UuidInFuture {
        raw_uuid: String,
    },
    UnsupportedFileExtension {
        extension: String,
    },
    RouteNotFound {
        path: String,
        method: String,
    },
}

impl ErrorDetails {
    /// Defines the error level for logging this error
    fn level(&self) -> tracing::Level {
        match self {
            ErrorDetails::InvalidTemplatePath
            | ErrorDetails::AllVariantsFailed { .. }
            | ErrorDetails::ApiKeyMissing { .. }
            | ErrorDetails::AppState { .. }
            | ErrorDetails::BadCredentialsPreInference { .. }
            | ErrorDetails::BadImageFetch { .. }
            | ErrorDetails::ChannelWrite { .. }
            | ErrorDetails::ClickHouseConnection { .. }
            | ErrorDetails::ClickHouseDeserialization { .. }
            | ErrorDetails::ClickHouseMigration { .. }
            | ErrorDetails::ClickHouseQuery { .. }
            | ErrorDetails::Config { .. }
            | ErrorDetails::ExtraBodyReplacement { .. }
            | ErrorDetails::FileRead { .. }
            | ErrorDetails::GCPCredentials { .. }
            | ErrorDetails::Inference { .. }
            | ErrorDetails::InferenceClient { .. }
            | ErrorDetails::InferenceServer { .. }
            | ErrorDetails::InternalError { .. }
            | ErrorDetails::InvalidBaseUrl { .. }
            | ErrorDetails::InvalidBatchParams { .. }
            | ErrorDetails::InvalidCandidate { .. }
            | ErrorDetails::InvalidClientMode { .. }
            | ErrorDetails::InvalidDiclConfig { .. }
            | ErrorDetails::InvalidDynamicEvaluationRun { .. }
            | ErrorDetails::InvalidFunctionVariants { .. }
            | ErrorDetails::InvalidModel { .. }
            | ErrorDetails::InvalidModelProvider { .. }
            | ErrorDetails::InvalidOpenAICompatibleRequest { .. }
            | ErrorDetails::InvalidProviderConfig { .. }
            | ErrorDetails::InvalidRenderedStoredInference { .. }
            | ErrorDetails::InvalidTool { .. }
            | ErrorDetails::InvalidUuid { .. }
            | ErrorDetails::JsonSchema { .. }
            | ErrorDetails::JsonSchemaValidation { .. }
            | ErrorDetails::MiniJinjaEnvironment { .. }
            | ErrorDetails::MiniJinjaTemplate { .. }
            | ErrorDetails::MiniJinjaTemplateMissing { .. }
            | ErrorDetails::MiniJinjaTemplateRender { .. }
            | ErrorDetails::MissingFunctionInVariants { .. }
            | ErrorDetails::ModelProvidersExhausted { .. }
            | ErrorDetails::ModelValidation { .. }
            | ErrorDetails::ObjectStoreUnconfigured { .. }
            | ErrorDetails::ObjectStoreWrite { .. }
            | ErrorDetails::OptimizationResponse { .. }
            | ErrorDetails::ProviderNotFound { .. }
            | ErrorDetails::Serialization { .. }
            | ErrorDetails::StreamError { .. }
            | ErrorDetails::ToolNotLoaded { .. }
            | ErrorDetails::TypeConversion { .. }
            | ErrorDetails::UnknownCandidate { .. }
            | ErrorDetails::UnknownModel { .. }
            | ErrorDetails::UnknownTool { .. }
            | ErrorDetails::UnsupportedVariantForFunctionType { .. } => tracing::Level::ERROR,
            ErrorDetails::BatchInputValidation { .. }
            | ErrorDetails::BatchNotFound { .. }
            | ErrorDetails::Cache { .. }
            | ErrorDetails::DatapointNotFound { .. }
            | ErrorDetails::DuplicateTool { .. }
            | ErrorDetails::DynamicJsonSchema { .. }
            | ErrorDetails::InferenceNotFound { .. }
            | ErrorDetails::InferenceTimeout { .. }
            | ErrorDetails::InputValidation { .. }
            | ErrorDetails::InvalidDatasetName { .. }
            | ErrorDetails::InvalidEncodedJobHandle
            | ErrorDetails::InvalidInferenceOutputSource { .. }
            | ErrorDetails::InvalidInferenceTarget { .. }
            | ErrorDetails::InvalidJobHandle { .. }
            | ErrorDetails::InvalidMessage { .. }
            | ErrorDetails::InvalidMetricName { .. }
            | ErrorDetails::InvalidRequest { .. }
            | ErrorDetails::InvalidTensorzeroUuid { .. }
            | ErrorDetails::InvalidValFraction { .. }
            | ErrorDetails::InvalidVariantForOptimization { .. }
            | ErrorDetails::JsonRequest { .. }
            | ErrorDetails::MissingBatchInferenceResponse { .. }
            | ErrorDetails::MissingFileExtension { .. }
            | ErrorDetails::ModelProviderTimeout { .. }
            | ErrorDetails::ModelTimeout { .. }
            | ErrorDetails::Observability { .. }
            | ErrorDetails::OutputParsing { .. }
            | ErrorDetails::OutputValidation { .. }
            | ErrorDetails::RouteNotFound { .. }
            | ErrorDetails::ToolNotFound { .. }
            | ErrorDetails::UnknownEvaluation { .. }
            | ErrorDetails::UnknownFunction { .. }
            | ErrorDetails::UnknownMetric { .. }
            | ErrorDetails::UnknownVariant { .. }
            | ErrorDetails::UnsupportedContentBlockType { .. }
            | ErrorDetails::UnsupportedFileExtension { .. }
            | ErrorDetails::UnsupportedModelProviderForBatchInference { .. }
            | ErrorDetails::UnsupportedVariantForBatchInference { .. }
            | ErrorDetails::UnsupportedVariantForStreamingInference { .. }
            | ErrorDetails::UuidInFuture { .. }
            | ErrorDetails::VariantTimeout { .. } => tracing::Level::WARN,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            ErrorDetails::AllVariantsFailed { .. }
            | ErrorDetails::ApiKeyMissing { .. }
            | ErrorDetails::BatchInputValidation { .. }
            | ErrorDetails::DuplicateTool { .. }
            | ErrorDetails::DynamicJsonSchema { .. }
            | ErrorDetails::ExtraBodyReplacement { .. }
            | ErrorDetails::InputValidation { .. }
            | ErrorDetails::InvalidBatchParams { .. }
            | ErrorDetails::InvalidClientMode { .. }
            | ErrorDetails::InvalidDatasetName { .. }
            | ErrorDetails::InvalidDynamicEvaluationRun { .. }
            | ErrorDetails::InvalidEncodedJobHandle
            | ErrorDetails::InvalidInferenceOutputSource { .. }
            | ErrorDetails::InvalidInferenceTarget { .. }
            | ErrorDetails::InvalidJobHandle { .. }
            | ErrorDetails::InvalidMessage { .. }
            | ErrorDetails::InvalidMetricName { .. }
            | ErrorDetails::InvalidOpenAICompatibleRequest { .. }
            | ErrorDetails::InvalidRenderedStoredInference { .. }
            | ErrorDetails::InvalidRequest { .. }
            | ErrorDetails::InvalidTensorzeroUuid { .. }
            | ErrorDetails::InvalidUuid { .. }
            | ErrorDetails::InvalidValFraction { .. }
            | ErrorDetails::InvalidVariantForOptimization { .. }
            | ErrorDetails::JsonRequest { .. }
            | ErrorDetails::JsonSchemaValidation { .. }
            | ErrorDetails::MissingBatchInferenceResponse { .. }
            | ErrorDetails::MissingFileExtension { .. }
            | ErrorDetails::MissingFunctionInVariants { .. }
            | ErrorDetails::ToolNotFound { .. }
            | ErrorDetails::UnsupportedContentBlockType { .. }
            | ErrorDetails::UnsupportedFileExtension { .. }
            | ErrorDetails::UnsupportedVariantForBatchInference { .. }
            | ErrorDetails::UuidInFuture { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::AppState { .. }
            | ErrorDetails::BadCredentialsPreInference { .. }
            | ErrorDetails::BadImageFetch { .. }
            | ErrorDetails::Cache { .. }
            | ErrorDetails::ChannelWrite { .. }
            | ErrorDetails::ClickHouseConnection { .. }
            | ErrorDetails::ClickHouseDeserialization { .. }
            | ErrorDetails::ClickHouseMigration { .. }
            | ErrorDetails::ClickHouseQuery { .. }
            | ErrorDetails::Config { .. }
            | ErrorDetails::FileRead { .. }
            | ErrorDetails::GCPCredentials { .. }
            | ErrorDetails::Inference { .. }
            | ErrorDetails::InferenceServer { .. }
            | ErrorDetails::InternalError { .. }
            | ErrorDetails::InvalidBaseUrl { .. }
            | ErrorDetails::InvalidCandidate { .. }
            | ErrorDetails::InvalidDiclConfig { .. }
            | ErrorDetails::InvalidFunctionVariants { .. }
            | ErrorDetails::InvalidModel { .. }
            | ErrorDetails::InvalidModelProvider { .. }
            | ErrorDetails::InvalidProviderConfig { .. }
            | ErrorDetails::InvalidTemplatePath
            | ErrorDetails::InvalidTool { .. }
            | ErrorDetails::JsonSchema { .. }
            | ErrorDetails::MiniJinjaEnvironment { .. }
            | ErrorDetails::MiniJinjaTemplate { .. }
            | ErrorDetails::MiniJinjaTemplateMissing { .. }
            | ErrorDetails::MiniJinjaTemplateRender { .. }
            | ErrorDetails::ModelProvidersExhausted { .. }
            | ErrorDetails::ModelValidation { .. }
            | ErrorDetails::ObjectStoreUnconfigured { .. }
            | ErrorDetails::ObjectStoreWrite { .. }
            | ErrorDetails::Observability { .. }
            | ErrorDetails::OptimizationResponse { .. }
            | ErrorDetails::OutputParsing { .. }
            | ErrorDetails::OutputValidation { .. }
            | ErrorDetails::Serialization { .. }
            | ErrorDetails::StreamError { .. }
            | ErrorDetails::ToolNotLoaded { .. }
            | ErrorDetails::TypeConversion { .. }
            | ErrorDetails::UnknownCandidate { .. }
            | ErrorDetails::UnknownModel { .. }
            | ErrorDetails::UnknownTool { .. }
            | ErrorDetails::UnsupportedModelProviderForBatchInference { .. }
            | ErrorDetails::UnsupportedVariantForFunctionType { .. }
            | ErrorDetails::UnsupportedVariantForStreamingInference { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            ErrorDetails::BatchNotFound { .. }
            | ErrorDetails::DatapointNotFound { .. }
            | ErrorDetails::InferenceNotFound { .. }
            | ErrorDetails::ProviderNotFound { .. }
            | ErrorDetails::RouteNotFound { .. }
            | ErrorDetails::UnknownEvaluation { .. }
            | ErrorDetails::UnknownFunction { .. }
            | ErrorDetails::UnknownMetric { .. }
            | ErrorDetails::UnknownVariant { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::InferenceTimeout { .. }
            | ErrorDetails::ModelProviderTimeout { .. }
            | ErrorDetails::ModelTimeout { .. }
            | ErrorDetails::VariantTimeout { .. } => StatusCode::REQUEST_TIMEOUT,
            ErrorDetails::InferenceClient { status_code, .. } => {
                status_code.unwrap_or_else(|| StatusCode::INTERNAL_SERVER_ERROR)
            }
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
                        .map(|(variant_name, error)| format!("{variant_name}: {error}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            }
            ErrorDetails::ModelProviderTimeout {
                provider_name,
                timeout,
                streaming,
            } => {
                if *streaming {
                    write!(
                        f,
                        "Model provider {provider_name} timed out due to configured `streaming.ttft_ms` timeout ({timeout:?})"
                    )
                } else {
                    write!(
                        f,
                        "Model provider {provider_name} timed out due to configured `non_streaming.total_ms` timeout ({timeout:?})"
                    )
                }
            }
            ErrorDetails::ModelTimeout {
                model_name,
                timeout,
                streaming,
            } => {
                if *streaming {
                    write!(f, "Model {model_name} timed out due to configured `streaming.ttft_ms` timeout ({timeout:?})")
                } else {
                    write!(f, "Model {model_name} timed out due to configured `non_streaming.total_ms` timeout ({timeout:?})")
                }
            }
            ErrorDetails::VariantTimeout {
                variant_name,
                timeout,
                streaming,
            } => {
                let variant_description = if let Some(variant_name) = variant_name {
                    format!("Variant `{variant_name}`")
                } else {
                    "Unknown variant".to_string()
                };
                if *streaming {
                    write!(f, "{variant_description} timed out due to configured `streaming.ttft_ms` timeout ({timeout:?})")
                } else {
                    write!(f, "{variant_description} timed out due to configured `non_streaming.total_ms` timeout ({timeout:?})")
                }
            }
            ErrorDetails::ObjectStoreWrite { message, path } => {
                write!(
                    f,
                    "Error writing to object store: `{message}`. Path: {path:?}"
                )
            }
            ErrorDetails::InvalidInferenceTarget { message } => {
                write!(f, "Invalid inference target: {message}")
            }
            ErrorDetails::BadImageFetch { url, message } => {
                write!(f, "Error fetching image from {url}: {message}")
            }
            ErrorDetails::ObjectStoreUnconfigured { block_type } => {
                write!(f, "Object storage is not configured. You must configure `[object_storage]` before making requests containing a `{block_type}` content block. If you don't want to use object storage, you can explicitly set `object_storage.type = \"disabled\"` in your configuration.")
            }
            ErrorDetails::UnsupportedContentBlockType {
                content_block_type,
                provider_type,
            } => {
                write!(
                    f,
                    "Unsupported content block type `{content_block_type}` for provider `{provider_type}`",
                )
            }
            ErrorDetails::ExtraBodyReplacement { message, pointer } => {
                write!(
                    f,
                    "Error replacing extra body: `{message}` with pointer: `{pointer}`"
                )
            }
            ErrorDetails::ApiKeyMissing { provider_name } => {
                write!(f, "API key missing for provider: {provider_name}")
            }
            ErrorDetails::AppState { message } => {
                write!(f, "Error initializing AppState: {message}")
            }
            ErrorDetails::BadCredentialsPreInference { provider_name } => {
                write!(
                    f,
                    "Bad credentials at inference time for provider: {provider_name}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new"
                )
            }
            ErrorDetails::BatchInputValidation { index, message } => {
                write!(f, "Input at index {index} failed validation: {message}",)
            }
            ErrorDetails::BatchNotFound { id } => {
                write!(f, "Batch request not found for id: {id}")
            }
            ErrorDetails::Cache { message } => {
                write!(f, "Error in cache: {message}")
            }
            ErrorDetails::ChannelWrite { message } => {
                write!(f, "Error writing to channel: {message}")
            }
            ErrorDetails::ClickHouseConnection { message } => {
                write!(f, "Error connecting to ClickHouse: {message}")
            }
            ErrorDetails::ClickHouseDeserialization { message } => {
                write!(f, "Error deserializing ClickHouse response: {message}")
            }
            ErrorDetails::ClickHouseMigration { id, message } => {
                write!(f, "Error running ClickHouse migration {id}: {message}")
            }
            ErrorDetails::ClickHouseQuery { message } => {
                write!(f, "Failed to run ClickHouse query: {message}")
            }
            ErrorDetails::DatapointNotFound {
                dataset_name,
                datapoint_id,
            } => {
                write!(
                    f,
                    "Datapoint not found for dataset: {dataset_name} and id: {datapoint_id}"
                )
            }
            ErrorDetails::DuplicateTool { name } => {
                write!(f, "Duplicate tool name: {name}. Tool names must be unique.")
            }
            ErrorDetails::DynamicJsonSchema { message } => {
                write!(
                    f,
                    "Error in compiling client-provided JSON schema: {message}"
                )
            }
            ErrorDetails::FileRead { message, file_path } => {
                write!(f, "Error reading file {file_path}: {message}")
            }
            ErrorDetails::GCPCredentials { message } => {
                write!(f, "Error in acquiring GCP credentials: {message}")
            }
            ErrorDetails::Config { message }
            | ErrorDetails::Inference { message }
            | ErrorDetails::InvalidBatchParams { message }
            | ErrorDetails::InvalidFunctionVariants { message }
            | ErrorDetails::InvalidMessage { message }
            | ErrorDetails::InvalidProviderConfig { message }
            | ErrorDetails::InvalidRequest { message }
            | ErrorDetails::InvalidTool { message }
            | ErrorDetails::JsonRequest { message }
            | ErrorDetails::JsonSchema { message }
            | ErrorDetails::Serialization { message }
            | ErrorDetails::TypeConversion { message } => write!(f, "{message}"),
            ErrorDetails::InferenceClient {
                message,
                provider_type,
                raw_request,
                raw_response,
                status_code,
            } => {
                // `debug` defaults to false so we don't log raw request and response by default
                if *DEBUG.get().unwrap_or(&false) {
                    write!(
                        f,
                        "Error from {} client: {}{}{}",
                        provider_type,
                        message,
                        raw_request
                            .as_ref()
                            .map_or("".to_string(), |r| format!("\nRaw request: {r}")),
                        raw_response
                            .as_ref()
                            .map_or("".to_string(), |r| format!("\nRaw response: {r}"))
                    )
                } else {
                    write!(
                        f,
                        "Error{} from {} client: {}",
                        status_code.map_or("".to_string(), |s| format!(" {s}")),
                        provider_type,
                        message
                    )
                }
            }
            ErrorDetails::InferenceNotFound { inference_id } => {
                write!(f, "Inference not found for id: {inference_id}")
            }
            ErrorDetails::InferenceServer {
                message,
                provider_type,
                raw_request,
                raw_response,
            } => {
                // `debug` defaults to false so we don't log raw request and response by default
                if *DEBUG.get().unwrap_or(&false) {
                    write!(
                        f,
                        "Error from {} server: {}{}{}",
                        provider_type,
                        message,
                        raw_request
                            .as_ref()
                            .map_or("".to_string(), |r| format!("\nRaw request: {r}")),
                        raw_response
                            .as_ref()
                            .map_or("".to_string(), |r| format!("\nRaw response: {r}"))
                    )
                } else {
                    write!(f, "Error from {provider_type} server: {message}")
                }
            }
            ErrorDetails::InferenceTimeout { variant_name } => {
                write!(f, "Inference timed out for variant: {variant_name}")
            }
            ErrorDetails::InputValidation { source } => {
                write!(f, "Input validation failed with messages: {source}")
            }
            ErrorDetails::InternalError { message } => {
                write!(f, "Internal error: {message}")
            }
            ErrorDetails::InvalidBaseUrl { message } => {
                write!(f, "Invalid batch params retrieved from database: {message}")
            }
            ErrorDetails::InvalidCandidate {
                variant_name,
                message,
            } => {
                write!(
                    f,
                    "Invalid candidate variant as a component of variant {variant_name}: {message}"
                )
            }
            ErrorDetails::InvalidClientMode { mode, message } => {
                write!(f, "Invalid client mode: {mode}. {message}")
            }
            ErrorDetails::InvalidDiclConfig { message } => {
                write!(f, "Invalid dynamic in-context learning config: {message}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new")
            }
            ErrorDetails::InvalidDatasetName { dataset_name } => {
                write!(f, "Invalid dataset name: {dataset_name}. Datasets cannot be named \"builder\" or begin with \"tensorzero::\"")
            }
            ErrorDetails::InvalidDynamicEvaluationRun { episode_id } => {
                write!(
                    f,
                    "Dynamic evaluation run not found for episode id: {episode_id}",
                )
            }
            ErrorDetails::InvalidEncodedJobHandle => {
                write!(
                    f,
                    "Invalid encoded job handle. Failed to decode using URL-safe Base64."
                )
            }
            ErrorDetails::InvalidJobHandle { message } => {
                write!(f, "Failed to deserialize job handle: {message}")
            }
            ErrorDetails::InvalidTensorzeroUuid { message, kind } => {
                write!(f, "Invalid {kind} ID: {message}")
            }
            ErrorDetails::InvalidInferenceOutputSource { source } => {
                write!(f, "Invalid inference output source: {source}. Should be one of: \"inference\" or \"demonstration\".")
            }
            ErrorDetails::InvalidMetricName { metric_name } => {
                write!(f, "Invalid metric name: {metric_name}")
            }
            ErrorDetails::InvalidModel { model_name } => {
                write!(f, "Invalid model: {model_name}")
            }
            ErrorDetails::InvalidModelProvider {
                model_name,
                provider_name,
            } => {
                write!(
                    f,
                    "Invalid model provider: {provider_name} for model: {model_name}"
                )
            }
            ErrorDetails::InvalidValFraction { val_fraction } => {
                write!(
                    f,
                    "Invalid val fraction: {val_fraction}. Must be between 0 and 1."
                )
            }
            ErrorDetails::InvalidOpenAICompatibleRequest { message } => write!(
                f,
                "Invalid request to OpenAI-compatible endpoint: {message}"
            ),
            ErrorDetails::InvalidRenderedStoredInference { message } => {
                write!(f, "Invalid rendered stored inference: {message}")
            }
            ErrorDetails::InvalidTemplatePath => {
                write!(f, "Template path failed to convert to Rust string")
            }
            ErrorDetails::InvalidUuid { raw_uuid } => {
                write!(f, "Failed to parse UUID as v7: {raw_uuid}")
            }
            ErrorDetails::InvalidVariantForOptimization {
                function_name,
                variant_name,
            } => {
                write!(f, "Invalid variant for optimization: {variant_name} for function: {function_name}")
            }
            ErrorDetails::JsonSchemaValidation {
                messages,
                data,
                schema,
            } => {
                write!(f, "JSON Schema validation failed:\n{}", messages.join("\n"))?;
                // `debug` defaults to false so we don't log data by default
                if *DEBUG.get().unwrap_or(&false) {
                    write!(
                        f,
                        "\n\nData:\n{}",
                        serde_json::to_string(data).map_err(|_| std::fmt::Error)?
                    )?;
                }
                write!(
                    f,
                    "\n\nSchema:\n{}",
                    serde_json::to_string(schema).map_err(|_| std::fmt::Error)?
                )
            }
            ErrorDetails::MiniJinjaEnvironment { message } => {
                write!(f, "Error initializing MiniJinja environment: {message}")
            }
            ErrorDetails::MiniJinjaTemplate {
                template_name,
                message,
            }
            | ErrorDetails::MiniJinjaTemplateRender {
                template_name,
                message,
            } => {
                write!(f, "Error rendering template {template_name}: {message}")
            }
            ErrorDetails::MiniJinjaTemplateMissing { template_name } => {
                write!(f, "Template not found: {template_name}")
            }
            ErrorDetails::MissingBatchInferenceResponse { inference_id } => match inference_id {
                Some(inference_id) => write!(
                    f,
                    "Missing batch inference response for inference id: {inference_id}"
                ),
                None => write!(f, "Missing batch inference response"),
            },
            ErrorDetails::MissingFunctionInVariants { function_name } => {
                write!(f, "Missing function in variants: {function_name}")
            }
            ErrorDetails::MissingFileExtension { file_name } => {
                write!(
                    f,
                    "Could not determine file extension for file: {file_name}"
                )
            }
            ErrorDetails::ModelProvidersExhausted { provider_errors } => {
                write!(
                    f,
                    "All model providers failed to infer with errors: {}",
                    provider_errors
                        .iter()
                        .map(|(provider_name, error)| format!("{provider_name}: {error}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            ErrorDetails::ModelValidation { message } => {
                write!(f, "Failed to validate model: {message}")
            }
            ErrorDetails::Observability { message } => {
                write!(f, "{message}")
            }
            ErrorDetails::OptimizationResponse {
                message,
                provider_type,
            } => {
                write!(
                    f,
                    "Error from {provider_type} optimization response: {message}"
                )
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
                write!(f, "Output validation failed with messages: {source}")
            }
            ErrorDetails::ProviderNotFound { provider_name } => {
                write!(f, "Provider not found: {provider_name}")
            }
            ErrorDetails::StreamError { source } => {
                write!(f, "Error in streaming response: {source}")
            }
            ErrorDetails::ToolNotFound { name } => write!(f, "Tool not found: {name}"),
            ErrorDetails::ToolNotLoaded { name } => write!(f, "Tool not loaded: {name}"),
            ErrorDetails::UnknownCandidate { name } => {
                write!(f, "Unknown candidate variant: {name}")
            }
            ErrorDetails::UnknownEvaluation { name } => write!(f, "Unknown evaluation: {name}"),
            ErrorDetails::UnknownFunction { name } => write!(f, "Unknown function: {name}"),
            ErrorDetails::UnknownModel { name } => write!(f, "Unknown model: {name}"),
            ErrorDetails::UnknownTool { name } => write!(f, "Unknown tool: {name}"),
            ErrorDetails::UnknownVariant { name } => write!(f, "Unknown variant: {name}"),
            ErrorDetails::UnknownMetric { name } => write!(f, "Unknown metric: {name}"),
            ErrorDetails::UnsupportedModelProviderForBatchInference { provider_type } => {
                write!(
                    f,
                    "Unsupported model provider for batch inference: {provider_type}"
                )
            }
            ErrorDetails::UnsupportedFileExtension { extension } => {
                write!(f, "Unsupported file extension: {extension}")
            }
            ErrorDetails::UnsupportedVariantForBatchInference { variant_name } => {
                match variant_name {
                    Some(variant_name) => {
                        write!(f, "Unsupported variant for batch inference: {variant_name}")
                    }
                    None => write!(f, "Unsupported variant for batch inference"),
                }
            }
            ErrorDetails::UnsupportedVariantForStreamingInference {
                variant_type,
                issue_link,
            } => {
                if let Some(link) = issue_link {
                    write!(
                        f,
                        "Unsupported variant for streaming inference of type {variant_type}. For more information, see: {link}"
                    )
                } else {
                    write!(
                        f,
                        "Unsupported variant for streaming inference of type {variant_type}"
                    )
                }
            }
            ErrorDetails::UnsupportedVariantForFunctionType {
                function_name,
                variant_name,
                function_type,
                variant_type,
            } => {
                write!(f, "Unsupported variant `{variant_name}` of type `{variant_type}` for function `{function_name}` of type `{function_type}`")
            }
            ErrorDetails::UuidInFuture { raw_uuid } => {
                write!(f, "UUID is in the future: {raw_uuid}")
            }
            ErrorDetails::RouteNotFound { path, method } => {
                write!(f, "Route not found: {method} {path}")
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
