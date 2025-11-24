use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use indexmap::IndexMap;
use minijinja_utils::AnalysisError;
use opentelemetry::trace::Status;
use serde::{Serialize, Serializer};
use serde_json::{json, Value};
use std::fmt::{Debug, Display};
use thiserror::Error;
use tokio::sync::OnceCell;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use url::Url;
use uuid::Uuid;

use crate::db::clickhouse::migration_manager::get_run_migrations_command;
use crate::inference::types::storage::StoragePath;
use crate::inference::types::Thought;
use crate::rate_limiting::{FailedRateLimit, RateLimitingConfigScopes};

pub mod delayed_error;
pub use delayed_error::DelayedError;

/// Controls whether to include raw request/response details in error output
///
/// When true:
/// - Raw request/response details are logged for inference provider errors
/// - Raw details are included in error response bodies
/// - Most commonly affects errors from provider API requests/responses
///
/// WARNING: Setting this to true will expose potentially sensitive request/response
/// data in logs and error responses. Use with caution.
static DEBUG: OnceCell<bool> = if cfg!(feature = "e2e_tests") {
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

static UNSTABLE_ERROR_JSON: OnceCell<bool> = OnceCell::const_new();

pub fn set_unstable_error_json(unstable_error_json: bool) -> Result<(), Error> {
    UNSTABLE_ERROR_JSON.set(unstable_error_json).map_err(|_| {
        Error::new(ErrorDetails::Config {
            message: "Failed to set unstable error JSON".to_string(),
        })
    })
}

pub fn warn_discarded_cache_write(raw_response: &str) {
    if *DEBUG.get().unwrap_or(&false) {
        tracing::warn!("Skipping cache write due to invalid output:\nRaw response: {raw_response}");
    } else {
        tracing::warn!("Skipping cache write due to invalid output");
    }
}

pub fn warn_discarded_thought_block(provider_type: &str, thought: &Thought) {
    if *DEBUG.get().unwrap_or(&false) {
        tracing::warn!(
            "TensorZero doesn't support input thought blocks for the `{provider_type}` provider. Many providers don't support them; if this provider does, please let us know: https://github.com/tensorzero/tensorzero/discussions/categories/feature-requests\n\n{thought:?}"
        );
    } else {
        tracing::warn!(
            "TensorZero doesn't support input thought blocks for the `{provider_type}` provider. Many providers don't support them; if this provider does, please let us know: https://github.com/tensorzero/tensorzero/discussions/categories/feature-requests"
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

#[derive(Clone, Debug, Error, Serialize)]
#[cfg_attr(any(test, feature = "e2e_tests"), derive(PartialEq))]
#[error(transparent)]
// As long as the struct member is private, we force people to use the `new` method and log the error.
// We arc `ErrorDetails` per the `clippy::result_large_err` lint, as well as to make it cloneable
pub struct Error(Arc<ErrorDetails>);

impl Error {
    pub fn new(details: ErrorDetails) -> Self {
        details.log();
        Error(Arc::new(details))
    }

    // If you need to construct an error without logging it, use `DelayedError` instead.
    // This method should only be called within `DelayedError` itself.
    fn new_without_logging(details: ErrorDetails) -> Self {
        Error(Arc::new(details))
    }

    pub fn status_code(&self) -> StatusCode {
        self.0.status_code()
    }

    pub fn underlying_status_code(&self) -> Option<StatusCode> {
        self.0.underlying_status_code()
    }

    pub fn get_details(&self) -> &ErrorDetails {
        &self.0
    }

    /// Ensures that the OpenTelemetry span corresponding to `span` is marked as an error.
    /// If our level is `ERROR`, then we'll do nothing, since logging an error automatically marks the span as an error.
    /// If our level is anything else, then we explicitly mark the span as an error using our own messages
    /// This is used by callers that only want to log a warning to the console, but want an error to show up in OpenTelemetry
    /// for a particular span.
    pub fn ensure_otel_span_errored(&self, span: &Span) {
        if self.0.level() != tracing::Level::ERROR {
            span.set_status(Status::Error {
                description: self.to_string().into(),
            });
        }
    }

    pub fn log(&self) {
        self.0.log();
    }

    pub fn log_at_level(&self, prefix: &str, level: tracing::Level) {
        self.0.log_at_level(prefix, level);
    }

    pub fn is_retryable(&self) -> bool {
        self.0.is_retryable()
    }
}

// Expect for derive Serialize
#[expect(clippy::trivially_copy_pass_by_ref)]
fn serialize_status<S>(code: &Option<StatusCode>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match code {
        Some(c) => serializer.serialize_u16(c.as_u16()),
        None => serializer.serialize_none(),
    }
}

fn serialize_if_debug<T, S>(data: T, serializer: S) -> Result<S::Ok, S::Error>
where
    T: Serialize,
    S: Serializer,
{
    if *DEBUG.get().unwrap_or(&false) {
        return data.serialize(serializer);
    }
    serializer.serialize_none()
}

impl From<ErrorDetails> for Error {
    fn from(details: ErrorDetails) -> Self {
        Error::new(details)
    }
}

#[derive(Debug, Error, Serialize)]
#[cfg_attr(any(test, feature = "e2e_tests"), derive(PartialEq))]
pub enum ErrorDetails {
    AllVariantsFailed {
        // We use an `IndexMap` to preserve the insertion order for `underlying_status_code`
        errors: IndexMap<String, Error>,
    },
    TensorZeroAuth {
        message: String,
    },
    InvalidInferenceTarget {
        message: String,
    },
    ApiKeyMissing {
        provider_name: String,
        message: String,
    },
    AppState {
        message: String,
    },
    BadCredentialsPreInference {
        provider_name: String,
    },
    Base64 {
        message: String,
    },
    BatchInputValidation {
        index: usize,
        message: String,
    },
    BatchNotFound {
        id: Uuid,
    },
    BadFileFetch {
        url: Url,
        message: String,
    },
    Cache {
        message: String,
    },
    Glob {
        glob: String,
        message: String,
    },
    ChannelWrite {
        message: String,
    },
    ClickHouseConfiguration {
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
    ClickHouseMigrationsDisabled,
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
    DiclMissingOutput,
    DuplicateTool {
        name: String,
    },
    DuplicateRateLimitingConfigScope {
        scope: RateLimitingConfigScopes,
    },
    DynamicEndpointNotFound {
        key_name: String,
    },
    DynamicJsonSchema {
        message: String,
    },
    DynamicTemplateLoad {
        internal: AnalysisError,
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
        #[serde(serialize_with = "serialize_status")]
        status_code: Option<StatusCode>,
        provider_type: String,
        #[serde(serialize_with = "serialize_if_debug")]
        raw_request: Option<String>,
        #[serde(serialize_with = "serialize_if_debug")]
        raw_response: Option<String>,
    },
    InferenceNotFound {
        inference_id: Uuid,
    },
    FatalStreamError {
        message: String,
        provider_type: String,
        #[serde(serialize_with = "serialize_if_debug")]
        raw_request: Option<String>,
        #[serde(serialize_with = "serialize_if_debug")]
        raw_response: Option<String>,
    },
    InferenceServer {
        message: String,
        provider_type: String,
        #[serde(serialize_with = "serialize_if_debug")]
        raw_request: Option<String>,
        #[serde(serialize_with = "serialize_if_debug")]
        raw_response: Option<String>,
    },
    InvalidClientMode {
        mode: String,
        message: String,
    },
    InvalidDynamicEndpoint {
        url: String,
    },
    InvalidEncodedJobHandle,
    InvalidJobHandle {
        message: String,
    },
    InvalidInferenceOutputSource {
        source_kind: String,
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
        variant_name: String,
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
    InvalidWorkflowEvaluationRun {
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
    ModelNotFound {
        model_name: String,
    },
    ModelProvidersExhausted {
        // We use an `IndexMap` to preserve the insertion order for `underlying_status_code`
        provider_errors: IndexMap<String, Error>,
    },
    ModelValidation {
        message: String,
    },
    NoFallbackVariantsRemaining,
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
    // TODO(shuyang): once 3496 merges: change all of these to taking sqlx errors rather than string. Note also sqlx::Error is not Serialize?
    PostgresConnectionInitialization {
        message: String,
    },
    PostgresConnection {
        message: String,
    },
    PostgresMigration {
        message: String,
    },
    PostgresQuery {
        function_name: Option<String>,
        message: String,
    },
    PostgresResult {
        result_type: &'static str,
        message: String,
    },
    ProviderNotFound {
        provider_name: String,
    },
    RateLimitExceeded {
        failed_rate_limits: Vec<FailedRateLimit>,
    },
    RateLimitMissingMaxTokens,
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
    IncompatibleTool {
        message: String,
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
    UnsupportedModelProviderForStreamingInference {
        provider_type: String,
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
            ErrorDetails::AllVariantsFailed { .. } => tracing::Level::ERROR,
            ErrorDetails::TensorZeroAuth { .. } => tracing::Level::WARN,
            ErrorDetails::ApiKeyMissing { .. } => tracing::Level::ERROR,
            ErrorDetails::AppState { .. } => tracing::Level::ERROR,
            ErrorDetails::ObjectStoreUnconfigured { .. } => tracing::Level::ERROR,
            ErrorDetails::ExtraBodyReplacement { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidInferenceTarget { .. } => tracing::Level::WARN,
            ErrorDetails::BadCredentialsPreInference { .. } => tracing::Level::ERROR,
            ErrorDetails::Base64 { .. } => tracing::Level::ERROR,
            ErrorDetails::UnsupportedContentBlockType { .. } => tracing::Level::WARN,
            ErrorDetails::BatchInputValidation { .. } => tracing::Level::WARN,
            ErrorDetails::BatchNotFound { .. } => tracing::Level::WARN,
            ErrorDetails::Cache { .. } => tracing::Level::WARN,
            ErrorDetails::ChannelWrite { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseConnection { .. } => tracing::Level::ERROR,
            ErrorDetails::BadFileFetch { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseConfiguration { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseDeserialization { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseMigration { .. } => tracing::Level::ERROR,
            ErrorDetails::ClickHouseMigrationsDisabled => tracing::Level::ERROR,
            ErrorDetails::ClickHouseQuery { .. } => tracing::Level::ERROR,
            ErrorDetails::ObjectStoreWrite { .. } => tracing::Level::ERROR,
            ErrorDetails::Config { .. } => tracing::Level::ERROR,
            ErrorDetails::DatapointNotFound { .. } => tracing::Level::WARN,
            ErrorDetails::DiclMissingOutput => tracing::Level::ERROR,
            ErrorDetails::DuplicateTool { .. } => tracing::Level::WARN,
            ErrorDetails::DuplicateRateLimitingConfigScope { .. } => tracing::Level::WARN,
            ErrorDetails::DynamicJsonSchema { .. } => tracing::Level::WARN,
            ErrorDetails::DynamicEndpointNotFound { .. } => tracing::Level::WARN,
            ErrorDetails::DynamicTemplateLoad { .. } => tracing::Level::ERROR,
            ErrorDetails::FileRead { .. } => tracing::Level::ERROR,
            ErrorDetails::GCPCredentials { .. } => tracing::Level::ERROR,
            ErrorDetails::Inference { .. } => tracing::Level::ERROR,
            ErrorDetails::InferenceClient { .. } => tracing::Level::ERROR,
            ErrorDetails::InferenceNotFound { .. } => tracing::Level::WARN,
            ErrorDetails::InferenceServer { .. } => tracing::Level::ERROR,
            ErrorDetails::FatalStreamError { .. } => tracing::Level::ERROR,
            ErrorDetails::InferenceTimeout { .. } => tracing::Level::WARN,
            ErrorDetails::ModelProviderTimeout { .. } => tracing::Level::WARN,
            ErrorDetails::ModelTimeout { .. } => tracing::Level::WARN,
            ErrorDetails::VariantTimeout { .. } => tracing::Level::WARN,
            ErrorDetails::InputValidation { .. } => tracing::Level::WARN,
            ErrorDetails::InternalError { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidBaseUrl { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidBatchParams { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidCandidate { .. } => tracing::Level::ERROR,
            ErrorDetails::Glob { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidClientMode { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidDiclConfig { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidDatasetName { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidWorkflowEvaluationRun { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidInferenceOutputSource { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidTensorzeroUuid { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidFunctionVariants { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidVariantForOptimization { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidEncodedJobHandle => tracing::Level::WARN,
            ErrorDetails::InvalidJobHandle { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidRenderedStoredInference { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidMetricName { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidMessage { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidModel { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidModelProvider { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidOpenAICompatibleRequest { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidProviderConfig { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidRequest { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidTemplatePath => tracing::Level::ERROR,
            ErrorDetails::InvalidTool { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidUuid { .. } => tracing::Level::ERROR,
            ErrorDetails::InvalidDynamicEndpoint { .. } => tracing::Level::WARN,
            ErrorDetails::InvalidValFraction { .. } => tracing::Level::WARN,
            ErrorDetails::JsonRequest { .. } => tracing::Level::WARN,
            ErrorDetails::JsonSchema { .. } => tracing::Level::ERROR,
            ErrorDetails::JsonSchemaValidation { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaEnvironment { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaTemplate { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaTemplateMissing { .. } => tracing::Level::ERROR,
            ErrorDetails::MiniJinjaTemplateRender { .. } => tracing::Level::ERROR,
            ErrorDetails::MissingFunctionInVariants { .. } => tracing::Level::ERROR,
            ErrorDetails::MissingBatchInferenceResponse { .. } => tracing::Level::WARN,
            ErrorDetails::MissingFileExtension { .. } => tracing::Level::WARN,
            ErrorDetails::ModelProvidersExhausted { .. } => tracing::Level::ERROR,
            ErrorDetails::ModelNotFound { .. } => tracing::Level::WARN,
            ErrorDetails::ModelValidation { .. } => tracing::Level::ERROR,
            ErrorDetails::NoFallbackVariantsRemaining => tracing::Level::WARN,
            ErrorDetails::Observability { .. } => tracing::Level::WARN,
            ErrorDetails::OutputParsing { .. } => tracing::Level::WARN,
            ErrorDetails::OutputValidation { .. } => tracing::Level::WARN,
            ErrorDetails::OptimizationResponse { .. } => tracing::Level::ERROR,
            ErrorDetails::ProviderNotFound { .. } => tracing::Level::ERROR,
            ErrorDetails::PostgresConnectionInitialization { .. } => tracing::Level::ERROR,
            ErrorDetails::PostgresConnection { .. } => tracing::Level::ERROR,
            ErrorDetails::PostgresMigration { .. } => tracing::Level::ERROR,
            ErrorDetails::PostgresResult { .. } => tracing::Level::ERROR,
            ErrorDetails::PostgresQuery { .. } => tracing::Level::ERROR,
            ErrorDetails::RateLimitExceeded { .. } => tracing::Level::WARN,
            ErrorDetails::RateLimitMissingMaxTokens => tracing::Level::WARN,
            ErrorDetails::Serialization { .. } => tracing::Level::ERROR,
            ErrorDetails::StreamError { .. } => tracing::Level::ERROR,
            ErrorDetails::ToolNotFound { .. } => tracing::Level::WARN,
            ErrorDetails::ToolNotLoaded { .. } => tracing::Level::ERROR,
            ErrorDetails::IncompatibleTool { .. } => tracing::Level::WARN,
            ErrorDetails::TypeConversion { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownCandidate { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownFunction { .. } => tracing::Level::WARN,
            ErrorDetails::UnknownEvaluation { .. } => tracing::Level::WARN,
            ErrorDetails::UnknownModel { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownTool { .. } => tracing::Level::ERROR,
            ErrorDetails::UnknownVariant { .. } => tracing::Level::WARN,
            ErrorDetails::UnknownMetric { .. } => tracing::Level::WARN,
            ErrorDetails::UnsupportedFileExtension { .. } => tracing::Level::WARN,
            ErrorDetails::UnsupportedModelProviderForBatchInference { .. } => tracing::Level::WARN,
            ErrorDetails::UnsupportedModelProviderForStreamingInference { .. } => {
                tracing::Level::ERROR
            }
            ErrorDetails::UnsupportedVariantForBatchInference { .. } => tracing::Level::WARN,
            ErrorDetails::UnsupportedVariantForFunctionType { .. } => tracing::Level::ERROR,
            ErrorDetails::UnsupportedVariantForStreamingInference { .. } => tracing::Level::WARN,
            ErrorDetails::UuidInFuture { .. } => tracing::Level::WARN,
            ErrorDetails::RouteNotFound { .. } => tracing::Level::WARN,
        }
    }

    /// Returns the most recent 'underlying' status code for this error.
    /// For example, if an inference fails due to all models failing, this will
    /// return the status code of the last model that failed.
    ///
    /// Returns `None` if the error doesn't have a concept of a 'last' status code.
    fn underlying_status_code(&self) -> Option<StatusCode> {
        match self {
            ErrorDetails::AllVariantsFailed { errors } => errors
                .values()
                .last()
                .and_then(|error| error.underlying_status_code()),
            ErrorDetails::InferenceClient { status_code, .. } => *status_code,
            ErrorDetails::ModelProvidersExhausted { provider_errors } => provider_errors
                .values()
                .last()
                .and_then(|error| error.underlying_status_code()),
            _ => None,
        }
    }

    /// Defines the HTTP status code for responses involving this error
    fn status_code(&self) -> StatusCode {
        match self {
            ErrorDetails::AllVariantsFailed { .. } => StatusCode::BAD_GATEWAY,
            ErrorDetails::TensorZeroAuth { .. } => StatusCode::UNAUTHORIZED,
            ErrorDetails::ApiKeyMissing { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::Glob { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ExtraBodyReplacement { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::AppState { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::BadCredentialsPreInference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::BatchInputValidation { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::BatchNotFound { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::Cache { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ChannelWrite { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseConfiguration { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseConnection { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseDeserialization { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseMigration { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseMigrationsDisabled => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ClickHouseQuery { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ObjectStoreUnconfigured { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::DatapointNotFound { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::Config { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::DiclMissingOutput => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::DuplicateTool { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::DuplicateRateLimitingConfigScope { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::DynamicJsonSchema { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::DynamicTemplateLoad { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::DynamicEndpointNotFound { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::FileRead { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::GCPCredentials { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidInferenceTarget { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::Inference { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ObjectStoreWrite { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InferenceClient { status_code, .. } => {
                status_code.unwrap_or_else(|| StatusCode::INTERNAL_SERVER_ERROR)
            }
            ErrorDetails::Base64 { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::BadFileFetch { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InferenceNotFound { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::InferenceServer { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::FatalStreamError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InferenceTimeout { .. } => StatusCode::REQUEST_TIMEOUT,
            ErrorDetails::ModelProviderTimeout { .. } => StatusCode::REQUEST_TIMEOUT,
            ErrorDetails::ModelTimeout { .. } => StatusCode::REQUEST_TIMEOUT,
            ErrorDetails::VariantTimeout { .. } => StatusCode::REQUEST_TIMEOUT,
            ErrorDetails::InvalidClientMode { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidEncodedJobHandle => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidJobHandle { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidTensorzeroUuid { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidUuid { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InputValidation { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InternalError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidBaseUrl { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidValFraction { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::UnsupportedContentBlockType { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidBatchParams { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidCandidate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidDiclConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidDatasetName { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidDynamicEndpoint { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidWorkflowEvaluationRun { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidFunctionVariants { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidInferenceOutputSource { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidMessage { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidMetricName { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidModel { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidModelProvider { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidOpenAICompatibleRequest { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidProviderConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidRenderedStoredInference { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::InvalidTemplatePath => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::InvalidVariantForOptimization { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::JsonRequest { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::JsonSchema { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::JsonSchemaValidation { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::MiniJinjaEnvironment { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::MiniJinjaTemplate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::MiniJinjaTemplateMissing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::MiniJinjaTemplateRender { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::MissingBatchInferenceResponse { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::MissingFunctionInVariants { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::MissingFileExtension { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::ModelNotFound { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::ModelProvidersExhausted { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ModelValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::NoFallbackVariantsRemaining => StatusCode::BAD_GATEWAY,
            ErrorDetails::Observability { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::OptimizationResponse { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::OutputParsing { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::OutputValidation { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ProviderNotFound { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::PostgresConnectionInitialization { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            ErrorDetails::PostgresConnection { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::PostgresQuery { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::PostgresResult { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::PostgresMigration { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::RateLimitExceeded { .. } => StatusCode::TOO_MANY_REQUESTS,
            ErrorDetails::RateLimitMissingMaxTokens => StatusCode::BAD_REQUEST,
            ErrorDetails::Serialization { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::StreamError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::ToolNotFound { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::ToolNotLoaded { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::IncompatibleTool { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::TypeConversion { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownCandidate { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownFunction { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::UnknownEvaluation { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::UnknownModel { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownTool { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorDetails::UnknownVariant { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::UnknownMetric { .. } => StatusCode::NOT_FOUND,
            ErrorDetails::UnsupportedFileExtension { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::UnsupportedModelProviderForBatchInference { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            ErrorDetails::UnsupportedModelProviderForStreamingInference { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            ErrorDetails::UnsupportedVariantForBatchInference { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::UnsupportedVariantForStreamingInference { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            ErrorDetails::UnsupportedVariantForFunctionType { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            ErrorDetails::UuidInFuture { .. } => StatusCode::BAD_REQUEST,
            ErrorDetails::RouteNotFound { .. } => StatusCode::NOT_FOUND,
        }
    }

    pub fn log_at_level(&self, prefix: &str, level: tracing::Level) {
        match level {
            tracing::Level::ERROR => tracing::error!("{prefix}{self}"),
            tracing::Level::WARN => tracing::warn!("{prefix}{self}"),
            tracing::Level::INFO => tracing::info!("{prefix}{self}"),
            tracing::Level::DEBUG => tracing::debug!("{prefix}{self}"),
            tracing::Level::TRACE => tracing::trace!("{prefix}{self}"),
        }
    }

    /// Log the error using the `tracing` library
    pub fn log(&self) {
        self.log_at_level("", self.level());
    }

    pub fn is_retryable(&self) -> bool {
        match &self {
            ErrorDetails::RateLimitExceeded { .. } => false,
            // For ModelProvidersExhausted we will retry if any provider error is retryable
            ErrorDetails::ModelProvidersExhausted { provider_errors } => provider_errors
                .iter()
                .any(|(_, error)| error.is_retryable()),
            _ => true,
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
            ErrorDetails::TensorZeroAuth { message } => {
                write!(f, "TensorZero authentication error: {message}")
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
                let variant_description = format!("Variant `{variant_name}`");
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
            ErrorDetails::BadFileFetch { url, message } => {
                write!(f, "Error fetching file from {url}: {message}")
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
            ErrorDetails::Glob { glob, message } => {
                write!(f, "Error using glob: `{glob}`: {message}")
            }
            ErrorDetails::ApiKeyMissing {
                provider_name,
                message,
            } => {
                write!(f, "API key missing for provider {provider_name}: {message}")
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
            ErrorDetails::Base64 { message } => {
                write!(f, "Error decoding base64: {message}")
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
            ErrorDetails::ClickHouseConfiguration { message } => {
                write!(f, "Error in ClickHouse configuration: {message}")
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
            ErrorDetails::ClickHouseMigrationsDisabled => {
                let run_migrations_command: String = get_run_migrations_command();
                write!(f, "Automatic ClickHouse migrations were disabled, but not all migrations were run. Please run `{run_migrations_command}`")
            }
            ErrorDetails::ClickHouseQuery { message } => {
                write!(f, "Failed to run ClickHouse query: {message}")
            }
            ErrorDetails::Config { message } => {
                write!(f, "{message}")
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
            ErrorDetails::DiclMissingOutput => {
                write!(f, "DICL example missing output. There was a bug in a notebook from 2025-08 that may have caused the output to not be written to ClickHouse. You can remove the examples with missing output by running the query `DELETE FROM DynamicInContextLearningExample WHERE empty(output)`.")
            }
            ErrorDetails::DuplicateTool { name } => {
                write!(f, "Duplicate tool name: {name}. Tool names must be unique.")
            }
            ErrorDetails::DuplicateRateLimitingConfigScope { scope } => {
                write!(f, "Duplicate rate limiting config scope: {scope:?}. Rate limiting config scopes must be unique.")
            }
            ErrorDetails::DynamicJsonSchema { message } => {
                write!(
                    f,
                    "Error in compiling client-provided JSON schema: {message}"
                )
            }
            ErrorDetails::DynamicEndpointNotFound { key_name } => {
                write!(f, "Dynamic endpoint '{key_name}' not found in credentials")
            }
            ErrorDetails::DynamicTemplateLoad { internal } => match internal {
                AnalysisError::ParseError(err) => {
                    write!(
                        f,
                        "Failed to parse template during validation of loads: {err}"
                    )
                }
                AnalysisError::DynamicLoadsFound(loads) => {
                    writeln!(
                        f,
                        "TensorZero does not allow templates with dynamic paths to be loaded. Found {} dynamic load(s):",
                        loads.len()
                    )?;
                    for (i, load) in loads.iter().enumerate() {
                        if i > 0 {
                            writeln!(f)?;
                        }
                        write!(
                            f,
                            "  {}:{}:{}: dynamic {} - {}:\n    {}",
                            load.template_name,
                            load.line,
                            load.column,
                            load.load_kind,
                            load.reason,
                            load.source_quote
                        )?;
                    }
                    writeln!(f, "Please use explicit paths to templates instead of variables. You may be able to use if / else statements to achieve the desired behavior.")?;
                    Ok(())
                }
            },

            ErrorDetails::FileRead { message, file_path } => {
                write!(f, "Error reading file {file_path}: {message}")
            }
            ErrorDetails::GCPCredentials { message } => {
                write!(f, "Error in acquiring GCP credentials: {message}")
            }
            ErrorDetails::Inference { message } => write!(f, "{message}"),
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
                            .map_or(String::new(), |r| format!("\nRaw request: {r}")),
                        raw_response
                            .as_ref()
                            .map_or(String::new(), |r| format!("\nRaw response: {r}"))
                    )
                } else {
                    write!(
                        f,
                        "Error{} from {} client: {}",
                        status_code.map_or(String::new(), |s| format!(" {s}")),
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
                            .map_or(String::new(), |r| format!("\nRaw request: {r}")),
                        raw_response
                            .as_ref()
                            .map_or(String::new(), |r| format!("\nRaw response: {r}"))
                    )
                } else {
                    write!(f, "Error from {provider_type} server: {message}")
                }
            }
            ErrorDetails::FatalStreamError {
                message,
                provider_type,
                raw_request,
                raw_response,
            } => {
                // `debug` defaults to false so we don't log raw request and response by default
                if *DEBUG.get().unwrap_or(&false) {
                    write!(
                        f,
                        "Inference stream closed due to error from {} server: {}{}{}",
                        provider_type,
                        message,
                        raw_request
                            .as_ref()
                            .map_or(String::new(), |r| format!("\nRaw request: {r}")),
                        raw_response
                            .as_ref()
                            .map_or(String::new(), |r| format!("\nRaw response: {r}"))
                    )
                } else {
                    write!(f, "Inference stream closed due to error from {provider_type} server: {message}")
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
            ErrorDetails::InvalidBatchParams { message } => write!(f, "{message}"),
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
            ErrorDetails::InvalidWorkflowEvaluationRun { episode_id } => {
                write!(
                    f,
                    "Workflow evaluation run not found for episode id: {episode_id}",
                )
            }
            ErrorDetails::InvalidDynamicEndpoint { url } => {
                write!(f, "Invalid dynamic endpoint URL: {url}")
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
            ErrorDetails::InvalidFunctionVariants { message } => write!(f, "{message}"),
            ErrorDetails::InvalidTensorzeroUuid { message, kind } => {
                write!(f, "Invalid {kind} ID: {message}")
            }
            ErrorDetails::InvalidInferenceOutputSource { source_kind } => {
                write!(f, "Invalid inference output source: {source_kind}. Should be one of: \"inference\" or \"demonstration\".")
            }
            ErrorDetails::InvalidMetricName { metric_name } => {
                write!(f, "Invalid metric name: {metric_name}")
            }
            ErrorDetails::InvalidMessage { message } => write!(f, "{message}"),
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
            ErrorDetails::InvalidProviderConfig { message } => write!(f, "{message}"),
            ErrorDetails::InvalidRequest { message } => write!(f, "{message}"),
            ErrorDetails::InvalidRenderedStoredInference { message } => {
                write!(f, "Invalid rendered stored inference: {message}")
            }
            ErrorDetails::InvalidTemplatePath => {
                write!(f, "Template path failed to convert to Rust string")
            }
            ErrorDetails::InvalidTool { message } => write!(f, "{message}"),
            ErrorDetails::InvalidUuid { raw_uuid } => {
                write!(f, "Failed to parse UUID as v7: {raw_uuid}")
            }
            ErrorDetails::InvalidVariantForOptimization {
                function_name,
                variant_name,
            } => {
                write!(f, "Invalid variant for optimization: {variant_name} for function: {function_name}")
            }
            ErrorDetails::JsonRequest { message } => write!(f, "{message}"),
            ErrorDetails::JsonSchema { message } => write!(f, "{message}"),
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
            } => {
                write!(f, "Error rendering template {template_name}: {message}")
            }
            ErrorDetails::MiniJinjaTemplateMissing { template_name } => {
                write!(f, "Template not found: {template_name}")
            }
            ErrorDetails::MiniJinjaTemplateRender {
                template_name,
                message,
            } => {
                write!(f, "Error rendering template {template_name}: {message}")
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
            ErrorDetails::ModelNotFound { model_name } => {
                write!(f, "Model not found: {model_name}")
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
            ErrorDetails::NoFallbackVariantsRemaining => {
                write!(f, "No fallback variants remaining.")
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
            ErrorDetails::PostgresConnectionInitialization { message } => {
                write!(
                    f,
                    "Postgres connection initialization failed with message: {message}"
                )
            }
            ErrorDetails::PostgresConnection { message } => {
                write!(f, "Error connecting to Postgres: {message}")
            }
            ErrorDetails::PostgresMigration { message } => {
                write!(f, "Postgres migration failed with message: {message}")
            }
            ErrorDetails::PostgresResult {
                result_type,
                message,
            } => {
                write!(
                    f,
                    "Unexpected Postgres result of type {result_type}: {message}"
                )
            }
            ErrorDetails::PostgresQuery {
                function_name,
                message,
            } => match function_name {
                Some(function_name) => write!(
                    f,
                    "Postgres query failed in function {function_name} with message: {message}"
                ),
                None => write!(f, "Postgres query failed: {message}"),
            },
            ErrorDetails::ProviderNotFound { provider_name } => {
                write!(f, "Provider not found: {provider_name}")
            }
            ErrorDetails::RateLimitExceeded { failed_rate_limits } => {
                if failed_rate_limits.len() == 1 {
                    let limit = &failed_rate_limits[0];
                    let scope = limit
                        .scope_key
                        .iter()
                        .map(|key| key.to_config_representation())
                        .collect::<Vec<_>>()
                        .join(", ");

                    write!(
                        f,
                        "TensorZero rate limit exceeded for `{}` resource.\nScope: {}\nRequested: {}\nAvailable: {}",
                        limit.resource.as_str(), scope, limit.requested, limit.available
                    )
                } else {
                    writeln!(
                        f,
                        "TensorZero rate limits exceeded for {} rules:",
                        failed_rate_limits.len()
                    )?;
                    for (i, limit) in failed_rate_limits.iter().enumerate() {
                        if i > 0 {
                            writeln!(f)?;
                        }
                        let scope = limit
                            .scope_key
                            .iter()
                            .map(|key| key.to_config_representation())
                            .collect::<Vec<_>>()
                            .join(", ");

                        write!(
                            f,
                            "- Resource: `{}`\n    ├ Scope: {}\n    ├ Requested: {}\n    └ Available: {}",
                            limit.resource.as_str(), scope, limit.requested, limit.available
                        )?;
                    }
                    Ok(())
                }
            }
            ErrorDetails::RateLimitMissingMaxTokens => {
                write!(
                    f,
                    "Missing `max_tokens` for request subject to rate limiting rules."
                )
            }
            ErrorDetails::StreamError { source } => {
                write!(f, "Error in streaming response: {source}")
            }
            ErrorDetails::Serialization { message } => write!(f, "{message}"),
            ErrorDetails::TypeConversion { message } => write!(f, "{message}"),
            ErrorDetails::ToolNotFound { name } => write!(f, "Tool not found: {name}"),
            ErrorDetails::ToolNotLoaded { name } => write!(f, "Tool not loaded: {name}"),
            ErrorDetails::IncompatibleTool { message } => write!(f, "{message}"),
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
            ErrorDetails::UnsupportedModelProviderForStreamingInference { provider_type } => {
                write!(
                    f,
                    "Unsupported model provider for streaming inference: {provider_type}"
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

impl IntoResponse for Error {
    /// Log the error and convert it into an Axum response
    fn into_response(self) -> Response {
        let message = self.to_string();
        let mut body = json!({
            "error": message,
        });
        if *UNSTABLE_ERROR_JSON.get().unwrap_or(&false) {
            body["error_json"] =
                serde_json::to_value(self.get_details()).unwrap_or_else(|e| json!(e.to_string()));
        }
        let mut response = (self.status_code(), Json(body)).into_response();
        // Attach the error to the response, so that we can set a nice message in our
        // `apply_otel_http_trace_layer` middleware
        response.extensions_mut().insert(self);
        response
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::new(ErrorDetails::Serialization {
            message: err.to_string(),
        })
    }
}

impl From<sqlx::Error> for Error {
    fn from(err: sqlx::Error) -> Self {
        Self::new(ErrorDetails::PostgresQuery {
            message: err.to_string(),
            function_name: None,
        })
    }
}

impl From<AnalysisError> for Error {
    fn from(err: AnalysisError) -> Self {
        Self::new(ErrorDetails::DynamicTemplateLoad { internal: err })
    }
}
