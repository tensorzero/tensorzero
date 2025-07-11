use std::sync::OnceLock;

use serde::Serialize;
use uuid::Uuid;

use crate::{clickhouse::ClickHouseConnectionInfo, error::ErrorDetails};

/// Holds the `ClickHouseConnectionInfo` that we use for writing to the `TensorZeroError` table.
/// Note that this is currently only supported with the standalone gateway - adding
/// support for an embedded gateway will require more work.
// TODO - come up with a better name for this
static GLOBAL_CLICKHOUSE_ERRORS: OnceLock<ClickHouseConnectionInfo> = OnceLock::new();

#[expect(clippy::expect_used)]
pub fn initialize_clickhouse_errors(clickhouse: ClickHouseConnectionInfo) {
    GLOBAL_CLICKHOUSE_ERRORS
        .set(clickhouse)
        .expect("GLOBAL_CLICKHOUSE_ERRORS already initialized");
}

pub fn get_clickhouse_errors() -> Option<&'static ClickHouseConnectionInfo> {
    GLOBAL_CLICKHOUSE_ERRORS.get()
}

tokio::task_local! {
    static SUPPRESS_CLICKHOUSE_ERROR_INSERT: ();
}

fn should_suppress_clickhouse_error_insert() -> bool {
    SUPPRESS_CLICKHOUSE_ERROR_INSERT.try_with(|_| ()).is_ok()
}

pub fn spawn_write_error(error: &ErrorDetails) {
    if !error.should_write_to_clickhouse() || should_suppress_clickhouse_error_insert() {
        return;
    }

    // If we produce a new error when trying to write `error` to ClickHouse, don't
    // try to write the new error to ClickHouse. This avoids the possibility of infinite
    // recursion if we have a bug in `write_error`.
    SUPPRESS_CLICKHOUSE_ERROR_INSERT.sync_scope((), || {
        if let Some(clickhouse) = get_clickhouse_errors() {
            let mut error_database_insert = ErrorDatabaseInsert {
                id_uint: Uuid::now_v7().as_u128().to_string(),
                error_kind: "TODO".to_string(),
                message: error.to_string(),
                http_status: None,
                raw_request: None,
                raw_response: None,
                function_name: None,
                variant_name: None,
                model_name: None,
                model_provider_name: None,
                episode_id: None,
                inference_id: None,
            };
            match error {
                ErrorDetails::AllVariantsFailed { errors: _ } => {}
                ErrorDetails::InvalidInferenceTarget { message: _ } => {}
                ErrorDetails::ApiKeyMissing { provider_name: _ } => {}
                ErrorDetails::AppState { message: _ } => {}
                ErrorDetails::BadCredentialsPreInference { provider_name } => {
                    error_database_insert.model_provider_name = Some(provider_name);
                }
                ErrorDetails::BatchInputValidation {
                    index: _,
                    message: _,
                } => {}
                ErrorDetails::BatchNotFound { id: _ } => {}
                ErrorDetails::BadImageFetch { url: _, message: _ } => {}
                ErrorDetails::Cache { message: _ } => {}
                ErrorDetails::ChannelWrite { message: _ } => {}
                ErrorDetails::ClickHouseConnection { message: _ } => {}
                ErrorDetails::ClickHouseDeserialization { message: _ } => {}
                ErrorDetails::ClickHouseMigration { id: _, message: _ } => {}
                ErrorDetails::ClickHouseQuery { message: _ } => {}
                ErrorDetails::Config { message: _ } => {}
                ErrorDetails::ObjectStoreUnconfigured { block_type: _ } => {}
                ErrorDetails::DatapointNotFound {
                    dataset_name: _,
                    datapoint_id: _,
                } => {}
                ErrorDetails::DuplicateTool { name: _ } => {}
                ErrorDetails::DynamicJsonSchema { message: _ } => {}
                ErrorDetails::FileRead {
                    message: _,
                    file_path: _,
                } => {}
                ErrorDetails::GCPCredentials { message: _ } => {}
                ErrorDetails::Inference { message: _ } => {}
                ErrorDetails::InferenceClient {
                    message: _,
                    status_code,
                    provider_type: _,
                    raw_request,
                    raw_response,
                } => {
                    error_database_insert.http_status = status_code.map(|s| s.as_u16());
                    error_database_insert.raw_request = raw_request.as_deref();
                    error_database_insert.raw_response = raw_response.as_deref();
                }
                ErrorDetails::InferenceNotFound { inference_id: _ } => {}
                ErrorDetails::InferenceServer {
                    message: _,
                    provider_type: _,
                    raw_request,
                    raw_response,
                } => {
                    error_database_insert.raw_request = raw_request.as_deref();
                    error_database_insert.raw_response = raw_response.as_deref();
                }
                ErrorDetails::InvalidClientMode {
                    mode: _,
                    message: _,
                } => {}
                ErrorDetails::InvalidEncodedJobHandle => {}
                ErrorDetails::InvalidJobHandle { message: _ } => {}
                ErrorDetails::InvalidInferenceOutputSource { source: _ } => {}
                ErrorDetails::ObjectStoreWrite {
                    message: _,
                    path: _,
                } => {}
                ErrorDetails::InternalError { message: _ } => {}
                ErrorDetails::InferenceTimeout { variant_name: _ } => {}
                ErrorDetails::VariantTimeout {
                    variant_name,
                    timeout: _,
                    streaming: _,
                } => {
                    error_database_insert.variant_name = variant_name.as_deref();
                }
                ErrorDetails::ModelTimeout {
                    model_name,
                    timeout: _,
                    streaming: _,
                } => {
                    error_database_insert.model_name = Some(model_name);
                }
                ErrorDetails::ModelProviderTimeout {
                    provider_name,
                    timeout: _,
                    streaming: _,
                } => {
                    error_database_insert.model_provider_name = Some(provider_name);
                }
                ErrorDetails::InputValidation { source: _ } => {}
                ErrorDetails::InvalidBatchParams { message: _ } => {}
                ErrorDetails::InvalidBaseUrl { message: _ } => {}
                ErrorDetails::InvalidCandidate {
                    variant_name,
                    message: _,
                } => {
                    error_database_insert.variant_name = Some(variant_name);
                }
                ErrorDetails::InvalidDatasetName { dataset_name: _ } => {}
                ErrorDetails::InvalidDiclConfig { message: _ } => {}
                ErrorDetails::InvalidDynamicEvaluationRun { episode_id: _ } => {}
                ErrorDetails::InvalidTensorzeroUuid {
                    kind: _,
                    message: _,
                } => {}
                ErrorDetails::InvalidFunctionVariants { message: _ } => {}
                ErrorDetails::InvalidMetricName { metric_name: _ } => {}
                ErrorDetails::InvalidMessage { message: _ } => {}
                ErrorDetails::InvalidModel { model_name } => {
                    error_database_insert.model_name = Some(model_name);
                }
                ErrorDetails::InvalidModelProvider {
                    model_name,
                    provider_name,
                } => {
                    error_database_insert.model_name = Some(model_name);
                    error_database_insert.model_provider_name = Some(provider_name);
                }
                ErrorDetails::InvalidOpenAICompatibleRequest { message: _ } => {}
                ErrorDetails::InvalidProviderConfig { message: _ } => {}
                ErrorDetails::InvalidRenderedStoredInference { message: _ } => {}
                ErrorDetails::InvalidRequest { message: _ } => {}
                ErrorDetails::InvalidTemplatePath => {}
                ErrorDetails::InvalidTool { message: _ } => {}
                ErrorDetails::InvalidVariantForOptimization {
                    function_name,
                    variant_name,
                } => {
                    error_database_insert.function_name = Some(function_name);
                    error_database_insert.variant_name = Some(variant_name);
                }
                ErrorDetails::InvalidValFraction { val_fraction: _ } => {}
                ErrorDetails::InvalidUuid { raw_uuid: _ } => {}
                ErrorDetails::JsonRequest { message: _ } => {}
                ErrorDetails::JsonSchema { message: _ } => {}
                ErrorDetails::JsonSchemaValidation {
                    messages: _,
                    data: _,
                    schema: _,
                } => {}
                ErrorDetails::MissingFunctionInVariants { function_name } => {
                    error_database_insert.function_name = Some(function_name);
                }
                ErrorDetails::MiniJinjaEnvironment { message: _ } => {}
                ErrorDetails::MiniJinjaTemplate {
                    template_name: _,
                    message: _,
                } => {}
                ErrorDetails::MiniJinjaTemplateMissing { template_name: _ } => {}
                ErrorDetails::MiniJinjaTemplateRender {
                    template_name: _,
                    message: _,
                } => {}
                ErrorDetails::MissingBatchInferenceResponse { inference_id: _ } => {}
                ErrorDetails::MissingFileExtension { file_name: _ } => {}
                ErrorDetails::ModelProvidersExhausted { provider_errors: _ } => {}
                ErrorDetails::ModelValidation { message: _ } => {}
                ErrorDetails::Observability { message: _ } => {}
                ErrorDetails::OutputParsing {
                    message: _,
                    raw_output: _,
                } => {}
                ErrorDetails::OptimizationResponse {
                    message: _,
                    provider_type: _,
                } => {}
                ErrorDetails::OutputValidation { source: _ } => {}
                ErrorDetails::ProviderNotFound { provider_name } => {
                    error_database_insert.model_provider_name = Some(provider_name);
                }
                ErrorDetails::Serialization { message: _ } => {}
                ErrorDetails::ExtraBodyReplacement {
                    message: _,
                    pointer: _,
                } => {}
                ErrorDetails::StreamError { source: _ } => {}
                ErrorDetails::ToolNotFound { name: _ } => {}
                ErrorDetails::ToolNotLoaded { name: _ } => {}
                ErrorDetails::TypeConversion { message: _ } => {}
                ErrorDetails::UnknownCandidate { name: _ } => {}
                ErrorDetails::UnknownEvaluation { name: _ } => {}
                ErrorDetails::UnknownFunction { name: _ } => {}
                ErrorDetails::UnknownModel { name: _ } => {}
                ErrorDetails::UnknownTool { name: _ } => {}
                ErrorDetails::UnknownVariant { name: _ } => {}
                ErrorDetails::UnknownMetric { name: _ } => {}
                ErrorDetails::UnsupportedModelProviderForBatchInference { provider_type: _ } => {}
                ErrorDetails::UnsupportedVariantForBatchInference { variant_name } => {
                    error_database_insert.variant_name = variant_name.as_deref();
                }
                ErrorDetails::UnsupportedVariantForStreamingInference {
                    variant_type: _,
                    issue_link: _,
                } => {}
                ErrorDetails::UnsupportedVariantForFunctionType {
                    function_name,
                    variant_name,
                    function_type: _,
                    variant_type: _,
                } => {
                    error_database_insert.function_name = Some(function_name);
                    error_database_insert.variant_name = Some(variant_name);
                }
                ErrorDetails::UnsupportedContentBlockType {
                    content_block_type: _,
                    provider_type: _,
                } => {}
                ErrorDetails::UuidInFuture { raw_uuid: _ } => {}
                ErrorDetails::UnsupportedFileExtension { extension: _ } => {}
                ErrorDetails::RouteNotFound { path: _, method: _ } => {}
            };
            match serde_json::to_value(error_database_insert) {
                Ok(value) => {
                    tokio::spawn(SUPPRESS_CLICKHOUSE_ERROR_INSERT.scope((), async move {
                        if let Err(e) = clickhouse.write(&[value], "TensorZeroError").await {
                            tracing::error!("Failed to write error to ClickHouse: {}", e);
                        }
                    }));
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to serialize error to when writing to ClickHouse: {}",
                        e
                    );
                }
            }
        }
    });
}

#[derive(Debug, Serialize)]
struct ErrorDatabaseInsert<'a> {
    // TODO - is there a better way to handle UUId->UInt128 conversions?
    id_uint: String,
    error_kind: String,
    message: String,
    http_status: Option<u16>,
    raw_request: Option<&'a str>,
    raw_response: Option<&'a str>,
    function_name: Option<&'a str>,
    variant_name: Option<&'a str>,
    model_name: Option<&'a str>,
    model_provider_name: Option<&'a str>,
    episode_id: Option<Uuid>,
    inference_id: Option<Uuid>,
}
