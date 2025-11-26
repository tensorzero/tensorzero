use axum::body::Body;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Extension, Json};
use futures::future::{join_all, try_join_all};
use indexmap::IndexMap;
use itertools::{izip, Itertools};
use metrics::counter;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::iter::repeat;
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

use super::inference::{
    ChatInferenceResponse, InferenceClients, InferenceCredentials, InferenceDatabaseInsertMetadata,
    InferenceIds, InferenceModels, InferenceParams, InferenceResponse, JsonInferenceResponse,
};
use crate::cache::{CacheEnabledMode, CacheOptions};
use crate::config::Config;
use crate::db::clickhouse::{ClickHouseConnectionInfo, TableName};
use crate::endpoints::RequestApiKeyExtension;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::function::FunctionConfig;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::batch::{
    BatchEpisodeIds, BatchEpisodeIdsWithSize, BatchInferenceDatabaseInsertMetadata,
    BatchInferenceParams, BatchInferenceParamsWithSize, BatchModelInferenceRow,
    BatchOutputSchemasWithSize, BatchRequestRow, BatchStatus, PollBatchInferenceResponse,
    ProviderBatchInferenceOutput, ProviderBatchInferenceResponse, UnparsedBatchRequestRow,
};
use crate::inference::types::resolved_input::LazyResolvedInput;
use crate::inference::types::RequestMessage;
use crate::inference::types::{batch::StartBatchModelInferenceWithMetadata, Input};
use crate::inference::types::{
    current_timestamp, ChatInferenceDatabaseInsert, ContentBlockChatOutput, FetchContext,
    FinishReason, InferenceDatabaseInsert, InferenceResult, JsonInferenceDatabaseInsert,
    JsonInferenceOutput, Latency, ModelInferenceResponseWithMetadata, RequestMessagesOrBatch,
    Usage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::model::ModelTable;
use crate::rate_limiting::ScopeInfo;
use crate::tool::{
    BatchDynamicToolParams, BatchDynamicToolParamsWithSize, DynamicToolParams, ToolCallConfig,
    ToolCallConfigDatabaseInsert,
};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};
use crate::variant::{BatchInferenceConfig, InferenceConfig, Variant, VariantInfo};

/// The expected payload to the `/start_batch_inference` endpoint.
/// It will be a JSON object with the following fields:
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartBatchInferenceParams {
    // the function name
    pub function_name: String,
    // the episode IDs for each inference (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    #[serde(default)]
    pub episode_ids: Option<BatchEpisodeIdInput>,
    // the inputs for the inferences
    pub inputs: Vec<Input>,
    // Inference-time overrides for variant types (use with caution)
    #[serde(default)]
    pub params: BatchInferenceParams,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. TESTING / DEBUGGING VARIANTS).
    pub variant_name: Option<String>,
    // the tags to add to the inference
    #[serde(default)]
    pub tags: Option<BatchTags>,
    // dynamic information about tool calling. Don't directly include `dynamic_tool_params` in `Params`.
    #[serde(flatten)]
    pub dynamic_tool_params: BatchDynamicToolParams,
    // `dynamic_tool_params` includes the following fields, passed at the top level of `Params`:
    // If provided, the inference will only use the specified tools (a subset of the function's tools)
    // allowed_tools: Option<Vec<Option<Vec<String>>>>,
    // If provided, the inference will use the specified tools in addition to the function's tools
    // additional_tools: Option<Vec<Option<Vec<Tool>>>>,
    // If provided, the inference will use the specified tool choice
    // tool_choice: Option<Vec<Option<ToolChoice>>>,
    // If true, the inference will use parallel tool calls
    // parallel_tool_calls: Option<Vec<Option<bool>>>,
    // If provided for a JSON inference, the inference will use the specified output schema instead of the
    // configured one. We only lazily validate this schema.
    #[serde(default)]
    pub output_schemas: Option<BatchOutputSchemas>,
    #[serde(default)]
    pub credentials: InferenceCredentials,
}

pub type BatchEpisodeIdInput = Vec<Option<Uuid>>;
pub type BatchTags = Vec<Option<HashMap<String, String>>>;
pub type BatchOutputSchemas = Vec<Option<Value>>;

/// This handler starts a batch inference request for a particular function.
/// The entire batch must use the same function and variant.
/// It will fail if we fail to kick off the batch request for any reason.
/// However, the batch request might still fail for other reasons after it has been started.
#[instrument(
    name="start_batch_inference",
    skip_all,
    fields(
        function_name = %params.function_name,
        variant_name = ?params.variant_name,
    )
)]
#[debug_handler(state = AppStateData)]
pub async fn start_batch_inference_handler(
    State(app_state): State<AppStateData>,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(params): StructuredJson<StartBatchInferenceParams>,
) -> Result<Response<Body>, Error> {
    Ok(Json(start_batch_inference(app_state, params, api_key_ext).await?).into_response())
}

pub async fn start_batch_inference(
    AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        ..
    }: AppStateData,
    params: StartBatchInferenceParams,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
) -> Result<PrepareBatchInferenceOutput, Error> {
    // Get the function config or return an error if it doesn't exist
    let function = config.get_function(&params.function_name)?;
    let num_inferences = params.inputs.len();
    if num_inferences == 0 {
        return Err(ErrorDetails::InvalidRequest {
            message: "No inputs provided".to_string(),
        }
        .into());
    }
    // Collect the tool params and output schemas into vectors of the same length as the batch
    let batch_dynamic_tool_params: Vec<DynamicToolParams> =
        BatchDynamicToolParamsWithSize(params.dynamic_tool_params, num_inferences).try_into()?;
    let batch_dynamic_output_schemas: Vec<Option<DynamicJSONSchema>> =
        BatchOutputSchemasWithSize(params.output_schemas, num_inferences).try_into()?;

    let tool_configs = batch_dynamic_tool_params
        .into_iter()
        .map(|dynamic_tool_params| function.prepare_tool_config(dynamic_tool_params, &config.tools))
        .collect::<Result<Vec<_>, _>>()?;
    let mut candidate_variants: BTreeMap<String, Arc<VariantInfo>> =
        function.variants().clone().into_iter().collect();

    let inference_ids = (0..num_inferences)
        .map(|_| Uuid::now_v7())
        .collect::<Vec<_>>();

    // If the function has no variants, return an error
    if candidate_variants.is_empty() {
        return Err(ErrorDetails::InvalidFunctionVariants {
            message: format!("Function `{}` has no variants", params.function_name),
        }
        .into());
    }

    // Validate the input
    params
        .inputs
        .iter()
        .enumerate()
        .try_for_each(|(i, input)| {
            function.validate_input(input).map_err(|e| {
                Error::new(ErrorDetails::BatchInputValidation {
                    index: i,
                    message: e.to_string(),
                })
            })
        })?;

    // If a variant is pinned, only that variant should be attempted
    if let Some(ref variant_name) = params.variant_name {
        candidate_variants.retain(|k, _| k == variant_name);

        // If the pinned variant doesn't exist, return an error
        if candidate_variants.is_empty() {
            return Err(ErrorDetails::UnknownVariant {
                name: variant_name.to_string(),
            }
            .into());
        }
    }

    // Retrieve or generate the episode IDs and validate them (in the impl)
    let episode_ids: BatchEpisodeIds =
        BatchEpisodeIdsWithSize(params.episode_ids, num_inferences).try_into()?;

    // Increment the request count
    counter!(
        "request_count",
        "endpoint" => "batch_inference",
        "function_name" => params.function_name.to_string(),
    )
    .increment(1);
    counter!(
        "tensorzero_requests_total",
        "endpoint" => "batch_inference",
        "function_name" => params.function_name.to_string(),
    )
    .increment(1);
    counter!(
        "inference_count",
        "endpoint" => "batch_inference",
        "function_name" => params.function_name.to_string(),
    )
    .increment(num_inferences as u64);
    counter!(
        "tensorzero_inferences_total",
        "endpoint" => "batch_inference",
        "function_name" => params.function_name.to_string(),
    )
    .increment(num_inferences as u64);

    // Keep track of which variants failed
    let mut variant_errors = IndexMap::new();

    let cache_options = CacheOptions {
        max_age_s: None,
        enabled: CacheEnabledMode::WriteOnly,
    };

    let tags = Arc::new(HashMap::default()); // NOTE: we currently do not rate limit batch inference

    let inference_clients = InferenceClients {
        http_client: http_client.clone(),
        clickhouse_connection_info: clickhouse_connection_info.clone(),
        postgres_connection_info: postgres_connection_info.clone(),
        credentials: Arc::new(params.credentials.clone()),
        cache_options: cache_options.clone(),
        rate_limiting_config: Arc::new(config.rate_limiting.clone()),
        tags: tags.clone(),
        otlp_config: config.gateway.export.otlp.clone(),
        deferred_tasks,
        scope_info: ScopeInfo::new(tags.clone(), api_key_ext),
    };

    let inference_models = InferenceModels {
        models: config.models.clone(),
        embedding_models: config.embedding_models.clone(),
    };
    let inference_params: Vec<InferenceParams> =
        BatchInferenceParamsWithSize(params.params, num_inferences).try_into()?;

    let context = FetchContext {
        client: &http_client,
        object_store_info: &config.object_store_info,
    };

    let resolved_inputs = params
        .inputs
        .into_iter()
        .map(|input| input.into_lazy_resolved_input(&context))
        .collect::<Result<Vec<LazyResolvedInput>, Error>>()?;

    // If we have a pinned variant (only one candidate), skip sampling and directly start the batch inference
    if candidate_variants.len() == 1 {
        let (variant_name, variant) = candidate_variants
            .into_iter()
            .next()
            .ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: format!("No candidate variants available for batch inference. {IMPOSSIBLE_ERROR_MESSAGE}"),
                })
            })?;

        return start_variant_batch_inference(StartVariantBatchInferenceArgs {
            variant_name,
            variant,
            function: &function,
            function_name: &params.function_name,
            episode_ids: &episode_ids,
            inference_ids: &inference_ids,
            resolved_inputs: resolved_inputs.clone(),
            inference_models: &inference_models,
            inference_clients,
            inference_params: inference_params.clone(),
            tool_configs: &tool_configs,
            batch_dynamic_output_schemas: &batch_dynamic_output_schemas,
            config: &config,
            clickhouse_connection_info: &clickhouse_connection_info,
            tags: params.tags.clone(),
        })
        .await
        .map(|(batch_id, inference_ids)| PrepareBatchInferenceOutput {
            batch_id,
            inference_ids,
            episode_ids,
        });
    }

    // Keep sampling variants until one succeeds
    // We already guarantee there is at least one inference
    let first_episode_id = episode_ids
        .first()
        .ok_or_else(|| Error::new(ErrorDetails::Inference {
            message: "batch episode_ids unexpectedly empty. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
        }))?;

    while !candidate_variants.is_empty() {
        // We sample the same variant for the whole batch
        let result = function
            .experimentation()
            .sample(
                &params.function_name,
                *first_episode_id,
                &mut candidate_variants,
                &postgres_connection_info,
            )
            .await;
        let (variant_name, variant) = match result {
            Ok((variant_name, variant)) => (variant_name, variant),
            Err(e) => {
                if variant_errors.is_empty() {
                    return Err(e);
                }
                // If the sampling fails we break out of the loop and return the AllVariantsExhausted error
                // It is more informative to the caller that variants have failed than that there's some internal error with the sampling strategy.
                // As we continue work on experimentation we will make sure that the sampler only errors if there is no way to provide a valid variant.
                break;
            }
        };

        let result = start_variant_batch_inference(StartVariantBatchInferenceArgs {
            variant_name: variant_name.clone(),
            variant,
            function: &function,
            function_name: &params.function_name,
            episode_ids: &episode_ids,
            inference_ids: &inference_ids,
            resolved_inputs: resolved_inputs.clone(),
            inference_models: &inference_models,
            inference_clients: inference_clients.clone(),
            inference_params: inference_params.clone(),
            tool_configs: &tool_configs,
            batch_dynamic_output_schemas: &batch_dynamic_output_schemas,
            config: &config,
            clickhouse_connection_info: &clickhouse_connection_info,
            tags: params.tags.clone(),
        })
        .await;

        match result {
            Ok((batch_id, inference_ids)) => {
                return Ok(PrepareBatchInferenceOutput {
                    batch_id,
                    inference_ids,
                    episode_ids,
                });
            }
            Err(e) => {
                tracing::warn!(
                    "functions.{function_name}.variants.{variant_name} failed during inference: {e}",
                    function_name = params.function_name,
                    variant_name = variant_name,
                );
                variant_errors.insert(variant_name, e);
                continue;
            }
        }
    }

    // Eventually, if we get here, it means we tried every variant and none of them worked
    Err(ErrorDetails::AllVariantsFailed {
        errors: variant_errors,
    }
    .into())
}

struct StartVariantBatchInferenceArgs<'a> {
    variant_name: String,
    variant: Arc<VariantInfo>,
    function: &'a Arc<FunctionConfig>,
    function_name: &'a str,
    episode_ids: &'a BatchEpisodeIds,
    inference_ids: &'a [Uuid],
    resolved_inputs: Vec<LazyResolvedInput>,
    inference_models: &'a InferenceModels,
    inference_clients: InferenceClients,
    inference_params: Vec<InferenceParams>,
    tool_configs: &'a Vec<Option<ToolCallConfig>>,
    batch_dynamic_output_schemas: &'a Vec<Option<DynamicJSONSchema>>,
    config: &'a Arc<Config>,
    clickhouse_connection_info: &'a ClickHouseConnectionInfo,
    tags: Option<BatchTags>,
}

async fn start_variant_batch_inference(
    args: StartVariantBatchInferenceArgs<'_>,
) -> Result<(Uuid, Vec<Uuid>), Error> {
    let StartVariantBatchInferenceArgs {
        variant_name,
        variant,
        function,
        function_name,
        episode_ids,
        inference_ids,
        resolved_inputs,
        inference_models,
        inference_clients,
        inference_params,
        tool_configs,
        batch_dynamic_output_schemas,
        config,
        clickhouse_connection_info,
        tags,
    } = args;

    let tool_configs_arc: Vec<Option<Arc<ToolCallConfig>>> = tool_configs
        .iter()
        .map(|opt| opt.as_ref().map(|tc| Arc::new(tc.clone())))
        .collect();
    let schemas_arc: Vec<Option<Arc<DynamicJSONSchema>>> = batch_dynamic_output_schemas
        .iter()
        .map(|opt| opt.as_ref().map(|s| Arc::new(s.clone())))
        .collect();
    let inference_config = BatchInferenceConfig::new(
        Arc::clone(&config.templates),
        tool_configs_arc,
        schemas_arc,
        Arc::from(function_name),
        Arc::from(variant_name.as_str()),
        config.gateway.fetch_and_encode_input_files_before_inference,
    );
    let inference_configs = inference_config.inference_configs(episode_ids, inference_ids);
    // Will be edited by the variant as part of making the request so we must clone here
    // This could potentially be improved by decoupling the variant name from the rest of the inference params
    let variant_inference_params = inference_params.clone();

    let result = variant
        .start_batch_inference(
            &resolved_inputs,
            inference_models.clone(),
            function,
            &inference_configs,
            inference_clients,
            variant_inference_params,
        )
        .await?;

    // Write to ClickHouse (don't spawn a thread for this because it's required and we should fail loudly)
    let write_metadata = BatchInferenceDatabaseInsertMetadata {
        function_name,
        variant_name: variant_name.as_str(),
        episode_ids,
        tags,
    };

    write_start_batch_inference(
        clickhouse_connection_info,
        config,
        resolved_inputs,
        result,
        write_metadata,
        tool_configs,
        &inference_configs,
    )
    .await
}

// Determines the return type of the `/start_batch_inference` endpoint upon success
#[derive(Debug, Serialize)]
pub struct PrepareBatchInferenceOutput {
    pub batch_id: Uuid,
    pub inference_ids: Vec<Uuid>,
    pub episode_ids: Vec<Uuid>,
}

#[derive(Debug, Deserialize)]
pub struct PollPathParams {
    pub batch_id: Uuid,
    pub inference_id: Option<Uuid>,
}

/// Polls a batch inference request that was made using the `/start_batch_inference` endpoint
/// Semantics: if the batch is pending, it will actually poll the model provider
/// If the batch is failed, it will return a failed response immediately
/// If the batch is completed, it will return the appropriate response immediately from ClickHouse
#[instrument(name = "poll_batch_inference", skip_all, fields(query))]
#[debug_handler(state = AppStateData)]
pub async fn poll_batch_inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        ..
    }): AppState,
    Path(path_params): Path<PollPathParams>,
) -> Result<Response<Body>, Error> {
    let batch_request = get_batch_request(&clickhouse_connection_info, &path_params).await?;
    match batch_request.status {
        BatchStatus::Pending => {
            // For now, we don't support dynamic API keys for batch inference
            let credentials = InferenceCredentials::default();
            let response =
                poll_batch_inference(&batch_request, http_client, &config.models, &credentials)
                    .await?;
            let response = write_poll_batch_inference(
                &clickhouse_connection_info,
                &batch_request,
                response,
                &config,
            )
            .await?;
            Ok(Json(response.filter_by_query(path_params)).into_response())
        }
        BatchStatus::Completed => {
            let function = config.get_function(&batch_request.function_name)?;
            let response = get_completed_batch_inference_response(
                &clickhouse_connection_info,
                &batch_request,
                &path_params,
                &function,
            )
            .await?;
            let response = PollInferenceResponse::Completed(response);
            Ok(Json(response.filter_by_query(path_params)).into_response())
        }
        BatchStatus::Failed => Ok(Json(PollInferenceResponse::Failed).into_response()),
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum PollInferenceResponse {
    Pending,
    Completed(CompletedBatchInferenceResponse),
    Failed,
}

impl PollInferenceResponse {
    /// Filters the response by the provided query
    /// If the query is by an inference ID, it will return a single inference response
    /// Otherwise, it will return the entire batch
    fn filter_by_query(self, path_params: PollPathParams) -> PollInferenceResponse {
        match self {
            PollInferenceResponse::Completed(response) => {
                PollInferenceResponse::Completed(response.filter_by_query(path_params))
            }
            other => other,
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub struct CompletedBatchInferenceResponse {
    pub batch_id: Uuid,
    pub inferences: Vec<InferenceResponse>,
}

impl CompletedBatchInferenceResponse {
    /// Filters the response by the provided query
    /// If the query is by an inference ID, it will return a single inference response
    /// Otherwise, it will return the entire batch
    fn filter_by_query(self, path_params: PollPathParams) -> CompletedBatchInferenceResponse {
        match path_params {
            PollPathParams {
                inference_id: None, ..
            } => self,
            PollPathParams {
                inference_id: Some(inference_id),
                ..
            } => {
                let inferences = self
                    .inferences
                    .into_iter()
                    .filter(|r| r.inference_id() == inference_id)
                    .collect();
                CompletedBatchInferenceResponse {
                    batch_id: self.batch_id,
                    inferences,
                }
            }
        }
    }
}

pub async fn get_batch_request(
    clickhouse: &ClickHouseConnectionInfo,
    path_params: &PollPathParams,
) -> Result<BatchRequestRow<'static>, Error> {
    let response = match path_params {
        PollPathParams {
            batch_id,
            inference_id: None,
        } => {
            let query = format!(
                r"
                    SELECT
                        batch_id,
                        id,
                        batch_params,
                        model_name,
                        model_provider_name,
                        status,
                        function_name,
                        variant_name,
                        raw_request,
                        raw_response,
                        errors
                    FROM BatchRequest
                    WHERE batch_id = '{batch_id}'
                    ORDER BY timestamp DESC
                    LIMIT 1
                    FORMAT JSONEachRow
                "
            );
            let response = clickhouse.run_query_synchronous_no_params(query).await?;
            if response.response.is_empty() {
                return Err(ErrorDetails::BatchNotFound { id: *batch_id }.into());
            }
            response
        }
        PollPathParams {
            batch_id,
            inference_id: Some(inference_id),
        } => {
            let query = format!(
                r"
                    SELECT br.batch_id as batch_id,
                        br.id as id,
                        br.batch_params as batch_params,
                        br.model_name as model_name,
                        br.model_provider_name as model_provider_name,
                        br.status as status,
                        br.function_name as function_name,
                        br.variant_name as variant_name,
                        br.raw_request as raw_request,
                        br.raw_response as raw_response,
                        br.errors as errors
                    FROM BatchIdByInferenceId bi
                    JOIN BatchRequest br ON bi.batch_id = br.batch_id
                    WHERE bi.inference_id = '{inference_id}' AND bi.batch_id = '{batch_id}'
                    ORDER BY br.timestamp DESC
                    LIMIT 1
                    FORMAT JSONEachRow
                ",
            );
            let response = clickhouse.run_query_synchronous_no_params(query).await?;
            if response.response.is_empty() {
                return Err(ErrorDetails::BatchNotFound { id: *inference_id }.into());
            }
            response
        }
    };
    let batch_request =
        serde_json::from_str::<BatchRequestRow>(&response.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })?;
    Ok(batch_request)
}

/// Polls a batch inference request from the model provider that
/// the original request was sent to
///
/// Returns: a `PollBatchInferenceResponse` which is the current status of the batch
/// and if it's newly completed, the response.
async fn poll_batch_inference(
    batch_request: &BatchRequestRow<'static>,
    http_client: TensorzeroHttpClient,
    models: &ModelTable,
    credentials: &InferenceCredentials,
) -> Result<PollBatchInferenceResponse, Error> {
    // Retrieve the relevant model provider
    // Call model.poll_batch_inference on it
    let model_config = models
        .get(batch_request.model_name.as_ref())
        .await?
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidModel {
                model_name: batch_request.model_name.to_string(),
            })
        })?;
    let model_provider = model_config
        .providers
        .get(batch_request.model_provider_name.as_ref())
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidModelProvider {
                model_name: batch_request.model_name.to_string(),
                provider_name: batch_request.model_provider_name.to_string(),
            })
        })?;
    model_provider
        .poll_batch_inference(batch_request, &http_client, credentials)
        .await
}

// Helper struct for writing to the `BatchModelInference` table in ClickHouse
// This is only used to help with iteration in the `write_batch_inference` function
struct BatchInferenceRowHelper<'a> {
    inference_id: &'a Uuid,
    input: LazyResolvedInput,
    input_messages: Vec<RequestMessage>,
    system: Option<&'a str>,
    tool_config: Option<&'a ToolCallConfig>,
    inference_params: &'a InferenceParams,
    output_schema: Option<&'a Value>,
    raw_request: &'a str,
    tags: Option<HashMap<String, String>>,
}

async fn write_start_batch_inference<'a>(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    config: &Config,
    inputs: Vec<LazyResolvedInput>,
    result: StartBatchModelInferenceWithMetadata<'a>,
    metadata: BatchInferenceDatabaseInsertMetadata<'a>,
    tool_configs: &[Option<ToolCallConfig>],
    inference_configs: &[InferenceConfig],
) -> Result<(Uuid, Vec<Uuid>), Error> {
    let model_name = &result.model_name;
    let model_provider_name = &result.model_provider_name;
    // Collect all the data into BatchInferenceRow structs
    let inference_rows = izip!(
        inference_configs.iter(),
        inputs,
        result.input_messages.into_iter(),
        result.systems.iter(),
        tool_configs.iter(),
        result.inference_params.iter(),
        result.output_schemas.into_iter(),
        result.raw_requests.iter(),
        metadata
            .tags
            .unwrap_or_default()
            .into_iter()
            .chain(repeat(None)),
    )
    .map(
        |(
            inference_config,
            input,
            input_messages,
            system,
            tool_config,
            inference_params,
            output_schema,
            raw_request,
            tags,
        )| {
            BatchInferenceRowHelper {
                inference_id: &inference_config.ids.inference_id,
                input,
                input_messages,
                system: system.as_deref(),
                tool_config: tool_config.as_ref(),
                inference_params,
                output_schema,
                raw_request,
                tags,
            }
        },
    );
    let rows = join_all(inference_rows.enumerate().map(|(i, row)| async move {
        let tool_params: Option<ToolCallConfigDatabaseInsert> =
            row.tool_config.map(|tc| tc.clone().into());

        let resolved_input = row.input.clone().resolve().await?;
        join_all(resolved_input.clone().write_all_files(config)).await;

        Ok::<_, Error>(BatchModelInferenceRow {
            inference_id: *row.inference_id,
            batch_id: result.batch_id,
            function_name: metadata.function_name.into(),
            variant_name: metadata.variant_name.into(),
            episode_id: metadata.episode_ids[i],
            input: resolved_input.into_stored_input(),
            input_messages: try_join_all(
                row.input_messages
                    .into_iter()
                    .map(RequestMessage::into_stored_message),
            )
            .await?,
            system: row.system.map(Cow::Borrowed),
            tool_params,
            inference_params: Cow::Borrowed(row.inference_params),
            output_schema: row.output_schema.map(Value::to_string),
            raw_request: Cow::Borrowed(row.raw_request),
            model_name: Cow::Borrowed(model_name),
            model_provider_name: Cow::Borrowed(model_provider_name),
            tags: row.tags.unwrap_or_default(),
        })
    }))
    .await;

    let success_rows = rows
        .into_iter()
        .flat_map(|res| match res {
            Ok(row) => Some(row),
            Err(e) => {
                tracing::error!("Failed to resolve batch inference input: {e:?}");
                None
            }
        })
        .collect::<Vec<_>>();

    clickhouse_connection_info
        .write_batched(success_rows.as_slice(), TableName::BatchModelInference)
        .await?;

    let batch_request_insert = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id: result.batch_id,
        batch_params: &result.batch_params,
        function_name: metadata.function_name,
        variant_name: metadata.variant_name,
        raw_request: &result.raw_request,
        raw_response: &result.raw_response,
        model_name: result.model_name,
        model_provider_name: &result.model_provider_name,
        status: BatchStatus::Pending,
        errors: result.errors,
    });
    write_batch_request_row(clickhouse_connection_info, &batch_request_insert).await?;

    Ok((
        result.batch_id,
        inference_configs
            .iter()
            .map(|c| c.ids.inference_id)
            .collect(),
    ))
}

pub async fn write_batch_request_row(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_request: &BatchRequestRow<'_>,
) -> Result<(), Error> {
    clickhouse_connection_info
        .write_batched(&[batch_request], TableName::BatchRequest)
        .await
}

/// Writes the status of a batch inference request to the database
/// This is a light operation unless the batch is freshly completed, in which case it writes
/// ChatInferences / JsonInferences and ModelInferences as well as updating the
/// BatchRequest table with a new row.
///
/// Note: only call this function if the batch was Pending prior to being polled.
/// We don't need to poll if the batch is failed or completed because the status will not change.
pub async fn write_poll_batch_inference(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_request: &BatchRequestRow<'_>,
    response: PollBatchInferenceResponse,
    config: &Config,
) -> Result<PollInferenceResponse, Error> {
    match response {
        PollBatchInferenceResponse::Pending {
            raw_request,
            raw_response,
        } => {
            write_batch_request_status_update(
                clickhouse_connection_info,
                batch_request,
                BatchStatus::Pending,
                raw_request,
                raw_response,
            )
            .await?;
            Ok(PollInferenceResponse::Pending)
        }
        PollBatchInferenceResponse::Completed(response) => {
            let raw_request = response.raw_request.clone();
            let raw_response = response.raw_response.clone();
            let inferences = write_completed_batch_inference(
                clickhouse_connection_info,
                batch_request,
                response,
                config,
            )
            .await?;
            // NOTE - in older versions of TensorZero, we were missing this call.
            // As a result, some customers may have databases with duplicate inferences.
            write_batch_request_status_update(
                clickhouse_connection_info,
                batch_request,
                BatchStatus::Completed,
                raw_request,
                raw_response,
            )
            .await?;
            Ok(PollInferenceResponse::Completed(
                CompletedBatchInferenceResponse {
                    batch_id: batch_request.batch_id,
                    inferences,
                },
            ))
        }
        PollBatchInferenceResponse::Failed {
            raw_request,
            raw_response,
        } => {
            write_batch_request_status_update(
                clickhouse_connection_info,
                batch_request,
                BatchStatus::Failed,
                raw_request,
                raw_response,
            )
            .await?;
            Ok(PollInferenceResponse::Failed)
        }
    }
}

/// This function updates the status of a batch request in the database
/// It only updates the status of the batch request and does not write any other data to the database
async fn write_batch_request_status_update(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_request: &BatchRequestRow<'_>,
    status: BatchStatus,
    raw_request: String,
    raw_response: String,
) -> Result<(), Error> {
    let batch_request_insert = BatchRequestRow::new(UnparsedBatchRequestRow {
        batch_id: batch_request.batch_id,
        batch_params: &batch_request.batch_params,
        function_name: &batch_request.function_name,
        variant_name: &batch_request.variant_name,
        model_name: &batch_request.model_name,
        raw_request: &raw_request,
        raw_response: &raw_response,
        model_provider_name: &batch_request.model_provider_name,
        status,
        errors: vec![], // TODO (#503): add better error handling
    });
    clickhouse_connection_info
        .write_batched(&[batch_request_insert], TableName::BatchRequest)
        .await?;
    Ok(())
}

/// This function writes ChatInferences / JsonInferences and ModelInferences to the database
/// and updates the BatchRequest table with a new row.
///
/// It takes a `ProviderBatchInferenceResponse` which is the response from the model provider
/// and converts it into a `Vec<InferenceResponse>` which is what gets written to the database.
/// As part of this, it also constructs the `ModelInferenceResponseWithMetadata` struct which is
/// used to serialize the `ModelInference` table.
///
/// TODO: this function has a large number of Clones that are not necessary.
/// To avoid these, the types that are calling for clones must be changed to Cows and then the code in the non-batch inference
/// handler must be adjusted to deal with it and also the lifetimes associated there.
pub async fn write_completed_batch_inference<'a>(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_request: &'a BatchRequestRow<'a>,
    mut response: ProviderBatchInferenceResponse,
    config: &Config,
) -> Result<Vec<InferenceResponse>, Error> {
    let inference_ids: Vec<Uuid> = response.elements.keys().copied().collect();
    let batch_model_inferences = get_batch_inferences(
        clickhouse_connection_info,
        batch_request.batch_id,
        &inference_ids,
    )
    .await?;
    let function_name = &batch_model_inferences
        .first()
        .ok_or_else(|| {
            Error::new(ErrorDetails::MissingBatchInferenceResponse { inference_id: None })
        })?
        .function_name
        .clone();
    let function = config.get_function(function_name)?;
    let mut inferences: Vec<InferenceResponse> = Vec::new();
    let mut inference_rows_to_write: Vec<InferenceDatabaseInsert> = Vec::new();
    let mut model_inference_rows_to_write: Vec<Value> = Vec::new();
    for batch_model_inference in batch_model_inferences {
        let BatchModelInferenceRow {
            inference_id,
            batch_id: _,
            function_name: _,
            variant_name,
            episode_id,
            input,
            input_messages,
            system,
            tool_params,
            inference_params,
            output_schema,
            raw_request,
            model_name: _,
            model_provider_name: _,
            tags,
        } = batch_model_inference;
        let ProviderBatchInferenceOutput {
            id: _,
            output,
            raw_response,
            usage,
            finish_reason,
        } = match response.elements.remove(&inference_id) {
            Some(inference_response) => inference_response,
            None => {
                Error::new(ErrorDetails::MissingBatchInferenceResponse {
                    inference_id: Some(inference_id),
                });
                continue;
            }
        };
        let model_inference_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            output: output.clone(),
            system: system.map(Cow::into_owned),
            input_messages: RequestMessagesOrBatch::BatchInput(input_messages),
            raw_request: raw_request.into_owned(),
            raw_response,
            usage,
            latency: Latency::Batch,
            model_name: batch_request.model_name.clone(),
            model_provider_name: batch_request.model_provider_name.clone().into(),
            cached: false,
            finish_reason,
        };
        let tool_config: Option<ToolCallConfig> = match tool_params {
            Some(db_insert) => match db_insert.into_tool_call_config(&function, &config.tools) {
                Ok(config) => config,
                Err(_) => {
                    // Skip this inference if we can't convert the tool config
                    // Error will be logged on construction in `into_tool_call_config`
                    continue;
                }
            },
            None => None,
        };
        let output_schema = match output_schema
            .map(|s| DynamicJSONSchema::parse_from_str(&s))
            .transpose()
        {
            Ok(s) => s,
            Err(_) => continue,
        };
        let extra_body = Default::default();
        let extra_headers = Default::default();
        let inference_config = InferenceConfig {
            tool_config: tool_config.as_ref().map(|tc| Arc::new(tc.clone())),
            dynamic_output_schema: output_schema.as_ref().map(|s| Arc::new(s.clone())),
            templates: Arc::clone(&config.templates),
            function_name: Arc::from(function_name.as_ref()),
            variant_name: Arc::from(variant_name.as_ref()),
            ids: InferenceIds {
                inference_id,
                episode_id,
            },
            fetch_and_encode_input_files_before_inference: config
                .gateway
                .fetch_and_encode_input_files_before_inference,
            // Not currently supported as a batch inference parameter
            extra_body,
            extra_headers,
            extra_cache_key: None,
        };
        let inference_result = function
            .prepare_response(
                inference_id,
                output,
                vec![model_inference_response],
                &inference_config,
                inference_params.into_owned(),
                None,
            )
            .await?;
        let inference_response = InferenceResponse::new(
            inference_result.clone(),
            episode_id,
            variant_name.to_string(),
        );
        inferences.push(inference_response);
        let metadata = InferenceDatabaseInsertMetadata {
            function_name: function_name.to_string(),
            variant_name: variant_name.to_string(),
            episode_id,
            tool_config,
            processing_time: None,
            ttft_ms: None,
            tags,
            // Not currently supported as a batch inference parameter
            extra_body: Default::default(),
            extra_headers: Default::default(),
        };
        model_inference_rows_to_write
            .extend(inference_result.get_serialized_model_inferences().await);
        match inference_result {
            InferenceResult::Chat(chat_result) => {
                let chat_inference = ChatInferenceDatabaseInsert::new(chat_result, input, metadata);
                inference_rows_to_write.push(InferenceDatabaseInsert::Chat(chat_inference));
            }
            InferenceResult::Json(json_result) => {
                let json_inference = JsonInferenceDatabaseInsert::new(json_result, input, metadata);
                inference_rows_to_write.push(InferenceDatabaseInsert::Json(json_inference));
            }
        }
    }
    // Write all the *Inference rows to the database
    match &**function {
        FunctionConfig::Chat(_chat_function) => {
            clickhouse_connection_info
                .write_batched(&inference_rows_to_write, TableName::ChatInference)
                .await?;
        }
        FunctionConfig::Json(_json_function) => {
            clickhouse_connection_info
                .write_batched(&inference_rows_to_write, TableName::JsonInference)
                .await?;
        }
    }
    // Write all the ModelInference rows to the database
    clickhouse_connection_info
        .write_batched(&model_inference_rows_to_write, TableName::ModelInference)
        .await?;

    Ok(inferences)
}

/// This function gets the batch inferences from the database for a given batch id and inference ids
pub async fn get_batch_inferences(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_id: Uuid,
    inference_ids: &[Uuid],
) -> Result<Vec<BatchModelInferenceRow<'static>>, Error> {
    // Guard against the provider not giving us any inference ids
    if inference_ids.is_empty() {
        return Ok(vec![]);
    }
    let query = format!(
        "SELECT * FROM BatchModelInference WHERE batch_id = '{}' AND inference_id IN ({}) FORMAT JSONEachRow",
        batch_id,
        inference_ids.iter().map(|id| format!("'{id}'")).join(",")
    );
    let response = clickhouse_connection_info
        .run_query_synchronous_no_params(query)
        .await?;
    let rows = response
        .response
        .lines()
        .filter(|line| !line.is_empty())
        .map(serde_json::from_str::<BatchModelInferenceRow>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to deserialize batch model inference row: {e}"),
            })
        })?;
    Ok(rows)
}

/// This function gets the already-completed batch inference response from the database
/// It takes a `BatchRequestRow` and a `PollPathParams` and returns a `CompletedBatchInferenceResponse`
/// The `PollPathParams` is used to determine which inference to get (a single inference or all inferences in the batch)
/// The `FunctionConfig` is helpful in determining which table to query for the inference
pub async fn get_completed_batch_inference_response(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_request: &BatchRequestRow<'_>,
    path_params: &PollPathParams,
    function: &FunctionConfig,
) -> Result<CompletedBatchInferenceResponse, Error> {
    match function {
        FunctionConfig::Chat(_) => match path_params {
            PollPathParams {
                batch_id,
                inference_id: None,
            } => {
                let query = format!(
                    "WITH batch_inferences AS (
                        SELECT inference_id
                        FROM BatchModelInference
                        WHERE batch_id = '{}'
                    )
                    SELECT
                        ci.id as inference_id,
                        ci.episode_id as episode_id,
                        ci.variant_name as variant_name,
                        ci.output as output,
                        toUInt32(SUM(mi.input_tokens)) as input_tokens,
                        toUInt32(SUM(mi.output_tokens)) as output_tokens,
                        argMax(mi.finish_reason, toUInt128(mi.id)) as finish_reason
                    FROM ChatInference ci
                    LEFT JOIN ModelInference mi ON ci.id = mi.inference_id
                    WHERE ci.id IN (SELECT inference_id FROM batch_inferences)
                    AND ci.function_name = '{}'
                    AND ci.variant_name = '{}'
                    GROUP BY ci.id, ci.episode_id, ci.variant_name, ci.output
                    FORMAT JSONEachRow",
                    batch_id, batch_request.function_name, batch_request.variant_name
                );
                let response = clickhouse_connection_info
                    .run_query_synchronous_no_params(query)
                    .await?;
                let mut inference_responses = Vec::new();
                for row in response.response.lines() {
                    let inference_response: ChatInferenceResponseDatabaseRead =
                        serde_json::from_str(row).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: e.to_string(),
                            })
                        })?;
                    inference_responses
                        .push(InferenceResponse::Chat(inference_response.try_into()?));
                }
                Ok(CompletedBatchInferenceResponse {
                    batch_id: *batch_id,
                    inferences: inference_responses,
                })
            }
            PollPathParams {
                inference_id: Some(inference_id),
                ..
            } => {
                let query = format!(
                    "WITH inf_lookup AS (
                        SELECT episode_id
                        FROM InferenceById
                        WHERE id_uint = toUInt128(toUUID('{}'))
                        LIMIT 1
                    )
                    SELECT
                        ci.id as inference_id,
                        ci.episode_id as episode_id,
                        ci.variant_name as variant_name,
                        ci.output as output,
                        toUInt32(SUM(mi.input_tokens)) as input_tokens,
                        toUInt32(SUM(mi.output_tokens)) as output_tokens,
                        argMax(mi.finish_reason, toUInt128(mi.id)) as finish_reason
                    FROM ChatInference ci \
                    LEFT JOIN ModelInference mi ON ci.id = mi.inference_id \
                    JOIN inf_lookup ON ci.episode_id = inf_lookup.episode_id \
                    WHERE ci.id = '{}' \
                    AND ci.function_name = '{}' \
                    AND ci.variant_name = '{}' \
                    GROUP BY ci.id, ci.episode_id, ci.variant_name, ci.output \
                    FORMAT JSONEachRow",
                    inference_id,
                    inference_id,
                    batch_request.function_name,
                    batch_request.variant_name
                );
                let response = clickhouse_connection_info
                    .run_query_synchronous_no_params(query)
                    .await?;
                if response.response.is_empty() {
                    return Err(ErrorDetails::InferenceNotFound {
                        inference_id: *inference_id,
                    }
                    .into());
                }
                let inference_response: ChatInferenceResponseDatabaseRead =
                    serde_json::from_str(&response.response).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: e.to_string(),
                        })
                    })?;
                let inference_response = InferenceResponse::Chat(inference_response.try_into()?);
                Ok(CompletedBatchInferenceResponse {
                    batch_id: batch_request.batch_id,
                    inferences: vec![inference_response],
                })
            }
        },
        FunctionConfig::Json(_) => match path_params {
            PollPathParams {
                inference_id: None, ..
            } => {
                let query = format!(
                    "WITH batch_inferences AS (
                        SELECT inference_id
                        FROM BatchModelInference
                        WHERE batch_id = '{}'
                    )
                    SELECT
                        ji.id as inference_id,
                        ji.episode_id as episode_id,
                        ji.variant_name as variant_name,
                        ji.output as output,
                        toUInt32(SUM(mi.input_tokens)) as input_tokens,
                        toUInt32(SUM(mi.output_tokens)) as output_tokens,
                        argMax(mi.finish_reason, toUInt128(mi.id)) as finish_reason
                    FROM JsonInference ji
                    LEFT JOIN ModelInference mi ON ji.id = mi.inference_id
                    WHERE ji.id IN (SELECT inference_id FROM batch_inferences)
                    AND ji.function_name = '{}'
                    AND ji.variant_name = '{}'
                    GROUP BY ji.id, ji.episode_id, ji.variant_name, ji.output
                    FORMAT JSONEachRow",
                    path_params.batch_id, batch_request.function_name, batch_request.variant_name
                );
                let response = clickhouse_connection_info
                    .run_query_synchronous_no_params(query)
                    .await?;
                let mut inference_responses = Vec::new();
                for row in response.response.lines() {
                    let inference_response: JsonInferenceResponseDatabaseRead =
                        serde_json::from_str(row).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: e.to_string(),
                            })
                        })?;
                    inference_responses
                        .push(InferenceResponse::Json(inference_response.try_into()?));
                }
                Ok(CompletedBatchInferenceResponse {
                    batch_id: batch_request.batch_id,
                    inferences: inference_responses,
                })
            }
            PollPathParams {
                inference_id: Some(inference_id),
                ..
            } => {
                let query = format!(
                    "WITH inf_lookup AS (
                        SELECT episode_id
                        FROM InferenceById
                        WHERE id_uint = toUInt128(toUUID('{}'))
                        LIMIT 1
                    )
                    SELECT
                        ji.id as inference_id,
                        ji.episode_id as episode_id,
                        ji.variant_name as variant_name,
                        ji.output as output,
                        toUInt32(SUM(mi.input_tokens)) as input_tokens,
                        toUInt32(SUM(mi.output_tokens)) as output_tokens,
                        argMax(mi.finish_reason, toUInt128(mi.id)) as finish_reason
                    FROM JsonInference ji \
                    LEFT JOIN ModelInference mi ON ji.id = mi.inference_id \
                    JOIN inf_lookup ON ji.episode_id = inf_lookup.episode_id \
                    WHERE ji.id = '{}' \
                    AND ji.function_name = '{}' \
                    AND ji.variant_name = '{}' \
                    GROUP BY ji.id, ji.episode_id, ji.variant_name, ji.output \
                    FORMAT JSONEachRow",
                    inference_id,
                    inference_id,
                    batch_request.function_name,
                    batch_request.variant_name
                );
                let response = clickhouse_connection_info
                    .run_query_synchronous_no_params(query)
                    .await?;
                if response.response.is_empty() {
                    return Err(ErrorDetails::InferenceNotFound {
                        inference_id: *inference_id,
                    }
                    .into());
                }
                let inference_response: JsonInferenceResponseDatabaseRead =
                    serde_json::from_str(&response.response).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: e.to_string(),
                        })
                    })?;
                let inference_response = InferenceResponse::Json(inference_response.try_into()?);
                Ok(CompletedBatchInferenceResponse {
                    batch_id: batch_request.batch_id,
                    inferences: vec![inference_response],
                })
            }
        },
    }
}

#[derive(Debug, Deserialize)]
struct ChatInferenceResponseDatabaseRead {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub output: String,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub finish_reason: Option<FinishReason>,
}

impl TryFrom<ChatInferenceResponseDatabaseRead> for ChatInferenceResponse {
    type Error = Error;

    fn try_from(value: ChatInferenceResponseDatabaseRead) -> Result<Self, Self::Error> {
        let usage = Usage {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
        };
        let output: Vec<ContentBlockChatOutput> =
            serde_json::from_str(&value.output).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: e.to_string(),
                })
            })?;
        Ok(ChatInferenceResponse {
            inference_id: value.inference_id,
            episode_id: value.episode_id,
            variant_name: value.variant_name,
            content: output,
            usage,
            // This is currently unsupported in the batch API
            original_response: None,
            finish_reason: value.finish_reason,
        })
    }
}

#[derive(Debug, Deserialize)]
struct JsonInferenceResponseDatabaseRead {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub output: String,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub finish_reason: Option<FinishReason>,
}

impl TryFrom<JsonInferenceResponseDatabaseRead> for JsonInferenceResponse {
    type Error = Error;

    fn try_from(value: JsonInferenceResponseDatabaseRead) -> Result<Self, Self::Error> {
        let usage = Usage {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
        };
        let output: JsonInferenceOutput = serde_json::from_str(&value.output).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: e.to_string(),
            })
        })?;
        Ok(JsonInferenceResponse {
            inference_id: value.inference_id,
            episode_id: value.episode_id,
            variant_name: value.variant_name,
            output,
            usage,
            // This is currently unsupported in the batch API
            original_response: None,
            finish_reason: value.finish_reason,
        })
    }
}
