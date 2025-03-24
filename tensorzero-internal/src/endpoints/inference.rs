use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Json};
use futures::stream::Stream;
use metrics::counter;
use object_store::{ObjectStore, PutMode, PutOptions};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use tokio_stream::StreamExt;
use tracing::instrument;
use uuid::Uuid;

use crate::cache::{CacheOptions, CacheParamsOptions};
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::{Config, ObjectStoreInfo};
use crate::embeddings::EmbeddingModelTable;
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::function::{sample_variant, FunctionConfigChat};
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::resolved_input::ImageWithPath;
use crate::inference::types::storage::StoragePath;
use crate::inference::types::{
    collect_chunks, Base64Image, ChatInferenceDatabaseInsert, CollectChunksArgs,
    ContentBlockChatOutput, ContentBlockChunk, FetchContext, FinishReason, InferenceResult,
    InferenceResultChunk, InferenceResultStream, Input, JsonInferenceDatabaseInsert,
    JsonInferenceOutput, ModelInferenceResponseWithMetadata, RequestMessage, ResolvedInput,
    ResolvedInputMessageContent, Usage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::model::ModelTable;
use crate::tool::{DynamicToolParams, ToolCallConfig, ToolChoice};
use crate::uuid_util::validate_tensorzero_uuid;
use crate::variant::chat_completion::ChatCompletionConfig;
use crate::variant::{InferenceConfig, JsonMode, Variant, VariantConfig};

use super::validate_tags;

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    // The function name. Exactly one of `function_name` or `model_name` must be provided.
    pub function_name: Option<String>,
    // The model name to run using a default function. Exactly one of `function_name` or `model_name` must be provided.
    pub model_name: Option<String>,
    // the episode ID (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    pub episode_id: Option<Uuid>,
    // the input for the inference
    pub input: Input,
    // default False
    pub stream: Option<bool>,
    // Inference-time overrides for variant types (use with caution)
    #[serde(default)]
    pub params: InferenceParams,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. TESTING / DEBUGGING VARIANTS).
    pub variant_name: Option<String>,
    // if true, the inference will not be stored
    pub dryrun: Option<bool>,
    // if true, the inference will be internal and validation of tags will be skipped
    #[serde(default)]
    pub internal: bool,
    // the tags to add to the inference
    #[serde(default)]
    pub tags: HashMap<String, String>,
    // dynamic information about tool calling. Don't directly include `dynamic_tool_params` in `Params`.
    #[serde(flatten)]
    pub dynamic_tool_params: DynamicToolParams,
    // `dynamic_tool_params` includes the following fields, passed at the top level of `Params`:
    // If provided, the inference will only use the specified tools (a subset of the function's tools)
    // allowed_tools: Option<Vec<String>>,
    // If provided, the inference will use the specified tools in addition to the function's tools
    // additional_tools: Option<Vec<Tool>>,
    // If provided, the inference will use the specified tool choice
    // tool_choice: Option<ToolChoice>,
    // If true, the inference will use parallel tool calls
    // parallel_tool_calls: Option<bool>,
    // If provided for a JSON inference, the inference will use the specified output schema instead of the
    // configured one. We only lazily validate this schema.
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub cache_options: CacheParamsOptions,
    #[serde(default)]
    pub credentials: InferenceCredentials,
    /// If `true`, add an `original_response` field to the response, containing the raw string response from the model.
    /// Note that for complex variants (e.g. `experimental_best_of_n_sampling`), the response may not contain `original_response`
    /// if the fuser/judge model failed
    #[serde(default)]
    pub include_original_response: bool,
    #[serde(default)]
    pub extra_body: UnfilteredInferenceExtraBody,
}

#[derive(Clone, Debug)]
struct InferenceMetadata {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub input: ResolvedInput,
    pub dryrun: bool,
    pub start_time: Instant,
    pub inference_params: InferenceParams,
    pub model_name: Arc<str>,
    pub model_provider_name: Arc<str>,
    pub raw_request: String,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub previous_model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
    pub tags: HashMap<String, String>,
    pub tool_config: Option<ToolCallConfig>,
    pub dynamic_output_schema: Option<DynamicJSONSchema>,
    #[allow(dead_code)] // We may start exposing this in the response
    pub cached: bool,
    pub extra_body: UnfilteredInferenceExtraBody,
}

pub type InferenceCredentials = HashMap<String, SecretString>;

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Response<Body>, Error> {
    let inference_output =
        inference(config, &http_client, clickhouse_connection_info, params).await?;
    match inference_output {
        InferenceOutput::NonStreaming(response) => Ok(Json(response).into_response()),
        InferenceOutput::Streaming(stream) => {
            let event_stream = prepare_serialized_events(stream);

            Ok(Sse::new(event_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response())
        }
    }
}

pub type InferenceStream =
    Pin<Box<dyn Stream<Item = Result<InferenceResponseChunk, Error>> + Send>>;

pub enum InferenceOutput {
    NonStreaming(InferenceResponse),
    Streaming(InferenceStream),
}

impl std::fmt::Debug for InferenceOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceOutput::NonStreaming(response) => write!(f, "NonStreaming({:?})", response),
            InferenceOutput::Streaming(_) => write!(f, "Streaming"),
        }
    }
}

pub const DEFAULT_FUNCTION_NAME: &str = "tensorzero::default";

#[derive(Copy, Clone, Debug)]
pub struct InferenceIds {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
}

#[instrument(
    name="inference",
    skip(config, http_client, clickhouse_connection_info, params),
    fields(
        function_name = ?params.function_name,
        model_name = ?params.model_name,
        variant_name = ?params.variant_name,
        inference_id,
        episode_id,
    )
)]
pub async fn inference(
    config: Arc<Config<'static>>,
    http_client: &reqwest::Client,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    params: Params,
) -> Result<InferenceOutput, Error> {
    // To be used for the Inference table processing_time measurements
    let start_time = Instant::now();
    let inference_id = Uuid::now_v7();
    tracing::Span::current().record("inference_id", inference_id.to_string());

    if params.include_original_response && params.stream.unwrap_or(false) {
        return Err(ErrorDetails::InvalidRequest {
            message: "Cannot set both `include_original_response` and `stream` to `true`"
                .to_string(),
        }
        .into());
    }

    // Retrieve or generate the episode ID
    let episode_id = params.episode_id.unwrap_or(Uuid::now_v7());
    validate_tensorzero_uuid(episode_id, "Episode")?;
    tracing::Span::current().record("episode_id", episode_id.to_string());

    validate_tags(&params.tags, params.internal)?;
    let (function, function_name) = find_function(&params, &config)?;
    // Collect the function variant names as a Vec<&str>
    let mut candidate_variant_names: Vec<&str> =
        function.variants().keys().map(AsRef::as_ref).collect();

    // If the function has no variants, return an error
    if candidate_variant_names.is_empty() {
        return Err(ErrorDetails::InvalidFunctionVariants {
            message: format!("Function `{}` has no variants", function_name),
        }
        .into());
    }

    // Validate the input
    function.validate_inference_params(&params)?;

    let tool_config = function.prepare_tool_config(params.dynamic_tool_params, &config.tools)?;

    // If a variant is pinned, only that variant should be attempted
    if let Some(ref variant_name) = params.variant_name {
        candidate_variant_names.retain(|k| k == variant_name);

        // If the pinned variant doesn't exist, return an error
        if candidate_variant_names.is_empty() {
            return Err(ErrorDetails::UnknownVariant {
                name: variant_name.to_string(),
            }
            .into());
        }
    } else {
        // Remove all zero-weight variants - these can only be used if explicitly pinned above
        candidate_variant_names.retain(|name| {
            if let Some(variant) = function.variants().get(*name) {
                // Retain 'None' and positive-weight variants, discarding zero-weight variants
                variant.weight().is_none_or(|w| w > 0.0)
            } else {
                // Keep missing variants - later code will error if we try to use them
                true
            }
        });
    }

    // Should we store the results?
    let dryrun = params.dryrun.unwrap_or(false);

    // Increment the request count if we're not in dryrun mode
    if !dryrun {
        let mut labels = vec![
            ("endpoint", "inference".to_string()),
            ("function_name", function_name.clone()),
        ];
        if let Some(model_name) = params.model_name {
            labels.push(("model_name", model_name.clone()));
        }
        counter!("request_count", &labels).increment(1);
        counter!("inference_count", &labels).increment(1);
    }

    // Should we stream the inference?
    let stream = params.stream.unwrap_or(false);

    // Keep track of which variants failed
    let mut variant_errors = std::collections::HashMap::new();

    // Set up inference config
    let output_schema = params.output_schema.map(DynamicJSONSchema::new);
    let mut inference_config = InferenceConfig {
        function_name: &function_name,
        variant_name: None,
        templates: &config.templates,
        tool_config: tool_config.as_ref(),
        dynamic_output_schema: output_schema.as_ref(),
        ids: InferenceIds {
            inference_id,
            episode_id,
        },
        extra_cache_key: None,
        extra_body: Default::default(),
    };
    let inference_clients = InferenceClients {
        http_client,
        clickhouse_connection_info: &clickhouse_connection_info,
        credentials: &params.credentials,
        cache_options: &(params.cache_options, dryrun).into(),
    };

    let inference_models = InferenceModels {
        models: &config.models,
        embedding_models: &config.embedding_models,
    };
    let resolved_input = params
        .input
        .resolve(&FetchContext {
            client: http_client,
            object_store_info: &config.object_store_info,
        })
        .await?;
    // Keep sampling variants until one succeeds
    while !candidate_variant_names.is_empty() {
        let (variant_name, variant) = sample_variant(
            &mut candidate_variant_names,
            function.variants(),
            &function_name,
            &episode_id,
        )?;
        // Will be edited by the variant as part of making the request so we must clone here
        let variant_inference_params = params.params.clone();

        inference_config.variant_name = Some(variant_name);
        inference_config.extra_body = params.extra_body.clone();
        if stream {
            let result = variant
                .infer_stream(
                    &resolved_input,
                    &inference_models,
                    function.as_ref(),
                    &inference_config,
                    &inference_clients,
                    variant_inference_params,
                )
                .await;

            // Make sure the response worked prior to launching the thread and starting to return chunks.
            // The provider has already checked that the first chunk is OK.
            let (stream, model_used_info) = match result {
                Ok((stream, model_used_info)) => (stream, model_used_info),
                Err(e) => {
                    tracing::warn!(
                        "functions.{function_name:?}.variants.{variant_name:?} failed during inference: {e}",
                        function_name = params.function_name,
                        variant_name = variant_name,
                    );
                    variant_errors.insert(variant_name.to_string(), e);
                    continue;
                }
            };

            let extra_body = inference_config.extra_body.clone();

            // Create InferenceMetadata for a streaming inference
            let inference_metadata = InferenceMetadata {
                function_name: function_name.to_string(),
                variant_name: variant_name.to_string(),
                inference_id,
                episode_id,
                input: resolved_input.clone(),
                dryrun,
                start_time,
                inference_params: model_used_info.inference_params,
                model_name: model_used_info.model_name,
                model_provider_name: model_used_info.model_provider_name,
                raw_request: model_used_info.raw_request,
                system: model_used_info.system,
                input_messages: model_used_info.input_messages,
                previous_model_inference_results: model_used_info.previous_model_inference_results,
                tags: params.tags,
                tool_config,
                dynamic_output_schema: output_schema,
                cached: model_used_info.cached,
                extra_body,
            };

            let stream = create_stream(
                function,
                config.clone(),
                inference_metadata,
                stream,
                clickhouse_connection_info,
            );

            return Ok(InferenceOutput::Streaming(Box::pin(stream)));
        } else {
            let result = variant
                .infer(
                    &resolved_input,
                    &inference_models,
                    function.as_ref(),
                    &inference_config,
                    &inference_clients,
                    variant_inference_params,
                )
                .await;

            let mut result = match result {
                Ok(result) => result,
                Err(e) => {
                    tracing::warn!(
                        "functions.{function_name}.variants.{variant_name} failed during inference: {e}",
                        function_name = function_name,
                        variant_name = variant_name,
                    );
                    variant_errors.insert(variant_name.to_string(), e);
                    continue;
                }
            };

            if !dryrun {
                // Spawn a thread for a trailing write to ClickHouse so that it doesn't block the response
                let extra_body = inference_config.extra_body.clone();
                let result_to_write = result.clone();
                let write_metadata = InferenceDatabaseInsertMetadata {
                    function_name: function_name.to_string(),
                    variant_name: variant_name.to_string(),
                    episode_id,
                    tool_config,
                    processing_time: Some(start_time.elapsed()),
                    tags: params.tags,
                    extra_body,
                };

                let async_writes = config.gateway.observability.async_writes;
                // Always spawn a tokio task here. This ensures that 'write_inference' will
                // not be cancelled partway through execution if the outer '/inference' request
                // is cancelled. This reduces the chances that we only write to some tables and not others
                // (but this is inherently best-effort due to ClickHouse's lack of transactions).
                let write_future = tokio::spawn(async move {
                    write_inference(
                        &clickhouse_connection_info,
                        &config,
                        resolved_input,
                        result_to_write,
                        write_metadata,
                    )
                    .await;
                });
                if !async_writes {
                    write_future.await.map_err(|e| {
                        Error::new(ErrorDetails::InternalError {
                            message: format!("Failed to await ClickHouse inference write: {e:?}"),
                        })
                    })?;
                }
            }

            if !params.include_original_response {
                result.set_original_response(None);
            }

            let response = InferenceResponse::new(result, episode_id, variant_name.to_string());

            return Ok(InferenceOutput::NonStreaming(response));
        }
    }

    // Eventually, if we get here, it means we tried every variant and none of them worked
    Err(ErrorDetails::AllVariantsFailed {
        errors: variant_errors,
    }
    .into())
}

/// Finds a function by `function_name` or `model_name`, erroring if an
/// invalid combination of parameters is provided.
/// If `model_name` is specified, then we use the special 'default' function
/// Returns the function config and the function name
fn find_function(params: &Params, config: &Config) -> Result<(Arc<FunctionConfig>, String), Error> {
    match (&params.function_name, &params.model_name) {
        // Get the function config or return an error if it doesn't exist
        (Some(function_name), None) => Ok((
            config.get_function(function_name)?.clone(),
            function_name.to_string(),
        )),
        (None, Some(model_name)) => {
            if params.variant_name.is_some() {
                return Err(ErrorDetails::InvalidInferenceTarget {
                    message: "`variant_name` cannot be provided when using `model_name`"
                        .to_string(),
                }
                .into());
            }
            if let Err(e) = config.models.validate(model_name) {
                return Err(ErrorDetails::InvalidInferenceTarget {
                    message: format!("Invalid model name: {e}"),
                }
                .into());
            }

            Ok((
                Arc::new(FunctionConfig::Chat(FunctionConfigChat {
                    variants: [(
                        model_name.clone(),
                        VariantConfig::ChatCompletion(ChatCompletionConfig {
                            model: (&**model_name).into(),
                            ..Default::default()
                        }),
                    )]
                    .into_iter()
                    .collect(),
                    system_schema: None,
                    user_schema: None,
                    assistant_schema: None,
                    tools: vec![],
                    tool_choice: ToolChoice::None,
                    parallel_tool_calls: None,
                })),
                DEFAULT_FUNCTION_NAME.to_string(),
            ))
        }
        (Some(_), Some(_)) => Err(ErrorDetails::InvalidInferenceTarget {
            message: "Only one of `function_name` or `model_name` can be provided".to_string(),
        }
        .into()),
        (None, None) => Err(ErrorDetails::InvalidInferenceTarget {
            message: "Either `function_name` or `model_name` must be provided".to_string(),
        }
        .into()),
    }
}

fn create_stream(
    function: Arc<FunctionConfig>,
    config: Arc<Config<'static>>,
    metadata: InferenceMetadata,
    mut stream: InferenceResultStream,
    clickhouse_connection_info: ClickHouseConnectionInfo,
) -> impl Stream<Item = Result<InferenceResponseChunk, Error>> + Send {
    async_stream::stream! {
        let mut buffer = vec![];
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(chunk) => {
                    buffer.push(chunk.clone());
                    if let Some(chunk) = prepare_response_chunk(&metadata, chunk) {
                        yield Ok(chunk);
                    }
                }
                Err(e) => yield Err(e),
            }
        }
        if !metadata.dryrun {
            // IMPORTANT: The following code will not be reached if the stream is interrupted.
            // Only do things that would be ok to skip in that case.
            //
            // For example, if we were using ClickHouse for billing, we would want to store the interrupted requests.
            //
            // If we really care about storing interrupted requests, we should use a drop guard:
            // https://github.com/tokio-rs/axum/discussions/1060
            let InferenceMetadata {
                function_name,
                variant_name,
                inference_id,
                episode_id,
                input,
                dryrun: _,
                start_time,
                inference_params,
                model_name,
                model_provider_name,
                raw_request,
                system,
                input_messages,
                previous_model_inference_results,
                tags,
                tool_config,
                dynamic_output_schema,
                cached,
                extra_body,
            } = metadata;

            let config = config.clone();
            let async_write = config.gateway.observability.async_writes;
            let write_future = async move {
                let templates = &config.templates;
                let collect_chunks_args = CollectChunksArgs {
                    value: buffer,
                    inference_id,
                    episode_id,
                    system,
                    input_messages,
                    function,
                    model_name,
                    model_provider_name,
                    raw_request,
                    inference_params,
                    function_name: &function_name,
                    variant_name: &variant_name,
                    dynamic_output_schema,
                    templates,
                    tool_config: tool_config.as_ref(),
                    cached,
                    extra_body: extra_body.clone(),
                };
                let inference_response: Result<InferenceResult, Error> =
                    collect_chunks(collect_chunks_args).await;

                let inference_response = inference_response.ok();

                if let Some(inference_response) = inference_response {
                    let mut inference_response = inference_response;
                    inference_response.mut_model_inference_results().extend(previous_model_inference_results);
                    let write_metadata = InferenceDatabaseInsertMetadata {
                        function_name,
                        variant_name,
                        episode_id,
                        tool_config,
                        processing_time: Some(start_time.elapsed()),
                        tags,
                        extra_body,
                    };
                    let config = config.clone();

                        let clickhouse_connection_info = clickhouse_connection_info.clone();
                        write_inference(
                            &clickhouse_connection_info,
                            &config,
                            input,
                            inference_response,
                            write_metadata,
                        ).await;

                }
            };
            if async_write {
                tokio::spawn(write_future);
            } else {
                write_future.await;
            }
        }
    }
}

fn prepare_response_chunk(
    metadata: &InferenceMetadata,
    chunk: InferenceResultChunk,
) -> Option<InferenceResponseChunk> {
    InferenceResponseChunk::new(
        chunk,
        metadata.inference_id,
        metadata.episode_id,
        metadata.variant_name.clone(),
        metadata.cached,
    )
}

// Prepares an Event for SSE on the way out of the gateway
// When None is passed in, we send "[DONE]" to the client to signal the end of the stream
fn prepare_serialized_events(
    mut stream: InferenceStream,
) -> impl Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        while let Some(chunk) = stream.next().await {
            let chunk_json = match chunk {
                Ok(chunk) => {
                    serde_json::to_value(chunk).map_err(|e| {
                        Error::new(ErrorDetails::Inference {
                            message: format!("Failed to convert chunk to JSON: {}", e),
                        })
                    })?
                },
                Err(e) => {
                    // NOTE - in the future, we may want to end the stream early if we get an error
                    serde_json::json!({"error": e.to_string()})
                }
            };
            yield Event::default().json_data(chunk_json).map_err(|e| {
                Error::new(ErrorDetails::Inference {
                    message: format!("Failed to convert Value to Event: {}", e),
                })
            })
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

#[derive(Debug)]
pub struct InferenceDatabaseInsertMetadata {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub tool_config: Option<ToolCallConfig>,
    pub processing_time: Option<Duration>,
    pub tags: HashMap<String, String>,
    pub extra_body: UnfilteredInferenceExtraBody,
}

async fn write_image(
    object_store: &Option<ObjectStoreInfo>,
    raw: &Base64Image,
    storage_path: &StoragePath,
) -> Result<(), Error> {
    if let Some(object_store) = object_store {
        // The store might be explicitly disabled
        if let Some(store) = object_store.object_store.as_ref() {
            let data = raw.data()?;
            let bytes = aws_smithy_types::base64::decode(data).map_err(|e| {
                Error::new(ErrorDetails::ObjectStoreWrite {
                    message: format!("Failed to decode image as base64: {e:?}"),
                    path: storage_path.clone(),
                })
            })?;
            let res = store
                .put_opts(
                    &storage_path.path,
                    bytes.into(),
                    PutOptions {
                        mode: PutMode::Create,
                        ..Default::default()
                    },
                )
                .await;
            match res {
                Ok(_) | Err(object_store::Error::AlreadyExists { .. }) => {}
                Err(e) => {
                    return Err(ErrorDetails::ObjectStoreWrite {
                        message: format!("Failed to write image to object store: {e:?}"),
                        path: storage_path.clone(),
                    }
                    .into());
                }
            }
        }
    } else {
        return Err(ErrorDetails::InternalError {
            message: "Called `write_image` with no object store configured".to_string(),
        }
        .into());
    }
    Ok(())
}

async fn write_inference(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    config: &Config<'_>,
    input: ResolvedInput,
    result: InferenceResult,
    metadata: InferenceDatabaseInsertMetadata,
) {
    let mut futures: Vec<Pin<Box<dyn Future<Output = ()> + Send>>> = Vec::new();
    if config.gateway.observability.enabled.unwrap_or(true) {
        for message in &input.messages {
            for content_block in &message.content {
                if let ResolvedInputMessageContent::Image(ImageWithPath {
                    image: raw,
                    storage_path,
                }) = content_block
                {
                    futures.push(Box::pin(async {
                        if let Err(e) =
                            write_image(&config.object_store_info, raw, storage_path).await
                        {
                            tracing::error!("Failed to write image to object store: {e:?}");
                        }
                    }));
                }
            }
        }
    }
    let model_responses: Vec<serde_json::Value> = result.get_serialized_model_inferences();
    futures.push(Box::pin(async {
        // Write the model responses to the ModelInference table
        for response in model_responses {
            let _ = clickhouse_connection_info
                .write(&[response], "ModelInference")
                .await;
        }
        // Write the inference to the Inference table
        match result {
            InferenceResult::Chat(result) => {
                let chat_inference =
                    ChatInferenceDatabaseInsert::new(result, input.clone(), metadata);
                let _ = clickhouse_connection_info
                    .write(&[chat_inference], "ChatInference")
                    .await;
            }
            InferenceResult::Json(result) => {
                let json_inference =
                    JsonInferenceDatabaseInsert::new(result, input.clone(), metadata);
                let _ = clickhouse_connection_info
                    .write(&[json_inference], "JsonInference")
                    .await;
            }
        }
    }));
    futures::future::join_all(futures).await;
}

/// InferenceResponse and InferenceResultChunk determine what gets serialized and sent to the client

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
    Json(JsonInferenceResponse),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub content: Vec<ContentBlockChatOutput>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub output: JsonInferenceOutput,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

impl InferenceResponse {
    pub fn new(inference_result: InferenceResult, episode_id: Uuid, variant_name: String) -> Self {
        match inference_result {
            InferenceResult::Chat(result) => InferenceResponse::Chat(ChatInferenceResponse {
                inference_id: result.inference_id,
                episode_id,
                variant_name,
                content: result.content,
                usage: result.usage,
                original_response: result.original_response,
                finish_reason: result.finish_reason,
            }),
            InferenceResult::Json(result) => InferenceResponse::Json(JsonInferenceResponse {
                inference_id: result.inference_id,
                episode_id,
                variant_name,
                output: result.output,
                usage: result.usage,
                original_response: result.original_response,
                finish_reason: result.finish_reason,
            }),
        }
    }

    pub fn variant_name(&self) -> &str {
        match self {
            InferenceResponse::Chat(c) => &c.variant_name,
            InferenceResponse::Json(j) => &j.variant_name,
        }
    }

    pub fn inference_id(&self) -> Uuid {
        match self {
            InferenceResponse::Chat(c) => c.inference_id,
            InferenceResponse::Json(j) => j.inference_id,
        }
    }

    pub fn episode_id(&self) -> Uuid {
        match self {
            InferenceResponse::Chat(c) => c.episode_id,
            InferenceResponse::Json(j) => j.episode_id,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InferenceResponseChunk {
    Chat(ChatInferenceResponseChunk),
    Json(JsonInferenceResponseChunk),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatInferenceResponseChunk {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub content: Vec<ContentBlockChunk>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonInferenceResponseChunk {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub raw: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

impl InferenceResponseChunk {
    fn new(
        inference_result: InferenceResultChunk,
        inference_id: Uuid,
        episode_id: Uuid,
        variant_name: String,
        cached: bool,
    ) -> Option<Self> {
        Some(match inference_result {
            InferenceResultChunk::Chat(result) => {
                InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
                    inference_id,
                    episode_id,
                    variant_name,
                    content: result.content,
                    usage: if cached { None } else { result.usage },
                    finish_reason: result.finish_reason,
                })
            }
            InferenceResultChunk::Json(result) => {
                if result.raw.is_none() && result.usage.is_none() {
                    return None;
                }
                InferenceResponseChunk::Json(JsonInferenceResponseChunk {
                    inference_id,
                    episode_id,
                    variant_name,
                    raw: result.raw.unwrap_or_default(),
                    usage: if cached { None } else { result.usage },
                    finish_reason: result.finish_reason,
                })
            }
        })
    }
}

// Carryall struct for clients used in inference
pub struct InferenceClients<'a> {
    pub http_client: &'a reqwest::Client,
    pub clickhouse_connection_info: &'a ClickHouseConnectionInfo,
    pub credentials: &'a InferenceCredentials,
    pub cache_options: &'a CacheOptions,
}

// Carryall struct for models used in inference
#[derive(Debug)]
pub struct InferenceModels<'a> {
    pub models: &'a ModelTable,
    pub embedding_models: &'a EmbeddingModelTable,
}

/// InferenceParams is the top-level struct for inference parameters.
/// We backfill these from the configs given in the variants used and ultimately write them to the database.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceParams {
    pub chat_completion: ChatCompletionInferenceParams,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionInferenceParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_mode: Option<JsonMode>,
}

impl ChatCompletionInferenceParams {
    pub fn backfill_with_variant_params(
        &mut self,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        seed: Option<u32>,
        top_p: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
    ) {
        if self.temperature.is_none() {
            self.temperature = temperature;
        }
        if self.max_tokens.is_none() {
            self.max_tokens = max_tokens;
        }
        if self.seed.is_none() {
            self.seed = seed;
        }
        if self.top_p.is_none() {
            self.top_p = top_p;
        }
        if self.presence_penalty.is_none() {
            self.presence_penalty = presence_penalty;
        }
        if self.frequency_penalty.is_none() {
            self.frequency_penalty = frequency_penalty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;
    use uuid::Uuid;

    use crate::inference::types::{
        ChatInferenceResultChunk, ContentBlockChunk, JsonInferenceResultChunk, TextChunk,
    };

    #[tokio::test]
    async fn test_prepare_event() {
        // Test case 1: Valid Chat ProviderInferenceResponseChunk
        let content = vec![ContentBlockChunk::Text(TextChunk {
            text: "Test content".to_string(),
            id: "0".to_string(),
        })];
        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk {
            content: content.clone(),
            created: 0,
            usage: None,
            finish_reason: Some(FinishReason::Stop),
            raw_response: "".to_string(),
            latency: Duration::from_millis(100),
        });
        let raw_request = "raw request".to_string();
        let inference_metadata = InferenceMetadata {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            input: ResolvedInput {
                messages: vec![],
                system: None,
            },
            dryrun: false,
            inference_params: InferenceParams::default(),
            start_time: Instant::now(),
            model_name: "test_model".into(),
            model_provider_name: "test_provider".into(),
            raw_request: raw_request.clone(),
            system: None,
            input_messages: vec![],
            previous_model_inference_results: vec![],
            tags: HashMap::new(),
            tool_config: None,
            dynamic_output_schema: None,
            cached: false,
            extra_body: Default::default(),
        };

        let result = prepare_response_chunk(&inference_metadata, chunk).unwrap();
        match result {
            InferenceResponseChunk::Chat(c) => {
                assert_eq!(c.inference_id, inference_metadata.inference_id);
                assert_eq!(c.episode_id, inference_metadata.episode_id);
                assert_eq!(c.variant_name, inference_metadata.variant_name);
                assert_eq!(c.content, content);
                assert!(c.usage.is_none());
                assert_eq!(c.finish_reason, Some(FinishReason::Stop));
            }
            InferenceResponseChunk::Json(_) => {
                panic!("Expected ChatInferenceResponseChunk, got JsonInferenceResponseChunk");
            }
        }

        // TODO (#86): You could get the values of the private members using unsafe Rust.
        // For now, we won't and will rely on E2E testing here.
        // This test doesn't do much so consider deleting or doing more.

        // Test case 2: Valid JSON ProviderInferenceResponseChunk
        let chunk = InferenceResultChunk::Json(JsonInferenceResultChunk {
            raw: Some("Test content".to_string()),
            thought: Some("Thought 1".to_string()),
            created: 0,
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(100),
            finish_reason: Some(FinishReason::Stop),
        });
        let inference_metadata = InferenceMetadata {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            input: ResolvedInput {
                messages: vec![],
                system: None,
            },
            dryrun: false,
            inference_params: InferenceParams::default(),
            start_time: Instant::now(),
            model_name: "test_model".into(),
            model_provider_name: "test_provider".into(),
            raw_request: raw_request.clone(),
            system: None,
            input_messages: vec![],
            previous_model_inference_results: vec![],
            tags: HashMap::new(),
            tool_config: None,
            dynamic_output_schema: None,
            cached: false,
            extra_body: Default::default(),
        };

        let result = prepare_response_chunk(&inference_metadata, chunk).unwrap();
        match result {
            InferenceResponseChunk::Json(c) => {
                assert_eq!(c.inference_id, inference_metadata.inference_id);
                assert_eq!(c.episode_id, inference_metadata.episode_id);
                assert_eq!(c.variant_name, inference_metadata.variant_name);
                assert_eq!(c.raw, "Test content".to_string());
                assert!(c.usage.is_none());
                assert_eq!(c.finish_reason, Some(FinishReason::Stop));
            }
            InferenceResponseChunk::Chat(_) => {
                panic!("Expected JsonInferenceResponseChunk, got ChatInferenceResponseChunk");
            }
        }
    }

    #[test]
    fn test_find_function_no_function_model() {
        let err = find_function(
            &Params {
                function_name: None,
                model_name: None,
                ..Default::default()
            },
            &Config::default(),
        )
        .expect_err("find_function should fail without either arg");
        assert!(
            err.to_string()
                .contains("Either `function_name` or `model_name` must be provided"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_find_function_both_function_model() {
        let err = find_function(
            &Params {
                function_name: Some("my_function".to_string()),
                model_name: Some("my_model".to_string()),
                ..Default::default()
            },
            &Config::default(),
        )
        .expect_err("find_function should fail with both args provided");
        assert!(
            err.to_string()
                .contains("Only one of `function_name` or `model_name` can be provided"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_find_function_model_and_variant() {
        let err = find_function(
            &Params {
                function_name: None,
                model_name: Some("my_model".to_string()),
                variant_name: Some("my_variant".to_string()),
                ..Default::default()
            },
            &Config::default(),
        )
        .expect_err("find_function should fail without model_name");
        assert!(
            err.to_string()
                .contains("`variant_name` cannot be provided when using `model_name`"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_find_function_shorthand_model() {
        let (function_config, function_name) = find_function(
            &Params {
                function_name: None,
                model_name: Some("openai::gpt-9000".to_string()),
                ..Default::default()
            },
            &Config::default(),
        )
        .expect("Failed to find shorthand function");
        assert_eq!(function_name, "tensorzero::default");
        assert_eq!(function_config.variants().len(), 1);
        assert_eq!(
            function_config.variants().keys().next().unwrap(),
            "openai::gpt-9000"
        );
    }

    #[test]
    fn test_find_function_shorthand_missing_provider() {
        let err = find_function(
            &Params {
                model_name: Some("fake_provider::gpt-9000".to_string()),
                ..Default::default()
            },
            &Config::default(),
        )
        .expect_err("find_function should fail with invalid provider");
        assert!(
            err.to_string()
                .contains("Model name 'fake_provider::gpt-9000' not found in model table"),
            "Unexpected error: {err}"
        );
    }
}
