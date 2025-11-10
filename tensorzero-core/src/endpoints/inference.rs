use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Extension, Json};
use futures::stream::Stream;
use futures::FutureExt;
use futures_core::FusedStream;
use metrics::counter;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use tokio_stream::StreamExt;
use tokio_util::task::TaskTracker;
use tracing::instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use uuid::Uuid;

use crate::cache::{CacheOptions, CacheParamsOptions};
use crate::config::{Config, ErrorContext, OtlpConfig, SchemaData, UninitializedVariantInfo};
use crate::db::clickhouse::{ClickHouseConnectionInfo, TableName};
use crate::db::postgres::PostgresConnectionInfo;
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::RequestApiKeyExtension;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::experimentation::ExperimentationConfig;
use crate::function::FunctionConfig;
use crate::function::FunctionConfigChat;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, ServiceTier,
};
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::resolved_input::LazyResolvedInput;
use crate::inference::types::{
    collect_chunks, ChatInferenceDatabaseInsert, ChatInferenceResultChunk, CollectChunksArgs,
    ContentBlockChatOutput, ContentBlockChunk, FetchContext, FinishReason, InferenceResult,
    InferenceResultChunk, InferenceResultStream, Input, InternalJsonInferenceOutput,
    JsonInferenceDatabaseInsert, JsonInferenceOutput, JsonInferenceResultChunk,
    ModelInferenceResponseWithMetadata, RequestMessage, ResolvedInput, Usage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::rate_limiting::{RateLimitingConfig, ScopeInfo};
use crate::tool::{DynamicToolParams, ToolCallConfig, ToolChoice};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};
use crate::variant::chat_completion::UninitializedChatCompletionConfig;
use crate::variant::dynamic::load_dynamic_variant_info;
use crate::variant::{InferenceConfig, JsonMode, Variant, VariantConfig, VariantInfo};

use crate::endpoints::validate_tags;
use crate::endpoints::workflow_evaluation_run::validate_inference_episode_id_and_apply_workflow_evaluation_run;

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
    // provider_tools: Vec<ProviderTool> (defaults to [])
    // If set, will attempt to pass this vector of tools to providers which run server-side tools
    // that satisfy the scopes required.
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
    #[serde(default)]
    pub extra_headers: UnfilteredInferenceExtraHeaders,
    #[serde(default)]
    pub internal_dynamic_variant_config: Option<UninitializedVariantInfo>,
}

#[derive(Debug)]
struct InferenceMetadata {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub input: Arc<LazyResolvedInput>,
    pub dryrun: bool,
    pub start_time: Instant,
    pub inference_params: InferenceParams,
    pub model_name: Arc<str>,
    pub model_provider_name: Arc<str>,
    pub raw_request: String,
    pub raw_response: Option<String>,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub previous_model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
    pub tags: HashMap<String, String>,
    pub tool_config: Option<ToolCallConfig>,
    pub dynamic_output_schema: Option<DynamicJSONSchema>,
    pub cached: bool,
    pub extra_body: UnfilteredInferenceExtraBody,
    pub extra_headers: UnfilteredInferenceExtraHeaders,
    pub fetch_and_encode_input_files_before_inference: bool,
    pub include_original_response: bool,
}

pub type InferenceCredentials = HashMap<String, SecretString>;

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Response<Body>, Error> {
    let inference_output = inference(
        config,
        &http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        params,
        api_key_ext,
    )
    .await?;
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
    Pin<Box<dyn FusedStream<Item = Result<InferenceResponseChunk, Error>> + Send>>;

pub enum InferenceOutput {
    NonStreaming(InferenceResponse),
    Streaming(InferenceStream),
}

impl std::fmt::Debug for InferenceOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceOutput::NonStreaming(response) => write!(f, "NonStreaming({response:?})"),
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
    skip_all
    fields(
        function_name,
        model_name,
        variant_name,
        inference_id,
        episode_id,
        otel.name = "function_inference"
    )
)]
pub async fn inference(
    config: Arc<Config>,
    http_client: &TensorzeroHttpClient,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    postgres_connection_info: PostgresConnectionInfo,
    deferred_tasks: TaskTracker,
    mut params: Params,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
) -> Result<InferenceOutput, Error> {
    let span = tracing::Span::current();
    if let Some(function_name) = &params.function_name {
        span.record("function_name", function_name);
    }
    if let Some(model_name) = &params.model_name {
        span.record("model_name", model_name);
    }
    if let Some(variant_name) = &params.variant_name {
        span.record("variant_name", variant_name);
    }
    if let Some(episode_id) = &params.episode_id {
        span.record("episode_id", episode_id.to_string());
    }

    config
        .gateway
        .export
        .otlp
        .mark_openinference_chain_span(&span);

    // Automatically add internal tag when internal=true
    if params.internal {
        params
            .tags
            .insert("tensorzero::internal".to_string(), "true".to_string());
    }

    for (tag_key, tag_value) in &params.tags {
        span.set_attribute(format!("tags.{tag_key}"), tag_value.clone());
    }
    // To be used for the Inference table processing_time measurements
    let start_time = Instant::now();
    let inference_id = Uuid::now_v7();
    span.record("inference_id", inference_id.to_string());
    validate_tags(&params.tags, params.internal)?;

    // Retrieve or generate the episode ID
    let episode_id = params.episode_id.unwrap_or_else(Uuid::now_v7);

    validate_inference_episode_id_and_apply_workflow_evaluation_run(
        episode_id,
        params.function_name.as_ref(),
        &mut params.variant_name,
        &mut params.tags,
        &clickhouse_connection_info,
    )
    .await?;
    // Record the episode id if we didn't already have one
    if params.episode_id.is_none() {
        tracing::Span::current().record("episode_id", episode_id.to_string());
    }
    if let Some(api_key_ext) = &api_key_ext {
        params.tags.insert(
            "tensorzero::api_key_public_id".to_string(),
            api_key_ext.0.api_key.get_public_id().into(),
        );
    }

    let (function, function_name) = find_function(&params, &config)?;
    let mut candidate_variants: BTreeMap<String, Arc<VariantInfo>> =
        function.variants().clone().into_iter().collect();

    // If the function has no variants, return an error
    if candidate_variants.is_empty() {
        return Err(ErrorDetails::InvalidFunctionVariants {
            message: format!("Function `{function_name}` has no variants"),
        }
        .into());
    }

    // Validate the input
    function.validate_inference_params(&params)?;

    // Should we store the results?
    let dryrun = params.dryrun.unwrap_or(false);
    if params.internal_dynamic_variant_config.is_some() && !dryrun {
        return Err(ErrorDetails::InvalidRequest {
            message:
                "If `internal_dynamic_variant_config` is used, `dryrun` must also be set to true"
                    .to_string(),
        }
        .into());
    }

    let tool_config = function.prepare_tool_config(params.dynamic_tool_params, &config.tools)?;
    let mut templates = Arc::clone(&config.templates);

    let needs_sampling = prepare_candidate_variants(
        &mut candidate_variants,
        &mut params.tags,
        params.variant_name.as_deref(),
        params.internal_dynamic_variant_config,
        &mut templates,
        &function,
        function_name.clone(),
    )?;
    let templates = &templates;

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
        counter!("tensorzero_requests_total", &labels).increment(1);
        counter!("inference_count", &labels).increment(1);
        counter!("tensorzero_inferences_total", &labels).increment(1);
    }

    // Should we stream the inference?
    let stream = params.stream.unwrap_or(false);

    // Keep track of which variants failed
    let mut variant_errors: HashMap<String, Error> = HashMap::new();

    // Set up inference config
    let output_schema = params.output_schema.map(DynamicJSONSchema::new);

    let tags = Arc::new(params.tags.clone());

    let inference_clients = InferenceClients {
        http_client: http_client.clone(),
        clickhouse_connection_info: clickhouse_connection_info.clone(),
        postgres_connection_info: postgres_connection_info.clone(),
        credentials: Arc::new(params.credentials.clone()),
        cache_options: (params.cache_options, dryrun).into(),
        tags: tags.clone(),
        rate_limiting_config: Arc::new(config.rate_limiting.clone()),
        otlp_config: config.gateway.export.otlp.clone(),
        deferred_tasks,
        scope_info: ScopeInfo::new(tags.clone(), api_key_ext),
    };

    let inference_models = InferenceModels {
        models: config.models.clone(),
        embedding_models: config.embedding_models.clone(),
    };
    let resolved_input = Arc::new(params.input.into_lazy_resolved_input(FetchContext {
        client: http_client,
        object_store_info: &config.object_store_info,
    })?);

    // If we don't need sampling (pinned or dynamic variant), directly infer with the single variant
    if !needs_sampling {
        // Extract the single variant (should be exactly one)
        let (variant_name, variant) = candidate_variants
            .into_iter()
            .next()
            .ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: format!("No candidate variants available for direct inference. {IMPOSSIBLE_ERROR_MESSAGE}"),
                })
            })?;

        return infer_variant(InferVariantArgs {
            variant_name,
            variant,
            function: &function,
            function_name: &function_name,
            inference_id,
            episode_id,
            dryrun,
            start_time,
            stream,
            resolved_input,
            inference_models,
            inference_clients,
            inference_params: params.params.clone(),
            templates,
            tool_config: &tool_config,
            output_schema: &output_schema,
            config: &config,
            clickhouse_connection_info: &clickhouse_connection_info,
            tags: &params.tags,
            extra_body: &params.extra_body,
            extra_headers: &params.extra_headers,
            include_original_response: params.include_original_response,
        })
        .await;
    }

    // Keep sampling variants until one succeeds
    while !candidate_variants.is_empty() {
        let result = function
            .experimentation()
            .sample(
                &function_name,
                episode_id,
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

        let result = infer_variant(InferVariantArgs {
            variant_name: variant_name.clone(),
            variant,
            function: &function,
            function_name: &function_name,
            inference_id,
            episode_id,
            dryrun,
            start_time,
            stream,
            resolved_input: resolved_input.clone(),
            inference_models: inference_models.clone(),
            inference_clients: inference_clients.clone(),
            inference_params: params.params.clone(),
            templates,
            tool_config: &tool_config,
            output_schema: &output_schema,
            config: &config,
            clickhouse_connection_info: &clickhouse_connection_info,
            tags: &params.tags,
            extra_body: &params.extra_body,
            extra_headers: &params.extra_headers,
            include_original_response: params.include_original_response,
        })
        .await;

        match result {
            Ok(output) => return Ok(output),
            Err(e) => {
                tracing::warn!(
                    "functions.{function_name}.variants.{variant_name} failed during inference: {e}",
                    function_name = function_name,
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

struct InferVariantArgs<'a> {
    variant_name: String,
    variant: Arc<VariantInfo>,
    function: &'a Arc<FunctionConfig>,
    function_name: &'a str,
    inference_id: Uuid,
    episode_id: Uuid,
    dryrun: bool,
    start_time: Instant,
    stream: bool,
    resolved_input: Arc<LazyResolvedInput>,
    inference_models: InferenceModels,
    inference_clients: InferenceClients,
    inference_params: InferenceParams,
    templates: &'a Arc<TemplateConfig<'static>>,
    tool_config: &'a Option<ToolCallConfig>,
    output_schema: &'a Option<DynamicJSONSchema>,
    config: &'a Arc<Config>,
    clickhouse_connection_info: &'a ClickHouseConnectionInfo,
    tags: &'a HashMap<String, String>,
    extra_body: &'a UnfilteredInferenceExtraBody,
    extra_headers: &'a UnfilteredInferenceExtraHeaders,
    include_original_response: bool,
}

async fn infer_variant(args: InferVariantArgs<'_>) -> Result<InferenceOutput, Error> {
    let InferVariantArgs {
        variant_name,
        variant,
        function,
        function_name,
        inference_id,
        episode_id,
        dryrun,
        start_time,
        stream,
        resolved_input,
        inference_models,
        inference_clients,
        inference_params,
        templates,
        tool_config,
        output_schema,
        config,
        clickhouse_connection_info,
        tags,
        extra_body,
        extra_headers,
        include_original_response,
    } = args;

    // Will be edited by the variant as part of making the request so we must clone here
    let variant_inference_params = inference_params.clone();
    let inference_config = Arc::new(InferenceConfig {
        function_name: Arc::from(function_name),
        variant_name: Arc::from(variant_name.as_str()),
        templates: Arc::clone(templates),
        tool_config: tool_config.as_ref().map(|tc| Arc::new(tc.clone())),
        dynamic_output_schema: output_schema
            .as_ref()
            .map(|schema| Arc::new(schema.clone())),
        ids: InferenceIds {
            inference_id,
            episode_id,
        },
        fetch_and_encode_input_files_before_inference: config
            .gateway
            .fetch_and_encode_input_files_before_inference,
        extra_cache_key: None,
        extra_body: extra_body.clone(),
        extra_headers: extra_headers.clone(),
    });

    if stream {
        let result = variant
            .infer_stream(
                resolved_input.clone(),
                inference_models,
                function.clone(),
                inference_config.clone(),
                inference_clients,
                variant_inference_params,
            )
            .await;

        // Make sure the response worked prior to launching the thread and starting to return chunks.
        // The provider has already checked that the first chunk is OK.
        let (stream, model_used_info) = result?;

        let extra_body = inference_config.extra_body.clone();
        let extra_headers = inference_config.extra_headers.clone();
        // Create InferenceMetadata for a streaming inference
        let inference_metadata = InferenceMetadata {
            function_name: function_name.to_string(),
            variant_name: inference_config.variant_name.to_string(),
            inference_id,
            episode_id,
            input: resolved_input,
            dryrun,
            start_time,
            inference_params: model_used_info.inference_params,
            model_name: model_used_info.model_name,
            model_provider_name: model_used_info.model_provider_name,
            raw_request: model_used_info.raw_request,
            raw_response: model_used_info.raw_response,
            system: model_used_info.system,
            input_messages: model_used_info.input_messages,
            previous_model_inference_results: model_used_info.previous_model_inference_results,
            tags: tags.clone(),
            tool_config: tool_config.clone(),
            dynamic_output_schema: output_schema.clone(),
            cached: model_used_info.cached,
            extra_body,
            extra_headers,
            include_original_response,
            fetch_and_encode_input_files_before_inference: config
                .gateway
                .fetch_and_encode_input_files_before_inference,
        };

        let stream = create_stream(
            function.clone(),
            config.clone(),
            inference_metadata,
            stream,
            clickhouse_connection_info.clone(),
        );

        Ok(InferenceOutput::Streaming(Box::pin(stream)))
    } else {
        let result = variant
            .infer(
                Arc::clone(&resolved_input),
                inference_models,
                function.clone(),
                Arc::clone(&inference_config),
                inference_clients,
                variant_inference_params,
            )
            .await;

        let mut result = result?;

        if !dryrun {
            // Spawn a thread for a trailing write to ClickHouse so that it doesn't block the response
            let result_to_write = result.clone();
            let extra_body = inference_config.extra_body.clone();
            let extra_headers = inference_config.extra_headers.clone();
            let write_metadata = InferenceDatabaseInsertMetadata {
                function_name: function_name.to_string(),
                variant_name: inference_config.variant_name.to_string(),
                episode_id,
                tool_config: tool_config.clone(),
                processing_time: Some(start_time.elapsed()),
                ttft_ms: None,
                tags: tags.clone(),
                extra_body,
                extra_headers,
            };

            let async_writes = config.gateway.observability.async_writes;
            let clickhouse_connection_info = clickhouse_connection_info.clone();
            let config = config.clone();
            let resolved_input = resolved_input.clone();
            // Always spawn a tokio task here. This ensures that 'write_inference' will
            // not be cancelled partway through execution if the outer '/inference' request
            // is cancelled. This reduces the chances that we only write to some tables and not others
            // (but this is inherently best-effort due to ClickHouse's lack of transactions).
            // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
            #[expect(clippy::disallowed_methods)]
            let write_future = tokio::spawn(async move {
                let _: () = write_inference(
                    &clickhouse_connection_info,
                    &config,
                    Arc::unwrap_or_clone(resolved_input).resolve().await?,
                    result_to_write,
                    write_metadata,
                )
                .await;
                Ok::<_, Error>(())
            });
            if !async_writes {
                write_future.await.map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!("Failed to await ClickHouse inference write: {e:?}"),
                    })
                })??;
            }
        }

        if !include_original_response {
            result.set_original_response(None);
        }

        let response = InferenceResponse::new(result, episode_id, variant_name.clone());

        Ok(InferenceOutput::NonStreaming(response))
    }
}

/// Finds a function by `function_name` or `model_name`, erroring if an
/// invalid combination of parameters is provided.
/// If `model_name` is specified, then we use the special 'default' function
/// Returns the function config and the function name
fn find_function(params: &Params, config: &Config) -> Result<(Arc<FunctionConfig>, String), Error> {
    match (
        &params.function_name,
        &params.model_name,
        &params.internal_dynamic_variant_config,
    ) {
        // Get the function config or return an error if it doesn't exist
        (Some(function_name), None, _) => Ok((
            config.get_function(function_name)?.clone().into_owned(),
            function_name.to_string(),
        )),
        (None, Some(model_name), None) => {
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
                        Arc::new(VariantInfo {
                            timeouts: Default::default(),
                            inner: VariantConfig::ChatCompletion(
                                UninitializedChatCompletionConfig {
                                    model: (&**model_name).into(),
                                    ..Default::default()
                                }
                                .load(
                                    &SchemaData::default(),
                                    &ErrorContext {
                                        function_name: "tensorzero::default".to_string(),
                                        variant_name: model_name.clone(),
                                    },
                                )?,
                            ),
                        }),
                    )]
                    .into_iter()
                    .collect(),
                    schemas: SchemaData::default(),
                    tools: vec![],
                    tool_choice: ToolChoice::Auto,
                    parallel_tool_calls: None,
                    description: None,
                    all_explicit_templates_names: HashSet::new(),
                    experimentation: ExperimentationConfig::default(),
                })),
                DEFAULT_FUNCTION_NAME.to_string(),
            ))
        }
        (Some(_), Some(_), None) => Err(ErrorDetails::InvalidInferenceTarget {
            message: "Only one of `function_name` or `model_name` can be provided".to_string(),
        }
        .into()),
        (None, None, None) => Err(ErrorDetails::InvalidInferenceTarget {
            message: "Either `function_name` or `model_name` must be provided".to_string(),
        }
        .into()),
        (_, _, Some(_)) => Err(ErrorDetails::InvalidInferenceTarget {
            message: "If a dynamic variant config is passed, `function_name` must be specified."
                .to_string(),
        }
        .into()),
    }
}

fn create_stream(
    function: Arc<FunctionConfig>,
    config: Arc<Config>,
    metadata: InferenceMetadata,
    mut stream: InferenceResultStream,
    clickhouse_connection_info: ClickHouseConnectionInfo,
) -> impl FusedStream<Item = Result<InferenceResponseChunk, Error>> + Send {
    async_stream::stream! {
        let mut buffer = vec![];
        let mut extra_usage = Some(metadata.previous_model_inference_results.iter().map(ModelInferenceResponseWithMetadata::usage_considering_cached).sum());
        if extra_usage == Some(Usage { input_tokens: 0, output_tokens: 0 }) {
            extra_usage = None;
        }
        let mut inference_ttft = None;
        while let Some(chunk) = stream.next().await {
            if inference_ttft.is_none() {
                inference_ttft = Some(metadata.start_time.elapsed());
            }
            match chunk {
                Ok(chunk) => {
                    buffer.push(chunk.clone());
                    if let Some(chunk) = prepare_response_chunk(&metadata, chunk, &mut extra_usage) {
                        yield Ok(chunk);
                    }
                }
                Err(e) => yield Err(e),
            }
        }
        // We didn't find an existing chunk to add 'extra_usage' (either because the underlying
        // stream had no usage information, or because we returned zero usage due to caching)
        if let Some(extra_usage) = extra_usage {
            let usage_chunk = match &*function {
                FunctionConfig::Chat(_model_provider) => {
                    InferenceResultChunk::Chat(ChatInferenceResultChunk {
                        created: 0,
                        content: vec![],
                        usage: Some(extra_usage),
                        finish_reason: None,
                        latency: Duration::from_millis(0),
                        raw_response: String::new(),
                    })
                }
                FunctionConfig::Json(_) => {
                    InferenceResultChunk::Json(JsonInferenceResultChunk {
                        thought: None,
                        created: 0,
                        usage: Some(extra_usage),
                        latency: Duration::from_millis(0),
                        raw: None,
                        raw_response: String::new(),
                        finish_reason: None,
                    })
                }
            };
            buffer.push(usage_chunk.clone());
            if let Some(chunk) = prepare_response_chunk(&metadata, usage_chunk, &mut None) {
                yield Ok(chunk);
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
                raw_response,
                system,
                input_messages,
                previous_model_inference_results,
                tags,
                tool_config,
                dynamic_output_schema,
                cached,
                extra_body,
                extra_headers,
                fetch_and_encode_input_files_before_inference,
                include_original_response: _,
            } = metadata;

            let config = config.clone();
            let async_write = config.gateway.observability.async_writes;
            let write_future = async move {
                let templates = Arc::clone(&config.templates);
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
                    raw_response,
                    inference_params,
                    function_name: Arc::from(function_name.as_str()),
                    variant_name: Arc::from(variant_name.as_str()),
                    dynamic_output_schema: dynamic_output_schema.map(Arc::new),
                    templates,
                    tool_config: tool_config.as_ref().map(|tc| Arc::new(tc.clone())),
                    cached,
                    extra_body: extra_body.clone(),
                    extra_headers: extra_headers.clone(),
                    fetch_and_encode_input_files_before_inference,
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
                        ttft_ms: inference_ttft.map(|ttft| ttft.as_millis() as u32),
                        extra_body,
                        extra_headers,
                    };
                    let config = config.clone();
                        match Arc::unwrap_or_clone(input).resolve().await {
                            Ok(input) => {
                                let clickhouse_connection_info = clickhouse_connection_info.clone();
                                write_inference(
                                    &clickhouse_connection_info,
                                    &config,
                                    input,
                                    inference_response,
                                    write_metadata,
                                ).await;
                            },
                            Err(e) => {
                                tracing::error!("Failed to resolve input: {e:?}");

                            }
                        };


                }
                drop(clickhouse_connection_info);
            };
            if async_write {
                // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
                #[expect(clippy::disallowed_methods)]
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
    extra_usage: &mut Option<Usage>,
) -> Option<InferenceResponseChunk> {
    InferenceResponseChunk::new(
        chunk,
        metadata.inference_id,
        metadata.episode_id,
        metadata.variant_name.clone(),
        metadata.cached,
        metadata.include_original_response,
        extra_usage,
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
                            message: format!("Failed to convert chunk to JSON: {e}"),
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
                    message: format!("Failed to convert Value to Event: {e}"),
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
    pub ttft_ms: Option<u32>,
    pub tags: HashMap<String, String>,
    pub extra_body: UnfilteredInferenceExtraBody,
    pub extra_headers: UnfilteredInferenceExtraHeaders,
}

async fn write_inference(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    config: &Config,
    input: ResolvedInput,
    result: InferenceResult,
    metadata: InferenceDatabaseInsertMetadata,
) {
    let model_responses: Vec<serde_json::Value> = result.get_serialized_model_inferences().await;
    let mut futures: Vec<Pin<Box<dyn Future<Output = ()> + Send>>> =
        input.clone().write_all_files(config);
    // Write the model responses to the ModelInference table
    futures.push(
        async {
            let _ = clickhouse_connection_info
                .write_batched(&model_responses, TableName::ModelInference)
                .await;
        }
        .boxed(),
    );
    futures.push(Box::pin(async {
        // Write the inference to the Inference table
        match result {
            InferenceResult::Chat(result) => {
                let stored_input = input.clone().into_stored_input();
                let chat_inference =
                    ChatInferenceDatabaseInsert::new(result, stored_input, metadata);
                let _ = clickhouse_connection_info
                    .write_batched(&[chat_inference], TableName::ChatInference)
                    .await;
            }
            InferenceResult::Json(result) => {
                let stored_input = input.clone().into_stored_input();
                let json_inference =
                    JsonInferenceDatabaseInsert::new(result, stored_input, metadata);
                let _ = clickhouse_connection_info
                    .write_batched(&[json_inference], TableName::JsonInference)
                    .await;
            }
        }
    }));
    futures::future::join_all(futures).await;
}

/// InferenceResponse and InferenceResultChunk determine what gets serialized and sent to the client

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
#[serde(untagged, rename_all = "snake_case")]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
    Json(JsonInferenceResponse),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
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
        let usage = inference_result.usage_considering_cached();
        match inference_result {
            InferenceResult::Chat(result) => InferenceResponse::Chat(ChatInferenceResponse {
                inference_id: result.inference_id,
                episode_id,
                variant_name,
                content: result.content,
                usage,
                original_response: result.original_response,
                finish_reason: result.finish_reason,
            }),
            InferenceResult::Json(result) => {
                let InternalJsonInferenceOutput { raw, parsed, .. } = result.output;
                let output = JsonInferenceOutput { raw, parsed };
                InferenceResponse::Json(JsonInferenceResponse {
                    inference_id: result.inference_id,
                    episode_id,
                    variant_name,
                    output,
                    usage,
                    original_response: result.original_response,
                    finish_reason: result.finish_reason,
                })
            }
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

    pub fn get_serialized_output(&self) -> Result<String, Error> {
        match self {
            InferenceResponse::Chat(c) => c.get_serialized_output(),
            InferenceResponse::Json(j) => j.get_serialized_output(),
        }
    }
}

impl ChatInferenceResponse {
    pub fn get_serialized_output(&self) -> Result<String, Error> {
        serde_json::to_string(&self.content).map_err(|e| {
            Error::new(ErrorDetails::Inference {
                message: format!("Failed to serialize chat inference response: {e:?}"),
            })
        })
    }
}

impl JsonInferenceResponse {
    pub fn get_serialized_output(&self) -> Result<String, Error> {
        serde_json::to_string(&self.output).map_err(|e| {
            Error::new(ErrorDetails::Inference {
                message: format!("Failed to serialize json inference response: {e:?}"),
            })
        })
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_chunk: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_chunk: Option<String>,
}

const ZERO_USAGE: Usage = Usage {
    input_tokens: 0,
    output_tokens: 0,
};

impl InferenceResponseChunk {
    fn new(
        inference_result: InferenceResultChunk,
        inference_id: Uuid,
        episode_id: Uuid,
        variant_name: String,
        cached: bool,
        include_original_response: bool,
        extra_usage: &mut Option<Usage>,
    ) -> Option<Self> {
        let mut result_usage = if cached {
            // When our outer inference result is cached, don't
            // add `extra_usage` to it. We'll append a final usage chunk
            // in `create_stream` if needed
            Some(ZERO_USAGE)
        } else {
            inference_result.usage().copied()
        };
        // The first time we encounter an empty chunk that already has usage information set,
        // add `extra_usage` to the chunk.
        // If we never encounter any empty chunks with usage, we'll append one ourselves
        // in `create_stream`
        // We do this in both cached and non-cached mode, so that our decision to emit
        // an extra usage chunk is consistent across both modes.
        if let Some(result_usage) = &mut result_usage {
            let is_empty = match &inference_result {
                InferenceResultChunk::Chat(result) => result.content.is_empty(),
                InferenceResultChunk::Json(result) => result.raw.is_none(),
            };
            if is_empty {
                if let Some(extra_usage) = extra_usage.take() {
                    result_usage.input_tokens += extra_usage.input_tokens;
                    result_usage.output_tokens += extra_usage.output_tokens;
                }
            }
        }
        Some(match inference_result {
            InferenceResultChunk::Chat(result) => {
                InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
                    inference_id,
                    episode_id,
                    variant_name,
                    content: result.content,
                    // Token usage is intended to represent 'billed tokens',
                    // so set it to zero if the result is cached
                    usage: result_usage,
                    finish_reason: result.finish_reason,
                    original_chunk: include_original_response.then_some(result.raw_response),
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
                    // Token usage is intended to represent 'billed tokens',
                    // so set it to zero if the result is cached
                    usage: result_usage,
                    finish_reason: result.finish_reason,
                    original_chunk: include_original_response.then_some(result.raw_response),
                })
            }
        })
    }

    pub fn episode_id(&self) -> Uuid {
        match self {
            InferenceResponseChunk::Chat(c) => c.episode_id,
            InferenceResponseChunk::Json(j) => j.episode_id,
        }
    }

    pub fn inference_id(&self) -> Uuid {
        match self {
            InferenceResponseChunk::Chat(c) => c.inference_id,
            InferenceResponseChunk::Json(j) => j.inference_id,
        }
    }

    pub fn variant_name(&self) -> &str {
        match self {
            InferenceResponseChunk::Chat(c) => &c.variant_name,
            InferenceResponseChunk::Json(j) => &j.variant_name,
        }
    }
}

// Carryall struct for clients used in inference
#[derive(Clone)]
pub struct InferenceClients {
    pub http_client: TensorzeroHttpClient,
    pub clickhouse_connection_info: ClickHouseConnectionInfo,
    pub postgres_connection_info: PostgresConnectionInfo,
    pub credentials: Arc<InferenceCredentials>,
    pub cache_options: CacheOptions,
    pub tags: Arc<HashMap<String, String>>,
    pub rate_limiting_config: Arc<RateLimitingConfig>,
    pub otlp_config: OtlpConfig,
    pub deferred_tasks: TaskTracker,
    pub scope_info: ScopeInfo,
}

// Carryall struct for models used in inference
#[derive(Clone, Debug)]
pub struct InferenceModels {
    pub models: Arc<ModelTable>,
    pub embedding_models: Arc<EmbeddingModelTable>,
}

/// InferenceParams is the top-level struct for inference parameters.
/// We backfill these from the configs given in the variants used and ultimately write them to the database.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct InferenceParams {
    pub chat_completion: ChatCompletionInferenceParams,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

impl ChatCompletionInferenceParams {
    #[expect(clippy::too_many_arguments)]
    pub fn backfill_with_variant_params(
        &mut self,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        seed: Option<u32>,
        top_p: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        stop_sequences: Option<Vec<String>>,
        inference_params_v2: ChatCompletionInferenceParamsV2,
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
        if self.stop_sequences.is_none() {
            self.stop_sequences = stop_sequences;
        }
        let ChatCompletionInferenceParamsV2 {
            reasoning_effort,
            service_tier,
            thinking_budget_tokens,
            verbosity,
        } = inference_params_v2;

        if self.reasoning_effort.is_none() {
            self.reasoning_effort = reasoning_effort;
        }
        if self.service_tier.is_none() {
            self.service_tier = service_tier;
        }
        if self.thinking_budget_tokens.is_none() {
            self.thinking_budget_tokens = thinking_budget_tokens;
        }
        if self.verbosity.is_none() {
            self.verbosity = verbosity;
        }
    }
}

/// Prepares the candidate variants map using inference parameters prior to sampling
/// This function handles 2 cases:
/// 1. If a variant is pinned, only that variant should be attempted
/// 2. If a dynamic variant is configured, only that variant should be attempted
///
/// It also errors if both are configured or there is a failure to initialize the dynamic variant
///
/// Returns `Ok(true)` if experimentation/sampling is needed (multiple candidate variants)
/// Returns `Ok(false)` if direct inference is needed (single predetermined variant - no sampling)
fn prepare_candidate_variants(
    candidate_variants: &mut BTreeMap<String, Arc<VariantInfo>>,
    tags: &mut HashMap<String, String>,
    pinned_variant_name: Option<&str>,
    dynamic_variant_config: Option<UninitializedVariantInfo>,
    template_config: &mut Arc<TemplateConfig<'static>>,
    function: &FunctionConfig,
    function_name: String,
) -> Result<bool, Error> {
    let needs_sampling = match (pinned_variant_name, dynamic_variant_config) {
        // If a variant is pinned, only that variant should be attempted
        (Some(variant_name), None) => {
            candidate_variants.retain(|k, _| k == variant_name);

            // If the pinned variant doesn't exist, return an error
            if candidate_variants.is_empty() {
                return Err(ErrorDetails::UnknownVariant {
                    name: variant_name.to_string(),
                }
                .into());
            }
            tags.insert(
                "tensorzero::variant_pinned".to_string(),
                variant_name.to_string(),
            );
            false // Direct inference - no sampling needed
        }
        (None, Some(dynamic_variant_config)) => {
            // Replace the variant config with just the dynamic variant
            let candidate_variant_info = load_dynamic_variant_info(
                dynamic_variant_config,
                function.schemas(),
                function_name,
            )?;

            // Replace templates in the template config with the ones passed in
            // We Clone here so that we can still reference the old templates that don't conflict
            let mut dynamic_template_config: TemplateConfig = (**template_config).clone();
            for path_with_contents in candidate_variant_info.get_all_template_paths() {
                let template_name = path_with_contents.path.get_template_key();
                if dynamic_template_config.contains_template(&template_name) {
                    return Err(ErrorDetails::InvalidDynamicTemplatePath {
                        name: template_name,
                    }
                    .into());
                }
                dynamic_template_config
                    .add_template(template_name, path_with_contents.contents.clone())?;
            }
            *template_config = Arc::new(dynamic_template_config);
            candidate_variants.clear();
            candidate_variants.insert(
                "tensorzero::dynamic_variant".to_string(),
                Arc::new(candidate_variant_info),
            );
            false // Direct inference - no sampling needed
        }
        // If neither variant_name nor internal_dynamic_variant_config is set, we need sampling
        (None, None) => true,
        (Some(_), Some(_)) => {
            return Err(ErrorDetails::InvalidRequest {
                message: "`variant_name` and `internal_dynamic_variant_config` cannot both be set."
                    .to_string(),
            }
            .into())
        }
    };
    Ok(needs_sampling)
}

#[cfg(test)]
mod tests {
    use super::*;

    use object_store::path::Path;
    use serde_json::json;
    use std::time::Duration;
    use uuid::Uuid;

    use crate::inference::types::{
        storage::{StorageKind, StoragePath},
        Base64File, ChatInferenceResultChunk, ContentBlockChunk, File, InputMessageContent,
        JsonInferenceResultChunk, ObjectStoragePointer, Role, TextChunk, UrlFile,
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
            raw_response: String::new(),
            latency: Duration::from_millis(100),
        });
        let raw_request = "raw request".to_string();
        let inference_metadata = InferenceMetadata {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            input: Arc::new(LazyResolvedInput {
                messages: vec![],
                system: None,
            }),
            dryrun: false,
            inference_params: InferenceParams::default(),
            start_time: Instant::now(),
            model_name: "test_model".into(),
            model_provider_name: "test_provider".into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            system: None,
            input_messages: vec![],
            previous_model_inference_results: vec![],
            tags: HashMap::new(),
            tool_config: None,
            dynamic_output_schema: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            include_original_response: false,
        };

        let result = prepare_response_chunk(&inference_metadata, chunk, &mut None).unwrap();
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

        // Test case 2: Valid JSON ProviderInferenceResponseChunk
        let chunk = InferenceResultChunk::Json(JsonInferenceResultChunk {
            raw: Some("Test content".to_string()),
            thought: Some("Thought 1".to_string()),
            created: 0,
            usage: None,
            raw_response: String::new(),
            latency: Duration::from_millis(100),
            finish_reason: Some(FinishReason::Stop),
        });
        let inference_metadata = InferenceMetadata {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            input: Arc::new(LazyResolvedInput {
                messages: vec![],
                system: None,
            }),
            dryrun: false,
            inference_params: InferenceParams::default(),
            start_time: Instant::now(),
            model_name: "test_model".into(),
            model_provider_name: "test_provider".into(),
            raw_request: raw_request.clone(),
            raw_response: None,
            system: None,
            input_messages: vec![],
            previous_model_inference_results: vec![],
            tags: HashMap::new(),
            tool_config: None,
            dynamic_output_schema: None,
            cached: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            include_original_response: false,
        };

        let result = prepare_response_chunk(&inference_metadata, chunk, &mut None).unwrap();
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

    #[test]
    fn test_deserialize_file_content_untagged_url() {
        // Test backwards compatibility: untagged URL should still deserialize
        let input_with_url = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "url": "https://example.com/file.txt",
                            "mime_type": "image/png"
                        }
                    ]
                }
            ]
        });

        let input_with_url: Input = serde_json::from_value(input_with_url).unwrap();
        assert_eq!(input_with_url.messages.len(), 1);
        assert_eq!(input_with_url.messages[0].role, Role::User);
        assert_eq!(input_with_url.messages[0].content.len(), 1);
        assert_eq!(
            input_with_url.messages[0].content[0],
            InputMessageContent::File(File::Url(UrlFile {
                url: "https://example.com/file.txt".parse().unwrap(),
                mime_type: Some(mime::IMAGE_PNG),
                detail: None,
            }))
        );
    }

    #[test]
    fn test_deserialize_file_content_untagged_base64() {
        // Test backwards compatibility: untagged Base64 should still deserialize
        let input_with_base64 = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "data": "fake_base64_data",
                            "mime_type": "image/png"
                        }
                    ]
                }
            ]
        });

        let input_with_base64: Input = serde_json::from_value(input_with_base64).unwrap();
        assert_eq!(input_with_base64.messages.len(), 1);
        assert_eq!(input_with_base64.messages[0].role, Role::User);
        assert_eq!(input_with_base64.messages[0].content.len(), 1);
        assert_eq!(
            input_with_base64.messages[0].content[0],
            InputMessageContent::File(File::Base64(
                Base64File::new(None, mime::IMAGE_PNG, "fake_base64_data".to_string(), None,)
                    .expect("test data should be valid")
            ))
        );
    }

    #[test]
    fn test_serialize_file_content_always_tagged() {
        // Test that serialization always produces tagged format
        let file_url = File::Url(UrlFile {
            url: "https://example.com/file.txt".parse().unwrap(),
            mime_type: Some(mime::IMAGE_PNG),
            detail: None,
        });
        let serialized = serde_json::to_value(&file_url).unwrap();
        assert_eq!(serialized["file_type"], "url");
        assert_eq!(serialized["url"], "https://example.com/file.txt");
        assert_eq!(serialized["mime_type"], "image/png");

        let file_base64 = File::Base64(
            Base64File::new(None, mime::IMAGE_PNG, "fake_base64_data".to_string(), None)
                .expect("test data should be valid"),
        );
        let serialized = serde_json::to_value(&file_base64).unwrap();
        assert_eq!(serialized["file_type"], "base64");
        assert_eq!(serialized["mime_type"], "image/png");
        assert_eq!(serialized["data"], "fake_base64_data");
    }

    #[test]
    fn test_deserialize_file_content_for_url() {
        let input_with_url = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file_type": "url",
                            "url": "https://example.com/file.txt",
                        }
                    ]
                }
            ]
        });

        let input_with_url: Input = serde_json::from_value(input_with_url).unwrap();
        assert_eq!(input_with_url.messages.len(), 1);
        assert_eq!(input_with_url.messages[0].role, Role::User);
        assert_eq!(input_with_url.messages[0].content.len(), 1);
        assert_eq!(
            input_with_url.messages[0].content[0],
            InputMessageContent::File(File::Url(UrlFile {
                url: "https://example.com/file.txt".parse().unwrap(),
                mime_type: None,
                detail: None,
            }))
        );
    }

    #[test]
    fn test_deserialize_file_content_for_base64() {
        let input_with_base64 = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file_type": "base64",
                            "data": "fake_base64_data",
                            "mime_type": "image/png"
                        }
                    ]
                }
            ]
        });

        let input_with_base64: Input = serde_json::from_value(input_with_base64).unwrap();
        assert_eq!(input_with_base64.messages.len(), 1);
        assert_eq!(input_with_base64.messages[0].role, Role::User);
        assert_eq!(input_with_base64.messages[0].content.len(), 1);
        assert_eq!(
            input_with_base64.messages[0].content[0],
            InputMessageContent::File(File::Base64(
                Base64File::new(None, mime::IMAGE_PNG, "fake_base64_data".to_string(), None,)
                    .expect("test data should be valid")
            ))
        );
    }

    #[test]
    fn test_deserialize_file_content_for_object_storage() {
        let input_with_object_storage = json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file_type": "object_storage",
                            "mime_type": "image/png",
                            "storage_path": {
                                "kind": {
                                    "type": "s3_compatible",
                                    "bucket_name": "test-bucket",
                                    "region": "test-region",
                                    "endpoint": "test-endpoint",
                                    "allow_http": false,
                                    "prefix": ""
                                },
                                "path": "test-path"
                            }
                        }
                    ]
                }
            ]
        });

        let input_with_object_storage: Input =
            serde_json::from_value(input_with_object_storage).unwrap();
        assert_eq!(input_with_object_storage.messages.len(), 1);
        assert_eq!(input_with_object_storage.messages[0].role, Role::User);
        assert_eq!(input_with_object_storage.messages[0].content.len(), 1);
        assert_eq!(
            input_with_object_storage.messages[0].content[0],
            InputMessageContent::File(File::ObjectStoragePointer(ObjectStoragePointer {
                source_url: None,
                mime_type: mime::IMAGE_PNG,
                storage_path: StoragePath {
                    kind: StorageKind::S3Compatible {
                        bucket_name: Some("test-bucket".to_string()),
                        region: Some("test-region".to_string()),
                        endpoint: Some("test-endpoint".to_string()),
                        allow_http: Some(false),
                        prefix: String::new(),
                    },
                    path: Path::from("test-path"),
                },
                detail: None,
            }))
        );
    }

    #[test]
    fn test_file_roundtrip_serialization() {
        // Test that serialize -> deserialize maintains data integrity
        let original = File::Base64(
            Base64File::new(None, mime::IMAGE_JPEG, "abcdef".to_string(), None)
                .expect("test data should be valid"),
        );

        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: File = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original, deserialized);

        // Verify serialized format is tagged
        let serialized_value: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(serialized_value["file_type"], "base64");
    }
}
