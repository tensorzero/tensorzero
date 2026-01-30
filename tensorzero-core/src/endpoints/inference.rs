use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json, debug_handler};
use futures::FutureExt;
use futures::stream::Stream;
use futures_core::FusedStream;
use indexmap::IndexMap;
use metrics::{Label, counter};
use schemars::JsonSchema;
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
use tracing_futures::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use uuid::Uuid;

use crate::cache::{CacheOptions, CacheParamsOptions};
use crate::config::snapshot::SnapshotHash;
use crate::config::{Config, ErrorContext, OtlpConfig, SchemaData, UninitializedVariantInfo};
use crate::db::clickhouse::{ClickHouseConnectionInfo, TableName};
use crate::db::postgres::PostgresConnectionInfo;
use crate::embeddings::EmbeddingModelTable;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use crate::experimentation::ExperimentationConfig;
use crate::function::{DEFAULT_FUNCTION_NAME, FunctionConfig, FunctionConfigChat};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, ServiceTier,
};
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::extra_stuff::validate_inference_filters;
use crate::inference::types::resolved_input::LazyResolvedInput;
use crate::inference::types::usage::{
    aggregate_usage_across_model_inferences, aggregate_usage_from_single_streaming_model_inference,
};
use crate::inference::types::{
    ApiType, ChatInferenceDatabaseInsert, ChatInferenceResultChunk, CollectChunksArgs,
    ContentBlockChatOutput, ContentBlockChunk, FetchContext, FinishReason, InferenceResult,
    InferenceResultChunk, InferenceResultStream, Input, InputExt, InternalJsonInferenceOutput,
    JsonInferenceDatabaseInsert, JsonInferenceOutput, JsonInferenceResultChunk,
    ModelInferenceResponseWithMetadata, RawResponseEntry, RawUsageEntry, RequestMessage,
    ResolvedInput, TextChunk, Usage, collect_chunks,
};
use crate::jsonschema_util::JSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::observability::request_logging::HttpMetricData;
use crate::rate_limiting::{RateLimitingManager, ScopeInfo};
use crate::relay::TensorzeroRelay;
use crate::tool::{DynamicToolParams, ToolCallConfig, ToolChoice};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};
use crate::variant::chat_completion::UninitializedChatCompletionConfig;
use crate::variant::dynamic::load_dynamic_variant_info;
use crate::variant::{InferenceConfig, JsonMode, Variant, VariantConfig, VariantInfo};
use tensorzero_auth::middleware::RequestApiKeyExtension;

use crate::endpoints::validate_tags;
use crate::endpoints::workflow_evaluation_run::validate_inference_episode_id_and_apply_workflow_evaluation_run;

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Default, Deserialize, Serialize)]
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
    #[serde(default, skip_serializing)]
    pub credentials: InferenceCredentials,
    /// DEPRECATED (#5697 / 2026.4+): Use `include_raw_response` instead.
    /// If `true`, add an `original_response` field to the response, containing the raw string response from the model.
    /// Note that for complex variants (e.g. `experimental_best_of_n_sampling`), the response may not contain `original_response`
    /// if the fuser/judge model failed.
    #[serde(default)]
    pub include_original_response: bool,
    /// If `true`, add a `raw_response` field to the response, containing the raw string response from the model.
    /// Note that for complex variants (e.g. `experimental_best_of_n_sampling`), the response may not contain `raw_response`
    /// if the fuser/judge model failed.
    #[serde(default)]
    pub include_raw_response: bool,
    /// If `true`, include `raw_usage` in the response's `usage` field, containing the raw usage data from each model inference.
    #[serde(default)]
    pub include_raw_usage: bool,
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
    pub dynamic_output_schema: Option<JSONSchema>,
    pub cached: bool,
    pub extra_body: UnfilteredInferenceExtraBody,
    pub json_mode: Option<JsonMode>,
    pub extra_headers: UnfilteredInferenceExtraHeaders,
    pub fetch_and_encode_input_files_before_inference: bool,
    pub include_original_response: bool,
    pub include_raw_response: bool,
    pub include_raw_usage: bool,
    pub model_inference_id: Uuid,
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
        rate_limiting_manager,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(params): StructuredJson<Params>,
) -> Response<Body> {
    let mut metric_data = HttpMetricData {
        extra_overhead_labels: vec![],
    };
    // If 'function_name' and 'model_name' are both provided, we'll emit
    // an error when we call `inference`
    if let Some(function_name) = &params.function_name {
        metric_data
            .extra_overhead_labels
            .push(Label::new("function_name", function_name.clone()));
    } else if params.model_name.is_some() {
        metric_data
            .extra_overhead_labels
            .push(Label::new("function_name", "tensorzero::default"));
    }
    let inference_output = Box::pin(inference(
        config,
        &http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        rate_limiting_manager,
        params,
        api_key_ext,
    ))
    .await;
    let mut response = match inference_output {
        Ok(data) => {
            if let Some(variant_name) = data.exactly_one_variant {
                metric_data
                    .extra_overhead_labels
                    .push(Label::new("variant_name", variant_name));
            }
            match data.output {
                InferenceOutput::NonStreaming(response) => Json(response).into_response(),
                InferenceOutput::Streaming(stream) => {
                    let event_stream = prepare_serialized_events(stream);

                    Sse::new(event_stream)
                        .keep_alive(axum::response::sse::KeepAlive::new())
                        .into_response()
                }
            }
        }
        Err(e) => e.into_response(),
    };
    response.extensions_mut().insert(metric_data);
    response
}

pub type InferenceStream =
    Pin<Box<dyn FusedStream<Item = Result<InferenceResponseChunk, Error>> + Send>>;

#[expect(clippy::large_enum_variant)]
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

#[derive(Copy, Clone, Debug)]
pub struct InferenceIds {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
}

pub struct InferenceOutputData {
    pub output: InferenceOutput,
    /// If `Some`, then we tried exactly one variant (which succeeded)
    /// If multiple variants were tried (regardless of whether or not one eventually succeeded), then this will be `None`
    pub exactly_one_variant: Option<String>,
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
#[expect(
    clippy::too_many_arguments,
    reason = "Function signature matches existing API pattern"
)]
pub async fn inference(
    config: Arc<Config>,
    http_client: &TensorzeroHttpClient,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    postgres_connection_info: PostgresConnectionInfo,
    deferred_tasks: TaskTracker,
    rate_limiting_manager: Arc<RateLimitingManager>,
    mut params: Params,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
) -> Result<InferenceOutputData, Error> {
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

    if params.include_original_response {
        tracing::warn!(
            "The `include_original_response` parameter is deprecated. Use `include_raw_response` instead."
        );
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

    let (function, function_name) = find_function(&params, &config).await?;
    let mut candidate_variants: BTreeMap<String, Arc<VariantInfo>> =
        function.variants().clone().into_iter().collect();

    // If the function has no variants and no dynamic variant config, return an error
    if candidate_variants.is_empty() && params.internal_dynamic_variant_config.is_none() {
        return Err(ErrorDetails::InvalidFunctionVariants {
            message: format!("Function `{function_name}` has no variants"),
        }
        .into());
    }

    // Validate the input
    function.validate_inference_params(&params).await?;

    // Validate extra_body and extra_headers filters
    validate_inference_filters(
        &params.extra_body,
        &params.extra_headers,
        Some(&function),
        &config.models,
        &config.gateway.relay,
    )
    .await?;

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
        counter!("tensorzero_requests_total", &labels).increment(1);
        counter!("tensorzero_inferences_total", &labels).increment(1);
    }

    // Should we stream the inference?
    let stream = params.stream.unwrap_or(false);

    // Keep track of which variants failed
    let mut variant_errors: IndexMap<String, Error> = IndexMap::new();

    // Set up inference config
    let output_schema = params.output_schema.map(JSONSchema::compile_background);

    let tags = Arc::new(params.tags.clone());

    let inference_clients = InferenceClients {
        http_client: http_client.clone(),
        clickhouse_connection_info: clickhouse_connection_info.clone(),
        postgres_connection_info: postgres_connection_info.clone(),
        credentials: Arc::new(params.credentials.clone()),
        cache_options: (params.cache_options, dryrun).into(),
        tags: tags.clone(),
        rate_limiting_manager,
        otlp_config: config.gateway.export.otlp.clone(),
        deferred_tasks,
        scope_info: ScopeInfo::new(tags.clone(), api_key_ext),
        relay: config.gateway.relay.clone(),
        include_raw_usage: params.include_raw_usage,
        include_raw_response: params.include_raw_response,
    };

    let inference_models = InferenceModels {
        models: config.models.clone(),
        embedding_models: config.embedding_models.clone(),
    };
    let fetch_context = FetchContext {
        client: http_client,
        object_store_info: &config.object_store_info,
    };
    let resolved_input = Arc::new(params.input.into_lazy_resolved_input(&fetch_context)?);

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

        let output = Box::pin(infer_variant(InferVariantArgs {
            variant_name: variant_name.clone(),
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
            include_raw_response: params.include_raw_response,
            include_raw_usage: params.include_raw_usage,
        }))
        .await?;
        return Ok(InferenceOutputData {
            output,
            exactly_one_variant: Some(variant_name),
        });
    }

    // Keep sampling variants until one succeeds
    let mut already_sampled = false;
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

        let result = Box::pin(infer_variant(InferVariantArgs {
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
            include_raw_response: params.include_raw_response,
            include_raw_usage: params.include_raw_usage,
        }))
        .await;

        match result {
            Ok(output) => {
                return Ok(InferenceOutputData {
                    output,
                    exactly_one_variant: if already_sampled {
                        None
                    } else {
                        Some(variant_name)
                    },
                });
            }
            Err(e) => {
                tracing::warn!(
                    "functions.{function_name}.variants.{variant_name} failed during inference: {e}",
                    function_name = function_name,
                    variant_name = variant_name,
                );
                variant_errors.insert(variant_name, e);
                already_sampled = true;
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
    output_schema: &'a Option<JSONSchema>,
    config: &'a Arc<Config>,
    clickhouse_connection_info: &'a ClickHouseConnectionInfo,
    tags: &'a HashMap<String, String>,
    extra_body: &'a UnfilteredInferenceExtraBody,
    extra_headers: &'a UnfilteredInferenceExtraHeaders,
    include_original_response: bool,
    include_raw_response: bool,
    include_raw_usage: bool,
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
        include_raw_response,
        include_raw_usage,
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
        let deferred_tasks = inference_clients.deferred_tasks.clone();
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
            inference_params: model_used_info.inference_params.clone(),
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
            json_mode: model_used_info.inference_params.chat_completion.json_mode,
            extra_headers,
            include_original_response,
            include_raw_response,
            include_raw_usage,
            fetch_and_encode_input_files_before_inference: config
                .gateway
                .fetch_and_encode_input_files_before_inference,
            model_inference_id: model_used_info.model_inference_id,
        };

        let stream = create_stream(
            function.clone(),
            config.clone(),
            inference_metadata,
            stream,
            clickhouse_connection_info.clone(),
            deferred_tasks.clone(),
        );

        Ok(InferenceOutput::Streaming(Box::pin(stream)))
    } else {
        let deferred_tasks = inference_clients.deferred_tasks.clone();
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

        let result = result?;

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
                snapshot_hash: config.hash.clone(),
            };

            let async_writes = config.gateway.observability.async_writes;
            let clickhouse_connection_info = clickhouse_connection_info.clone();
            let config = config.clone();
            let resolved_input = resolved_input.clone();
            // Capture the parent span (function_inference) so we can use it as the parent
            // for write_inference, even if the task is spawned.
            let parent_span = tracing::Span::current();
            // Always spawn a tokio task here. This ensures that 'write_inference' will
            // not be cancelled partway through execution if the outer '/inference' request
            // is cancelled. This reduces the chances that we only write to some tables and not others
            // (but this is inherently best-effort due to ClickHouse's lack of transactions).
            let write_future = deferred_tasks.spawn(async move {
                let _: () = write_inference(
                    &clickhouse_connection_info,
                    &config,
                    Arc::unwrap_or_clone(resolved_input).resolve().await?,
                    result_to_write,
                    write_metadata,
                )
                .await;
                Ok::<_, Error>(())
            }.instrument(tracing::debug_span!(parent: &parent_span, "write_inference", otel.name = "write_inference", stream = false, inference_id = %inference_id, async_writes = async_writes)));
            if !async_writes {
                write_future.await.map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!("Failed to await ClickHouse inference write: {e:?}"),
                    })
                })??;
            }
        }

        let response = InferenceResponse::new(
            result,
            episode_id,
            variant_name.clone(),
            include_raw_usage,
            include_original_response,
            include_raw_response,
        );

        Ok(InferenceOutput::NonStreaming(response))
    }
}

/// Finds a function by `function_name` or `model_name`, erroring if an
/// invalid combination of parameters is provided.
/// If `model_name` is specified, then we use the special 'default' function
/// Returns the function config and the function name
async fn find_function(
    params: &Params,
    config: &Config,
) -> Result<(Arc<FunctionConfig>, String), Error> {
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

            // Validate extra_body and extra_headers filters
            validate_inference_filters(
                &params.extra_body,
                &params.extra_headers,
                None,
                &config.models,
                &config.gateway.relay,
            )
            .await?;

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

/// Creates an artificial chunk containing `raw_usage` from previous model inferences (e.g. best-of-N candidates).
/// Returns `None` if `include_raw_usage` is false or there are no non-cached entries with raw_usage.
fn create_previous_raw_usage_chunk(
    metadata: &InferenceMetadata,
    function: &FunctionConfig,
) -> Option<InferenceResultChunk> {
    if !metadata.include_raw_usage {
        return None;
    }

    // Filter out `raw_usage` from cached model inferences
    let entries: Vec<RawUsageEntry> = metadata
        .previous_model_inference_results
        .iter()
        .filter(|r| !r.cached)
        .flat_map(|r| r.raw_usage.clone().unwrap_or_default())
        .collect();

    if entries.is_empty() {
        return None;
    }

    let raw_usage = Some(entries);
    let chunk = match function {
        FunctionConfig::Chat(_) => InferenceResultChunk::Chat(ChatInferenceResultChunk {
            raw_usage,
            ..Default::default()
        }),
        FunctionConfig::Json(_) => InferenceResultChunk::Json(JsonInferenceResultChunk {
            raw_usage,
            ..Default::default()
        }),
    };
    Some(chunk)
}

/// Creates an artificial chunk containing `raw_response` entries from previous model inferences (e.g. best-of-N candidates).
/// Returns `None` if `include_raw_response` is false or there are no non-cached entries.
fn create_previous_raw_response_chunk(
    metadata: &InferenceMetadata,
    function: &FunctionConfig,
) -> Option<InferenceResultChunk> {
    if !metadata.include_raw_response {
        return None;
    }

    // Collect raw response entries, preferring passed-through entries from relay
    let entries: Vec<RawResponseEntry> = metadata
        .previous_model_inference_results
        .iter()
        .filter(|r| !r.cached)
        .flat_map(|r| {
            // If there are passed-through relay_raw_response (from relay), use them
            if let Some(passed_through) = &r.relay_raw_response {
                passed_through.clone()
            } else {
                // Otherwise, generate entries from the model inference result
                let api_type = r
                    .raw_usage
                    .as_ref()
                    .and_then(|entries| entries.first())
                    .map(|entry| entry.api_type)
                    .unwrap_or(ApiType::ChatCompletions);
                vec![RawResponseEntry {
                    model_inference_id: r.id,
                    provider_type: r.model_provider_name.to_string(),
                    api_type,
                    data: r.raw_response.clone(),
                }]
            }
        })
        .collect();

    if entries.is_empty() {
        return None;
    }

    let raw_response = Some(entries);
    let chunk = match function {
        FunctionConfig::Chat(_) => InferenceResultChunk::Chat(ChatInferenceResultChunk {
            raw_response,
            ..Default::default()
        }),
        FunctionConfig::Json(_) => InferenceResultChunk::Json(JsonInferenceResultChunk {
            raw_response,
            ..Default::default()
        }),
    };
    Some(chunk)
}

/// Transform the response(s) from the model providers for our inference APIs.
///
/// NB: After this function, the stream is then further processed by:
/// - TensorZero Inference API: `prepare_serialized_events`
/// - OpenAI-Compatible Inference API: `prepare_serialized_openai_compatible_events`
fn create_stream(
    function: Arc<FunctionConfig>,
    config: Arc<Config>,
    metadata: InferenceMetadata,
    mut stream: InferenceResultStream,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    deferred_tasks: TaskTracker,
) -> impl FusedStream<Item = Result<InferenceResponseChunk, Error>> + Send {
    // Capture the parent span (function_inference) so we can use it as the parent
    // for write_inference later, even after function_inference has completed.
    let parent_span = tracing::Span::current();

    async_stream::stream! {
        let mut buffer = vec![];

        // If previous model inferences (e.g. best-of-N candidates) had `raw_usage`, emit them immediately in an artificial chunk.
        if let Some(chunk) = create_previous_raw_usage_chunk(&metadata, &function) {
            buffer.push(chunk.clone());
            yield Ok(prepare_response_chunk(&metadata, chunk));
        }

        // If previous model inferences (e.g. best-of-N candidates) had `raw_response`, emit them immediately in an artificial chunk.
        if let Some(chunk) = create_previous_raw_response_chunk(&metadata, &function) {
            buffer.push(chunk.clone());
            yield Ok(prepare_response_chunk(&metadata, chunk));
        }

        // Then, send all chunks but strip usage and finish reason
        let mut usages: Vec<Usage> = vec![];
        let mut finish_reasons: Vec<FinishReason> = vec![];
        let mut inference_ttft = None;
        while let Some(chunk) = stream.next().await {
            let mut chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(e);
                    continue;
                }
            };

            // Compute TTFT
            if inference_ttft.is_none() {
                inference_ttft = Some(metadata.start_time.elapsed());
            }

            // Strip usage
            if let Some(u) = chunk.usage() {
                usages.push(*u);
                chunk.set_usage(None);
            }

            // Strip finish reason
            if let Some(fr) = chunk.finish_reason() {
                finish_reasons.push(*fr);
                chunk.set_finish_reason(None);
            }

            buffer.push(chunk.clone());

            // Stream chunk, unless we've stripped all useful information
            if should_stream_chunk_in_create_stream(&chunk, metadata.include_original_response, metadata.include_raw_response, metadata.include_raw_usage) {
                yield Ok(prepare_response_chunk(&metadata, chunk));
            }
        }

        // If we saw multiple chunks with `finish_reason`, warn (unexpected behavior)
        if finish_reasons.len() > 1 {
            tracing::warn!("Received multiple chunks with `finish_reason`, returning the last one: {}", finish_reasons.iter().map(|fr| format!("{fr:?}")).collect::<Vec<_>>().join(", "));
        }
        let finish_reason = finish_reasons.pop();

        // If we saw multiple chunks with `usage`, compute the field-wise max and warn if they are non-cumulative
        // This is the current model's usage (used for database storage)
        let model_inference_usage = aggregate_usage_from_single_streaming_model_inference(usages);
        // Then add the usage from previous inferences (e.g. best-of-N candidates)
        // This is the total usage for the TensorZero inference
        let inference_usage = aggregate_usage_across_model_inferences(
            metadata.previous_model_inference_results.iter().map(ModelInferenceResponseWithMetadata::usage_considering_cached).chain(std::iter::once(model_inference_usage))
        );

        let chunk = match *function {
            FunctionConfig::Chat(_) => InferenceResultChunk::Chat(ChatInferenceResultChunk {
                finish_reason,
                usage: Some(inference_usage),
                ..Default::default()
            }),
            FunctionConfig::Json(_) => InferenceResultChunk::Json(JsonInferenceResultChunk {
                finish_reason,
                usage: Some(inference_usage),
                ..Default::default()
            }),
        };

        buffer.push(chunk.clone());

        yield Ok(prepare_response_chunk(&metadata, chunk));

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
                json_mode: _,
                extra_headers,
                fetch_and_encode_input_files_before_inference,
                include_original_response: _,
                include_raw_response: _,
                include_raw_usage: _,
                model_inference_id,
            } = metadata;

            let config = config.clone();
            let async_writes = config.gateway.observability.async_writes;
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
                    model_inference_id,
                    // Use only the current model's usage, not the aggregated total
                    // (previous model inferences are added separately below)
                    model_inference_usage,
                    finish_reason,
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
                        snapshot_hash: config.hash.clone(),
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
            }.instrument(tracing::debug_span!(parent: &parent_span, "write_inference", otel.name = "write_inference", stream = true, inference_id = %inference_id, async_writes = async_writes));
            if async_writes {
                deferred_tasks.spawn(write_future);
            } else {
                write_future.await;
            }
        }
    }
}

/// Decide whether we should stream an intermediate chunk in `create_stream`.
///
/// We want to stream chunks that have useful information (e.g. content, usage).
///
/// We always want to stream a chunk if `include_original_response` or `include_raw_response` is enabled.
fn should_stream_chunk_in_create_stream(
    chunk: &InferenceResultChunk,
    include_original_response: bool,
    include_raw_response: bool,
    include_raw_usage: bool,
) -> bool {
    if include_original_response || include_raw_response {
        return true;
    }

    match chunk {
        InferenceResultChunk::Chat(c) => {
            let ChatInferenceResultChunk {
                // Always stream these fields
                content,
                // These fields should've been cleared for intermediate chunks in `create_stream`; if they're here, stream
                usage,
                finish_reason,
                // Only stream if `include_raw_usage` is enabled
                raw_usage,
                // Only stream if `include_raw_response` is enabled
                raw_response,
                // We already handled `include_original_response` above
                raw_chunk: _,
                // We don't care about streaming the following fields in isolation
                provider_latency: _,
            } = c;

            // We want to stream the chunk if `raw_usage` is relevant
            if include_raw_usage && raw_usage.as_ref().is_some_and(|x| !x.is_empty()) {
                return true;
            }

            // We want to stream the chunk if `raw_response` is relevant
            if include_raw_response && raw_response.as_ref().is_some_and(|x| !x.is_empty()) {
                return true;
            }

            !content.is_empty() || usage.is_some() || finish_reason.is_some()
        }
        InferenceResultChunk::Json(c) => {
            let JsonInferenceResultChunk {
                // Always stream these fields
                raw,
                // These fields should've been cleared for intermediate chunks in `create_stream`; if they're here, stream
                usage,
                finish_reason,
                // Only stream if `include_raw_usage` is enabled
                raw_usage,
                // Only stream if `include_raw_response` is enabled
                raw_response,
                // We already handled `include_original_response` above
                raw_chunk: _,
                // We never actually stream this field, so we don't need it
                thought_chunks: _,
                // We don't care about streaming the following fields in isolation
                provider_latency: _,
            } = c;

            // We want to stream the chunk if `raw_usage` is relevant
            if include_raw_usage && raw_usage.as_ref().is_some_and(|x| !x.is_empty()) {
                return true;
            }

            // We want to stream the chunk if `raw_response` is relevant
            if include_raw_response && raw_response.as_ref().is_some_and(|x| !x.is_empty()) {
                return true;
            }

            raw.as_ref().is_some_and(|x| !x.is_empty())
                || usage.is_some()
                || finish_reason.is_some()
        }
    }
}

fn prepare_response_chunk(
    metadata: &InferenceMetadata,
    chunk: InferenceResultChunk,
) -> InferenceResponseChunk {
    InferenceResponseChunk::new(
        chunk,
        metadata.inference_id,
        metadata.episode_id,
        metadata.variant_name.clone(),
        metadata.cached,
        metadata.include_original_response,
        metadata.include_raw_response,
        metadata.json_mode,
        metadata.include_raw_usage,
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
    pub snapshot_hash: SnapshotHash,
}

async fn write_inference(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    config: &Config,
    input: ResolvedInput,
    result: InferenceResult,
    metadata: InferenceDatabaseInsertMetadata,
) {
    let model_responses: Vec<serde_json::Value> = result
        .get_serialized_model_inferences(metadata.snapshot_hash.clone())
        .await;
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(untagged, rename_all = "snake_case")]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
    Json(JsonInferenceResponse),
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ChatInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub content: Vec<ContentBlockChatOutput>,
    pub usage: Usage,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// DEPRECATED (#5697 / 2026.4+): Use `raw_response` instead.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_response: Option<String>,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<Vec<RawResponseEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct JsonInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub output: JsonInferenceOutput,
    pub usage: Usage,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// DEPRECATED (#5697 / 2026.4+): Use `raw_response` instead.
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_response: Option<String>,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<Vec<RawResponseEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

impl InferenceResponse {
    pub fn new(
        inference_result: InferenceResult,
        episode_id: Uuid,
        variant_name: String,
        include_raw_usage: bool,
        include_original_response: bool,
        include_raw_response: bool,
    ) -> Self {
        let usage = inference_result.usage_considering_cached();

        // Build raw_usage if requested
        // Returns Some(entries) if requested (even if empty when all cached), None if not requested
        let raw_usage = if include_raw_usage {
            let entries: Vec<RawUsageEntry> = inference_result
                .model_inference_results()
                .iter()
                .filter(|r| !r.cached) // Exclude TensorZero cache hits
                .flat_map(|r| r.raw_usage.clone().unwrap_or_default())
                .collect();
            Some(entries)
        } else {
            None
        };

        // Build raw_response if requested
        // Returns Some(entries) if requested (even if empty when all cached), None if not requested
        let raw_response = if include_raw_response {
            let entries: Vec<RawResponseEntry> = inference_result
                .model_inference_results()
                .iter()
                .filter(|r| !r.cached) // Exclude TensorZero cache hits
                .flat_map(|r| {
                    // If there are passed-through relay_raw_response (from relay), use them
                    if let Some(passed_through) = &r.relay_raw_response {
                        passed_through.clone()
                    } else {
                        // Otherwise, generate entries from the model inference result
                        let api_type = r
                            .raw_usage
                            .as_ref()
                            .and_then(|entries| entries.first())
                            .map(|entry| entry.api_type)
                            .unwrap_or(ApiType::ChatCompletions);
                        vec![RawResponseEntry {
                            model_inference_id: r.id,
                            provider_type: r.model_provider_name.to_string(),
                            api_type,
                            data: r.raw_response.clone(),
                        }]
                    }
                })
                .collect();
            Some(entries)
        } else {
            None
        };

        match inference_result {
            InferenceResult::Chat(result) => {
                // Populate original_response if deprecated flag was set
                let original_response = if include_original_response {
                    result.original_response
                } else {
                    None
                };
                InferenceResponse::Chat(ChatInferenceResponse {
                    inference_id: result.inference_id,
                    episode_id,
                    variant_name,
                    content: result.content,
                    usage,
                    raw_usage: raw_usage.clone(),
                    original_response,
                    raw_response: raw_response.clone(),
                    finish_reason: result.finish_reason,
                })
            }
            InferenceResult::Json(result) => {
                let InternalJsonInferenceOutput { raw, parsed, .. } = result.output;
                let output = JsonInferenceOutput { raw, parsed };
                // Populate original_response if deprecated flag was set
                let original_response = if include_original_response {
                    result.original_response
                } else {
                    None
                };
                InferenceResponse::Json(JsonInferenceResponse {
                    inference_id: result.inference_id,
                    episode_id,
                    variant_name,
                    output,
                    usage,
                    raw_usage,
                    original_response,
                    raw_response,
                    finish_reason: result.finish_reason,
                })
            }
        }
    }

    pub fn usage(&self) -> Usage {
        match self {
            InferenceResponse::Chat(c) => c.usage,
            InferenceResponse::Json(j) => j.usage,
        }
    }

    pub fn raw_usage(&self) -> Option<&Vec<RawUsageEntry>> {
        match self {
            InferenceResponse::Chat(c) => c.raw_usage.as_ref(),
            InferenceResponse::Json(j) => j.raw_usage.as_ref(),
        }
    }

    pub fn raw_response(&self) -> Option<&Vec<RawResponseEntry>> {
        match self {
            InferenceResponse::Chat(c) => c.raw_response.as_ref(),
            InferenceResponse::Json(j) => j.raw_response.as_ref(),
        }
    }

    pub fn finish_reason(&self) -> Option<FinishReason> {
        match self {
            InferenceResponse::Chat(c) => c.finish_reason,
            InferenceResponse::Json(j) => j.finish_reason,
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
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// Raw responses from previous model inferences (e.g., best-of-n candidates).
    /// Emitted in the first chunk of a streaming response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<Vec<RawResponseEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// DEPRECATED (#5697 / 2026.4+): Use `raw_chunk` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_chunk: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_chunk: Option<String>,
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
    pub raw_usage: Option<Vec<RawUsageEntry>>,
    /// Raw responses from previous model inferences (e.g., best-of-n candidates).
    /// Emitted in the first chunk of a streaming response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<Vec<RawResponseEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// DEPRECATED (#5697 / 2026.4+): Use `raw_chunk` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_chunk: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_chunk: Option<String>,
}

impl InferenceResponseChunk {
    #[expect(clippy::too_many_arguments)]
    fn new(
        inference_result: InferenceResultChunk,
        inference_id: Uuid,
        episode_id: Uuid,
        variant_name: String,
        cached: bool,
        include_original_response: bool,
        include_raw_response: bool,
        json_mode: Option<JsonMode>,
        include_raw_usage: bool,
    ) -> Self {
        // Compute the usage
        let usage = if cached {
            // `usage` represents billed tokens. We set values to 0 if TensorZero cached the inference.
            // Only include usage on chunks that originally had it (i.e., the final chunk).
            inference_result.usage().map(|_| Usage {
                input_tokens: Some(0),
                output_tokens: Some(0),
            })
        } else {
            inference_result.usage().copied()
        };

        // Compute the raw usage
        let raw_usage = if include_raw_usage {
            inference_result.raw_usage().cloned()
        } else {
            None
        };

        // Pass through raw_response if include_raw_response is set
        // This is populated by create_previous_raw_response_chunk for artificial chunks
        let raw_response = if include_raw_response {
            inference_result.raw_response().cloned()
        } else {
            None
        };

        match inference_result {
            InferenceResultChunk::Chat(result) => {
                // For chat functions with json_mode="tool", convert tool call chunks to text chunks
                let content = if json_mode == Some(JsonMode::Tool) {
                    result
                        .content
                        .into_iter()
                        .map(|chunk| match chunk {
                            ContentBlockChunk::ToolCall(tool_call) => {
                                // Convert tool call arguments to text chunk
                                ContentBlockChunk::Text(TextChunk {
                                    id: tool_call.id,
                                    text: tool_call.raw_arguments,
                                })
                            }
                            other => other,
                        })
                        .collect()
                } else {
                    result.content
                };

                // Compute chunk fields based on request flags
                let (original_chunk, raw_chunk) = Self::compute_chunk_fields(
                    result.raw_chunk,
                    include_original_response,
                    include_raw_response,
                );

                InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
                    inference_id,
                    episode_id,
                    variant_name,
                    content,
                    // Token usage is intended to represent 'billed tokens',
                    // so set it to zero if the result is cached
                    usage,
                    raw_usage,
                    raw_response: raw_response.clone(),
                    finish_reason: result.finish_reason,
                    original_chunk,
                    raw_chunk,
                })
            }
            InferenceResultChunk::Json(result) => {
                // Compute chunk fields based on request flags
                let (original_chunk, raw_chunk) = Self::compute_chunk_fields(
                    result.raw_chunk,
                    include_original_response,
                    include_raw_response,
                );

                InferenceResponseChunk::Json(JsonInferenceResponseChunk {
                    inference_id,
                    episode_id,
                    variant_name,
                    raw: result.raw.unwrap_or_default(),
                    // Token usage is intended to represent 'billed tokens',
                    // so set it to zero if the result is cached
                    usage,
                    raw_usage,
                    raw_response,
                    finish_reason: result.finish_reason,
                    original_chunk,
                    raw_chunk,
                })
            }
        }
    }

    /// Helper to compute original_chunk and raw_chunk fields based on request flags.
    /// If both flags are true, both fields get the same value (cloned).
    /// Returns None if the source is empty (e.g., for fake streams).
    fn compute_chunk_fields(
        source: String,
        include_original: bool,
        include_raw: bool,
    ) -> (Option<String>, Option<String>) {
        // Don't serialize empty strings - return None instead
        let source = if source.is_empty() {
            None
        } else {
            Some(source)
        };
        match (include_original, include_raw) {
            (true, true) => (source.clone(), source),
            (true, false) => (source, None),
            (false, true) => (None, source),
            (false, false) => (None, None),
        }
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
    pub rate_limiting_manager: Arc<RateLimitingManager>,
    pub otlp_config: OtlpConfig,
    pub deferred_tasks: TaskTracker,
    pub scope_info: ScopeInfo,
    pub relay: Option<TensorzeroRelay>,
    pub include_raw_usage: bool,
    pub include_raw_response: bool,
}

// Carryall struct for models used in inference
#[derive(Clone, Debug)]
pub struct InferenceModels {
    pub models: Arc<ModelTable>,
    pub embedding_models: Arc<EmbeddingModelTable>,
}

/// InferenceParams is the top-level struct for inference parameters.
/// We backfill these from the configs given in the variants used and ultimately write them to the database.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(deny_unknown_fields)]
pub struct InferenceParams {
    pub chat_completion: ChatCompletionInferenceParams,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
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
            // We clone here so that we can still reference the old templates that don't conflict
            let mut dynamic_template_config: TemplateConfig = (**template_config).clone();
            for path_with_contents in candidate_variant_info.get_all_template_paths() {
                let template_name = path_with_contents.path.get_template_key();
                if dynamic_template_config.contains_template(&template_name) {
                    tracing::debug!(
                        "Dynamic template `{}` is overriding an existing template.",
                        template_name
                    );
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
            .into());
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
        ApiType, Base64File, ChatInferenceResultChunk, ContentBlockChunk, ContentBlockOutput, File,
        InputMessageContent, JsonInferenceResultChunk, Latency, ModelInferenceResponseWithMetadata,
        ObjectStoragePointer, RequestMessagesOrBatch, Role, Text, TextChunk, ThoughtChunk, UrlFile,
        storage::{StorageKind, StoragePath},
        usage::RawUsageEntry,
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
            usage: None,
            raw_usage: None,
            raw_response: None,
            finish_reason: Some(FinishReason::Stop),
            raw_chunk: String::new(),
            provider_latency: Some(Duration::from_millis(100)),
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
            json_mode: None,
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            include_original_response: false,
            include_raw_response: false,
            include_raw_usage: false,
            model_inference_id: Uuid::now_v7(),
        };

        let result = prepare_response_chunk(&inference_metadata, chunk);
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
            thought_chunks: vec![ThoughtChunk {
                id: "0".to_string(),
                text: Some("Thought 1".to_string()),
                signature: None,
                summary_id: None,
                summary_text: None,
                provider_type: None,
                extra_data: None,
            }],
            usage: None,
            raw_usage: None,
            raw_response: None,
            raw_chunk: String::new(),
            provider_latency: Some(Duration::from_millis(100)),
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
            json_mode: None,
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            include_original_response: false,
            include_raw_response: false,
            include_raw_usage: false,
            model_inference_id: Uuid::now_v7(),
        };

        let result = prepare_response_chunk(&inference_metadata, chunk);
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

    #[tokio::test]
    async fn test_find_function_no_function_model() {
        let err = find_function(
            &Params {
                function_name: None,
                model_name: None,
                ..Default::default()
            },
            &Config::default(),
        )
        .await
        .expect_err("find_function should fail without either arg");
        assert!(
            err.to_string()
                .contains("Either `function_name` or `model_name` must be provided"),
            "Unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_find_function_both_function_model() {
        let err = find_function(
            &Params {
                function_name: Some("my_function".to_string()),
                model_name: Some("my_model".to_string()),
                ..Default::default()
            },
            &Config::default(),
        )
        .await
        .expect_err("find_function should fail with both args provided");
        assert!(
            err.to_string()
                .contains("Only one of `function_name` or `model_name` can be provided"),
            "Unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_find_function_model_and_variant() {
        let err = find_function(
            &Params {
                function_name: None,
                model_name: Some("my_model".to_string()),
                variant_name: Some("my_variant".to_string()),
                ..Default::default()
            },
            &Config::default(),
        )
        .await
        .expect_err("find_function should fail without model_name");
        assert!(
            err.to_string()
                .contains("`variant_name` cannot be provided when using `model_name`"),
            "Unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_find_function_shorthand_model() {
        let (function_config, function_name) = find_function(
            &Params {
                function_name: None,
                model_name: Some("openai::gpt-9000".to_string()),
                ..Default::default()
            },
            &Config::default(),
        )
        .await
        .expect("Failed to find shorthand function");
        assert_eq!(function_name, "tensorzero::default");
        assert_eq!(function_config.variants().len(), 1);
        assert_eq!(
            function_config.variants().keys().next().unwrap(),
            "openai::gpt-9000"
        );
    }

    #[tokio::test]
    async fn test_find_function_shorthand_missing_provider() {
        let err = find_function(
            &Params {
                model_name: Some("fake_provider::gpt-9000".to_string()),
                ..Default::default()
            },
            &Config::default(),
        )
        .await
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
                filename: None,
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
                Base64File::new(
                    None,
                    Some(mime::IMAGE_PNG),
                    "fake_base64_data".to_string(),
                    None,
                    None
                )
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
            filename: None,
        });
        let serialized = serde_json::to_value(&file_url).unwrap();
        assert_eq!(serialized["file_type"], "url");
        assert_eq!(serialized["url"], "https://example.com/file.txt");
        assert_eq!(serialized["mime_type"], "image/png");

        let file_base64 = File::Base64(
            Base64File::new(
                None,
                Some(mime::IMAGE_PNG),
                "fake_base64_data".to_string(),
                None,
                None,
            )
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
                filename: None,
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
                Base64File::new(
                    None,
                    Some(mime::IMAGE_PNG),
                    "fake_base64_data".to_string(),
                    None,
                    None
                )
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
                filename: None,
            }))
        );
    }

    #[test]
    fn test_file_roundtrip_serialization() {
        // Test that serialize -> deserialize maintains data integrity
        let original = File::Base64(
            Base64File::new(
                None,
                Some(mime::IMAGE_JPEG),
                "abcdef".to_string(),
                None,
                None,
            )
            .expect("test data should be valid"),
        );

        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: File = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original, deserialized);

        // Verify serialized format is tagged
        let serialized_value: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(serialized_value["file_type"], "base64");
    }

    /// Helper to create an InferenceMetadata for testing
    fn create_test_metadata() -> InferenceMetadata {
        InferenceMetadata {
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
            raw_request: "raw request".to_string(),
            raw_response: None,
            system: None,
            input_messages: vec![],
            previous_model_inference_results: vec![],
            tags: HashMap::new(),
            tool_config: None,
            dynamic_output_schema: None,
            cached: false,
            extra_body: Default::default(),
            json_mode: None,
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            include_original_response: false,
            include_raw_response: false,
            include_raw_usage: true,
            model_inference_id: Uuid::now_v7(),
        }
    }

    /// Test that raw_usage is passed through from the chunk when include_raw_usage is true
    #[test]
    fn test_prepare_response_chunk_passes_through_raw_usage() {
        let metadata = create_test_metadata();

        let raw_usage_entries = vec![RawUsageEntry {
            model_inference_id: Uuid::now_v7(),
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: json!({"prompt_tokens": 10, "completion_tokens": 20}),
        }];

        // Create a chunk WITH raw_usage already set
        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk {
            content: vec![ContentBlockChunk::Text(TextChunk {
                text: "Test content".to_string(),
                id: "0".to_string(),
            })],
            usage: Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            }),
            raw_usage: Some(raw_usage_entries.clone()),
            raw_response: None,
            finish_reason: Some(FinishReason::Stop),
            raw_chunk: String::new(),
            provider_latency: Some(Duration::from_millis(100)),
        });

        let result = prepare_response_chunk(&metadata, chunk);

        match result {
            InferenceResponseChunk::Chat(c) => {
                assert!(c.usage.is_some(), "usage should be present");
                let raw_usage = c
                    .raw_usage
                    .expect("raw_usage should be passed through from chunk");
                assert_eq!(
                    raw_usage.len(),
                    1,
                    "raw_usage should contain the expected entries"
                );
                assert_eq!(
                    raw_usage[0].provider_type, "openai",
                    "raw_usage entry should have correct provider_type"
                );
            }
            InferenceResponseChunk::Json(_) => {
                panic!("Expected ChatInferenceResponseChunk");
            }
        }
    }

    /// Test that raw_usage is NOT included when include_raw_usage is false
    #[test]
    fn test_prepare_response_chunk_excludes_raw_usage_when_disabled() {
        let mut metadata = create_test_metadata();
        metadata.include_raw_usage = false;

        let raw_usage_entries = vec![RawUsageEntry {
            model_inference_id: Uuid::now_v7(),
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: json!({"prompt_tokens": 10}),
        }];

        // Create a chunk WITH raw_usage set
        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk {
            content: vec![],
            usage: Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            }),
            raw_usage: Some(raw_usage_entries),
            raw_response: None,
            finish_reason: Some(FinishReason::Stop),
            raw_chunk: String::new(),
            provider_latency: Some(Duration::from_millis(100)),
        });

        let result = prepare_response_chunk(&metadata, chunk);

        match result {
            InferenceResponseChunk::Chat(c) => {
                assert!(
                    c.raw_usage.is_none(),
                    "raw_usage should NOT be included when include_raw_usage is false"
                );
            }
            InferenceResponseChunk::Json(_) => panic!("Expected Chat chunk"),
        }
    }

    /// Test that chunks without raw_usage return None for raw_usage
    #[test]
    fn test_prepare_response_chunk_without_raw_usage() {
        let metadata = create_test_metadata();

        // Create a chunk WITHOUT raw_usage
        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk {
            content: vec![ContentBlockChunk::Text(TextChunk {
                text: "Content".to_string(),
                id: "0".to_string(),
            })],
            usage: Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            }),
            raw_usage: None,
            raw_response: None,
            finish_reason: None,
            raw_chunk: String::new(),
            provider_latency: Some(Duration::from_millis(50)),
        });

        let result = prepare_response_chunk(&metadata, chunk);

        match result {
            InferenceResponseChunk::Chat(c) => {
                assert!(
                    c.raw_usage.is_none(),
                    "raw_usage should be None when chunk has no raw_usage"
                );
            }
            InferenceResponseChunk::Json(_) => panic!("Expected Chat chunk"),
        }
    }

    /// Test that cached inferences have usage zeroed
    #[test]
    fn test_prepare_response_chunk_cached_zeros_usage() {
        let mut metadata = create_test_metadata();
        metadata.cached = true;

        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk {
            content: vec![],
            usage: Some(Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
            }),
            raw_usage: None,
            raw_response: None,
            finish_reason: Some(FinishReason::Stop),
            raw_chunk: String::new(),
            provider_latency: Some(Duration::from_millis(100)),
        });

        let result = prepare_response_chunk(&metadata, chunk);

        match result {
            InferenceResponseChunk::Chat(c) => {
                let usage = c.usage.expect("usage should be present");
                assert_eq!(
                    usage.input_tokens,
                    Some(0),
                    "cached inference should have input_tokens zeroed"
                );
                assert_eq!(
                    usage.output_tokens,
                    Some(0),
                    "cached inference should have output_tokens zeroed"
                );
            }
            InferenceResponseChunk::Json(_) => panic!("Expected Chat chunk"),
        }
    }

    /// Test raw_usage with JSON chunks
    #[test]
    fn test_prepare_response_chunk_json_with_raw_usage() {
        let metadata = create_test_metadata();

        let raw_usage_entries = vec![RawUsageEntry {
            model_inference_id: Uuid::now_v7(),
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: json!({"total_tokens": 50}),
        }];

        let chunk = InferenceResultChunk::Json(JsonInferenceResultChunk {
            raw: Some(r#"{"key": "value"}"#.to_string()),
            thought_chunks: vec![],
            usage: Some(Usage {
                input_tokens: Some(30),
                output_tokens: Some(20),
            }),
            raw_usage: Some(raw_usage_entries),
            raw_response: None,
            raw_chunk: String::new(),
            provider_latency: Some(Duration::from_millis(100)),
            finish_reason: Some(FinishReason::Stop),
        });

        let result = prepare_response_chunk(&metadata, chunk);

        match result {
            InferenceResponseChunk::Json(j) => {
                assert!(j.usage.is_some(), "usage should be present");
                assert!(
                    j.raw_usage.is_some(),
                    "raw_usage should be passed through for JSON chunks"
                );
            }
            InferenceResponseChunk::Chat(_) => panic!("Expected Json chunk"),
        }
    }

    /// Test that metadata fields are correctly applied to response chunk
    #[test]
    fn test_prepare_response_chunk_applies_metadata() {
        let metadata = create_test_metadata();
        let expected_inference_id = metadata.inference_id;
        let expected_episode_id = metadata.episode_id;
        let expected_variant_name = metadata.variant_name.clone();

        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk {
            content: vec![ContentBlockChunk::Text(TextChunk {
                text: "Test".to_string(),
                id: "0".to_string(),
            })],
            usage: None,
            raw_usage: None,
            raw_response: None,
            finish_reason: None,
            raw_chunk: String::new(),
            provider_latency: Some(Duration::from_millis(50)),
        });

        let result = prepare_response_chunk(&metadata, chunk);

        match result {
            InferenceResponseChunk::Chat(c) => {
                assert_eq!(
                    c.inference_id, expected_inference_id,
                    "inference_id should be set from metadata"
                );
                assert_eq!(
                    c.episode_id, expected_episode_id,
                    "episode_id should be set from metadata"
                );
                assert_eq!(
                    c.variant_name, expected_variant_name,
                    "variant_name should be set from metadata"
                );
            }
            InferenceResponseChunk::Json(_) => panic!("Expected Chat chunk"),
        }
    }

    /// Test that create_stream emits raw_usage from previous model inferences in an artificial chunk
    #[tokio::test]
    async fn test_create_stream_emits_previous_inference_raw_usage() {
        // Create previous model inference with raw_usage
        let raw_usage_entries = vec![RawUsageEntry {
            model_inference_id: Uuid::now_v7(),
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: json!({"prompt_tokens": 100, "completion_tokens": 50}),
        }];

        let previous_inference = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            output: vec![ContentBlockOutput::Text(Text {
                text: "previous output".to_string(),
            })],
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            raw_request: "{}".to_string(),
            raw_response: "{}".to_string(),
            usage: Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            model_provider_name: "openai".into(),
            model_name: "gpt-4".into(),
            cached: false, // NOT cached, so raw_usage should be emitted
            finish_reason: Some(FinishReason::Stop),
            raw_usage: Some(raw_usage_entries.clone()),
            relay_raw_response: None,
        };

        let mut metadata = create_test_metadata();
        metadata.previous_model_inference_results = vec![previous_inference];
        metadata.include_raw_usage = true;

        // Create a simple function config
        let function = Arc::new(FunctionConfig::Chat(FunctionConfigChat::default()));
        let config = Arc::new(Config::default());
        let clickhouse = ClickHouseConnectionInfo::new_fake();
        let deferred_tasks = TaskTracker::new();

        // Create a simple stream with one chunk
        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk::default());
        let input_stream: Pin<Box<dyn Stream<Item = Result<InferenceResultChunk, Error>> + Send>> =
            Box::pin(futures::stream::iter(vec![Ok(chunk)]));
        let input_stream: InferenceResultStream = futures::StreamExt::peekable(input_stream);

        let mut stream = std::pin::pin!(create_stream(
            function,
            config,
            metadata,
            input_stream,
            clickhouse,
            deferred_tasks,
        ));

        // Collect all chunks
        let mut chunks = vec![];
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        // We should have at least 2 chunks:
        // 1. Artificial chunk with raw_usage from previous inferences
        // 2. Final chunk with aggregated usage
        assert!(
            chunks.len() >= 2,
            "should have at least 2 chunks (raw_usage + final), got {}",
            chunks.len()
        );

        // First chunk should have raw_usage from previous inference
        let first_chunk = chunks[0].as_ref().expect("first chunk should be Ok");
        match first_chunk {
            InferenceResponseChunk::Chat(c) => {
                let raw_usage = c
                    .raw_usage
                    .as_ref()
                    .expect("first chunk should have raw_usage from previous inferences");
                assert_eq!(
                    raw_usage.len(),
                    1,
                    "raw_usage should contain entries from previous inference"
                );
                assert_eq!(
                    raw_usage[0].provider_type, "openai",
                    "raw_usage should have correct provider_type"
                );
            }
            InferenceResponseChunk::Json(_) => panic!("Expected Chat chunk"),
        }
    }

    /// Test that cached previous inferences are filtered out from raw_usage
    #[tokio::test]
    async fn test_create_stream_filters_cached_from_raw_usage() {
        // Create a CACHED previous model inference with raw_usage (should be filtered out)
        let cached_raw_usage = vec![RawUsageEntry {
            model_inference_id: Uuid::now_v7(),
            provider_type: "cached_provider".to_string(),
            api_type: ApiType::ChatCompletions,
            data: json!({"should_not_appear": true}),
        }];

        let cached_inference = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            output: vec![ContentBlockOutput::Text(Text {
                text: "cached output".to_string(),
            })],
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            raw_request: "{}".to_string(),
            raw_response: "{}".to_string(),
            usage: Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            model_provider_name: "cached".into(),
            model_name: "cached-model".into(),
            cached: true, // CACHED - should be filtered out
            finish_reason: Some(FinishReason::Stop),
            raw_usage: Some(cached_raw_usage),
            relay_raw_response: None,
        };

        let mut metadata = create_test_metadata();
        metadata.previous_model_inference_results = vec![cached_inference];
        metadata.include_raw_usage = true;

        let function = Arc::new(FunctionConfig::Chat(FunctionConfigChat::default()));
        let config = Arc::new(Config::default());
        let clickhouse = ClickHouseConnectionInfo::new_fake();
        let deferred_tasks = TaskTracker::new();

        // Create a simple stream with one chunk
        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk::default());
        let input_stream: Pin<Box<dyn Stream<Item = Result<InferenceResultChunk, Error>> + Send>> =
            Box::pin(futures::stream::iter(vec![Ok(chunk)]));
        let input_stream: InferenceResultStream = futures::StreamExt::peekable(input_stream);

        let mut stream = std::pin::pin!(create_stream(
            function,
            config,
            metadata,
            input_stream,
            clickhouse,
            deferred_tasks,
        ));

        // Collect all chunks
        let mut chunks = vec![];
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        // Check that none of the chunks have raw_usage with "cached_provider"
        for chunk in &chunks {
            if let Ok(InferenceResponseChunk::Chat(c)) = chunk
                && let Some(raw_usage) = &c.raw_usage
            {
                for entry in raw_usage {
                    assert_ne!(
                        entry.provider_type, "cached_provider",
                        "cached inference raw_usage should be filtered out"
                    );
                }
            }
        }
    }

    /// Test that raw_usage is NOT emitted when include_raw_usage is false
    #[tokio::test]
    async fn test_create_stream_no_raw_usage_when_disabled() {
        let raw_usage_entries = vec![RawUsageEntry {
            model_inference_id: Uuid::now_v7(),
            provider_type: "openai".to_string(),
            api_type: ApiType::ChatCompletions,
            data: json!({"prompt_tokens": 100}),
        }];

        let previous_inference = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            output: vec![ContentBlockOutput::Text(Text {
                text: "output".to_string(),
            })],
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            raw_request: "{}".to_string(),
            raw_response: "{}".to_string(),
            usage: Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            model_provider_name: "openai".into(),
            model_name: "gpt-4".into(),
            cached: false,
            finish_reason: Some(FinishReason::Stop),
            raw_usage: Some(raw_usage_entries),
            relay_raw_response: None,
        };

        let mut metadata = create_test_metadata();
        metadata.previous_model_inference_results = vec![previous_inference];
        metadata.include_raw_usage = false; // DISABLED

        let function = Arc::new(FunctionConfig::Chat(FunctionConfigChat::default()));
        let config = Arc::new(Config::default());
        let clickhouse = ClickHouseConnectionInfo::new_fake();
        let deferred_tasks = TaskTracker::new();

        // Create a simple stream with one chunk
        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk::default());
        let input_stream: Pin<Box<dyn Stream<Item = Result<InferenceResultChunk, Error>> + Send>> =
            Box::pin(futures::stream::iter(vec![Ok(chunk)]));
        let input_stream: InferenceResultStream = futures::StreamExt::peekable(input_stream);

        let mut stream = std::pin::pin!(create_stream(
            function,
            config,
            metadata,
            input_stream,
            clickhouse,
            deferred_tasks,
        ));

        // Collect all chunks
        let mut chunks = vec![];
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        // With one input chunk and include_raw_usage=false, we should have:
        // - The input chunk (with metadata applied)
        // No artificial raw_usage chunk because include_raw_usage is false
        // Check that no chunks have raw_usage
        for chunk in &chunks {
            if let Ok(InferenceResponseChunk::Chat(c)) = chunk {
                assert!(
                    c.raw_usage.is_none(),
                    "raw_usage should not be present when include_raw_usage is false"
                );
            }
        }
    }
}
