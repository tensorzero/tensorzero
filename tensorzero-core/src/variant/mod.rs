use futures::StreamExt;
use itertools::izip;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::time::error::Elapsed;
use tracing::instrument;
use uuid::Uuid;

use crate::config::{PathWithContents, TimeoutsConfig};
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::InferenceIds;
use crate::endpoints::inference::{InferenceClients, InferenceModels, InferenceParams};
use crate::error::Error;
use crate::error::ErrorDetails;
#[cfg(feature = "pyo3")]
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::function::FunctionConfig;
use crate::inference::types::batch::StartBatchModelInferenceWithMetadata;
use crate::inference::types::chat_completion_inference_params::ChatCompletionInferenceParamsV2;
use crate::inference::types::extra_body::{FullExtraBodyConfig, UnfilteredInferenceExtraBody};
use crate::inference::types::extra_headers::{
    FullExtraHeadersConfig, UnfilteredInferenceExtraHeaders,
};
use crate::inference::types::resolved_input::LazyResolvedInput;
#[cfg(feature = "pyo3")]
use crate::inference::types::Role;
use crate::inference::types::{
    FunctionType, InferenceResultChunk, InferenceResultStream, ModelInferenceRequest,
    ModelInferenceResponseWithMetadata, RequestMessage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::model::StreamResponse;
use crate::model::StreamResponseAndMessages;
use crate::tool::{create_dynamic_implicit_tool_config, ToolCallConfig};
use crate::utils::retries::RetryConfig;
use crate::{inference::types::InferenceResult, model::ModelConfig};

pub mod best_of_n_sampling;
pub mod chain_of_thought;
pub mod chat_completion;
pub mod dicl;
pub mod dynamic;
pub mod mixture_of_n;

/// Holds a particular variant implementation, plus additional top-level configuration
/// that is applicable to any variant type.
#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct VariantInfo {
    pub inner: VariantConfig,
    pub timeouts: TimeoutsConfig,
}

impl VariantInfo {
    pub fn set_weight(&mut self, weight: Option<f64>) {
        self.inner.set_weight(weight);
    }
}

#[derive(ts_rs::TS, Debug, Serialize)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VariantConfig {
    ChatCompletion(chat_completion::ChatCompletionConfig),
    BestOfNSampling(best_of_n_sampling::BestOfNSamplingConfig),
    Dicl(dicl::DiclConfig),
    MixtureOfN(mixture_of_n::MixtureOfNConfig),
    ChainOfThought(chain_of_thought::ChainOfThoughtConfig),
}

#[cfg(feature = "pyo3")]
#[pyclass(name = "ChatCompletionConfig")]
pub struct ChatCompletionConfigPyClass {
    pub inner: Arc<VariantInfo>,
}

#[cfg(feature = "pyo3")]
#[pyclass(name = "BestOfNSamplingConfig")]
pub struct BestOfNSamplingConfigPyClass {
    pub inner: Arc<VariantInfo>,
}

#[cfg(feature = "pyo3")]
#[pyclass(name = "DICLConfig")]
pub struct DiclConfigPyClass {
    pub inner: Arc<VariantInfo>,
}

#[cfg(feature = "pyo3")]
#[pyclass(name = "MixtureOfNConfig")]
pub struct MixtureOfNConfigPyClass {
    pub inner: Arc<VariantInfo>,
}

#[cfg(feature = "pyo3")]
#[pyclass(name = "ChainOfThoughtConfig")]
pub struct ChainOfThoughtConfigPyClass {
    pub inner: Arc<VariantInfo>,
}

/// This type is used to determine how to enforce JSON mode for a given variant.
/// Variants represent JSON mode in a slightly more abstract sense than ModelInferenceRequests, as
/// we support coercing tool calls into JSON mode.
/// This is represented as a tool config in the
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum JsonMode {
    Off,
    On,
    Strict,
    #[serde(alias = "implicit_tool")] // Legacy name (stored in CH --> permanent alias)
    Tool,
}

/// Configuration that applies to the current inference request.
#[derive(Clone, Debug)]
pub struct InferenceConfig {
    pub tool_config: Option<Arc<ToolCallConfig>>,
    pub templates: Arc<TemplateConfig<'static>>,
    pub dynamic_output_schema: Option<Arc<DynamicJSONSchema>>,
    pub function_name: Arc<str>,
    pub variant_name: Arc<str>,
    pub ids: InferenceIds,
    pub extra_body: UnfilteredInferenceExtraBody,
    pub extra_headers: UnfilteredInferenceExtraHeaders,
    pub fetch_and_encode_input_files_before_inference: bool,
    /// Optional arbitrary data, only used when constructing the cache key.
    /// This is used by best_of_n/mixture_of_n to force different sub-variants
    /// to have different cache keys.
    /// This field should only ever be forwarded to `ModelInferenceRequest`
    pub extra_cache_key: Option<String>,
}

/// Maps to the subset of Config that applies to the current inference request.
#[derive(Clone, Debug)]
pub struct BatchInferenceConfig {
    pub tool_configs: Vec<Option<Arc<ToolCallConfig>>>,
    pub templates: Arc<TemplateConfig<'static>>,
    pub dynamic_output_schemas: Vec<Option<Arc<DynamicJSONSchema>>>,
    pub function_name: Arc<str>,
    pub variant_name: Arc<str>,
    pub fetch_and_encode_input_files_before_inference: bool,
}
impl BatchInferenceConfig {
    pub fn inference_configs(
        &self,
        episode_ids: &[Uuid],
        inference_ids: &[Uuid],
    ) -> Vec<InferenceConfig> {
        izip!(
            self.tool_configs.iter(),
            self.dynamic_output_schemas.iter(),
            episode_ids.iter(),
            inference_ids.iter()
        )
        .map(
            |(tool_config, dynamic_output_schema, episode_id, inference_id)| InferenceConfig {
                templates: Arc::clone(&self.templates),
                tool_config: tool_config.clone(),
                dynamic_output_schema: dynamic_output_schema.clone(),
                function_name: Arc::clone(&self.function_name),
                variant_name: Arc::clone(&self.variant_name),
                ids: InferenceIds {
                    inference_id: *inference_id,
                    episode_id: *episode_id,
                },
                fetch_and_encode_input_files_before_inference: self
                    .fetch_and_encode_input_files_before_inference,
                // Not yet supported for batch inference requests
                extra_body: Default::default(),
                extra_headers: Default::default(),
                extra_cache_key: None,
            },
        )
        .collect()
    }
}

#[derive(Debug)]
pub struct ModelUsedInfo {
    pub model_name: Arc<str>,
    pub model_provider_name: Arc<str>,
    pub raw_request: String,
    pub raw_response: Option<String>,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub inference_params: InferenceParams,
    pub cached: bool,
    // These responses will get added into the final inference result (after `collect_chunks` finishes)
    pub previous_model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
}

pub trait Variant {
    async fn infer(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error>;

    async fn infer_stream(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error>;

    #[expect(clippy::too_many_arguments)]
    async fn validate(
        &self,
        function: Arc<FunctionConfig>,
        models: &ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
        global_outbound_http_timeout: &chrono::Duration,
    ) -> Result<(), Error>;

    fn get_all_template_paths(&self) -> Vec<&PathWithContents>;
    fn get_all_explicit_template_names(&self) -> HashSet<String>;

    async fn start_batch_inference<'a>(
        &'a self,
        input: &[LazyResolvedInput],
        models: InferenceModels,
        function: &'a FunctionConfig,
        inference_configs: &'a [InferenceConfig],
        clients: InferenceClients,
        inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error>;
}

impl VariantConfig {
    pub fn weight(&self) -> Option<f64> {
        match self {
            VariantConfig::ChatCompletion(params) => params.weight(),
            VariantConfig::BestOfNSampling(params) => params.weight(),
            VariantConfig::Dicl(params) => params.weight(),
            VariantConfig::MixtureOfN(params) => params.weight(),
            VariantConfig::ChainOfThought(params) => params.inner.weight(),
        }
    }

    pub fn set_weight(&mut self, weight: Option<f64>) {
        match self {
            VariantConfig::ChatCompletion(params) => params.set_weight(weight),
            VariantConfig::BestOfNSampling(params) => params.set_weight(weight),
            VariantConfig::Dicl(params) => params.set_weight(weight),
            VariantConfig::MixtureOfN(params) => params.set_weight(weight),
            VariantConfig::ChainOfThought(params) => params.inner.set_weight(weight),
        }
    }
}

impl Variant for VariantInfo {
    #[instrument(
        fields(function_name = %inference_config.function_name, variant_name = %inference_config.variant_name, otel.name="variant_inference", stream=false),
        skip_all
    )]
    async fn infer(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let variant_name = inference_config.variant_name.clone();

        clients
            .otlp_config
            .mark_openinference_chain_span(&tracing::Span::current());

        let fut = async {
            match &self.inner {
                VariantConfig::ChatCompletion(params) => {
                    params
                        .infer(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }
                VariantConfig::BestOfNSampling(params) => {
                    params
                        .infer(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }

                VariantConfig::Dicl(params) => {
                    params
                        .infer(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }
                VariantConfig::MixtureOfN(params) => {
                    params
                        .infer(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }
                VariantConfig::ChainOfThought(params) => {
                    params
                        .infer(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }
            }
        };
        if let Some(timeout) = self.timeouts.non_streaming.total_ms {
            let timeout = tokio::time::Duration::from_millis(timeout);
            tokio::time::timeout(timeout, fut)
                .await
                // Convert the outer `Elapsed` error into a TensorZero error,
                // so that it can be handled by the `match response` block below
                .unwrap_or_else(|_: Elapsed| {
                    Err(Error::new(ErrorDetails::VariantTimeout {
                        variant_name: variant_name.to_string(),
                        timeout,
                        streaming: false,
                    }))
                })
        } else {
            fut.await
        }
    }

    #[instrument(
        fields(function_name = %inference_config.function_name, variant_name = %inference_config.variant_name, otel.name="variant_inference", stream=true),
        skip_all
    )]
    async fn infer_stream(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        clients
            .otlp_config
            .mark_openinference_chain_span(&tracing::Span::current());
        let variant_name = inference_config.variant_name.clone();
        let fut = async {
            match &self.inner {
                VariantConfig::ChatCompletion(params) => {
                    params
                        .infer_stream(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }
                VariantConfig::BestOfNSampling(params) => {
                    params
                        .infer_stream(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }
                VariantConfig::Dicl(params) => {
                    params
                        .infer_stream(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }
                VariantConfig::MixtureOfN(params) => {
                    params
                        .infer_stream(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }
                VariantConfig::ChainOfThought(params) => {
                    params
                        .infer_stream(
                            Arc::clone(&input),
                            models,
                            function,
                            inference_config,
                            clients,
                            inference_params,
                        )
                        .await
                }
            }
        };

        // This future includes a call to `peek_first_chunk`, so applying
        // `streaming_ttft_timeout` is correct.
        if let Some(timeout) = self.timeouts.streaming.ttft_ms {
            let timeout = tokio::time::Duration::from_millis(timeout);
            tokio::time::timeout(timeout, fut)
                .await
                .unwrap_or_else(|_: Elapsed| {
                    Err(Error::new(ErrorDetails::VariantTimeout {
                        variant_name: variant_name.to_string(),
                        timeout,
                        streaming: true,
                    }))
                })
        } else {
            fut.await
        }
    }

    #[instrument(skip_all, fields(variant_name = %inference_configs.first().map(|x| x.variant_name.as_ref()).unwrap_or("")))]
    async fn start_batch_inference<'a>(
        &'a self,
        inputs: &[LazyResolvedInput],
        models: InferenceModels,
        function: &'a FunctionConfig,
        inference_configs: &'a [InferenceConfig],
        clients: InferenceClients,
        inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        match &self.inner {
            VariantConfig::ChatCompletion(params) => {
                params
                    .start_batch_inference(
                        inputs,
                        models,
                        function,
                        inference_configs,
                        clients,
                        inference_params,
                    )
                    .await
            }
            _ => {
                Err(ErrorDetails::UnsupportedVariantForBatchInference { variant_name: None }.into())
            }
        }
    }

    #[instrument(skip_all, fields(variant_name = %variant_name))]
    async fn validate(
        &self,
        function: Arc<FunctionConfig>,
        models: &ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig<'_>,
        function_name: &str,
        variant_name: &str,
        global_outbound_http_timeout: &chrono::Duration,
    ) -> Result<(), Error> {
        self.timeouts.validate(global_outbound_http_timeout)?;
        match &self.inner {
            VariantConfig::ChatCompletion(params) => {
                params
                    .validate(
                        function,
                        models,
                        embedding_models,
                        templates,
                        function_name,
                        variant_name,
                        global_outbound_http_timeout,
                    )
                    .await
            }
            VariantConfig::BestOfNSampling(params) => {
                params
                    .validate(
                        function,
                        models,
                        embedding_models,
                        templates,
                        function_name,
                        variant_name,
                        global_outbound_http_timeout,
                    )
                    .await
            }
            VariantConfig::Dicl(params) => {
                params
                    .validate(
                        function,
                        models,
                        embedding_models,
                        templates,
                        function_name,
                        variant_name,
                        global_outbound_http_timeout,
                    )
                    .await
            }
            VariantConfig::MixtureOfN(params) => {
                params
                    .validate(
                        function,
                        models,
                        embedding_models,
                        templates,
                        function_name,
                        variant_name,
                        global_outbound_http_timeout,
                    )
                    .await
            }
            VariantConfig::ChainOfThought(params) => {
                params
                    .validate(
                        function,
                        models,
                        embedding_models,
                        templates,
                        function_name,
                        variant_name,
                        global_outbound_http_timeout,
                    )
                    .await
            }
        }
    }

    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        match &self.inner {
            VariantConfig::ChatCompletion(params) => params.get_all_template_paths(),
            VariantConfig::BestOfNSampling(params) => params.get_all_template_paths(),
            VariantConfig::Dicl(params) => params.get_all_template_paths(),
            VariantConfig::MixtureOfN(params) => params.get_all_template_paths(),
            VariantConfig::ChainOfThought(params) => params.get_all_template_paths(),
        }
    }

    fn get_all_explicit_template_names(&self) -> HashSet<String> {
        match &self.inner {
            VariantConfig::ChatCompletion(params) => params.get_all_explicit_template_names(),
            VariantConfig::BestOfNSampling(params) => params.get_all_explicit_template_names(),
            VariantConfig::Dicl(params) => params.get_all_explicit_template_names(),
            VariantConfig::MixtureOfN(params) => params.get_all_explicit_template_names(),
            VariantConfig::ChainOfThought(params) => params.get_all_explicit_template_names(),
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn prepare_model_inference_request<'request>(
    messages: Vec<RequestMessage>,
    system: Option<String>,
    function: &'request FunctionConfig,
    inference_config: &'request InferenceConfig,
    stream: bool,
    inference_params: &InferenceParams,
    base_json_mode: Option<JsonMode>,
    extra_body: FullExtraBodyConfig,
    extra_headers: FullExtraHeadersConfig,
) -> Result<ModelInferenceRequest<'request>, Error> {
    let json_mode = inference_params
        .chat_completion
        .json_mode
        .or(base_json_mode);

    Ok(match function {
        FunctionConfig::Chat(_) => {
            // For chat functions with `json_mode="tool"`, create a tool config based on the output schema
            let tool_config = match json_mode {
                Some(JsonMode::Tool) => {
                    // We know dynamic_output_schema exists because validation already checked this
                    match &inference_config.dynamic_output_schema {
                        Some(schema) => Some(Cow::Owned(create_dynamic_implicit_tool_config(
                            schema.value.clone(),
                        ))),
                        None => {
                            return Err(ErrorDetails::InvalidRequest {
                                message: "JSON mode `tool` requires `output_schema` to be provided at inference time.".to_string(),
                            }
                            .into());
                        }
                    }
                }
                _ => inference_config
                    .tool_config
                    .as_ref()
                    .map(|arc| Cow::Borrowed(arc.as_ref())),
            };

            ModelInferenceRequest {
                messages,
                system,
                inference_id: inference_config.ids.inference_id,
                tool_config,
                temperature: inference_params.chat_completion.temperature,
                top_p: inference_params.chat_completion.top_p,
                max_tokens: inference_params.chat_completion.max_tokens,
                presence_penalty: inference_params.chat_completion.presence_penalty,
                frequency_penalty: inference_params.chat_completion.frequency_penalty,
                seed: inference_params.chat_completion.seed,
                stream,
                // In chat mode, we fall back to 'JsonMode::Off' - json mode will only be enabled if
                // explicitly requested in `chat_completion` params.
                json_mode: json_mode.unwrap_or(JsonMode::Off).into(),
                function_type: FunctionType::Chat,
                output_schema: inference_config
                    .dynamic_output_schema
                    .as_ref()
                    .map(|v| &v.value),
                stop_sequences: inference_params
                    .chat_completion
                    .stop_sequences
                    .clone()
                    .map(Cow::Owned),
                extra_body,
                extra_headers,
                fetch_and_encode_input_files_before_inference: inference_config
                    .fetch_and_encode_input_files_before_inference,
                extra_cache_key: inference_config.extra_cache_key.clone(),
                inference_params_v2: ChatCompletionInferenceParamsV2 {
                    reasoning_effort: inference_params.chat_completion.reasoning_effort.clone(),
                    service_tier: inference_params.chat_completion.service_tier.clone(),
                    thinking_budget_tokens: inference_params.chat_completion.thinking_budget_tokens,
                    verbosity: inference_params.chat_completion.verbosity.clone(),
                },
            }
        }
        FunctionConfig::Json(json_config) => {
            let tool_config = match json_mode {
                Some(JsonMode::Tool) => match &inference_config.dynamic_output_schema {
                    Some(schema) => Some(Cow::Owned(create_dynamic_implicit_tool_config(
                        schema.value.clone(),
                    ))),
                    None => Some(Cow::Borrowed(&json_config.json_mode_tool_call_config)),
                },
                _ => None,
            };
            let output_schema = match &inference_config.dynamic_output_schema {
                Some(schema) => Some(&schema.value),
                None => Some(&json_config.output_schema.value),
            };
            ModelInferenceRequest {
                messages,
                system,
                tool_config,
                inference_id: inference_config.ids.inference_id,
                temperature: inference_params.chat_completion.temperature,
                top_p: inference_params.chat_completion.top_p,
                max_tokens: inference_params.chat_completion.max_tokens,
                presence_penalty: inference_params.chat_completion.presence_penalty,
                frequency_penalty: inference_params.chat_completion.frequency_penalty,
                seed: inference_params.chat_completion.seed,
                fetch_and_encode_input_files_before_inference: inference_config
                    .fetch_and_encode_input_files_before_inference,
                stream,
                // In json mode, we fall back to 'JsonMode::Strict' if it was unset in both
                // the `chat_completions` params and the variant config.
                json_mode: json_mode.unwrap_or(JsonMode::Strict).into(),
                function_type: FunctionType::Json,
                output_schema,
                stop_sequences: inference_params
                    .chat_completion
                    .stop_sequences
                    .clone()
                    .map(Cow::Owned),
                extra_body,
                extra_headers,
                extra_cache_key: inference_config.extra_cache_key.clone(),
                inference_params_v2: ChatCompletionInferenceParamsV2 {
                    reasoning_effort: inference_params.chat_completion.reasoning_effort.clone(),
                    service_tier: inference_params.chat_completion.service_tier.clone(),
                    thinking_budget_tokens: inference_params.chat_completion.thinking_budget_tokens,
                    verbosity: inference_params.chat_completion.verbosity.clone(),
                },
            }
        }
    })
}

/// Encapsulates all arguments for the `infer_model_request` function
struct InferModelRequestArgs<'a, 'request> {
    request: ModelInferenceRequest<'request>,
    model_name: Arc<str>,
    model_config: &'a ModelConfig,
    function: &'a FunctionConfig,
    inference_config: Arc<InferenceConfig>,
    clients: InferenceClients,
    inference_params: InferenceParams,
    retry_config: &'a RetryConfig,
}

/// Refactored `infer_model_request` function accepting a single struct argument
#[instrument(fields(model_name = %args.model_name), skip_all)]
async fn infer_model_request(
    args: InferModelRequestArgs<'_, '_>,
) -> Result<InferenceResult, Error> {
    let clients = args.clients.clone();
    let model_inference_response = args
        .retry_config
        .retry(|| async {
            args.model_config
                .infer(&args.request, &clients, &args.model_name)
                .await
        })
        .await?;

    let original_response = model_inference_response.raw_response.clone();
    let model_inference_result =
        ModelInferenceResponseWithMetadata::new(model_inference_response, args.model_name);
    let raw_content = model_inference_result.output.clone();
    let model_inference_results = vec![model_inference_result];

    args.function
        .prepare_response(
            args.inference_config.ids.inference_id,
            raw_content,
            model_inference_results,
            &args.inference_config,
            args.inference_params,
            Some(original_response),
        )
        .await
}

#[instrument(fields(model_name = %model_name), skip_all)]
// Note: this is due to a bug in Clippy 1.86 which runs on CI
// when we upgrate it we should be able to remove this attribute
#[allow(clippy::needless_lifetimes, clippy::allow_attributes)]
async fn infer_model_request_stream<'request>(
    request: ModelInferenceRequest<'request>,
    model_name: Arc<str>,
    model_config: &ModelConfig,
    function: &FunctionConfig,
    clients: InferenceClients,
    inference_params: InferenceParams,
    retry_config: RetryConfig,
) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
    let StreamResponseAndMessages {
        response:
            StreamResponse {
                stream,
                raw_request,
                model_provider_name,
                cached,
            },
        messages: input_messages,
    } = retry_config
        .retry(|| async {
            model_config
                .infer_stream(&request, &clients, &model_name)
                .await
        })
        .await?;
    let system = request.system.clone();
    let model_used_info = ModelUsedInfo {
        model_name,
        model_provider_name,
        raw_request,
        raw_response: None,
        inference_params,
        previous_model_inference_results: vec![],
        system,
        input_messages,
        cached,
    };
    let config_type = function.config_type();
    let stream =
        stream.map(move |chunk| chunk.map(|chunk| InferenceResultChunk::new(chunk, config_type)));
    Ok((StreamExt::peekable(Box::pin(stream)), model_used_info))
}

impl BatchInferenceConfig {
    pub fn new(
        templates: Arc<TemplateConfig<'static>>,
        tool_configs: Vec<Option<Arc<ToolCallConfig>>>,
        dynamic_output_schemas: Vec<Option<Arc<DynamicJSONSchema>>>,
        function_name: Arc<str>,
        variant_name: Arc<str>,
        fetch_and_encode_input_files_before_inference: bool,
    ) -> Self {
        Self {
            tool_configs,
            templates,
            dynamic_output_schemas,
            function_name,
            variant_name,
            fetch_and_encode_input_files_before_inference,
        }
    }
}

#[cfg(feature = "pyo3")]
impl ChatCompletionConfigPyClass {
    fn extract_chat_completion_config(
        variant_info: &VariantInfo,
    ) -> Result<&chat_completion::ChatCompletionConfig, PyErr> {
        match &variant_info.inner {
            VariantConfig::ChatCompletion(config) => Ok(config),
            _ => Err(PyValueError::new_err(format!(
                "Variant is not a chat completion variant: {IMPOSSIBLE_ERROR_MESSAGE}"
            ))),
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl ChatCompletionConfigPyClass {
    #[getter]
    fn get_system_template(&self) -> PyResult<Option<String>> {
        let config = Self::extract_chat_completion_config(&self.inner)?;
        Ok(config
            .templates()
            .get_implicit_system_template()
            .as_ref()
            .map(|t| t.template.contents.clone()))
    }

    #[getter]
    fn get_user_template(&self) -> PyResult<Option<String>> {
        let config = Self::extract_chat_completion_config(&self.inner)?;
        Ok(config
            .templates()
            .get_implicit_template(Role::User)
            .as_ref()
            .map(|t| t.template.contents.clone()))
    }

    #[getter]
    fn get_assistant_template(&self) -> PyResult<Option<String>> {
        let config = Self::extract_chat_completion_config(&self.inner)?;
        Ok(config
            .templates()
            .get_implicit_template(Role::Assistant)
            .as_ref()
            .map(|t| t.template.contents.clone()))
    }

    #[getter]
    fn get_model(&self) -> PyResult<String> {
        let config = Self::extract_chat_completion_config(&self.inner)?;
        Ok(config.model().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CacheEnabledMode, CacheOptions};
    use crate::config::SchemaData;
    use crate::db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo};
    use crate::endpoints::inference::{ChatCompletionInferenceParams, InferenceCredentials};
    use crate::error::ErrorDetails;
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::http::TensorzeroHttpClient;
    use crate::inference::types::{
        ContentBlockChunk, ModelInferenceRequestJsonMode, RequestMessage, Role, Usage,
    };
    use crate::jsonschema_util::StaticJSONSchema;
    use crate::minijinja_util::tests::get_test_template_config;
    use crate::model::{ModelProvider, ProviderConfig};
    use crate::providers::dummy::{
        DummyProvider, DUMMY_INFER_RESPONSE_CONTENT, DUMMY_JSON_RESPONSE_RAW,
        DUMMY_STREAMING_RESPONSE,
    };
    use crate::rate_limiting::ScopeInfo;
    use crate::tool::{ToolCallConfig, ToolChoice};

    use serde_json::json;
    use std::collections::HashMap;
    #[tokio::test]
    async fn test_prepare_model_inference_request() {
        // Setup common variables
        let templates = get_test_template_config().await;
        let stream = false;

        // Define a dummy tool config for testing
        let tool_config = ToolCallConfig::default();
        let tool_config_arc = Arc::new(tool_config.clone());

        // Create a sample inference config
        let inference_config = InferenceConfig {
            templates: Arc::new(templates.clone()),
            tool_config: Some(tool_config_arc),
            function_name: "test_function".into(),
            variant_name: "test_variant".into(),
            dynamic_output_schema: None,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };

        // Define common inference parameters
        let inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams {
                temperature: Some(0.7),
                max_tokens: Some(50),
                top_p: Some(0.9),
                presence_penalty: Some(0.0),
                frequency_penalty: Some(0.0),
                seed: Some(42),
                json_mode: None,
                stop_sequences: None,
                ..Default::default()
            },
        };

        // Prepare sample messages and system prompt
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["Hello, how are you?".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["I'm fine, thank you!".to_string().into()],
            },
        ];
        let system = Some("You are a helpful assistant.".to_string());

        // Test case 1: FunctionConfig::Chat with JsonMode::Off
        let function_config_chat = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        });
        let json_mode = JsonMode::Off;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_chat,
            &inference_config,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
            Default::default(),
        )
        .unwrap();

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.system, system);
        assert_eq!(result.tool_config, Some(Cow::Borrowed(&tool_config)));
        assert_eq!(result.temperature, Some(0.7));
        assert_eq!(result.top_p, Some(0.9));
        assert_eq!(result.max_tokens, Some(50));
        assert_eq!(result.presence_penalty, Some(0.0));
        assert_eq!(result.frequency_penalty, Some(0.0));
        assert_eq!(result.seed, Some(42));
        assert_eq!(result.stream, stream);
        assert_eq!(result.json_mode, ModelInferenceRequestJsonMode::Off);
        assert_eq!(result.function_type, FunctionType::Chat);
        assert_eq!(result.output_schema, None);

        // Test case 2: FunctionConfig::Json with JsonMode::On and static output schema
        let output_schema_value = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"],
        });
        let output_schema = StaticJSONSchema::from_value(output_schema_value.clone()).unwrap();
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema_value);

        let function_config_json = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema: output_schema.clone(),
            json_mode_tool_call_config: json_mode_tool_call_config.clone(),
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        });

        let json_mode = JsonMode::On;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_json,
            &inference_config,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
            Default::default(),
        )
        .unwrap();

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.system, system.clone());
        assert_eq!(result.tool_config, None);
        assert_eq!(result.temperature, Some(0.7));
        assert_eq!(result.max_tokens, Some(50));
        assert_eq!(result.seed, Some(42));
        assert_eq!(result.stream, stream);
        assert_eq!(result.json_mode, ModelInferenceRequestJsonMode::On);
        assert_eq!(result.function_type, FunctionType::Json);
        assert_eq!(result.output_schema, Some(&output_schema_value));

        // Test case 3: FunctionConfig::Json with JsonMode::ImplicitTool and dynamic output schema
        let dynamic_output_schema_value = json!({
            "type": "object",
            "properties": {
                "result": { "type": "string" }
            },
            "required": ["result"],
        });
        let dynamic_output_schema = DynamicJSONSchema::new(dynamic_output_schema_value.clone());
        let inference_config_dynamic = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: Arc::new(templates.clone()),
            tool_config: Some(Arc::new(tool_config)),
            function_name: "test_function".into(),
            variant_name: "test_variant".into(),
            dynamic_output_schema: Some(Arc::new(dynamic_output_schema)),
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let json_mode = JsonMode::Tool;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_json,
            &inference_config_dynamic,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
            Default::default(),
        )
        .unwrap();

        assert_eq!(
            result.tool_config,
            Some(Cow::Owned(create_dynamic_implicit_tool_config(
                dynamic_output_schema_value.clone(),
            )))
        );
        assert_eq!(result.output_schema, Some(&dynamic_output_schema_value));

        // Test case 4: FunctionConfig::Json with JsonMode::Strict
        let json_mode = JsonMode::Strict;

        let result = prepare_model_inference_request(
            messages.clone(),
            system.clone(),
            &function_config_json,
            &inference_config,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
            Default::default(),
        )
        .unwrap();

        assert_eq!(result.tool_config, None);
        assert_eq!(result.output_schema, Some(&output_schema_value));
        assert_eq!(result.json_mode, ModelInferenceRequestJsonMode::Strict);

        // Test case 5: FunctionConfig::Json with JsonMode::Off (should still set output_schema)
        let json_mode = JsonMode::Off;

        let result = prepare_model_inference_request(
            messages,
            system,
            &function_config_json,
            &inference_config,
            stream,
            &inference_params,
            Some(json_mode),
            Default::default(),
            Default::default(),
        )
        .unwrap();

        assert_eq!(result.tool_config, None);
        assert_eq!(result.output_schema, Some(&output_schema_value));
        assert_eq!(result.json_mode, ModelInferenceRequestJsonMode::Off);
    }

    #[tokio::test]
    async fn test_infer_model_request() {
        // Setup common variables
        let api_keys = InferenceCredentials::default();
        let client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            http_client: client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
        };
        let templates = Arc::new(get_test_template_config().await);
        let inference_params = InferenceParams::default();
        let inference_config = InferenceConfig {
            templates,
            tool_config: None,
            function_name: "test_function".into(),
            variant_name: "test_variant".into(),
            dynamic_output_schema: None,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };

        // Test case 1: Successful inference with ChatCompletionConfig and FunctionConfigChat
        let model_name = "dummy_chat_model";
        let function_config_chat = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        });

        let request_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];

        let model_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            tool_config: None,
            function_type: FunctionType::Chat,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            ..Default::default()
        };

        // Create a dummy provider config with the desired model name
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name.to_string(),
            ..Default::default()
        });

        // Create a model config with the dummy provider
        let model_config = ModelConfig {
            routing: vec![model_name.into()],
            providers: HashMap::from([(
                model_name.into(),
                ModelProvider {
                    name: model_name.into(),
                    config: dummy_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let retry_config = Box::leak(Box::new(RetryConfig::default()));

        // Create the arguments struct
        let args = InferModelRequestArgs {
            request: model_request.clone(),
            model_name: model_name.into(),
            model_config: &model_config,
            function: &function_config_chat,
            inference_config: Arc::new(inference_config.clone()),
            clients: clients.clone(),
            inference_params: inference_params.clone(),
            retry_config,
        };

        // Refactored function call
        let result = infer_model_request(args).await;

        let inference_result = result.unwrap();
        assert_eq!(
            inference_result.usage_considering_cached(),
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
        match inference_result {
            InferenceResult::Chat(chat_result) => {
                // The DummyProvider returns DUMMY_INFER_RESPONSE_CONTENT by default
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(chat_result.content, expected_content);
                assert_eq!(chat_result.model_inference_results.len(), 1);
                assert_eq!(
                    &*chat_result.model_inference_results[0].model_name,
                    model_name
                );
                // Need to recreate to make this ContentBlock rather than ContentBlockOutput
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(
                    &*chat_result.model_inference_results[0].output,
                    expected_content
                );
            }
            InferenceResult::Json(_) => panic!("Expected Chat inference result"),
        }

        // Test case 2: Successful inference with FunctionConfigJson
        let model_name_json = "json";
        let function_config_json = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema: StaticJSONSchema::from_value(json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"]
            }))
            .unwrap(),
            json_mode_tool_call_config: ToolCallConfig::default(),
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        });
        let output_schema = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"]
        });

        let model_request_json = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            seed: None,
            stream: false,
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            json_mode: ModelInferenceRequestJsonMode::On,
            output_schema: Some(&output_schema),
            tool_config: None,
            function_type: FunctionType::Json,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            ..Default::default()
        };

        // Create a dummy provider config with model_name "json" to trigger JSON response
        let dummy_provider_config_json = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name_json.to_string(),
            ..Default::default()
        });

        let model_config_json = ModelConfig {
            routing: vec![model_name_json.into()],
            providers: HashMap::from([(
                model_name_json.into(),
                ModelProvider {
                    name: model_name_json.into(),
                    config: dummy_provider_config_json,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };

        // Create the arguments struct
        let args = InferModelRequestArgs {
            request: model_request_json.clone(),
            model_name: model_name_json.into(),
            model_config: &model_config_json,
            function: &function_config_json,
            inference_config: Arc::new(inference_config.clone()),
            clients: clients.clone(),
            inference_params: inference_params.clone(),
            retry_config,
        };

        // Refactored function call
        let result = infer_model_request(args).await;

        let inference_result = result.unwrap();
        assert_eq!(
            inference_result.usage_considering_cached(),
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
        match inference_result {
            InferenceResult::Json(json_result) => {
                assert_eq!(
                    json_result.output.raw,
                    Some(DUMMY_JSON_RESPONSE_RAW.to_string())
                );
                assert_eq!(json_result.output.parsed, Some(json!({"answer": "Hello"})));
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(
                    &*json_result.model_inference_results[0].model_name,
                    model_name_json
                );
                assert_eq!(
                    json_result.model_inference_results[0].output,
                    vec![DUMMY_JSON_RESPONSE_RAW.to_string().into()]
                );
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference result"),
        }

        // Test case 3: Model inference failure
        let error_model_name = "error";
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: error_model_name.to_string(),
            ..Default::default()
        });

        let error_model_config = ModelConfig {
            routing: vec![error_model_name.into()],
            providers: HashMap::from([(
                error_model_name.into(),
                ModelProvider {
                    name: error_model_name.into(),
                    config: error_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };

        // Create the arguments struct
        let args = InferModelRequestArgs {
            request: model_request.clone(),
            model_name: error_model_name.into(),
            model_config: &error_model_config,
            function: &function_config_chat,
            inference_config: Arc::new(inference_config.clone()),
            clients: clients.clone(),
            inference_params: inference_params.clone(),
            retry_config,
        };

        // Refactored function call
        let result = infer_model_request(args).await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(matches!(
            error.get_details(),
            ErrorDetails::ModelProvidersExhausted { .. }
        ));
    }

    #[tokio::test]
    async fn test_infer_model_request_errors() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Setup common variables
        let api_keys = InferenceCredentials::default();
        let client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            http_client: client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
        };
        let templates = Arc::new(get_test_template_config().await);
        let inference_params = InferenceParams::default();
        let inference_config = InferenceConfig {
            templates,
            tool_config: None,
            function_name: "test_function".into(),
            variant_name: "test_variant".into(),
            dynamic_output_schema: None,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };

        let model_name = "dummy_chat_model";
        let error_model_name = "error";
        let function_config_chat = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        });

        let request_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];

        let model_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            tool_config: None,
            function_type: FunctionType::Chat,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            ..Default::default()
        };

        // Create a dummy provider config with the error model name
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: error_model_name.to_string(),
            ..Default::default()
        });

        // Create a dummy provider config with the good model name
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name.to_string(),
            ..Default::default()
        });

        // Create a model config with the dummy provider
        let model_config = ModelConfig {
            routing: vec![error_model_name.into(), model_name.into()],
            providers: HashMap::from([
                (
                    error_model_name.into(),
                    ModelProvider {
                        name: error_model_name.into(),
                        config: error_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                    },
                ),
                (
                    model_name.into(),
                    ModelProvider {
                        name: model_name.into(),
                        config: dummy_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                    },
                ),
            ]),
            timeouts: Default::default(),
        };
        let retry_config = Box::leak(Box::new(RetryConfig::default()));

        // Create the arguments struct
        let args = InferModelRequestArgs {
            request: model_request.clone(),
            model_name: model_name.into(),
            model_config: &model_config,
            function: &function_config_chat,
            inference_config: Arc::new(inference_config.clone()),
            clients: clients.clone(),
            inference_params: inference_params.clone(),
            retry_config,
        };

        // Refactored function call
        let result = infer_model_request(args).await;

        let inference_result = result.unwrap();
        assert_eq!(
            inference_result.usage_considering_cached(),
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
        match inference_result {
            InferenceResult::Chat(chat_result) => {
                // The DummyProvider returns DUMMY_INFER_RESPONSE_CONTENT by default
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(chat_result.content, expected_content);
                assert_eq!(chat_result.model_inference_results.len(), 1);
                assert_eq!(
                    &*chat_result.model_inference_results[0].model_name,
                    model_name
                );
                // Need to recreate to make this ContentBlock rather than ContentBlockOutput
                let expected_content = vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()];
                assert_eq!(
                    chat_result.model_inference_results[0].output,
                    expected_content
                );
            }
            InferenceResult::Json(_) => panic!("Expected Chat inference result"),
        }
        assert!(logs_contain(
            r#"ERROR infer_model_request{model_name=dummy_chat_model}:infer{model_name="dummy_chat_model" otel.name="model_inference" stream=false}:infer{provider_name="error"}:infer{provider_name="error" otel.name="model_provider_inference" stream=false}: tensorzero_core::error: Error from dummy client: Error sending request to Dummy provider for model 'error'."#
        ));
    }

    #[tokio::test]
    async fn test_infer_model_request_stream() {
        // Set up the HTTP client and ClickHouse connection info
        let client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let api_keys = InferenceCredentials::default();
        let clients = InferenceClients {
            http_client: client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
        };
        let retry_config = RetryConfig::default();
        // Create a dummy function config (chat completion)
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: crate::tool::ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        });

        // Create an input message
        let messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];
        let system = Some("You are a helpful assistant.".to_string());

        // Create a dummy model config with a provider
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            ..Default::default()
        });

        let model_config = Box::leak(Box::new(ModelConfig {
            routing: vec!["good_provider".into()],
            providers: HashMap::from([(
                "good_provider".into(),
                ModelProvider {
                    name: "good_provider".into(),
                    config: dummy_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        }));

        // Prepare the model inference request
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages,
            system,
            temperature: Some(0.7),
            max_tokens: Some(50),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            seed: None,
            tool_config: None,
            function_type: FunctionType::Chat,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            ..Default::default()
        };

        // Initialize inference parameters
        let inference_params = InferenceParams::default();

        // Call infer_model_request_stream
        let result = infer_model_request_stream(
            request,
            "good_model".into(),
            model_config,
            &function_config,
            clients.clone(),
            inference_params.clone(),
            retry_config,
        )
        .await;

        // Assert that the result is OK
        assert!(result.is_ok());

        // Unwrap the result
        let (mut stream, model_used_info) = result.unwrap();

        // Check the first chunk
        if let InferenceResultChunk::Chat(chat_chunk) = stream.next().await.unwrap().unwrap() {
            assert_eq!(chat_chunk.content.len(), 1);
            if let ContentBlockChunk::Text(text_chunk) = &chat_chunk.content[0] {
                assert_eq!(text_chunk.text, DUMMY_STREAMING_RESPONSE[0]);
            } else {
                panic!("Expected text chunk in first inference result chunk.");
            }
        } else {
            panic!("Expected chat inference result chunk.");
        }

        // Verify the model used information
        assert_eq!(&*model_used_info.model_name, "good_model");
        assert_eq!(&*model_used_info.model_provider_name, "good_provider");
        assert_eq!(model_used_info.inference_params, inference_params);

        // Iterate over the stream and collect the remaining chunks
        let mut received_text = String::new();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.expect("Stream chunk should be OK.");

            if let InferenceResultChunk::Chat(chat_chunk) = chunk {
                for content_block in chat_chunk.content {
                    if let ContentBlockChunk::Text(text_chunk) = content_block {
                        received_text.push_str(&text_chunk.text);
                    }
                }
            } else if let Some(usage) = chunk.usage() {
                // Verify the usage information
                assert_eq!(usage.input_tokens, Some(10));
                assert_eq!(
                    usage.output_tokens,
                    Some(DUMMY_STREAMING_RESPONSE.len() as u32)
                );
            } else {
                panic!("Unexpected inference result chunk.");
            }
        }

        // Combine the first chunk's text with the received text
        let mut full_response = DUMMY_STREAMING_RESPONSE[0].to_string();
        full_response.push_str(&received_text);

        // Verify the full response
        let expected_response: String = DUMMY_STREAMING_RESPONSE.iter().cloned().collect();
        assert_eq!(full_response, expected_response);
    }

    #[tokio::test]
    async fn test_infer_model_request_errors_stream() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Setup common variables
        let api_keys = InferenceCredentials::default();
        let client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            http_client: client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
        };
        let inference_params = InferenceParams::default();

        let model_name = "dummy_chat_model";
        let error_model_name = "error";
        let function_config_chat = Box::leak(Box::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        })));

        let request_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        }];

        let model_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: request_messages.clone(),
            system: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            output_schema: None,
            tool_config: None,
            function_type: FunctionType::Chat,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            ..Default::default()
        };

        // Create a dummy provider config with the error model name
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: error_model_name.to_string(),
            ..Default::default()
        });

        // Create a dummy provider config with the good model name
        let dummy_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: model_name.to_string(),
            ..Default::default()
        });

        // Create a model config with the dummy provider
        let model_config = Box::leak(Box::new(ModelConfig {
            routing: vec![error_model_name.into(), model_name.into()],
            providers: HashMap::from([
                (
                    error_model_name.into(),
                    ModelProvider {
                        name: error_model_name.into(),
                        config: error_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                    },
                ),
                (
                    model_name.into(),
                    ModelProvider {
                        name: model_name.into(),
                        config: dummy_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                    },
                ),
            ]),
            timeouts: Default::default(),
        }));
        let retry_config = RetryConfig::default();

        // Call infer_model_request_stream
        let result = infer_model_request_stream(
            model_request,
            model_name.into(),
            model_config,
            function_config_chat,
            clients.clone(),
            inference_params.clone(),
            retry_config,
        )
        .await;

        // Assert that the result is OK
        assert!(result.is_ok());

        // Unwrap the result
        let (mut stream, model_used_info) = result.unwrap();

        // Check the first chunk
        if let InferenceResultChunk::Chat(chat_chunk) = stream.next().await.unwrap().unwrap() {
            assert_eq!(chat_chunk.content.len(), 1);
            if let ContentBlockChunk::Text(text_chunk) = &chat_chunk.content[0] {
                assert_eq!(text_chunk.text, DUMMY_STREAMING_RESPONSE[0]);
            } else {
                panic!("Expected text chunk in first inference result chunk.");
            }
        } else {
            panic!("Expected chat inference result chunk.");
        }

        // Verify the model used information
        assert_eq!(&*model_used_info.model_name, model_name);
        assert_eq!(&*model_used_info.model_provider_name, model_name);
        assert_eq!(model_used_info.inference_params, inference_params);

        // Iterate over the stream and collect the remaining chunks
        let mut received_text = String::new();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.expect("Stream chunk should be OK.");

            if let InferenceResultChunk::Chat(chat_chunk) = chunk {
                for content_block in chat_chunk.content {
                    if let ContentBlockChunk::Text(text_chunk) = content_block {
                        received_text.push_str(&text_chunk.text);
                    }
                }
            } else if let Some(usage) = chunk.usage() {
                // Verify the usage information
                assert_eq!(usage.input_tokens, Some(10));
                assert_eq!(
                    usage.output_tokens,
                    Some(DUMMY_STREAMING_RESPONSE.len() as u32)
                );
            } else {
                panic!("Unexpected inference result chunk.");
            }
        }

        // Combine the first chunk's text with the received text
        let mut full_response = DUMMY_STREAMING_RESPONSE[0].to_string();
        full_response.push_str(&received_text);

        // Verify the full response
        let expected_response: String = DUMMY_STREAMING_RESPONSE.iter().cloned().collect();
        assert_eq!(full_response, expected_response);

        assert!(logs_contain(
            r#"ERROR infer_model_request_stream{model_name=dummy_chat_model}:infer_stream{model_name="dummy_chat_model" otel.name="model_inference" stream=true}:infer_stream{provider_name="error" otel.name="model_provider_inference" stream=true}: tensorzero_core::error: Error from dummy client: Error sending request to Dummy provider for model 'error'."#
        ));
    }
}
