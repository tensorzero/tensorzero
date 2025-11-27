use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use futures::future::{join_all, try_join_all};
use futures::StreamExt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time::timeout;

use crate::config::{ErrorContext, PathWithContents, SchemaData};
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::{InferenceClients, InferenceModels};
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::inference::types::extra_body::FullExtraBodyConfig;
use crate::inference::types::extra_headers::FullExtraHeadersConfig;
use crate::inference::types::resolved_input::LazyResolvedInput;
use crate::inference::types::{
    batch::StartBatchModelInferenceWithMetadata, ModelInferenceRequest, RequestMessage, Role,
    System,
};
use crate::inference::types::{
    ChatInferenceResultChunk, ContentBlockChatOutput, ContentBlockChunk, InferenceResultChunk,
    JsonInferenceResultChunk, RequestMessagesOrBatch, TextChunk, ThoughtChunk, Usage,
};
use crate::model::ModelTable;
use crate::tool::ToolCallChunk;
use crate::utils::unbounded_recursion_wrapper;
use crate::{
    endpoints::inference::InferenceParams,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    inference::types::{InferenceResult, InferenceResultStream},
    minijinja_util::TemplateConfig,
    variant::chat_completion::ChatCompletionConfig,
};

use crate::variant::chat_completion::UninitializedChatCompletionConfig;

use super::{
    infer_model_request, infer_model_request_stream, prepare_model_inference_request,
    InferModelRequestArgs, InferenceConfig, ModelUsedInfo, Variant,
};

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct MixtureOfNConfig {
    weight: Option<f64>,
    timeout_s: f64,
    candidates: Vec<String>,
    fuser: FuserConfig,
}

impl MixtureOfNConfig {
    pub fn weight(&self) -> Option<f64> {
        self.weight
    }

    pub fn set_weight(&mut self, weight: Option<f64>) {
        self.weight = weight;
    }

    pub fn timeout_s(&self) -> f64 {
        self.timeout_s
    }

    pub fn candidates(&self) -> &Vec<String> {
        &self.candidates
    }

    pub fn fuser(&self) -> &FuserConfig {
        &self.fuser
    }

    /// Converts this initialized config back to its uninitialized form.
    pub fn as_uninitialized(self) -> UninitializedMixtureOfNConfig {
        UninitializedMixtureOfNConfig {
            weight: self.weight,
            timeout_s: self.timeout_s,
            candidates: self.candidates,
            fuser: UninitializedFuserConfig {
                inner: self.fuser.inner.as_uninitialized(),
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[serde(deny_unknown_fields)]
#[ts(export)]
pub struct UninitializedMixtureOfNConfig {
    #[serde(default)]
    pub weight: Option<f64>,
    #[serde(default = "default_timeout")]
    pub timeout_s: f64,
    pub candidates: Vec<String>,
    pub fuser: UninitializedFuserConfig,
}

fn default_timeout() -> f64 {
    300.0
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct FuserConfig {
    #[serde(flatten)]
    pub inner: ChatCompletionConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[serde(deny_unknown_fields)]
#[ts(export)]
pub struct UninitializedFuserConfig {
    #[serde(flatten)]
    pub inner: UninitializedChatCompletionConfig,
}

impl UninitializedMixtureOfNConfig {
    pub fn load(
        self,
        schemas: &SchemaData,
        error_context: &ErrorContext,
    ) -> Result<MixtureOfNConfig, Error> {
        Ok(MixtureOfNConfig {
            weight: self.weight,
            timeout_s: self.timeout_s,
            candidates: self.candidates,
            fuser: FuserConfig {
                inner: self.fuser.inner.load(
                    schemas,
                    // Our stored fuser is a plain `UninitializedChatCompletionConfig`, so we need
                    // to explicitly add `fuser` to any error messages it produces.
                    &ErrorContext {
                        function_name: error_context.function_name.clone(),
                        variant_name: format!("{}.fuser", error_context.variant_name),
                    },
                )?,
            },
        })
    }
}

impl Variant for MixtureOfNConfig {
    // The compiler gives us 'cycle detected when looking up the hidden types stored across await points in a coroutine'
    // if we try to use 'async fn' here
    #[expect(refining_impl_trait, clippy::manual_async_fn)]
    fn infer(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        _inference_params: InferenceParams,
    ) -> impl Future<Output = Result<InferenceResult, Error>> + Send {
        async move {
            let candidate_inference_results = self
                .infer_candidates(
                    &input,
                    &models,
                    &function,
                    Arc::clone(&inference_config),
                    clients.clone(),
                )
                .await?;
            match self
            .fuse_candidates(
                &input,
                &function,
                &models.models,
                Arc::clone(&inference_config),
                clients,
                candidate_inference_results,
                false,
            )
            .await?
        {
            InferenceOrStreamResult::NonStream(inference_result) => {
                Ok(inference_result)
            },
            InferenceOrStreamResult::Stream(..) => {
                Err(ErrorDetails::InternalError { message: format!("MixtureOfNConfig.fuse_candidates returned a stream for a non-streaming request. {IMPOSSIBLE_ERROR_MESSAGE}") }.into())
                }
            }
        }
    }

    async fn infer_stream(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        // We infer the candidates in non-streaming mode, since we need to pass the full candidate results to the fuser
        let candidate_inference_results = self
            .infer_candidates(
                &input,
                &models,
                &function,
                Arc::clone(&inference_config),
                clients.clone(),
            )
            .await?;

        match self
            .fuse_candidates(
                &input,
                &function,
                &models.models,
                Arc::clone(&inference_config),
                clients,
                candidate_inference_results,
                true,
            )
            .await?
        {
            // We get a NonStream result if we don't have fuser result (either the fuser failed, or it wasn't run at all due to only one candidate existing)
            InferenceOrStreamResult::NonStream(inference_result) => {
                stream_inference_from_non_stream(inference_result, inference_params)
            }
            InferenceOrStreamResult::Stream(stream, model_used_info) => {
                Ok((stream, model_used_info))
            }
        }
    }

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
        // Validate each candidate variant
        for candidate in &self.candidates {
            let variant = function.variants().get(candidate).ok_or_else(|| {
                Error::new(ErrorDetails::UnknownCandidate {
                    name: candidate.to_string(),
                })
            })?;
            // Required by the compiler due to recursion (we call the top-level `validate`)
            Box::pin(variant.validate(
                Arc::clone(&function),
                models,
                embedding_models,
                templates,
                function_name,
                candidate,
                global_outbound_http_timeout,
            ))
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InvalidCandidate {
                    variant_name: variant_name.to_string(),
                    message: e.to_string(),
                })
            })?;
        }
        // Validate the fuser variant
        self.fuser
            .inner
            .validate(
                Arc::clone(&function),
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
                global_outbound_http_timeout,
            )
            .await?;
        Ok(())
    }

    // We do not return templates for the candidates, as they are required to be variants in the same function
    // and will therefore also have the same templates.
    // We only return templates for the fuser variant.
    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        self.fuser.inner.get_all_template_paths()
    }

    fn get_all_explicit_template_names(&self) -> HashSet<String> {
        self.fuser.inner.get_all_explicit_template_names()
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _input: &[LazyResolvedInput],
        _models: InferenceModels,
        _function: &'a FunctionConfig,
        _inference_configs: &'a [InferenceConfig],
        _clients: InferenceClients,
        _inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        Err(ErrorDetails::UnsupportedVariantForBatchInference { variant_name: None }.into())
    }
}

/// The result of calling attempts to fuse our candidates
enum InferenceOrStreamResult {
    /// If the user requested a non-streaming inference, then we'll call the fuser
    /// in non-streaming mode and return its result.
    /// If the fuser fails (or we only have a single candidate) when the user
    /// requested a streaming inference, we'll also return `InferenceOrStreamResult::NonStream`
    /// The `infer_stream` method is responsible for converting a non-streaming response
    /// into a 'fake' stream
    NonStream(InferenceResult),
    /// We only produce `InferenceOrStreamResult::Stream` if the user requested a streaming inference,
    /// and the fuser successfully starts a stream.
    Stream(InferenceResultStream, ModelUsedInfo),
}

/// Constructs an `infer_stream` response `(InferenceResultStream, ModelUsedInfo)`,
/// built from the information contained in the `InferenceResult`.
/// Each content block in the `InferenceResult` is converted into a chunk in the `InferenceResultStream`.
/// This is used by `best_of_n` and `mixture_of_n` when the user requests a stream response,
/// but our candidate/judge has a non-streaming response.
pub fn stream_inference_from_non_stream(
    inference_result: InferenceResult,
    inference_params: InferenceParams,
) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
    // Use the first model inference result to construct our top-level result (since we don't have a fuser/judge result)
    let model_inference_result = inference_result
        .model_inference_results()
        .first()
        .ok_or_else(|| {
            Error::new(ErrorDetails::Inference {
                message: format!(
                    "Expected one candidate but found none. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            })
        })?;
    // Copy the actual usage from the model inference result (without considering cached)
    // We set the 'cached' flag on the 'ModelUsedInfo, which will adjust the usage as needed when producing
    // the HTTP response stream.
    let usage = model_inference_result.usage;
    let model_used_info = ModelUsedInfo {
        model_name: model_inference_result.model_name.clone(),
        model_provider_name: model_inference_result.model_provider_name.clone(),
        raw_request: model_inference_result.raw_request.clone(),
        inference_params: inference_params.clone(),
        // Preserve the raw response from the candidate we chose (rather than attempting
        // to concatenate the raw_response from the chunks in our fake stream)
        raw_response: Some(model_inference_result.raw_response.clone()),
        // Preserve any other model inference results (we already processed the first one),
        // in case we're doing something like chained best-of-n/mixture-of-n variants.
        previous_model_inference_results: inference_result.model_inference_results()[1..].to_vec(),
        system: model_inference_result.system.clone(),
        input_messages: match model_inference_result.input_messages.clone() {
            RequestMessagesOrBatch::Message(input_messages) => input_messages,
            RequestMessagesOrBatch::BatchInput(_) => {
                return Err(Error::new(ErrorDetails::InternalError {
                    message: format!("Unexpected RequestMessagesOrBatch::BatchInput in model inference result. {IMPOSSIBLE_ERROR_MESSAGE}")
                        .to_string(),
                }));
            }
        },
        cached: model_inference_result.cached,
    };
    let stream = make_stream_from_non_stream(inference_result, Some(usage))?;
    Ok((stream, model_used_info))
}

fn make_stream_from_non_stream(
    inference_result: InferenceResult,
    usage: Option<Usage>,
) -> Result<InferenceResultStream, Error> {
    let mut id = 0;
    let chunk = match inference_result {
        InferenceResult::Chat(chat) => {
            let content_blocks = chat.content.into_iter().map(|content| {
                match content {
                ContentBlockChatOutput::Text(text) => {
                    let chunk = ContentBlockChunk::Text(TextChunk {
                        id: id.to_string(),
                        text: text.text,
                    });
                    id += 1;
                    Ok(chunk)
                }
                ContentBlockChatOutput::ToolCall(tool_call) => {
                    // Ues the tool call id as the chunk id, as this id needs to be
                    // passed back in when providing a tool call response.
                    let chunk = ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: tool_call.id.to_string(),
                        raw_name: Some(tool_call.raw_name),
                        raw_arguments: tool_call.raw_arguments,
                    });
                    Ok(chunk)
                }
                ContentBlockChatOutput::Thought(thought) => {
                    let chunk = ContentBlockChunk::Thought(ThoughtChunk {
                        id: id.to_string(),
                        text: thought.text,
                        signature: thought.signature,
                        summary_id: None,
                        summary_text: None,
                        provider_type: thought.provider_type,
                    });
                    id += 1;
                    Ok(chunk)
                }
                ContentBlockChatOutput::Unknown(_) => {
                    Err(ErrorDetails::Inference {
                        message: "MixtureOfNConfig variant does not support unknown content blocks in streaming mode".to_string(),
                    }
                    .into())
                }
            }
        }).collect::<Result<Vec<_>, Error>>()?;
            Ok(InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: content_blocks,
                created: chat.created,
                usage,
                latency: tokio::time::Duration::from_secs(0),
                raw_response: chat.original_response.unwrap_or_default(),
                finish_reason: chat.finish_reason,
            }))
        }
        InferenceResult::Json(json) => Ok(InferenceResultChunk::Json(JsonInferenceResultChunk {
            raw: json.output.raw,
            thought: None,
            created: json.created,
            usage,
            latency: tokio::time::Duration::from_secs(0),
            raw_response: json.original_response.unwrap_or_default(),
            finish_reason: json.finish_reason,
        })),
    };
    Ok(StreamExt::peekable(Box::pin(tokio_stream::once(chunk))))
}

impl fmt::Debug for InferenceOrStreamResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonStream(result) => write!(f, "NonStream({result:?})"),
            Self::Stream(_, model_used_info) => {
                write!(f, "Stream(..., {model_used_info:?})")
            }
        }
    }
}
impl MixtureOfNConfig {
    /// Infer each candidate variant concurrently and return the results.
    async fn infer_candidates(
        &self,
        input: &LazyResolvedInput,
        models: &InferenceModels,
        function: &Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
    ) -> Result<Vec<InferenceResult>, Error> {
        // Get all the variants we are going to infer
        let candidate_variants = self
            .candidates
            .iter()
            .enumerate()
            .map(|(i, candidate)| {
                let variant = function.variants().get(candidate).ok_or_else(|| {
                    Error::new(ErrorDetails::UnknownCandidate {
                        name: candidate.to_string(),
                    })
                })?;
                // Inject the candidate index into the cache key. This prevents us from using the same cache entry
                // for identical candidates, allowing users to evaluate the same candidate multiple times
                // to generate (potentially) different responses.
                // Note - we intentionally *only* inject the index, and not any other variant/model name
                // information. This means that multiple top-level 'best_of_n' variants will be able to share
                // the same cache entries. For example, consider two top-level best-of-n variants with
                // sub variants:
                // [A, B, A, C]
                // [A, B, C, D]
                //
                // The first two evaluations (A and B) will share the same cache key, since
                // the sub-variant will make the same request (and have the same injected index)
                // However, the 'A, C' and 'C, D' evaluations will all have distinct cache keys:
                // (A, 2), (C, 3), (C, 2), (D, 4)
                let config = InferenceConfig {
                    variant_name: Arc::from(candidate.as_str()),
                    extra_cache_key: Some(format!("candidate_{i}")),
                    ..inference_config.as_ref().clone()
                };
                Ok((candidate.to_string(), variant.clone(), Arc::new(config)))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Start the inference tasks (we keep the names around for logging)
        let mut inference_futures = Vec::new();
        for (candidate_name, candidate_variant, config) in candidate_variants {
            let input = Arc::new(input.clone());
            let models = models.clone();
            let function = Arc::clone(function);
            let clients = clients.clone();
            inference_futures.push((
                candidate_name.clone(),
                timeout(
                    tokio::time::Duration::from_secs_f64(self.timeout_s),
                    unbounded_recursion_wrapper(async move {
                        candidate_variant
                            .infer(
                                input,
                                models,
                                function,
                                config,
                                clients,
                                InferenceParams::default(),
                            )
                            .await
                    }),
                ),
            ));
        }

        // Wait for all the inference tasks to complete
        let inference_results: Vec<_> = join_all(
            inference_futures
                .into_iter()
                .map(|(candidate_name, future)| async move { (candidate_name, future.await) }),
        )
        .await;

        // Collect the successful results
        let mut successful_results = Vec::new();
        for (candidate_name, result) in inference_results {
            match result {
                Ok(inner_result) => {
                    if let Ok(res) = inner_result {
                        successful_results.push(res);
                    }
                }
                Err(_timeout_error) => {
                    // Map the Tokio timeout error to our own TimeoutError type
                    Error::new(ErrorDetails::InferenceTimeout {
                        variant_name: candidate_name.clone(),
                    });
                }
            }
        }

        Ok(successful_results)
    }

    /// Fuses the candidates using the fuser config.
    /// If the fuser fails to return a valid response,
    /// we randomly select one of the candidates.
    #[expect(clippy::too_many_arguments)]
    async fn fuse_candidates<'a>(
        &'a self,
        input: &LazyResolvedInput,
        function: &Arc<FunctionConfig>,
        models: &'a Arc<ModelTable>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        mut candidates: Vec<InferenceResult>,
        stream: bool,
    ) -> Result<InferenceOrStreamResult, Error> {
        if candidates.is_empty() {
            return Err(ErrorDetails::Inference {
                message: "No candidates to fuse in the mixture of n".to_string(),
            }
            .into());
        }
        if candidates.len() == 1 {
            return Ok(InferenceOrStreamResult::NonStream(candidates.pop().ok_or_else(|| Error::new(ErrorDetails::Inference {
                message: "Expected one candidate but found none. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
            }))?));
        }
        let mut candidates = candidates;

        let inference_result = if stream {
            inner_fuse_candidates_stream(
                &self.fuser,
                input,
                models,
                function,
                inference_config,
                clients,
                &candidates,
            )
            .await
            .map(|(stream, model_used_info)| {
                InferenceOrStreamResult::Stream(stream, model_used_info)
            })
        } else {
            inner_fuse_candidates(
                &self.fuser,
                input,
                models,
                function,
                inference_config,
                clients,
                &candidates,
            )
            .await
            .map(InferenceOrStreamResult::NonStream)
        };
        // As long as the fuser returns an inference result, we want to include it in the observability
        let mut inference_result = match inference_result {
            Ok(inf_result) => inf_result,
            Err(_) => {
                let random_index = rand::rng().random_range(0..candidates.len());
                if random_index >= candidates.len() {
                    return Err(Error::new(ErrorDetails::Inference {
                        message: "Failed to get random candidate (should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                    }));
                }
                // If the fuser fails, don't provide any 'original_response' to the user
                let mut candidate = candidates.swap_remove(random_index);
                candidate.set_original_response(None);
                InferenceOrStreamResult::NonStream(candidate)
            }
        };

        match &mut inference_result {
            InferenceOrStreamResult::NonStream(inference_result) => {
                for candidate in candidates {
                    inference_result
                        .mut_model_inference_results()
                        .extend(candidate.owned_model_inference_results());
                }
            }
            InferenceOrStreamResult::Stream(_stream, model_used_info) => {
                for candidate in candidates {
                    model_used_info
                        .previous_model_inference_results
                        .extend(candidate.owned_model_inference_results());
                }
            }
        }

        Ok(inference_result)
    }
}

/// Attempts to fuse the candidates for the mixture of n.
/// If this function returns an error, we will randomly select one
/// of the candidates in the outer function.
///
/// Here are the steps in the function:
///  * Prepare the request for the fuser variant.
///  * Infer the request using the model specified in the fuser config.
///  * Return the output of the fuser.
async fn inner_fuse_candidates<'a>(
    fuser: &'a FuserConfig,
    input: &LazyResolvedInput,
    models: &'a ModelTable,
    function: &Arc<FunctionConfig>,
    inference_config: Arc<InferenceConfig>,
    clients: InferenceClients,
    candidates: &[InferenceResult],
) -> Result<InferenceResult, Error> {
    let inference_config_clone = Arc::clone(&inference_config);
    let mut inference_params = InferenceParams::default();
    let (inference_request, included_indices) = fuser
        .prepare_fuser_request(
            input,
            function,
            &inference_config,
            candidates,
            &mut inference_params,
            false,
        )
        .await?;
    if included_indices.is_empty() {
        return Err(ErrorDetails::Inference {
            message: "No valid candidates available to prepare request.".to_string(),
        }
        .into());
    }
    let model_config = models.get(fuser.inner.model()).await?.ok_or_else(|| {
        Error::new(ErrorDetails::UnknownModel {
            name: fuser.inner.model().to_string(),
        })
    })?;
    let infer_model_request_args = InferModelRequestArgs {
        request: inference_request,
        model_name: fuser.inner.model().clone(),
        model_config: &model_config,
        function: function.as_ref(),
        inference_config: inference_config_clone,
        retry_config: fuser.inner.retries(),
        clients,
        inference_params: InferenceParams::default(),
    };
    let inference_result = infer_model_request(infer_model_request_args).await?;
    Ok(inference_result)
}

/// Attempts to fuse the candidates for the mixture of n.
/// If this function returns an error, we will randomly select one
/// of the candidates in the outer function.
///
/// Here are the steps in the function:
///  * Prepare the request for the fuser variant.
///  * Infer the request using the model specified in the fuser config.
///  * Return the output of the fuser.
async fn inner_fuse_candidates_stream<'a>(
    fuser: &'a FuserConfig,
    input: &LazyResolvedInput,
    models: &'a ModelTable,
    function: &Arc<FunctionConfig>,
    inference_config: Arc<InferenceConfig>,
    clients: InferenceClients,
    candidates: &[InferenceResult],
) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
    let mut params = InferenceParams::default();
    let (inference_request, included_indices) = fuser
        .prepare_fuser_request(
            input,
            function,
            &inference_config,
            candidates,
            &mut params,
            true,
        )
        .await?;
    if included_indices.is_empty() {
        return Err(ErrorDetails::Inference {
            message: "No valid candidates available to prepare request.".to_string(),
        }
        .into());
    }
    let model_config = models.get(fuser.inner.model()).await?.ok_or_else(|| {
        Error::new(ErrorDetails::UnknownModel {
            name: fuser.inner.model().to_string(),
        })
    })?;
    infer_model_request_stream(
        inference_request,
        fuser.inner.model().clone(),
        &model_config,
        function.as_ref(),
        clients,
        params,
        *fuser.inner.retries(),
    )
    .await
}

impl FuserConfig {
    /// Prepares the system message for the fuser variant.
    /// We use the system_template of the fuser variant to generate a system message as if we
    /// were using the fuser variant directly to solve the problem.
    /// Then, we template that system message into a broader set of instructions that includes
    /// information about what the fuser will be asked to do (choose a candidate).
    fn prepare_system_message(
        &self,
        templates: &TemplateConfig,
        system: Option<&System>,
        max_index: usize,
    ) -> Result<String, Error> {
        let inner_system_message = self.inner.prepare_system_message(templates, system)?;
        let template_context = match inner_system_message {
            Some(inner_system_message) => {
                json!({"inner_system_message": inner_system_message, "max_index": max_index})
            }
            None => json!({"max_index": max_index}),
        };
        templates.template_message("t0:mixture_of_n_fuser_system", &template_context)
    }

    /// Prepares the final candidate message for the fuser variant.
    ///
    /// This function constructs a `RequestMessage` that includes all valid candidate outputs
    /// by templating them into a predefined fuser template. It handles different types of
    /// inference results:
    ///
    /// - **Chat Inference**: Serializes the content blocks to a JSON string.
    /// - **JSON Inference**: Uses the raw JSON output if it contains correctly parsed data; otherwise,
    ///   skips the candidate.
    ///
    /// Additionally, it tracks and returns the indices of any candidates that were successfully included in the fuser message.
    ///
    /// # Parameters
    ///
    /// - `templates`: Reference to the `TemplateConfig` used for templating messages.
    /// - `candidates`: A vector of `InferenceResult` instances representing the candidate outputs.
    ///
    /// # Returns
    ///
    /// On success, returns a tuple containing:
    /// - `RequestMessage`: The templated message to be sent to the fuser.
    /// - `Vec<usize>`: A sorted vector of indices indicating which candidates were successfully included in the fuser message.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if any of the candidate outputs fail to serialize or if templating fails.
    fn prepare_candidate_message(
        templates: &TemplateConfig,
        candidates: &[InferenceResult],
    ) -> Result<(RequestMessage, Vec<usize>), Error> {
        let mut candidate_outputs = Vec::new();
        let mut included_indices = Vec::new();
        for (i, candidate) in candidates.iter().enumerate() {
            match candidate {
                InferenceResult::Chat(chat_result) => {
                    let serialized_content =
                        serde_json::to_string(&chat_result.content).map_err(|e| {
                            Error::new(ErrorDetails::Inference {
                                message: format!("Error converting chat result to string: {e}"),
                            })
                        })?;
                    candidate_outputs.push(serialized_content);
                    included_indices.push(i);
                }
                InferenceResult::Json(json_result) => {
                    if let (Some(raw), Some(_)) =
                        (&json_result.output.raw, &json_result.output.parsed)
                    {
                        candidate_outputs.push(raw.clone());
                        included_indices.push(i);
                    }
                }
            }
        }
        let template_context = json!({
            "candidates": candidate_outputs,
        });
        let message_text =
            templates.template_message("t0:mixture_of_n_fuser_candidates", &template_context)?;
        Ok((
            RequestMessage {
                role: Role::User,
                content: vec![message_text.into()],
            },
            included_indices,
        ))
    }

    /// Prepares the request for the fuser variant.
    /// We use the `prepare_system_message` and `prepare_candidate_message` functions to generate
    /// the system and candidate messages for the fuser, which take candidate selection into account.
    ///
    /// Additionally, this function returns the indices of candidates that were successfully included in the fuser message.
    ///
    /// # Returns
    ///
    /// On success, returns a tuple containing:
    /// - `ModelInferenceRequest`: The request prepared for the model inference.
    /// - `Vec<usize>`: A sorted vector of indices indicating which candidates were successfully included in the fuser message.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if any of the candidate outputs fail to serialize or if templating fails.
    async fn prepare_fuser_request<'a, 'request>(
        &'a self,
        input: &'request LazyResolvedInput,
        function: &'request Arc<FunctionConfig>,
        inference_config: &'request InferenceConfig,
        candidates: &[InferenceResult],
        inference_params: &mut InferenceParams,
        stream: bool,
    ) -> Result<(ModelInferenceRequest<'request>, Vec<usize>), Error>
    where
        'a: 'request,
    {
        // Do this before we prepare the system message so we can use the correct max index in the system message
        let (candidate_message, included_indices) =
            Self::prepare_candidate_message(&inference_config.templates, candidates)?;
        let max_index = included_indices.len().saturating_sub(1);
        let system = Some(self.prepare_system_message(
            &inference_config.templates,
            input.system.as_ref(),
            max_index,
        )?);
        let mut messages = try_join_all(input.messages.iter().map(|message| {
            self.inner
                .prepare_request_message(&inference_config.templates, message)
        }))
        .await?;

        messages.push(candidate_message);
        inference_params
            .chat_completion
            .backfill_with_variant_params(
                self.inner.temperature(),
                self.inner.max_tokens(),
                self.inner.seed(),
                self.inner.top_p(),
                self.inner.presence_penalty(),
                self.inner.frequency_penalty(),
                self.inner.stop_sequences().cloned(),
                self.inner.inference_params_v2.clone(),
            );

        if !inference_config.extra_body.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message:
                    "Inference-level `extra_body` is not yet supported for mixture_of_n variant"
                        .to_string(),
            }
            .into());
        }
        let extra_body = FullExtraBodyConfig {
            extra_body: self.inner.extra_body().cloned(),
            inference_extra_body: Default::default(),
        };
        let extra_headers = FullExtraHeadersConfig {
            variant_extra_headers: self.inner.extra_headers().cloned(),
            inference_extra_headers: inference_config
                .extra_headers
                .clone()
                .filter(&inference_config.variant_name),
        };
        let model_inference_request = prepare_model_inference_request(
            messages,
            system,
            function.as_ref(),
            inference_config,
            stream,
            inference_params,
            self.inner.json_mode().cloned(),
            extra_body,
            extra_headers,
        )?;
        Ok((model_inference_request, included_indices))
    }
}

#[cfg(test)]
mod tests {
    use crate::rate_limiting::ScopeInfo;
    use std::collections::HashMap;

    use tokio_stream::StreamExt;
    use uuid::Uuid;

    use crate::{
        cache::{CacheEnabledMode, CacheOptions},
        config::{provider_types::ProviderTypesConfig, SchemaData, UninitializedSchemas},
        db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
        endpoints::inference::{InferenceCredentials, InferenceIds},
        experimentation::ExperimentationConfig,
        function::{FunctionConfigChat, FunctionConfigJson},
        http::TensorzeroHttpClient,
        inference::types::{
            Arguments, ChatInferenceResult, FinishReason, InternalJsonInferenceOutput,
            JsonInferenceResult, Latency, ModelInferenceResponseWithMetadata, Text, Thought,
        },
        jsonschema_util::StaticJSONSchema,
        minijinja_util::tests::{
            get_system_filled_template, get_system_template, get_test_template_config,
            test_system_template_schema,
        },
        model::{ModelConfig, ModelProvider, ProviderConfig},
        model_table::ProviderTypeDefaultCredentials,
        providers::dummy::DummyProvider,
        tool::{InferenceResponseToolCall, ToolCallConfig, ToolChoice},
    };

    use super::*;

    #[tokio::test]
    async fn test_prepare_system_message() {
        let templates = get_test_template_config().await;

        // Test without templates, string message
        let fuser_config = FuserConfig {
            inner: UninitializedChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let input_message = System::Text("You are a helpful assistant.".to_string());
        let max_index = 2;
        let result =
            fuser_config.prepare_system_message(&templates, Some(&input_message), max_index);
        let prepared_message = result.unwrap();
        let expected_message = templates
            .template_message(
                "t0:mixture_of_n_fuser_system",
                &json!({"inner_system_message": "You are a helpful assistant.", "max_index": max_index}),
            )
            .unwrap();
        assert_eq!(prepared_message, expected_message);

        // Test without templates, object message
        let fuser_config = FuserConfig {
            inner: UninitializedChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let input_message = System::Template(Arguments(
            json!({"message": "You are a helpful assistant."})
                .as_object()
                .unwrap()
                .clone(),
        ));
        let max_index = 3;
        let result =
            fuser_config.prepare_system_message(&templates, Some(&input_message), max_index);
        assert!(result.is_err());
        let prepared_message = result.unwrap_err();
        assert_eq!(
            prepared_message,
            ErrorDetails::InvalidMessage {
                message: "System message content is a template but there is no variant template"
                    .to_string()
            }
            .into()
        );

        // Test without templates, no message
        let fuser_config = FuserConfig {
            inner: UninitializedChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let max_index = 5;
        let result = fuser_config.prepare_system_message(&templates, None, max_index);
        let expected_message = templates
            .template_message(
                "t0:mixture_of_n_fuser_system",
                &json!({"max_index": max_index}),
            )
            .unwrap();
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(prepared_message, expected_message);

        // Test with templates that need new info
        let system_template_name = "system";
        let system_template = get_system_template();

        let fuser_config = FuserConfig {
            inner: UninitializedChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                system_template: Some(system_template),
                user_template: None,
                assistant_template: None,
                input_wrappers: None,
                ..Default::default()
            }
            .load(
                &SchemaData::load(
                    None,
                    None,
                    Some(test_system_template_schema()),
                    UninitializedSchemas::default(),
                    "test",
                )
                .unwrap(),
                &ErrorContext {
                    function_name: "test".to_string(),
                    variant_name: "test".to_string(),
                },
            )
            .unwrap(),
        };

        let max_index = 6;
        let input_message = System::Template(Arguments(
            serde_json::json!({"assistant_name": "ChatGPT"})
                .as_object()
                .unwrap()
                .clone(),
        ));
        let prepared_message = fuser_config
            .prepare_system_message(&templates, Some(&input_message), max_index)
            .unwrap();
        let inner_system_message = templates
            .template_message(
                system_template_name,
                &json!({"assistant_name": "ChatGPT", "max_index": max_index}),
            )
            .unwrap();
        let expected_message = templates
            .template_message(
                "t0:mixture_of_n_fuser_system",
                &json!({"inner_system_message": inner_system_message, "max_index": max_index}),
            )
            .unwrap();
        assert_eq!(prepared_message, expected_message);

        // Test with template that is complete as is (string)
        let system_template_name = "system_filled";
        let system_template = get_system_filled_template();

        let fuser_config = FuserConfig {
            inner: UninitializedChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                system_template: Some(system_template),
                user_template: None,
                assistant_template: None,
                input_wrappers: None,
                ..Default::default()
            }
            .load(
                &SchemaData::load(
                    None,
                    None,
                    Some(test_system_template_schema()),
                    UninitializedSchemas::default(),
                    "test",
                )
                .unwrap(),
                &ErrorContext {
                    function_name: "test".to_string(),
                    variant_name: "test".to_string(),
                },
            )
            .unwrap(),
        };

        let max_index = 10;
        let result = fuser_config.prepare_system_message(&templates, None, max_index);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        let inner_system_message = templates
            .template_message(system_template_name, &json!({}))
            .unwrap();
        let expected_message = templates
            .template_message(
                "t0:mixture_of_n_fuser_system",
                &json!({"inner_system_message": inner_system_message, "max_index": max_index}),
            )
            .unwrap();
        assert_eq!(prepared_message, expected_message);
    }

    #[tokio::test]
    async fn test_prepare_candidate_message() {
        let templates = get_test_template_config().await;

        // Prepare some candidate InferenceResults
        let model_inference_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            output: vec!["Candidate answer 1".to_string().into()],
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Example response\"}".to_string(),
            usage: Usage {
                input_tokens: Some(50),
                output_tokens: Some(100),
            },
            latency: Latency::NonStreaming {
                response_time: std::time::Duration::from_millis(500),
            },
            model_provider_name: "ExampleProvider".into(),
            model_name: "ExampleModel".into(),
            finish_reason: Some(FinishReason::Stop),
            cached: false,
        };

        let candidate1 = InferenceResult::Chat(
            ChatInferenceResult::new(
                Uuid::now_v7(),
                vec!["Candidate answer 1".to_string().into()],
                vec![model_inference_response],
                None,
                InferenceParams::default(),
                None,
                None,
            )
            .await,
        );

        let model_inference_response2 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 201u64,
            output: vec!["Candidate answer 2".to_string().into()],
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            raw_request: "{\"prompt\": \"Example prompt 2\"}".to_string(),
            raw_response: "{\"response\": \"Example response 2\"}".to_string(),
            usage: Usage {
                input_tokens: Some(15),
                output_tokens: Some(25),
            },
            latency: Latency::NonStreaming {
                response_time: std::time::Duration::from_millis(550),
            },
            model_provider_name: "ExampleProvider2".into(),
            model_name: "ExampleModel2".into(),
            finish_reason: Some(FinishReason::Stop),
            cached: false,
        };

        let candidate2 = InferenceResult::Chat(
            ChatInferenceResult::new(
                Uuid::now_v7(),
                vec!["Candidate answer 2".to_string().into()],
                vec![model_inference_response2],
                None,
                InferenceParams::default(),
                None,
                None,
            )
            .await,
        );

        let candidates = vec![candidate1, candidate2];

        // Call prepare_candidate_message
        let result = FuserConfig::prepare_candidate_message(&templates, &candidates);
        assert!(result.is_ok());
        let (request_message, included_indices) = result.unwrap();
        assert_eq!(included_indices, vec![0, 1]);

        let expected_message_text = "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n[{\"type\":\"text\",\"text\":\"Candidate answer 1\"}]\n------\n1:\n[{\"type\":\"text\",\"text\":\"Candidate answer 2\"}]\n------".to_string();
        // Now check that the request_message has the expected role and content
        assert_eq!(request_message.role, Role::User);
        assert_eq!(request_message.content, vec![expected_message_text.into()]);
    }

    #[tokio::test]
    async fn test_prepare_candidate_message_json() {
        let templates = get_test_template_config().await;

        // Prepare some candidate InferenceResults - some valid, some malformed
        let model_inference_response_valid = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            output: vec!["{\"response\": \"Valid JSON response\"}".to_string().into()],
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Valid JSON response\"}".to_string(),
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            },
            latency: Latency::NonStreaming {
                response_time: std::time::Duration::from_millis(500),
            },
            model_provider_name: "ExampleProvider".into(),
            model_name: "ExampleModel".into(),
            finish_reason: Some(FinishReason::Stop),
            cached: false,
        };

        let candidate1 = InferenceResult::Json(JsonInferenceResult::new(
            Uuid::now_v7(),
            Some("{\"response\": \"Valid JSON response\"}".to_string()),
            Some(json!({"response": "Valid JSON response"})),
            Some(0),
            vec![],
            vec![model_inference_response_valid],
            json!({"type": "object", "properties": {"response": {"type": "string"}}}),
            InferenceParams::default(),
            None,
        ));

        let model_inference_response_malformed = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 201u64,
            output: vec!["{\"response\": \"Malformed JSON response\""
                .to_string()
                .into()], // missing closing brace
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            raw_request: "{\"prompt\": \"Example prompt 2\"}".to_string(),
            raw_response: "{\"response\": \"Malformed JSON response\"".to_string(), // malformed
            usage: Usage {
                input_tokens: Some(15),
                output_tokens: Some(25),
            },
            latency: Latency::NonStreaming {
                response_time: std::time::Duration::from_millis(550),
            },
            model_provider_name: "ExampleProvider2".into(),
            model_name: "ExampleModel2".into(),
            finish_reason: Some(FinishReason::Stop),
            cached: false,
        };

        let candidate2 = InferenceResult::Json(JsonInferenceResult::new(
            Uuid::now_v7(),
            Some("{\"oops: \"Malformed JSON response\"".to_string()),
            None, // malformed
            Some(0),
            vec![],
            vec![model_inference_response_malformed],
            json!({"type": "object", "properties": {"response": {"type": "string"}}}),
            InferenceParams::default(),
            None,
        ));

        let candidates = vec![candidate1, candidate2];

        // Call prepare_candidate_message
        let result = FuserConfig::prepare_candidate_message(&templates, &candidates);
        assert!(result.is_ok());
        let (request_message, included_indices) = result.unwrap();

        // Expect included_indices to contain index 0
        assert_eq!(included_indices, vec![0]);

        let expected_message_text = "Here are the candidate answers (with the index and a row of ------ separating):\n0:\n{\"response\": \"Valid JSON response\"}\n------".to_string();

        // Check that the request_message has the expected role and content
        assert_eq!(request_message.role, Role::User);
        assert_eq!(request_message.content, vec![expected_message_text.into()]);
    }

    #[tokio::test]
    async fn test_fuse_candidates() {
        // Set up fuser with a provider that returns a valid answer_choice
        let fuser_config = FuserConfig {
            inner: UninitializedChatCompletionConfig {
                model: "json".into(),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let mixture_of_n_variant = MixtureOfNConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec![],
            fuser: fuser_config,
        };

        let templates = get_test_template_config().await;
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema: StaticJSONSchema::from_value(json!({})).unwrap(),
            json_mode_tool_call_config: ToolCallConfig::default(),
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));
        // Prepare some candidate InferenceResults
        let model_inference_response0 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            output: vec!["Candidate answer 0".to_string().into()],
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Example response\"}".to_string(),
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            },
            latency: Latency::NonStreaming {
                response_time: std::time::Duration::from_millis(500),
            },
            model_provider_name: "ExampleProvider".into(),
            model_name: "ExampleModel".into(),
            finish_reason: Some(FinishReason::Stop),
            cached: false,
        };
        let inference_id0 = Uuid::now_v7();
        let candidate0 = InferenceResult::Chat(
            ChatInferenceResult::new(
                inference_id0,
                vec!["Candidate answer 0".to_string().into()],
                vec![model_inference_response0],
                None,
                InferenceParams::default(),
                None,
                None,
            )
            .await,
        );

        let model_inference_response1 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 201u64,
            output: vec!["Candidate answer 1".to_string().into()],
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            raw_request: "{\"prompt\": \"Example prompt 1\"}".to_string(),
            raw_response: "{\"response\": \"Example response 1\"}".to_string(),
            usage: Usage {
                input_tokens: Some(15),
                output_tokens: Some(25),
            },
            latency: Latency::NonStreaming {
                response_time: std::time::Duration::from_millis(550),
            },
            model_provider_name: "ExampleProvider1".into(),
            model_name: "ExampleModel1".into(),
            finish_reason: Some(FinishReason::Stop),
            cached: false,
        };
        let inference_id1 = Uuid::now_v7();
        let candidate1 = InferenceResult::Chat(
            ChatInferenceResult::new(
                inference_id1,
                vec!["Candidate answer 1".to_string().into()],
                vec![model_inference_response1],
                None,
                InferenceParams::default(),
                None,
                None,
            )
            .await,
        );
        let candidates = vec![candidate0, candidate1];
        let provider_types = ProviderTypesConfig::default();
        let models = ModelTable::new(
            HashMap::from([(
                "json".into(),
                ModelConfig {
                    routing: vec!["json".into()],
                    providers: HashMap::from([(
                        "json".into(),
                        ModelProvider {
                            name: "json".into(),
                            config: ProviderConfig::Dummy(DummyProvider {
                                model_name: "json".into(),
                                ..Default::default()
                            }),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            timeouts: Default::default(),
                            discard_unknown_chunks: false,
                        },
                    )]),
                    timeouts: Default::default(),
                },
            )]),
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            chrono::Duration::seconds(120),
        )
        .expect("Failed to create model table");
        let client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let api_keys = InferenceCredentials::default();
        let inference_clients = InferenceClients {
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
        let input = LazyResolvedInput {
            system: None,
            messages: vec![],
        };
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: Arc::new(templates),
            tool_config: None,
            dynamic_output_schema: None,
            function_name: "".into(),
            variant_name: "".into(),
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };

        let InferenceOrStreamResult::NonStream(fused) = mixture_of_n_variant
            .fuse_candidates(
                &input,
                &json_function_config,
                &Arc::new(models),
                Arc::new(inference_config.clone()),
                inference_clients.clone(),
                candidates.clone(),
                false,
            )
            .await
            .expect("Failed to select best candidate")
        else {
            panic!("Expected a non-stream result");
        };

        let expected_usage = Usage {
            input_tokens: Some(35),
            output_tokens: Some(46),
        };
        let expected_content = InternalJsonInferenceOutput {
            raw: Some("{\"answer\":\"Hello\"}".to_string()),
            parsed: Some(json!({"answer": "Hello"})),
            auxiliary_content: vec![],
            json_block_index: Some(0),
        };
        assert_eq!(fused.usage_considering_cached(), expected_usage);
        match fused {
            InferenceResult::Json(fused) => {
                assert_eq!(fused.output, expected_content);
                assert_eq!(fused.model_inference_results.len(), 3);
            }
            InferenceResult::Chat(_) => {
                panic!("Expected a Json inference result");
            }
        }
        // Set up fuser with a provider that fails
        let fuser_config = FuserConfig {
            inner: UninitializedChatCompletionConfig {
                model: "error".into(),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let mixture_of_n_variant = MixtureOfNConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec![],
            fuser: fuser_config,
        };

        let models = {
            let mut map = HashMap::new();
            map.insert(
                "error".into(),
                ModelConfig {
                    routing: vec!["error".into()],
                    providers: HashMap::from([(
                        "error".into(),
                        ModelProvider {
                            name: "error".into(),
                            config: ProviderConfig::Dummy(DummyProvider {
                                model_name: "error".into(),
                                ..Default::default()
                            }),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            timeouts: Default::default(),
                            discard_unknown_chunks: false,
                        },
                    )]),
                    timeouts: Default::default(),
                },
            );
            let provider_types = ProviderTypesConfig::default();
            ModelTable::new(
                map,
                ProviderTypeDefaultCredentials::new(&provider_types).into(),
                chrono::Duration::seconds(120),
            )
            .expect("Failed to create model table")
        };
        let input = LazyResolvedInput {
            system: None,
            messages: vec![],
        };

        let InferenceOrStreamResult::NonStream(result) = mixture_of_n_variant
            .fuse_candidates(
                &input,
                &json_function_config,
                &Arc::new(models),
                Arc::new(inference_config.clone()),
                inference_clients.clone(),
                candidates.clone(),
                false,
            )
            .await
            .unwrap()
        else {
            panic!("Expected a non-stream result");
        };

        // Expect an error and a random candidate to be selected
        let choice = result;
        // We know that the model will fail, so there should only be two results
        match choice {
            InferenceResult::Chat(chat_choice) => {
                assert_eq!(chat_choice.model_inference_results.len(), 2);
            }
            InferenceResult::Json(_) => {
                panic!("Expected a Chat inference result");
            }
        }
        // Depending on implementation, you might check which candidate was selected

        // Set up fuser with a provider that returns invalid JSON
        let fuser_config = FuserConfig {
            inner: UninitializedChatCompletionConfig {
                model: "regular".into(),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let mixture_of_n_variant = MixtureOfNConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec![],
            fuser: fuser_config,
        };

        let models = {
            let mut map = HashMap::new();
            map.insert(
                "regular".into(),
                ModelConfig {
                    routing: vec!["regular".into()],
                    providers: HashMap::from([(
                        "regular".into(),
                        ModelProvider {
                            name: "regular".into(),
                            config: ProviderConfig::Dummy(DummyProvider {
                                model_name: "regular".into(),
                                ..Default::default()
                            }),
                            extra_body: Default::default(),
                            extra_headers: Default::default(),
                            timeouts: Default::default(),
                            discard_unknown_chunks: false,
                        },
                    )]),
                    timeouts: Default::default(),
                },
            );
            let provider_types = ProviderTypesConfig::default();
            ModelTable::new(
                map,
                ProviderTypeDefaultCredentials::new(&provider_types).into(),
                chrono::Duration::seconds(120),
            )
            .expect("Failed to create model table")
        };
        let input = LazyResolvedInput {
            system: None,
            messages: vec![],
        };
        let chat_function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));

        let models_arc = Arc::new(models);
        let InferenceOrStreamResult::NonStream(result) = mixture_of_n_variant
            .fuse_candidates(
                &input,
                &chat_function_config,
                &models_arc,
                Arc::new(inference_config.clone()),
                inference_clients.clone(),
                candidates.clone(),
                false,
            )
            .await
            .unwrap()
        else {
            panic!("Expected a non-stream result");
        };

        let choice = result;
        match choice {
            InferenceResult::Chat(chat_choice) => {
                // Should return 3 results since model has been called 3 times
                // But, it's a random choice, so we can't assert on the specific index
                assert!(chat_choice.model_inference_results.len() == 3);
            }
            InferenceResult::Json(_) => {
                panic!("Expected a Chat inference result");
            }
        }
        // Test case: No answer choices (should return an error)
        let empty_candidates = vec![];
        let result = mixture_of_n_variant
            .fuse_candidates(
                &input,
                &json_function_config,
                &models_arc,
                Arc::new(inference_config.clone()),
                inference_clients.clone(),
                empty_candidates.clone(),
                false,
            )
            .await;
        let err = result.unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::Inference {
                message: "No candidates to fuse in the mixture of n".to_string()
            }
            .into()
        );
    }

    #[tokio::test]
    async fn test_make_stream_from_non_stream_chat() {
        let stream = make_stream_from_non_stream(
            InferenceResult::Chat(ChatInferenceResult {
                inference_id: Uuid::now_v7(),
                content: vec![
                    ContentBlockChatOutput::Text(Text {
                        text: "First text message".to_string(),
                    }),
                    ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
                        id: "123".into(),
                        name: Some("first_tool".into()),
                        raw_name: "first_tool".into(),
                        arguments: Some(serde_json::json!({
                            "my": "first_arg"
                        })),
                        raw_arguments: r#"{"my"  :  "first_arg"}"#.to_string(),
                    }),
                    ContentBlockChatOutput::Thought(Thought {
                        text: Some("My first thought".into()),
                        signature: Some("my_first_signature".into()),
                        summary: None,
                        provider_type: Some("my_first_provider_type".into()),
                    }),
                    ContentBlockChatOutput::Thought(Thought {
                        text: Some("My second thought".into()),
                        signature: Some("my_second_signature".into()),
                        summary: None,
                        provider_type: None,
                    }),
                    ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
                        id: "456".into(),
                        name: Some("second_tool".into()),
                        raw_name: "second_tool".into(),
                        arguments: Some(serde_json::json!({
                            "my": "second_arg"
                        })),
                        raw_arguments: r#"{"my"  :  "second_arg"}"#.to_string(),
                    }),
                    ContentBlockChatOutput::Text(Text {
                        text: "Second text message".to_string(),
                    }),
                ],
                created: 123456,
                model_inference_results: vec![],
                inference_params: InferenceParams::default(),
                original_response: Some("My raw response".to_string()),
                finish_reason: Some(FinishReason::Length),
            }),
            Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            }),
        )
        .unwrap();

        let stream_chunks = stream.collect::<Vec<_>>().await;
        assert_eq!(
            stream_chunks,
            [Ok(InferenceResultChunk::Chat(ChatInferenceResultChunk {
                content: vec![
                    ContentBlockChunk::Text(TextChunk {
                        id: "0".into(),
                        text: "First text message".to_string(),
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "123".into(),
                        raw_name: Some("first_tool".into()),
                        raw_arguments: r#"{"my"  :  "first_arg"}"#.to_string(),
                    }),
                    ContentBlockChunk::Thought(ThoughtChunk {
                        id: "1".into(),
                        text: Some("My first thought".into()),
                        signature: Some("my_first_signature".into()),
                        summary_id: None,
                        summary_text: None,
                        provider_type: Some("my_first_provider_type".into()),
                    }),
                    ContentBlockChunk::Thought(ThoughtChunk {
                        id: "2".into(),
                        text: Some("My second thought".into()),
                        signature: Some("my_second_signature".into()),
                        summary_id: None,
                        summary_text: None,
                        provider_type: None,
                    }),
                    ContentBlockChunk::ToolCall(ToolCallChunk {
                        id: "456".into(),
                        raw_name: Some("second_tool".into()),
                        raw_arguments: r#"{"my"  :  "second_arg"}"#.to_string(),
                    }),
                    ContentBlockChunk::Text(TextChunk {
                        id: "3".into(),
                        text: "Second text message".to_string(),
                    }),
                ],
                created: 123456,
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                }),
                latency: std::time::Duration::from_secs(0),
                raw_response: "My raw response".to_string(),
                finish_reason: Some(FinishReason::Length),
            })),]
        );
    }

    #[test]
    fn test_as_uninitialized_preserves_basic_fields() {
        let uninitialized = UninitializedMixtureOfNConfig {
            weight: Some(1.0),
            timeout_s: 60.0,
            candidates: vec!["variant1".to_string(), "variant2".to_string()],
            fuser: UninitializedFuserConfig {
                inner: UninitializedChatCompletionConfig {
                    model: "gpt-4".into(),
                    temperature: Some(0.3),
                    ..Default::default()
                },
            },
        };

        let config = uninitialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.weight, Some(1.0));
        assert_eq!(exported.timeout_s, 60.0);
        assert_eq!(
            exported.candidates,
            vec!["variant1".to_string(), "variant2".to_string()]
        );
        assert_eq!(exported.fuser.inner.model, "gpt-4".into());
        assert_eq!(exported.fuser.inner.temperature, Some(0.3));
    }

    #[test]
    fn test_as_uninitialized_preserves_nested_fuser() {
        let uninitialized = UninitializedMixtureOfNConfig {
            weight: None,
            timeout_s: 300.0,
            candidates: vec!["v1".to_string()],
            fuser: UninitializedFuserConfig {
                inner: UninitializedChatCompletionConfig {
                    model: "fuser-model".into(),
                    temperature: Some(0.1),
                    max_tokens: Some(50),
                    seed: Some(99),
                    ..Default::default()
                },
            },
        };

        let config = uninitialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.fuser.inner.model, "fuser-model".into());
        assert_eq!(exported.fuser.inner.temperature, Some(0.1));
        assert_eq!(exported.fuser.inner.max_tokens, Some(50));
        assert_eq!(exported.fuser.inner.seed, Some(99));
    }

    #[test]
    fn test_as_uninitialized_with_empty_candidates() {
        let uninitialized = UninitializedMixtureOfNConfig {
            weight: None,
            timeout_s: 300.0,
            candidates: vec![],
            fuser: UninitializedFuserConfig {
                inner: UninitializedChatCompletionConfig {
                    model: "gpt-4".into(),
                    ..Default::default()
                },
            },
        };

        let config = uninitialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        assert!(exported.candidates.is_empty());
    }

    #[test]
    fn test_as_uninitialized_serialization_round_trip() {
        let original = UninitializedMixtureOfNConfig {
            weight: Some(0.7),
            timeout_s: 120.0,
            candidates: vec!["a".to_string(), "b".to_string()],
            fuser: UninitializedFuserConfig {
                inner: UninitializedChatCompletionConfig {
                    model: "gpt-3.5-turbo".into(),
                    ..Default::default()
                },
            },
        };

        let config = original
            .clone()
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        // Serialize and deserialize
        let json = serde_json::to_string(&exported).unwrap();
        let deserialized: UninitializedMixtureOfNConfig = serde_json::from_str(&json).unwrap();

        // Should be able to load again
        let reloaded = deserialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        assert_eq!(reloaded.weight(), Some(0.7));
        assert_eq!(reloaded.timeout_s(), 120.0);
        assert_eq!(
            reloaded.candidates(),
            &vec!["a".to_string(), "b".to_string()]
        );
    }
}
