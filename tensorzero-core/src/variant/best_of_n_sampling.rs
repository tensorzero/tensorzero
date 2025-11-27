use std::borrow::Cow;
use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;

use futures::future::{join_all, try_join_all};
use lazy_static::lazy_static;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::time::timeout;

use crate::config::{ErrorContext, PathWithContents, SchemaData};
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::{InferenceClients, InferenceModels};
use crate::error::ErrorDetails;
use crate::inference::types::chat_completion_inference_params::ChatCompletionInferenceParamsV2;
use crate::inference::types::extra_body::FullExtraBodyConfig;
use crate::inference::types::extra_headers::FullExtraHeadersConfig;
use crate::inference::types::resolved_input::LazyResolvedInput;
use crate::inference::types::ContentBlockOutput;
use crate::inference::types::{
    batch::StartBatchModelInferenceWithMetadata, FunctionType, ModelInferenceRequest,
    ModelInferenceResponseWithMetadata, RequestMessage, Role, System,
};
use crate::jsonschema_util::StaticJSONSchema;
use crate::model::ModelTable;
use crate::tool::create_json_mode_tool_call_config_with_allowed_tools;
use crate::tool::{AllowedTools, AllowedToolsChoice, ToolCallConfig};
use crate::utils::unbounded_recursion_wrapper;
use crate::variant::mixture_of_n::stream_inference_from_non_stream;
use crate::{
    endpoints::inference::InferenceParams,
    error::Error,
    function::FunctionConfig,
    inference::types::{InferenceResult, InferenceResultStream},
    minijinja_util::TemplateConfig,
    variant::chat_completion::ChatCompletionConfig,
};

use super::chat_completion::UninitializedChatCompletionConfig;
use super::{InferenceConfig, JsonMode, ModelUsedInfo, Variant};

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct BestOfNSamplingConfig {
    weight: Option<f64>,
    timeout_s: f64,
    candidates: Vec<String>,
    evaluator: BestOfNEvaluatorConfig,
}

impl BestOfNSamplingConfig {
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

    pub fn evaluator(&self) -> &BestOfNEvaluatorConfig {
        &self.evaluator
    }

    /// Converts this initialized config back to its uninitialized form.
    pub fn as_uninitialized(self) -> UninitializedBestOfNSamplingConfig {
        UninitializedBestOfNSamplingConfig {
            weight: self.weight,
            timeout_s: self.timeout_s,
            candidates: self.candidates,
            evaluator: UninitializedBestOfNEvaluatorConfig {
                inner: self.evaluator.inner.as_uninitialized(),
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct UninitializedBestOfNSamplingConfig {
    #[serde(default)]
    pub weight: Option<f64>,
    #[serde(default = "default_timeout")]
    pub timeout_s: f64,
    pub candidates: Vec<String>,
    pub evaluator: UninitializedBestOfNEvaluatorConfig,
}

fn default_timeout() -> f64 {
    300.0
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct BestOfNEvaluatorConfig {
    #[serde(flatten)]
    pub inner: ChatCompletionConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct UninitializedBestOfNEvaluatorConfig {
    #[serde(flatten)]
    pub inner: UninitializedChatCompletionConfig,
}

impl UninitializedBestOfNSamplingConfig {
    pub fn load(
        self,
        schemas: &SchemaData,
        error_context: &ErrorContext,
    ) -> Result<BestOfNSamplingConfig, Error> {
        Ok(BestOfNSamplingConfig {
            weight: self.weight,
            timeout_s: self.timeout_s,
            candidates: self.candidates,
            evaluator: BestOfNEvaluatorConfig {
                inner: self.evaluator.inner.load(
                    schemas,
                    // Our stored evaluator is a plain `UninitializedChatCompletionConfig`, so we need
                    // to explicitly add `evaluator` to any error messages it produces.
                    &ErrorContext {
                        function_name: error_context.function_name.clone(),
                        variant_name: format!("{}.evaluator", error_context.variant_name),
                    },
                )?,
            },
        })
    }
}

const IMPLICIT_TOOL_NAME: &str = "respond";

lazy_static! {
    static ref EVALUATOR_OUTPUT_SCHEMA: StaticJSONSchema = {
        #[expect(clippy::expect_used)]
        StaticJSONSchema::from_value(json!({
            "type": "object",
            "properties": {
                "thinking": { "type": "string" },
                "answer_choice": { "type": "integer" }
            },
            "required": ["thinking", "answer_choice"],
            "additionalProperties": false
        }))
        .expect("Failed to create schema for evaluator output")
    };
    static ref JSON_MODE_TOOL_CALL_CONFIG: ToolCallConfig = {
        create_json_mode_tool_call_config_with_allowed_tools(
            EVALUATOR_OUTPUT_SCHEMA.clone(),
            AllowedTools {
                tools: [IMPLICIT_TOOL_NAME.to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::FunctionDefault,
            },
        )
    };
}

impl Variant for BestOfNSamplingConfig {
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
                    &clients,
                )
                .await?;
            self.select_best_candidate(
                &input,
                &models.models,
                &inference_config,
                &clients,
                candidate_inference_results,
            )
            .await
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
        let candidate_inference_results = self
            .infer_candidates(
                &input,
                &models,
                &function,
                Arc::clone(&inference_config),
                &clients,
            )
            .await?;
        let inference_result = self
            .select_best_candidate(
                &input,
                &models.models,
                &inference_config,
                &clients,
                candidate_inference_results,
            )
            .await?;

        // We always invoke our candidates in non-streaming mode (since we need to concatenate their responses
        // to produce the judge input)
        // Take the judge's chosen candidate, and convert the candidate response to a stream
        stream_inference_from_non_stream(inference_result, inference_params)
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
        // Validate the evaluator variant
        self.evaluator
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
    // We only return templates for the evaluator variant.
    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        self.evaluator.inner.get_all_template_paths()
    }

    fn get_all_explicit_template_names(&self) -> HashSet<String> {
        // The candidate variants will  already have 'get_all_explicit_template_names' called on them,
        // so we don't need to look them up here
        self.evaluator.inner.get_all_explicit_template_names()
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

impl BestOfNSamplingConfig {
    /// Infer each candidate variant concurrently and return the results.
    async fn infer_candidates(
        &self,
        input: &LazyResolvedInput,
        models: &InferenceModels,
        function: &Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: &InferenceClients,
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
            let models = models.clone();
            let clients = clients.clone();
            let function = Arc::clone(function);
            let input = Arc::new(input.clone());
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
                    // It logs on construction
                    Error::new(ErrorDetails::InferenceTimeout {
                        variant_name: candidate_name.clone(),
                    });
                }
            }
        }

        Ok(successful_results)
    }

    /// Gets the best candidate using the evaluator config.
    /// If at any point the evaluator fails to return a valid response,
    /// we randomly select one of the candidates.
    async fn select_best_candidate(
        &self,
        input: &LazyResolvedInput,
        models: &Arc<ModelTable>,
        inference_config: &InferenceConfig,
        clients: &InferenceClients,
        candidates: Vec<InferenceResult>,
    ) -> Result<InferenceResult, Error> {
        if candidates.is_empty() {
            return Err(ErrorDetails::Inference {
                message: "No candidates to select from in best of n".to_string(),
            }
            .into());
        }
        if candidates.len() == 1 {
            let mut candidates = candidates;
            return candidates.pop().ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: "Expected one candidate but found none".to_string(),
                })
            });
        }
        // If the evaluator fails, we randomly select one of the candidates
        // As long as the evaluator returns an inference result, we want to include it in the observability
        let (selection_idx, inference_result) = match inner_select_best_candidate(
            &self.evaluator,
            input,
            models,
            inference_config,
            clients,
            &candidates,
        )
        .await
        {
            Ok((idx_opt, inf_result)) => (
                idx_opt.unwrap_or_else(|| rand::rng().random_range(0..candidates.len())),
                inf_result,
            ),
            Err(_) => (rand::rng().random_range(0..candidates.len()), None),
        };

        // Safely remove the selected candidate without panicking
        let mut candidates = candidates;
        let mut selected_candidate = if selection_idx < candidates.len() {
            candidates.swap_remove(selection_idx)
        } else {
            return Err(ErrorDetails::Inference {
                message: "The index chosen by the evaluator is out of bounds (should never happen)"
                    .to_string(),
            }
            .into());
        };
        if let Some(inference_result) = &inference_result {
            // Pass the evaluator response back to the user as 'original_response'
            selected_candidate.set_original_response(Some(inference_result.raw_response.clone()));
        } else {
            // If the evaluator failed, don't provide an 'original_response' to the uesr
            selected_candidate.set_original_response(None);
        }
        for candidate in candidates {
            selected_candidate
                .mut_model_inference_results()
                .extend(candidate.owned_model_inference_results());
        }
        if let Some(inference_result) = inference_result {
            selected_candidate
                .mut_model_inference_results()
                .push(inference_result);
        }

        Ok(selected_candidate)
    }
}

/// Attempts to select the best candidate for best of n.
/// If this function returns an error or the index is None, we will randomly select one
/// of the candidates in the outer function.
/// If a model inference actually occurs, we return None and the model inference result instead of Err() so
/// that we can still observe the model inference result in ClickHouse.
///
/// Here are the steps in the function:
///  * Prepare the request for the evaluator variant.
///  * Infer the request using the model specified in the evaluator config.
///  * Parse the output of the evaluator.
///  * Map the evaluator's index to the actual index in the original candidate list (prior to skipping any).
///  * Check if the index is out of bounds.
///  * Return the index and the model inference result.
async fn inner_select_best_candidate<'a>(
    evaluator: &'a BestOfNEvaluatorConfig,
    input: &LazyResolvedInput,
    models: &'a ModelTable,
    inference_config: &InferenceConfig,
    clients: &InferenceClients,
    candidates: &[InferenceResult],
) -> Result<(Option<usize>, Option<ModelInferenceResponseWithMetadata>), Error> {
    let mut inference_params = InferenceParams::default();
    let (inference_request, skipped_indices) = evaluator
        .prepare_evaluator_request(input, inference_config, candidates, &mut inference_params)
        .await?;
    if skipped_indices.len() == candidates.len() {
        return Err(ErrorDetails::Inference {
            message: "No valid candidates available to prepare request.".to_string(),
        }
        .into());
    }
    // If there is only one candidate that was not skipped, we return that one without running inference
    if skipped_indices.len() == candidates.len() - 1 {
        let selected_index = (0..candidates.len())
            .find(|&i| !skipped_indices.contains(&i))
            .ok_or_else(|| Error::new(ErrorDetails::Inference {
                message:
                    "No valid candidates available to prepare request (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new"
                        .to_string(),
            }))?;
        // Return the selected index and None for the model inference result
        return Ok((Some(selected_index), None));
    }
    let model_config = models.get(evaluator.inner.model()).await?.ok_or_else(|| {
        Error::new(ErrorDetails::UnknownModel {
            name: evaluator.inner.model().to_string(),
        })
    })?;
    let model_inference_response = evaluator
        .inner
        .retries()
        .retry(|| async {
            model_config
                .infer(&inference_request, clients, evaluator.inner.model())
                .await
        })
        .await?;
    let model_inference_result = ModelInferenceResponseWithMetadata::new(
        model_inference_response,
        evaluator.inner.model().clone(),
    );
    let raw = match model_inference_result
        .output
        .iter()
        .find_map(|block| match block {
            ContentBlockOutput::Text(text) => Some(&text.text),
            ContentBlockOutput::ToolCall(tool_call) => Some(&tool_call.arguments),
            ContentBlockOutput::Thought(_) | ContentBlockOutput::Unknown(_) => None,
        }) {
        Some(text) => text,
        None => {
            Error::new(ErrorDetails::Inference {
                message: "The evaluator did not return a text response".to_string(),
            });
            return Ok((None, Some(model_inference_result)));
        }
    };
    let parsed_output = match serde_json::from_str::<Value>(raw) {
        Ok(value) => value,
        Err(e) => {
            Error::new(ErrorDetails::Inference {
                message: format!("The evaluator did not return a valid JSON response: {e}"),
            });
            return Ok((None, Some(model_inference_result)));
        }
    };
    let answer_choice = match parsed_output.get("answer_choice") {
        Some(val) => match val.as_u64() {
            Some(num) => num as usize,
            None => {
                Error::new(ErrorDetails::Inference {
                    message: format!(
                        "The evaluator did not return a valid integer answer choice: {val}"
                    ),
                });
                return Ok((None, Some(model_inference_result)));
            }
        },
        None => {
            Error::new(ErrorDetails::Inference {
                message: format!(
                    "The evaluator returned a JSON response without an answer_choice field: {parsed_output}"
                ),
            });
            return Ok((None, Some(model_inference_result)));
        }
    };
    // Map the evaluator's index to the actual index
    let answer_choice = map_evaluator_to_actual_index(answer_choice, &skipped_indices);
    if answer_choice >= candidates.len() {
        Error::new(ErrorDetails::Inference {
            message: format!(
                "The index chosen by the evaluator is out of bounds: {} >= {}",
                answer_choice,
                candidates.len()
            ),
        });
        return Ok((None, Some(model_inference_result)));
    }
    Ok((Some(answer_choice as usize), Some(model_inference_result)))
}

impl BestOfNEvaluatorConfig {
    /// Prepares the system message for the evaluator variant.
    /// We use the system_template of the evaluator variant to generate a system message as if we
    /// were using the evaluator variant directly to solve the problem.
    /// Then, we template that system message into a broader set of instructions that includes
    /// information about what the evaluator will be asked to do (choose a candidate).
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
        templates.template_message("t0:best_of_n_evaluator_system", &template_context)
    }

    /// Prepares the final candidate message for the evaluator variant.
    ///
    /// This function constructs a `RequestMessage` that includes all valid candidate outputs
    /// by templating them into a predefined evaluation template. It handles different types of
    /// inference results:
    ///
    /// - **Chat Inference**: Serializes the content blocks to a JSON string.
    /// - **JSON Inference**: Uses the raw JSON output if it contains correctly parsed data; otherwise,
    ///   skips the candidate.
    ///
    /// Additionally, it tracks and returns the indices of any candidates that were skipped due
    /// to missing or invalid parsed outputs. This allows the caller to be aware of which
    /// candidates were not included in the evaluation message.
    ///
    /// # Parameters
    ///
    /// - `templates`: Reference to the `TemplateConfig` used for templating messages.
    /// - `candidates`: A vector of `InferenceResult` instances representing the candidate outputs.
    ///
    /// # Returns
    ///
    /// On success, returns a tuple containing:
    /// - `RequestMessage`: The templated message to be sent to the evaluator.
    /// - `Vec<usize>`: A sorted vector of indices indicating which candidates were skipped.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if any of the candidate outputs fail to serialize or if templating fails.
    fn prepare_candidate_message(
        templates: &TemplateConfig,
        candidates: &[InferenceResult],
    ) -> Result<(RequestMessage, Vec<usize>), Error> {
        let mut candidate_outputs = Vec::new();
        let mut skipped_indices = Vec::new();
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
                }
                InferenceResult::Json(json_result) => {
                    match (&json_result.output.raw, &json_result.output.parsed) {
                        (Some(raw), Some(_)) => {
                            candidate_outputs.push(raw.clone());
                        }
                        _ => skipped_indices.push(i),
                    }
                }
            }
        }
        let template_context = json!({
            "candidates": candidate_outputs,
        });
        let message_text =
            templates.template_message("t0:best_of_n_evaluator_candidates", &template_context)?;
        Ok((
            RequestMessage {
                role: Role::User,
                content: vec![message_text.into()],
            },
            skipped_indices,
        ))
    }

    /// Prepares the request for the evaluator variant.
    /// We use the `prepare_system_message` and `prepare_candidate_message` functions to generate
    /// the system and candidate messages for the evaluator, which take candidate selection into account.
    ///
    /// Additionally, this function returns the indices of candidates that were skipped due to
    /// serialization or parsing issues, allowing the caller to handle or log these skipped candidates as needed.
    ///
    /// # Returns
    ///
    /// On success, returns a tuple containing:
    /// - `ModelInferenceRequest`: The request prepared for the model inference.
    /// - `Vec<usize>`: A sorted vector of indices indicating which candidates were skipped.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if any of the candidate outputs fail to serialize or if templating fails.
    async fn prepare_evaluator_request<'a>(
        &self,
        input: &LazyResolvedInput,
        inference_config: &InferenceConfig,
        candidates: &[InferenceResult],
        inference_params: &mut InferenceParams,
    ) -> Result<(ModelInferenceRequest<'a>, Vec<usize>), Error> {
        // Do this before we prepare the system message so we can use the correct max index in the system message
        let (candidate_message, skipped_indices) =
            Self::prepare_candidate_message(&inference_config.templates, candidates)?;
        // Need to subtract the skipped indices from the total number of candidates to get the correct max index
        let max_index = candidates
            .len()
            .checked_sub(skipped_indices.len())
            .and_then(|len| len.checked_sub(1))
            .ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: "No valid candidates available to prepare request.".to_string(),
                })
            })?;
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
        let json_mode = inference_params
            .chat_completion
            .json_mode
            .or_else(|| self.inner.json_mode().cloned())
            .unwrap_or(JsonMode::Strict);
        let tool_config = match json_mode {
            JsonMode::Tool => Some(Cow::Borrowed(&*JSON_MODE_TOOL_CALL_CONFIG)),
            JsonMode::Off | JsonMode::On | JsonMode::Strict => None,
        };
        if !inference_config.extra_body.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message: "Inference-level `extra_body` is not yet supported for best_of_n variant"
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
            inference_extra_headers: Default::default(),
        };
        Ok((
            ModelInferenceRequest {
                inference_id: inference_config.ids.inference_id,
                messages,
                system,
                tool_config,
                temperature: inference_params.chat_completion.temperature,
                max_tokens: inference_params.chat_completion.max_tokens,
                seed: inference_params.chat_completion.seed,
                top_p: inference_params.chat_completion.top_p,
                presence_penalty: inference_params.chat_completion.presence_penalty,
                frequency_penalty: inference_params.chat_completion.frequency_penalty,
                stop_sequences: inference_params
                    .chat_completion
                    .stop_sequences
                    .clone()
                    .map(Cow::Owned),
                stream: false,
                json_mode: json_mode.into(),
                function_type: FunctionType::Json,
                fetch_and_encode_input_files_before_inference: inference_config
                    .fetch_and_encode_input_files_before_inference,
                output_schema: Some(&EVALUATOR_OUTPUT_SCHEMA.value),
                extra_body,
                extra_headers,
                extra_cache_key: inference_config.extra_cache_key.clone(),
                inference_params_v2: ChatCompletionInferenceParamsV2 {
                    reasoning_effort: inference_params.chat_completion.reasoning_effort.clone(),
                    service_tier: inference_params.chat_completion.service_tier.clone(),
                    thinking_budget_tokens: inference_params.chat_completion.thinking_budget_tokens,
                    verbosity: inference_params.chat_completion.verbosity.clone(),
                },
            },
            skipped_indices,
        ))
    }
}

/// Maps the evaluator's selected index to the actual index in the original candidate list.
///
/// # Parameters
/// - `evaluator_idx`: The index selected by the evaluator from the filtered list.
/// - `skipped_indices`: A sorted list of indices that were skipped.
///
/// # Returns
/// - `usize`: The corresponding actual index in the original list.
fn map_evaluator_to_actual_index(evaluator_idx: usize, skipped_indices: &[usize]) -> usize {
    let mut actual_idx = evaluator_idx;
    for &skipped in skipped_indices {
        if skipped <= actual_idx {
            actual_idx += 1;
        }
    }
    actual_idx
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use uuid::Uuid;

    use crate::rate_limiting::ScopeInfo;
    use crate::{
        cache::{CacheEnabledMode, CacheOptions},
        config::{provider_types::ProviderTypesConfig, UninitializedSchemas},
        db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
        endpoints::inference::{InferenceCredentials, InferenceIds},
        http::TensorzeroHttpClient,
        inference::types::{
            Arguments, ChatInferenceResult, FinishReason, JsonInferenceResult, Latency,
            RequestMessagesOrBatch, Usage,
        },
        minijinja_util::tests::{
            get_system_filled_template, get_system_template, get_test_template_config,
        },
        model::{ModelConfig, ModelProvider, ProviderConfig},
        model_table::ProviderTypeDefaultCredentials,
        providers::dummy::DummyProvider,
    };

    use super::*;

    #[test]
    fn test_static_schema() {
        // Also covers the fact that the lazy schema works
        let instance = json!({
            "thinking": "I am thinking",
            "answer_choice": 0
        });
        let result = EVALUATOR_OUTPUT_SCHEMA.validate(&instance);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_prepare_system_message() {
        let templates = get_test_template_config().await;

        let system_schema = StaticJSONSchema::from_value(serde_json::json!({
            "type": "object",
            "properties": {
                "assistant_name": {
                    "type": "string"
                }
            },
            "required": ["assistant_name"]
        }))
        .unwrap();

        // Test without templates, string message
        let evaluator_config = BestOfNEvaluatorConfig {
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
            evaluator_config.prepare_system_message(&templates, Some(&input_message), max_index);
        let prepared_message = result.unwrap();
        let expected_message = templates
            .template_message(
                "t0:best_of_n_evaluator_system",
                &json!({"inner_system_message": "You are a helpful assistant.", "max_index": max_index}),
            )
            .unwrap();
        assert_eq!(prepared_message, expected_message);

        // Test without templates, object message
        let evaluator_config = BestOfNEvaluatorConfig {
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
            evaluator_config.prepare_system_message(&templates, Some(&input_message), max_index);
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
        let evaluator_config = BestOfNEvaluatorConfig {
            inner: UninitializedChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let max_index = 5;
        let result = evaluator_config.prepare_system_message(&templates, None, max_index);
        let expected_message = templates
            .template_message(
                "t0:best_of_n_evaluator_system",
                &json!({"max_index": max_index}),
            )
            .unwrap();
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(prepared_message, expected_message);

        let system_template = get_system_template();

        let evaluator_config = BestOfNEvaluatorConfig {
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
                    Some(system_schema),
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
        let result =
            evaluator_config.prepare_system_message(&templates, Some(&input_message), max_index);
        let prepared_message = result.unwrap();
        let inner_system_message = templates
            .template_message(
                "system",
                &json!({"assistant_name": "ChatGPT", "max_index": max_index}),
            )
            .unwrap();
        let expected_message = templates
            .template_message(
                "t0:best_of_n_evaluator_system",
                &json!({"inner_system_message": inner_system_message, "max_index": max_index}),
            )
            .unwrap();
        assert_eq!(prepared_message, expected_message);

        // Test with template that is complete as is (string)
        let system_template_name = "system_filled";
        let system_template = get_system_filled_template();

        let evaluator_config = BestOfNEvaluatorConfig {
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
                &SchemaData::load(None, None, None, UninitializedSchemas::default(), "test")
                    .unwrap(),
                &ErrorContext {
                    function_name: "test".to_string(),
                    variant_name: "test".to_string(),
                },
            )
            .unwrap(),
        };

        let max_index = 10;
        let prepared_message = evaluator_config
            .prepare_system_message(&templates, None, max_index)
            .unwrap();
        let inner_system_message = templates
            .template_message(system_template_name, &json!({}))
            .unwrap();
        let expected_message = templates
            .template_message(
                "t0:best_of_n_evaluator_system",
                &json!({"inner_system_message": inner_system_message, "max_index": max_index}),
            )
            .unwrap();
        assert_eq!(prepared_message, expected_message);
    }

    #[tokio::test]
    async fn test_prepare_candidate_message() {
        let templates = get_test_template_config().await;

        // Create an EvaluatorConfig
        // Prepare some candidate InferenceResults
        let model_inference_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            output: vec!["Candidate answer 1".to_string().into()],
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]),
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
            system: Some("test_system".to_string()),
            input_messages: RequestMessagesOrBatch::Message(vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]),
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
        let result = BestOfNEvaluatorConfig::prepare_candidate_message(&templates, &candidates);
        assert!(result.is_ok());
        let (request_message, skipped_indices) = result.unwrap();
        assert!(skipped_indices.is_empty());

        let expected_message_text = "Here are the candidate answers (with the index and a row of ------ separating):\n0: [{\"type\":\"text\",\"text\":\"Candidate answer 1\"}]\n------\n1: [{\"type\":\"text\",\"text\":\"Candidate answer 2\"}]\n------\nPlease evaluate these candidates and provide the index of the best one.".to_string();
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
            system: Some("test_system".to_string()),
            input_messages: RequestMessagesOrBatch::Message(vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]),
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Valid JSON response\"}".to_string(),
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
            system: Some("test_system".to_string()),
            input_messages: RequestMessagesOrBatch::Message(vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]),
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
            finish_reason: Some(FinishReason::ToolCall),
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
        let result = BestOfNEvaluatorConfig::prepare_candidate_message(&templates, &candidates);
        assert!(result.is_ok());
        let (request_message, skipped_indices) = result.unwrap();

        // Expect skipped_indices to contain index 1
        assert_eq!(skipped_indices, vec![1]);

        let expected_message_text = "Here are the candidate answers (with the index and a row of ------ separating):\n0: {\"response\": \"Valid JSON response\"}\n------\nPlease evaluate these candidates and provide the index of the best one.".to_string();

        // Check that the request_message has the expected role and content
        assert_eq!(request_message.role, Role::User);
        assert_eq!(request_message.content, vec![expected_message_text.into()]);
    }

    #[tokio::test]
    async fn test_select_best_candidate() {
        // Set up evaluator with a provider that returns a valid answer_choice
        let evaluator_config = BestOfNEvaluatorConfig {
            inner: UninitializedChatCompletionConfig {
                model: "best_of_n_1".into(),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let best_of_n_variant = BestOfNSamplingConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec![],
            evaluator: evaluator_config,
        };

        let templates = get_test_template_config().await;
        // Prepare some candidate InferenceResults
        let model_inference_response0 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            output: vec!["Candidate answer 0".to_string().into()],
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Example response\"}".to_string(),
            system: Some("test_system".to_string()),
            input_messages: RequestMessagesOrBatch::Message(vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]),
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
            system: Some("test_system".to_string()),
            input_messages: RequestMessagesOrBatch::Message(vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]),
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
                "best_of_n_1".into(),
                ModelConfig {
                    routing: vec!["best_of_n_1".into()],
                    providers: HashMap::from([(
                        "best_of_n_1".into(),
                        ModelProvider {
                            name: "best_of_n_1".into(),
                            config: ProviderConfig::Dummy(DummyProvider {
                                model_name: "best_of_n_1".into(),
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
            credentials: Arc::new(api_keys),
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

        let selected = best_of_n_variant
            .select_best_candidate(
                &input,
                &Arc::new(models),
                &inference_config,
                &inference_clients,
                candidates.clone(),
            )
            .await
            .expect("Failed to select best candidate");

        // Expect the second candidate to be selected (index 1)
        // based on "answer": 1 in best_of_n_1
        let expected_id = inference_id1;
        let expected_usage = Usage {
            input_tokens: Some(75),
            output_tokens: Some(126),
        };
        let expected_content = vec!["Candidate answer 1".to_string().into()];
        assert_eq!(selected.usage_considering_cached(), expected_usage);
        match selected {
            InferenceResult::Chat(selected) => {
                assert_eq!(selected.inference_id, expected_id);
                assert_eq!(selected.content, expected_content);
                assert_eq!(selected.model_inference_results.len(), 3);
                assert_eq!(selected.finish_reason, Some(FinishReason::Stop));
            }
            InferenceResult::Json(_) => {
                panic!("Expected a Chat inference result");
            }
        }
        // Set up evaluator with a provider that fails
        let evaluator_config = BestOfNEvaluatorConfig {
            inner: UninitializedChatCompletionConfig {
                model: "error".into(),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let best_of_n_variant = BestOfNSamplingConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec![],
            evaluator: evaluator_config,
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

        let result = best_of_n_variant
            .select_best_candidate(
                &input,
                &Arc::new(models),
                &inference_config,
                &inference_clients,
                candidates.clone(),
            )
            .await;

        // Expect an error and a random candidate to be selected
        let choice = result.unwrap();
        // We know that the model will fail, so there should only be two results
        match choice {
            InferenceResult::Chat(chat_choice) => {
                assert!(chat_choice.model_inference_results.len() == 2);
            }
            InferenceResult::Json(_) => {
                panic!("Expected a Chat inference result");
            }
        }
        // Depending on implementation, you might check which candidate was selected

        // Set up evaluator with a provider that returns invalid JSON
        let evaluator_config = BestOfNEvaluatorConfig {
            inner: UninitializedChatCompletionConfig {
                model: "regular".into(),
                ..Default::default()
            }
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap(),
        };
        let best_of_n_variant = BestOfNSamplingConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec![],
            evaluator: evaluator_config,
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
                crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT,
            )
            .expect("Failed to create model table")
        };
        let input = LazyResolvedInput {
            system: None,
            messages: vec![],
        };

        let models_arc = Arc::new(models);
        let result = best_of_n_variant
            .select_best_candidate(
                &input,
                &models_arc,
                &inference_config,
                &inference_clients,
                candidates.clone(),
            )
            .await;

        let choice = result.unwrap();
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
        let result = best_of_n_variant
            .select_best_candidate(
                &input,
                &models_arc,
                &inference_config,
                &inference_clients,
                empty_candidates.clone(),
            )
            .await;
        let err = result.unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::Inference {
                message: "No candidates to select from in best of n".to_string()
            }
            .into()
        );

        // Test case: Index returned too large (should return an error)
        let best_of_n_big_variant = BestOfNSamplingConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec![],
            evaluator: BestOfNEvaluatorConfig {
                inner: UninitializedChatCompletionConfig {
                    model: "best_of_n_big".into(),
                    weight: Some(1.0),
                    ..Default::default()
                }
                .load(&SchemaData::default(), &ErrorContext::new_test())
                .unwrap(),
            },
        };

        let mut big_models = HashMap::new();
        big_models.insert(
            "best_of_n_big".into(),
            ModelConfig {
                routing: vec!["best_of_n_big".into()],
                providers: HashMap::from([(
                    "best_of_n_big".into(),
                    ModelProvider {
                        name: "best_of_n_big".into(),
                        config: ProviderConfig::Dummy(DummyProvider {
                            model_name: "best_of_n_big".into(),
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
        let big_models = ModelTable::new(
            big_models,
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT,
        )
        .expect("Failed to create model table");

        let result_big = best_of_n_big_variant
            .select_best_candidate(
                &input,
                &Arc::new(big_models),
                &inference_config,
                &inference_clients,
                candidates.clone(),
            )
            .await;
        // we gracefully handle the error and return a random candidate
        let _result = result_big.unwrap();
    }

    #[test]
    fn test_map_evaluator_to_actual_index() {
        // Case 1: No skipped indices
        let skipped = vec![];
        assert_eq!(map_evaluator_to_actual_index(0, &skipped), 0);
        assert_eq!(map_evaluator_to_actual_index(1, &skipped), 1);

        // Case 2: Skipped index before evaluator's choice
        let skipped = vec![1];
        assert_eq!(map_evaluator_to_actual_index(0, &skipped), 0);
        assert_eq!(map_evaluator_to_actual_index(1, &skipped), 2);

        // Case 3: Multiple skipped indices
        let skipped = vec![1, 3];
        assert_eq!(map_evaluator_to_actual_index(0, &skipped), 0);
        assert_eq!(map_evaluator_to_actual_index(1, &skipped), 2);
        assert_eq!(map_evaluator_to_actual_index(2, &skipped), 4);
        assert_eq!(map_evaluator_to_actual_index(3, &skipped), 5);

        // Case 4: All possible skipped
        let skipped = vec![0, 1, 2, 3, 4];
        assert_eq!(map_evaluator_to_actual_index(0, &skipped), 5);
        assert_eq!(map_evaluator_to_actual_index(1, &skipped), 6);

        // Case 5: Skipped indices out of range
        let skipped = vec![10, 20];
        assert_eq!(map_evaluator_to_actual_index(5, &skipped), 5);
    }

    #[test]
    fn test_as_uninitialized_preserves_basic_fields() {
        let uninitialized = UninitializedBestOfNSamplingConfig {
            weight: Some(1.0),
            timeout_s: 60.0,
            candidates: vec!["variant1".to_string(), "variant2".to_string()],
            evaluator: UninitializedBestOfNEvaluatorConfig {
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
        assert_eq!(exported.evaluator.inner.model, "gpt-4".into());
        assert_eq!(exported.evaluator.inner.temperature, Some(0.3));
    }

    #[test]
    fn test_as_uninitialized_preserves_nested_evaluator() {
        let uninitialized = UninitializedBestOfNSamplingConfig {
            weight: None,
            timeout_s: 300.0,
            candidates: vec!["v1".to_string()],
            evaluator: UninitializedBestOfNEvaluatorConfig {
                inner: UninitializedChatCompletionConfig {
                    model: "judge-model".into(),
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

        assert_eq!(exported.evaluator.inner.model, "judge-model".into());
        assert_eq!(exported.evaluator.inner.temperature, Some(0.1));
        assert_eq!(exported.evaluator.inner.max_tokens, Some(50));
        assert_eq!(exported.evaluator.inner.seed, Some(99));
    }

    #[test]
    fn test_as_uninitialized_with_empty_candidates() {
        let uninitialized = UninitializedBestOfNSamplingConfig {
            weight: None,
            timeout_s: 300.0,
            candidates: vec![],
            evaluator: UninitializedBestOfNEvaluatorConfig {
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
        let original = UninitializedBestOfNSamplingConfig {
            weight: Some(0.7),
            timeout_s: 120.0,
            candidates: vec!["a".to_string(), "b".to_string()],
            evaluator: UninitializedBestOfNEvaluatorConfig {
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
        let deserialized: UninitializedBestOfNSamplingConfig = serde_json::from_str(&json).unwrap();

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
