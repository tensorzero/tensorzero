use std::path::Path;

use futures::future::join_all;
use rand::Rng;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::time::{timeout, Duration};

use crate::config_parser::PathWithContents;
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::{InferenceClients, InferenceModels};
use crate::inference::types::extra_body::FullExtraBodyConfig;
use crate::inference::types::ResolvedInput;
use crate::inference::types::{
    batch::StartBatchModelInferenceWithMetadata, ModelInferenceRequest, RequestMessage, Role, Usage,
};
use crate::model::ModelTable;
use crate::{
    endpoints::inference::InferenceParams,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    inference::types::{InferenceResult, InferenceResultStream},
    minijinja_util::TemplateConfig,
    variant::chat_completion::ChatCompletionConfig,
};

use crate::config_parser::LoadableConfig;
use crate::variant::chat_completion::UninitializedChatCompletionConfig;

use super::{
    infer_model_request, prepare_model_inference_request, InferModelRequestArgs, InferenceConfig,
    ModelUsedInfo, Variant,
};

#[derive(Debug)]
pub struct MixtureOfNConfig {
    pub weight: Option<f64>,
    pub timeout_s: f64,
    pub candidates: Vec<String>,
    pub fuser: FuserConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug)]
pub struct FuserConfig {
    pub inner: ChatCompletionConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedFuserConfig {
    #[serde(flatten)]
    pub inner: UninitializedChatCompletionConfig,
}

impl LoadableConfig<MixtureOfNConfig> for UninitializedMixtureOfNConfig {
    fn load<P: AsRef<Path>>(self, base_path: P) -> Result<MixtureOfNConfig, Error> {
        Ok(MixtureOfNConfig {
            weight: self.weight,
            timeout_s: self.timeout_s,
            candidates: self.candidates,
            fuser: FuserConfig {
                inner: self.fuser.inner.load(base_path)?,
            },
        })
    }
}

impl Variant for MixtureOfNConfig {
    async fn infer<'a: 'request, 'request>(
        &self,
        input: &ResolvedInput,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'static, 'request>,
        clients: &'request InferenceClients<'request>,
        _inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let candidate_inference_results = self
            .infer_candidates(input, models, function, inference_config, clients)
            .await?;
        self.fuse_candidates(
            input,
            function,
            models.models,
            inference_config,
            clients,
            candidate_inference_results,
        )
        .await
    }

    async fn infer_stream<'request>(
        &self,
        _input: &ResolvedInput,
        _models: &'request InferenceModels<'_>,
        _function: &FunctionConfig,
        _inference_config: &'request InferenceConfig<'static, 'request>,
        _clients: &'request InferenceClients<'request>,
        _inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        Err(ErrorDetails::InvalidRequest {
            message: "Best of n variants do not support streaming inference.".to_string(),
        }
        .into())
    }

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &mut ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        // Validate each candidate variant
        for candidate in &self.candidates {
            let variant = function.variants().get(candidate).ok_or_else(|| {
                Error::new(ErrorDetails::UnknownCandidate {
                    name: candidate.to_string(),
                })
            })?;
            variant
                .validate(
                    function,
                    models,
                    embedding_models,
                    templates,
                    function_name,
                    candidate,
                )
                .map_err(|e| {
                    Error::new(ErrorDetails::InvalidCandidate {
                        variant_name: variant_name.to_string(),
                        message: e.to_string(),
                    })
                })?;
        }
        // Validate the evaluator variant
        self.fuser.inner.validate(
            function,
            models,
            embedding_models,
            templates,
            function_name,
            variant_name,
        )?;
        Ok(())
    }

    // We do not return templates for the candidates, as they are required to be variants in the same function
    // and will therefore also have the same templates.
    // We only return templates for the evaluator variant.
    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        self.fuser.inner.get_all_template_paths()
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _input: &[ResolvedInput],
        _models: &'a InferenceModels<'a>,
        _function: &'a FunctionConfig,
        _inference_configs: &'a [InferenceConfig<'a, 'a>],
        _clients: &'a InferenceClients<'a>,
        _inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        Err(ErrorDetails::UnsupportedVariantForBatchInference { variant_name: None }.into())
    }
}

impl MixtureOfNConfig {
    /// Infer each candidate variant concurrently and return the results.
    async fn infer_candidates<'a, 'request>(
        &self,
        input: &ResolvedInput,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'static, 'request>,
        clients: &'request InferenceClients<'request>,
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
                let mut config = inference_config.clone();
                config.extra_cache_key = Some(format!("candidate_{i}"));
                Ok((candidate.to_string(), variant, config))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Start the inference tasks (we keep the names around for logging)
        let mut inference_futures = Vec::new();
        for (candidate_name, candidate_variant, config) in &candidate_variants {
            inference_futures.push((
                candidate_name.clone(),
                timeout(
                    Duration::from_secs_f64(self.timeout_s),
                    candidate_variant.infer(
                        input,
                        models,
                        function,
                        config,
                        clients,
                        InferenceParams::default(),
                    ),
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
                        successful_results.push(res)
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
    async fn fuse_candidates<'a, 'request>(
        &'a self,
        input: &ResolvedInput,
        function: &'a FunctionConfig,
        models: &'a ModelTable,
        inference_config: &'request InferenceConfig<'a, 'request>,
        clients: &'request InferenceClients<'request>,
        mut candidates: Vec<InferenceResult>,
    ) -> Result<InferenceResult, Error> {
        if candidates.is_empty() {
            return Err(ErrorDetails::Inference {
                message: "No candidates to fuse in the mixture of n".to_string(),
            }
            .into());
        }
        if candidates.len() == 1 {
            return candidates.pop().ok_or_else(|| Error::new(ErrorDetails::Inference {
                message: "Expected one candidate but found none. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
            }));
        }
        let mut candidates = candidates;
        // If the fuser fails, we randomly select one of the candidates
        // As long as the fuser returns an inference result, we want to include it in the observability
        let mut inference_result = match inner_fuse_candidates(
            &self.fuser,
            input,
            models,
            function,
            inference_config,
            clients,
            &candidates,
        )
        .await
        {
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
                candidate
            }
        };

        // Safely remove the selected candidate without panicking
        let mut total_usage: Usage = candidates.iter().map(|c| c.usage()).sum();
        total_usage.input_tokens += inference_result.usage().input_tokens;
        total_usage.output_tokens += inference_result.usage().output_tokens;
        inference_result.set_usage(total_usage);
        for candidate in candidates {
            inference_result
                .mut_model_inference_results()
                .extend(candidate.owned_model_inference_results());
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
async fn inner_fuse_candidates<'a, 'request>(
    fuser: &'a FuserConfig,
    input: &'request ResolvedInput,
    models: &'a ModelTable,
    function: &'a FunctionConfig,
    inference_config: &'request InferenceConfig<'a, 'request>,
    clients: &'request InferenceClients<'request>,
    candidates: &[InferenceResult],
) -> Result<InferenceResult, Error> {
    let (inference_request, included_indices) = fuser.prepare_request(
        input,
        function,
        inference_config,
        candidates,
        &mut InferenceParams::default(),
    )?;
    if included_indices.is_empty() {
        return Err(ErrorDetails::Inference {
            message: "No valid candidates available to prepare request.".to_string(),
        }
        .into());
    }
    let model_config = models.get(&fuser.inner.model)?.ok_or_else(|| {
        Error::new(ErrorDetails::UnknownModel {
            name: fuser.inner.model.to_string(),
        })
    })?;
    let infer_model_request_args = InferModelRequestArgs {
        request: inference_request,
        model_name: fuser.inner.model.clone(),
        model_config: &model_config,
        function,
        inference_config,
        retry_config: &fuser.inner.retries,
        clients,
        inference_params: InferenceParams::default(),
    };
    let inference_result = infer_model_request(infer_model_request_args).await?;
    Ok(inference_result)
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
        system: Option<&Value>,
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
    /// - `RequestMessage`: The templated message to be sent to the evaluator.
    /// - `Vec<usize>`: A sorted vector of indices indicating which candidates were successfully included in the fuser message.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if any of the candidate outputs fail to serialize or if templating fails.
    fn prepare_candidate_message(
        &self,
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
                    if json_result.output.parsed.is_some() {
                        candidate_outputs.push(json_result.output.raw.clone());
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

    /// Prepares the request for the evaluator variant.
    /// We use the `prepare_system_message` and `prepare_candidate_message` functions to generate
    /// the system and candidate messages for the evaluator, which take candidate selection into account.
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
    fn prepare_request<'a, 'request>(
        &'a self,
        input: &'request ResolvedInput,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'a, 'request>,
        candidates: &[InferenceResult],
        inference_params: &mut InferenceParams,
    ) -> Result<(ModelInferenceRequest<'request>, Vec<usize>), Error>
    where
        'a: 'request,
    {
        // Do this before we prepare the system message so we can use the correct max index in the system message
        let (candidate_message, included_indices) =
            self.prepare_candidate_message(inference_config.templates, candidates)?;
        let max_index = included_indices.len().saturating_sub(1);
        let system = Some(self.prepare_system_message(
            inference_config.templates,
            input.system.as_ref(),
            max_index,
        )?);
        let messages = input
            .messages
            .iter()
            .map(|message| {
                self.inner
                    .prepare_request_message(inference_config.templates, message)
            })
            .chain(std::iter::once(Ok(candidate_message)))
            .collect::<Result<Vec<_>, _>>()?;
        inference_params
            .chat_completion
            .backfill_with_variant_params(
                self.inner.temperature,
                self.inner.max_tokens,
                self.inner.seed,
                self.inner.top_p,
                self.inner.presence_penalty,
                self.inner.frequency_penalty,
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
            extra_body: self.inner.extra_body.clone(),
            variant_extra_headers: self.inner.extra_headers.clone(),
            inference_extra_body: Default::default(),
        };
        let model_inference_request = prepare_model_inference_request(
            messages,
            system,
            function,
            inference_config,
            false,
            inference_params,
            self.inner.json_mode,
            extra_body,
        )?;
        Ok((model_inference_request, included_indices))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use reqwest::Client;
    use uuid::Uuid;

    use crate::{
        cache::{CacheEnabledMode, CacheOptions},
        clickhouse::ClickHouseConnectionInfo,
        endpoints::inference::{InferenceCredentials, InferenceIds},
        function::{FunctionConfigChat, FunctionConfigJson},
        inference::{
            providers::dummy::DummyProvider,
            types::{
                ChatInferenceResult, FinishReason, JsonInferenceOutput, JsonInferenceResult,
                Latency, ModelInferenceResponseWithMetadata,
            },
        },
        jsonschema_util::JSONSchemaFromPath,
        minijinja_util::tests::get_test_template_config,
        model::{ModelConfig, ModelProvider, ProviderConfig},
        tool::{ToolCallConfig, ToolChoice},
    };

    use super::*;

    #[test]
    fn test_prepare_system_message() {
        let templates = get_test_template_config();

        // Test without templates, string message
        let fuser_config = FuserConfig {
            inner: ChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                ..Default::default()
            },
        };
        let input_message = Value::String("You are a helpful assistant.".to_string());
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
            inner: ChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                ..Default::default()
            },
        };
        let input_message = json!({"message": "You are a helpful assistant."});
        let max_index = 3;
        let result =
            fuser_config.prepare_system_message(&templates, Some(&input_message), max_index);
        assert!(result.is_err());
        let prepared_message = result.unwrap_err();
        assert_eq!(
        prepared_message,
        ErrorDetails::InvalidMessage { message: "System message content {\"message\":\"You are a helpful assistant.\"} is not a string but there is no variant template".to_string() }.into()
        );

        // Test without templates, no message
        let fuser_config = FuserConfig {
            inner: ChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                ..Default::default()
            },
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

        let fuser_config = FuserConfig {
            inner: ChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                system_template: Some(PathWithContents {
                    path: system_template_name.into(),
                    contents: "".to_string(),
                }),
                ..Default::default()
            },
        };

        let max_index = 6;
        let input_message = serde_json::json!({"assistant_name": "ChatGPT"});
        let result =
            fuser_config.prepare_system_message(&templates, Some(&input_message), max_index);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
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

        let fuser_config = FuserConfig {
            inner: ChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                system_template: Some(PathWithContents {
                    path: system_template_name.into(),
                    contents: "".to_string(),
                }),
                ..Default::default()
            },
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
        let templates = get_test_template_config();

        // Create an FuserConfig
        let fuser_config = FuserConfig {
            inner: ChatCompletionConfig {
                model: "dummy".into(),
                weight: Some(1.0),
                ..Default::default()
            },
        };

        // Prepare some candidate InferenceResults
        let model_inference_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            output: vec!["Candidate answer 1".to_string().into()],
            system: None,
            input_messages: vec![],
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Example response\"}".to_string(),
            usage: Usage {
                input_tokens: 50,
                output_tokens: 100,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(500),
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
                Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                },
                vec![model_inference_response],
                None,
                InferenceParams::default(),
                None,
            )
            .await,
        );

        let model_inference_response2 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 201u64,
            output: vec!["Candidate answer 2".to_string().into()],
            system: None,
            input_messages: vec![],
            raw_request: "{\"prompt\": \"Example prompt 2\"}".to_string(),
            raw_response: "{\"response\": \"Example response 2\"}".to_string(),
            usage: Usage {
                input_tokens: 15,
                output_tokens: 25,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(550),
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
                Usage {
                    input_tokens: 15,
                    output_tokens: 25,
                },
                vec![model_inference_response2],
                None,
                InferenceParams::default(),
                None,
            )
            .await,
        );

        let candidates = vec![candidate1, candidate2];

        // Call prepare_candidate_message
        let result = fuser_config.prepare_candidate_message(&templates, &candidates);
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
        let templates = get_test_template_config();

        // Create a FuserConfig
        let fuser_config = FuserConfig {
            inner: ChatCompletionConfig {
                model: "dummy_json".into(),
                weight: Some(1.0),
                ..Default::default()
            },
        };

        // Prepare some candidate InferenceResults - some valid, some malformed
        let model_inference_response_valid = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            output: vec!["{\"response\": \"Valid JSON response\"}".to_string().into()],
            system: None,
            input_messages: vec![],
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Valid JSON response\"}".to_string(),
            usage: Usage {
                input_tokens: 50,
                output_tokens: 100,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(500),
            },
            model_provider_name: "ExampleProvider".into(),
            model_name: "ExampleModel".into(),
            finish_reason: Some(FinishReason::Stop),
            cached: false,
        };

        let candidate1 = InferenceResult::Json(JsonInferenceResult::new(
            Uuid::now_v7(),
            "{\"response\": \"Valid JSON response\"}".to_string(),
            Some(json!({"response": "Valid JSON response"})),
            Usage {
                input_tokens: 10,
                output_tokens: 20,
            },
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
            input_messages: vec![],
            raw_request: "{\"prompt\": \"Example prompt 2\"}".to_string(),
            raw_response: "{\"response\": \"Malformed JSON response\"".to_string(), // malformed
            usage: Usage {
                input_tokens: 15,
                output_tokens: 25,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(550),
            },
            model_provider_name: "ExampleProvider2".into(),
            model_name: "ExampleModel2".into(),
            finish_reason: Some(FinishReason::Stop),
            cached: false,
        };

        let candidate2 = InferenceResult::Json(JsonInferenceResult::new(
            Uuid::now_v7(),
            "{\"oops: \"Malformed JSON response\"".to_string(),
            None, // malformed
            Usage {
                input_tokens: 15,
                output_tokens: 25,
            },
            vec![model_inference_response_malformed],
            json!({"type": "object", "properties": {"response": {"type": "string"}}}),
            InferenceParams::default(),
            None,
        ));

        let candidates = vec![candidate1, candidate2];

        // Call prepare_candidate_message
        let result = fuser_config.prepare_candidate_message(&templates, &candidates);
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
            inner: ChatCompletionConfig {
                model: "json".into(),
                ..Default::default()
            },
        };
        let mixture_of_n_variant = MixtureOfNConfig {
            weight: Some(1.0),
            timeout_s: 10.0,
            candidates: vec![],
            fuser: fuser_config,
        };

        let templates = get_test_template_config();
        let json_function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&json!({})).unwrap(),
            implicit_tool_call_config: ToolCallConfig::default(),
        });
        // Prepare some candidate InferenceResults
        let model_inference_response0 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            output: vec!["Candidate answer 0".to_string().into()],
            system: None,
            input_messages: vec![],
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Example response\"}".to_string(),
            usage: Usage {
                input_tokens: 50,
                output_tokens: 100,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(500),
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
                Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                },
                vec![model_inference_response0],
                None,
                InferenceParams::default(),
                None,
            )
            .await,
        );

        let model_inference_response1 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 201u64,
            output: vec!["Candidate answer 1".to_string().into()],
            system: None,
            input_messages: vec![],
            raw_request: "{\"prompt\": \"Example prompt 1\"}".to_string(),
            raw_response: "{\"response\": \"Example response 1\"}".to_string(),
            usage: Usage {
                input_tokens: 15,
                output_tokens: 25,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(550),
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
                Usage {
                    input_tokens: 15,
                    output_tokens: 25,
                },
                vec![model_inference_response1],
                None,
                InferenceParams::default(),
                None,
            )
            .await,
        );
        let candidates = vec![candidate0, candidate1];
        let models = ModelTable::try_from(HashMap::from([(
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
                    },
                )]),
            },
        )]))
        .expect("Failed to create model table");
        let client = Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let api_keys = InferenceCredentials::default();
        let inference_clients = InferenceClients {
            http_client: &client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };
        let input = ResolvedInput {
            system: None,
            messages: vec![],
        };
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
            function_name: "",
            variant_name: Some(""),
            extra_body: Default::default(),
            extra_cache_key: None,
        };

        let fused = mixture_of_n_variant
            .fuse_candidates(
                &input,
                &json_function_config,
                &models,
                &inference_config,
                &inference_clients,
                candidates.clone(),
            )
            .await
            .expect("Failed to select best candidate");

        let expected_usage = Usage {
            input_tokens: 35,
            output_tokens: 55,
        };
        let expected_content = JsonInferenceOutput {
            raw: "{\"answer\":\"Hello\"}".to_string(),
            parsed: Some(json!({"answer": "Hello"})),
        };
        match fused {
            InferenceResult::Json(fused) => {
                assert_eq!(fused.usage, expected_usage);
                assert_eq!(fused.output, expected_content);
                assert_eq!(fused.model_inference_results.len(), 3);
            }
            _ => {
                panic!("Expected a Chat inference result");
            }
        }
        // Set up fuser with a provider that fails
        let fuser_config = FuserConfig {
            inner: ChatCompletionConfig {
                model: "error".into(),
                ..Default::default()
            },
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
                        },
                    )]),
                },
            );
            ModelTable::try_from(map).expect("Failed to create model table")
        };
        let input = ResolvedInput {
            system: None,
            messages: vec![],
        };

        let result = mixture_of_n_variant
            .fuse_candidates(
                &input,
                &json_function_config,
                &models,
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
                assert_eq!(chat_choice.model_inference_results.len(), 2);
            }
            _ => {
                panic!("Expected a Chat inference result");
            }
        }
        // Depending on implementation, you might check which candidate was selected

        // Set up evaluator with a provider that returns invalid JSON
        let fuser_config = FuserConfig {
            inner: ChatCompletionConfig {
                model: "regular".into(),
                ..Default::default()
            },
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
                        },
                    )]),
                },
            );
            ModelTable::try_from(map).expect("Failed to create model table")
        };
        let input = ResolvedInput {
            system: None,
            messages: vec![],
        };
        let chat_function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
        });

        let result = mixture_of_n_variant
            .fuse_candidates(
                &input,
                &chat_function_config,
                &models,
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
            _ => {
                panic!("Expected a Chat inference result");
            }
        }
        // Test case: No answer choices (should return an error)
        let empty_candidates = vec![];
        let result = mixture_of_n_variant
            .fuse_candidates(
                &input,
                &json_function_config,
                &models,
                &inference_config,
                &inference_clients,
                empty_candidates.clone(),
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
}
