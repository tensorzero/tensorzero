use std::{collections::HashMap, path::PathBuf};

use futures::future::join_all;
use lazy_static::lazy_static;
use rand::Rng;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::time::{timeout, Duration};

use crate::inference::types::{
    ContentBlock, FunctionType, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    ModelInferenceResponseWithMetadata, RequestMessage, Role, Usage,
};
use crate::jsonschema_util::JSONSchemaFromPath;
use crate::{
    endpoints::inference::InferenceParams,
    error::Error,
    function::FunctionConfig,
    inference::types::{InferenceResult, InferenceResultChunk, InferenceResultStream, Input},
    minijinja_util::TemplateConfig,
    model::ModelConfig,
    variant::chat_completion::ChatCompletionConfig,
};

use super::{InferenceConfig, ModelUsedInfo, Variant};

#[derive(Debug, Deserialize)]
pub struct RejectionSamplingConfig {
    pub weight: f64,
    #[serde(default = "default_timeout")]
    pub timeout_s: f64,
    pub candidates: Vec<String>,
    pub evaluator: EvaluatorConfig,
}

fn default_timeout() -> f64 {
    300.0
}

#[derive(Debug, Deserialize)]
pub struct EvaluatorConfig {
    #[serde(flatten)]
    inner: ChatCompletionConfig,
}

lazy_static! {
    static ref EVALUATOR_OUTPUT_SCHEMA: JSONSchemaFromPath = {
        #[allow(clippy::expect_used)]
        JSONSchemaFromPath::from_value(&json!({
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
}

impl Variant for RejectionSamplingConfig {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'a HashMap<String, ModelConfig>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        _inference_params: InferenceParams,
    ) -> Result<InferenceResult<'a>, Error> {
        let candidate_inference_results = self
            .infer_candidates(input, models, function, inference_config, client)
            .await?;
        self.select_best_candidate(
            input,
            models,
            inference_config,
            client,
            candidate_inference_results,
        )
        .await
    }

    async fn infer_stream<'request>(
        &'static self,
        _input: &Input,
        _models: &'static HashMap<String, ModelConfig>,
        _function: &'static FunctionConfig,
        _inference_config: &'request InferenceConfig<'request>,
        _client: &'request Client,
        _inference_params: InferenceParams,
    ) -> Result<
        (
            InferenceResultChunk,
            InferenceResultStream,
            ModelUsedInfo<'static>,
        ),
        Error,
    > {
        Err(Error::InvalidRequest {
            message: "Rejection sampling variants do not support streaming inference.".to_string(),
        })
    }

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        // Validate each candidate variant
        for candidate in &self.candidates {
            let variant = function
                .variants()
                .get(candidate)
                .ok_or(Error::UnknownCandidate {
                    name: candidate.to_string(),
                })?;
            variant
                .validate(function, models, templates, function_name, candidate)
                .map_err(|e| Error::InvalidCandidate {
                    variant_name: variant_name.to_string(),
                    message: e.to_string(),
                })?;
        }
        // Validate the evaluator variant
        self.evaluator
            .inner
            .validate(function, models, templates, function_name, variant_name)?;
        Ok(())
    }

    // We do not return templates for the candidates, as they are required to be variants in the same function
    // and will therefore also have the same templates.
    // We only return templates for the evaluator variant.
    fn get_all_template_paths(&self) -> Vec<&PathBuf> {
        self.evaluator.inner.get_all_template_paths()
    }
}

impl RejectionSamplingConfig {
    /// Infer each candidate variant concurrently and return the results.
    async fn infer_candidates<'a, 'request>(
        &self,
        input: &Input,
        models: &'a HashMap<String, ModelConfig>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
    ) -> Result<Vec<InferenceResult<'a>>, Error> {
        // Get all the variants we are going to infer
        let candidate_variants = self
            .candidates
            .iter()
            .map(|candidate| {
                let variant =
                    function
                        .variants()
                        .get(candidate)
                        .ok_or(Error::UnknownCandidate {
                            name: candidate.to_string(),
                        })?;
                Ok((candidate.to_string(), variant))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Start the inference tasks (we keep the names around for logging)
        let mut inference_futures = Vec::new();
        for (candidate_name, candidate_variant) in &candidate_variants {
            inference_futures.push((
                candidate_name.clone(),
                timeout(
                    Duration::from_secs_f64(self.timeout_s),
                    candidate_variant.infer(
                        input,
                        models,
                        function,
                        inference_config,
                        client,
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
                Ok(inner_result) => match inner_result {
                    Ok(res) => successful_results.push(res),
                    Err(e) => {
                        e.log();
                    }
                },
                Err(_timeout_error) => {
                    // Map the Tokio timeout error to our own TimeoutError type
                    let mapped_timeout_error = Error::InferenceTimeout {
                        variant_name: candidate_name.clone(),
                    };
                    // Log the mapped timeout error
                    mapped_timeout_error.log();
                }
            }
        }

        Ok(successful_results)
    }

    /// Gets the best candidate using the evaluator config.
    /// If at any point the evaluator fails to return a valid response,
    /// we randomly select one of the candidates.
    async fn select_best_candidate<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'a HashMap<String, ModelConfig>,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        candidates: Vec<InferenceResult<'a>>,
    ) -> Result<InferenceResult<'a>, Error> {
        if candidates.is_empty() {
            return Err(Error::Inference {
                message: "No candidates to select from in rejection sampling".to_string(),
            });
        }
        if candidates.len() == 1 {
            let mut candidates = candidates;
            return candidates.pop().ok_or_else(|| Error::Inference {
                message: "Expected one candidate but found none".to_string(),
            });
        }
        // If the evaluator fails, we randomly select one of the candidates
        // As long as the evaluator returns an inference result, we want to include it in the observability
        let (selection_idx, inference_result) = match inner_select_best_candidate(
            &self.evaluator,
            input,
            models,
            inference_config,
            client,
            &candidates,
        )
        .await
        {
            Ok((idx_opt, inf_result)) => (
                idx_opt.unwrap_or_else(|| rand::thread_rng().gen_range(0..candidates.len())),
                Some(inf_result),
            ),
            Err(e) => {
                e.log();
                (rand::thread_rng().gen_range(0..candidates.len()), None)
            }
        };

        // Safely remove the selected candidate without panicking
        let mut total_usage: Usage = candidates.iter().map(|c| c.usage()).sum();
        let mut candidates = candidates;
        let mut selected_candidate = if selection_idx < candidates.len() {
            candidates.swap_remove(selection_idx)
        } else {
            return Err(Error::Inference {
                message: "The index chosen by the evaluator is out of bounds (should never happen)"
                    .to_string(),
            });
        };
        if let Some(inference_result) = &inference_result {
            total_usage.input_tokens += inference_result.usage.input_tokens;
            total_usage.output_tokens += inference_result.usage.output_tokens;
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

/// Attempts to select the best candidate for rejection sampling.
/// If this function returns an error or the index is None, we will randomly select one
/// of the candidates in the outer function.
/// If a model inference actually occurs, we return None and the model inference result instead of Err() so
/// that we can still observe the model inference result in ClickHouse.
///
/// Here are the steps in the function:
///  * Prepare the request for the evaluator variant.
///  * Infer the request using the model specified in the evaluator config.
///  * Parse the output of the evaluator.
///  * Check if the index is out of bounds.
///  * Return the index and the model inference result.
async fn inner_select_best_candidate<'a, 'request>(
    evaluator: &'a EvaluatorConfig,
    input: &'request Input,
    models: &'a HashMap<String, ModelConfig>,
    inference_config: &'request InferenceConfig<'request>,
    client: &'request Client,
    candidates: &Vec<InferenceResult<'request>>,
) -> Result<(Option<usize>, ModelInferenceResponseWithMetadata<'a>), Error> {
    let inference_request = evaluator.prepare_request(
        input,
        inference_config,
        candidates,
        &mut InferenceParams::default(),
    )?;
    let model_config = models
        .get(&evaluator.inner.model)
        .ok_or(Error::UnknownModel {
            name: evaluator.inner.model.clone(),
        })?;
    let model_inference_response = model_config.infer(&inference_request, client).await?;
    let model_inference_result =
        ModelInferenceResponseWithMetadata::new(model_inference_response, &evaluator.inner.model);
    let text_content = match model_inference_result
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::Text(text) => Some(text),
            _ => None,
        }) {
        Some(text) => text,
        None => return Ok((None, model_inference_result)),
    };
    let parsed_output = match serde_json::from_str::<Value>(&text_content.text) {
        Ok(value) => value,
        Err(_) => {
            return Ok((None, model_inference_result));
        }
    };
    let answer_choice = match parsed_output.get("answer_choice") {
        Some(val) => match val.as_u64() {
            Some(num) => num as usize,
            None => return Ok((None, model_inference_result)),
        },
        None => return Ok((None, model_inference_result)),
    };
    if answer_choice >= candidates.len() {
        let err = Error::Inference {
            message: format!(
                "The index chosen by the evaluator is out of bounds: {} >= {}",
                answer_choice,
                candidates.len()
            ),
        };
        err.log();
        return Ok((None, model_inference_result));
    }
    Ok((Some(answer_choice as usize), model_inference_result))
}

impl EvaluatorConfig {
    /// Prepares the system message for the evaluator variant.
    /// We use the system_template of the evaluator variant to generate a system message as if we
    /// were using the evaluator variant directly to solve the problem.
    /// Then, we template that system message into a broader set of instructions that includes
    /// information about what the evaluator will be asked to do (choose a candidate).
    fn prepare_system_message(
        &self,
        templates: &TemplateConfig,
        system: Option<&Value>,
    ) -> Result<String, Error> {
        let inner_system_message = self.inner.prepare_system_message(templates, system)?;
        let template_context = match inner_system_message {
            Some(inner_system_message) => json!({"inner_system_message": inner_system_message}),
            None => json!({}),
        };
        templates.template_message("t0:rejection_sampling_evaluator_system", &template_context)
    }

    /// Prepares the final candidate message for the evaluator variant.
    /// We include each candidate's output in the final message to the evaluator by templating
    /// them into our hardcoded template.
    /// For chat functions we serialize the content blocks to a string and for json functions
    /// we use the raw output from the json field.
    fn prepare_candidate_message(
        &self,
        templates: &TemplateConfig,
        candidates: &Vec<InferenceResult>,
    ) -> Result<RequestMessage, Error> {
        let mut candidate_outputs = Vec::new();
        for candidate in candidates {
            match candidate {
                InferenceResult::Chat(chat_result) => {
                    candidate_outputs.push(serde_json::to_string(&chat_result.content).map_err(
                        |e| Error::Inference {
                            message: format!("Error converting chat result to string: {e}"),
                        },
                    )?);
                }
                InferenceResult::Json(json_result) => {
                    candidate_outputs.push(json_result.output.raw.clone());
                }
            }
        }
        let template_context = json!({
            "candidates": candidates,
        });
        let message_text = templates.template_message(
            "t0:rejection_sampling_evaluator_candidates",
            &template_context,
        )?;
        Ok(RequestMessage {
            role: Role::User,
            content: vec![message_text.into()],
        })
    }

    /// Prepares the request for the evaluator variant.
    /// We use the prepare_system_message and prepare_candidate_message functions to generate
    /// the system and candidate messages for the evaluator which take candidate selection into account.
    ///
    /// We also enforce the output schema of the evalutator variant, which is used to force the model
    /// to choose an answer choice that is valid.
    fn prepare_request(
        &self,
        input: &Input,
        inference_config: &InferenceConfig,
        candidates: &Vec<InferenceResult>,
        inference_params: &mut InferenceParams,
    ) -> Result<ModelInferenceRequest, Error> {
        let system =
            Some(self.prepare_system_message(inference_config.templates, input.system.as_ref())?);
        let messages = input
            .messages
            .iter()
            .map(|message| {
                self.inner
                    .prepare_request_message(inference_config.templates, message)
            })
            .chain(std::iter::once(self.prepare_candidate_message(
                inference_config.templates,
                candidates,
            )))
            .collect::<Result<Vec<_>, _>>()?;
        inference_params
            .chat_completion
            .backfill_with_variant_params(
                self.inner.temperature,
                self.inner.max_tokens,
                self.inner.seed,
            );
        Ok(ModelInferenceRequest {
            messages,
            system,
            tool_config: None,
            temperature: inference_params.chat_completion.temperature,
            max_tokens: inference_params.chat_completion.max_tokens,
            seed: inference_params.chat_completion.seed,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Strict,
            function_type: FunctionType::Json,
            output_schema: Some(EVALUATOR_OUTPUT_SCHEMA.value),
        })
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::{
        inference::{
            providers::dummy::DummyProvider,
            types::{ChatInferenceResult, Latency},
        },
        minijinja_util::tests::get_test_template_config,
        model::ProviderConfig,
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

    #[test]
    fn test_prepare_system_message() {
        let templates = get_test_template_config();

        // Test without templates, string message
        let evaluator_config = EvaluatorConfig {
            inner: ChatCompletionConfig {
                model: "dummy".to_string(),
                weight: 1.0,
                ..Default::default()
            },
        };
        let input_message = Value::String("You are a helpful assistant.".to_string());
        let result = evaluator_config.prepare_system_message(&templates, Some(&input_message));
        let prepared_message = result.unwrap();
        let expected_message = templates
            .template_message(
                "t0:rejection_sampling_evaluator_system",
                &json!({"inner_system_message": "You are a helpful assistant."}),
            )
            .unwrap();
        assert_eq!(prepared_message, expected_message);

        // Test without templates, object message
        let evaluator_config = EvaluatorConfig {
            inner: ChatCompletionConfig {
                model: "dummy".to_string(),
                weight: 1.0,
                ..Default::default()
            },
        };
        let input_message = json!({"message": "You are a helpful assistant."});
        let result = evaluator_config.prepare_system_message(&templates, Some(&input_message));
        assert!(result.is_err());
        let prepared_message = result.unwrap_err();
        assert_eq!(
        prepared_message,
        Error::InvalidMessage { message: "System message content {\"message\":\"You are a helpful assistant.\"} is not a string but there is no variant template".to_string() }
        );

        // Test without templates, no message
        let evaluator_config = EvaluatorConfig {
            inner: ChatCompletionConfig {
                model: "dummy".to_string(),
                weight: 1.0,
                ..Default::default()
            },
        };
        let result = evaluator_config.prepare_system_message(&templates, None);
        let expected_message = templates
            .template_message("t0:rejection_sampling_evaluator_system", &json!({}))
            .unwrap();
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(prepared_message, expected_message);

        // Test with templates that need new info
        let system_template_name = "system";

        let evaluator_config = EvaluatorConfig {
            inner: ChatCompletionConfig {
                model: "dummy".to_string(),
                weight: 1.0,
                system_template: Some(system_template_name.into()),
                ..Default::default()
            },
        };

        let input_message = serde_json::json!({"assistant_name": "ChatGPT"});
        let result = evaluator_config.prepare_system_message(&templates, Some(&input_message));
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        let inner_system_message = templates
            .template_message(system_template_name, &json!({"assistant_name": "ChatGPT"}))
            .unwrap();
        let expected_message = templates
            .template_message(
                "t0:rejection_sampling_evaluator_system",
                &json!({"inner_system_message": inner_system_message}),
            )
            .unwrap();
        assert_eq!(prepared_message, expected_message);

        // Test with template that is complete as is (string)
        let system_template_name = "system_filled";

        let evaluator_config = EvaluatorConfig {
            inner: ChatCompletionConfig {
                model: "dummy".to_string(),
                weight: 1.0,
                system_template: Some(system_template_name.into()),
                ..Default::default()
            },
        };

        let result = evaluator_config.prepare_system_message(&templates, None);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        let inner_system_message = templates
            .template_message(system_template_name, &json!({}))
            .unwrap();
        let expected_message = templates
            .template_message(
                "t0:rejection_sampling_evaluator_system",
                &json!({"inner_system_message": inner_system_message}),
            )
            .unwrap();
        assert_eq!(prepared_message, expected_message);
    }

    #[tokio::test]
    async fn test_prepare_candidate_message() {
        let templates = get_test_template_config();

        // Create an EvaluatorConfig
        let evaluator_config = EvaluatorConfig {
            inner: ChatCompletionConfig {
                model: "dummy".to_string(),
                weight: 1.0,
                ..Default::default()
            },
        };

        // Prepare some candidate InferenceResults
        let model_inference_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            content: vec!["Candidate answer 1".to_string().into()],
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Example response\"}".to_string(),
            usage: Usage {
                input_tokens: 50,
                output_tokens: 100,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(500),
            },
            model_provider_name: "ExampleProvider",
            model_name: "ExampleModel",
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
            )
            .await,
        );

        let model_inference_response2 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 201u64,
            content: vec!["Candidate answer 2".to_string().into()],
            raw_request: "{\"prompt\": \"Example prompt 2\"}".to_string(),
            raw_response: "{\"response\": \"Example response 2\"}".to_string(),
            usage: Usage {
                input_tokens: 15,
                output_tokens: 25,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(550),
            },
            model_provider_name: "ExampleProvider2",
            model_name: "ExampleModel2",
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
            )
            .await,
        );

        let candidates = vec![candidate1, candidate2];

        // Call prepare_candidate_message
        let result = evaluator_config.prepare_candidate_message(&templates, &candidates);
        assert!(result.is_ok());
        let request_message = result.unwrap();

        // Expected message
        let template_context = json!({
            "candidates": candidates,
        });
        let expected_message_text = templates
            .template_message(
                "t0:rejection_sampling_evaluator_candidates",
                &template_context,
            )
            .unwrap();

        // Now check that the request_message has the expected role and content
        assert_eq!(request_message.role, Role::User);
        assert_eq!(request_message.content, vec![expected_message_text.into()]);
    }

    #[tokio::test]
    async fn test_select_best_candidate() {
        // Set up evaluator with a provider that returns a valid answer_choice
        let evaluator_config = EvaluatorConfig {
            inner: ChatCompletionConfig {
                model: "rejection_sampling_1".to_string(),
                ..Default::default()
            },
        };
        let rejection_sampling_variant = RejectionSamplingConfig {
            weight: 1.0,
            timeout_s: 10.0,
            candidates: vec![],
            evaluator: evaluator_config,
        };

        let templates = get_test_template_config();
        // Prepare some candidate InferenceResults
        let model_inference_response0 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 200u64,
            content: vec!["Candidate answer 0".to_string().into()],
            raw_request: "{\"prompt\": \"Example prompt\"}".to_string(),
            raw_response: "{\"response\": \"Example response\"}".to_string(),
            usage: Usage {
                input_tokens: 50,
                output_tokens: 100,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(500),
            },
            model_provider_name: "ExampleProvider",
            model_name: "ExampleModel",
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
            )
            .await,
        );

        let model_inference_response1 = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: 201u64,
            content: vec!["Candidate answer 1".to_string().into()],
            raw_request: "{\"prompt\": \"Example prompt 1\"}".to_string(),
            raw_response: "{\"response\": \"Example response 1\"}".to_string(),
            usage: Usage {
                input_tokens: 15,
                output_tokens: 25,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(550),
            },
            model_provider_name: "ExampleProvider1",
            model_name: "ExampleModel1",
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
            )
            .await,
        );
        let candidates = vec![candidate0, candidate1];
        let models = HashMap::from([(
            "rejection_sampling_1".to_string(),
            ModelConfig {
                routing: vec!["rejection_sampling_1".to_string()],
                providers: HashMap::from([(
                    "rejection_sampling_1".to_string(),
                    ProviderConfig::Dummy(DummyProvider {
                        model_name: "rejection_sampling_1".to_string(),
                    }),
                )]),
            },
        )]);
        let client = Client::new();
        let input = Input {
            system: None,
            messages: vec![],
        };
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };

        let selected = rejection_sampling_variant
            .select_best_candidate(
                &input,
                &models,
                &inference_config,
                &client,
                candidates.clone(),
            )
            .await
            .expect("Failed to select best candidate");

        // Expect the second candidate to be selected (index 1)
        // based on "answer": 1 in rejection_sampling_1
        let expected_id = inference_id1;
        let expected_usage = Usage {
            input_tokens: 15,
            output_tokens: 25,
        };
        let expected_content = vec!["Candidate answer 1".to_string().into()];
        match selected {
            InferenceResult::Chat(selected) => {
                assert_eq!(selected.inference_id, expected_id);
                assert_eq!(selected.usage, expected_usage);
                assert_eq!(selected.content, expected_content);
                assert_eq!(selected.model_inference_results.len(), 3);
            }
            _ => {
                panic!("Expected a Chat inference result");
            }
        }
        // Set up evaluator with a provider that fails
        let evaluator_config = EvaluatorConfig {
            inner: ChatCompletionConfig {
                model: "error".to_string(),
                ..Default::default()
            },
        };
        let rejection_sampling_variant = RejectionSamplingConfig {
            weight: 1.0,
            timeout_s: 10.0,
            candidates: vec![],
            evaluator: evaluator_config,
        };

        let models = {
            let mut map = HashMap::new();
            map.insert(
                "error".to_string(),
                ModelConfig {
                    routing: vec!["error".to_string()],
                    providers: HashMap::from([(
                        "error".to_string(),
                        ProviderConfig::Dummy(DummyProvider {
                            model_name: "error".to_string(),
                        }),
                    )]),
                },
            );
            map
        };
        let client = Client::new();
        let input = Input {
            system: None,
            messages: vec![],
        };

        let result = rejection_sampling_variant
            .select_best_candidate(
                &input,
                &models,
                &inference_config,
                &client,
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
            _ => {
                panic!("Expected a Chat inference result");
            }
        }
        // Depending on implementation, you might check which candidate was selected

        // Set up evaluator with a provider that returns invalid JSON
        let evaluator_config = EvaluatorConfig {
            inner: ChatCompletionConfig {
                model: "regular".to_string(),
                ..Default::default()
            },
        };
        let rejection_sampling_variant = RejectionSamplingConfig {
            weight: 1.0,
            timeout_s: 10.0,
            candidates: vec![],
            evaluator: evaluator_config,
        };

        let models = {
            let mut map = HashMap::new();
            map.insert(
                "regular".to_string(),
                ModelConfig {
                    routing: vec!["regular".to_string()],
                    providers: HashMap::from([(
                        "regular".to_string(),
                        ProviderConfig::Dummy(DummyProvider {
                            model_name: "regular".to_string(),
                        }),
                    )]),
                },
            );
            map
        };
        let input = Input {
            system: None,
            messages: vec![],
        };

        let result = rejection_sampling_variant
            .select_best_candidate(
                &input,
                &models,
                &inference_config,
                &client,
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
        let result = rejection_sampling_variant
            .select_best_candidate(
                &input,
                &models,
                &inference_config,
                &client,
                empty_candidates.clone(),
            )
            .await;
        let err = result.unwrap_err();
        assert_eq!(
            err,
            Error::Inference {
                message: "No candidates to select from in rejection sampling".to_string()
            }
        );

        // Test case: Index returned too large (should return an error)
        let rejection_sampling_big_variant = RejectionSamplingConfig {
            weight: 1.0,
            timeout_s: 10.0,
            candidates: vec![],
            evaluator: EvaluatorConfig {
                inner: ChatCompletionConfig {
                    model: "rejection_sampling_big".to_string(),
                    weight: 1.0,
                    ..Default::default()
                },
            },
        };

        let mut big_models = HashMap::new();
        big_models.insert(
            "rejection_sampling_big".to_string(),
            ModelConfig {
                routing: vec!["rejection_sampling_big".to_string()],
                providers: HashMap::from([(
                    "rejection_sampling_big".to_string(),
                    ProviderConfig::Dummy(DummyProvider {
                        model_name: "rejection_sampling_big".to_string(),
                    }),
                )]),
            },
        );

        let result_big = rejection_sampling_big_variant
            .select_best_candidate(
                &input,
                &big_models,
                &inference_config,
                &client,
                candidates.clone(),
            )
            .await;
        // we gracefully handle the error and return a random candidate
        let _result = result_big.unwrap();
    }
}
