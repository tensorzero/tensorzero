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
            Ok(choice) => {
                let (idx, inference_result) = choice;
                (idx, Some(inference_result))
            }
            Err(e) => {
                e.log();
                let idx = rand::thread_rng().gen_range(0..candidates.len());
                (idx, None)
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

/// Actually does the work of selecting the best candidate for rejection sampling.
/// We factor this into a separate function so that we can gracefully handle any error here in the caller.
async fn inner_select_best_candidate<'a, 'request>(
    evaluator: &'a EvaluatorConfig,
    input: &'request Input,
    models: &'a HashMap<String, ModelConfig>,
    inference_config: &'request InferenceConfig<'request>,
    client: &'request Client,
    candidates: &Vec<InferenceResult<'request>>,
) -> Result<(usize, ModelInferenceResponseWithMetadata<'a>), Error> {
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
    let text_content = model_inference_response
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::Text(text) => Some(text),
            _ => None,
        })
        .ok_or(Error::Inference {
            message: "No valid content blocks found in evaluator response".to_string(),
        })?;
    let parsed_output =
        serde_json::from_str::<Value>(&text_content.text).map_err(|e| Error::OutputParsing {
            message: format!("Failed to parse output from evaluator response {}", e),
            raw_output: text_content.text.clone(),
        })?;
    let answer_choice = parsed_output
        .get("answer_choice")
        .ok_or(Error::OutputParsing {
            message: "Missing answer_choice in evaluator response".to_string(),
            raw_output: text_content.text.clone(),
        })?
        .as_u64()
        .ok_or(Error::OutputParsing {
            message: "answer_choice is not a valid integer".to_string(),
            raw_output: text_content.text.clone(),
        })?;
    let model_inference_result =
        ModelInferenceResponseWithMetadata::new(model_inference_response, &evaluator.inner.model);
    Ok((answer_choice as usize, model_inference_result))
}

impl EvaluatorConfig {
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
        templates.template_message("rejection_sampling_evaluator_system", &template_context)
    }

    fn prepare_candidate_message(
        &self,
        templates: &TemplateConfig,
        candidates: &Vec<InferenceResult>,
    ) -> Result<RequestMessage, Error> {
        let template_context = json!({
            "candidates": candidates,
        });
        let message_text = templates
            .template_message("rejection_sampling_evaluator_candidate", &template_context)?;
        Ok(RequestMessage {
            role: Role::User,
            content: vec![message_text.into()],
        })
    }

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
}
