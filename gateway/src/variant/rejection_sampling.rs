use std::future::Future;
use std::pin::Pin;
use std::{collections::HashMap, path::PathBuf};

use futures::future::join_all;
use rand::Rng;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::time::{timeout, Duration};

use crate::inference::types::{
    ContentBlock, FunctionType, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    ModelInferenceResponseWithMetadata, RequestMessage, Role,
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
use lazy_static::lazy_static;

use super::{InferenceConfig, ModelUsedInfo, Variant};

#[derive(Debug, Deserialize)]
pub struct RejectionSamplingConfig {
    pub weight: f64,
    pub timeout_s: Option<f64>,
    pub candidates: Vec<String>,
    pub evaluator: EvaluatorConfig,
}

#[derive(Debug, Deserialize)]
pub struct EvaluatorConfig {
    #[serde(flatten)]
    inner: ChatCompletionConfig,
}

lazy_static! {
    static ref EVALUATOR_OUTPUT_SCHEMA: JSONSchemaFromPath = {
        JSONSchemaFromPath::from_value(&json!({
            "type": "object",
            "properties": {
                "thinking": { "type": "string" },
                "answer_choice": { "type": "integer" }
            },
            "required": ["thinking", "answer_choice"],
            "additionalProperties": false
        }))
        .unwrap()
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
        inference_params: &mut InferenceParams,
    ) -> Result<InferenceResult<'a>, Error> {
        let candidate_inference_results = self
            .infer_candidates(
                input,
                models,
                function,
                inference_config,
                client,
                inference_params,
            )
            .await?;
        let best_candidate = self
            .select_best_candidate(
                input,
                models,
                inference_config,
                client,
                inference_params,
                candidate_inference_results,
            )
            .await?;

        // TODO(Viraj): do the bookkeeping for all ModelInferences and such
        todo!()
    }

    async fn infer_stream<'request>(
        &'static self,
        _input: &Input,
        _models: &'static HashMap<String, ModelConfig>,
        _function: &'static FunctionConfig,
        _inference_config: &'request InferenceConfig<'request>,
        _client: &'request Client,
        _inference_params: &mut InferenceParams,
    ) -> Result<
        (
            InferenceResultChunk,
            InferenceResultStream,
            ModelUsedInfo<'static>,
        ),
        Error,
    > {
        return Err(Error::InvalidRequest {
            message: "Rejection sampling variants do not support streaming inference.".to_string(),
        });
    }

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        todo!()
    }

    fn get_all_template_paths(&self) -> Vec<&PathBuf> {
        todo!()
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
        inference_params: &mut InferenceParams,
    ) -> Result<Vec<InferenceResult<'a>>, Error> {
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
        let mut inference_futures = Vec::new();
        for (candidate_name, candidate_variant) in &candidate_variants {
            let future = candidate_variant.infer(
                input,
                models,
                function,
                inference_config,
                client,
                inference_params, // TODO(Viraj): Change the structure of inference_params to avoid mutable borrows
            );

            let future: Pin<Box<dyn Future<Output = Result<InferenceResult, Error>>>> =
                if let Some(timeout_s) = self.timeout_s {
                    Box::pin(async move {
                        timeout(Duration::from_secs_f64(timeout_s), future)
                            .await
                            .map_err(|_| Error::InferenceTimeout {
                                variant_name: candidate_name.clone(),
                            })?
                    })
                } else {
                    Box::pin(future)
                };

            inference_futures.push(future);
        }

        let inference_results: Vec<_> = join_all(inference_futures).await.into_iter().collect();
        let successful_results: Vec<InferenceResult> = inference_results
            .into_iter()
            .filter_map(|result| match result {
                Ok(res) => Some(res),
                Err(e) => {
                    e.log();
                    None
                }
            })
            .collect();

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
        inference_params: &mut InferenceParams,
        candidates: Vec<InferenceResult<'a>>,
    ) -> Result<ModelInferenceResponseWithMetadata<'a>, Error> {
        let choice_idx = match inner_select_best_candidate(
            &self.evaluator,
            input,
            models,
            inference_config,
            client,
            inference_params,
            &candidates,
        )
        .await
        {
            Ok(choice) => choice,
            Err(e) => {
                e.log();
                rand::thread_rng().gen_range(0..candidates.len())
            }
        };

        todo!()
    }
}

/// Actually does the work of selecting the best candidate for rejection sampling.
/// We factor this into a separate function so that we can gracefully handle any error here in the caller.
async fn inner_select_best_candidate(
    evaluator: &EvaluatorConfig,
    input: &Input,
    models: &HashMap<String, ModelConfig>,
    inference_config: &InferenceConfig<'_>,
    client: &Client,
    inference_params: &mut InferenceParams,
    candidates: &Vec<InferenceResult<'_>>,
) -> Result<usize, Error> {
    let inference_request =
        evaluator.prepare_request(input, inference_config, candidates, inference_params)?;
    let model_config = models
        .get(&evaluator.inner.model)
        .ok_or(Error::UnknownModel {
            name: evaluator.inner.model.clone(),
        })?;
    let model_inference_response = model_config.infer(&inference_request, client).await?;
    let text_content = model_inference_response
        .content
        .into_iter()
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
    Ok(answer_choice as usize)
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
            output_schema: Some(&EVALUATOR_OUTPUT_SCHEMA.value),
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
