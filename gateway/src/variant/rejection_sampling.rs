use std::future::Future;
use std::pin::Pin;
use std::{collections::HashMap, path::PathBuf};

use futures::future::join_all;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::time::{timeout, Duration};

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
    pub timeout_s: Option<f64>,
    pub candidates: Vec<String>,
    pub evaluator: EvaluatorConfig,
}

#[derive(Debug, Deserialize)]
pub struct EvaluatorConfig {
    #[serde(flatten)]
    inner: ChatCompletionConfig,
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
        Ok(self
            .select_best_candidate(
                input,
                models,
                function,
                inference_config,
                client,
                inference_params,
                candidate_inference_results,
            )
            .await?)
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
        // Start of Selection
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
                inference_params, // TODO: Change the structure of inference_params to avoid mutable borrows
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

    async fn select_best_candidate<'a, 'request>(
        &self,
        input: &Input,
        models: &'a HashMap<String, ModelConfig>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        inference_params: &mut InferenceParams,
        candidates: Vec<InferenceResult<'a>>,
    ) -> Result<InferenceResult<'a>, Error> {
        let system_message = self
            .evaluator
            .prepare_system_message(inference_config.templates, input.system.as_ref())?;
        let mut messages = input
            .messages
            .iter()
            .map(|message| {
                self.evaluator
                    .inner
                    .prepare_request_message(inference_config.templates, message)
            })
            .collect::<Result<Vec<_>, _>>()?;
        todo!()
    }
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
        candidates: Vec<InferenceResult>,
    ) -> Result<String, Error> {
        let template_context = json!({
            "candidates": candidates,
        });
        templates.template_message("rejection_sampling_evaluator_candidate", &template_context)
    }
}
