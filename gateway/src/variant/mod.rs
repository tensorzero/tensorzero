use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::endpoints::inference::InferenceParams;
use crate::error::Error;
use crate::function::FunctionConfig;
use crate::inference::types::{InferenceResultChunk, InferenceResultStream, Input};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::tool::ToolCallConfig;
use crate::{inference::types::InferenceResult, model::ModelConfig};
pub mod chat_completion;
pub mod rejection_sampling;

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum VariantConfig {
    ChatCompletion(chat_completion::ChatCompletionConfig),
    RejectionSampling(rejection_sampling::RejectionSamplingConfig),
}

/// This type is used to determine how to enforce JSON mode for a given variant.
/// Variants represent JSON mode in a slightly more abstract sense than ModelInferenceRequests, as
/// we support coercing tool calls into JSON mode.
/// This is represented as a tool config in the
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JsonMode {
    Off,
    #[default]
    On,
    Strict,
    ImplicitTool,
}

/// Maps to the subset of Config that applies to the current inference request.
/// It doesn't take into account inference-time overrides (e.g. dynamic tools).
pub struct InferenceConfig<'a> {
    pub tool_config: Option<ToolCallConfig>,
    pub templates: &'a TemplateConfig<'a>,
    pub dynamic_output_schema: Option<DynamicJSONSchema>,
}

pub struct ModelUsedInfo<'a> {
    pub model_name: &'a str,
    pub model_provider_name: &'a str,
    pub raw_request: String,
}

pub trait Variant {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'a HashMap<String, ModelConfig>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        inference_params: &mut InferenceParams,
    ) -> Result<InferenceResult<'a>, Error>;

    async fn infer_stream<'request>(
        &'static self,
        input: &Input,
        models: &'static HashMap<String, ModelConfig>,
        function: &'static FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        inference_params: &mut InferenceParams,
    ) -> Result<
        (
            InferenceResultChunk,
            InferenceResultStream,
            ModelUsedInfo<'static>,
        ),
        Error,
    >;

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error>;

    fn get_all_template_paths(&self) -> Vec<&PathBuf>;
}

impl VariantConfig {
    pub fn weight(&self) -> f64 {
        match self {
            VariantConfig::ChatCompletion(params) => params.weight,
            VariantConfig::RejectionSampling(params) => params.weight,
        }
    }
}

impl Variant for VariantConfig {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'a HashMap<String, ModelConfig>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        inference_params: &mut InferenceParams,
    ) -> Result<InferenceResult<'a>, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer(
                        input,
                        models,
                        function,
                        inference_config,
                        client,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::RejectionSampling(params) => {
                params
                    .infer(
                        input,
                        models,
                        function,
                        inference_config,
                        client,
                        inference_params,
                    )
                    .await
            }
        }
    }

    async fn infer_stream<'request>(
        &'static self,
        input: &Input,
        models: &'static HashMap<String, ModelConfig>,
        function: &'static FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        inference_params: &mut InferenceParams,
    ) -> Result<
        (
            InferenceResultChunk,
            InferenceResultStream,
            ModelUsedInfo<'static>,
        ),
        Error,
    > {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer_stream(
                        input,
                        models,
                        function,
                        inference_config,
                        client,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::RejectionSampling(params) => {
                params
                    .infer_stream(
                        input,
                        models,
                        function,
                        inference_config,
                        client,
                        inference_params,
                    )
                    .await
            }
        }
    }

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params.validate(function, models, templates, function_name, variant_name)
            }
            VariantConfig::RejectionSampling(params) => {
                params.validate(function, models, templates, function_name, variant_name)
            }
        }
    }

    fn get_all_template_paths(&self) -> Vec<&PathBuf> {
        match self {
            VariantConfig::ChatCompletion(params) => params.get_all_template_paths(),
            VariantConfig::RejectionSampling(params) => params.get_all_template_paths(),
        }
    }
}
