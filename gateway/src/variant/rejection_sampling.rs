use std::{collections::HashMap, path::PathBuf};

use reqwest::Client;
use serde::Deserialize;

use crate::{
    endpoints::inference::InferenceParams,
    error::Error,
    function::FunctionConfig,
    inference::types::{InferenceResult, InferenceResultChunk, InferenceResultStream, Input},
    minijinja_util::TemplateConfig,
    model::ModelConfig,
    variant::VariantConfig,
};

use super::{InferenceConfig, ModelUsedInfo, Variant};

#[derive(Debug, Deserialize)]
pub struct RejectionSamplingConfig {
    pub weight: f64,
    pub candidates: Vec<String>,
    pub evaluator: EvaluatorConfig,
}

#[derive(Debug, Deserialize)]
pub struct EvaluatorConfig {
    #[serde(flatten)]
    variant: Box<VariantConfig>,
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
        todo!()
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
        todo!()
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
