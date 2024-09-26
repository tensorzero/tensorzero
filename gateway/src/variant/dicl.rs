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
};

use super::{InferenceConfig, JsonMode, ModelUsedInfo, Variant};

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiclConfig {
    #[serde(default)]
    pub weight: f64,
    pub embedding_model: String,
    pub model: String,
    pub system_instructions: Option<PathBuf>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u32>,
    #[serde(default)]
    pub json_mode: JsonMode,
}

impl Variant for DiclConfig {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'a HashMap<String, ModelConfig>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let serialized_input = serde_json::to_string(&input).map_err(|e| Error::Serialization {
            message: format!(
                "Error in serializing Input in dynamic in-context learning variant: {}",
                e
            ),
        })?;

        todo!()
    }

    async fn infer_stream<'request>(
        &'static self,
        input: &Input,
        models: &'static HashMap<String, ModelConfig>,
        function: &'static FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        inference_params: InferenceParams,
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

impl DiclConfig {
    fn retrieve_relevant_examples(&self, serialized_input: &str) -> Result<Vec<String>, Error> {}
}
