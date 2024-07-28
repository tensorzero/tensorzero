use serde::Deserialize;
use std::{collections::HashMap, path::PathBuf};

use crate::error::Error;
use crate::{
    config_parser::ModelConfig,
    inference::types::{InferenceResponse, InferenceResponseStream, InputMessage},
};

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum VariantConfig {
    ChatCompletion(ChatCompletionConfig),
}

pub trait Variant {
    async fn infer(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
    ) -> Result<InferenceResponse, Error>;

    async fn infer_stream(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
    ) -> Result<InferenceResponseStream, Error>;
}

impl VariantConfig {
    pub fn weight(&self) -> f64 {
        match self {
            VariantConfig::ChatCompletion(params) => params.weight,
        }
    }
}

impl Variant for VariantConfig {
    async fn infer(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
    ) -> Result<InferenceResponse, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => params.infer(messages, models).await,
        }
    }

    async fn infer_stream(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
    ) -> Result<InferenceResponseStream, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => params.infer_stream(messages, models).await,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionConfig {
    pub weight: f64,
    pub model: String, // TODO: validate that the model is valid given the rest of the config
    pub system_template: Option<PathBuf>,
    pub user_template: Option<PathBuf>,
    pub assistant_template: Option<PathBuf>,
}

impl Variant for ChatCompletionConfig {
    async fn infer(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
    ) -> Result<InferenceResponse, Error> {
        todo!()
    }

    async fn infer_stream(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
    ) -> Result<InferenceResponseStream, Error> {
        todo!()
    }
}
