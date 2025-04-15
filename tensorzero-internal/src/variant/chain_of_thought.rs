use std::path::Path;

use serde::Deserialize;

use crate::error::Error;
use crate::{
    config_parser::LoadableConfig,
    variant::chat_completion::{ChatCompletionConfig, UninitializedChatCompletionConfig},
};

#[derive(Debug)]
pub struct ChainOfThoughtConfig {
    pub inner: ChatCompletionConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedChainOfThoughtConfig {
    #[serde(flatten)]
    pub inner: UninitializedChatCompletionConfig,
}

impl LoadableConfig<ChainOfThoughtConfig> for UninitializedChainOfThoughtConfig {
    fn load<P: AsRef<Path>>(self, base_path: P) -> Result<ChainOfThoughtConfig, Error> {
        Ok(ChainOfThoughtConfig {
            inner: self.inner.load(base_path)?,
        })
    }
}
