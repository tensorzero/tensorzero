use reqwest::Client;
use secrecy::SecretString;
use std::collections::HashMap;
use std::env;

#[cfg(test)]
use crate::inference::providers::dummy::DummyProvider;
use crate::{
    error::Error,
    inference::{
        providers::{
            anthropic::AnthropicProvider,
            openai::{FireworksProvider, OpenAIProvider},
            provider_trait::InferenceProvider,
        },
        types::{
            InferenceResponseStream, ModelInferenceRequest, ModelInferenceResponse,
            ModelInferenceResponseChunk,
        },
    },
};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub routing: Vec<String>, // [provider name A, provider name B, ...]
    pub providers: HashMap<String, ProviderConfig>, // provider name => provider config
}

impl ModelConfig {
    pub async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let mut provider_errors = Vec::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config.infer(request, client).await;
            match response {
                Ok(response) => return Ok(response),
                Err(error) => provider_errors.push(error),
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }

    pub async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let mut provider_errors = Vec::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config.infer_stream(request, client).await;
            match response {
                Ok(response) => return Ok(response),
                Err(error) => provider_errors.push(error),
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }
}

// TODO: think about how we can manage typing here so we don't have to check every time this is passed that it is the correct type.
// TODO(Viraj): implement an arm of the ProviderConfig enum with the #[cfg(test)] attribut
// so that we can use it as a mock of a model provider that can exercise all needed behaviors.
#[derive(Clone, Debug)]
pub enum ProviderConfig {
    Anthropic {
        model_name: String,
        api_key: Option<SecretString>,
    },
    Azure {
        model_name: String,
        api_base: String,
        api_key: Option<SecretString>,
    },
    OpenAI {
        model_name: String,
        api_base: Option<String>,
        api_key: Option<SecretString>,
    },
    Fireworks {
        model_name: String,
        api_key: Option<SecretString>,
    },
    #[cfg(test)]
    Dummy { model_name: String },
}

impl<'de> Deserialize<'de> for ProviderConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// Helper struct for deserializing the ProviderConfig.
        /// This is necessary because we want to load environment variables as we deserialize
        /// and we need to be able to deserialize the correct one based on the "type" field.
        /// Use the ProviderConfig struct for all post-initialization logic.
        #[derive(Deserialize)]
        #[serde(tag = "type")]
        #[serde(rename_all = "lowercase")]
        #[serde(deny_unknown_fields)]
        enum ProviderConfigHelper {
            Anthropic {
                model_name: String,
            },
            Azure {
                model_name: String,
                api_base: String,
            },
            #[serde(rename = "openai")]
            OpenAI {
                model_name: String,
                api_base: Option<String>,
            },
            #[serde(rename = "fireworks")]
            Fireworks {
                model_name: String,
            },
            #[cfg(test)]
            #[serde(rename = "dummy")]
            Dummy {
                model_name: String,
            },
        }

        let helper = ProviderConfigHelper::deserialize(deserializer)?;

        Ok(match helper {
            ProviderConfigHelper::Anthropic { model_name } => ProviderConfig::Anthropic {
                model_name,
                api_key: env::var("ANTHROPIC_API_KEY").ok().map(SecretString::new),
            },
            ProviderConfigHelper::Azure {
                model_name,
                api_base,
            } => ProviderConfig::Azure {
                model_name,
                api_base,
                api_key: env::var("AZURE_OPENAI_API_KEY").ok().map(SecretString::new),
            },
            ProviderConfigHelper::OpenAI {
                model_name,
                api_base,
            } => ProviderConfig::OpenAI {
                model_name,
                api_base,
                api_key: env::var("OPENAI_API_KEY").ok().map(SecretString::new),
            },
            ProviderConfigHelper::Fireworks { model_name } => ProviderConfig::Fireworks {
                model_name,
                api_key: env::var("FIREWORKS_API_KEY").ok().map(SecretString::new),
            },
            #[cfg(test)]
            ProviderConfigHelper::Dummy { model_name } => ProviderConfig::Dummy { model_name },
        })
    }
}

impl ProviderConfig {
    pub async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<ModelInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic { .. } => {
                AnthropicProvider::infer(request, self, client).await
            }
            ProviderConfig::Azure { .. } => {
                todo!()
            }
            ProviderConfig::OpenAI { .. } => OpenAIProvider::infer(request, self, client).await,
            ProviderConfig::Fireworks { .. } => {
                FireworksProvider::infer(request, self, client).await
            }
            #[cfg(test)]
            ProviderConfig::Dummy { .. } => DummyProvider::infer(request, self, client).await,
        }
    }

    pub async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        match self {
            ProviderConfig::Anthropic { .. } => {
                AnthropicProvider::infer_stream(request, self, client).await
            }
            ProviderConfig::Azure { .. } => {
                todo!()
            }
            ProviderConfig::OpenAI { .. } => {
                OpenAIProvider::infer_stream(request, self, client).await
            }
            ProviderConfig::Fireworks { .. } => {
                FireworksProvider::infer_stream(request, self, client).await
            }
            #[cfg(test)]
            ProviderConfig::Dummy { .. } => {
                DummyProvider::infer_stream(request, self, client).await
            }
        }
    }
}
