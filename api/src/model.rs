use reqwest::Client;
use secrecy::SecretString;
use std::collections::HashMap;
use std::env;

use crate::{
    error::Error,
    inference::{
        providers::anthropic::AnthropicProvider,
        providers::openai::{FireworksProvider, OpenAIProvider},
        providers::provider_trait::InferenceProvider,
        types::{InferenceResponseStream, ModelInferenceRequest, ModelInferenceResponse},
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
    ) -> Result<InferenceResponseStream, Error> {
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
                // TODO: do the thing where we get a single chunk and make sure it is OK before moving on.
                // This is going to require us to pass some kind of tx for a channel.
                Ok(response) => return Ok(response),
                Err(error) => provider_errors.push(error),
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }
}

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
}

impl<'de> Deserialize<'de> for ProviderConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
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
        })
    }
}

static ANTHROPIC_PROVIDER: AnthropicProvider = AnthropicProvider;
static OPENAI_PROVIDER: OpenAIProvider = OpenAIProvider;
static FIREWORKS_PROVIDER: FireworksProvider = FireworksProvider;

impl ProviderConfig {
    pub async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<ModelInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic { .. } => {
                ANTHROPIC_PROVIDER.infer(request, self, client).await
            }
            ProviderConfig::Azure { .. } => {
                todo!()
            }
            ProviderConfig::OpenAI { .. } => OPENAI_PROVIDER.infer(request, self, client).await,
            ProviderConfig::Fireworks { .. } => {
                FIREWORKS_PROVIDER.infer(request, self, client).await
            }
        }
    }

    pub async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
    ) -> Result<InferenceResponseStream, Error> {
        match self {
            ProviderConfig::Anthropic { .. } => {
                ANTHROPIC_PROVIDER.infer_stream(request, self, client).await
            }
            ProviderConfig::Azure { .. } => {
                todo!()
            }
            ProviderConfig::OpenAI { .. } => {
                OPENAI_PROVIDER.infer_stream(request, self, client).await
            }
            ProviderConfig::Fireworks { .. } => {
                FIREWORKS_PROVIDER.infer_stream(request, self, client).await
            }
        }
    }
}
