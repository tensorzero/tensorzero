use reqwest::Client;
use secrecy::SecretString;
use std::collections::HashMap;

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
        api_key: &'a SecretString,
    ) -> Result<ModelInferenceResponse, Error> {
        let mut provider_errors = Vec::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config.infer(request, client, api_key).await;
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
        api_key: &'a SecretString,
    ) -> Result<InferenceResponseStream, Error> {
        let mut provider_errors = Vec::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config.infer_stream(request, client, api_key).await;
            match response {
                Ok(response) => return Ok(response),
                Err(error) => provider_errors.push(error),
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum ProviderConfig {
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

static ANTHROPIC_PROVIDER: AnthropicProvider = AnthropicProvider;
static OPENAI_PROVIDER: OpenAIProvider = OpenAIProvider;
static FIREWORKS_PROVIDER: FireworksProvider = FireworksProvider;

impl ProviderConfig {
    pub async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
        api_key: &'a SecretString,
    ) -> Result<ModelInferenceResponse, Error> {
        match self {
            ProviderConfig::Anthropic { .. } => {
                ANTHROPIC_PROVIDER
                    .infer(request, self, client, api_key)
                    .await
            }
            ProviderConfig::Azure { .. } => {
                todo!()
            }
            ProviderConfig::OpenAI { .. } => {
                OPENAI_PROVIDER.infer(request, self, client, api_key).await
            }
            ProviderConfig::Fireworks { .. } => {
                FIREWORKS_PROVIDER
                    .infer(request, self, client, api_key)
                    .await
            }
        }
    }

    pub async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        client: &'a Client,
        api_key: &'a SecretString,
    ) -> Result<InferenceResponseStream, Error> {
        match self {
            ProviderConfig::Anthropic { .. } => {
                ANTHROPIC_PROVIDER
                    .infer_stream(request, self, client, api_key)
                    .await
            }
            ProviderConfig::Azure { .. } => {
                todo!()
            }
            ProviderConfig::OpenAI { .. } => {
                OPENAI_PROVIDER
                    .infer_stream(request, self, client, api_key)
                    .await
            }
            ProviderConfig::Fireworks { .. } => {
                FIREWORKS_PROVIDER
                    .infer_stream(request, self, client, api_key)
                    .await
            }
        }
    }
}
