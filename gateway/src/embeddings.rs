use std::{collections::HashMap, future::Future};

use reqwest::Client;
use serde::Deserialize;
use uuid::Uuid;

use crate::{
    error::Error,
    inference::{
        providers::openai::OpenAIProvider,
        types::{current_timestamp, Latency, Usage},
    },
    model::ProviderConfig,
};

#[derive(Debug, Deserialize)]
pub struct EmbeddingModelConfig {
    pub routing: Vec<String>,
    pub providers: HashMap<String, EmbeddingProviderConfig>,
}

impl EmbeddingModelConfig {
    pub async fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &Client,
    ) -> Result<EmbeddingResponse, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let provider_config =
                self.providers
                    .get(provider_name)
                    .ok_or(Error::ProviderNotFound {
                        provider_name: provider_name.clone(),
                    })?;
            let response = provider_config.embed(request, client).await;
            match response {
                Ok(response) => {
                    for error in provider_errors.values() {
                        error.log();
                    }
                    return Ok(response);
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingRequest {
    pub input: String,
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingResponse {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

pub trait EmbeddingProvider {
    fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &Client,
    ) -> impl Future<Output = Result<EmbeddingResponse, Error>> + Send;
}

#[derive(Debug)]
pub enum EmbeddingProviderConfig {
    OpenAI(OpenAIProvider),
}

impl<'de> Deserialize<'de> for EmbeddingProviderConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let provider_config = ProviderConfig::deserialize(deserializer)?;

        Ok(match provider_config {
            ProviderConfig::OpenAI(provider) => EmbeddingProviderConfig::OpenAI(provider),
            _ => {
                return Err(serde::de::Error::custom(format!(
                    "Unsupported provider config: {:?}",
                    provider_config
                )));
            }
        })
    }
}

impl EmbeddingProvider for EmbeddingProviderConfig {
    fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &Client,
    ) -> impl Future<Output = Result<EmbeddingResponse, Error>> + Send {
        match self {
            EmbeddingProviderConfig::OpenAI(provider) => provider.embed(request, client),
        }
    }
}

impl EmbeddingResponse {
    pub fn new(
        embedding: Vec<f32>,
        raw_request: String,
        raw_response: String,
        usage: Usage,
        latency: Latency,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            embedding,
            created: current_timestamp(),
            raw_request,
            raw_response,
            usage,
            latency,
        }
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_embedding_fallbacks() {
        todo!()
    }
}
