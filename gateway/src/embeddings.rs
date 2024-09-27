use std::{collections::HashMap, future::Future};

use reqwest::Client;
use serde::Deserialize;
use uuid::Uuid;

use crate::{
    error::Error,
    inference::{
        providers::{openai::OpenAIProvider, provider_trait::HasCredentials},
        types::{current_timestamp, Latency, ModelInferenceResponseWithMetadata, Usage},
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
                    let embedding_response = EmbeddingResponse::new(response, provider_name);
                    return Ok(embedding_response);
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::ModelProvidersExhausted { provider_errors })
    }

    pub fn validate(&self) -> Result<(), Error> {
        // Ensure that all providers have credentials
        if !self
            .providers
            .values()
            .all(|provider| provider.has_credentials())
        {
            return Err(Error::ModelValidation {
                message: "At least one provider lacks credentials".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingRequest {
    pub input: String,
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingProviderResponse {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingResponse<'a> {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub embedding_provider_name: &'a str,
}

pub struct EmbeddingResponseWithMetadata<'a> {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub embedding_provider_name: &'a str,
    pub embedding_model_name: &'a str,
}

impl<'a> EmbeddingResponse<'a> {
    pub fn new(
        embedding_provider_response: EmbeddingProviderResponse,
        embedding_provider_name: &'a str,
    ) -> Self {
        Self {
            id: embedding_provider_response.id,
            embedding: embedding_provider_response.embedding,
            created: embedding_provider_response.created,
            raw_request: embedding_provider_response.raw_request,
            raw_response: embedding_provider_response.raw_response,
            usage: embedding_provider_response.usage,
            latency: embedding_provider_response.latency,
            embedding_provider_name,
        }
    }
}

impl<'a> EmbeddingResponseWithMetadata<'a> {
    pub fn new(embedding_response: EmbeddingResponse<'a>, embedding_model_name: &'a str) -> Self {
        Self {
            id: embedding_response.id,
            embedding: embedding_response.embedding,
            created: embedding_response.created,
            raw_request: embedding_response.raw_request,
            raw_response: embedding_response.raw_response,
            usage: embedding_response.usage,
            latency: embedding_response.latency,
            embedding_provider_name: embedding_response.embedding_provider_name,
            embedding_model_name,
        }
    }
}

impl<'a> From<EmbeddingResponseWithMetadata<'a>> for ModelInferenceResponseWithMetadata<'a> {
    fn from(response: EmbeddingResponseWithMetadata<'a>) -> Self {
        Self {
            id: response.id,
            content: vec![],
            created: response.created,
            raw_request: response.raw_request,
            raw_response: response.raw_response,
            usage: response.usage,
            latency: response.latency,
            model_provider_name: response.embedding_provider_name,
            model_name: response.embedding_model_name,
        }
    }
}

pub trait EmbeddingProvider: HasCredentials {
    fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &Client,
    ) -> impl Future<Output = Result<EmbeddingProviderResponse, Error>> + Send;
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

impl HasCredentials for EmbeddingProviderConfig {
    fn has_credentials(&self) -> bool {
        match self {
            EmbeddingProviderConfig::OpenAI(provider) => provider.has_credentials(),
        }
    }
}

impl EmbeddingProvider for EmbeddingProviderConfig {
    fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &Client,
    ) -> impl Future<Output = Result<EmbeddingProviderResponse, Error>> + Send {
        match self {
            EmbeddingProviderConfig::OpenAI(provider) => provider.embed(request, client),
        }
    }
}

impl EmbeddingProviderResponse {
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
    // TODO(Viraj): this
    // #[tokio::test]
    // async fn test_embedding_fallbacks() {
    //     todo!()
    // }
}
