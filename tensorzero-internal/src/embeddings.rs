use std::future::Future;
use std::time::Duration;
use std::{collections::HashMap, sync::Arc};

use crate::cache::{
    embedding_cache_lookup, start_cache_write, CacheData, EmbeddingCacheData,
    EmbeddingModelProviderRequest,
};
use crate::endpoints::inference::InferenceClients;
use crate::model_table::BaseModelTable;
use crate::model_table::ShorthandModelConfig;
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::{
        providers::openai::OpenAIProvider,
        types::{
            current_timestamp, Latency, ModelInferenceResponseWithMetadata, RequestMessage, Role,
            Usage,
        },
    },
    model::ProviderConfig,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyProvider;

pub type EmbeddingModelTable = BaseModelTable<EmbeddingModelConfig>;

impl ShorthandModelConfig for EmbeddingModelConfig {
    const SHORTHAND_MODEL_PREFIXES: &[&str] = &["openai::"];
    const MODEL_TYPE: &str = "Embedding model";
    fn from_shorthand(provider_type: &str, model_name: &str) -> Result<Self, Error> {
        let model_name = model_name.to_string();
        let provider_config = match provider_type {
            "openai" => {
                EmbeddingProviderConfig::OpenAI(OpenAIProvider::new(model_name, None, None)?)
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            "dummy" => EmbeddingProviderConfig::Dummy(DummyProvider::new(model_name, None)?),
            _ => {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!("Invalid provider type: {}", provider_type),
                }));
            }
        };
        Ok(EmbeddingModelConfig {
            routing: vec![provider_type.to_string().into()],
            providers: HashMap::from([(provider_type.to_string().into(), provider_config)]),
        })
    }

    fn validate(&self, _key: &str) -> Result<(), Error> {
        // Credentials are validated during deserialization
        // We may add additional validation here in the future
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingModelConfig {
    pub routing: Vec<Arc<str>>,
    pub providers: HashMap<Arc<str>, EmbeddingProviderConfig>,
}

impl EmbeddingModelConfig {
    #[instrument(skip_all)]
    pub async fn embed(
        &self,
        request: &EmbeddingRequest,
        model_name: &str,
        clients: &InferenceClients<'_>,
    ) -> Result<EmbeddingResponse, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let provider_config = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.to_string(),
                })
            })?;
            let provider_request = EmbeddingModelProviderRequest {
                request,
                provider_name,
                model_name,
            };
            // TODO: think about how to best handle errors here
            if clients.cache_options.enabled.read() {
                let cache_lookup = embedding_cache_lookup(
                    clients.clickhouse_connection_info,
                    &provider_request,
                    clients.cache_options.max_age_s,
                )
                .await
                .ok()
                .flatten();
                if let Some(cache_lookup) = cache_lookup {
                    return Ok(cache_lookup);
                }
            }
            let response = provider_config
                .embed(request, clients.http_client, clients.credentials)
                .await;
            match response {
                Ok(response) => {
                    if clients.cache_options.enabled.write() {
                        let _ = start_cache_write(
                            clients.clickhouse_connection_info,
                            provider_request.get_cache_key()?,
                            EmbeddingCacheData {
                                embedding: response.embedding.clone(),
                            },
                            &response.raw_request,
                            &response.raw_response,
                            &response.usage,
                            None,
                        );
                    }
                    let embedding_response =
                        EmbeddingResponse::new(response, provider_name.clone());
                    return Ok(embedding_response);
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(ErrorDetails::ModelProvidersExhausted { provider_errors }.into())
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub struct EmbeddingRequest {
    pub input: String,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingProviderRequest<'request> {
    pub request: &'request EmbeddingRequest,
    pub model_name: &'request str,
    pub provider_name: &'request str,
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingProviderResponse {
    pub id: Uuid,
    pub input: String,
    pub embedding: Vec<f32>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingResponse {
    pub id: Uuid,
    pub input: String,
    pub embedding: Vec<f32>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub embedding_provider_name: Arc<str>,
    pub cached: bool,
}

impl EmbeddingResponse {
    pub fn from_cache(
        cache_lookup: CacheData<EmbeddingCacheData>,
        request: &EmbeddingModelProviderRequest,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            input: request.request.input.clone(),
            embedding: cache_lookup.output.embedding,
            raw_request: cache_lookup.raw_request,
            raw_response: cache_lookup.raw_response,
            usage: Usage {
                input_tokens: cache_lookup.input_tokens,
                output_tokens: cache_lookup.output_tokens,
            },
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            embedding_provider_name: Arc::from(request.provider_name),
            cached: true,
        }
    }
}

pub struct EmbeddingResponseWithMetadata {
    pub id: Uuid,
    pub input: String,
    pub embedding: Vec<f32>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub embedding_provider_name: Arc<str>,
    pub embedding_model_name: Arc<str>,
}

impl EmbeddingResponse {
    pub fn new(
        embedding_provider_response: EmbeddingProviderResponse,
        embedding_provider_name: Arc<str>,
    ) -> Self {
        Self {
            id: embedding_provider_response.id,
            input: embedding_provider_response.input,
            embedding: embedding_provider_response.embedding,
            created: embedding_provider_response.created,
            raw_request: embedding_provider_response.raw_request,
            raw_response: embedding_provider_response.raw_response,
            usage: embedding_provider_response.usage,
            latency: embedding_provider_response.latency,
            embedding_provider_name,
            cached: false,
        }
    }
}

impl EmbeddingResponseWithMetadata {
    pub fn new(embedding_response: EmbeddingResponse, embedding_model_name: Arc<str>) -> Self {
        Self {
            id: embedding_response.id,
            input: embedding_response.input,
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

impl From<EmbeddingResponseWithMetadata> for ModelInferenceResponseWithMetadata {
    fn from(response: EmbeddingResponseWithMetadata) -> Self {
        Self {
            id: response.id,
            output: vec![],
            created: response.created,
            system: None,
            input_messages: vec![RequestMessage {
                role: Role::User,
                content: vec![response.input.into()],
            }], // TODO (#399): Store this information in a more appropriate way for this kind of request
            raw_request: response.raw_request,
            raw_response: response.raw_response,
            usage: response.usage,
            latency: response.latency,
            model_provider_name: response.embedding_provider_name,
            model_name: response.embedding_model_name,
            cached: false,
            finish_reason: None,
        }
    }
}

pub trait EmbeddingProvider {
    fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> impl Future<Output = Result<EmbeddingProviderResponse, Error>> + Send;
}

#[derive(Debug)]
pub enum EmbeddingProviderConfig {
    OpenAI(OpenAIProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
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
    async fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<EmbeddingProviderResponse, Error> {
        match self {
            EmbeddingProviderConfig::OpenAI(provider) => {
                provider.embed(request, client, dynamic_api_keys).await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            EmbeddingProviderConfig::Dummy(provider) => {
                provider.embed(request, client, dynamic_api_keys).await
            }
        }
    }
}

impl EmbeddingProviderResponse {
    pub fn new(
        embedding: Vec<f32>,
        input: String,
        raw_request: String,
        raw_response: String,
        usage: Usage,
        latency: Latency,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            input,
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
    use tracing_test::traced_test;

    use crate::{
        cache::{CacheEnabledMode, CacheOptions},
        clickhouse::ClickHouseConnectionInfo,
    };

    use super::*;

    #[traced_test]
    #[tokio::test]
    async fn test_embedding_fallbacks() {
        let bad_provider = EmbeddingProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            ..Default::default()
        });
        let good_provider = EmbeddingProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            ..Default::default()
        });
        let fallback_embedding_model = EmbeddingModelConfig {
            routing: vec!["error".to_string().into(), "good".to_string().into()],
            providers: HashMap::from([
                ("error".to_string().into(), bad_provider),
                ("good".to_string().into(), good_provider),
            ]),
        };
        let request = EmbeddingRequest {
            input: "Hello, world!".to_string(),
        };
        let response = fallback_embedding_model
            .embed(
                &request,
                "fallback",
                &InferenceClients {
                    http_client: &Client::new(),
                    credentials: &InferenceCredentials::default(),
                    cache_options: &CacheOptions {
                        max_age_s: None,
                        enabled: CacheEnabledMode::Off,
                    },
                    clickhouse_connection_info: &ClickHouseConnectionInfo::new_disabled(),
                },
            )
            .await;
        assert!(response.is_ok());
        assert!(logs_contain(
            "Error sending request to Dummy provider for model 'error'"
        ))
    }
}
