use std::future::Future;
use std::time::Duration;
use std::{collections::HashMap, sync::Arc};

use crate::cache::{
    embedding_cache_lookup, start_cache_write, CacheData, EmbeddingCacheData,
    EmbeddingModelProviderRequest,
};
use crate::config_parser::ProviderTypesConfig;
use crate::endpoints::inference::InferenceClients;
use crate::model::UninitializedProviderConfig;
use crate::model_table::BaseModelTable;
use crate::model_table::ShorthandModelConfig;
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::{
        providers::{openai::OpenAIProvider, vllm::VLLMProvider},
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
    const SHORTHAND_MODEL_PREFIXES: &[&str] = &["openai::", "vllm::"];
    const MODEL_TYPE: &str = "Embedding model";
    async fn from_shorthand(provider_type: &str, model_name: &str) -> Result<Self, Error> {
        let model_name = model_name.to_string();
        let provider_config = match provider_type {
            "openai" => {
                EmbeddingProviderConfig::OpenAI(OpenAIProvider::new(model_name, None, None)?)
            }
            "vllm" => {
                // For shorthand, we'll use a default localhost URL
                let default_url = url::Url::parse("http://localhost:8000").map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Failed to parse default vLLM URL: {e}"),
                    })
                })?;
                EmbeddingProviderConfig::VLLM(VLLMProvider::new(model_name, default_url, None)?)
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            "dummy" => EmbeddingProviderConfig::Dummy(DummyProvider::new(model_name, None)?),
            _ => {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!("Invalid provider type: {provider_type}"),
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
pub struct UninitializedEmbeddingModelConfig {
    pub routing: Vec<Arc<str>>,
    pub providers: HashMap<Arc<str>, UninitializedEmbeddingProviderConfig>,
}

impl UninitializedEmbeddingModelConfig {
    pub fn load(self, provider_types: &ProviderTypesConfig) -> Result<EmbeddingModelConfig, Error> {
        let providers = self
            .providers
            .into_iter()
            .map(|(name, config)| {
                let provider_config = config.load(provider_types)?;
                Ok((name, provider_config))
            })
            .collect::<Result<HashMap<_, _>, Error>>()?;
        Ok(EmbeddingModelConfig {
            routing: self.routing,
            providers,
        })
    }
}

#[derive(Debug)]
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
                                embeddings: response.embeddings.clone(),
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

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    /// Get all input strings as a vector
    pub fn as_vec(&self) -> Vec<&str> {
        match self {
            EmbeddingInput::Single(text) => vec![text],
            EmbeddingInput::Batch(texts) => texts.iter().map(|s| s.as_str()).collect(),
        }
    }

    /// Get the number of inputs
    pub fn len(&self) -> usize {
        match self {
            EmbeddingInput::Single(_) => 1,
            EmbeddingInput::Batch(texts) => texts.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        match self {
            EmbeddingInput::Single(text) => text.is_empty(),
            EmbeddingInput::Batch(texts) => texts.is_empty() || texts.iter().all(|t| t.is_empty()),
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
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
    pub input: EmbeddingInput,
    pub embeddings: Vec<Vec<f32>>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingResponse {
    pub id: Uuid,
    pub input: EmbeddingInput,
    pub embeddings: Vec<Vec<f32>>,
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
            embeddings: cache_lookup.output.embeddings,
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
            embeddings: embedding_provider_response.embeddings,
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
        // For backward compatibility, take the first embedding and convert input to string
        let input_string = match &embedding_response.input {
            EmbeddingInput::Single(text) => text.clone(),
            EmbeddingInput::Batch(texts) => texts.first().unwrap_or(&String::new()).clone(),
        };
        let embedding = embedding_response
            .embeddings
            .into_iter()
            .next()
            .unwrap_or_default();

        Self {
            id: embedding_response.id,
            input: input_string,
            embedding,
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
    VLLM(VLLMProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
}

#[derive(Debug, Deserialize)]
pub struct UninitializedEmbeddingProviderConfig {
    #[serde(flatten)]
    config: UninitializedProviderConfig,
}

impl UninitializedEmbeddingProviderConfig {
    pub fn load(
        self,
        provider_types: &ProviderTypesConfig,
    ) -> Result<EmbeddingProviderConfig, Error> {
        let provider_config = self.config.load(provider_types)?;
        Ok(match provider_config {
            ProviderConfig::OpenAI(provider) => EmbeddingProviderConfig::OpenAI(provider),
            ProviderConfig::VLLM(provider) => EmbeddingProviderConfig::VLLM(provider),
            _ => {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!(
                        "Unsupported provider config for embedding: {provider_config:?}"
                    ),
                }));
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
            EmbeddingProviderConfig::VLLM(provider) => {
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
        embeddings: Vec<Vec<f32>>,
        input: EmbeddingInput,
        raw_request: String,
        raw_response: String,
        usage: Usage,
        latency: Latency,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            input,
            embeddings,
            created: current_timestamp(),
            raw_request,
            raw_response,
            usage,
            latency,
        }
    }

    /// Create a response from a single embedding (for backward compatibility)
    pub fn new_single(
        embedding: Vec<f32>,
        input: String,
        raw_request: String,
        raw_response: String,
        usage: Usage,
        latency: Latency,
    ) -> Self {
        Self::new(
            vec![embedding],
            EmbeddingInput::Single(input),
            raw_request,
            raw_response,
            usage,
            latency,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_input_variants() {
        // Test Single variant
        let single = EmbeddingInput::Single("Hello, world!".to_string());
        assert_eq!(single.len(), 1);
        assert!(!single.is_empty());
        assert_eq!(single.as_vec(), vec!["Hello, world!"]);

        // Test Batch variant
        let batch = EmbeddingInput::Batch(vec![
            "First text".to_string(),
            "Second text".to_string(),
            "Third text".to_string(),
        ]);
        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());
        assert_eq!(
            batch.as_vec(),
            vec!["First text", "Second text", "Third text"]
        );

        // Test empty cases
        let empty_single = EmbeddingInput::Single("".to_string());
        assert!(empty_single.is_empty());

        let empty_batch = EmbeddingInput::Batch(vec![]);
        assert!(empty_batch.is_empty());

        let batch_with_empty = EmbeddingInput::Batch(vec!["".to_string(), "".to_string()]);
        assert!(batch_with_empty.is_empty());
    }

    #[test]
    fn test_embedding_input_serialization() {
        // Test Single serialization
        let single = EmbeddingInput::Single("Hello".to_string());
        let serialized = serde_json::to_value(&single).unwrap();
        assert_eq!(serialized, serde_json::Value::String("Hello".to_string()));

        // Test Batch serialization
        let batch = EmbeddingInput::Batch(vec!["Hello".to_string(), "World".to_string()]);
        let serialized = serde_json::to_value(&batch).unwrap();
        assert_eq!(
            serialized,
            serde_json::Value::Array(vec![
                serde_json::Value::String("Hello".to_string()),
                serde_json::Value::String("World".to_string())
            ])
        );
    }

    #[test]
    fn test_embedding_request_with_batch_input() {
        let request = EmbeddingRequest {
            input: EmbeddingInput::Batch(vec!["Text 1".to_string(), "Text 2".to_string()]),
            encoding_format: None,
        };

        assert_eq!(request.input.len(), 2);
        assert_eq!(request.input.as_vec(), vec!["Text 1", "Text 2"]);
    }

    #[test]
    fn test_embedding_provider_response_batch() {
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let input = EmbeddingInput::Batch(vec!["Text 1".to_string(), "Text 2".to_string()]);

        let response = EmbeddingProviderResponse::new(
            embeddings.clone(),
            input.clone(),
            "raw_request".to_string(),
            "raw_response".to_string(),
            Usage {
                input_tokens: 10,
                output_tokens: 0,
            },
            Latency::NonStreaming {
                response_time: std::time::Duration::from_millis(100),
            },
        );

        assert_eq!(response.embeddings, embeddings);
        assert_eq!(response.input, input);
    }

    #[test]
    fn test_embedding_provider_response_new_single_compatibility() {
        // Test backward compatibility method
        let embedding = vec![0.1, 0.2, 0.3];
        let input = "Test input".to_string();

        let response = EmbeddingProviderResponse::new_single(
            embedding.clone(),
            input.clone(),
            "raw_request".to_string(),
            "raw_response".to_string(),
            Usage {
                input_tokens: 5,
                output_tokens: 0,
            },
            Latency::NonStreaming {
                response_time: std::time::Duration::from_millis(50),
            },
        );

        assert_eq!(response.embeddings, vec![embedding]);
        assert_eq!(response.input, EmbeddingInput::Single(input));
    }
    use tracing_test::traced_test;

    use crate::{
        cache::{CacheEnabledMode, CacheOptions},
        clickhouse::ClickHouseConnectionInfo,
    };

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
            input: EmbeddingInput::Single("Hello, world!".to_string()),
            encoding_format: None,
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
