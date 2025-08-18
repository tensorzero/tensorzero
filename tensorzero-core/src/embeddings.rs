use std::future::Future;
use std::time::Duration;
use std::{collections::HashMap, sync::Arc};

use crate::cache::{
    embedding_cache_lookup, start_cache_write, CacheData, EmbeddingCacheData,
    EmbeddingModelProviderRequest,
};
use crate::config_parser::{ProviderTypesConfig, TimeoutsConfig};
use crate::endpoints::inference::InferenceClients;
use crate::model::UninitializedProviderConfig;
use crate::model_table::BaseModelTable;
use crate::model_table::ShorthandModelConfig;
use crate::providers::azure::AzureProvider;
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    inference::types::{
        current_timestamp, Latency, ModelInferenceResponseWithMetadata, RequestMessage, Role, Usage,
    },
    model::ProviderConfig,
    providers::openai::OpenAIProvider,
};
use futures::future::try_join_all;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::error::Elapsed;
use tracing::instrument;
use uuid::Uuid;

#[cfg(any(test, feature = "e2e_tests"))]
use crate::providers::dummy::DummyProvider;

pub type EmbeddingModelTable = BaseModelTable<EmbeddingModelConfig>;

impl ShorthandModelConfig for EmbeddingModelConfig {
    const SHORTHAND_MODEL_PREFIXES: &[&str] = &["openai::"];
    const MODEL_TYPE: &str = "Embedding model";
    async fn from_shorthand(provider_type: &str, model_name: &str) -> Result<Self, Error> {
        let model_name = model_name.to_string();
        let provider_config = match provider_type {
            "openai" => {
                EmbeddingProviderConfig::OpenAI(OpenAIProvider::new(model_name, None, None)?)
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            "dummy" => EmbeddingProviderConfig::Dummy(DummyProvider::new(model_name, None)?),
            _ => {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!("Invalid provider type: {provider_type}"),
                }));
            }
        };
        let provider_info = EmbeddingProviderInfo {
            inner: provider_config,
            timeouts: TimeoutsConfig::default(),
            provider_name: Arc::from(provider_type.to_string()),
        };
        Ok(EmbeddingModelConfig {
            routing: vec![provider_type.to_string().into()],
            providers: HashMap::from([(provider_type.to_string().into(), provider_info)]),
            timeouts: TimeoutsConfig::default(),
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
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
}

impl UninitializedEmbeddingModelConfig {
    pub async fn load(
        self,
        provider_types: &ProviderTypesConfig,
    ) -> Result<EmbeddingModelConfig, Error> {
        let providers = try_join_all(self.providers.into_iter().map(|(name, config)| async {
            let provider_config = config.load(provider_types, name.clone()).await?;
            Ok::<_, Error>((name, provider_config))
        }))
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();
        Ok(EmbeddingModelConfig {
            routing: self.routing,
            providers,
            timeouts: self.timeouts,
        })
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct EmbeddingModelConfig {
    pub routing: Vec<Arc<str>>,
    pub providers: HashMap<Arc<str>, EmbeddingProviderInfo>,
    pub timeouts: TimeoutsConfig,
}

impl EmbeddingModelConfig {
    #[instrument(skip_all)]
    pub async fn embed(
        &self,
        request: &EmbeddingRequest,
        model_name: &str,
        clients: &InferenceClients<'_>,
    ) -> Result<EmbeddingModelResponse, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        let run_all_embedding_models = async {
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
                        if clients.cache_options.enabled.write() && response.embeddings.len() == 1 {
                            let Some(first_embedding) = response.embeddings.first() else {
                                return Err(ErrorDetails::InternalError{
                             message: format!("Failed to get first embedding for cache {IMPOSSIBLE_ERROR_MESSAGE}")
                            }
                            .into());
                            };
                            if let Some(float_data) = first_embedding.as_float() {
                                let _ = start_cache_write(
                                    clients.clickhouse_connection_info,
                                    provider_request.get_cache_key()?,
                                    EmbeddingCacheData {
                                        embedding: float_data.clone(),
                                    },
                                    &response.raw_request,
                                    &response.raw_response,
                                    &response.usage,
                                    None,
                                );
                            }
                        };
                        let embedding_response =
                            EmbeddingModelResponse::new(response, provider_name.clone());
                        return Ok(embedding_response);
                    }
                    Err(error) => {
                        provider_errors.insert(provider_name.to_string(), error);
                    }
                }
            }
            Err(ErrorDetails::ModelProvidersExhausted { provider_errors }.into())
        };
        // This is the top-level embedding model timeout, which limits the total time taken to run all providers.
        // Some of the providers may themselves have timeouts, which is fine. Provider timeouts
        // are treated as just another kind of provider error - a timeout of N ms is equivalent
        // to a provider taking N ms, and then producing a normal HTTP error.
        if let Some(timeout) = self.timeouts.non_streaming.total_ms {
            let timeout = Duration::from_millis(timeout);
            tokio::time::timeout(timeout, run_all_embedding_models)
                .await
                // Convert the outer `Elapsed` error into a TensorZero error,
                // so that it can be handled by the `match response` block below
                .unwrap_or_else(|_: Elapsed| {
                    Err(Error::new(ErrorDetails::ModelTimeout {
                        model_name: model_name.to_string(),
                        timeout,
                        streaming: false,
                    }))
                })
        } else {
            run_all_embedding_models.await
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    pub fn num_inputs(&self) -> usize {
        match self {
            EmbeddingInput::Single(_) => 1,
            EmbeddingInput::Batch(texts) => texts.len(),
        }
    }

    pub fn first(&self) -> Option<&String> {
        match self {
            EmbeddingInput::Single(text) => Some(text),
            EmbeddingInput::Batch(texts) => texts.first(),
        }
    }
}

impl From<String> for EmbeddingInput {
    fn from(text: String) -> Self {
        EmbeddingInput::Single(text)
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    pub dimensions: Option<u32>,
    pub encoding_format: EmbeddingEncodingFormat,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingProviderRequest<'request> {
    pub request: &'request EmbeddingRequest,
    pub model_name: &'request str,
    pub provider_name: &'request str,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingEncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingProviderResponse {
    pub id: Uuid,
    pub input: EmbeddingInput,
    pub embeddings: Vec<Embedding>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

#[derive(Debug, PartialEq)]
pub struct EmbeddingModelResponse {
    pub id: Uuid,
    pub input: EmbeddingInput,
    pub embeddings: Vec<Embedding>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub embedding_provider_name: Arc<str>,
    pub cached: bool,
}

impl EmbeddingModelResponse {
    pub fn from_cache(
        cache_lookup: CacheData<EmbeddingCacheData>,
        request: &EmbeddingModelProviderRequest,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            created: current_timestamp(),
            input: request.request.input.clone(),
            embeddings: vec![Embedding::Float(cache_lookup.output.embedding)],
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
    pub input: EmbeddingInput,
    pub embeddings: Vec<Embedding>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub embedding_provider_name: Arc<str>,
    pub embedding_model_name: Arc<str>,
}

impl EmbeddingModelResponse {
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
    pub fn new(embedding_response: EmbeddingModelResponse, embedding_model_name: Arc<str>) -> Self {
        Self {
            id: embedding_response.id,
            input: embedding_response.input,
            embeddings: embedding_response.embeddings,
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

impl TryFrom<EmbeddingResponseWithMetadata> for ModelInferenceResponseWithMetadata {
    type Error = Error;

    fn try_from(response: EmbeddingResponseWithMetadata) -> Result<Self, Self::Error> {
        if response.input.num_inputs() != 1 {
            return Err(ErrorDetails::InternalError { message: format!("Can't convert batched embedding response to model inference response. {IMPOSSIBLE_ERROR_MESSAGE}") }.into());
        }
        let Some(input) = response.input.first() else {
            return Err(ErrorDetails::InternalError { message: format!("Can't convert batched embedding response to model inference response. {IMPOSSIBLE_ERROR_MESSAGE}") }.into());
        };
        Ok(Self {
            id: response.id,
            output: vec![],
            created: response.created,
            system: None,
            input_messages: vec![RequestMessage {
                role: Role::User,
                content: vec![input.clone().into()],
            }], // TODO (#399): Store this information in a more appropriate way for this kind of request
            raw_request: response.raw_request,
            raw_response: response.raw_response,
            usage: response.usage,
            latency: response.latency,
            model_provider_name: response.embedding_provider_name,
            model_name: response.embedding_model_name,
            cached: false,
            finish_reason: None,
        })
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

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum EmbeddingProviderConfig {
    OpenAI(OpenAIProvider),
    Azure(AzureProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
}

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct EmbeddingProviderInfo {
    pub inner: EmbeddingProviderConfig,
    pub timeouts: TimeoutsConfig,
    pub provider_name: Arc<str>,
}

impl EmbeddingProvider for EmbeddingProviderInfo {
    async fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<EmbeddingProviderResponse, Error> {
        let response_fut = self.inner.embed(request, client, dynamic_api_keys);
        Ok(
            if let Some(timeout_ms) = self.timeouts.non_streaming.total_ms {
                let timeout = Duration::from_millis(timeout_ms);
                tokio::time::timeout(timeout, response_fut)
                    .await
                    .unwrap_or_else(|_: Elapsed| {
                        Err(Error::new(ErrorDetails::ModelProviderTimeout {
                            provider_name: self.provider_name.to_string(),
                            timeout,
                            streaming: false,
                        }))
                    })?
            } else {
                response_fut.await?
            },
        )
    }
}

#[derive(Debug, Deserialize)]
pub struct UninitializedEmbeddingProviderConfig {
    #[serde(flatten)]
    config: UninitializedProviderConfig,
    #[serde(default)]
    timeouts: TimeoutsConfig,
}

impl UninitializedEmbeddingProviderConfig {
    pub async fn load(
        self,
        provider_types: &ProviderTypesConfig,
        provider_name: Arc<str>,
    ) -> Result<EmbeddingProviderInfo, Error> {
        let provider_config = self.config.load(provider_types).await?;
        let timeouts = self.timeouts;
        Ok(match provider_config {
            ProviderConfig::OpenAI(provider) => EmbeddingProviderInfo {
                inner: EmbeddingProviderConfig::OpenAI(provider),
                timeouts,
                provider_name,
            },
            ProviderConfig::Azure(provider) => EmbeddingProviderInfo {
                inner: EmbeddingProviderConfig::Azure(provider),
                timeouts,
                provider_name,
            },
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => EmbeddingProviderInfo {
                inner: EmbeddingProviderConfig::Dummy(provider),
                timeouts,
                provider_name,
            },
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
            EmbeddingProviderConfig::Azure(provider) => {
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
        embeddings: Vec<Embedding>,
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
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Embedding {
    Float(Vec<f32>),
    Base64(String),
}

impl<'a> Embedding {
    pub fn as_float(&'a self) -> Option<&'a Vec<f32>> {
        match self {
            Embedding::Float(vec) => Some(vec),
            Embedding::Base64(_) => None,
        }
    }

    pub fn ndims(&self) -> usize {
        match self {
            Embedding::Float(vec) => vec.len(),
            Embedding::Base64(encoded) => encoded.len() * 3 / 16,
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
        let bad_provider_info = EmbeddingProviderInfo {
            inner: bad_provider,
            timeouts: Default::default(),
            provider_name: Arc::from("error".to_string()),
        };
        let good_provider = EmbeddingProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            ..Default::default()
        });
        let good_provider_info = EmbeddingProviderInfo {
            inner: good_provider,
            timeouts: Default::default(),
            provider_name: Arc::from("good".to_string()),
        };
        let fallback_embedding_model = EmbeddingModelConfig {
            routing: vec!["error".to_string().into(), "good".to_string().into()],
            providers: HashMap::from([
                ("error".to_string().into(), bad_provider_info),
                ("good".to_string().into(), good_provider_info),
            ]),
            timeouts: TimeoutsConfig::default(),
        };
        let request = EmbeddingRequest {
            input: "Hello, world!".to_string().into(),
            dimensions: None,
            encoding_format: EmbeddingEncodingFormat::Float,
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
        ));
    }
}
