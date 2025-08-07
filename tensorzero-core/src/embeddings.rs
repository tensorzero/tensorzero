use std::borrow::Cow;
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
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    inference::types::{
        current_timestamp, Latency, ModelInferenceResponseWithMetadata, RequestMessage, Role, Usage,
    },
    model::ProviderConfig,
    providers::openai::OpenAIProvider,
};
use base64::prelude::*;
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
                            let _ = start_cache_write(
                                clients.clickhouse_connection_info,
                                provider_request.get_cache_key()?,
                                EmbeddingCacheData {
                                    embedding: first_embedding.into_float()?.into_owned(),
                                },
                                &response.raw_request,
                                &response.raw_response,
                                &response.usage,
                                None,
                            );
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
            #[cfg(feature = "e2e_tests")]
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
    pub fn into_float(&'a self) -> Result<Cow<'a, Vec<f32>>, Error> {
        match self {
            Embedding::Float(vec) => Ok(Cow::Borrowed(vec)),
            Embedding::Base64(base64) => {
                let bytes = BASE64_STANDARD.decode(base64).map_err(|e| {
                    Error::new(ErrorDetails::Base64 {
                        message: e.to_string(),
                    })
                })?;
                let mut floats = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    floats.push(f32::from_le_bytes(chunk.try_into().map_err(
                        |e: std::array::TryFromSliceError| {
                            Error::new(ErrorDetails::Base64 {
                                message: e.to_string(),
                            })
                        },
                    )?));
                }
                Ok(Cow::Owned(floats))
            }
        }
    }

    pub fn into_base64(self) -> String {
        match self {
            Embedding::Float(vec) => {
                let mut bytes = Vec::with_capacity(vec.len() * 4);
                for float in vec {
                    bytes.extend_from_slice(&float.to_le_bytes());
                }
                BASE64_STANDARD.encode(bytes)
            }
            Embedding::Base64(base64) => base64,
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

    #[test]
    fn test_base64_to_float() {
        let base64 = "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";
        let embedding = Embedding::Base64(base64.to_string());
        let embedding_len = embedding.ndims();
        let float = embedding.into_float().unwrap();
        let float_len = float.len();
        assert_eq!(float_len, embedding_len);
    }
}
