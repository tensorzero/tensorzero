use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;

use crate::cache::{
    embedding_cache_lookup, start_cache_write, CacheData, CacheValidationInfo, EmbeddingCacheData,
    EmbeddingModelProviderRequest,
};
use crate::config::{provider_types::ProviderTypesConfig, TimeoutsConfig};
use crate::endpoints::inference::InferenceClients;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::RequestMessagesOrBatch;
use crate::inference::types::{ContentBlock, Text};
use crate::model::{ModelProviderRequestInfo, UninitializedProviderConfig};
use crate::model_table::{BaseModelTable, ProviderKind, ProviderTypeDefaultCredentials};
use crate::model_table::{OpenAIKind, ShorthandModelConfig};
use crate::providers::azure::AzureProvider;
use crate::providers::openrouter::OpenRouterProvider;
use crate::rate_limiting::{
    get_estimated_tokens, EstimatedRateLimitResourceUsage, RateLimitResource,
    RateLimitResourceUsage, RateLimitedInputContent, RateLimitedRequest, RateLimitedResponse,
};
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    inference::types::{
        current_timestamp, Latency, ModelInferenceResponseWithMetadata, RequestMessage, Role, Usage,
    },
    model::ProviderConfig,
    providers::openai::{OpenAIAPIType, OpenAIProvider},
};
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use tokio::time::error::Elapsed;
use tracing::{instrument, Span};
use tracing_futures::Instrument;
use uuid::Uuid;

#[cfg(any(test, feature = "e2e_tests"))]
use crate::providers::dummy::DummyProvider;

pub type EmbeddingModelTable = BaseModelTable<EmbeddingModelConfig>;

impl ShorthandModelConfig for EmbeddingModelConfig {
    const SHORTHAND_MODEL_PREFIXES: &[&str] = &["openai::"];
    const MODEL_TYPE: &str = "Embedding model";
    async fn from_shorthand(
        provider_type: &str,
        model_name: &str,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self, Error> {
        let model_name = model_name.to_string();
        let provider_config = match provider_type {
            "openai" => EmbeddingProviderConfig::OpenAI(OpenAIProvider::new(
                model_name,
                None,
                OpenAIKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
                // TODO: handle the fact that there are also embeddings
                OpenAIAPIType::ChatCompletions,
                false,
                Vec::new(),
            )?),
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
            timeout_ms: None,
            provider_name: Arc::from(provider_type.to_string()),
            extra_body: Default::default(),
        };
        Ok(EmbeddingModelConfig {
            routing: vec![provider_type.to_string().into()],
            providers: HashMap::from([(provider_type.to_string().into(), provider_info)]),
            timeout_ms: None,
        })
    }

    fn validate(
        &self,
        _key: &str,
        global_outbound_http_timeout: &chrono::Duration,
    ) -> Result<(), Error> {
        let global_ms = global_outbound_http_timeout.num_milliseconds();
        if let Some(timeout_ms) = self.timeout_ms {
            if chrono::Duration::milliseconds(timeout_ms as i64) > *global_outbound_http_timeout {
                return Err(Error::new(ErrorDetails::Config {
                    message: format!("The `timeout_ms` value `{timeout_ms}` is greater than `gateway.global_outbound_http_timeout_ms`: `{global_ms}`"),
                }));
            }
        }
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
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
}

impl UninitializedEmbeddingModelConfig {
    pub async fn load(
        self,
        provider_types: &ProviderTypesConfig,
        default_credentials: &ProviderTypeDefaultCredentials,
        http_client: TensorzeroHttpClient,
    ) -> Result<EmbeddingModelConfig, Error> {
        // Handle timeout deprecation
        let timeout_ms = match (self.timeout_ms, self.timeouts.non_streaming.total_ms) {
            (Some(timeout_ms), None) => Some(timeout_ms),
            (None, Some(old_timeout)) => {
                crate::utils::deprecation_warning(
                    "`timeouts` is deprecated for embedding models. \
                    Please use `timeout_ms` instead.",
                );
                Some(old_timeout)
            }
            (None, None) => None,
            (Some(_), Some(_)) => {
                return Err(Error::new(ErrorDetails::Config {
                    message: "`timeout_ms` and `timeouts` cannot both be set for embedding models"
                        .to_string(),
                }));
            }
        };

        let providers = try_join_all(self.providers.into_iter().map(|(name, config)| async {
            let provider_config = config
                .load(
                    provider_types,
                    name.clone(),
                    default_credentials,
                    http_client.clone(),
                )
                .await?;
            Ok::<_, Error>((name, provider_config))
        }))
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();
        Ok(EmbeddingModelConfig {
            routing: self.routing,
            providers,
            timeout_ms,
        })
    }
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct EmbeddingModelConfig {
    pub routing: Vec<Arc<str>>,
    pub providers: HashMap<Arc<str>, EmbeddingProviderInfo>,
    pub timeout_ms: Option<u64>,
}

impl EmbeddingModelConfig {
    #[instrument(skip_all)]
    pub async fn embed(
        &self,
        request: &EmbeddingRequest,
        model_name: &str,
        clients: &InferenceClients,
    ) -> Result<EmbeddingModelResponse, Error> {
        let mut provider_errors: IndexMap<String, Error> = IndexMap::new();
        let run_all_embedding_models = async {
            for provider_name in &self.routing {
                let provider_config = self.providers.get(provider_name).ok_or_else(|| {
                    Error::new(ErrorDetails::ProviderNotFound {
                        provider_name: provider_name.to_string(),
                    })
                })?;
                let provider_request = EmbeddingModelProviderRequest {
                    request,
                    model_name,
                    provider_name,
                    otlp_config: &clients.otlp_config,
                };
                // TODO: think about how to best handle errors here
                if clients.cache_options.enabled.read() {
                    let cache_lookup = embedding_cache_lookup(
                        &clients.clickhouse_connection_info,
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
                    .embed(request, clients, &provider_config.into())
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
                                &clients.clickhouse_connection_info,
                                provider_request.get_cache_key()?,
                                CacheData {
                                    output: EmbeddingCacheData {
                                        embedding: first_embedding.clone(),
                                    },
                                    raw_request: response.raw_request.clone(),
                                    raw_response: response.raw_response.clone(),
                                    input_tokens: response.usage.input_tokens,
                                    output_tokens: response.usage.output_tokens,
                                    finish_reason: None,
                                },
                                CacheValidationInfo { tool_config: None },
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
        if let Some(timeout_ms) = self.timeout_ms {
            let timeout = Duration::from_millis(timeout_ms);
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
    SingleTokens(Vec<u32>),
    BatchTokens(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    pub fn num_inputs(&self) -> usize {
        match self {
            EmbeddingInput::Single(_) => 1,
            EmbeddingInput::Batch(texts) => texts.len(),
            EmbeddingInput::SingleTokens(_) => 1,
            EmbeddingInput::BatchTokens(tokens) => tokens.len(),
        }
    }

    pub fn first(&self) -> Option<&String> {
        match self {
            EmbeddingInput::Single(text) => Some(text),
            EmbeddingInput::Batch(texts) => texts.first(),
            EmbeddingInput::SingleTokens(_) => None,
            EmbeddingInput::BatchTokens(_) => None,
        }
    }
}

impl RateLimitedInputContent for EmbeddingInput {
    fn estimated_input_token_usage(&self) -> u64 {
        match self {
            EmbeddingInput::Single(text) => get_estimated_tokens(text),
            EmbeddingInput::Batch(texts) => texts
                .iter()
                .map(|text| get_estimated_tokens(text))
                .sum::<u64>(),
            // For token arrays, we have exact counts, not estimates
            EmbeddingInput::SingleTokens(tokens) => tokens.len() as u64,
            EmbeddingInput::BatchTokens(token_arrays) => token_arrays
                .iter()
                .map(|tokens| tokens.len() as u64)
                .sum::<u64>(),
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

impl RateLimitedRequest for EmbeddingRequest {
    fn estimated_resource_usage(
        &self,
        resources: &[RateLimitResource],
    ) -> Result<EstimatedRateLimitResourceUsage, Error> {
        let EmbeddingRequest {
            input,
            dimensions: _,
            encoding_format: _,
        } = self;

        let tokens = if resources.contains(&RateLimitResource::Token) {
            Some(input.estimated_input_token_usage())
        } else {
            None
        };

        let model_inferences = if resources.contains(&RateLimitResource::ModelInference) {
            Some(1)
        } else {
            None
        };

        Ok(EstimatedRateLimitResourceUsage {
            model_inferences,
            tokens,
        })
    }
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

impl RateLimitedResponse for EmbeddingProviderResponse {
    fn resource_usage(&self) -> RateLimitResourceUsage {
        if let Some(tokens) = self.usage.total_tokens() {
            RateLimitResourceUsage::Exact {
                model_inferences: 1,
                tokens: tokens as u64,
            }
        } else {
            RateLimitResourceUsage::UnderEstimate {
                model_inferences: 1,
                tokens: 0,
            }
        }
    }
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
            embeddings: vec![cache_lookup.output.embedding],
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

    /// We return the actual usage (meaning the number of tokens the user would be billed for)
    /// in the HTTP response.
    /// However, we store the number of tokens that would have been used in the database.
    /// So we need this function to compute the actual usage in order to send it in the HTTP response.
    pub fn usage_considering_cached(&self) -> Usage {
        if self.cached {
            Usage {
                input_tokens: Some(0),
                output_tokens: Some(0),
            }
        } else {
            self.usage
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
            input_messages: RequestMessagesOrBatch::Message(vec![RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: input.clone(),
                })],
            }]), // TODO (#399): Store this information in a more appropriate way for this kind of request
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
        client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
        model_provider_data: &EmbeddingProviderRequestInfo,
    ) -> impl Future<Output = Result<EmbeddingProviderResponse, Error>> + Send;
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub enum EmbeddingProviderConfig {
    OpenAI(OpenAIProvider),
    Azure(AzureProvider),
    OpenRouter(OpenRouterProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct EmbeddingProviderInfo {
    pub inner: EmbeddingProviderConfig,
    pub timeout_ms: Option<u64>,
    pub provider_name: Arc<str>,
    #[cfg_attr(test, ts(skip))]
    pub extra_body: Option<ExtraBodyConfig>,
}

#[derive(Clone, Debug)]
pub struct EmbeddingProviderRequestInfo {
    pub provider_name: Arc<str>,
    pub extra_body: Option<ExtraBodyConfig>,
}

impl From<&EmbeddingProviderInfo> for EmbeddingProviderRequestInfo {
    fn from(val: &EmbeddingProviderInfo) -> Self {
        EmbeddingProviderRequestInfo {
            provider_name: val.provider_name.clone(),
            extra_body: val.extra_body.clone(),
        }
    }
}

impl From<&EmbeddingProviderRequestInfo> for ModelProviderRequestInfo {
    fn from(val: &EmbeddingProviderRequestInfo) -> Self {
        crate::model::ModelProviderRequestInfo {
            provider_name: val.provider_name.clone(),
            extra_headers: None, // Embeddings don't use extra headers yet
            extra_body: val.extra_body.clone(),
        }
    }
}

impl EmbeddingProviderInfo {
    pub async fn embed(
        &self,
        request: &EmbeddingRequest,
        clients: &InferenceClients,
        model_provider_data: &EmbeddingProviderRequestInfo,
    ) -> Result<EmbeddingProviderResponse, Error> {
        let ticket_borrow = clients
            .rate_limiting_config
            .consume_tickets(
                &clients.postgres_connection_info,
                &clients.scope_info,
                request,
            )
            .await?;
        let response_fut = self.inner.embed(
            request,
            &clients.http_client,
            &clients.credentials,
            model_provider_data,
        );
        let response = if let Some(timeout_ms) = self.timeout_ms {
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
        };
        let postgres_connection_info = clients.postgres_connection_info.clone();
        let resource_usage = response.resource_usage();
        // Make sure that we finish updating rate-limiting tickets if the gateway shuts down
        clients.deferred_tasks.spawn(
            async move {
                if let Err(e) = ticket_borrow
                    .return_tickets(&postgres_connection_info, resource_usage)
                    .await
                {
                    tracing::error!("Failed to return rate limit tickets: {}", e);
                }
            }
            .instrument(Span::current()),
        );
        Ok(response)
    }
}

#[derive(Debug, Deserialize)]
pub struct UninitializedEmbeddingProviderConfig {
    #[serde(flatten)]
    config: UninitializedProviderConfig,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    timeouts: TimeoutsConfig,
    #[serde(default)]
    pub extra_body: Option<ExtraBodyConfig>,
}

impl UninitializedEmbeddingProviderConfig {
    pub async fn load(
        self,
        provider_types: &ProviderTypesConfig,
        provider_name: Arc<str>,
        default_credentials: &ProviderTypeDefaultCredentials,
        http_client: TensorzeroHttpClient,
    ) -> Result<EmbeddingProviderInfo, Error> {
        let provider_config = self
            .config
            .load(provider_types, default_credentials, http_client)
            .await?;
        // Handle timeout deprecation
        let timeout_ms = match (self.timeout_ms, self.timeouts.non_streaming.total_ms) {
            (Some(timeout_ms), None) => Some(timeout_ms),
            (None, Some(old_timeout)) => {
                crate::utils::deprecation_warning(
                    "`timeouts` is deprecated for embedding providers. \
                    Please use `timeout_ms` instead.",
                );
                Some(old_timeout)
            }
            (None, None) => None,
            (Some(_), Some(_)) => {
                return Err(Error::new(ErrorDetails::Config {
                    message:
                        "`timeout_ms` and `timeouts` cannot both be set for embedding providers"
                            .to_string(),
                }));
            }
        };

        let extra_body = self.extra_body;
        Ok(match provider_config {
            ProviderConfig::OpenAI(provider) => EmbeddingProviderInfo {
                inner: EmbeddingProviderConfig::OpenAI(provider),
                timeout_ms,
                provider_name,
                extra_body,
            },
            ProviderConfig::Azure(provider) => EmbeddingProviderInfo {
                inner: EmbeddingProviderConfig::Azure(provider),
                timeout_ms,
                provider_name,
                extra_body,
            },
            ProviderConfig::OpenRouter(provider) => EmbeddingProviderInfo {
                inner: EmbeddingProviderConfig::OpenRouter(provider),
                timeout_ms,
                provider_name,
                extra_body,
            },
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => EmbeddingProviderInfo {
                inner: EmbeddingProviderConfig::Dummy(provider),
                timeout_ms,
                provider_name,
                extra_body,
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
        client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
        model_provider_data: &EmbeddingProviderRequestInfo,
    ) -> Result<EmbeddingProviderResponse, Error> {
        match self {
            EmbeddingProviderConfig::OpenAI(provider) => {
                provider
                    .embed(request, client, dynamic_api_keys, model_provider_data)
                    .await
            }
            EmbeddingProviderConfig::Azure(provider) => {
                provider
                    .embed(request, client, dynamic_api_keys, model_provider_data)
                    .await
            }
            EmbeddingProviderConfig::OpenRouter(provider) => {
                provider
                    .embed(request, client, dynamic_api_keys, model_provider_data)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            EmbeddingProviderConfig::Dummy(provider) => {
                provider
                    .embed(request, client, dynamic_api_keys, model_provider_data)
                    .await
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
    use crate::{
        cache::{CacheEnabledMode, CacheOptions},
        db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
        model_table::ProviderTypeDefaultCredentials,
    };

    use super::*;
    #[tokio::test]
    async fn test_embedding_fallbacks() {
        let logs_contain = crate::utils::testing::capture_logs();
        let bad_provider = EmbeddingProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            ..Default::default()
        });
        let bad_provider_info = EmbeddingProviderInfo {
            inner: bad_provider,
            timeout_ms: None,
            provider_name: Arc::from("error".to_string()),
            extra_body: None,
        };
        let good_provider = EmbeddingProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            ..Default::default()
        });
        let good_provider_info = EmbeddingProviderInfo {
            inner: good_provider,
            timeout_ms: None,
            provider_name: Arc::from("good".to_string()),
            extra_body: None,
        };
        let fallback_embedding_model = EmbeddingModelConfig {
            routing: vec!["error".to_string().into(), "good".to_string().into()],
            providers: HashMap::from([
                ("error".to_string().into(), bad_provider_info),
                ("good".to_string().into(), good_provider_info),
            ]),
            timeout_ms: None,
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
                    http_client: TensorzeroHttpClient::new_testing().unwrap(),
                    clickhouse_connection_info: ClickHouseConnectionInfo::new_disabled(),
                    postgres_connection_info: PostgresConnectionInfo::Disabled,
                    credentials: Arc::new(InferenceCredentials::default()),
                    cache_options: CacheOptions {
                        max_age_s: None,
                        enabled: CacheEnabledMode::Off,
                    },
                    tags: Arc::new(Default::default()),
                    rate_limiting_config: Arc::new(Default::default()),
                    otlp_config: Default::default(),
                    deferred_tasks: tokio_util::task::TaskTracker::new(),
                    scope_info: crate::rate_limiting::ScopeInfo {
                        tags: Arc::new(HashMap::new()),
                        api_key_public_id: None,
                    },
                },
            )
            .await;
        assert!(response.is_ok());
        assert!(logs_contain(
            "Error sending request to Dummy provider for model 'error'"
        ));
    }

    #[tokio::test]
    async fn test_embedding_provider_config_with_extra_body() {
        use crate::inference::types::extra_body::{
            ExtraBodyConfig, ExtraBodyReplacement, ExtraBodyReplacementKind,
        };
        use serde_json::json;

        let replacement = ExtraBodyReplacement {
            pointer: "/task".to_string(),
            kind: ExtraBodyReplacementKind::Value(json!("query")),
        };
        let extra_body_config = ExtraBodyConfig {
            data: vec![replacement.clone()],
        };

        let uninitialized_config = UninitializedEmbeddingProviderConfig {
            config: UninitializedProviderConfig::OpenAI {
                model_name: "text-embedding-ada-002".to_string(),
                api_base: None,
                api_key_location: Some(crate::model::CredentialLocationWithFallback::Single(
                    crate::model::CredentialLocation::None,
                )),
                api_type: Default::default(),
                include_encrypted_reasoning: false,
                provider_tools: Vec::new(),
            },
            timeout_ms: None,
            timeouts: TimeoutsConfig::default(),
            extra_body: Some(extra_body_config.clone()),
        };

        let provider_info = uninitialized_config
            .load(
                &ProviderTypesConfig::default(),
                Arc::from("test_provider"),
                &ProviderTypeDefaultCredentials::default(),
                TensorzeroHttpClient::new_testing().unwrap(),
            )
            .await
            .unwrap();

        // Verify the extra_body is preserved
        assert!(provider_info.extra_body.is_some());
        let loaded_extra_body = provider_info.extra_body.unwrap();
        assert_eq!(loaded_extra_body.data.len(), 1);
        assert_eq!(loaded_extra_body.data[0], replacement);
    }
}
