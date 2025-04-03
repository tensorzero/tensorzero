use futures::StreamExt;
use reqwest::Client;
use secrecy::SecretString;
use serde::de::Error as SerdeError;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use std::{env, fs};
use strum::VariantNames;
use tensorzero_derive::TensorZeroDeserialize;
#[allow(unused_imports)]
use tracing::{span, warn, Instrument, Level};
use url::Url;

use crate::cache::{
    cache_lookup, cache_lookup_streaming, start_cache_write, start_cache_write_streaming,
    CacheData, ModelProviderRequest, NonStreamingCacheData, StreamingCacheData,
};
use crate::config_parser::SKIP_CREDENTIAL_VALIDATION;
use crate::endpoints::inference::InferenceClients;
#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyProvider;
use crate::inference::providers::google_ai_studio_gemini::GoogleAIStudioGeminiProvider;

use crate::inference::providers::helpers::peek_first_chunk;
use crate::inference::providers::hyperbolic::HyperbolicProvider;
use crate::inference::providers::sglang::SGLangProvider;
use crate::inference::providers::tgi::TGIProvider;
use crate::inference::types::batch::{
    BatchRequestRow, PollBatchInferenceResponse, StartBatchModelInferenceResponse,
    StartBatchProviderInferenceResponse,
};
use crate::inference::types::extra_body::{ExtraBodyConfig, ExtraHeadersConfig};
use crate::inference::types::{
    current_timestamp, ContentBlock, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponseChunk, ProviderInferenceResponseStreamInner, RequestMessage, Usage,
};
use crate::model_table::{BaseModelTable, ShorthandModelConfig};
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::{
        providers::{
            anthropic::AnthropicProvider, aws_bedrock::AWSBedrockProvider, azure::AzureProvider,
            deepseek::DeepSeekProvider, fireworks::FireworksProvider,
            gcp_vertex_anthropic::GCPVertexAnthropicProvider,
            gcp_vertex_gemini::GCPVertexGeminiProvider, mistral::MistralProvider,
            openai::OpenAIProvider, provider_trait::InferenceProvider, together::TogetherProvider,
            vllm::VLLMProvider, xai::XAIProvider,
        },
        types::{ModelInferenceRequest, ModelInferenceResponse, ProviderInferenceResponse},
    },
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub routing: Vec<Arc<str>>, // [provider name A, provider name B, ...]
    #[serde(deserialize_with = "deserialize_model_providers")]
    pub providers: HashMap<Arc<str>, ModelProvider>, // provider name => provider config
}

// We want `ModelProvider` to know its own name (from the 'providers' config section).
// We first deserialize to `HashMap<Arc<str>, UninitializedModelProvider>`, and then
// build `ModelProvider`s using the name keys from the map.
pub fn deserialize_model_providers<'de, D>(
    deserializer: D,
) -> Result<HashMap<Arc<str>, ModelProvider>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let providers: HashMap<Arc<str>, UninitializedModelProvider> =
        HashMap::deserialize(deserializer)?;
    Ok(providers
        .into_iter()
        .map(|(name, provider)| {
            (
                name.clone(),
                ModelProvider {
                    name,
                    config: provider.config,
                    extra_body: provider.extra_body,
                    extra_headers: provider.extra_headers,
                },
            )
        })
        .collect())
}

pub struct StreamResponse {
    pub stream: PeekableProviderInferenceResponseStream,
    pub raw_request: String,
    pub model_provider_name: Arc<str>,
    pub cached: bool,
}

impl StreamResponse {
    pub fn from_cache(
        cache_lookup: CacheData<StreamingCacheData>,
        model_provider_name: Arc<str>,
    ) -> Self {
        let chunks = cache_lookup.output.chunks;
        let chunks_len = chunks.len();

        Self {
            stream: (Box::pin(futures::stream::iter(chunks.into_iter().enumerate().map(
                move |(index, c)| {
                    Ok(ProviderInferenceResponseChunk {
                        content: c.content,
                        raw_response: c.raw_response,
                        // We intentionally don't cache and re-use these values from the original
                        // request:
                        // The new result was 'created' now
                        created: current_timestamp(),
                        // Only include usage in the last chunk, None for all others
                        usage: if index == chunks_len - 1 {
                            Some(Usage {
                                input_tokens: cache_lookup.input_tokens,
                                output_tokens: cache_lookup.output_tokens,
                            })
                        } else {
                            None
                        },
                        // We didn't make any network calls to the model provider, so the latency is 0
                        latency: Duration::from_secs(0),
                        // For all chunks but the last one, the finish reason is None
                        // For the last chunk, the finish reason is the same as the cache lookup
                        finish_reason: if index == chunks_len - 1 {
                            cache_lookup.finish_reason.clone()
                        } else {
                            None
                        },
                    })
                },
            ))) as ProviderInferenceResponseStreamInner)
                .peekable(),
            raw_request: cache_lookup.raw_request,
            model_provider_name,
            cached: true,
        }
    }
}

/// Creates a fully-qualified name from a model and provider name, suitable for using
/// in `ContentBlock::Unknown.model_provider_name`
/// Note that 'model_name' is a name from `[models]`, which is not necessarily
/// the same as the underlying name passed to a specific provider api
pub fn fully_qualified_name(model_name: &str, provider_name: &str) -> String {
    format!("tensorzero::model_name::{model_name}::provider_name::{provider_name}")
}

impl ModelConfig {
    fn filter_content_blocks<'a>(
        &self,
        request: &'a ModelInferenceRequest<'a>,
        model_name: &str,
        provider_name: &str,
    ) -> Cow<'a, ModelInferenceRequest<'a>> {
        let name = fully_qualified_name(model_name, provider_name);
        let needs_filter = request.messages.iter().any(|m| {
            m.content.iter().any(|c| {
                if let ContentBlock::Unknown {
                    model_provider_name,
                    data: _,
                } = c
                {
                    model_provider_name.as_ref().is_some_and(|n| n != &name)
                } else {
                    false
                }
            })
        });
        if needs_filter {
            let new_messages = request
                .messages
                .iter()
                .map(|m| RequestMessage {
                    content: m
                        .content
                        .iter()
                        .flat_map(|c| {
                            if let ContentBlock::Unknown {
                                model_provider_name,
                                data: _,
                            } = c
                            {
                                if model_provider_name.as_ref().is_some_and(|n| n != &name) {
                                    None
                                } else {
                                    Some(c.clone())
                                }
                            } else {
                                Some(c.clone())
                            }
                        })
                        .collect(),
                    ..m.clone()
                })
                .collect();
            Cow::Owned(ModelInferenceRequest {
                messages: new_messages,
                ..request.clone()
            })
        } else {
            Cow::Borrowed(request)
        }
    }
    pub async fn infer<'request>(
        &self,
        request: &'request ModelInferenceRequest<'request>,
        clients: &'request InferenceClients<'request>,
        model_name: &'request str,
    ) -> Result<ModelInferenceResponse, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let request = self.filter_content_blocks(request, model_name, provider_name);
            let model_provider_request = ModelProviderRequest {
                request: &request,
                model_name,
                provider_name,
            };
            let cache_key = model_provider_request.get_cache_key()?;
            // TODO: think about how to best handle errors here
            if clients.cache_options.enabled.read() {
                let cache_lookup = cache_lookup(
                    clients.clickhouse_connection_info,
                    model_provider_request,
                    clients.cache_options.max_age_s,
                )
                .await
                .ok()
                .flatten();
                if let Some(cache_lookup) = cache_lookup {
                    return Ok(cache_lookup);
                }
            }
            let provider = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.to_string(),
                })
            })?;
            let response = provider
                .infer(
                    model_provider_request,
                    clients.http_client,
                    clients.credentials,
                )
                .instrument(span!(
                    Level::INFO,
                    "infer",
                    provider_name = &**provider_name
                ))
                .await;

            match response {
                Ok(response) => {
                    if clients.cache_options.enabled.write() {
                        let _ = start_cache_write(
                            clients.clickhouse_connection_info,
                            cache_key,
                            NonStreamingCacheData {
                                blocks: response.output.clone(),
                            },
                            &response.raw_request,
                            &response.raw_response,
                            &response.usage,
                            response.finish_reason.as_ref(),
                        );
                    }
                    // We already checked the cache above (and returned early if it was a hit), so this response was not from the cache
                    let model_inference_response =
                        ModelInferenceResponse::new(response, provider_name.clone(), false);

                    return Ok(model_inference_response);
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        let err = Error::new(ErrorDetails::ModelProvidersExhausted { provider_errors });
        Err(err)
    }

    pub async fn infer_stream<'request>(
        &self,
        request: &'request ModelInferenceRequest<'request>,
        clients: &'request InferenceClients<'request>,
        model_name: &'request str,
    ) -> Result<(StreamResponse, Vec<RequestMessage>), Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let request = self.filter_content_blocks(request, model_name, provider_name);
            let model_provider_request = ModelProviderRequest {
                request: &request,
                model_name,
                provider_name,
            };
            // TODO: think about how to best handle errors here
            if clients.cache_options.enabled.read() {
                let cache_lookup = cache_lookup_streaming(
                    clients.clickhouse_connection_info,
                    model_provider_request,
                    clients.cache_options.max_age_s,
                )
                .await
                .ok()
                .flatten();
                if let Some(cache_lookup) = cache_lookup {
                    return Ok((cache_lookup, request.messages.clone()));
                }
            }

            let provider = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.to_string(),
                })
            })?;
            let response = provider
                .infer_stream(
                    model_provider_request,
                    clients.http_client,
                    clients.credentials,
                )
                .instrument(span!(
                    Level::INFO,
                    "infer_stream",
                    provider_name = &**provider_name
                ))
                .await;
            match response {
                Ok(response) => {
                    let (stream, raw_request) = response;
                    // Note - we cache the chunks here so that we store the raw model provider input and response chunks
                    // in the cache. We don't want this logic in `collect_chunks`, which would cause us to cache the result
                    // of higher-level transformations (e.g. dicl)
                    let mut stream = if clients.cache_options.enabled.write() {
                        stream_with_cache_write(
                            raw_request.clone(),
                            model_provider_request,
                            clients,
                            stream,
                        )
                        .await?
                    } else {
                        stream
                    };
                    // Get a single chunk from the stream and make sure it is OK then send to client.
                    // We want to do this here so that we can tell that the request is working.
                    peek_first_chunk(&mut stream, &raw_request, provider_name).await?;
                    return Ok((
                        StreamResponse {
                            stream,
                            raw_request,
                            model_provider_name: provider_name.clone(),
                            cached: false,
                        },
                        request.messages.clone(),
                    ));
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::new(ErrorDetails::ModelProvidersExhausted {
            provider_errors,
        }))
    }

    pub async fn start_batch_inference<'request>(
        &self,
        requests: &'request [ModelInferenceRequest<'request>],
        client: &'request Client,
        api_keys: &'request InferenceCredentials,
    ) -> Result<StartBatchModelInferenceResponse, Error> {
        let mut provider_errors: HashMap<String, Error> = HashMap::new();
        for provider_name in &self.routing {
            let provider = self.providers.get(provider_name).ok_or_else(|| {
                Error::new(ErrorDetails::ProviderNotFound {
                    provider_name: provider_name.to_string(),
                })
            })?;
            let response = provider
                .start_batch_inference(requests, client, api_keys)
                .instrument(span!(
                    Level::INFO,
                    "start_batch_inference",
                    provider_name = &**provider_name
                ))
                .await;
            match response {
                Ok(response) => {
                    return Ok(StartBatchModelInferenceResponse::new(
                        response,
                        provider_name.clone(),
                    ));
                }
                Err(error) => {
                    provider_errors.insert(provider_name.to_string(), error);
                }
            }
        }
        Err(Error::new(ErrorDetails::ModelProvidersExhausted {
            provider_errors,
        }))
    }
}

async fn stream_with_cache_write(
    raw_request: String,
    model_request: ModelProviderRequest<'_>,
    clients: &InferenceClients<'_>,
    mut stream: PeekableProviderInferenceResponseStream,
) -> Result<PeekableProviderInferenceResponseStream, Error> {
    let cache_key = model_request.get_cache_key()?;
    let clickhouse_info = clients.clickhouse_connection_info.clone();
    Ok((Box::pin(async_stream::stream! {
        let mut buffer = vec![];
        let mut errored = false;
        while let Some(chunk) = stream.next().await {
            if !errored {
                match chunk.as_ref() {
                    Ok(chunk) => {
                        buffer.push(chunk.clone());
                    }
                    Err(e) => {
                        tracing::warn!("Skipping cache write for stream response due to error in stream: {e}");
                        errored = true;
                    }
                }
            }
            yield chunk;
        }
        if !errored {
            let usage = consolidate_usage(&buffer);
            let _ = start_cache_write_streaming(
                &clickhouse_info,
                cache_key,
                buffer,
                &raw_request,
                &usage,
            );
        }
    }) as ProviderInferenceResponseStreamInner).peekable())
}

fn consolidate_usage(chunks: &[ProviderInferenceResponseChunk]) -> Usage {
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks {
        if let Some(usage) = &chunk.usage {
            input_tokens += usage.input_tokens;
            output_tokens += usage.output_tokens;
        }
    }
    Usage {
        input_tokens,
        output_tokens,
    }
}

#[derive(Debug, Deserialize)]
pub struct UninitializedModelProvider {
    #[serde(flatten)]
    pub config: ProviderConfig,
    pub extra_body: Option<ExtraBodyConfig>,
    pub extra_headers: Option<ExtraHeadersConfig>,
}

#[derive(Debug)]
pub struct ModelProvider {
    pub name: Arc<str>,
    pub config: ProviderConfig,
    pub extra_headers: Option<ExtraHeadersConfig>,
    pub extra_body: Option<ExtraBodyConfig>,
}

impl From<&ModelProvider> for ModelProviderRequestInfo {
    fn from(val: &ModelProvider) -> Self {
        ModelProviderRequestInfo {
            provider_name: val.name.clone(),
            extra_headers: val.extra_headers.clone(),
            extra_body: val.extra_body.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelProviderRequestInfo {
    pub provider_name: Arc<str>,
    pub extra_headers: Option<ExtraHeadersConfig>,
    pub extra_body: Option<ExtraBodyConfig>,
}

#[derive(Debug)]
pub enum ProviderConfig {
    Anthropic(AnthropicProvider),
    AWSBedrock(AWSBedrockProvider),
    Azure(AzureProvider),
    Fireworks(FireworksProvider),
    GCPVertexAnthropic(GCPVertexAnthropicProvider),
    GCPVertexGemini(GCPVertexGeminiProvider),
    GoogleAIStudioGemini(GoogleAIStudioGeminiProvider),
    Hyperbolic(HyperbolicProvider),
    Mistral(MistralProvider),
    OpenAI(OpenAIProvider),
    Together(TogetherProvider),
    VLLM(VLLMProvider),
    XAI(XAIProvider),
    TGI(TGIProvider),
    SGLang(SGLangProvider),
    DeepSeek(DeepSeekProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
}

/// Helper struct for deserializing the ProviderConfig.
/// This is necessary because we want to load environment variables as we deserialize
/// and we need to be able to deserialize the correct one based on the "type" field.
/// Use the ProviderConfig struct for all post-initialization logic.
#[derive(TensorZeroDeserialize, VariantNames)]
#[strum(serialize_all = "lowercase")]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub(super) enum ProviderConfigHelper {
    Anthropic {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    #[strum(serialize = "aws_bedrock")]
    #[serde(rename = "aws_bedrock")]
    AWSBedrock {
        model_id: String,
        region: Option<String>,
        #[serde(default)]
        allow_auto_detect_region: bool,
    },
    Azure {
        deployment_id: String,
        endpoint: Url,
        api_key_location: Option<CredentialLocation>,
    },
    #[strum(serialize = "gcp_vertex_anthropic")]
    #[serde(rename = "gcp_vertex_anthropic")]
    GCPVertexAnthropic {
        model_id: String,
        location: String,
        project_id: String,
        credential_location: Option<CredentialLocation>,
    },
    #[strum(serialize = "gcp_vertex_gemini")]
    #[serde(rename = "gcp_vertex_gemini")]
    GCPVertexGemini {
        model_id: String,
        location: String,
        project_id: String,
        credential_location: Option<CredentialLocation>,
    },
    #[strum(serialize = "google_ai_studio_gemini")]
    #[serde(rename = "google_ai_studio_gemini")]
    GoogleAIStudioGemini {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    Hyperbolic {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    #[strum(serialize = "fireworks")]
    #[serde(rename = "fireworks")]
    Fireworks {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    Mistral {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    OpenAI {
        model_name: String,
        api_base: Option<Url>,
        api_key_location: Option<CredentialLocation>,
    },
    Together {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
        #[serde(default = "crate::inference::providers::together::default_parse_think_blocks")]
        parse_think_blocks: bool,
    },
    #[allow(clippy::upper_case_acronyms)]
    VLLM {
        model_name: String,
        api_base: Url,
        api_key_location: Option<CredentialLocation>,
    },
    #[allow(clippy::upper_case_acronyms)]
    XAI {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    #[allow(clippy::upper_case_acronyms)]
    TGI {
        api_base: Url,
        api_key_location: Option<CredentialLocation>,
    },
    #[allow(clippy::upper_case_acronyms)]
    SGLang {
        model_name: String,
        api_base: Url,
        api_key_location: Option<CredentialLocation>,
    },
    #[allow(clippy::upper_case_acronyms)]
    DeepSeek {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy {
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    },
}

impl<'de> Deserialize<'de> for ProviderConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = ProviderConfigHelper::deserialize(deserializer)?;
        Ok(match helper {
            ProviderConfigHelper::Anthropic {
                model_name,
                api_key_location,
            } => ProviderConfig::Anthropic(
                AnthropicProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::AWSBedrock {
                model_id,
                region,
                allow_auto_detect_region,
            } => {
                let region = region.map(aws_types::region::Region::new);
                if region.is_none() && !allow_auto_detect_region {
                    return Err(D::Error::custom("AWS bedrock provider requires a region to be provided, or `allow_auto_detect_region = true`."));
                }

                // NB: We need to make an async call here to initialize the AWS Bedrock client.

                let provider = tokio::task::block_in_place(move || {
                    tokio::runtime::Handle::current()
                        .block_on(async { AWSBedrockProvider::new(model_id, region).await })
                        .map_err(|e| {
                            serde::de::Error::custom(format!(
                                "Failed to initialize AWS Bedrock provider: {e}"
                            ))
                        })
                })?;

                ProviderConfig::AWSBedrock(provider)
            }
            ProviderConfigHelper::Azure {
                deployment_id,
                endpoint,
                api_key_location,
            } => ProviderConfig::Azure(
                AzureProvider::new(deployment_id, endpoint, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::Fireworks {
                model_name,
                api_key_location,
            } => ProviderConfig::Fireworks(
                FireworksProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::GCPVertexAnthropic {
                model_id,
                location,
                project_id,
                credential_location: api_key_location,
            } => ProviderConfig::GCPVertexAnthropic(
                GCPVertexAnthropicProvider::new(model_id, location, project_id, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::GCPVertexGemini {
                model_id,
                location,
                project_id,
                credential_location: api_key_location,
            } => ProviderConfig::GCPVertexGemini(
                GCPVertexGeminiProvider::new(model_id, location, project_id, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::GoogleAIStudioGemini {
                model_name,
                api_key_location,
            } => ProviderConfig::GoogleAIStudioGemini(
                GoogleAIStudioGeminiProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::Hyperbolic {
                model_name,
                api_key_location,
            } => ProviderConfig::Hyperbolic(
                HyperbolicProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::Mistral {
                model_name,
                api_key_location,
            } => ProviderConfig::Mistral(
                MistralProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::OpenAI {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::OpenAI(
                OpenAIProvider::new(model_name, api_base, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::Together {
                model_name,
                api_key_location,
                parse_think_blocks,
            } => ProviderConfig::Together(
                TogetherProvider::new(model_name, api_key_location, parse_think_blocks)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::VLLM {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::VLLM(
                VLLMProvider::new(model_name, api_base, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::XAI {
                model_name,
                api_key_location,
            } => ProviderConfig::XAI(
                XAIProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::SGLang {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::SGLang(
                SGLangProvider::new(model_name, api_base, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::TGI {
                api_base,
                api_key_location,
            } => ProviderConfig::TGI(
                TGIProvider::new(api_base, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            ProviderConfigHelper::DeepSeek {
                model_name,
                api_key_location,
            } => ProviderConfig::DeepSeek(
                DeepSeekProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfigHelper::Dummy {
                model_name,
                api_key_location,
            } => ProviderConfig::Dummy(
                DummyProvider::new(model_name, api_key_location)
                    .map_err(|e| D::Error::custom(e.to_string()))?,
            ),
        })
    }
}

impl ModelProvider {
    async fn infer(
        &self,
        request: ModelProviderRequest<'_>,
        client: &Client,
        api_keys: &InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        match &self.config {
            ProviderConfig::Anthropic(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::Azure(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::Fireworks(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::Mistral(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::OpenAI(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::Together(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::SGLang(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            ProviderConfig::VLLM(provider) => provider.infer(request, client, api_keys, self).await,
            ProviderConfig::XAI(provider) => provider.infer(request, client, api_keys, self).await,
            ProviderConfig::TGI(provider) => provider.infer(request, client, api_keys, self).await,
            ProviderConfig::DeepSeek(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider.infer(request, client, api_keys, self).await
            }
        }
    }

    async fn infer_stream(
        &self,
        request: ModelProviderRequest<'_>,
        client: &Client,
        api_keys: &InferenceCredentials,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        match &self.config {
            ProviderConfig::Anthropic(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::Azure(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::Fireworks(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::Mistral(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::OpenAI(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::Together(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::SGLang(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::XAI(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::VLLM(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::TGI(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            ProviderConfig::DeepSeek(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider.infer_stream(request, client, api_keys, self).await
            }
        }
    }

    async fn start_batch_inference<'a>(
        &self,
        requests: &'a [ModelInferenceRequest<'a>],
        client: &'a Client,
        api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        match &self.config {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::SGLang(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::DeepSeek(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            ProviderConfig::TGI(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .start_batch_inference(requests, client, api_keys)
                    .await
            }
        }
    }

    pub async fn poll_batch_inference<'a>(
        &self,
        batch_request: &'a BatchRequestRow<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        match &self.config {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::TGI(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::SGLang(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            ProviderConfig::DeepSeek(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .poll_batch_inference(batch_request, http_client, dynamic_api_keys)
                    .await
            }
        }
    }
}

pub enum CredentialLocation {
    /// Environment variable containing the actual credential
    Env(String),
    /// Environment variable containing the path to a credential file
    PathFromEnv(String),
    /// For dynamic credential resolution
    Dynamic(String),
    /// Direct path to a credential file
    Path(String),
    None,
}

impl<'de> Deserialize<'de> for CredentialLocation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if let Some(inner) = s.strip_prefix("env::") {
            Ok(CredentialLocation::Env(inner.to_string()))
        } else if let Some(inner) = s.strip_prefix("path_from_env::") {
            Ok(CredentialLocation::PathFromEnv(inner.to_string()))
        } else if let Some(inner) = s.strip_prefix("dynamic::") {
            Ok(CredentialLocation::Dynamic(inner.to_string()))
        } else if let Some(inner) = s.strip_prefix("path::") {
            Ok(CredentialLocation::Path(inner.to_string()))
        } else if s == "none" {
            Ok(CredentialLocation::None)
        } else {
            Err(serde::de::Error::custom(format!(
                "Invalid ApiKeyLocation format: {}",
                s
            )))
        }
    }
}

#[derive(Clone)]
pub enum Credential {
    Static(SecretString),
    FileContents(SecretString),
    Dynamic(String),
    None,
    Missing,
}

/// Builds a credential type from the provided `CredentialLocation` and default location.
/// This is a convenience function that calls `get_creds_with_cache_and_fn` using
/// the `TryFrom<Credential>` implementation for `T`.
///
/// Most providers should be able to use this function to build their credentials,
/// unless they have special requirements (e.g. calling an `async fn`)
pub fn build_creds_caching_default<T: Clone + TryFrom<Credential, Error = Error>>(
    location: Option<CredentialLocation>,
    default_location: CredentialLocation,
    provider_type: &str,
    cache: &OnceLock<T>,
) -> Result<T, Error> {
    build_creds_caching_default_with_fn(location, default_location, provider_type, cache, |creds| {
        T::try_from(creds)
    })
}

/// Builds a credential type from the provided `CredentialLocation` and default location.
/// If the location is `None`, we'll use the provided `OnceLock` to cache the result
/// of `f(default_location)`.
/// Otherwise, we'll call `f(location)` without caching the result.
///
/// **NOTE** - `f` may be run multiple times in parallel even when `default_location` is used,
/// due to a limitation of the current `OnceLock` api.
pub fn build_creds_caching_default_with_fn<T: Clone, F: FnOnce(Credential) -> Result<T, Error>>(
    location: Option<CredentialLocation>,
    default_location: CredentialLocation,
    provider_type: &str,
    cache: &OnceLock<T>,
    f: F,
) -> Result<T, Error> {
    let make_creds = |location| {
        let creds = Credential::try_from((location, provider_type))?;
        let provider_creds = f(creds)?;
        Ok(provider_creds)
    };
    if let Some(location) = location {
        make_creds(location)
    } else {
        racy_get_or_try_init(cache, || make_creds(default_location))
    }
}

/// Gets the value from a `OnceLock` or initializes it with the result of `f`
/// If this is called simultaneously from multiple threads, it may call `f` multiple times
/// If `f` returns an error, the `OnceLock` will remain uninitialized
fn racy_get_or_try_init<T: Clone, E>(
    once_lock: &OnceLock<T>,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    if let Some(val) = once_lock.get() {
        Ok(val.clone())
    } else {
        let val = f()?;
        // We don't care if the value was est
        let _ = once_lock.set(val.clone());
        Ok(val)
    }
}

impl TryFrom<(CredentialLocation, &str)> for Credential {
    type Error = Error;
    #[allow(unused_variables)]
    fn try_from(
        (location, provider_type): (CredentialLocation, &str),
    ) -> Result<Self, Self::Error> {
        match location {
            CredentialLocation::Env(key_name) => match env::var(key_name) {
                Ok(value) => Ok(Credential::Static(SecretString::from(value))),
                Err(_) => {
                    if SKIP_CREDENTIAL_VALIDATION.is_set() {
                        #[cfg(any(test, feature = "e2e_tests"))]
                        {
                            warn!(
                            "You are missing the credentials required for a model provider of type {}, so the associated tests will likely fail.",
                            provider_type
                        );
                        }
                        Ok(Credential::Missing)
                    } else {
                        Err(Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: provider_type.to_string(),
                        }))
                    }
                }
            },
            CredentialLocation::PathFromEnv(env_key) => {
                // First get the path from environment variable
                let path = match env::var(&env_key) {
                    Ok(path) => path,
                    Err(_) => {
                        if SKIP_CREDENTIAL_VALIDATION.is_set() {
                            #[cfg(any(test, feature = "e2e_tests"))]
                            {
                                warn!(
                                "Environment variable {} is required for a model provider of type {} but is missing, so the associated tests will likely fail.",
                                env_key, provider_type

                            );
                            }
                            return Ok(Credential::Missing);
                        } else {
                            return Err(Error::new(ErrorDetails::ApiKeyMissing {
                                provider_name: format!(
                                    "{}: Environment variable {} for credentials path is missing",
                                    provider_type, env_key
                                ),
                            }));
                        }
                    }
                };
                // Then read the file contents
                match fs::read_to_string(path) {
                    Ok(contents) => Ok(Credential::FileContents(SecretString::from(contents))),
                    Err(e) => {
                        if SKIP_CREDENTIAL_VALIDATION.is_set() {
                            #[cfg(any(test, feature = "e2e_tests"))]
                            {
                                warn!(
                                "Failed to read credentials file for a model provider of type {}, so the associated tests will likely fail: {}",
                                provider_type, e
                            );
                            }
                            Ok(Credential::Missing)
                        } else {
                            Err(Error::new(ErrorDetails::ApiKeyMissing {
                                provider_name: format!(
                                    "{}: Failed to read credentials file - {}",
                                    provider_type, e
                                ),
                            }))
                        }
                    }
                }
            }
            CredentialLocation::Path(path) => match fs::read_to_string(path) {
                Ok(contents) => Ok(Credential::FileContents(SecretString::from(contents))),
                Err(e) => {
                    if SKIP_CREDENTIAL_VALIDATION.is_set() {
                        #[cfg(any(test, feature = "e2e_tests"))]
                        {
                            warn!(
                                "Failed to read credentials file for a model provider of type {}, so the associated tests will likely fail: {}",
                            provider_type, e
                        );
                        }
                        Ok(Credential::Missing)
                    } else {
                        Err(Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: format!(
                                "{}: Failed to read credentials file - {}",
                                provider_type, e
                            ),
                        }))
                    }
                }
            },
            CredentialLocation::Dynamic(key_name) => Ok(Credential::Dynamic(key_name.clone())),
            CredentialLocation::None => Ok(Credential::None),
        }
    }
}

const SHORTHAND_MODEL_PREFIXES: &[&str] = &[
    "anthropic::",
    "deepseek::",
    "fireworks::",
    "google_ai_studio_gemini::",
    "hyperbolic::",
    "mistral::",
    "openai::",
    "together::",
    "xai::",
    "dummy::",
];

pub type ModelTable = BaseModelTable<ModelConfig>;

impl ShorthandModelConfig for ModelConfig {
    const SHORTHAND_MODEL_PREFIXES: &[&str] = SHORTHAND_MODEL_PREFIXES;
    const MODEL_TYPE: &str = "Model";
    fn from_shorthand(provider_type: &str, model_name: &str) -> Result<Self, Error> {
        let model_name = model_name.to_string();
        let provider_config = match provider_type {
            "anthropic" => ProviderConfig::Anthropic(AnthropicProvider::new(model_name, None)?),
            "deepseek" => ProviderConfig::DeepSeek(DeepSeekProvider::new(model_name, None)?),
            "fireworks" => ProviderConfig::Fireworks(FireworksProvider::new(model_name, None)?),
            "google_ai_studio_gemini" => ProviderConfig::GoogleAIStudioGemini(
                GoogleAIStudioGeminiProvider::new(model_name, None)?,
            ),
            "hyperbolic" => ProviderConfig::Hyperbolic(HyperbolicProvider::new(model_name, None)?),
            "mistral" => ProviderConfig::Mistral(MistralProvider::new(model_name, None)?),
            "openai" => ProviderConfig::OpenAI(OpenAIProvider::new(model_name, None, None)?),
            "together" => ProviderConfig::Together(TogetherProvider::new(
                model_name,
                None,
                crate::inference::providers::together::default_parse_think_blocks(),
            )?),
            "xai" => ProviderConfig::XAI(XAIProvider::new(model_name, None)?),
            #[cfg(any(test, feature = "e2e_tests"))]
            "dummy" => ProviderConfig::Dummy(DummyProvider::new(model_name, None)?),
            _ => {
                return Err(ErrorDetails::Config {
                    message: format!("Invalid provider type: {}", provider_type),
                }
                .into());
            }
        };
        Ok(ModelConfig {
            routing: vec![provider_type.to_string().into()],
            providers: HashMap::from([(
                provider_type.to_string().into(),
                ModelProvider {
                    name: provider_type.into(),
                    config: provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        })
    }

    fn validate(&self, model_name: &str) -> Result<(), Error> {
        // Ensure that the model has at least one provider
        if self.routing.is_empty() {
            return Err(ErrorDetails::Config {
                message: format!("`models.{model_name}`: `routing` must not be empty"),
            }
            .into());
        }

        // Ensure that routing entries are unique and exist as keys in providers
        let mut seen_providers = std::collections::HashSet::new();
        for provider in &self.routing {
            if provider.starts_with("tensorzero::") {
                return Err(ErrorDetails::Config {
                    message: format!("`models.{model_name}.routing`: Provider name cannot start with 'tensorzero::': {provider}"),
                }
                .into());
            }
            if !seen_providers.insert(provider) {
                return Err(ErrorDetails::Config {
                    message: format!("`models.{model_name}.routing`: duplicate entry `{provider}`"),
                }
                .into());
            }

            if !self.providers.contains_key(provider) {
                return Err(ErrorDetails::Config {
            message: format!(
                "`models.{model_name}`: `routing` contains entry `{provider}` that does not exist in `providers`"
            ),
        }
        .into());
            }
        }

        // Validate each provider
        for provider_name in self.providers.keys() {
            if !seen_providers.contains(provider_name) {
                return Err(ErrorDetails::Config {
                    message: format!(
                "`models.{model_name}`: Provider `{provider_name}` is not listed in `routing`"
            ),
                }
                .into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, cell::Cell};

    use crate::cache::CacheEnabledMode;
    use crate::tool::{ToolCallConfig, ToolChoice};
    use crate::{
        cache::CacheOptions,
        clickhouse::ClickHouseConnectionInfo,
        inference::{
            providers::dummy::{
                DummyCredentials, DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW,
                DUMMY_INFER_USAGE, DUMMY_STREAMING_RESPONSE,
            },
            types::{ContentBlockChunk, FunctionType, ModelInferenceRequestJsonMode, TextChunk},
        },
        model_table::RESERVED_MODEL_PREFIXES,
    };
    use secrecy::SecretString;
    use tokio_stream::StreamExt;
    use tracing_test::traced_test;
    use uuid::Uuid;

    use super::*;

    #[tokio::test]
    async fn test_model_config_infer_routing() {
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            credentials: DummyCredentials::None,
        });
        let model_config = ModelConfig {
            routing: vec!["good_provider".into()],
            providers: HashMap::from([(
                "good_provider".into(),
                ModelProvider {
                    name: "good_provider".into(),
                    config: good_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        };
        let api_keys = InferenceCredentials::default();
        let http_client = Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let clients = InferenceClients {
            http_client: &http_client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };

        // Try inferring the good model only
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let model_name = "test model";
        let response = model_config
            .infer(&request, &clients, model_name)
            .await
            .unwrap();
        let content = response.output;
        assert_eq!(
            content,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
        let raw = response.raw_response;
        assert_eq!(raw, DUMMY_INFER_RESPONSE_RAW);
        let usage = response.usage;
        assert_eq!(usage, DUMMY_INFER_USAGE);
        assert_eq!(&*response.model_provider_name, "good_provider");

        // Try inferring the bad model
        let model_config = ModelConfig {
            routing: vec!["error".into()],
            providers: HashMap::from([(
                "error".into(),
                ModelProvider {
                    name: "error".into(),
                    config: bad_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };
        let response = model_config
            .infer(&request, &clients, model_name)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "error".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Error sending request to Dummy provider for model 'error'."
                            .to_string(),
                        status_code: None,
                        provider_type: "dummy".to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                    }
                    .into()
                )])
            }
            .into()
        );
    }

    #[tokio::test]
    #[traced_test]
    async fn test_model_config_infer_routing_fallback() {
        // Test that fallback works with bad --> good model provider

        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            credentials: DummyCredentials::None,
        });
        let api_keys = InferenceCredentials::default();
        let http_client = Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let clients = InferenceClients {
            http_client: &http_client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };
        // Try inferring the good model only
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let model_config = ModelConfig {
            routing: vec![
                "error_provider".to_string().into(),
                "good_provider".to_string().into(),
            ],
            providers: HashMap::from([
                (
                    "error_provider".to_string().into(),
                    ModelProvider {
                        name: "error_provider".into(),
                        config: bad_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                    },
                ),
                (
                    "good_provider".to_string().into(),
                    ModelProvider {
                        name: "good_provider".into(),
                        config: good_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                    },
                ),
            ]),
        };

        let model_name = "test model";
        let response = model_config
            .infer(&request, &clients, model_name)
            .await
            .unwrap();
        // Ensure that the error for the bad provider was logged, but the request worked nonetheless
        assert!(logs_contain(
            "Error sending request to Dummy provider for model 'error'."
        ));
        let content = response.output;
        assert_eq!(
            content,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
        let raw = response.raw_response;
        assert_eq!(raw, DUMMY_INFER_RESPONSE_RAW);
        let usage = response.usage;
        assert_eq!(usage, DUMMY_INFER_USAGE);
        assert_eq!(&*response.model_provider_name, "good_provider");
    }

    #[tokio::test]
    async fn test_model_config_infer_stream_routing() {
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            credentials: DummyCredentials::None,
        });
        let api_keys = InferenceCredentials::default();
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Test good model
        let model_config = ModelConfig {
            routing: vec!["good_provider".to_string().into()],
            providers: HashMap::from([(
                "good_provider".to_string().into(),
                ModelProvider {
                    name: "good_provider".into(),
                    config: good_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };
        let (
            StreamResponse {
                mut stream,
                raw_request,
                model_provider_name,
                cached: _,
            },
            _input,
        ) = model_config
            .infer_stream(
                &request,
                &InferenceClients {
                    http_client: &Client::new(),
                    clickhouse_connection_info: &ClickHouseConnectionInfo::Disabled,
                    credentials: &api_keys,
                    cache_options: &CacheOptions {
                        max_age_s: None,
                        enabled: CacheEnabledMode::Off,
                    },
                },
                "my_model",
            )
            .await
            .unwrap();
        let initial_chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(
            initial_chunk.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })],
        );
        assert_eq!(raw_request, "raw request");
        assert_eq!(&*model_provider_name, "good_provider");
        let mut collected_content: Vec<ContentBlockChunk> =
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })];
        let mut stream = Box::pin(stream);
        while let Some(Ok(chunk)) = stream.next().await {
            let mut content = chunk.content;
            assert!(content.len() <= 1);
            if content.len() == 1 {
                collected_content.push(content.pop().unwrap());
            }
        }
        let mut collected_content_str = String::new();
        for content in collected_content {
            match content {
                ContentBlockChunk::Text(text) => collected_content_str.push_str(&text.text),
                _ => panic!("Expected a text content block"),
            }
        }
        assert_eq!(collected_content_str, DUMMY_STREAMING_RESPONSE.join(""));

        // Test bad model
        let model_config = ModelConfig {
            routing: vec!["error".to_string().into()],
            providers: HashMap::from([(
                "error".to_string().into(),
                ModelProvider {
                    name: "error".to_string().into(),
                    config: bad_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };
        let response = model_config
            .infer_stream(
                &request,
                &InferenceClients {
                    http_client: &Client::new(),
                    clickhouse_connection_info: &ClickHouseConnectionInfo::Disabled,
                    credentials: &api_keys,
                    cache_options: &CacheOptions {
                        max_age_s: None,
                        enabled: CacheEnabledMode::Off,
                    },
                },
                "my_model",
            )
            .await;
        assert!(response.is_err());
        let error = match response {
            Err(error) => error,
            Ok(_) => panic!("Expected error, got Ok(_)"),
        };
        assert_eq!(
            error,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "error".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Error sending request to Dummy provider for model 'error'."
                            .to_string(),
                        status_code: None,
                        provider_type: "dummy".to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                    }
                    .into()
                )])
            }
            .into()
        );
    }

    #[tokio::test]
    #[traced_test]
    async fn test_model_config_infer_stream_routing_fallback() {
        // Test that fallback works with bad --> good model provider (streaming)

        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            credentials: DummyCredentials::None,
        });
        let bad_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            credentials: DummyCredentials::None,
        });
        let api_keys = InferenceCredentials::default();
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        // Test fallback
        let model_config = ModelConfig {
            routing: vec!["error_provider".into(), "good_provider".into()],
            providers: HashMap::from([
                (
                    "error_provider".to_string().into(),
                    ModelProvider {
                        name: "error_provider".to_string().into(),
                        config: bad_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                    },
                ),
                (
                    "good_provider".to_string().into(),
                    ModelProvider {
                        name: "good_provider".to_string().into(),
                        config: good_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                    },
                ),
            ]),
        };
        let (
            StreamResponse {
                mut stream,
                raw_request,
                model_provider_name,
                cached: _,
            },
            _input,
        ) = model_config
            .infer_stream(
                &request,
                &InferenceClients {
                    http_client: &Client::new(),
                    clickhouse_connection_info: &ClickHouseConnectionInfo::Disabled,
                    credentials: &api_keys,
                    cache_options: &CacheOptions {
                        max_age_s: None,
                        enabled: CacheEnabledMode::Off,
                    },
                },
                "my_model",
            )
            .await
            .unwrap();
        let initial_chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(&*model_provider_name, "good_provider");
        // Ensure that the error for the bad provider was logged, but the request worked nonetheless
        assert!(logs_contain(
            "Error sending request to Dummy provider for model 'error'"
        ));
        assert_eq!(raw_request, "raw request");

        assert_eq!(
            initial_chunk.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string(),
            })],
        );

        let mut collected_content = initial_chunk.content;
        let mut stream = Box::pin(stream);
        while let Some(Ok(chunk)) = stream.next().await {
            let mut content = chunk.content;
            assert!(content.len() <= 1);
            if content.len() == 1 {
                collected_content.push(content.pop().unwrap());
            }
        }
        let mut collected_content_str = String::new();
        for content in collected_content {
            match content {
                ContentBlockChunk::Text(text) => collected_content_str.push_str(&text.text),
                _ => panic!("Expected a text content block"),
            }
        }
        assert_eq!(collected_content_str, DUMMY_STREAMING_RESPONSE.join(""));
    }

    #[tokio::test]
    async fn test_dynamic_api_keys() {
        let provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "test_key".into(),
            credentials: DummyCredentials::Dynamic("TEST_KEY".to_string()),
        });
        let model_config = ModelConfig {
            routing: vec!["model".into()],
            providers: HashMap::from([(
                "model".into(),
                ModelProvider {
                    name: "model".into(),
                    config: provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        };
        let api_keys = InferenceCredentials::default();
        let http_client = Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let clients = InferenceClients {
            http_client: &http_client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let model_name = "test model";
        let error = model_config
            .infer(&request, &clients, model_name)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "model".to_string(),
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Dummy".to_string()
                    }
                    .into()
                )])
            }
            .into()
        );

        let api_keys = HashMap::from([(
            "TEST_KEY".to_string(),
            SecretString::from("notgoodkey".to_string()),
        )]);
        let clients = InferenceClients {
            http_client: &http_client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };
        let response = model_config
            .infer(&request, &clients, model_name)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "model".to_string(),
                    ErrorDetails::InferenceClient {
                        message: "Invalid API key for Dummy provider".to_string(),
                        status_code: None,
                        provider_type: "dummy".to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                    }
                    .into()
                )])
            }
            .into()
        );

        let provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "test_key".into(),
            credentials: DummyCredentials::Dynamic("TEST_KEY".to_string()),
        });
        let model_config = ModelConfig {
            routing: vec!["model".to_string().into()],
            providers: HashMap::from([(
                "model".to_string().into(),
                ModelProvider {
                    name: "model".to_string().into(),
                    config: provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        };
        let api_keys = InferenceCredentials::default();
        let http_client = Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
        let clients = InferenceClients {
            http_client: &http_client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };

        let request = ModelInferenceRequest {
            messages: vec![],
            inference_id: Uuid::now_v7(),
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let error = model_config
            .infer(&request, &clients, model_name)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "model".to_string(),
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Dummy".to_string()
                    }
                    .into()
                )])
            }
            .into()
        );

        let api_keys = HashMap::from([(
            "TEST_KEY".to_string(),
            SecretString::from("good_key".to_string()),
        )]);
        let clients = InferenceClients {
            http_client: &http_client,
            clickhouse_connection_info: &clickhouse_connection_info,
            credentials: &api_keys,
            cache_options: &CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
        };
        let response = model_config
            .infer(&request, &clients, model_name)
            .await
            .unwrap();
        assert_eq!(
            response.output,
            vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
        );
    }

    #[test]
    fn test_validate_or_create_model_config() {
        let model_table = ModelTable::default();
        // Test that we can get or create a model config
        model_table.validate("dummy::gpt-4o").unwrap();
        // Shorthand models are not added to the model table
        assert_eq!(model_table.static_model_len(), 0);
        let model_config = model_table
            .get("dummy::gpt-4o")
            .unwrap()
            .expect("Missing dummy model");
        assert_eq!(model_config.routing, vec!["dummy".into()]);
        let provider_config = &model_config.providers.get("dummy").unwrap().config;
        match provider_config {
            ProviderConfig::Dummy(provider) => assert_eq!(&*provider.model_name, "gpt-4o"),
            _ => panic!("Expected Dummy provider"),
        }

        // Test that it fails if the model is not well-formed
        let model_config = model_table.validate("foo::bar");
        assert!(model_config.is_err());
        assert_eq!(
            model_config.unwrap_err(),
            ErrorDetails::Config {
                message: "Model name 'foo::bar' not found in model table".to_string()
            }
            .into()
        );
        // Test that it works with an initialized model
        let anthropic_provider_config = SKIP_CREDENTIAL_VALIDATION.set(&(), || {
            ProviderConfig::Anthropic(AnthropicProvider::new("claude".to_string(), None).unwrap())
        });
        let anthropic_model_config = ModelConfig {
            routing: vec!["anthropic".into()],
            providers: HashMap::from([(
                "anthropic".into(),
                ModelProvider {
                    name: "anthropic".into(),
                    config: anthropic_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                },
            )]),
        };
        let model_table: ModelTable = HashMap::from([("claude".into(), anthropic_model_config)])
            .try_into()
            .unwrap();

        model_table.validate("dummy::claude").unwrap();
    }

    #[test]
    fn test_shorthand_prefixes_subset_of_reserved() {
        for &shorthand in SHORTHAND_MODEL_PREFIXES {
            assert!(
                RESERVED_MODEL_PREFIXES.contains(&shorthand.to_string()),
                "Shorthand prefix '{}' is not in RESERVED_MODEL_PREFIXES",
                shorthand
            );
        }
    }

    #[test]
    fn test_racy_get_or_try_init() {
        let lock: OnceLock<bool> = OnceLock::new();

        // If the closure returns an error, `racy_get_or_try_init` should return an error
        racy_get_or_try_init(&lock, || {
            Err::<_, Box<dyn std::error::Error>>("Test error".into())
        })
        .expect_err("Test error");
        assert!(
            lock.get().is_none(),
            "OnceLock was initialized after an error"
        );

        racy_get_or_try_init(&lock, || Ok::<_, Box<dyn std::error::Error>>(true))
            .expect("racy_get_or_try_init should succeed with successful closure");

        assert_eq!(lock.get(), Some(&true));
    }

    #[test]
    fn test_cache_default_creds() {
        let make_creds_call_count = Cell::new(0);
        let make_creds = |_| {
            make_creds_call_count.set(make_creds_call_count.get() + 1);
            Ok(())
        };

        let cache = OnceLock::new();

        build_creds_caching_default_with_fn(
            None,
            CredentialLocation::None,
            "test",
            &cache,
            make_creds,
        )
        .expect("Failed to build creds");
        // The first call should initialize the OnceLock, and call `make_creds`
        assert_eq!(make_creds_call_count.get(), 1);
        assert_eq!(cache.get(), Some(&()));

        // Subsequent calls should not call `make_creds`
        build_creds_caching_default_with_fn(
            None,
            CredentialLocation::None,
            "test",
            &cache,
            make_creds,
        )
        .expect("Failed to build creds");
        assert_eq!(make_creds_call_count.get(), 1);
    }

    #[test]
    fn test_dont_cache_non_default_creds() {
        let make_creds_call_count = Cell::new(0);
        let make_creds = |_| {
            make_creds_call_count.set(make_creds_call_count.get() + 1);
            Ok(())
        };

        let cache = OnceLock::new();

        // When we provide a `Some(credential_location)`, we should not cache the creds.
        build_creds_caching_default_with_fn(
            Some(CredentialLocation::None),
            CredentialLocation::None,
            "test",
            &cache,
            make_creds,
        )
        .expect("Failed to build creds");

        assert_eq!(cache.get(), None);
        assert_eq!(make_creds_call_count.get(), 1);
    }
}
