use futures::future::try_join_all;
use futures::StreamExt;
use indexmap::IndexMap;
use secrecy::SecretString;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use strum::VariantNames;
use tensorzero_derive::TensorZeroDeserialize;
use tokio::time::error::Elapsed;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{span, Level, Span};
use tracing_futures::{Instrument, Instrumented};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use url::Url;

use crate::cache::{
    cache_lookup, cache_lookup_streaming, start_cache_write, start_cache_write_streaming,
    CacheData, CacheValidationInfo, ModelProviderRequest, NonStreamingCacheData,
    StreamingCacheData,
};
use crate::config::{
    provider_types::ProviderTypesConfig, OtlpConfig, OtlpTracesFormat, TimeoutsConfig,
};
use crate::endpoints::inference::InferenceClients;
use crate::http::TensorzeroHttpClient;
use crate::model_table::ProviderKind;
use crate::providers::aws_sagemaker::AWSSagemakerProvider;
#[cfg(any(test, feature = "e2e_tests"))]
use crate::providers::dummy::DummyProvider;
use crate::providers::google_ai_studio_gemini::GoogleAIStudioGeminiProvider;

use crate::inference::types::batch::{
    BatchRequestRow, PollBatchInferenceResponse, StartBatchModelInferenceResponse,
    StartBatchProviderInferenceResponse,
};
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::extra_headers::ExtraHeadersConfig;
use crate::inference::types::{
    current_timestamp, ContentBlock, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponseChunk, ProviderInferenceResponseStreamInner, RequestMessage, Thought,
    Unknown, Usage,
};
use crate::inference::WrappedProvider;
use crate::model_table::{
    AnthropicKind, AzureKind, BaseModelTable, DeepSeekKind, FireworksKind,
    GoogleAIStudioGeminiKind, GroqKind, HyperbolicKind, MistralKind, OpenAIKind, OpenRouterKind,
    ProviderTypeDefaultCredentials, SGLangKind, ShorthandModelConfig, TGIKind, TogetherKind,
    VLLMKind, XAIKind,
};
use crate::providers::helpers::peek_first_chunk;
use crate::providers::hyperbolic::HyperbolicProvider;
use crate::providers::openai::OpenAIAPIType;
use crate::providers::sglang::SGLangProvider;
use crate::providers::tgi::TGIProvider;
use crate::rate_limiting::{RateLimitResourceUsage, TicketBorrows};
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::{
        types::{ModelInferenceRequest, ModelInferenceResponse, ProviderInferenceResponse},
        InferenceProvider,
    },
};
use serde::{Deserialize, Serialize};

use crate::providers::{
    anthropic::AnthropicProvider, aws_bedrock::AWSBedrockProvider, azure::AzureProvider,
    deepseek::DeepSeekProvider, fireworks::FireworksProvider,
    gcp_vertex_anthropic::GCPVertexAnthropicProvider, gcp_vertex_gemini::GCPVertexGeminiProvider,
    groq::GroqProvider, mistral::MistralProvider, openai::OpenAIProvider,
    openrouter::OpenRouterProvider, together::TogetherProvider, vllm::VLLMProvider,
    xai::XAIProvider,
};

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ModelConfig {
    pub routing: Vec<Arc<str>>, // [provider name A, provider name B, ...]
    pub providers: HashMap<Arc<str>, ModelProvider>, // provider name => provider config
    pub timeouts: TimeoutsConfig,
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct UninitializedModelConfig {
    pub routing: Vec<Arc<str>>, // [provider name A, provider name B, ...]
    pub providers: HashMap<Arc<str>, UninitializedModelProvider>, // provider name => provider config
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
}

impl UninitializedModelConfig {
    pub async fn load(
        self,
        model_name: &str,
        provider_types: &ProviderTypesConfig,
        provider_type_default_credentials: &ProviderTypeDefaultCredentials,
        http_client: TensorzeroHttpClient,
    ) -> Result<ModelConfig, Error> {
        // We want `ModelProvider` to know its own name (from the 'providers' config section).
        // We first deserialize to `HashMap<Arc<str>, UninitializedModelProvider>`, and then
        // build `ModelProvider`s using the name keys from the map.
        let providers = try_join_all(self.providers.into_iter().map(|(name, provider)| {
            let http_client = http_client.clone();
            async move {
                Ok::<_, Error>((
                    name.clone(),
                    ModelProvider {
                        name: name.clone(),
                        config: provider
                            .config
                            .load(
                                provider_types,
                                provider_type_default_credentials,
                                http_client,
                            )
                            .await
                            .map_err(|e| {
                                Error::new(ErrorDetails::Config {
                                    message: format!("models.{model_name}.providers.{name}: {e}"),
                                })
                            })?,
                        extra_body: provider.extra_body,
                        extra_headers: provider.extra_headers,
                        timeouts: provider.timeouts,
                        discard_unknown_chunks: provider.discard_unknown_chunks,
                    },
                ))
            }
        }))
        .await?
        .into_iter()
        .collect::<HashMap<_, _>>();
        Ok(ModelConfig {
            routing: self.routing,
            providers,
            timeouts: self.timeouts,
        })
    }
}

pub struct StreamResponse {
    pub stream: Instrumented<PeekableProviderInferenceResponseStream>,
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
                        // Use the real usage (so that the `ModelInference` row we write is accurate)
                        // The usage returned to over HTTP is adjusted in `InferenceResponseChunk::new`
                        usage: c.usage,
                        // We didn't make any network calls to the model provider, so the latency is 0
                        latency: Duration::from_secs(0),
                        // For all chunks but the last one, the finish reason is None
                        // For the last chunk, the finish reason is the same as the cache lookup
                        finish_reason: if index == chunks_len - 1 {
                            cache_lookup.finish_reason
                        } else {
                            None
                        },
                    })
                },
            ))) as ProviderInferenceResponseStreamInner)
                .peekable()
                .instrument(tracing::info_span!(
                    "stream_from_cache",
                    otel.name = "stream_from_cache"
                )),
            raw_request: cache_lookup.raw_request,
            model_provider_name,
            cached: true,
        }
    }
}

/// Creates a fully-qualified name from a model and provider name.
/// This format was previously used in `ContentBlock::Unknown.model_provider_name`
/// and is still used for the deprecated `DynamicExtraBody::Provider` variant.
pub fn fully_qualified_name(model_name: &str, provider_name: &str) -> String {
    format!("tensorzero::model_name::{model_name}::provider_name::{provider_name}")
}

impl ModelConfig {
    /// Checks if an Unknown content block should be filtered out based on model_name and provider_name.
    /// Returns true if the block should be filtered (removed), false if it should be kept.
    fn should_filter_unknown_block(
        block_model_name: &Option<String>,
        block_provider_name: &Option<String>,
        target_model_name: &str,
        target_provider_name: &str,
    ) -> bool {
        // If model_name is specified and doesn't match, filter it out
        if let Some(ref m) = block_model_name {
            if m != target_model_name {
                return true;
            }
        }
        // If provider_name is specified and doesn't match, filter it out
        if let Some(ref p) = block_provider_name {
            if p != target_provider_name {
                return true;
            }
        }
        // Keep the block if both match (or are None)
        false
    }

    fn filter_content_blocks<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model_name: &str,
        provider: &ModelProvider,
    ) -> Cow<'a, ModelInferenceRequest<'a>> {
        let provider_name = provider.name.as_ref();
        let needs_filter = request.messages.iter().any(|m| {
            m.content.iter().any(|c| match c {
                ContentBlock::Unknown(Unknown {
                    model_name: block_model_name,
                    provider_name: block_provider_name,
                    data: _,
                }) => Self::should_filter_unknown_block(
                    block_model_name,
                    block_provider_name,
                    model_name,
                    provider_name,
                ),
                ContentBlock::Thought(Thought {
                    text: _,
                    signature: _,
                    summary: _,
                    provider_type,
                }) => provider_type
                    .as_ref()
                    .is_some_and(|t| t != &provider.config.thought_block_provider_type()),
                _ => false,
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
                        .flat_map(|c| match c {
                            ContentBlock::Unknown(Unknown {
                                model_name: block_model_name,
                                provider_name: block_provider_name,
                                data: _,
                            }) => {
                                if Self::should_filter_unknown_block(
                                    block_model_name,
                                    block_provider_name,
                                    model_name,
                                    provider_name,
                                ) {
                                    None
                                } else {
                                    Some(c.clone())
                                }
                            }
                            ContentBlock::Thought(Thought {
                                text: _,
                                signature: _,
                                summary: _,
                                provider_type,
                            }) => {
                                // When a thought is scoped to a particular provider type, we discard
                                // if it doesn't match our target provider.
                                // Thoughts without a `provider_type` are used for all providers.
                                if provider_type.as_ref().is_some_and(|t| {
                                    t != &provider.config.thought_block_provider_type()
                                }) {
                                    None
                                } else {
                                    Some(c.clone())
                                }
                            }
                            _ => Some(c.clone()),
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

    /// Performs a non-streaming request to a specific provider, performing a cache lookup if enabled.
    /// We apply model-provider timeouts to the future produced by this function
    /// (as we want to apply the timeout to ClickHouse cache lookups)
    async fn non_streaming_provider_request<'request>(
        &self,
        model_provider_request: ModelProviderRequest<'request>,
        provider: &'request ModelProvider,
        clients: &InferenceClients,
    ) -> Result<ModelInferenceResponse, Error> {
        // TODO: think about how to best handle errors here
        if clients.cache_options.enabled.read() {
            let cache_lookup = cache_lookup(
                &clients.clickhouse_connection_info,
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
        let response = provider
            .infer(model_provider_request, clients)
            .instrument(span!(
                Level::INFO,
                "infer",
                provider_name = model_provider_request.provider_name
            ))
            .await?;
        // We already checked the cache above (and returned early if it was a hit), so this response was not from the cache
        Ok(ModelInferenceResponse::new(
            response,
            model_provider_request.provider_name.into(),
            false,
        ))
    }

    /// Performs a streaming request to a specific provider, performing a cache lookup if enabled.
    /// We apply model-provider timeouts to the future produced by this function
    /// (as we want to apply the timeout to ClickHouse cache lookups).
    ///
    /// This function also includes a call to `peek_first_chunk` - this ensure that the
    /// duration of the returned future includes the time taken to get the first chunk
    /// from the model provider.
    async fn streaming_provider_request<'request>(
        &self,
        model_provider_request: ModelProviderRequest<'request>,
        provider: &'request ModelProvider,
        clients: &InferenceClients,
    ) -> Result<StreamResponseAndMessages, Error> {
        // TODO: think about how to best handle errors here
        if clients.cache_options.enabled.read() {
            let cache_lookup = cache_lookup_streaming(
                &clients.clickhouse_connection_info,
                model_provider_request,
                clients.cache_options.max_age_s,
            )
            .await
            .ok()
            .flatten();
            if let Some(cache_lookup) = cache_lookup {
                return Ok(StreamResponseAndMessages {
                    response: cache_lookup,
                    messages: model_provider_request.request.messages.clone(),
                });
            }
        }

        let StreamAndRawRequest {
            stream,
            raw_request,
            ticket_borrow,
        } = provider
            .infer_stream(model_provider_request, clients)
            .await?;

        // Note - we cache the chunks here so that we store the raw model provider input and response chunks
        // in the cache. We don't want this logic in `collect_chunks`, which would cause us to cache the result
        // of higher-level transformations (e.g. dicl)
        let write_to_cache = clients.cache_options.enabled.write();
        let span = stream.span().clone();
        let mut stream = wrap_provider_stream(
            raw_request.clone(),
            model_provider_request,
            ticket_borrow,
            clients,
            stream,
            write_to_cache,
        )
        .await?
        .instrument(span);
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        peek_first_chunk(
            stream.inner_mut(),
            &raw_request,
            model_provider_request.provider_name,
        )
        .await?;
        Ok(StreamResponseAndMessages {
            response: StreamResponse {
                stream,
                raw_request,
                model_provider_name: model_provider_request.provider_name.into(),
                cached: false,
            },
            messages: model_provider_request.request.messages.clone(),
        })
    }

    #[tracing::instrument(skip_all, fields(model_name = model_name, otel.name = "model_inference", stream = false))]
    pub async fn infer<'request>(
        &self,
        request: &'request ModelInferenceRequest<'request>,
        clients: &InferenceClients,
        model_name: &'request str,
    ) -> Result<ModelInferenceResponse, Error> {
        let span = tracing::Span::current();
        clients.otlp_config.mark_openinference_chain_span(&span);

        let mut provider_errors: IndexMap<String, Error> = IndexMap::new();
        let run_all_models = async {
            for provider_name in &self.routing {
                let provider = self.providers.get(provider_name).ok_or_else(|| {
                    Error::new(ErrorDetails::ProviderNotFound {
                        provider_name: provider_name.to_string(),
                    })
                })?;
                let request = Self::filter_content_blocks(request, model_name, provider);
                let model_provider_request = ModelProviderRequest {
                    request: &request,
                    model_name,
                    provider_name,
                    otlp_config: &clients.otlp_config,
                };
                let cache_key = model_provider_request.get_cache_key()?;

                let response_fut =
                    self.non_streaming_provider_request(model_provider_request, provider, clients);
                let response = if let Some(timeout) = provider.non_streaming_total_timeout() {
                    tokio::time::timeout(timeout, response_fut)
                        .await
                        // Convert the outer `Elapsed` error into a TensorZero error,
                        // so that it can be handled by the `match response` block below
                        .unwrap_or_else(|_: Elapsed| {
                            Err(Error::new(ErrorDetails::ModelProviderTimeout {
                                provider_name: provider_name.to_string(),
                                timeout,
                                streaming: false,
                            }))
                        })
                } else {
                    response_fut.await
                };

                match response {
                    Ok(response) => {
                        // Perform the cache write outside of the `non_streaming_total_timeout` timeout future,
                        // (in case we ever add a blocking cache write option)
                        if !response.cached && clients.cache_options.enabled.write() {
                            let _ = start_cache_write(
                                &clients.clickhouse_connection_info,
                                cache_key,
                                CacheData {
                                    output: NonStreamingCacheData {
                                        blocks: response.output.clone(),
                                    },
                                    raw_request: response.raw_request.clone(),
                                    raw_response: response.raw_response.clone(),
                                    input_tokens: response.usage.input_tokens,
                                    output_tokens: response.usage.output_tokens,
                                    finish_reason: response.finish_reason,
                                },
                                CacheValidationInfo {
                                    tool_config: request
                                        .tool_config
                                        .clone()
                                        .map(std::borrow::Cow::into_owned),
                                },
                            );
                        }

                        return Ok(response);
                    }
                    Err(error) => {
                        provider_errors.insert(provider_name.to_string(), error);
                    }
                }
            }
            Err(Error::new(ErrorDetails::ModelProvidersExhausted {
                provider_errors,
            }))
        };
        // This is the top-level model timeout, which limits the total time taken to run all providers.
        // Some of the providers may themselves have timeouts, which is fine. Provider timeouts
        // are treated as just another kind of provider error - a timeout of N ms is equivalent
        // to a provider taking N ms, and then producing a normal HTTP error.
        if let Some(timeout) = self.timeouts.non_streaming.total_ms {
            let timeout = Duration::from_millis(timeout);
            tokio::time::timeout(timeout, run_all_models)
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
            run_all_models.await
        }
    }

    #[tracing::instrument(skip_all, fields(model_name = model_name, otel.name = "model_inference", stream = true))]
    pub async fn infer_stream<'request>(
        &self,
        request: &'request ModelInferenceRequest<'request>,
        clients: &InferenceClients,
        model_name: &'request str,
    ) -> Result<StreamResponseAndMessages, Error> {
        clients
            .otlp_config
            .mark_openinference_chain_span(&tracing::Span::current());
        let mut provider_errors: IndexMap<String, Error> = IndexMap::new();
        let run_all_models = async {
            for provider_name in &self.routing {
                let provider = self.providers.get(provider_name).ok_or_else(|| {
                    Error::new(ErrorDetails::ProviderNotFound {
                        provider_name: provider_name.to_string(),
                    })
                })?;
                let request = Self::filter_content_blocks(request, model_name, provider);
                let model_provider_request = ModelProviderRequest {
                    request: &request,
                    model_name,
                    provider_name,
                    otlp_config: &clients.otlp_config,
                };

                // This future includes a call to `peek_first_chunk`, so applying
                // `streaming_ttft_timeout` is correct.
                let response_fut =
                    self.streaming_provider_request(model_provider_request, provider, clients);
                let response = if let Some(timeout) = provider.streaming_ttft_timeout() {
                    tokio::time::timeout(timeout, response_fut)
                        .await
                        .unwrap_or_else(|_: Elapsed| {
                            Err(Error::new(ErrorDetails::ModelProviderTimeout {
                                provider_name: provider_name.to_string(),
                                timeout,
                                streaming: true,
                            }))
                        })
                } else {
                    response_fut.await
                };

                match response {
                    Ok(response) => return Ok(response),
                    Err(error) => {
                        provider_errors.insert(provider_name.to_string(), error);
                    }
                }
            }
            Err(Error::new(ErrorDetails::ModelProvidersExhausted {
                provider_errors,
            }))
        };
        // See the corresponding `non_streaming.total_ms` timeout in the `infer`
        // method above for more details.
        if let Some(timeout) = self.timeouts.streaming.ttft_ms {
            let timeout = Duration::from_millis(timeout);
            tokio::time::timeout(timeout, run_all_models)
                .await
                // Convert the outer `Elapsed` error into a TensorZero error,
                // so that it can be handled by the `match response` block below
                .unwrap_or_else(|_: Elapsed| {
                    Err(Error::new(ErrorDetails::ModelTimeout {
                        model_name: model_name.to_string(),
                        timeout,
                        streaming: true,
                    }))
                })
        } else {
            run_all_models.await
        }
    }

    pub async fn start_batch_inference<'request>(
        &self,
        requests: &'request [ModelInferenceRequest<'request>],
        client: &'request TensorzeroHttpClient,
        api_keys: &'request InferenceCredentials,
    ) -> Result<StartBatchModelInferenceResponse, Error> {
        let mut provider_errors: IndexMap<String, Error> = IndexMap::new();
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

/// Wraps a low-level model provider stream, adding in common functionality:
/// * Model inference cache writes
/// * OpenTelemetry usage attributes
///
/// This is used for functionality that needs access to individual chunks, which requires
/// us to wrap the underlying stream.
async fn wrap_provider_stream(
    raw_request: String,
    model_request: ModelProviderRequest<'_>,
    ticket_borrow: TicketBorrows,
    clients: &InferenceClients,
    stream: Instrumented<PeekableProviderInferenceResponseStream>,
    write_to_cache: bool,
) -> Result<PeekableProviderInferenceResponseStream, Error> {
    // Detach the span from the stream, and re-attach it to the 'async_stream::stream!' wrapper
    // This ensures that the span duration include the entire provider-specific processing time
    let span = stream.span().clone();
    let mut stream = stream.into_inner();
    let cache_key = model_request.get_cache_key()?;
    let clickhouse_info = clients.clickhouse_connection_info.clone();
    let tool_config = model_request
        .request
        .tool_config
        .clone()
        .map(std::borrow::Cow::into_owned);
    let otlp_config = clients.otlp_config.clone();
    let postgres_connection_info = clients.postgres_connection_info.clone();
    let deferred_tasks = clients.deferred_tasks.clone();
    let base_stream = async_stream::stream! {
        let mut buffer = vec![];
        let mut errored = false;
        // `total_usage` is `None` until we receive a chunk with usage information
        let mut total_usage: Option<Usage> = None;
        while let Some(chunk) = stream.next().await {
            if let Ok(chunk) = chunk.as_ref() {
                if let Some(chunk_usage) = &chunk.usage {
                    // `total_usage` will be `None` if this is the first chunk with usage information....
                    if total_usage.is_none() {
                        // ... so initialize it to zero ...
                        total_usage = Some(Usage::zero());
                    }
                    // ...and then add the chunk usage to it (handling `None` fields)
                    if let Some(ref mut u) = total_usage { u.sum_strict(chunk_usage); }
                }
            }
            // We can skip cloning the chunk if we know we're not going to write to the cache
            if write_to_cache && !errored {
                match chunk.as_ref() {
                    Ok(chunk) => {
                        buffer.push(chunk.clone());
                    }
                    Err(e) => {
                        tracing::warn!("Skipping cache write for stream response due to error in stream: {e}");
                        errored = true;
                        // If we see a `FatalStreamError`, then yield it and stop processing the stream,
                        // to avoid holding open a stream that might never produce more chunks.
                        // We'll still compute rate-limiting usage using all of the chunks that we've seen so far.
                        if let ErrorDetails::FatalStreamError { .. } = e.get_details() {
                            yield chunk;
                            break;
                        }
                    }
                }
            }
            yield chunk;
        }

        // If we don't see a chunk with usage information, set `total_usage` to the default value (fields as `None`)
        let total_usage = total_usage.unwrap_or_default();

        otlp_config.apply_usage_to_model_provider_span(&span, &total_usage);
        // Make sure that we finish updating rate-limiting tickets if the gateway shuts down
        deferred_tasks.spawn(async move {
            let usage = match (total_usage.total_tokens(), errored) {
                (Some(tokens), false) => {
                    RateLimitResourceUsage::Exact {
                        model_inferences: 1,
                        tokens: tokens as u64,
                    }
                }
                _ => {
                    RateLimitResourceUsage::UnderEstimate {
                        model_inferences: 1,
                        tokens: total_usage.total_tokens().unwrap_or(0) as u64,
                    }
                }
            };

            if let Err(e) = ticket_borrow
                .return_tickets(&postgres_connection_info, usage)
                .await
            {
                tracing::error!("Failed to return rate limit tickets: {}", e);
            }
        }.instrument(span));


        if write_to_cache && !errored {
            let _ = start_cache_write_streaming(
                &clickhouse_info,
                cache_key,
                buffer,
                &raw_request,
                &total_usage,
                tool_config
            );
        }
    };
    // We unconditionally create a stream, and forward items into it from a separate task
    // This ensures that we keep processing chunks (and call `return_tickets` to update rate-limiting information)
    // even if the top-level HTTP request is later dropped.
    let (send, recv) = tokio::sync::mpsc::unbounded_channel();
    // Make sure that we finish processing the stream (so that we call `return_tickets` to update rate-limiting information)
    // if the gateway shuts down.
    clients.deferred_tasks.spawn(async move {
        futures::pin_mut!(base_stream);
        while let Some(chunk) = base_stream.next().await {
            // Intentionally ignore errors - the receiver might be dropped, but we want to keep polling
            // `base_stream` anyway (so that we compute the final usage and call `return_tickets`)
            let _ = send.send(chunk);
        }
    });
    Ok(
        (UnboundedReceiverStream::new(recv).boxed() as ProviderInferenceResponseStreamInner)
            .peekable(),
    )
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct UninitializedModelProvider {
    #[serde(flatten)]
    pub config: UninitializedProviderConfig,
    #[cfg_attr(test, ts(skip))]
    pub extra_body: Option<ExtraBodyConfig>,
    #[cfg_attr(test, ts(skip))]
    pub extra_headers: Option<ExtraHeadersConfig>,
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    /// If `true`, we emit a warning and discard chunks that we don't recognize
    /// (on a best-effort, per-provider basis).
    /// By default, we produce an error in the stream
    /// We can't meaningfully return unknown chunks to the user, as we don't
    /// know how to correctly merge them.
    #[serde(default)]
    pub discard_unknown_chunks: bool,
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ModelProvider {
    pub name: Arc<str>,
    pub config: ProviderConfig,
    #[cfg_attr(test, ts(skip))]
    pub extra_headers: Option<ExtraHeadersConfig>,
    #[cfg_attr(test, ts(skip))]
    pub extra_body: Option<ExtraBodyConfig>,
    pub timeouts: TimeoutsConfig,
    /// See `UninitializedModelProvider.discard_unknown_chunks`.
    pub discard_unknown_chunks: bool,
}

impl ModelProvider {
    fn validate(&self, global_outbound_http_timeout: &chrono::Duration) -> Result<(), Error> {
        self.timeouts.validate(global_outbound_http_timeout)?;
        Ok(())
    }
    fn non_streaming_total_timeout(&self) -> Option<Duration> {
        Some(Duration::from_millis(self.timeouts.non_streaming.total_ms?))
    }

    fn streaming_ttft_timeout(&self) -> Option<Duration> {
        Some(Duration::from_millis(self.timeouts.streaming.ttft_ms?))
    }

    /// The name to report in the OTEL `gen_ai.system` attribute
    fn genai_system_name(&self) -> &'static str {
        match &self.config {
            ProviderConfig::Anthropic(_) => "anthropic",
            ProviderConfig::AWSBedrock(_) => "aws_bedrock",
            ProviderConfig::AWSSagemaker(_) => "aws_sagemaker",
            ProviderConfig::Azure(_) => "azure",
            ProviderConfig::Fireworks(_) => "fireworks",
            ProviderConfig::GCPVertexAnthropic(_) => "gcp_vertex_anthropic",
            ProviderConfig::GCPVertexGemini(_) => "gcp_vertex_gemini",
            ProviderConfig::GoogleAIStudioGemini(_) => "google_ai_studio_gemini",
            ProviderConfig::Groq(_) => "groq",
            ProviderConfig::Hyperbolic(_) => "hyperbolic",
            ProviderConfig::Mistral(_) => "mistral",
            ProviderConfig::OpenAI(_) => "openai",
            ProviderConfig::OpenRouter(_) => "openrouter",
            ProviderConfig::Together(_) => "together",
            ProviderConfig::VLLM(_) => "vllm",
            ProviderConfig::XAI(_) => "xai",
            ProviderConfig::TGI(_) => "tgi",
            ProviderConfig::SGLang(_) => "sglang",
            ProviderConfig::DeepSeek(_) => "deepseek",
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(_) => "dummy",
        }
    }

    /// The model name to report in the OTEL `gen_ai.request.model` attribute
    fn genai_model_name(&self) -> Option<&str> {
        match &self.config {
            ProviderConfig::Anthropic(provider) => Some(provider.model_name()),
            ProviderConfig::AWSBedrock(provider) => Some(provider.model_id()),
            // SageMaker doesn't have a meaningful model name concept, as we just invoke an endpoint
            ProviderConfig::AWSSagemaker(_) => None,
            ProviderConfig::Azure(provider) => Some(provider.deployment_id()),
            ProviderConfig::Fireworks(provider) => Some(provider.model_name()),
            ProviderConfig::GCPVertexAnthropic(provider) => Some(provider.model_id()),
            ProviderConfig::GCPVertexGemini(provider) => Some(provider.model_or_endpoint_id()),
            ProviderConfig::GoogleAIStudioGemini(provider) => Some(provider.model_name()),
            ProviderConfig::Groq(provider) => Some(provider.model_name()),
            ProviderConfig::Hyperbolic(provider) => Some(provider.model_name()),
            ProviderConfig::Mistral(provider) => Some(provider.model_name()),
            ProviderConfig::OpenAI(provider) => Some(provider.model_name()),
            ProviderConfig::OpenRouter(provider) => Some(provider.model_name()),
            ProviderConfig::Together(provider) => Some(provider.model_name()),
            ProviderConfig::VLLM(provider) => Some(provider.model_name()),
            ProviderConfig::XAI(provider) => Some(provider.model_name()),
            // TGI doesn't have a meaningful model name
            ProviderConfig::TGI(_) => None,
            ProviderConfig::SGLang(provider) => Some(provider.model_name()),
            ProviderConfig::DeepSeek(provider) => Some(provider.model_name()),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => Some(provider.model_name()),
        }
    }
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

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum ProviderConfig {
    Anthropic(AnthropicProvider),
    #[serde(rename = "aws_bedrock")]
    AWSBedrock(AWSBedrockProvider),
    #[serde(rename = "aws_sagemaker")]
    AWSSagemaker(AWSSagemakerProvider),
    Azure(AzureProvider),
    DeepSeek(DeepSeekProvider),
    Fireworks(FireworksProvider),
    #[serde(rename = "gcp_vertex_anthropic")]
    GCPVertexAnthropic(GCPVertexAnthropicProvider),
    #[serde(rename = "gcp_vertex_gemini")]
    GCPVertexGemini(GCPVertexGeminiProvider),
    #[serde(rename = "google_ai_studio_gemini")]
    GoogleAIStudioGemini(GoogleAIStudioGeminiProvider),
    Groq(GroqProvider),
    Hyperbolic(HyperbolicProvider),
    Mistral(MistralProvider),
    OpenAI(OpenAIProvider),
    OpenRouter(OpenRouterProvider),
    #[serde(rename = "sglang")]
    SGLang(SGLangProvider),
    #[serde(rename = "tgi")]
    TGI(TGIProvider),
    Together(TogetherProvider),
    #[serde(rename = "vllm")]
    VLLM(VLLMProvider),
    #[serde(rename = "xai")]
    XAI(XAIProvider),
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy(DummyProvider),
}

impl ProviderConfig {
    fn thought_block_provider_type(&self) -> Cow<'static, str> {
        match self {
            ProviderConfig::Anthropic(_) => {
                Cow::Borrowed(crate::providers::anthropic::PROVIDER_TYPE)
            }
            ProviderConfig::AWSBedrock(_) => {
                Cow::Borrowed(crate::providers::aws_bedrock::PROVIDER_TYPE)
            }
            // Note - none of our current  wrapped provider types emit thought blocks
            // If any of them ever start producing thoughts, we'll need to make sure that the `provider_type`
            // field uses `thought_block_provider_type` on the parent SageMaker provider.
            ProviderConfig::AWSSagemaker(sagemaker) => Cow::Owned(format!(
                "aws_sagemaker::{}",
                sagemaker
                    .hosted_provider
                    .thought_block_provider_type_suffix()
            )),
            ProviderConfig::Azure(_) => Cow::Borrowed(crate::providers::azure::PROVIDER_TYPE),
            ProviderConfig::DeepSeek(_) => Cow::Borrowed(crate::providers::deepseek::PROVIDER_TYPE),
            ProviderConfig::Fireworks(_) => {
                Cow::Borrowed(crate::providers::fireworks::PROVIDER_TYPE)
            }
            ProviderConfig::GCPVertexAnthropic(_) => {
                Cow::Borrowed(crate::providers::gcp_vertex_anthropic::PROVIDER_TYPE)
            }
            ProviderConfig::GCPVertexGemini(_) => {
                Cow::Borrowed(crate::providers::gcp_vertex_gemini::PROVIDER_TYPE)
            }
            ProviderConfig::GoogleAIStudioGemini(_) => {
                Cow::Borrowed(crate::providers::google_ai_studio_gemini::PROVIDER_TYPE)
            }
            ProviderConfig::Groq(_) => Cow::Borrowed(crate::providers::groq::PROVIDER_TYPE),
            ProviderConfig::Hyperbolic(_) => {
                Cow::Borrowed(crate::providers::hyperbolic::PROVIDER_TYPE)
            }
            ProviderConfig::Mistral(_) => Cow::Borrowed(crate::providers::mistral::PROVIDER_TYPE),
            ProviderConfig::OpenAI(_) => Cow::Borrowed(crate::providers::openai::PROVIDER_TYPE),
            ProviderConfig::OpenRouter(_) => {
                Cow::Borrowed(crate::providers::openrouter::PROVIDER_TYPE)
            }
            ProviderConfig::SGLang(_) => Cow::Borrowed(crate::providers::sglang::PROVIDER_TYPE),
            ProviderConfig::TGI(_) => Cow::Borrowed(crate::providers::tgi::PROVIDER_TYPE),
            ProviderConfig::Together(_) => Cow::Borrowed(crate::providers::together::PROVIDER_TYPE),
            ProviderConfig::VLLM(_) => Cow::Borrowed(crate::providers::vllm::PROVIDER_TYPE),
            ProviderConfig::XAI(_) => Cow::Borrowed(crate::providers::xai::PROVIDER_TYPE),
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(_) => Cow::Borrowed(crate::providers::dummy::PROVIDER_TYPE),
        }
    }
}

/// Contains all providers which implement `SelfHostedProvider` - these providers
/// can be used as the target provider hosted by AWS Sagemaker
#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum HostedProviderKind {
    OpenAI,
    TGI,
}

#[derive(ts_rs::TS)]
#[ts(export)]
#[derive(Debug, TensorZeroDeserialize, VariantNames, Serialize)]
#[strum(serialize_all = "lowercase")]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
#[serde(deny_unknown_fields)]
pub enum UninitializedProviderConfig {
    Anthropic {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_base: Option<Url>,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
        #[serde(default)]
        beta_structured_outputs: bool,
    },
    #[strum(serialize = "aws_bedrock")]
    #[serde(rename = "aws_bedrock")]
    AWSBedrock {
        model_id: String,
        region: Option<String>,
        #[serde(default)]
        allow_auto_detect_region: bool,
    },
    #[strum(serialize = "aws_sagemaker")]
    #[serde(rename = "aws_sagemaker")]
    AWSSagemaker {
        endpoint_name: String,
        model_name: String,
        region: Option<String>,
        #[serde(default)]
        allow_auto_detect_region: bool,
        hosted_provider: HostedProviderKind,
    },
    Azure {
        deployment_id: String,
        endpoint: EndpointLocation,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    #[strum(serialize = "gcp_vertex_anthropic")]
    #[serde(rename = "gcp_vertex_anthropic")]
    GCPVertexAnthropic {
        model_id: String,
        location: String,
        project_id: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        credential_location: Option<CredentialLocationWithFallback>,
    },
    #[strum(serialize = "gcp_vertex_gemini")]
    #[serde(rename = "gcp_vertex_gemini")]
    GCPVertexGemini {
        model_id: Option<String>,
        endpoint_id: Option<String>,
        location: String,
        project_id: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        credential_location: Option<CredentialLocationWithFallback>,
    },
    #[strum(serialize = "google_ai_studio_gemini")]
    #[serde(rename = "google_ai_studio_gemini")]
    GoogleAIStudioGemini {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    #[strum(serialize = "groq")]
    #[serde(rename = "groq")]
    Groq {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    Hyperbolic {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    #[strum(serialize = "fireworks")]
    #[serde(rename = "fireworks")]
    Fireworks {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
        #[serde(default = "crate::providers::fireworks::default_parse_think_blocks")]
        parse_think_blocks: bool,
    },
    Mistral {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    OpenAI {
        model_name: String,
        api_base: Option<Url>,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
        #[serde(default)]
        api_type: OpenAIAPIType,
        #[serde(default)]
        include_encrypted_reasoning: bool,
        #[serde(default)]
        provider_tools: Vec<Value>,
    },
    OpenRouter {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    Together {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
        #[serde(default = "crate::providers::together::default_parse_think_blocks")]
        parse_think_blocks: bool,
    },
    VLLM {
        model_name: String,
        api_base: Url,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    XAI {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    TGI {
        api_base: Url,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    SGLang {
        model_name: String,
        api_base: Url,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    DeepSeek {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
    #[cfg(any(test, feature = "e2e_tests"))]
    Dummy {
        model_name: String,
        #[cfg_attr(test, ts(type = "string | null"))]
        api_key_location: Option<CredentialLocationWithFallback>,
    },
}

impl UninitializedProviderConfig {
    pub async fn load(
        self,
        provider_types: &ProviderTypesConfig,
        provider_type_default_credentials: &ProviderTypeDefaultCredentials,
        http_client: TensorzeroHttpClient,
    ) -> Result<ProviderConfig, Error> {
        Ok(match self {
            UninitializedProviderConfig::Anthropic {
                model_name,
                api_base,
                api_key_location,
                beta_structured_outputs,
            } => ProviderConfig::Anthropic(AnthropicProvider::new(
                model_name,
                api_base,
                AnthropicKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
                beta_structured_outputs,
            )),
            UninitializedProviderConfig::AWSBedrock {
                model_id,
                region,
                allow_auto_detect_region,
            } => {
                let region = region.map(aws_types::region::Region::new);
                if region.is_none() && !allow_auto_detect_region {
                    return Err(Error::new(ErrorDetails::Config { message: "AWS bedrock provider requires a region to be provided, or `allow_auto_detect_region = true`.".to_string() }));
                }

                ProviderConfig::AWSBedrock(
                    AWSBedrockProvider::new(model_id, region, http_client).await?,
                )
            }
            UninitializedProviderConfig::AWSSagemaker {
                endpoint_name,
                region,
                allow_auto_detect_region,
                model_name,
                hosted_provider,
            } => {
                let region = region.map(aws_types::region::Region::new);
                if region.is_none() && !allow_auto_detect_region {
                    return Err(Error::new(ErrorDetails::Config { message: "AWS Sagemaker provider requires a region to be provided, or `allow_auto_detect_region = true`.".to_string() }));
                }

                let self_hosted: Box<dyn WrappedProvider + Send + Sync + 'static> =
                    match hosted_provider {
                        HostedProviderKind::OpenAI => Box::new(OpenAIProvider::new(
                            model_name,
                            None,

                            OpenAIKind
                                .get_defaulted_credential(
                                    Some(&CredentialLocationWithFallback::Single(CredentialLocation::None)),
                                    provider_type_default_credentials,
                                )
                                .await?,
                            // TODO - decide how to expose the responses api for wrapped providers
                            OpenAIAPIType::ChatCompletions,
                            false,
                            Vec::new(),
                            )?),
                        HostedProviderKind::TGI => Box::new(TGIProvider::new(
                            Url::parse("http://tensorzero-unreachable-domain-please-file-a-bug-report.invalid").map_err(|e| {
                                Error::new(ErrorDetails::InternalError { message: format!("Failed to parse fake TGI endpoint: `{e}`. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new") })
                            })?,
                            TGIKind
                                .get_defaulted_credential(
                                    Some(&CredentialLocationWithFallback::Single(CredentialLocation::None)),
                                    provider_type_default_credentials,
                                )
                                .await?,
                        )),
                    };

                ProviderConfig::AWSSagemaker(
                    AWSSagemakerProvider::new(endpoint_name, self_hosted, region).await?,
                )
            }
            UninitializedProviderConfig::Azure {
                deployment_id,
                endpoint,
                api_key_location,
            } => ProviderConfig::Azure(AzureProvider::new(
                deployment_id,
                endpoint,
                AzureKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )?),
            UninitializedProviderConfig::Fireworks {
                model_name,
                api_key_location,
                parse_think_blocks,
            } => ProviderConfig::Fireworks(FireworksProvider::new(
                model_name,
                FireworksKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
                parse_think_blocks,
            )),
            UninitializedProviderConfig::GCPVertexAnthropic {
                model_id,
                location,
                project_id,
                credential_location: api_key_location,
            } => ProviderConfig::GCPVertexAnthropic(
                GCPVertexAnthropicProvider::new(
                    model_id,
                    location,
                    project_id,
                    api_key_location,
                    provider_type_default_credentials,
                )
                .await?,
            ),
            UninitializedProviderConfig::GCPVertexGemini {
                model_id,
                endpoint_id,
                location,
                project_id,
                credential_location: api_key_location,
            } => {
                let provider = GCPVertexGeminiProvider::new(
                    model_id,
                    endpoint_id,
                    location,
                    project_id,
                    api_key_location,
                    provider_types,
                    provider_type_default_credentials,
                )
                .await?;

                ProviderConfig::GCPVertexGemini(provider)
            }
            UninitializedProviderConfig::GoogleAIStudioGemini {
                model_name,
                api_key_location,
            } => ProviderConfig::GoogleAIStudioGemini(GoogleAIStudioGeminiProvider::new(
                model_name,
                GoogleAIStudioGeminiKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )?),
            UninitializedProviderConfig::Groq {
                model_name,
                api_key_location,
            } => ProviderConfig::Groq(GroqProvider::new(
                model_name,
                GroqKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::Hyperbolic {
                model_name,
                api_key_location,
            } => ProviderConfig::Hyperbolic(HyperbolicProvider::new(
                model_name,
                HyperbolicKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::Mistral {
                model_name,
                api_key_location,
            } => ProviderConfig::Mistral(MistralProvider::new(
                model_name,
                MistralKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::OpenAI {
                model_name,
                api_base,
                api_key_location,
                api_type,
                include_encrypted_reasoning,
                provider_tools,
            } => {
                // This should only be used when we are mocking batch inferences, otherwise defer to the API base set
                #[cfg(feature = "e2e_tests")]
                let api_base = provider_types
                    .openai
                    .batch_inference_api_base
                    .clone()
                    .or(api_base);

                ProviderConfig::OpenAI(OpenAIProvider::new(
                    model_name,
                    api_base,
                    OpenAIKind
                        .get_defaulted_credential(
                            api_key_location.as_ref(),
                            provider_type_default_credentials,
                        )
                        .await?,
                    api_type,
                    include_encrypted_reasoning,
                    provider_tools,
                )?)
            }
            UninitializedProviderConfig::OpenRouter {
                model_name,
                api_key_location,
            } => ProviderConfig::OpenRouter(OpenRouterProvider::new(
                model_name,
                OpenRouterKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::Together {
                model_name,
                api_key_location,
                parse_think_blocks,
            } => ProviderConfig::Together(TogetherProvider::new(
                model_name,
                TogetherKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
                parse_think_blocks,
            )),
            UninitializedProviderConfig::VLLM {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::VLLM(VLLMProvider::new(
                model_name,
                api_base,
                VLLMKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::XAI {
                model_name,
                api_key_location,
            } => ProviderConfig::XAI(XAIProvider::new(
                model_name,
                XAIKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::SGLang {
                model_name,
                api_base,
                api_key_location,
            } => ProviderConfig::SGLang(SGLangProvider::new(
                model_name,
                api_base,
                SGLangKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::TGI {
                api_base,
                api_key_location,
            } => ProviderConfig::TGI(TGIProvider::new(
                api_base,
                TGIKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            UninitializedProviderConfig::DeepSeek {
                model_name,
                api_key_location,
            } => ProviderConfig::DeepSeek(DeepSeekProvider::new(
                model_name,
                DeepSeekKind
                    .get_defaulted_credential(
                        api_key_location.as_ref(),
                        provider_type_default_credentials,
                    )
                    .await?,
            )),
            #[cfg(any(test, feature = "e2e_tests"))]
            UninitializedProviderConfig::Dummy {
                model_name,
                api_key_location,
            } => ProviderConfig::Dummy(DummyProvider::new(model_name, api_key_location)?),
        })
    }
}

struct StreamAndRawRequest {
    stream: tracing_futures::Instrumented<PeekableProviderInferenceResponseStream>,
    raw_request: String,
    ticket_borrow: TicketBorrows,
}

pub struct StreamResponseAndMessages {
    pub response: StreamResponse,
    pub messages: Vec<RequestMessage>,
}

impl ModelProvider {
    fn apply_otlp_span_fields_input(&self, otlp_config: &OtlpConfig, span: &Span) {
        if otlp_config.traces.enabled {
            match otlp_config.traces.format {
                OtlpTracesFormat::OpenTelemetry => {
                    span.set_attribute("gen_ai.operation.name", "chat");
                    span.set_attribute("gen_ai.system", self.genai_system_name());

                    if let Some(model_name) = self.genai_model_name() {
                        span.set_attribute("gen_ai.request.model", model_name.to_string());
                    }
                }
                OtlpTracesFormat::OpenInference => {
                    span.set_attribute("openinference.span.kind", "LLM");
                    span.set_attribute("llm.system", self.genai_system_name());

                    if let Some(model_name) = self.genai_model_name() {
                        span.set_attribute("llm.model_name", model_name.to_string());
                    }
                }
            }
        }
    }

    #[expect(clippy::unused_self)] // We'll need 'self' for other attributes
    fn apply_otlp_span_fields_output(
        &self,
        otlp_config: &OtlpConfig,
        span: &Span,
        resp: &Result<ProviderInferenceResponse, Error>,
    ) {
        match resp {
            Ok(response) => {
                otlp_config.apply_usage_to_model_provider_span(span, &response.usage);
                match otlp_config.traces.format {
                    OtlpTracesFormat::OpenTelemetry => {}
                    OtlpTracesFormat::OpenInference => {
                        // If we ever add providers that don't use JSON, we'll need to update this.
                        span.set_attribute("input.mime_type", "application/json");
                        span.set_attribute("input.value", response.raw_request.clone());
                        span.set_attribute("output.mime_type", "application/json");
                        span.set_attribute("output.value", response.raw_response.clone());
                    }
                }
            }
            Err(e) => {
                // If an error occurs, try to extract the raw request/response to attach to the OpenTelemetry span
                match e.get_details() {
                    ErrorDetails::InferenceClient {
                        raw_request,
                        raw_response,
                        ..
                    }
                    | ErrorDetails::InferenceServer {
                        raw_request,
                        raw_response,
                        ..
                    } => {
                        match otlp_config.traces.format {
                            OtlpTracesFormat::OpenTelemetry => {}
                            OtlpTracesFormat::OpenInference => {
                                // If we ever add providers that don't use JSON, we'll need to update this.
                                if let Some(raw_request) = raw_request {
                                    span.set_attribute("input.mime_type", "application/json");
                                    span.set_attribute("input.value", raw_request.clone());
                                }
                                if let Some(raw_response) = raw_response {
                                    span.set_attribute("output.mime_type", "application/json");
                                    span.set_attribute("output.value", raw_response.clone());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    #[tracing::instrument(skip_all, fields(provider_name = &*self.name, otel.name = "model_provider_inference", stream = false))]
    async fn infer(
        &self,
        request: ModelProviderRequest<'_>,
        clients: &InferenceClients,
    ) -> Result<ProviderInferenceResponse, Error> {
        let span = Span::current();
        self.apply_otlp_span_fields_input(request.otlp_config, &span);
        let ticket_borrow = clients
            .rate_limiting_config
            .consume_tickets(
                &clients.postgres_connection_info,
                &clients.scope_info,
                request.request,
            )
            .await?;
        let res = match &self.config {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::AWSSagemaker(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Groq(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::OpenRouter(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::SGLang(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::TGI(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::DeepSeek(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .infer(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
        };
        self.apply_otlp_span_fields_output(request.otlp_config, &span, &res);
        let provider_inference_response = res?;
        if let Ok(actual_resource_usage) = provider_inference_response.resource_usage() {
            let postgres_connection_info = clients.postgres_connection_info.clone();
            // Make sure that we finish updating rate-limiting tickets if the gateway shuts down
            clients.deferred_tasks.spawn(
                async move {
                    if let Err(e) = ticket_borrow
                        .return_tickets(&postgres_connection_info, actual_resource_usage)
                        .await
                    {
                        tracing::error!("Failed to return rate limit tickets: {}", e);
                    }
                }
                .instrument(span),
            );
        }
        Ok(provider_inference_response)
    }

    #[tracing::instrument(skip_all, fields(provider_name = &*self.name, otel.name = "model_provider_inference", time_to_first_token, stream = true))]
    async fn infer_stream(
        &self,
        request: ModelProviderRequest<'_>,
        clients: &InferenceClients,
    ) -> Result<StreamAndRawRequest, Error> {
        self.apply_otlp_span_fields_input(request.otlp_config, &Span::current());
        let ticket_borrow = clients
            .rate_limiting_config
            .consume_tickets(
                &clients.postgres_connection_info,
                &clients.scope_info,
                request.request,
            )
            .await?;
        let (stream, raw_request) = match &self.config {
            ProviderConfig::Anthropic(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::AWSBedrock(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::AWSSagemaker(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Azure(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Fireworks(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::GCPVertexAnthropic(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::GCPVertexGemini(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::GoogleAIStudioGemini(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Groq(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Hyperbolic(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Mistral(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::OpenAI(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::OpenRouter(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::Together(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::SGLang(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::XAI(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::VLLM(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::TGI(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            ProviderConfig::DeepSeek(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            ProviderConfig::Dummy(provider) => {
                provider
                    .infer_stream(request, &clients.http_client, &clients.credentials, self)
                    .await
            }
        }?;

        // Attach the current `model_provider_inference` span to the stream.
        // This will cause the span to be entered every time the stream is polled,
        // extending the lifetime of the span in OpenTelemetry to include the entire
        // duration of the response stream.
        Ok(StreamAndRawRequest {
            stream: stream.instrument(Span::current()),
            raw_request,
            ticket_borrow,
        })
    }

    async fn start_batch_inference<'a>(
        &self,
        requests: &'a [ModelInferenceRequest<'a>],
        client: &'a TensorzeroHttpClient,
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
            ProviderConfig::AWSSagemaker(provider) => {
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
            ProviderConfig::Groq(provider) => {
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
            ProviderConfig::OpenRouter(provider) => {
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
        http_client: &'a TensorzeroHttpClient,
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
            ProviderConfig::AWSSagemaker(provider) => {
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
            ProviderConfig::Groq(provider) => {
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
            ProviderConfig::OpenRouter(provider) => {
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

#[derive(Debug, PartialEq, Clone)]
pub enum CredentialLocation {
    /// Environment variable containing the actual credential
    Env(String),
    /// Environment variable containing the path to a credential file
    PathFromEnv(String),
    /// For dynamic credential resolution
    Dynamic(String),
    /// Direct path to a credential file
    Path(String),
    /// Use a provider-specific SDK to determine credentials
    Sdk,
    None,
}

/// Credential location with optional fallback support
#[derive(Debug, PartialEq, Clone, Serialize, ts_rs::TS)]
#[serde(untagged)]
pub enum CredentialLocationWithFallback {
    /// Single credential location (backward compatible)
    Single(#[ts(type = "string")] CredentialLocation),
    /// Credential location with fallback
    WithFallback {
        #[ts(type = "string")]
        default: CredentialLocation,
        #[ts(type = "string")]
        fallback: CredentialLocation,
    },
}

impl CredentialLocationWithFallback {
    /// Get the default (primary) credential location
    pub fn default_location(&self) -> &CredentialLocation {
        match self {
            CredentialLocationWithFallback::Single(loc) => loc,
            CredentialLocationWithFallback::WithFallback { default, .. } => default,
        }
    }

    /// Get the fallback credential location if present
    pub fn fallback_location(&self) -> Option<&CredentialLocation> {
        match self {
            CredentialLocationWithFallback::Single(_) => None,
            CredentialLocationWithFallback::WithFallback { fallback, .. } => Some(fallback),
        }
    }
}

#[derive(Debug, PartialEq, Clone, ts_rs::TS)]
#[ts(export)]
pub enum EndpointLocation {
    /// Environment variable containing the actual endpoint URL
    Env(String),
    /// For dynamic endpoint resolution
    Dynamic(String),
    /// Direct endpoint URL
    Static(String),
}

impl<'de> Deserialize<'de> for EndpointLocation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if let Some(inner) = s.strip_prefix("env::") {
            Ok(EndpointLocation::Env(inner.to_string()))
        } else if let Some(inner) = s.strip_prefix("dynamic::") {
            Ok(EndpointLocation::Dynamic(inner.to_string()))
        } else {
            // Default to static endpoint
            Ok(EndpointLocation::Static(s))
        }
    }
}

impl Serialize for EndpointLocation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            EndpointLocation::Env(inner) => format!("env::{inner}"),
            EndpointLocation::Dynamic(inner) => format!("dynamic::{inner}"),
            EndpointLocation::Static(inner) => inner.clone(),
        };
        serializer.serialize_str(&s)
    }
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
        } else if s == "sdk" {
            Ok(CredentialLocation::Sdk)
        } else if s == "none" {
            Ok(CredentialLocation::None)
        } else {
            Err(serde::de::Error::custom(format!(
                "Invalid ApiKeyLocation format: {s}"
            )))
        }
    }
}

impl Serialize for CredentialLocation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            CredentialLocation::Env(inner) => format!("env::{inner}"),
            CredentialLocation::PathFromEnv(inner) => format!("path_from_env::{inner}"),
            CredentialLocation::Dynamic(inner) => format!("dynamic::{inner}"),
            CredentialLocation::Path(inner) => format!("path::{inner}"),
            CredentialLocation::Sdk => "sdk".to_string(),
            CredentialLocation::None => "none".to_string(),
        };
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for CredentialLocationWithFallback {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, MapAccess, Visitor};
        use std::fmt;

        struct CredentialLocationWithFallbackVisitor;

        impl<'de> Visitor<'de> for CredentialLocationWithFallbackVisitor {
            type Value = CredentialLocationWithFallback;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string or an object with 'default' and 'fallback' fields")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: Error,
            {
                // Parse as single CredentialLocation
                let location =
                    CredentialLocation::deserialize(serde::de::value::StrDeserializer::new(value))?;
                Ok(CredentialLocationWithFallback::Single(location))
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut default = None;
                let mut fallback = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "default" => {
                            if default.is_some() {
                                return Err(Error::duplicate_field("default"));
                            }
                            let value: String = map.next_value()?;
                            default = Some(CredentialLocation::deserialize(
                                serde::de::value::StrDeserializer::new(&value),
                            )?);
                        }
                        "fallback" => {
                            if fallback.is_some() {
                                return Err(Error::duplicate_field("fallback"));
                            }
                            let value: String = map.next_value()?;
                            fallback = Some(CredentialLocation::deserialize(
                                serde::de::value::StrDeserializer::new(&value),
                            )?);
                        }
                        _ => {
                            return Err(Error::unknown_field(&key, &["default", "fallback"]));
                        }
                    }
                }

                let default = default.ok_or_else(|| Error::missing_field("default"))?;
                let fallback = fallback.ok_or_else(|| Error::missing_field("fallback"))?;

                Ok(CredentialLocationWithFallback::WithFallback { default, fallback })
            }
        }

        deserializer.deserialize_any(CredentialLocationWithFallbackVisitor)
    }
}

#[derive(Clone, Debug)]
pub enum Credential {
    Static(SecretString),
    FileContents(SecretString),
    Dynamic(String),
    Sdk,
    None,
    Missing,
    WithFallback {
        default: Box<Credential>,
        fallback: Box<Credential>,
    },
}

pub const SHORTHAND_MODEL_PREFIXES: &[&str] = &[
    "anthropic::",
    "deepseek::",
    "fireworks::",
    "google_ai_studio_gemini::",
    "gcp_vertex_gemini::",
    "gcp_vertex_anthropic::",
    "hyperbolic::",
    "groq::",
    "mistral::",
    "openai::",
    "openrouter::",
    "together::",
    "xai::",
    "dummy::",
];

pub type ModelTable = BaseModelTable<ModelConfig>;

impl ShorthandModelConfig for ModelConfig {
    const SHORTHAND_MODEL_PREFIXES: &[&str] = SHORTHAND_MODEL_PREFIXES;
    const MODEL_TYPE: &str = "Model";
    async fn from_shorthand(
        provider_type: &str,
        model_name: &str,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self, Error> {
        let model_name = model_name.to_string();
        let provider_config = match provider_type {
            "anthropic" => ProviderConfig::Anthropic(AnthropicProvider::new(
                model_name,
                None,
                AnthropicKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
                // We don't support beta structured output for shorthand models
                false,
            )),
            "deepseek" => ProviderConfig::DeepSeek(DeepSeekProvider::new(
                model_name,
                DeepSeekKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "fireworks" => ProviderConfig::Fireworks(FireworksProvider::new(
                model_name,
                FireworksKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
                crate::providers::fireworks::default_parse_think_blocks(),
            )),
            "google_ai_studio_gemini" => {
                ProviderConfig::GoogleAIStudioGemini(GoogleAIStudioGeminiProvider::new(
                    model_name,
                    GoogleAIStudioGeminiKind
                        .get_defaulted_credential(None, default_credentials)
                        .await?,
                )?)
            }
            "gcp_vertex_gemini" => ProviderConfig::GCPVertexGemini(
                GCPVertexGeminiProvider::new_shorthand(model_name, default_credentials).await?,
            ),
            "gcp_vertex_anthropic" => ProviderConfig::GCPVertexAnthropic(
                GCPVertexAnthropicProvider::new_shorthand(model_name, default_credentials).await?,
            ),
            "groq" => ProviderConfig::Groq(GroqProvider::new(
                model_name,
                GroqKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "hyperbolic" => ProviderConfig::Hyperbolic(HyperbolicProvider::new(
                model_name,
                HyperbolicKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "mistral" => ProviderConfig::Mistral(MistralProvider::new(
                model_name,
                MistralKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "openai" => {
                if let Some(stripped_model_name) = model_name.strip_prefix("responses::") {
                    ProviderConfig::OpenAI(OpenAIProvider::new(
                        stripped_model_name.to_string(),
                        None,
                        OpenAIKind
                            .get_defaulted_credential(None, default_credentials)
                            .await?,
                        OpenAIAPIType::Responses,
                        false,
                        Vec::new(),
                    )?)
                } else {
                    ProviderConfig::OpenAI(OpenAIProvider::new(
                        model_name,
                        None,
                        OpenAIKind
                            .get_defaulted_credential(None, default_credentials)
                            .await?,
                        OpenAIAPIType::ChatCompletions,
                        false,
                        Vec::new(),
                    )?)
                }
            }
            "openrouter" => ProviderConfig::OpenRouter(OpenRouterProvider::new(
                model_name,
                OpenRouterKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            "together" => ProviderConfig::Together(TogetherProvider::new(
                model_name,
                TogetherKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
                crate::providers::together::default_parse_think_blocks(),
            )),
            "xai" => ProviderConfig::XAI(XAIProvider::new(
                model_name,
                XAIKind
                    .get_defaulted_credential(None, default_credentials)
                    .await?,
            )),
            #[cfg(any(test, feature = "e2e_tests"))]
            "dummy" => ProviderConfig::Dummy(DummyProvider::new(model_name, None)?),
            _ => {
                return Err(ErrorDetails::Config {
                    message: format!("Invalid provider type: {provider_type}"),
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
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        })
    }

    fn validate(
        &self,
        model_name: &str,
        global_outbound_http_timeout: &chrono::Duration,
    ) -> Result<(), Error> {
        self.timeouts.validate(global_outbound_http_timeout)?;
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
        for (provider_name, provider) in &self.providers {
            if !seen_providers.contains(provider_name) {
                return Err(ErrorDetails::Config {
                    message: format!(
                "`models.{model_name}`: Provider `{provider_name}` is not listed in `routing`"
            ),
                }
                .into());
            }
            provider.validate(global_outbound_http_timeout)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::cache::CacheEnabledMode;
    use crate::config::SKIP_CREDENTIAL_VALIDATION;
    use crate::rate_limiting::ScopeInfo;
    use crate::tool::ToolCallConfig;
    use crate::{
        cache::CacheOptions,
        db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo},
        inference::types::{
            ContentBlockChunk, FunctionType, ModelInferenceRequestJsonMode, TextChunk,
        },
        model_table::RESERVED_MODEL_PREFIXES,
        providers::anthropic::AnthropicCredentials,
        providers::dummy::{
            DummyCredentials, DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW,
            DUMMY_STREAMING_RESPONSE,
        },
        rate_limiting::{RateLimitingConfig, UninitializedRateLimitingConfig},
    };
    use secrecy::SecretString;
    use tokio_stream::StreamExt;
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
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let tool_config = ToolCallConfig::with_tools_available(vec![], vec![]);
        let api_keys = InferenceCredentials::default();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
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
        assert_eq!(
            usage,
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
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
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let response = model_config
            .infer(&request, &clients, model_name)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: IndexMap::from([(
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
    async fn test_model_provider_infer_max_tokens_check() {
        let provider = ModelProvider {
            name: "test_provider".into(),
            config: ProviderConfig::Dummy(DummyProvider {
                model_name: "good".into(),
                credentials: DummyCredentials::None,
            }),
            extra_body: Default::default(),
            extra_headers: Default::default(),
            timeouts: Default::default(),
            discard_unknown_chunks: false,
        };

        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let postgres_mock = PostgresConnectionInfo::Disabled;
        let api_keys = InferenceCredentials::default();
        let tags = HashMap::new();

        // With token rate limiting enabled and no max_tokens
        let toml_str = r"
            [[rules]]
            tokens_per_second = 10
            always = true
        ";
        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let rate_limit_config: RateLimitingConfig = uninitialized_config.try_into().unwrap();

        let clients = InferenceClients {
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: postgres_mock.clone(),
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(tags.clone()),
            rate_limiting_config: Arc::new(rate_limit_config.clone()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(tags.clone()),
                api_key_public_id: None,
            },
        };

        let request_no_max_tokens = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None, // No max_tokens!
            ..Default::default()
        };

        let provider_request = ModelProviderRequest {
            request: &request_no_max_tokens,
            model_name: "test",
            provider_name: "test_provider",
            otlp_config: &Default::default(),
        };

        // Should fail with RateLimitMissingMaxTokens
        let result = provider.infer(provider_request, &clients).await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            Error::new(ErrorDetails::RateLimitMissingMaxTokens)
        );

        // With token rate limiting enabled and max_tokens provided
        let request_with_max_tokens = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: Some(100), // max_tokens provided
            ..Default::default()
        };

        // This should error because postgres is disabled, but it should not be the RateLimitMissingMaxTokens error
        let provider_request = ModelProviderRequest {
            request: &request_with_max_tokens,
            model_name: "test",
            provider_name: "test_provider",
            otlp_config: &Default::default(),
        };

        let result = provider
            .infer(provider_request, &clients)
            .await
            .unwrap_err();
        assert_ne!(result, Error::new(ErrorDetails::RateLimitMissingMaxTokens));
    }

    #[tokio::test]
    async fn test_model_config_infer_routing_fallback() {
        let logs_contain = crate::utils::testing::capture_logs();
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
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
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
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                    },
                ),
                (
                    "good_provider".to_string().into(),
                    ModelProvider {
                        name: "good_provider".into(),
                        config: good_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                    },
                ),
            ]),
            timeouts: Default::default(),
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
        assert_eq!(
            usage,
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
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
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let StreamResponseAndMessages {
            response:
                StreamResponse {
                    mut stream,
                    raw_request,
                    model_provider_name,
                    cached: _,
                },
            messages: _input,
        } = model_config
            .infer_stream(
                &request,
                &InferenceClients {
                    http_client: TensorzeroHttpClient::new_testing().unwrap(),
                    clickhouse_connection_info: ClickHouseConnectionInfo::new_disabled(),
                    postgres_connection_info: PostgresConnectionInfo::Disabled,
                    credentials: Arc::new(api_keys.clone()),
                    cache_options: CacheOptions {
                        max_age_s: None,
                        enabled: CacheEnabledMode::Off,
                    },
                    tags: Arc::new(Default::default()),
                    rate_limiting_config: Arc::new(Default::default()),
                    otlp_config: Default::default(),
                    deferred_tasks: tokio_util::task::TaskTracker::new(),
                    scope_info: ScopeInfo {
                        tags: Arc::new(HashMap::new()),
                        api_key_public_id: None,
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
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let response = model_config
            .infer_stream(
                &request,
                &InferenceClients {
                    http_client: TensorzeroHttpClient::new_testing().unwrap(),
                    clickhouse_connection_info: ClickHouseConnectionInfo::new_disabled(),
                    postgres_connection_info: PostgresConnectionInfo::Disabled,
                    credentials: Arc::new(api_keys.clone()),
                    cache_options: CacheOptions {
                        max_age_s: None,
                        enabled: CacheEnabledMode::Off,
                    },
                    tags: Arc::new(Default::default()),
                    rate_limiting_config: Arc::new(Default::default()),
                    otlp_config: Default::default(),
                    deferred_tasks: tokio_util::task::TaskTracker::new(),
                    scope_info: ScopeInfo {
                        tags: Arc::new(HashMap::new()),
                        api_key_public_id: None,
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
                provider_errors: IndexMap::from([(
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
    async fn test_model_config_infer_stream_routing_fallback() {
        let logs_contain = crate::utils::testing::capture_logs();
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
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                    },
                ),
                (
                    "good_provider".to_string().into(),
                    ModelProvider {
                        name: "good_provider".to_string().into(),
                        config: good_provider_config,
                        extra_body: Default::default(),
                        extra_headers: Default::default(),
                        timeouts: Default::default(),
                        discard_unknown_chunks: false,
                    },
                ),
            ]),
            timeouts: Default::default(),
        };
        let StreamResponseAndMessages {
            response:
                StreamResponse {
                    mut stream,
                    raw_request,
                    model_provider_name,
                    cached: _,
                },
            messages: _,
        } = model_config
            .infer_stream(
                &request,
                &InferenceClients {
                    http_client: TensorzeroHttpClient::new_testing().unwrap(),
                    clickhouse_connection_info: ClickHouseConnectionInfo::new_disabled(),
                    postgres_connection_info: PostgresConnectionInfo::Disabled,
                    credentials: Arc::new(api_keys.clone()),
                    cache_options: CacheOptions {
                        max_age_s: None,
                        enabled: CacheEnabledMode::Off,
                    },
                    tags: Arc::new(Default::default()),
                    rate_limiting_config: Arc::new(Default::default()),
                    otlp_config: Default::default(),
                    deferred_tasks: tokio_util::task::TaskTracker::new(),
                    scope_info: ScopeInfo {
                        tags: Arc::new(HashMap::new()),
                        api_key_public_id: None,
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
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let tool_config = ToolCallConfig::with_tools_available(vec![], vec![]);
        let api_keys = InferenceCredentials::default();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
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
                provider_errors: IndexMap::from([(
                    "model".to_string(),
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Dummy".to_string(),
                        message: "Dynamic api key `TEST_KEY` is missing".to_string(),
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
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
        };
        let response = model_config
            .infer(&request, &clients, model_name)
            .await
            .unwrap_err();
        assert_eq!(
            response,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: IndexMap::from([(
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
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let tool_config = ToolCallConfig::with_tools_available(vec![], vec![]);
        let api_keys = InferenceCredentials::default();
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let clients = InferenceClients {
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
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
                provider_errors: IndexMap::from([(
                    "model".to_string(),
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Dummy".to_string(),
                        message: "Dynamic api key `TEST_KEY` is missing".to_string(),
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
            http_client: http_client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys.clone()),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
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

    #[tokio::test]
    async fn test_validate_or_create_model_config() {
        let model_table = ModelTable::default();
        // Test that we can get or create a model config
        model_table.validate("dummy::gpt-4o").unwrap();
        // Shorthand models are not added to the model table
        assert_eq!(model_table.static_model_len(), 0);
        let model_config = model_table
            .get("dummy::gpt-4o")
            .await
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
        let anthropic_provider_config = SKIP_CREDENTIAL_VALIDATION.sync_scope((), || {
            ProviderConfig::Anthropic(AnthropicProvider::new(
                "claude".to_string(),
                None,
                AnthropicCredentials::None,
                false,
            ))
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
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let provider_types = ProviderTypesConfig::default();
        let model_table: ModelTable = ModelTable::new(
            HashMap::from([("claude".into(), anthropic_model_config)]),
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            chrono::Duration::seconds(120),
        )
        .unwrap();

        model_table.validate("dummy::claude").unwrap();
    }

    #[test]
    fn test_shorthand_prefixes_subset_of_reserved() {
        for &shorthand in SHORTHAND_MODEL_PREFIXES {
            assert!(
                RESERVED_MODEL_PREFIXES.contains(&shorthand.to_string()),
                "Shorthand prefix '{shorthand}' is not in RESERVED_MODEL_PREFIXES"
            );
        }
    }

    #[test]
    fn test_credential_location_with_fallback_serialize_single() {
        // Test serializing a Single variant (backward compatible)
        let single =
            CredentialLocationWithFallback::Single(CredentialLocation::Env("API_KEY".to_string()));
        let json = serde_json::to_string(&single).unwrap();
        assert_eq!(json, r#""env::API_KEY""#);
    }

    #[test]
    fn test_credential_location_with_fallback_serialize_with_fallback() {
        // Test serializing a WithFallback variant
        let with_fallback = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::Dynamic("key1".to_string()),
            fallback: CredentialLocation::Env("FALLBACK_KEY".to_string()),
        };
        let json = serde_json::to_string(&with_fallback).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["default"], "dynamic::key1");
        assert_eq!(parsed["fallback"], "env::FALLBACK_KEY");
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_single_string() {
        // Test deserializing from a simple string (backward compatible)
        let json = r#""env::API_KEY""#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::Single(CredentialLocation::Env(key)) => {
                assert_eq!(key, "API_KEY");
            }
            _ => panic!("Expected Single(Env)"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_dynamic_string() {
        // Test deserializing a dynamic credential from a string
        let json = r#""dynamic::my_key""#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::Single(CredentialLocation::Dynamic(key)) => {
                assert_eq!(key, "my_key");
            }
            _ => panic!("Expected Single(Dynamic)"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_with_fallback() {
        // Test deserializing an object with default and fallback fields
        let json = r#"{"default":"dynamic::key1","fallback":"env::FALLBACK_KEY"}"#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::WithFallback { default, fallback } => {
                match default {
                    CredentialLocation::Dynamic(key) => assert_eq!(key, "key1"),
                    _ => panic!("Expected Dynamic for default"),
                }
                match fallback {
                    CredentialLocation::Env(key) => assert_eq!(key, "FALLBACK_KEY"),
                    _ => panic!("Expected Env for fallback"),
                }
            }
            CredentialLocationWithFallback::Single(..) => panic!("Expected WithFallback"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_path_variants() {
        // Test deserializing path-based credentials
        let json = r#"{"default":"path::/etc/key","fallback":"path_from_env::KEY_PATH"}"#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::WithFallback { default, fallback } => {
                match default {
                    CredentialLocation::Path(path) => assert_eq!(path, "/etc/key"),
                    _ => panic!("Expected Path for default"),
                }
                match fallback {
                    CredentialLocation::PathFromEnv(key) => assert_eq!(key, "KEY_PATH"),
                    _ => panic!("Expected PathFromEnv for fallback"),
                }
            }
            CredentialLocationWithFallback::Single(..) => panic!("Expected WithFallback"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_sdk() {
        // Test deserializing SDK credential
        let json = r#""sdk""#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::Single(CredentialLocation::Sdk) => {}
            _ => panic!("Expected Single(Sdk)"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_none() {
        // Test deserializing None credential
        let json = r#""none""#;
        let result: CredentialLocationWithFallback = serde_json::from_str(json).unwrap();
        match result {
            CredentialLocationWithFallback::Single(CredentialLocation::None) => {}
            _ => panic!("Expected Single(None)"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_roundtrip_single() {
        // Test serializing and deserializing a Single variant
        let original =
            CredentialLocationWithFallback::Single(CredentialLocation::Env("MY_KEY".to_string()));
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: CredentialLocationWithFallback = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_credential_location_with_fallback_roundtrip_with_fallback() {
        // Test serializing and deserializing a WithFallback variant
        let original = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::Dynamic("primary".to_string()),
            fallback: CredentialLocation::Env("SECONDARY".to_string()),
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: CredentialLocationWithFallback = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_missing_default() {
        // Test that missing default field returns an error
        let json = r#"{"fallback":"env::FALLBACK_KEY"}"#;
        let result: Result<CredentialLocationWithFallback, _> = serde_json::from_str(json);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("missing field `default`"));
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_missing_fallback() {
        // Test that missing fallback field returns an error
        let json = r#"{"default":"env::DEFAULT_KEY"}"#;
        let result: Result<CredentialLocationWithFallback, _> = serde_json::from_str(json);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("missing field `fallback`"));
    }

    #[test]
    fn test_credential_location_with_fallback_deserialize_unknown_field() {
        // Test that unknown fields return an error
        let json = r#"{"default":"env::KEY","fallback":"env::FALLBACK","unknown":"value"}"#;
        let result: Result<CredentialLocationWithFallback, _> = serde_json::from_str(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown field"));
    }

    #[test]
    fn test_credential_location_with_fallback_default_location() {
        // Test the default_location() method
        let single =
            CredentialLocationWithFallback::Single(CredentialLocation::Env("KEY".to_string()));
        match single.default_location() {
            CredentialLocation::Env(key) => assert_eq!(key, "KEY"),
            _ => panic!("Expected Env"),
        }

        let with_fallback = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::Dynamic("primary".to_string()),
            fallback: CredentialLocation::Env("secondary".to_string()),
        };
        match with_fallback.default_location() {
            CredentialLocation::Dynamic(key) => assert_eq!(key, "primary"),
            _ => panic!("Expected Dynamic"),
        }
    }

    #[test]
    fn test_credential_location_with_fallback_fallback_location() {
        // Test the fallback_location() method
        let single =
            CredentialLocationWithFallback::Single(CredentialLocation::Env("KEY".to_string()));
        assert!(single.fallback_location().is_none());

        let with_fallback = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::Dynamic("primary".to_string()),
            fallback: CredentialLocation::Env("secondary".to_string()),
        };
        match with_fallback.fallback_location() {
            Some(CredentialLocation::Env(key)) => assert_eq!(key, "secondary"),
            _ => panic!("Expected Some(Env)"),
        }
    }
}
