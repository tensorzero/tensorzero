//! Contains the main logic for 'relay' mode
//! When enabled, we redirect requests made to *any* model, and instead forward them to a downstream gateway
//! The providers in the initial ('edge') gateway are ignored entirely
use std::{collections::HashMap, time::Instant};

use futures::StreamExt;
use futures::future::try_join_all;
use secrecy::SecretString;
use url::Url;

use crate::client::{
    ClientBuilder, ClientBuilderMode, ClientSecretString, ContentBlockChunk, InferenceResponseChunk,
};
use crate::config::UninitializedRelayConfig;
use crate::endpoints::embeddings::{EmbeddingResponse, EmbeddingsParams};
use crate::endpoints::inference::InferenceCredentials;
use crate::endpoints::openai_compatible::types::embeddings::{
    OpenAICompatibleEmbeddingParams, OpenAIEmbedding, OpenAIEmbeddingResponse,
};
use crate::error::{DelayedError, IMPOSSIBLE_ERROR_MESSAGE};
use crate::inference::types::extra_body::{prepare_relay_extra_body, prepare_relay_extra_headers};
use crate::inference::types::{
    ModelInferenceRequest, PeekableProviderInferenceResponseStream, ProviderInferenceResponseChunk,
    TextChunk, Usage,
};
use crate::model::Credential;
use crate::{
    cache::CacheParamsOptions,
    client::{
        Base64File, Client, ClientInferenceParams, File, InferenceOutput, InferenceParams,
        InferenceResponse, Input, InputMessage, InputMessageContent, ObjectStoragePointer, System,
        UrlFile,
    },
    endpoints::inference::{ChatCompletionInferenceParams, InferenceClients},
    error::{Error, ErrorDetails},
    inference::types::{
        ContentBlock, ContentBlockChatOutput, ContentBlockOutput, Latency,
        ModelInferenceRequestJsonMode, ProviderInferenceResponse, ProviderInferenceResponseArgs,
        Text, resolved_input::LazyFile,
    },
    tool::{DynamicToolParams, FunctionTool, Tool, ToolCall, ToolCallWrapper, ToolConfigRef},
    variant::JsonMode,
};
use uuid::Uuid;

#[derive(Clone, Debug)]
pub enum RelayCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<RelayCredentials>,
        fallback: Box<RelayCredentials>,
    },
}

impl RelayCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, DelayedError> {
        match self {
            RelayCredentials::Static(api_key) => Ok(Some(api_key)),
            RelayCredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: "tensorzero::relay".to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                }))
                .transpose()
            }
            RelayCredentials::WithFallback { default, fallback } => {
                // Try default first, fall back to fallback if it fails
                match default.get_api_key(dynamic_api_keys) {
                    Ok(key) => Ok(key),
                    Err(e) => {
                        e.log_at_level(
                            "Using fallback credential, as default credential is unavailable: ",
                            tracing::Level::WARN,
                        );
                        fallback.get_api_key(dynamic_api_keys)
                    }
                }
            }
            RelayCredentials::None => Ok(None),
        }
    }
}

impl TryFrom<Credential> for RelayCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(RelayCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(RelayCredentials::Dynamic(key_name)),
            Credential::None => Ok(RelayCredentials::None),
            Credential::Missing => Ok(RelayCredentials::None),
            Credential::WithFallback { default, fallback } => Ok(RelayCredentials::WithFallback {
                default: Box::new((*default).try_into()?),
                fallback: Box::new((*fallback).try_into()?),
            }),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Tensorzero Relay mode".to_string(),
            })),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorzeroRelay {
    client: Client,
    credentials: RelayCredentials,
    pub original_config: UninitializedRelayConfig,
}

impl TensorzeroRelay {
    pub fn new(
        gateway_url: Url,
        credentials: RelayCredentials,
        original_config: UninitializedRelayConfig,
    ) -> Result<Self, Error> {
        Ok(Self {
            client: ClientBuilder::new(ClientBuilderMode::HTTPGateway {
                url: gateway_url.clone(),
            })
            .build_http()
            .map_err(|e| {
                Error::new(ErrorDetails::Config {
                    message: e.to_string(),
                })
            })?,
            credentials,
            original_config,
        })
    }
}

impl TensorzeroRelay {
    pub async fn relay_embeddings(
        &self,
        params: EmbeddingsParams,
    ) -> Result<EmbeddingResponse, Error> {
        let params = OpenAICompatibleEmbeddingParams {
            input: params.input,
            model: format!("tensorzero::embedding_model_name::{}", params.model_name),
            dimensions: params.dimensions,
            encoding_format: params.encoding_format,
            tensorzero_dryrun: None,
            tensorzero_credentials: params.credentials,
            tensorzero_cache_options: None,
            tensorzero_include_raw_response: params.include_raw_response,
        };

        let api_key = self
            .credentials
            .get_api_key(&params.tensorzero_credentials)
            .map_err(|e| e.log())?
            .cloned();

        let res = self
            .client
            .http_embeddings(params, api_key)
            .await
            .map_err(|e| {
                // TODO - include `raw_request`/`raw_response` here
                Error::new(ErrorDetails::InferenceClient {
                    message: e.to_string(),
                    status_code: None,
                    provider_type: "tensorzero_relay".to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;
        match res.response {
            OpenAIEmbeddingResponse::List {
                data,
                model,
                usage,
                tensorzero_raw_response,
            } => Ok(EmbeddingResponse {
                embeddings: data
                    .into_iter()
                    .map(|embedding| match embedding {
                        OpenAIEmbedding::Embedding {
                            embedding,
                            index: _,
                        } => embedding,
                    })
                    .collect(),
                usage: usage
                    .map(|usage| Usage {
                        input_tokens: usage.prompt_tokens,
                        output_tokens: match (usage.total_tokens, usage.prompt_tokens) {
                            (Some(total), Some(prompt)) => Some(total - prompt),
                            _ => None,
                        },
                    })
                    .unwrap_or_default(),
                model,
                tensorzero_raw_response,
            }),
        }
    }
    pub async fn relay_streaming<'a>(
        &self,
        model_name: &str,
        request: &ModelInferenceRequest<'a>,
        clients: &InferenceClients,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let start_time = Instant::now();

        let client_inference_params = self
            .build_client_inference_params(model_name, request, clients, true)
            .await?;
        let res = self
            .client
            .http_inference(client_inference_params)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::Inference {
                    message: e.to_string(),
                })
            })?;

        let InferenceOutput::Streaming(streaming) = res.response else {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Expected streaming inference response. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }));
        };

        let stream = streaming
            .map(move |chunk| match chunk {
                Ok(chunk) => {
                    let raw_chunk = serde_json::to_string(&chunk).unwrap_or_default();
                    match chunk {
                        InferenceResponseChunk::Chat(c) => {
                            Ok(ProviderInferenceResponseChunk::new_with_raw_usage(
                                c.content,
                                c.usage,
                                // TODO - get the original chunk as a string
                                raw_chunk,
                                start_time.elapsed(),
                                c.finish_reason,
                                c.raw_usage,
                            ))
                        }
                        InferenceResponseChunk::Json(c) => {
                            Ok(ProviderInferenceResponseChunk::new_with_raw_usage(
                                vec![ContentBlockChunk::Text(TextChunk {
                                    id: "0".to_string(),
                                    text: c.raw,
                                })],
                                c.usage,
                                // TODO - get the original chunk as a string
                                raw_chunk,
                                start_time.elapsed(),
                                c.finish_reason,
                                c.raw_usage,
                            ))
                        }
                    }
                }
                Err(e) => Err(e),
            })
            .boxed()
            .peekable();

        Ok((stream, res.raw_request))
    }
    pub async fn relay_non_streaming<'a>(
        &self,
        model_name: &str,
        request: &ModelInferenceRequest<'a>,
        clients: &InferenceClients,
    ) -> Result<ProviderInferenceResponse, Error> {
        let start_time = Instant::now();

        let client_inference_params = self
            .build_client_inference_params(model_name, request, clients, false)
            .await?;

        let http_data = self
            .client
            .http_inference(client_inference_params)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::Inference {
                    message: e.to_string(),
                })
            })?;

        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };

        let InferenceOutput::NonStreaming(non_streaming) = http_data.response else {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Expected non-streaming inference response. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }));
        };

        let usage = non_streaming.usage().to_owned();
        let finish_reason = non_streaming.finish_reason();
        // Extract raw_usage from downstream response for passthrough
        let raw_usage_entries = non_streaming.raw_usage().cloned();
        // Extract relay_raw_response entries from downstream response for passthrough
        let relay_raw_response = non_streaming.raw_response().cloned();

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: match non_streaming {
                    InferenceResponse::Chat(chat) => chat
                        .content
                        .into_iter()
                        .flat_map(|c| match c {
                            ContentBlockChatOutput::Text(text) => {
                                Some(ContentBlockOutput::Text(text))
                            }
                            ContentBlockChatOutput::ToolCall(tool_call) => {
                                Some(ContentBlockOutput::ToolCall(ToolCall {
                                    id: tool_call.id,
                                    name: tool_call.raw_name,
                                    arguments: tool_call.raw_arguments,
                                }))
                            }
                            ContentBlockChatOutput::Thought(thought) => {
                                Some(ContentBlockOutput::Thought(thought))
                            }
                            ContentBlockChatOutput::Unknown(unknown) => {
                                Some(ContentBlockOutput::Unknown(unknown))
                            }
                        })
                        .collect(),
                    InferenceResponse::Json(json) => match json.output.raw {
                        Some(raw) => vec![ContentBlockOutput::Text(Text { text: raw })],
                        None => vec![],
                    },
                },
                system: request.system.clone(),
                input_messages: request.messages.clone(),
                raw_request: http_data.raw_request,
                raw_response: http_data.raw_response.unwrap_or_default(),
                usage,
                raw_usage: raw_usage_entries,
                relay_raw_response,
                provider_latency: latency,
                finish_reason,
                id: Uuid::now_v7(),
            },
        ))
    }

    // Constructs the input for the downstream gateway `POST /inference` request
    // We always invoke a default function (using the model name for the 'redirected' model in the edge gateway),
    // and include all of the tools from the 'edge' gateway (both static and dynamic) as dynamic tools
    async fn build_client_inference_params(
        &self,
        model_name: &str,
        request: &ModelInferenceRequest<'_>,
        clients: &InferenceClients,
        stream: bool,
    ) -> Result<ClientInferenceParams, Error> {
        let input = Input {
            messages: try_join_all(request.messages.clone().into_iter().map(async |m| {
                Ok::<_, Error>(InputMessage {
                    role: m.role,
                    content: try_join_all(m.content.into_iter().map(async |c| match c {
                        ContentBlock::Text(t) => {
                            Ok::<_, Error>(InputMessageContent::Text(Text { text: t.text }))
                        }
                        ContentBlock::ToolCall(t) => {
                            Ok(InputMessageContent::ToolCall(ToolCallWrapper::ToolCall(t)))
                        }
                        ContentBlock::ToolResult(t) => Ok(InputMessageContent::ToolResult(t)),
                        ContentBlock::Thought(t) => Ok(InputMessageContent::Thought(t)),
                        ContentBlock::File(f) => match *f {
                            LazyFile::Url { file_url, future } => {
                                if request.fetch_and_encode_input_files_before_inference {
                                    let resolved_file = future.await?;
                                    Ok(InputMessageContent::File(File::ObjectStorage(
                                        resolved_file,
                                    )))
                                } else {
                                    Ok(InputMessageContent::File(File::Url(UrlFile {
                                        url: file_url.url,
                                        mime_type: file_url.mime_type,
                                        detail: file_url.detail,
                                        filename: file_url.filename,
                                    })))
                                }
                            }
                            LazyFile::Base64(pending) => {
                                Ok(InputMessageContent::File(File::Base64(Base64File::new(
                                    pending.file.source_url.clone(),
                                    Some(pending.file.mime_type.clone()),
                                    pending.data.clone(),
                                    pending.file.detail.clone(),
                                    pending.file.filename.clone(),
                                )?)))
                            }
                            LazyFile::ObjectStoragePointer {
                                metadata,
                                storage_path,
                                future,
                            } => {
                                if request.fetch_and_encode_input_files_before_inference {
                                    let resolved_file = future.await?;
                                    Ok(InputMessageContent::File(File::ObjectStorage(
                                        resolved_file,
                                    )))
                                } else {
                                    Ok(InputMessageContent::File(File::ObjectStoragePointer(
                                        ObjectStoragePointer {
                                            storage_path,
                                            source_url: metadata.source_url.clone(),
                                            mime_type: metadata.mime_type.clone(),
                                            detail: metadata.detail.clone(),
                                            filename: metadata.filename.clone(),
                                        },
                                    )))
                                }
                            }

                            LazyFile::ObjectStorage(object_storage_file) => Ok(
                                InputMessageContent::File(File::ObjectStorage(object_storage_file)),
                            ),
                        },
                        ContentBlock::Unknown(unknown) => Ok(InputMessageContent::Unknown(unknown)),
                    }))
                    .await?,
                })
            }))
            .await?,
            system: request.system.clone().map(System::Text),
        };

        let api_key = self
            .credentials
            .get_api_key(&clients.credentials)
            .map_err(|e| e.log())?
            .cloned();

        let res = ClientInferenceParams {
            function_name: None,
            variant_name: None,
            model_name: Some(model_name.to_string()),
            input,
            stream: Some(stream),
            params: InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    temperature: request.temperature,
                    max_tokens: request.max_tokens,
                    seed: request.seed,
                    top_p: request.top_p,
                    presence_penalty: request.presence_penalty,
                    frequency_penalty: request.frequency_penalty,
                    json_mode: Some(match request.json_mode {
                        ModelInferenceRequestJsonMode::Off => JsonMode::Off,
                        ModelInferenceRequestJsonMode::On => JsonMode::On,
                        ModelInferenceRequestJsonMode::Strict => JsonMode::Strict,
                    }),
                    stop_sequences: request
                        .borrow_stop_sequences()
                        .clone()
                        .map(|c| c.into_owned()),
                    reasoning_effort: request.inference_params_v2.reasoning_effort.clone(),
                    service_tier: request.inference_params_v2.service_tier.clone(),
                    thinking_budget_tokens: request.inference_params_v2.thinking_budget_tokens,
                    verbosity: request.inference_params_v2.verbosity.clone(),
                },
            },
            internal: false,
            dynamic_tool_params: DynamicToolParams {
                allowed_tools: request
                    .tool_config
                    .as_ref()
                    .map(|config| config.allowed_tools.tools.clone()),
                additional_tools: request.tool_config.as_ref().map(|tools| {
                    tools
                        .tools_available_with_openai_custom()
                        .map(|t| match t {
                            ToolConfigRef::Function(f) => Tool::Function(FunctionTool {
                                description: f.description().to_string(),
                                parameters: f.parameters().clone(),
                                name: f.name().to_string(),
                                strict: f.strict(),
                            }),
                            ToolConfigRef::OpenAICustom(o) => Tool::OpenAICustom(o.clone()),
                        })
                        .collect()
                }),
                tool_choice: request
                    .tool_config
                    .as_ref()
                    .map(|config| config.tool_choice.clone()),
                parallel_tool_calls: request
                    .tool_config
                    .as_ref()
                    .and_then(|config| config.parallel_tool_calls),
                provider_tools: request
                    .tool_config
                    .as_ref()
                    .map(|config| config.provider_tools.clone())
                    .unwrap_or_default(),
            },
            output_schema: request.output_schema.cloned(),
            extra_body: prepare_relay_extra_body(&request.extra_body),
            extra_headers: prepare_relay_extra_headers(&request.extra_headers),
            credentials: clients
                .credentials
                .iter()
                .map(|(k, v)| (k.clone(), ClientSecretString(v.clone())))
                .collect(),
            // TODO - how do we want this to interact with dryrun?
            cache_options: CacheParamsOptions {
                max_age_s: clients.cache_options.max_age_s,
                enabled: clients.cache_options.enabled,
            },
            internal_dynamic_variant_config: None,
            episode_id: None,
            dryrun: None,
            // Filter out internal tags (those starting with "tensorzero::") before forwarding
            // to the downstream gateway, as they will be rejected by tag validation
            tags: clients
                .tags
                .iter()
                .filter(|(k, _)| !k.starts_with("tensorzero::"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            otlp_traces_extra_headers: HashMap::new(),
            otlp_traces_extra_attributes: HashMap::new(),
            otlp_traces_extra_resources: HashMap::new(),
            include_original_response: false,
            include_raw_response: clients.include_raw_response,
            include_raw_usage: clients.include_raw_usage,
            api_key,
        };
        Ok(res)
    }
}
