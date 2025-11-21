use std::collections::HashMap;
use std::time::Instant;

use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use futures::future::try_join_all;
use futures::StreamExt;
use secrecy::SecretString;
use serde::Serialize;
use url::Url;

use crate::cache::ModelProviderRequest;
use crate::client::{
    Base64File, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    ContentBlockChunk, File, InferenceOutput, InferenceParams, InferenceResponse,
    InferenceResponseChunk, ObjectStoragePointer, System, Unknown, UrlFile,
};
use crate::endpoints::inference::{ChatCompletionInferenceParams, InferenceCredentials};
use crate::error::{DelayedError, Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::batch::{
    BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
};
use crate::inference::types::resolved_input::LazyFile;
use crate::inference::types::{
    ContentBlock, ContentBlockChatOutput, ContentBlockOutput, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseArgs, ProviderInferenceResponseChunk, Text,
    TextChunk, TextKind,
};
use crate::inference::InferenceProvider;
use crate::model::{Credential, ModelProvider};
use crate::providers::helpers::inject_extra_request_data;
use crate::tool::{DynamicToolParams, FunctionTool, ToolCall, ToolCallWrapper};
use crate::variant::JsonMode;

const PROVIDER_NAME: &str = "TensorZero Relay";
pub const PROVIDER_TYPE: &str = "tensorzero_relay";

#[derive(Debug, Serialize, ts_rs::TS)]
pub struct TensorZeroRelayProvider {
    #[serde(skip)]
    client: crate::client::Client,
    gateway_base_url: Url,
    target_type: TensorZeroRelayTargetType,
    #[serde(skip)]
    #[expect(dead_code)]
    credentials: TensorZeroRelayCredentials,
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TensorZeroRelayTargetType {
    Function {
        function_name: String,
        variant_name: Option<String>,
    },
    Model {
        model_name: String,
    },
}

fn reject_extra_headers_and_body(
    request: &ModelProviderRequest<'_>,
    model_provider: &ModelProvider,
) -> Result<(), Error> {
    let res = inject_extra_request_data(
        &request.request.extra_body,
        &request.request.extra_headers,
        model_provider,
        request.model_name,
        &mut serde_json::json!({}),
    )?;
    if res.matched_any_headers_or_body {
        return Err(Error::new(ErrorDetails::Inference {
            message: "Extra headers and body are not supported for `tensorzero_relay` provider"
                .to_string(),
        }));
    }
    Ok(())
}

impl TensorZeroRelayProvider {
    pub async fn new(
        gateway_base_url: Url,
        function_name: Option<String>,
        model_name: Option<String>,
        variant_name: Option<String>,
        credentials: TensorZeroRelayCredentials,
        http_client: TensorzeroHttpClient,
    ) -> Result<Self, Error> {
        match credentials {
            TensorZeroRelayCredentials::None => {}
            _ => {
                return Err(Error::new(ErrorDetails::Config {
                    message: "Credentials are not yet supported for `tensorzero_relay` provider"
                        .to_string(),
                }))
            }
        }
        // Validate that exactly one of function_name or model_name is set
        let target_type = match (&function_name, &model_name) {
            (Some(fn_name), None) => TensorZeroRelayTargetType::Function {
                function_name: fn_name.clone(),
                variant_name,
            },
            (None, Some(model)) => {
                // Validate that variant_name is not set when using model_name
                if variant_name.is_some() {
                    return Err(Error::new(ErrorDetails::Config {
                        message: "variant_name can only be set when function_name is provided"
                            .to_string(),
                    }));
                }
                TensorZeroRelayTargetType::Model {
                    model_name: model.clone(),
                }
            }
            (Some(_), Some(_)) => {
                return Err(Error::new(ErrorDetails::Config {
                    message: "Cannot provide both function_name and model_name".to_string(),
                }))
            }
            (None, None) => {
                return Err(Error::new(ErrorDetails::Config {
                    message: "Exactly one of function_name or model_name must be provided"
                        .to_string(),
                }))
            }
        };

        let client =
            crate::client::ClientBuilder::new(crate::client::ClientBuilderMode::HTTPGateway {
                url: gateway_base_url.clone(),
            })
            .with_http_client(http_client)
            .build()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: e.to_string(),
                })
            })?;

        Ok(TensorZeroRelayProvider {
            client,
            gateway_base_url,
            target_type,
            credentials,
        })
    }

    pub fn model_name(&self) -> Option<&str> {
        match &self.target_type {
            TensorZeroRelayTargetType::Model { model_name } => Some(model_name.as_str()),
            TensorZeroRelayTargetType::Function { .. } => None,
        }
    }

    async fn build_client_inference_params(
        &self,
        model_provider_request: ModelProviderRequest<'_>,
        model_provider: &ModelProvider,
        stream: bool,
    ) -> Result<ClientInferenceParams, Error> {
        reject_extra_headers_and_body(&model_provider_request, model_provider)?;
        let request = &model_provider_request.request;
        let (function_name, variant_name) = match &self.target_type {
            TensorZeroRelayTargetType::Function {
                function_name,
                variant_name,
            } => (Some(function_name.clone()), variant_name.clone()),
            TensorZeroRelayTargetType::Model { .. } => (None, None),
        };
        let model_name = match &self.target_type {
            TensorZeroRelayTargetType::Function { .. } => None,
            TensorZeroRelayTargetType::Model { model_name } => Some(model_name.clone()),
        };

        let input = ClientInput {
            messages: try_join_all(request.messages.clone().into_iter().map(async |m| {
                Ok::<_, Error>(ClientInputMessage {
                    role: m.role,
                    content: try_join_all(m.content.into_iter().map(async |c| match c {
                        ContentBlock::Text(t) => {
                            Ok::<_, Error>(ClientInputMessageContent::Text(TextKind::Text {
                                text: t.text,
                            }))
                        }
                        ContentBlock::ToolCall(t) => Ok(ClientInputMessageContent::ToolCall(
                            ToolCallWrapper::ToolCall(t),
                        )),
                        ContentBlock::ToolResult(t) => Ok(ClientInputMessageContent::ToolResult(t)),
                        ContentBlock::Thought(t) => Ok(ClientInputMessageContent::Thought(t)),
                        ContentBlock::File(f) => match *f {
                            LazyFile::Url { file_url, future } => {
                                if request.fetch_and_encode_input_files_before_inference {
                                    let resolved_file = future.await?;
                                    Ok(ClientInputMessageContent::File(File::ObjectStorage(
                                        resolved_file,
                                    )))
                                } else {
                                    Ok(ClientInputMessageContent::File(File::Url(UrlFile {
                                        url: file_url.url,
                                        mime_type: file_url.mime_type,
                                        detail: file_url.detail,
                                        filename: file_url.filename,
                                    })))
                                }
                            }
                            LazyFile::Base64(pending) => {
                                Ok(ClientInputMessageContent::File(File::Base64(Base64File {
                                    source_url: pending.file.source_url.clone(),
                                    mime_type: pending.file.mime_type.clone(),
                                    data: pending.data.clone(),
                                    detail: pending.file.detail.clone(),
                                    filename: pending.file.filename.clone(),
                                })))
                            }
                            LazyFile::ObjectStoragePointer {
                                metadata,
                                storage_path,
                                future,
                            } => {
                                if request.fetch_and_encode_input_files_before_inference {
                                    let resolved_file = future.await?;
                                    Ok(ClientInputMessageContent::File(File::ObjectStorage(
                                        resolved_file,
                                    )))
                                } else {
                                    Ok(ClientInputMessageContent::File(File::ObjectStoragePointer(
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

                            LazyFile::ObjectStorage(object_storage_file) => {
                                Ok(ClientInputMessageContent::File(File::ObjectStorage(
                                    object_storage_file,
                                )))
                            }
                        },
                        ContentBlock::Unknown {
                            data,
                            model_provider_name,
                        } => Ok(ClientInputMessageContent::Unknown(Unknown {
                            data,
                            model_provider_name,
                        })),
                    }))
                    .await?,
                })
            }))
            .await?,
            system: request.system.clone().map(System::Text),
        };

        let res = ClientInferenceParams {
            function_name,
            variant_name,
            model_name,
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
                additional_tools: request.tool_config.as_ref().map(|config| {
                    config
                        .tools_available()
                        .map(|t| FunctionTool {
                            description: t.description().to_string(),
                            parameters: t.parameters().clone(),
                            name: t.name().to_string(),
                            strict: t.strict(),
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
            // TODO - implement extra_body and extra_headers
            extra_body: Default::default(),
            extra_headers: Default::default(),
            // We intentionally do for not forward the rest of these parameters,
            // we only apply them to this gateway, not the next hop.
            credentials: HashMap::new(),
            cache_options: Default::default(),
            internal_dynamic_variant_config: None,
            episode_id: None,
            dryrun: None,
            tags: HashMap::new(),
            otlp_traces_extra_headers: HashMap::new(),
            include_original_response: false,
        };
        Ok(res)
    }
}

impl InferenceProvider for TensorZeroRelayProvider {
    async fn infer<'a>(
        &'a self,
        model_provider_request: ModelProviderRequest<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let start_time = Instant::now();

        let client_inference_params = self
            .build_client_inference_params(model_provider_request, model_provider, false)
            .await?;

        let http_data = self
            .client
            .http_inference(client_inference_params)
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: e.to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: None,
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
        let finish_reason = non_streaming.finish_reason().cloned();

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
                            ContentBlockChatOutput::Unknown {
                                data,
                                model_provider_name,
                            } => Some(ContentBlockOutput::Unknown {
                                data,
                                model_provider_name,
                            }),
                        })
                        .collect(),
                    InferenceResponse::Json(json) => match json.output.raw {
                        Some(raw) => vec![ContentBlockOutput::Text(Text { text: raw })],
                        None => vec![],
                    },
                },
                system: model_provider_request.request.system.clone(),
                input_messages: model_provider_request.request.messages.clone(),
                raw_request: http_data.raw_request,
                raw_response: http_data.raw_response.unwrap_or_default(),
                usage,
                latency,
                finish_reason,
            },
        ))
    }

    async fn infer_stream<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let start_time = Instant::now();

        let client_inference_params = self
            .build_client_inference_params(request, model_provider, true)
            .await?;
        let res = self
            .client
            .http_inference(client_inference_params)
            .await
            .map_err(|e| {
                // TODO - what kind of error do we want to return here?
                Error::new(ErrorDetails::Inference {
                    message: e.to_string(),
                })
            })?;

        let InferenceOutput::Streaming(streaming) = res.response else {
            return Err(Error::new(ErrorDetails::Inference {
                message: "Expected streaming inference response".to_string(),
            }));
        };

        let stream = streaming
            .map(move |chunk| match chunk {
                Ok(chunk) => {
                    let raw_chunk = serde_json::to_string(&chunk).unwrap_or_default();
                    match chunk {
                        InferenceResponseChunk::Chat(c) => Ok(ProviderInferenceResponseChunk::new(
                            c.content,
                            c.usage,
                            // TODO - get the original chunk as a string
                            raw_chunk,
                            start_time.elapsed(),
                            c.finish_reason,
                        )),
                        InferenceResponseChunk::Json(c) => Ok(ProviderInferenceResponseChunk::new(
                            vec![ContentBlockChunk::Text(TextChunk {
                                id: "0".to_string(),
                                text: c.raw,
                            })],
                            c.usage,
                            // TODO - get the original chunk as a string
                            raw_chunk,
                            start_time.elapsed(),
                            c.finish_reason,
                        )),
                    }
                }
                Err(e) => Err(e),
            })
            .boxed()
            .peekable();

        Ok((stream, res.raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

#[derive(Clone, Debug)]
pub enum TensorZeroRelayCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<TensorZeroRelayCredentials>,
        fallback: Box<TensorZeroRelayCredentials>,
    },
}

impl TryFrom<Credential> for TensorZeroRelayCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(TensorZeroRelayCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(TensorZeroRelayCredentials::Dynamic(key_name)),
            Credential::None => Ok(TensorZeroRelayCredentials::None),
            Credential::Missing => Ok(TensorZeroRelayCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(TensorZeroRelayCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for TensorZero Relay provider".to_string(),
            })),
        }
    }
}

impl TensorZeroRelayCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a crate::endpoints::inference::InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, DelayedError> {
        match self {
            TensorZeroRelayCredentials::Static(api_key) => Ok(Some(api_key)),
            TensorZeroRelayCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).map(Some).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            TensorZeroRelayCredentials::None => Ok(None),
            TensorZeroRelayCredentials::WithFallback { default, fallback } => {
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
        }
    }
}
