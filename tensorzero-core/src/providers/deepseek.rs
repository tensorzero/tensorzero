use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use super::helpers::{
    convert_stream_error, inject_extra_request_data_and_send,
    inject_extra_request_data_and_send_eventsource,
};
use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DelayedError, DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::{TensorZeroEventSource, TensorzeroHttpClient};
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2,
};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, ContentBlockChunk, ContentBlockOutput, Latency,
    ModelInferenceRequest, ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, TextChunk, Thought, ThoughtChunk,
};
use crate::inference::InferenceProvider;
use crate::model::{Credential, ModelProvider};
use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::chat_completions::{ChatCompletionTool, ChatCompletionToolChoice};
use crate::providers::openai::OpenAIMessagesConfig;
use crate::providers::openai::{
    get_chat_url, handle_openai_error, prepare_system_or_developer_message,
    tensorzero_to_openai_messages, OpenAIAssistantRequestMessage, OpenAIContentBlock,
    OpenAIFinishReason, OpenAIRequestMessage, OpenAIResponseToolCall, OpenAISystemRequestMessage,
    OpenAIUsage, OpenAIUserRequestMessage, StreamOptions, SystemOrDeveloper,
};
use crate::tool::ToolCallChunk;

lazy_static! {
    static ref DEEPSEEK_DEFAULT_BASE_URL: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.deepseek.com/v1")
            .expect("Failed to parse DEEPSEEK_DEFAULT_BASE_URL")
    };
}

const PROVIDER_NAME: &str = "DeepSeek";
pub const PROVIDER_TYPE: &str = "deepseek";

#[derive(Clone, Debug)]
pub enum DeepSeekCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<DeepSeekCredentials>,
        fallback: Box<DeepSeekCredentials>,
    },
}

impl TryFrom<Credential> for DeepSeekCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(DeepSeekCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(DeepSeekCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(DeepSeekCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(DeepSeekCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for DeepSeek provider".to_string(),
            })),
        }
    }
}

impl DeepSeekCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            DeepSeekCredentials::Static(api_key) => Ok(api_key),
            DeepSeekCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            DeepSeekCredentials::WithFallback { default, fallback } => {
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
            DeepSeekCredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
        }
    }
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct DeepSeekProvider {
    model_name: String,
    #[serde(skip)]
    credentials: DeepSeekCredentials,
}

impl DeepSeekProvider {
    pub fn new(model_name: String, credentials: DeepSeekCredentials) -> Self {
        DeepSeekProvider {
            model_name,
            credentials,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl InferenceProvider for DeepSeekProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(
            DeepSeekRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing DeepSeek request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let request_url = get_chat_url(&DEEPSEEK_DEFAULT_BASE_URL)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let request_builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;

        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(DeepSeekResponseWithMetadata {
                response,
                raw_response,
                latency,
                raw_request,
                generic_request: request,
            }
            .try_into()?)
        } else {
            let status = res.status();

            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            Err(handle_openai_error(
                &raw_request,
                status,
                &response,
                PROVIDER_TYPE,
            ))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(
            DeepSeekRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing DeepSeek request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let request_url = get_chat_url(&DEEPSEEK_DEFAULT_BASE_URL)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let request_builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());
        let (event_source, raw_request) = inject_extra_request_data_and_send_eventsource(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;

        let stream = stream_deepseek(event_source, start_time, &raw_request).peekable();
        Ok((stream, raw_request))
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

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum DeepSeekResponseFormat {
    #[default]
    Text,
    JsonObject,
}

impl DeepSeekResponseFormat {
    fn new(json_mode: ModelInferenceRequestJsonMode) -> Option<Self> {
        match json_mode {
            ModelInferenceRequestJsonMode::On => Some(DeepSeekResponseFormat::JsonObject),
            // For now, we never explicitly send `DeepSeekResponseFormat::Text`
            ModelInferenceRequestJsonMode::Off => None,
            ModelInferenceRequestJsonMode::Strict => Some(DeepSeekResponseFormat::JsonObject),
        }
    }
}

#[derive(Debug, Default, Serialize)]
struct DeepSeekRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,

    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ChatCompletionTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ChatCompletionToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<DeepSeekResponseFormat>,
}

fn apply_inference_params(
    _request: &mut DeepSeekRequest,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "reasoning_effort", None);
    }

    if service_tier.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier", None);
    }

    if thinking_budget_tokens.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "thinking_budget_tokens", None);
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
}

impl<'a> DeepSeekRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<DeepSeekRequest<'a>, Error> {
        let ModelInferenceRequest {
            temperature,
            max_tokens,
            seed,
            top_p,
            presence_penalty,
            frequency_penalty,
            stream,
            ..
        } = *request;

        let stream_options = if request.stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        };

        if request.json_mode == ModelInferenceRequestJsonMode::Strict {
            tracing::warn!("DeepSeek provider does not support strict JSON mode. Downgrading to normal JSON mode.");
        }

        let response_format = DeepSeekResponseFormat::new(request.json_mode);

        // NOTE: as mentioned by the DeepSeek team here: https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations
        // the R1 series of models does not perform well with the system prompt. As we move towards first-class support for reasoning models we should check
        // if a model is an R1 model and if so, remove the system prompt from the request and instead put it in the first user message.
        let messages = prepare_deepseek_messages(
            request,
            model,
            OpenAIMessagesConfig {
                json_mode: Some(&request.json_mode),
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: request
                    .fetch_and_encode_input_files_before_inference,
            },
        )
        .await?;

        let (tools, tool_choice, _) = prepare_chat_completion_tools(request, false)?;

        let mut deepseek_request = DeepSeekRequest {
            messages,
            model,
            temperature,
            max_tokens,
            seed,
            top_p,
            stop: request.borrow_stop_sequences(),
            presence_penalty,
            frequency_penalty,
            stream,
            stream_options,
            response_format,
            tools,
            tool_choice,
            // allowed_tools is now part of tool_choice (AllowedToolsChoice variant)
        };

        apply_inference_params(&mut deepseek_request, &request.inference_params_v2);

        Ok(deepseek_request)
    }
}

fn stream_deepseek(
    mut event_source: TensorZeroEventSource,
    start_time: Instant,
    raw_request: &str,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    let mut tool_call_ids = Vec::new();
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(raw_request.clone(), PROVIDER_TYPE.to_string(), e).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<DeepSeekChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {e}",
                                ),
                                raw_request: Some(raw_request.clone()),
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            deepseek_to_tensorzero_chunk(message.data, d, latency, &mut tool_call_ids)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    })
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct DeepSeekFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct DeepSeekToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: DeepSeekFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct DeepSeekDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<DeepSeekToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct DeepSeekChatChunkChoice {
    delta: DeepSeekDelta,
    finish_reason: Option<OpenAIFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct DeepSeekChatChunk {
    choices: Vec<DeepSeekChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAIUsage>,
}

/// Maps a DeepSeek chunk to a TensorZero chunk for streaming inferences
fn deepseek_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: DeepSeekChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
) -> Result<ProviderInferenceResponseChunk, Error> {
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            raw_request: None,
            raw_response: Some(serde_json::to_string(&chunk).unwrap_or_default()),
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into());
    }
    let usage = chunk.usage.map(OpenAIUsage::into);
    let mut content = vec![];
    let mut finish_reason = None;
    if let Some(choice) = chunk.choices.pop() {
        if let Some(choice_finish_reason) = choice.finish_reason {
            finish_reason = Some(choice_finish_reason.into());
        }
        if let Some(text) = choice.delta.content {
            content.push(ContentBlockChunk::Text(TextChunk {
                text,
                id: "0".to_string(),
            }));
        }
        if let Some(reasoning) = choice.delta.reasoning_content {
            content.push(ContentBlockChunk::Thought(ThoughtChunk {
                text: Some(reasoning),
                signature: None,
                id: "0".to_string(),
                summary_id: None,
                summary_text: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
            }));
        }
        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                let index = tool_call.index;
                let id = match tool_call.id {
                    Some(id) => {
                        tool_call_ids.push(id.clone());
                        id
                    }
                    None => {
                        tool_call_ids
                            .get(index as usize)
                            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                                message: "Tool call index out of bounds (meaning we haven't seen this many ids in the stream)".to_string(),
                                raw_request: None,
                                raw_response: None,
                                provider_type: PROVIDER_TYPE.to_string(),
                            }))?
                            .clone()
                    }
                };

                content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                    id,
                    raw_name: tool_call.function.name,
                    raw_arguments: tool_call.function.arguments.unwrap_or_default(),
                }));
            }
        }
    }

    Ok(ProviderInferenceResponseChunk::new(
        content,
        usage,
        raw_message,
        latency,
        finish_reason,
    ))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct DeepSeekResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct DeepSeekResponseChoice {
    index: u8,
    message: DeepSeekResponseMessage,
    finish_reason: Option<OpenAIFinishReason>,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct DeepSeekResponse {
    choices: Vec<DeepSeekResponseChoice>,
    usage: OpenAIUsage,
}

pub(super) async fn prepare_deepseek_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
    model_name: &'a str,
    config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        messages.extend(tensorzero_to_openai_messages(message, config).await?);
    }
    // If this is an R1 model, prepend the system message as the first user message instead of using it as a system message
    if model_name.to_lowercase().contains("reasoner") {
        if let Some(system) = request.system.as_deref() {
            messages.insert(
                0,
                OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                    content: vec![OpenAIContentBlock::Text {
                        text: Cow::Borrowed(system),
                    }],
                }),
            );
        }
    } else if let Some(system_msg) = prepare_system_or_developer_message(
        request
            .system
            .as_deref()
            .map(|m| SystemOrDeveloper::System(Cow::Borrowed(m))),
        Some(&request.json_mode),
        &messages,
    ) {
        messages.insert(0, system_msg);
    }
    messages = coalesce_consecutive_messages(messages);
    Ok(messages)
}

struct DeepSeekResponseWithMetadata<'a> {
    response: DeepSeekResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<DeepSeekResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: DeepSeekResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let DeepSeekResponseWithMetadata {
            mut response,
            raw_response,
            latency,
            raw_request,
            generic_request,
        } = value;

        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices {}, Expected 1",
                    response.choices.len()
                ),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }

        let usage = response.usage.into();
        let DeepSeekResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(reasoning) = message.reasoning_content {
            content.push(ContentBlockOutput::Thought(Thought {
                text: Some(reasoning),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
            }));
        }
        if let Some(text) = message.content {
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        }
        let system = generic_request.system.clone();
        let messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages: messages,
                raw_request,
                raw_response,
                usage,
                latency,
                finish_reason: finish_reason.map(OpenAIFinishReason::into),
            },
        ))
    }
}

/// If a message is a system, user, or assistant message and the next message is the same type, coalesce them into a single message
/// Required for DeepSeek reasoner type models
fn coalesce_consecutive_messages(messages: Vec<OpenAIRequestMessage>) -> Vec<OpenAIRequestMessage> {
    let mut result = messages;
    let mut i = 0;
    while i < result.len().saturating_sub(1) {
        let current = &result[i];
        let next = &result[i + 1];

        match (current, next) {
            (OpenAIRequestMessage::System(curr), OpenAIRequestMessage::System(next)) => {
                let combined = format!("{}\n\n{}", curr.content, next.content);
                result[i] = OpenAIRequestMessage::System(OpenAISystemRequestMessage {
                    content: Cow::Owned(combined),
                });
                result.remove(i + 1);
            }
            (OpenAIRequestMessage::User(curr), OpenAIRequestMessage::User(next)) => {
                let mut combined = curr.content.clone();
                combined.extend(next.content.iter().cloned());
                result[i] =
                    OpenAIRequestMessage::User(OpenAIUserRequestMessage { content: combined });
                result.remove(i + 1);
            }
            (OpenAIRequestMessage::Assistant(curr), OpenAIRequestMessage::Assistant(next)) => {
                let combined_content = match (curr.content.as_ref(), next.content.as_ref()) {
                    (Some(c1), Some(c2)) => {
                        let mut combined = c1.clone();
                        combined.extend(c2.iter().cloned());
                        Some(combined)
                    }
                    (Some(c), None) | (None, Some(c)) => Some(c.clone()),
                    (None, None) => None,
                };

                let combined_tool_calls = match (&curr.tool_calls, &next.tool_calls) {
                    (Some(t1), Some(t2)) => {
                        let mut combined = t1.clone();
                        combined.extend(t2.iter().cloned());
                        Some(combined)
                    }
                    (Some(t), None) | (None, Some(t)) => Some(t.clone()),
                    (None, None) => None,
                };

                result[i] = OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                    content: combined_content,
                    tool_calls: combined_tool_calls,
                });
                result.remove(i + 1);
            }
            _ => i += 1,
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;
    use std::time::Duration;
    use uuid::Uuid;

    use crate::inference::types::{
        FinishReason, FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };
    use crate::providers::chat_completions::{
        ChatCompletionSpecificToolChoice, ChatCompletionSpecificToolFunction,
        ChatCompletionToolChoice, ChatCompletionToolType,
    };
    use crate::providers::openai::{
        OpenAIRequestFunctionCall, OpenAIRequestToolCall, OpenAIToolRequestMessage, OpenAIToolType,
        OpenAIUsage,
    };
    use crate::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};

    #[tokio::test]
    async fn test_deepseek_request_new() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let deepseek_request = DeepSeekRequest::new("deepseek-chat", &request_with_tools)
            .await
            .expect("failed to create Deepseek Request during test");

        assert_eq!(deepseek_request.messages.len(), 1);
        assert_eq!(deepseek_request.temperature, Some(0.5));
        assert_eq!(deepseek_request.max_tokens, Some(100));
        assert!(!deepseek_request.stream);
        assert_eq!(deepseek_request.seed, Some(69));
        assert!(deepseek_request.tools.is_some());
        let tools = deepseek_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool = &tools[0];
        assert_eq!(tool.function.name, WEATHER_TOOL.name());
        assert_eq!(tool.function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            deepseek_request.tool_choice,
            Some(ChatCompletionToolChoice::Specific(
                ChatCompletionSpecificToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionSpecificToolFunction {
                        name: WEATHER_TOOL.name(),
                    }
                }
            ))
        );

        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Json,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let deepseek_request = DeepSeekRequest::new("deepseek-chat", &request_with_tools)
            .await
            .expect("failed to create Deepseek Request");

        assert_eq!(deepseek_request.messages.len(), 2);
        assert_eq!(deepseek_request.temperature, Some(0.5));
        assert_eq!(deepseek_request.max_tokens, Some(100));
        assert_eq!(deepseek_request.top_p, Some(0.9));
        assert_eq!(deepseek_request.presence_penalty, Some(0.1));
        assert_eq!(deepseek_request.frequency_penalty, Some(0.2));
        assert!(!deepseek_request.stream);
        assert_eq!(deepseek_request.seed, Some(69));

        assert!(deepseek_request.tools.is_some());
        let tools = deepseek_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool = &tools[0];
        assert_eq!(tool.function.name, WEATHER_TOOL.name());
        assert_eq!(tool.function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            deepseek_request.tool_choice,
            Some(ChatCompletionToolChoice::Specific(
                ChatCompletionSpecificToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionSpecificToolFunction {
                        name: WEATHER_TOOL.name(),
                    }
                }
            ))
        );

        let request_with_tools = ModelInferenceRequest {
            json_mode: ModelInferenceRequestJsonMode::Strict,
            ..request_with_tools
        };

        let deepseek_request = DeepSeekRequest::new("deepseek-chat", &request_with_tools).await;
        let deepseek_request = deepseek_request.unwrap();
        // We should downgrade the strict JSON mode to normal JSON mode for deepseek
        assert_eq!(
            deepseek_request.response_format,
            Some(DeepSeekResponseFormat::JsonObject)
        );
    }

    #[tokio::test]
    async fn test_deepseek_api_base() {
        assert_eq!(
            DEEPSEEK_DEFAULT_BASE_URL.as_str(),
            "https://api.deepseek.com/v1"
        );
    }

    #[tokio::test]
    async fn test_credential_to_deepseek_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: DeepSeekCredentials = DeepSeekCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, DeepSeekCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = DeepSeekCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, DeepSeekCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = DeepSeekCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, DeepSeekCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = DeepSeekCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
    #[tokio::test]
    async fn test_deepseek_response_with_metadata_try_into() {
        let valid_response = DeepSeekResponse {
            choices: vec![DeepSeekResponseChoice {
                index: 0,
                message: DeepSeekResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    reasoning_content: Some("I'm thinking about the weather".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(OpenAIFinishReason::Stop),
            }],
            usage: OpenAIUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let deepseek_response_with_metadata = DeepSeekResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &DeepSeekRequest::new("deepseek-chat", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
        };
        let inference_response: ProviderInferenceResponse =
            deepseek_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 2);
        assert_eq!(
            inference_response.output[0],
            ContentBlockOutput::Thought(Thought {
                text: Some("I'm thinking about the weather".to_string()),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
            })
        );

        assert_eq!(
            inference_response.output[1],
            "Hello, world!".to_string().into()
        );
        assert_eq!(inference_response.raw_response, "test_response");
        assert_eq!(inference_response.usage.input_tokens, Some(10));
        assert_eq!(inference_response.usage.output_tokens, Some(20));
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_secs(0)
            }
        );
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn test_prepare_deepseek_messages() {
        // Test case 1: Regular model with system message
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["Hello".to_string().into()],
            }],
            system: Some("System prompt".to_string()),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            stream: false,
            seed: None,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let messages = prepare_deepseek_messages(
            &request,
            "deepseek-chat",
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[0], OpenAIRequestMessage::System(_)));
        assert!(matches!(messages[1], OpenAIRequestMessage::User(_)));

        // Test case 2: Reasoner model with system message
        let messages = prepare_deepseek_messages(
            &request,
            "deepseek-reasoner",
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            OpenAIRequestMessage::User(user_msg) => {
                assert_eq!(
                    user_msg.content,
                    vec![
                        OpenAIContentBlock::Text {
                            text: "System prompt".into(),
                        },
                        OpenAIContentBlock::Text {
                            text: "Hello".into(),
                        },
                    ]
                );
            }
            _ => panic!("Expected a user message"),
        }

        // Test case 3: Regular model without system message
        let request_no_system = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["Hello".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            stream: false,
            seed: None,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let messages = prepare_deepseek_messages(
            &request_no_system,
            "deepseek-chat",
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(messages.len(), 1);
        assert!(matches!(messages[0], OpenAIRequestMessage::User(_)));

        // Test case 4: Multiple messages with different roles
        let request_multiple = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![
                RequestMessage {
                    role: Role::User,
                    content: vec!["Hello".to_string().into()],
                },
                RequestMessage {
                    role: Role::Assistant,
                    content: vec!["Hi there!".to_string().into()],
                },
                RequestMessage {
                    role: Role::User,
                    content: vec!["How are you?".to_string().into()],
                },
            ],
            system: Some("Be helpful".to_string()),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            stream: false,
            seed: None,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let messages = prepare_deepseek_messages(
            &request_multiple,
            "deepseek-chat",
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[0], OpenAIRequestMessage::System(_)));
        assert!(matches!(messages[1], OpenAIRequestMessage::User(_)));
        assert!(matches!(messages[2], OpenAIRequestMessage::Assistant(_)));
        assert!(matches!(messages[3], OpenAIRequestMessage::User(_)));
    }

    // Helper constructors for test messages.
    fn system_message(content: &str) -> OpenAIRequestMessage<'_> {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(content),
        })
    }
    fn user_message(content: &str) -> OpenAIRequestMessage<'_> {
        OpenAIRequestMessage::User(OpenAIUserRequestMessage {
            content: vec![OpenAIContentBlock::Text {
                text: content.into(),
            }],
        })
    }
    fn assistant_message<'a>(
        content: Option<&'a str>,
        tool_calls: Option<Vec<OpenAIRequestToolCall<'a>>>,
    ) -> OpenAIRequestMessage<'a> {
        OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
            content: content.map(|c| vec![OpenAIContentBlock::Text { text: c.into() }]),
            tool_calls,
        })
    }
    fn tool_message<'a>(content: &'a str, tool_call_id: &'a str) -> OpenAIRequestMessage<'a> {
        OpenAIRequestMessage::Tool(OpenAIToolRequestMessage {
            content,
            tool_call_id,
        })
    }

    #[test]
    fn test_deepseek_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = DeepSeekRequest::default();

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort warns
        assert!(logs_contain(
            "DeepSeek does not support the inference parameter `reasoning_effort`"
        ));

        // Test that thinking_budget_tokens warns
        assert!(logs_contain(
            "DeepSeek does not support the inference parameter `thinking_budget_tokens`"
        ));

        // Test that verbosity warns
        assert!(logs_contain(
            "DeepSeek does not support the inference parameter `verbosity`"
        ));
    }

    #[tokio::test]
    async fn test_coalesce_consecutive_messages() {
        // Create dummy tool calls to test assistant message merging.
        let tool_call1 = OpenAIRequestToolCall {
            id: "tc1".into(),
            r#type: OpenAIToolType::Function,
            function: OpenAIRequestFunctionCall {
                name: "func1".into(),
                arguments: "args1".into(),
            },
        };
        let tool_call2 = OpenAIRequestToolCall {
            id: "tc2".into(),
            r#type: OpenAIToolType::Function,
            function: OpenAIRequestFunctionCall {
                name: "func2".into(),
                arguments: "args2".into(),
            },
        };

        // Test 1: Empty input.
        let input: Vec<OpenAIRequestMessage> = vec![];
        let output = coalesce_consecutive_messages(input);
        assert!(output.is_empty());

        // Test 2: Single message (system) remains unchanged.
        let input = vec![system_message("Only system message")];
        let output = coalesce_consecutive_messages(input);
        assert_eq!(output, vec![system_message("Only system message")]);

        // Test 3: Consecutive system messages are merged.
        let input = vec![
            system_message("Sys1"),
            system_message("Sys2"),
            system_message("Sys3"),
        ];
        let output = coalesce_consecutive_messages(input);
        let expected = vec![system_message("Sys1\n\nSys2\n\nSys3")];
        assert_eq!(output, expected);

        // Test 4: Consecutive user messages are merged.
        let input = vec![user_message("User1"), user_message("User2")];
        let output = coalesce_consecutive_messages(input);
        let expected = vec![OpenAIRequestMessage::User(OpenAIUserRequestMessage {
            content: vec![
                OpenAIContentBlock::Text {
                    text: "User1".into(),
                },
                OpenAIContentBlock::Text {
                    text: "User2".into(),
                },
            ],
        })];
        assert_eq!(output, expected);

        // Test 5: Consecutive assistant messages with both content and tool_calls.
        let input = vec![
            assistant_message(Some("Ass1"), Some(vec![tool_call1.clone()])),
            assistant_message(Some("Ass2"), Some(vec![tool_call2.clone()])),
        ];
        let content = vec![
            OpenAIContentBlock::Text {
                text: "Ass1".into(),
            },
            OpenAIContentBlock::Text {
                text: "Ass2".into(),
            },
        ];
        let output = coalesce_consecutive_messages(input);
        let expected = vec![OpenAIRequestMessage::Assistant(
            OpenAIAssistantRequestMessage {
                content: Some(content),
                tool_calls: Some(vec![tool_call1.clone(), tool_call2.clone()]),
            },
        )];
        assert_eq!(output, expected);

        // Test 6: Consecutive assistant messages where one message lacks content/tool_calls.
        let input = vec![
            assistant_message(Some("Ass3"), None),
            assistant_message(None, Some(vec![tool_call1.clone()])),
        ];
        let output = coalesce_consecutive_messages(input);
        // Merging: (Some("Ass3"), None) gives Some("Ass3"), and (None, Some(...)) gives the available tool_calls.
        let expected = vec![assistant_message(
            Some("Ass3"),
            Some(vec![tool_call1.clone()]),
        )];
        assert_eq!(output, expected);

        // Test 7: Assistant messages with both None contents but with tool_calls.
        let input = vec![
            assistant_message(None, Some(vec![tool_call1.clone()])),
            assistant_message(None, Some(vec![tool_call2.clone()])),
        ];
        let output = coalesce_consecutive_messages(input);
        let expected = vec![assistant_message(
            None,
            Some(vec![tool_call1.clone(), tool_call2.clone()]),
        )];
        assert_eq!(output, expected);

        // Test 8: Mixed message types should not merge across types.
        let input = vec![
            system_message("Sys1"),
            user_message("User1"),
            system_message("Sys2"),
            assistant_message(Some("Ass1"), None),
            tool_message("Tool1", "id1"),
            tool_message("Tool2", "id2"),
            assistant_message(Some("Ass2"), None),
            assistant_message(Some("Ass3"), None),
        ];
        let output = coalesce_consecutive_messages(input);
        // Only the two assistant messages at the end should merge.
        let expected = vec![
            system_message("Sys1"),
            user_message("User1"),
            system_message("Sys2"),
            assistant_message(Some("Ass1"), None),
            tool_message("Tool1", "id1"),
            tool_message("Tool2", "id2"),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some(vec![
                    OpenAIContentBlock::Text {
                        text: "Ass2".into(),
                    },
                    OpenAIContentBlock::Text {
                        text: "Ass3".into(),
                    },
                ]),
                tool_calls: None,
            }),
        ];
        assert_eq!(output, expected);

        // Test 9: More than two consecutive assistant messages.
        let input = vec![
            assistant_message(Some("A1"), Some(vec![tool_call1.clone()])),
            assistant_message(None, None),
            assistant_message(Some("A3"), None),
        ];
        let output = coalesce_consecutive_messages(input);
        // First merge: ("A1", None) => "A1" with tool_calls preserved; then ("A1", "A3") => "A1\n\nA3".
        let expected = vec![OpenAIRequestMessage::Assistant(
            OpenAIAssistantRequestMessage {
                content: Some(vec![
                    OpenAIContentBlock::Text { text: "A1".into() },
                    OpenAIContentBlock::Text { text: "A3".into() },
                ]),
                tool_calls: Some(vec![tool_call1.clone()]),
            },
        )];
        assert_eq!(output, expected);
    }
}
