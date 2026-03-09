use std::{borrow::Cow, time::Duration};

use crate::{
    http::{TensorZeroEventSource, TensorzeroHttpClient},
    providers::openai::OpenAIMessagesConfig,
};
use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest::StatusCode;
use reqwest_sse_stream::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use serde_json::Value;
use tensorzero_types_providers::mistral::{
    MistralChatChunk, MistralContent, MistralContentChunk, MistralFinishReason, MistralResponse,
    MistralResponseChoice, MistralResponseFormat, MistralResponseToolCall, MistralThinkingSubChunk,
    MistralUsage,
};
use tokio::time::Instant;
use url::Url;

use crate::inference::types::usage::raw_usage_entries_from_value;
use crate::{
    cache::ModelProviderRequest,
    endpoints::inference::InferenceCredentials,
    error::{DelayedError, DisplayOrDebugGateway, Error, ErrorDetails},
    inference::{
        InferenceProvider,
        types::{
            ApiType, ContentBlock, ContentBlockChunk, ContentBlockOutput, FinishReason, Latency,
            ModelInferenceRequest, ModelInferenceRequestJsonMode,
            PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
            ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
            ProviderInferenceResponseStreamInner, RequestMessage, Role, TextChunk, Thought,
            ThoughtChunk, Usage,
            batch::{
                BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
            },
            chat_completion_inference_params::{
                ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported,
            },
        },
    },
    model::{Credential, ModelProvider},
    providers::helpers::{
        check_new_tool_call_name, convert_stream_error, inject_extra_request_data_and_send,
        inject_extra_request_data_and_send_eventsource,
    },
    tool::{FunctionToolConfig, ToolCall, ToolCallChunk, ToolChoice},
};
use uuid::Uuid;

use super::openai::{
    OpenAIFunction, OpenAIRequestFunctionCall, OpenAIRequestMessage, OpenAIRequestToolCall,
    OpenAISystemRequestMessage, OpenAIToolType, get_chat_url, tensorzero_to_openai_messages,
};

lazy_static! {
    static ref MISTRAL_API_BASE: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.mistral.ai/v1/").expect("Failed to parse MISTRAL_API_BASE")
    };
}

const PROVIDER_NAME: &str = "Mistral";
pub const PROVIDER_TYPE: &str = "mistral";

type PreparedMistralToolsResult<'a> = (
    Option<Vec<MistralTool<'a>>>,
    Option<MistralToolChoice<'a>>,
    Option<bool>,
);

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MistralProvider {
    model_name: String,
    #[serde(skip)]
    credentials: MistralCredentials,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_mode: Option<String>,
}

impl MistralProvider {
    pub fn new(
        model_name: String,
        credentials: MistralCredentials,
        prompt_mode: Option<String>,
    ) -> Self {
        MistralProvider {
            model_name,
            credentials,
            prompt_mode,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum MistralCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<MistralCredentials>,
        fallback: Box<MistralCredentials>,
    },
}

impl TryFrom<Credential> for MistralCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(MistralCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(MistralCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(MistralCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(MistralCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Mistral provider".to_string(),
            })),
        }
    }
}

impl MistralCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            MistralCredentials::Static(api_key) => Ok(api_key),
            MistralCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            MistralCredentials::WithFallback { default, fallback } => {
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
            MistralCredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
        }
    }
}

impl InferenceProvider for MistralProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
            model_inference_id,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(
            MistralRequest::new(&self.model_name, request, self.prompt_mode.as_deref()).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Mistral request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let request_url = get_chat_url(&MISTRAL_API_BASE)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            ApiType::ChatCompletions,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            builder,
        )
        .await?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    api_type: ApiType::ChatCompletions,
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    api_type: ApiType::ChatCompletions,
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            MistralResponseWithMetadata {
                response,
                latency,
                raw_response,
                raw_request,
                generic_request: request,
                model_inference_id,
            }
            .try_into()
        } else {
            handle_mistral_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: PROVIDER_TYPE.to_string(),
                        api_type: ApiType::ChatCompletions,
                        raw_request: Some(raw_request),
                        raw_response: None,
                    })
                })?,
                ApiType::ChatCompletions,
            )
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
            model_inference_id,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(
            MistralRequest::new(&self.model_name, request, self.prompt_mode.as_deref()).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Mistral request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let request_url = get_chat_url(&MISTRAL_API_BASE)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (event_source, raw_request) = inject_extra_request_data_and_send_eventsource(
            PROVIDER_TYPE,
            ApiType::ChatCompletions,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            builder,
        )
        .await?;
        let stream =
            stream_mistral(event_source, start_time, &raw_request, model_inference_id).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Mistral".to_string(),
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

fn handle_mistral_error(
    response_code: StatusCode,
    response_body: &str,
    api_type: ApiType,
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => Err(ErrorDetails::InferenceClient {
            message: response_body.to_string(),
            status_code: Some(response_code),
            provider_type: PROVIDER_TYPE.to_string(),
            api_type,
            raw_request: None,
            raw_response: None,
        }
        .into()),
        _ => Err(ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            api_type,
            raw_request: None,
            raw_response: None,
        }
        .into()),
    }
}

pub fn stream_mistral(
    mut event_source: TensorZeroEventSource,
    start_time: Instant,
    raw_request: &str,
    model_inference_id: Uuid,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            let mut last_tool_name = None;
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(raw_request.clone(), PROVIDER_TYPE.to_string(), ApiType::ChatCompletions, *e, None).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<MistralChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {}, Data: {}",
                                    e, message.data
                                ),
                                provider_type: PROVIDER_TYPE.to_string(),
                                api_type: ApiType::ChatCompletions,
                                raw_request: Some(raw_request.clone()),
                                raw_response: None,
                            }.into());
                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            mistral_to_tensorzero_chunk(
                                message.data,
                                d,
                                latency,
                                &mut last_tool_name,
                                model_inference_id,
                                PROVIDER_TYPE,
                            )
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

/// A request-side thinking sub-chunk. Mirrors the response format: `[{"type":"text","text":"..."}]`.
#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum MistralRequestThinkingSubChunk<'a> {
    Text { text: Cow<'a, str> },
}

/// A request-side content chunk for Mistral assistant messages.
/// When reasoning is enabled, thinking chunks are placed alongside text chunks in the content array.
#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum MistralRequestContentChunk<'a> {
    Text {
        text: Cow<'a, str>,
    },
    Thinking {
        thinking: Vec<MistralRequestThinkingSubChunk<'a>>,
    },
}

/// A Mistral-specific assistant message that puts thinking in the content array.
#[derive(Debug, Serialize, PartialEq)]
pub(super) struct MistralAssistantRequestMessage<'a> {
    role: &'static str,
    #[serde(serialize_with = "serialize_mistral_assistant_content")]
    content: Vec<MistralRequestContentChunk<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIRequestToolCall<'a>>>,
}

/// Custom serializer for assistant content:
/// - Empty: serialize as `null` (omitted via skip_serializing_if on the field would also work,
///   but Mistral expects the field to be present for assistant messages)
/// - Single Text chunk: serialize as a plain string for non-reasoning compatibility
/// - Otherwise: serialize as an array of typed chunks
fn serialize_mistral_assistant_content<S>(
    content: &Vec<MistralRequestContentChunk<'_>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match content.as_slice() {
        [] => serializer.serialize_none(),
        [MistralRequestContentChunk::Text { text }] => text.serialize(serializer),
        _ => content.serialize(serializer),
    }
}

/// Wraps OpenAI messages for system/user/tool roles, but uses a custom type for assistant.
#[derive(Debug, Serialize, PartialEq)]
#[serde(untagged)]
pub(super) enum MistralRequestMessage<'a> {
    OpenAI(OpenAIRequestMessage<'a>),
    MistralAssistant(MistralAssistantRequestMessage<'a>),
}

pub(super) async fn prepare_mistral_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
    config: OpenAIMessagesConfig<'a>,
    supports_reasoning: bool,
) -> Result<Vec<MistralRequestMessage<'a>>, Error> {
    // Convert all messages concurrently, then assemble in order.
    // Each slot holds either converted OpenAI messages or a Mistral reasoning assistant message.
    enum ConvertedMessage<'a> {
        OpenAI(Vec<OpenAIRequestMessage<'a>>),
        MistralAssistant(Option<MistralAssistantRequestMessage<'a>>),
    }

    let conversion_futures: Vec<_> = request
        .messages
        .iter()
        .map(|msg| async move {
            if supports_reasoning && msg.role == Role::Assistant {
                let assistant_msg = tensorzero_to_mistral_assistant_message(msg)?;
                Ok(ConvertedMessage::MistralAssistant(assistant_msg))
            } else {
                let openai_msgs = tensorzero_to_openai_messages(msg, config).await?;
                Ok(ConvertedMessage::OpenAI(openai_msgs))
            }
        })
        .collect();

    let converted: Vec<Result<ConvertedMessage<'a>, Error>> =
        futures::future::join_all(conversion_futures).await;

    let mut messages: Vec<MistralRequestMessage<'a>> = Vec::new();

    if let Some(system_msg) = tensorzero_to_mistral_system_message(request.system.as_deref()) {
        messages.push(MistralRequestMessage::OpenAI(system_msg));
    }

    for result in converted {
        match result? {
            ConvertedMessage::OpenAI(openai_msgs) => {
                for m in openai_msgs {
                    if !m.no_content() {
                        messages.push(MistralRequestMessage::OpenAI(m));
                    }
                }
            }
            ConvertedMessage::MistralAssistant(Some(assistant_msg)) => {
                messages.push(MistralRequestMessage::MistralAssistant(assistant_msg));
            }
            ConvertedMessage::MistralAssistant(None) => {}
        }
    }

    Ok(messages)
}

/// Convert a TensorZero assistant message to a Mistral assistant message with typed content chunks.
fn tensorzero_to_mistral_assistant_message<'a>(
    message: &'a RequestMessage,
) -> Result<Option<MistralAssistantRequestMessage<'a>>, Error> {
    let mut content_chunks: Vec<MistralRequestContentChunk<'a>> = Vec::new();
    let mut tool_calls: Vec<OpenAIRequestToolCall<'a>> = Vec::new();

    for block in &message.content {
        match block {
            ContentBlock::Text(text) => {
                content_chunks.push(MistralRequestContentChunk::Text {
                    text: Cow::Borrowed(&text.text),
                });
            }
            ContentBlock::ToolCall(tool_call) => {
                tool_calls.push(OpenAIRequestToolCall {
                    id: Cow::Borrowed(&tool_call.id),
                    r#type: OpenAIToolType::Function,
                    function: OpenAIRequestFunctionCall {
                        name: Cow::Borrowed(&tool_call.name),
                        arguments: Cow::Borrowed(&tool_call.arguments),
                    },
                });
            }
            ContentBlock::Thought(thought) => {
                if let Some(text) = &thought.text {
                    content_chunks.push(MistralRequestContentChunk::Thinking {
                        thinking: vec![MistralRequestThinkingSubChunk::Text {
                            text: Cow::Borrowed(text),
                        }],
                    });
                }
            }
            ContentBlock::ToolResult(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool results are not supported in assistant messages".to_string(),
                }));
            }
            _ => {
                // Ignore other block types (e.g. File, Unknown) in assistant messages
            }
        }
    }

    if content_chunks.is_empty() && tool_calls.is_empty() {
        return Ok(None);
    }

    let tool_calls = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    Ok(Some(MistralAssistantRequestMessage {
        role: "assistant",
        content: content_chunks,
        tool_calls,
    }))
}

fn tensorzero_to_mistral_system_message(system: Option<&str>) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(untagged)]
pub(super) enum MistralToolChoice<'a> {
    String(MistralToolChoiceString),
    Specific(MistralSpecificToolChoice<'a>),
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum MistralToolChoiceString {
    Auto,
    None,
    Any,
}

#[derive(Debug, Serialize, PartialEq)]
pub(super) struct MistralSpecificToolChoice<'a> {
    r#type: &'static str,
    function: MistralSpecificToolFunction<'a>,
}

#[derive(Debug, Serialize, PartialEq)]
struct MistralSpecificToolFunction<'a> {
    name: &'a str,
}

impl<'a> From<&'a ToolChoice> for MistralToolChoice<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::Auto => MistralToolChoice::String(MistralToolChoiceString::Auto),
            ToolChoice::Required => MistralToolChoice::String(MistralToolChoiceString::Any),
            ToolChoice::None => MistralToolChoice::String(MistralToolChoiceString::None),
            ToolChoice::Specific(tool_name) => {
                MistralToolChoice::Specific(MistralSpecificToolChoice {
                    r#type: "function",
                    function: MistralSpecificToolFunction { name: tool_name },
                })
            }
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct MistralTool<'a> {
    r#type: OpenAIToolType,
    function: OpenAIFunction<'a>,
}

impl<'a> From<&'a FunctionToolConfig> for MistralTool<'a> {
    fn from(tool: &'a FunctionToolConfig) -> Self {
        MistralTool {
            r#type: OpenAIToolType::Function,
            function: OpenAIFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
        }
    }
}

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to Mistral format
pub(super) fn prepare_mistral_tools<'a>(
    request: &'a ModelInferenceRequest<'a>,
) -> Result<PreparedMistralToolsResult<'a>, Error> {
    match &request.tool_config {
        None => Ok((None, None, None)),
        Some(tool_config) => {
            if !tool_config.any_tools_available() {
                return Ok((None, None, None));
            }
            let tools = Some(
                tool_config
                    .strict_tools_available()?
                    .map(Into::into)
                    .collect(),
            );
            let parallel_tool_calls = tool_config.parallel_tool_calls;

            // Mistral does not support allowed_tools constraint, use regular tool_choice
            let tool_choice = Some((&tool_config.tool_choice).into());
            Ok((tools, tool_choice, parallel_tool_calls))
        }
    }
}

/// This struct defines the supported parameters for the Mistral inference API
/// See the [Mistral API documentation](https://docs.mistral.ai/api/#tag/chat)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, service_tier, stop, user,
/// or context_length_exceeded_behavior.
/// NOTE: Mistral does not support seed.
#[derive(Debug, Serialize)]
struct MistralRequest<'a> {
    messages: Vec<MistralRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    random_seed: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<MistralResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MistralTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<MistralToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_mode: Option<&'a str>,
}

fn apply_inference_params(
    _request: &mut MistralRequest,
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

impl<'a> MistralRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
        prompt_mode: Option<&'a str>,
    ) -> Result<MistralRequest<'a>, Error> {
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(MistralResponseFormat::JsonObject)
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_mistral_messages(
            request,
            OpenAIMessagesConfig {
                json_mode: Some(&request.json_mode),
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: request
                    .fetch_and_encode_input_files_before_inference,
            },
            prompt_mode.is_some(),
        )
        .await?;
        let (tools, tool_choice, _) = prepare_mistral_tools(request)?;

        let mut mistral_request = MistralRequest {
            messages,
            model,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            random_seed: request.seed,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
            stop: request.borrow_stop_sequences(),
            prompt_mode,
        };

        apply_inference_params(&mut mistral_request, &request.inference_params_v2);

        Ok(mistral_request)
    }
}

fn mistral_usage_to_tensorzero_usage(usage: MistralUsage) -> Usage {
    Usage {
        input_tokens: Some(usage.prompt_tokens),
        output_tokens: Some(usage.completion_tokens),
        cost: None,
    }
}

fn mistral_response_tool_call_to_tensorzero_tool_call(
    tool_call: MistralResponseToolCall,
) -> ToolCall {
    ToolCall {
        id: tool_call.id,
        name: tool_call.function.name,
        arguments: tool_call.function.arguments,
    }
}

fn mistral_finish_reason_to_tensorzero_finish_reason(reason: MistralFinishReason) -> FinishReason {
    match reason {
        MistralFinishReason::Stop => FinishReason::Stop,
        MistralFinishReason::Length => FinishReason::Length,
        MistralFinishReason::ModelLength => FinishReason::Length,
        MistralFinishReason::Error => FinishReason::Unknown,
        MistralFinishReason::ToolCalls => FinishReason::ToolCall,
        MistralFinishReason::Unknown => FinishReason::Unknown,
    }
}

struct MistralResponseWithMetadata<'a> {
    response: MistralResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    model_inference_id: Uuid,
}

impl<'a> TryFrom<MistralResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: MistralResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let MistralResponseWithMetadata {
            mut response,
            raw_response,
            latency,
            raw_request,
            generic_request,
            model_inference_id,
        } = value;
        if response.choices.len() != 1 {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                api_type: ApiType::ChatCompletions,
                raw_request: None,
                raw_response: Some(raw_response.clone()),
            }));
        }
        let MistralResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                api_type: ApiType::ChatCompletions,
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
            }))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        match message.content {
            Some(MistralContent::String(text)) if !text.is_empty() => {
                content.push(text.into());
            }
            Some(MistralContent::Chunks(chunks)) => {
                for chunk in chunks {
                    match chunk {
                        MistralContentChunk::Thinking { thinking } => {
                            let text = extract_thinking_text(&thinking);
                            if !text.is_empty() {
                                content.push(ContentBlockOutput::Thought(Thought {
                                    text: Some(text),
                                    signature: None,
                                    summary: None,
                                    provider_type: Some(PROVIDER_TYPE.to_string()),
                                    extra_data: None,
                                }));
                            }
                        }
                        MistralContentChunk::Text { text } if !text.is_empty() => {
                            content.push(text.into());
                        }
                        MistralContentChunk::Text { .. } => {}
                    }
                }
            }
            Some(MistralContent::String(_)) => {}
            None => {}
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(
                    mistral_response_tool_call_to_tensorzero_tool_call(tool_call),
                ));
            }
        }
        let raw_usage = mistral_usage_from_raw_response(&raw_response).map(|usage| {
            raw_usage_entries_from_value(
                model_inference_id,
                PROVIDER_TYPE,
                ApiType::ChatCompletions,
                usage,
            )
        });
        let usage = mistral_usage_to_tensorzero_usage(response.usage);
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response: raw_response.clone(),
                usage,
                raw_usage,
                relay_raw_response: None,
                provider_latency: latency,
                finish_reason: Some(mistral_finish_reason_to_tensorzero_finish_reason(
                    finish_reason,
                )),
                id: model_inference_id,
            },
        ))
    }
}

/// Maps a Mistral chunk to a TensorZero chunk for streaming inferences
fn mistral_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: MistralChatChunk,
    latency: Duration,
    last_tool_name: &mut Option<String>,
    model_inference_id: Uuid,
    provider_type: &str,
) -> Result<ProviderInferenceResponseChunk, Error> {
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            api_type: ApiType::ChatCompletions,
            raw_request: None,
            raw_response: Some(raw_message.clone()),
        }
        .into());
    }
    let raw_usage = mistral_usage_from_raw_response(&raw_message).map(|usage| {
        raw_usage_entries_from_value(
            model_inference_id,
            provider_type,
            ApiType::ChatCompletions,
            usage,
        )
    });
    let usage = chunk.usage.map(mistral_usage_to_tensorzero_usage);
    let mut content = vec![];
    let mut finish_reason = None;
    if let Some(choice) = chunk.choices.pop() {
        if let Some(choice_finish_reason) = choice.finish_reason {
            finish_reason = Some(mistral_finish_reason_to_tensorzero_finish_reason(
                choice_finish_reason,
            ));
        }
        match choice.delta.content {
            Some(MistralContent::String(text)) if !text.is_empty() => {
                content.push(ContentBlockChunk::Text(TextChunk {
                    text,
                    id: "0".to_string(),
                }));
            }
            Some(MistralContent::Chunks(chunks)) => {
                for chunk in chunks {
                    match chunk {
                        MistralContentChunk::Thinking { thinking } => {
                            let text = extract_thinking_text(&thinking);
                            if !text.is_empty() {
                                content.push(ContentBlockChunk::Thought(ThoughtChunk {
                                    text: Some(text),
                                    signature: None,
                                    id: "thinking".to_string(),
                                    summary_id: None,
                                    summary_text: None,
                                    provider_type: Some(PROVIDER_TYPE.to_string()),
                                    extra_data: None,
                                }));
                            }
                        }
                        MistralContentChunk::Text { text } if !text.is_empty() => {
                            content.push(ContentBlockChunk::Text(TextChunk {
                                text,
                                id: "0".to_string(),
                            }));
                        }
                        MistralContentChunk::Text { .. } => {}
                    }
                }
            }
            Some(MistralContent::String(_)) => {}
            None => {}
        }
        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: tool_call.id,
                    raw_name: check_new_tool_call_name(tool_call.function.name, last_tool_name),
                    raw_arguments: tool_call.function.arguments,
                }));
            }
        }
    }

    Ok(match raw_usage {
        Some(entries) => ProviderInferenceResponseChunk::new_with_raw_usage(
            content,
            usage,
            raw_message,
            latency,
            finish_reason,
            Some(entries),
        ),
        None => {
            ProviderInferenceResponseChunk::new(content, usage, raw_message, latency, finish_reason)
        }
    })
}

/// Extracts and concatenates text from Mistral thinking sub-chunks.
/// The `thinking` field in a Thinking content chunk is an array of `{type: "text", text: "..."}`.
fn extract_thinking_text(sub_chunks: &[MistralThinkingSubChunk]) -> String {
    sub_chunks
        .iter()
        .map(|sub| match sub {
            MistralThinkingSubChunk::Text { text } => text.as_str(),
        })
        .collect::<Vec<_>>()
        .join("")
}

fn mistral_usage_from_raw_response(raw_response: &str) -> Option<Value> {
    serde_json::from_str::<Value>(raw_response)
        .ok()
        .and_then(|value| value.get("usage").filter(|v| !v.is_null()).cloned())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;

    use crate::inference::types::{FunctionType, RequestMessage, Role};
    use crate::providers::test_helpers::{QUERY_TOOL, WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::tool::{AllowedTools, ToolCallConfig};
    use tensorzero_types_providers::mistral::{
        MistralChatChunkChoice, MistralDelta, MistralResponseFunctionCall, MistralResponseMessage,
        MistralThinkingSubChunk,
    };
    #[tokio::test]
    async fn test_mistral_request_new() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let mistral_request =
            MistralRequest::new("mistral-small-latest", &request_with_tools, None)
                .await
                .unwrap();

        assert_eq!(mistral_request.model, "mistral-small-latest");
        assert_eq!(mistral_request.messages.len(), 1);
        assert_eq!(mistral_request.temperature, Some(0.5));
        assert_eq!(mistral_request.max_tokens, Some(100));
        assert!(!mistral_request.stream);
        assert_eq!(
            mistral_request.response_format,
            Some(MistralResponseFormat::JsonObject)
        );
        assert!(mistral_request.tools.is_some());
        let tools = mistral_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            mistral_request.tool_choice,
            Some(MistralToolChoice::Specific(MistralSpecificToolChoice {
                r#type: "function",
                function: MistralSpecificToolFunction {
                    name: "get_temperature"
                },
            }))
        );
    }

    #[test]
    fn test_prepare_mistral_tools_with_allowed_tools() {
        use crate::tool::{AllowedTools, AllowedToolsChoice};

        // Test with allowed_tools specified - Mistral doesn't support allowed_tools constraint
        let tool_config = ToolCallConfig {
            static_tools_available: vec![WEATHER_TOOL.clone(), QUERY_TOOL.clone()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: Some(false),
            allowed_tools: AllowedTools {
                tools: vec![WEATHER_TOOL.name().to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify only allowed tools are returned (strict_tools_available respects allowed_tools)
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());

        // Verify tool_choice
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::String(MistralToolChoiceString::Auto)
        );

        // Verify parallel_tool_calls
        assert_eq!(parallel_tool_calls, Some(false));
    }

    #[test]
    fn test_prepare_mistral_tools_auto_mode() {
        // Test Auto mode with default allowed_tools
        let tool_config = ToolCallConfig {
            static_tools_available: vec![WEATHER_TOOL.clone(), QUERY_TOOL.clone()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(),
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify tools
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2);

        // Verify tool_choice is Auto
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::String(MistralToolChoiceString::Auto)
        );

        // Verify parallel_tool_calls is None (default behavior)
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_prepare_mistral_tools_required_mode() {
        use crate::tool::AllowedTools;

        let tool_config = ToolCallConfig {
            static_tools_available: vec![WEATHER_TOOL.clone()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(),
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify tools
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);

        // Verify tool_choice is Any (Required maps to Any)
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::String(MistralToolChoiceString::Any)
        );

        // Verify parallel_tool_calls is None
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_prepare_mistral_tools_none_mode() {
        let tool_config = ToolCallConfig {
            static_tools_available: vec![WEATHER_TOOL.clone()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(),
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify tools are still returned
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);

        // Verify tool_choice is None
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::String(MistralToolChoiceString::None)
        );

        // Verify parallel_tool_calls is None
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_prepare_mistral_tools_specific_mode() {
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify tools
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());

        // Verify tool_choice is Specific
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::Specific(MistralSpecificToolChoice {
                r#type: "function",
                function: MistralSpecificToolFunction {
                    name: "get_temperature"
                },
            })
        );

        // Verify parallel_tool_calls is None (WEATHER_TOOL_CONFIG doesn't set parallel_tool_calls)
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_try_from_mistral_response() {
        // Test case 1: Valid response with content
        let valid_response = MistralResponse {
            choices: vec![MistralResponseChoice {
                index: 0,
                message: MistralResponseMessage {
                    content: Some(MistralContent::String("Hello, world!".to_string())),
                    tool_calls: None,
                },
                finish_reason: MistralFinishReason::Stop,
            }],
            usage: MistralUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
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
            max_tokens: Some(100),
            seed: Some(69),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop: None,
            prompt_mode: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test_response".to_string();
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: valid_response,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
            model_inference_id: Uuid::now_v7(),
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec!["Hello, world!".to_string().into()]
        );
        assert_eq!(inference_response.usage.input_tokens, Some(10));
        assert_eq!(inference_response.usage.output_tokens, Some(20));
        assert_eq!(
            inference_response.provider_latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(100)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.raw_response, raw_response);
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(inference_response.system, None);
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );

        // Test case 2: Valid response with tool calls
        let valid_response_with_tools = MistralResponse {
            choices: vec![MistralResponseChoice {
                index: 0,
                message: MistralResponseMessage {
                    content: None,
                    tool_calls: Some(vec![MistralResponseToolCall {
                        id: "call1".to_string(),
                        function: MistralResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
                finish_reason: MistralFinishReason::ToolCalls,
            }],
            usage: MistralUsage {
                prompt_tokens: 15,
                completion_tokens: 25,
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }],
            system: Some("test_system".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop: None,
            prompt_mode: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: valid_response_with_tools,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(110),
            },
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
            model_inference_id: Uuid::now_v7(),
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec![ContentBlockOutput::ToolCall(ToolCall {
                id: "call1".to_string(),
                name: "test_function".to_string(),
                arguments: "{}".to_string(),
            })]
        );
        assert_eq!(inference_response.usage.input_tokens, Some(15));
        assert_eq!(inference_response.usage.output_tokens, Some(25));
        assert_eq!(
            inference_response.provider_latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(110)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.raw_response, raw_response);
        assert_eq!(inference_response.system, Some("test_system".to_string()));
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]
        );
        // Test case 3: Invalid response with no choices
        let invalid_response_no_choices = MistralResponse {
            choices: vec![],
            usage: MistralUsage {
                prompt_tokens: 5,
                completion_tokens: 0,
            },
        };
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop: None,
            prompt_mode: None,
        };
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: invalid_response_no_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(120),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
            model_inference_id: Uuid::now_v7(),
        });
        let error = result.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        // Test case 4: Invalid response with multiple choices
        let invalid_response_multiple_choices = MistralResponse {
            choices: vec![
                MistralResponseChoice {
                    index: 0,
                    message: MistralResponseMessage {
                        content: Some(MistralContent::String("Choice 1".to_string())),
                        tool_calls: None,
                    },
                    finish_reason: MistralFinishReason::Stop,
                },
                MistralResponseChoice {
                    index: 1,
                    message: MistralResponseMessage {
                        content: Some(MistralContent::String("Choice 2".to_string())),
                        tool_calls: None,
                    },
                    finish_reason: MistralFinishReason::Stop,
                },
            ],
            usage: MistralUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
            },
        };
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop: None,
            prompt_mode: None,
        };
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: invalid_response_multiple_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(130),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
            model_inference_id: Uuid::now_v7(),
        });
        let error = result.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
    }

    #[test]
    fn test_handle_mistral_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_mistral_error(
            StatusCode::UNAUTHORIZED,
            "Unauthorized access",
            ApiType::ChatCompletions,
        );
        let error = unauthorized.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
            ..
        } = details
        {
            assert_eq!(*message, "Unauthorized access");
            assert_eq!(*status_code, Some(StatusCode::UNAUTHORIZED));
            assert_eq!(*provider, PROVIDER_TYPE.to_string());
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, None);
        }

        // Test forbidden error
        let forbidden = handle_mistral_error(
            StatusCode::FORBIDDEN,
            "Forbidden access",
            ApiType::ChatCompletions,
        );
        let error = forbidden.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
            ..
        } = details
        {
            assert_eq!(*message, "Forbidden access");
            assert_eq!(*status_code, Some(StatusCode::FORBIDDEN));
            assert_eq!(*provider, PROVIDER_TYPE.to_string());
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, None);
        }

        // Test rate limit error
        let rate_limit = handle_mistral_error(
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded",
            ApiType::ChatCompletions,
        );
        let error = rate_limit.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
            ..
        } = details
        {
            assert_eq!(*message, "Rate limit exceeded");
            assert_eq!(*status_code, Some(StatusCode::TOO_MANY_REQUESTS));
            assert_eq!(*provider, PROVIDER_TYPE.to_string());
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, None);
        }

        // Test server error
        let server_error = handle_mistral_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Server error",
            ApiType::ChatCompletions,
        );
        let error = server_error.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        if let ErrorDetails::InferenceServer {
            message,
            provider_type: provider,
            raw_request,
            raw_response,
            ..
        } = details
        {
            assert_eq!(*message, "Server error");
            assert_eq!(*provider, PROVIDER_TYPE.to_string());
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, None);
        }
    }

    #[test]
    fn test_mistral_api_base() {
        assert_eq!(MISTRAL_API_BASE.as_str(), "https://api.mistral.ai/v1/");
    }

    #[test]
    fn test_credential_to_mistral_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: MistralCredentials = MistralCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MistralCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = MistralCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MistralCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = MistralCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MistralCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = MistralCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_mistral_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = MistralRequest {
            messages: vec![],
            model: "test-model",
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            random_seed: None,
            stream: false,
            response_format: None,
            tools: None,
            tool_choice: None,
            stop: None,
            prompt_mode: None,
        };

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort warns
        assert!(logs_contain(
            "Mistral does not support the inference parameter `reasoning_effort`"
        ));

        // Test that thinking_budget_tokens warns
        assert!(logs_contain(
            "Mistral does not support the inference parameter `thinking_budget_tokens`"
        ));

        // Test that verbosity warns
        assert!(logs_contain(
            "Mistral does not support the inference parameter `verbosity`"
        ));
    }

    #[test]
    fn test_try_from_mistral_response_with_reasoning() {
        let response_with_reasoning = MistralResponse {
            choices: vec![MistralResponseChoice {
                index: 0,
                message: MistralResponseMessage {
                    content: Some(MistralContent::Chunks(vec![
                        MistralContentChunk::Thinking {
                            thinking: vec![MistralThinkingSubChunk::Text {
                                text: "Let me think about this...".to_string(),
                            }],
                        },
                        MistralContentChunk::Text {
                            text: "The answer is 42.".to_string(),
                        },
                    ])),
                    tool_calls: None,
                },
                finish_reason: MistralFinishReason::Stop,
            }],
            usage: MistralUsage {
                prompt_tokens: 10,
                completion_tokens: 30,
            },
        };

        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What is the meaning of life?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: Some(800),
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let request_body = MistralRequest {
            messages: vec![],
            model: "magistral-small-latest",
            temperature: None,
            max_tokens: Some(800),
            random_seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            response_format: None,
            tools: None,
            tool_choice: None,
            stop: None,
            prompt_mode: Some("reasoning"),
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();

        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: response_with_reasoning,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(200),
            },
            raw_request,
            generic_request: &generic_request,
            raw_response: "test_response".to_string(),
            model_inference_id: Uuid::now_v7(),
        });

        let response = result.expect("should parse reasoning response");
        assert_eq!(
            response.output.len(),
            2,
            "should have a Thought block and a Text block"
        );
        assert!(
            matches!(&response.output[0], ContentBlockOutput::Thought(thought) if thought.text.as_deref() == Some("Let me think about this...")),
            "first block should be a Thought"
        );
        assert!(
            matches!(&response.output[1], ContentBlockOutput::Text(text) if text.text == "The answer is 42."),
            "second block should be Text"
        );
        assert_eq!(response.usage.input_tokens, Some(10));
        assert_eq!(response.usage.output_tokens, Some(30));
    }

    #[test]
    fn test_mistral_streaming_chunk_with_reasoning() {
        let chunk = MistralChatChunk {
            choices: vec![MistralChatChunkChoice {
                delta: MistralDelta {
                    content: Some(MistralContent::Chunks(vec![
                        MistralContentChunk::Thinking {
                            thinking: vec![MistralThinkingSubChunk::Text {
                                text: "reasoning step".to_string(),
                            }],
                        },
                    ])),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = mistral_to_tensorzero_chunk(
            "test_raw".to_string(),
            chunk,
            Duration::from_millis(50),
            &mut None,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        );

        let response_chunk = result.expect("should parse streaming reasoning chunk");
        assert_eq!(
            response_chunk.content.len(),
            1,
            "should have one content block"
        );
        assert!(
            matches!(&response_chunk.content[0], ContentBlockChunk::Thought(thought) if thought.text.as_deref() == Some("reasoning step")),
            "content block should be a ThoughtChunk"
        );

        // Also test a chunk with a text block after reasoning
        let text_chunk = MistralChatChunk {
            choices: vec![MistralChatChunkChoice {
                delta: MistralDelta {
                    content: Some(MistralContent::Chunks(vec![MistralContentChunk::Text {
                        text: "hello".to_string(),
                    }])),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = mistral_to_tensorzero_chunk(
            "test_raw".to_string(),
            text_chunk,
            Duration::from_millis(60),
            &mut None,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        );

        let response_chunk = result.expect("should parse streaming text chunk");
        assert_eq!(
            response_chunk.content.len(),
            1,
            "should have one content block"
        );
        assert!(
            matches!(&response_chunk.content[0], ContentBlockChunk::Text(text) if text.text == "hello"),
            "content block should be a TextChunk"
        );
    }

    #[test]
    fn test_mistral_assistant_message_with_thought_blocks() {
        use crate::inference::types::{ContentBlock, Text, Thought};

        let message = RequestMessage {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thought(Thought {
                    text: Some("Let me reason...".to_string()),
                    signature: None,
                    summary: None,
                    provider_type: Some("mistral".to_string()),
                    extra_data: None,
                }),
                ContentBlock::Text(Text {
                    text: "Here's the answer.".to_string(),
                }),
            ],
        };

        let result = tensorzero_to_mistral_assistant_message(&message)
            .expect("should convert assistant message");
        let msg = result.expect("should produce a message");

        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.len(), 2, "should have thinking and text chunks");
        assert_eq!(
            msg.content[0],
            MistralRequestContentChunk::Thinking {
                thinking: vec![MistralRequestThinkingSubChunk::Text {
                    text: Cow::Borrowed("Let me reason...")
                }]
            },
            "first chunk should be thinking"
        );
        assert_eq!(
            msg.content[1],
            MistralRequestContentChunk::Text {
                text: Cow::Borrowed("Here's the answer.")
            },
            "second chunk should be text"
        );
        assert!(msg.tool_calls.is_none());

        // Test serialization: single text-only content becomes a string
        let text_only_msg = MistralAssistantRequestMessage {
            role: "assistant",
            content: vec![MistralRequestContentChunk::Text {
                text: Cow::Borrowed("Just text"),
            }],
            tool_calls: None,
        };
        let serialized = serde_json::to_string(&text_only_msg).unwrap();
        assert!(
            serialized.contains(r#""content":"Just text""#),
            "single text content should serialize as a plain string, got: {serialized}"
        );

        // Test serialization: mixed content becomes an array
        let serialized = serde_json::to_string(&msg).unwrap();
        assert!(
            serialized.contains(r#""content":[{"type":"thinking"#),
            "mixed content should serialize as an array, got: {serialized}"
        );

        // Test serialization: empty content with tool calls serializes content as null
        let tool_only_msg = MistralAssistantRequestMessage {
            role: "assistant",
            content: vec![],
            tool_calls: Some(vec![OpenAIRequestToolCall {
                id: Cow::Borrowed("call_123"),
                r#type: OpenAIToolType::Function,
                function: OpenAIRequestFunctionCall {
                    name: Cow::Borrowed("get_weather"),
                    arguments: Cow::Borrowed(r#"{"city":"Tokyo"}"#),
                },
            }]),
        };
        let serialized = serde_json::to_string(&tool_only_msg).unwrap();
        assert!(
            serialized.contains(r#""content":null"#),
            "empty content should serialize as null, got: {serialized}"
        );
    }

    #[test]
    fn test_mistral_request_with_prompt_mode() {
        let request = MistralRequest {
            messages: vec![],
            model: "magistral-small-latest",
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(800),
            random_seed: None,
            stream: false,
            response_format: None,
            tools: None,
            tool_choice: None,
            stop: None,
            prompt_mode: Some("reasoning"),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(
            serialized.contains(r#""prompt_mode":"reasoning""#),
            "serialized request should contain prompt_mode field, got: {serialized}"
        );

        // Test that None prompt_mode is omitted
        let request_no_reasoning = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            random_seed: None,
            stream: false,
            response_format: None,
            tools: None,
            tool_choice: None,
            stop: None,
            prompt_mode: None,
        };

        let serialized = serde_json::to_string(&request_no_reasoning).unwrap();
        assert!(
            !serialized.contains("prompt_mode"),
            "serialized request should not contain prompt_mode when None, got: {serialized}"
        );
    }

    #[test]
    fn test_mistral_content_deserialization() {
        // Test string content (non-reasoning)
        let json = r#"{"content": "Hello, world!", "tool_calls": null}"#;
        let msg: MistralResponseMessage = serde_json::from_str(json).unwrap();
        assert_eq!(
            msg.content,
            Some(MistralContent::String("Hello, world!".to_string())),
            "string content should deserialize as MistralContent::String"
        );

        // Test chunked content (reasoning)
        let json = r#"{"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "Let me think..."}]}, {"type": "text", "text": "The answer"}]}"#;
        let msg: MistralResponseMessage = serde_json::from_str(json).unwrap();
        assert_eq!(
            msg.content,
            Some(MistralContent::Chunks(vec![
                MistralContentChunk::Thinking {
                    thinking: vec![MistralThinkingSubChunk::Text {
                        text: "Let me think...".to_string()
                    }]
                },
                MistralContentChunk::Text {
                    text: "The answer".to_string()
                },
            ])),
            "chunked content should deserialize as MistralContent::Chunks"
        );

        // Test null content
        let json = r#"{"content": null}"#;
        let msg: MistralResponseMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, None, "null content should deserialize as None");
    }
}
