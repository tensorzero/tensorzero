use futures::future::try_join_all;
use futures::{Stream, StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::de::IntoDeserializer;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::time::Duration;
use tokio::time::Instant;

use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{
    warn_discarded_thought_block, DelayedError, DisplayOrDebugGateway, Error, ErrorDetails,
};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2, ServiceTier,
};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse,
    resolved_input::{FileUrl, LazyFile},
    ContentBlock, ContentBlockChunk, ContentBlockOutput, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, ObjectStorageFile, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, RequestMessage, Role, Text,
    TextChunk, Unknown, Usage,
};
use crate::inference::types::{
    FinishReason, ProviderInferenceResponseArgs, ProviderInferenceResponseStreamInner,
};
use crate::inference::{InferenceProvider, TensorZeroEventError};
use crate::model::{Credential, ModelProvider};
use crate::tool::{FunctionToolConfig, ToolCall, ToolCallChunk, ToolChoice};

use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::helpers::{
    convert_stream_error, inject_extra_request_data_and_send,
    inject_extra_request_data_and_send_eventsource, warn_cannot_forward_url_if_missing_mime_type,
};

use super::chat_completions::{
    ChatCompletionAllowedToolsMode, ChatCompletionTool, ChatCompletionToolChoice,
    ChatCompletionToolChoiceString,
};
use super::openai::{
    AllowedToolsChoice as OpenAIAllowedToolsChoice,
    AllowedToolsConstraint as OpenAIAllowedToolsConstraint, AllowedToolsMode,
    SpecificToolFunction as OpenAISpecificToolFunction, ToolReference,
};

const PROVIDER_NAME: &str = "Groq";
pub const PROVIDER_TYPE: &str = "groq";

type PreparedToolsResult<'a> = (
    Option<Vec<GroqTool<'a>>>,
    Option<GroqToolChoice<'a>>,
    Option<bool>,
);

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GroqProvider {
    model_name: String,
    #[serde(skip)]
    credentials: GroqCredentials,
}

impl GroqProvider {
    pub fn new(model_name: String, credentials: GroqCredentials) -> Self {
        GroqProvider {
            model_name,
            credentials,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum GroqCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<GroqCredentials>,
        fallback: Box<GroqCredentials>,
    },
}

impl TryFrom<Credential> for GroqCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(GroqCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(GroqCredentials::Dynamic(key_name)),
            Credential::None => Ok(GroqCredentials::None),
            Credential::Missing => Ok(GroqCredentials::None),
            Credential::WithFallback { default, fallback } => Ok(GroqCredentials::WithFallback {
                default: Box::new((*default).try_into()?),
                fallback: Box::new((*fallback).try_into()?),
            }),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Groq provider".to_string(),
            })),
        }
    }
}

impl GroqCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, DelayedError> {
        match self {
            GroqCredentials::Static(api_key) => Ok(Some(api_key)),
            GroqCredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                }))
                .transpose()
            }
            GroqCredentials::WithFallback { default, fallback } => {
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
            GroqCredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for GroqProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_url = "https://api.groq.com/openai/v1/chat/completions".to_string();
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();

        let request_body =
            serde_json::to_value(GroqRequest::new(&self.model_name, request.request).await?)
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing Groq request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;

        let mut request_builder = http_client.post(request_url);

        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.request.extra_body,
            &request.request.extra_headers,
            model_provider,
            request.model_name,
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
            Ok(GroqResponseWithMetadata {
                response,
                raw_response,
                latency,
                raw_request: raw_request.clone(),
                generic_request: request.request,
            }
            .try_into()?)
        } else {
            Err(handle_groq_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        raw_request: Some(raw_request),
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?,
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
        let request_body = serde_json::to_value(GroqRequest::new(&self.model_name, request).await?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Groq request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let request_url = "https://api.groq.com/openai/v1/chat/completions".to_string();
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let mut request_builder = http_client.post(request_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
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

        let stream = stream_groq(
            PROVIDER_TYPE.to_string(),
            event_source.map_err(TensorZeroEventError::EventSource),
            start_time,
            &raw_request,
        )
        .peekable();
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

pub fn stream_groq(
    provider_type: String,
    event_source: impl Stream<Item = Result<Event, TensorZeroEventError>> + Send + 'static,
    start_time: Instant,
    raw_request: &str,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    let mut tool_call_ids = Vec::new();
    Box::pin(async_stream::stream! {
        futures::pin_mut!(event_source);
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    match e {
                        TensorZeroEventError::TensorZero(e) => {
                            yield Err(e);
                        }
                        TensorZeroEventError::EventSource(e) => {
                            yield Err(convert_stream_error(raw_request.clone(), provider_type.clone(), e).await);
                        }
                    }
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<GroqChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {e}",
                                    ),
                                raw_request: Some(raw_request.clone()),
                                raw_response: Some(message.data.clone()),
                                provider_type: provider_type.clone(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            groq_to_tensorzero_chunk(d, latency, &mut tool_call_ids)
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

pub(super) fn handle_groq_error(
    response_code: StatusCode,
    response_body: &str,
    provider_type: &str,
) -> Error {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => ErrorDetails::InferenceClient {
            status_code: Some(response_code),
            message: response_body.to_string(),
            raw_request: None,
            raw_response: Some(response_body.to_string()),
            provider_type: provider_type.to_string(),
        }
        .into(),
        _ => ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            raw_request: None,
            raw_response: Some(response_body.to_string()),
            provider_type: provider_type.to_string(),
        }
        .into(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct GroqSystemRequestMessage<'a> {
    pub content: Cow<'a, str>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct GroqUserRequestMessage<'a> {
    #[serde(serialize_with = "serialize_text_content_vec")]
    pub(super) content: Vec<GroqContentBlock<'a>>,
}

fn serialize_text_content_vec<S>(
    content: &Vec<GroqContentBlock<'_>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // If we have a single text block, serialize it as a string
    // to stay compatible with older providers which may not support content blocks
    if let [GroqContentBlock::Text { text }] = &content.as_slice() {
        text.serialize(serializer)
    } else {
        content.serialize(serializer)
    }
}

fn serialize_optional_text_content_vec<S>(
    content: &Option<Vec<GroqContentBlock<'_>>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match content {
        Some(vec) => serialize_text_content_vec(vec, serializer),
        None => serializer.serialize_none(),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GroqContentBlock<'a> {
    Text { text: Cow<'a, str> },
    ImageUrl { image_url: GroqImageUrl },
    Unknown { data: Cow<'a, Value> },
}

impl Serialize for GroqContentBlock<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Helper<'a> {
            Text { text: &'a str },
            ImageUrl { image_url: &'a GroqImageUrl },
        }
        match self {
            GroqContentBlock::Text { text } => Helper::Text { text }.serialize(serializer),
            GroqContentBlock::ImageUrl { image_url } => {
                Helper::ImageUrl { image_url }.serialize(serializer)
            }
            GroqContentBlock::Unknown { data } => data.serialize(serializer),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct GroqImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GroqRequestFunctionCall<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct GroqRequestToolCall<'a> {
    pub id: &'a str,
    pub r#type: GroqToolType,
    pub function: GroqRequestFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for GroqRequestToolCall<'a> {
    fn from(tool_call: &'a ToolCall) -> Self {
        GroqRequestToolCall {
            id: &tool_call.id,
            r#type: GroqToolType::Function,
            function: GroqRequestFunctionCall {
                name: &tool_call.name,
                arguments: &tool_call.arguments,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct GroqAssistantRequestMessage<'a> {
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_text_content_vec"
    )]
    pub content: Option<Vec<GroqContentBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<GroqRequestToolCall<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct GroqToolRequestMessage<'a> {
    pub content: &'a str,
    pub tool_call_id: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub(super) enum GroqRequestMessage<'a> {
    System(GroqSystemRequestMessage<'a>),
    User(GroqUserRequestMessage<'a>),
    Assistant(GroqAssistantRequestMessage<'a>),
    Tool(GroqToolRequestMessage<'a>),
}

impl GroqRequestMessage<'_> {
    pub fn content_contains_case_insensitive(&self, value: &str) -> bool {
        match self {
            GroqRequestMessage::System(msg) => msg.content.to_lowercase().contains(value),
            GroqRequestMessage::User(msg) => msg.content.iter().any(|c| match c {
                GroqContentBlock::Text { text } => text.to_lowercase().contains(value),
                GroqContentBlock::ImageUrl { .. } => false,
                // Don't inspect the contents of 'unknown' blocks
                GroqContentBlock::Unknown { data: _ } => false,
            }),
            GroqRequestMessage::Assistant(msg) => {
                if let Some(content) = &msg.content {
                    content.iter().any(|c| match c {
                        GroqContentBlock::Text { text } => text.to_lowercase().contains(value),
                        GroqContentBlock::ImageUrl { .. } => false,
                        // Don't inspect the contents of `unknown` blocks
                        GroqContentBlock::Unknown { data: _ } => false,
                    })
                } else {
                    false
                }
            }
            GroqRequestMessage::Tool(msg) => msg.content.to_lowercase().contains(value),
        }
    }
}

pub(super) async fn prepare_groq_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Result<Vec<GroqRequestMessage<'a>>, Error> {
    let fetch_and_encode = request.fetch_and_encode_input_files_before_inference;
    let mut messages: Vec<_> = try_join_all(
        request
            .messages
            .iter()
            .map(|msg| tensorzero_to_groq_messages(msg, fetch_and_encode)),
    )
    .await?
    .into_iter()
    .flatten()
    .collect();
    if let Some(system_msg) =
        tensorzero_to_groq_system_message(request.system.as_deref(), request.json_mode, &messages)
    {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to Groq format
/// NOTE: parallel tool calls are unreliable, and specific tool choice doesn't work
pub(super) fn prepare_groq_tools<'a>(
    request: &'a ModelInferenceRequest,
) -> Result<PreparedToolsResult<'a>, Error> {
    let (tools, tool_choice, parallel_tool_calls) = prepare_chat_completion_tools(request, true)?;

    // Convert from ChatCompletionTool to GroqTool
    let groq_tools = tools.map(|t| t.into_iter().map(GroqTool::from).collect());

    // Convert from ChatCompletionToolChoice to GroqToolChoice
    let groq_tool_choice = tool_choice.map(GroqToolChoice::from);

    Ok((groq_tools, groq_tool_choice, parallel_tool_calls))
}

/// If ModelInferenceRequestJsonMode::On and the system message or instructions does not contain "JSON"
/// the request will return an error.
/// So, we need to format the instructions to include "Respond using JSON." if it doesn't already.
pub(super) fn tensorzero_to_groq_system_message<'a>(
    system: Option<&'a str>,
    json_mode: ModelInferenceRequestJsonMode,
    messages: &[GroqRequestMessage<'a>],
) -> Option<GroqRequestMessage<'a>> {
    match system {
        Some(system) => {
            match json_mode {
                ModelInferenceRequestJsonMode::On => {
                    if messages
                        .iter()
                        .any(|msg| msg.content_contains_case_insensitive("json"))
                        || system.to_lowercase().contains("json")
                    {
                        GroqRequestMessage::System(GroqSystemRequestMessage {
                            content: Cow::Borrowed(system),
                        })
                    } else {
                        let formatted_instructions = format!("Respond using JSON.\n\n{system}");
                        GroqRequestMessage::System(GroqSystemRequestMessage {
                            content: Cow::Owned(formatted_instructions),
                        })
                    }
                }

                // If JSON mode is either off or strict, we don't need to do anything special
                _ => GroqRequestMessage::System(GroqSystemRequestMessage {
                    content: Cow::Borrowed(system),
                }),
            }
            .into()
        }
        None => match json_mode {
            ModelInferenceRequestJsonMode::On => {
                Some(GroqRequestMessage::System(GroqSystemRequestMessage {
                    content: Cow::Owned("Respond using JSON.".to_string()),
                }))
            }
            _ => None,
        },
    }
}

pub(super) async fn tensorzero_to_groq_messages(
    message: &RequestMessage,
    fetch_and_encode_input_files_before_inference: bool,
) -> Result<Vec<GroqRequestMessage<'_>>, Error> {
    match message.role {
        Role::User => {
            tensorzero_to_groq_user_messages(
                &message.content,
                fetch_and_encode_input_files_before_inference,
            )
            .await
        }
        Role::Assistant => {
            tensorzero_to_groq_assistant_messages(
                &message.content,
                fetch_and_encode_input_files_before_inference,
            )
            .await
        }
    }
}

async fn tensorzero_to_groq_user_messages(
    content_blocks: &[ContentBlock],
    fetch_and_encode_input_files_before_inference: bool,
) -> Result<Vec<GroqRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the user content blocks.

    let mut messages = Vec::new();
    let mut user_content_blocks = Vec::new();

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                user_content_blocks.push(GroqContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool calls are not supported in user messages".to_string(),
                }));
            }
            ContentBlock::ToolResult(tool_result) => {
                messages.push(GroqRequestMessage::Tool(GroqToolRequestMessage {
                    content: &tool_result.result,
                    tool_call_id: &tool_result.id,
                }));
            }
            ContentBlock::File(file) => match file.as_ref() {
                LazyFile::Url {
                    file_url:
                        FileUrl {
                            mime_type,
                            url,
                            detail,
                        },
                    future: _,
                } if !fetch_and_encode_input_files_before_inference
                    && matches!(
                        mime_type.as_ref().map(mime::MediaType::type_),
                        Some(mime::IMAGE) | None
                    ) =>
                {
                    if detail.is_some() {
                        tracing::warn!(
                                "The image detail parameter is not supported by Groq. The `detail` field will be ignored."
                            );
                    }
                    warn_cannot_forward_url_if_missing_mime_type(
                        file,
                        fetch_and_encode_input_files_before_inference,
                        PROVIDER_TYPE,
                    );
                    user_content_blocks.push(GroqContentBlock::ImageUrl {
                        image_url: GroqImageUrl {
                            url: url.to_string(),
                        },
                    });
                }
                _ => {
                    let resolved_file = file.resolve().await?;
                    let ObjectStorageFile { file, data } = &*resolved_file;
                    if file.detail.is_some() {
                        tracing::warn!(
                                "The image detail parameter is not supported by Groq. The `detail` field will be ignored."
                            );
                    }
                    user_content_blocks.push(GroqContentBlock::ImageUrl {
                        image_url: GroqImageUrl {
                            url: format!("data:{};base64,{}", file.mime_type, data),
                        },
                    });
                }
            },
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(PROVIDER_TYPE, thought);
            }
            ContentBlock::Unknown(Unknown { data, .. }) => {
                user_content_blocks.push(GroqContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
        };
    }

    // If there are any user content blocks, combine them into a single user message.
    if !user_content_blocks.is_empty() {
        messages.push(GroqRequestMessage::User(GroqUserRequestMessage {
            content: user_content_blocks,
        }));
    }

    Ok(messages)
}

async fn tensorzero_to_groq_assistant_messages(
    content_blocks: &[ContentBlock],
    fetch_and_encode_input_files_before_inference: bool,
) -> Result<Vec<GroqRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the assistant content blocks.
    let mut assistant_content_blocks = Vec::new();
    let mut assistant_tool_calls = Vec::new();

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                assistant_content_blocks.push(GroqContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(tool_call) => {
                let tool_call = GroqRequestToolCall {
                    id: &tool_call.id,
                    r#type: GroqToolType::Function,
                    function: GroqRequestFunctionCall {
                        name: &tool_call.name,
                        arguments: &tool_call.arguments,
                    },
                };

                assistant_tool_calls.push(tool_call);
            }
            ContentBlock::ToolResult(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool results are not supported in assistant messages".to_string(),
                }));
            }
            ContentBlock::File(file) => match file.as_ref() {
                LazyFile::Url {
                    file_url:
                        FileUrl {
                            mime_type,
                            url,
                            detail,
                        },
                    future: _,
                } if !fetch_and_encode_input_files_before_inference
                    && matches!(
                        mime_type.as_ref().map(mime::MediaType::type_),
                        Some(mime::IMAGE) | None
                    ) =>
                {
                    if detail.is_some() {
                        tracing::warn!(
                                "The image detail parameter is not supported by Groq. The `detail` field will be ignored."
                            );
                    }
                    warn_cannot_forward_url_if_missing_mime_type(
                        file,
                        fetch_and_encode_input_files_before_inference,
                        PROVIDER_TYPE,
                    );
                    assistant_content_blocks.push(GroqContentBlock::ImageUrl {
                        image_url: GroqImageUrl {
                            url: url.to_string(),
                        },
                    });
                }
                _ => {
                    let resolved_file = file.resolve().await?;
                    let ObjectStorageFile { file, data } = &*resolved_file;
                    if file.detail.is_some() {
                        tracing::warn!(
                                "The image detail parameter is not supported by Groq. The `detail` field will be ignored."
                            );
                    }
                    assistant_content_blocks.push(GroqContentBlock::ImageUrl {
                        image_url: GroqImageUrl {
                            url: format!("data:{};base64,{}", file.mime_type, data),
                        },
                    });
                }
            },
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(PROVIDER_TYPE, thought);
            }
            ContentBlock::Unknown(Unknown { data, .. }) => {
                assistant_content_blocks.push(GroqContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
        }
    }

    let content = match assistant_content_blocks.len() {
        0 => None,
        _ => Some(assistant_content_blocks),
    };

    let tool_calls = match assistant_tool_calls.len() {
        0 => None,
        _ => Some(assistant_tool_calls),
    };

    let message = GroqRequestMessage::Assistant(GroqAssistantRequestMessage {
        content,
        tool_calls,
    });

    Ok(vec![message])
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum GroqResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: Value,
    },
}

impl GroqResponseFormat {
    fn new(json_mode: ModelInferenceRequestJsonMode, output_schema: Option<&Value>) -> Self {
        match json_mode {
            ModelInferenceRequestJsonMode::On => GroqResponseFormat::JsonObject,
            ModelInferenceRequestJsonMode::Off => GroqResponseFormat::Text,
            ModelInferenceRequestJsonMode::Strict => match output_schema {
                Some(schema) => {
                    let json_schema = json!({"name": "response", "strict": true, "schema": schema});
                    GroqResponseFormat::JsonSchema { json_schema }
                }
                None => GroqResponseFormat::JsonObject,
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GroqToolType {
    Function,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct GroqFunction<'a> {
    pub(super) name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<&'a str>,
    pub parameters: &'a Value,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct GroqTool<'a> {
    pub(super) r#type: GroqToolType,
    pub(super) function: GroqFunction<'a>,
    pub(super) strict: bool,
}

impl<'a> From<&'a FunctionToolConfig> for GroqTool<'a> {
    fn from(tool: &'a FunctionToolConfig) -> Self {
        GroqTool {
            r#type: GroqToolType::Function,
            function: GroqFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
            strict: tool.strict(),
        }
    }
}

impl<'a> From<ChatCompletionTool<'a>> for GroqTool<'a> {
    fn from(tool: ChatCompletionTool<'a>) -> Self {
        GroqTool {
            r#type: GroqToolType::Function,
            function: GroqFunction {
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters,
            },
            strict: tool.strict,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(super) enum GroqToolChoice<'a> {
    String(GroqToolChoiceString),
    Specific(SpecificToolChoice<'a>),
    AllowedTools(OpenAIAllowedToolsChoice<'a>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum GroqToolChoiceString {
    None,
    Auto,
    Required,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SpecificToolChoice<'a> {
    pub(super) r#type: GroqToolType,
    pub(super) function: SpecificToolFunction<'a>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct SpecificToolFunction<'a> {
    pub(super) name: &'a str,
}

impl Default for GroqToolChoice<'_> {
    fn default() -> Self {
        GroqToolChoice::String(GroqToolChoiceString::None)
    }
}

impl<'a> From<&'a ToolChoice> for GroqToolChoice<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::None => GroqToolChoice::String(GroqToolChoiceString::None),
            ToolChoice::Auto => GroqToolChoice::String(GroqToolChoiceString::Auto),
            ToolChoice::Required => GroqToolChoice::String(GroqToolChoiceString::Required),
            ToolChoice::Specific(tool_name) => GroqToolChoice::Specific(SpecificToolChoice {
                r#type: GroqToolType::Function,
                function: SpecificToolFunction { name: tool_name },
            }),
        }
    }
}

impl<'a> From<ChatCompletionToolChoice<'a>> for GroqToolChoice<'a> {
    fn from(tool_choice: ChatCompletionToolChoice<'a>) -> Self {
        match tool_choice {
            ChatCompletionToolChoice::String(tc_string) => match tc_string {
                ChatCompletionToolChoiceString::None => {
                    GroqToolChoice::String(GroqToolChoiceString::None)
                }
                ChatCompletionToolChoiceString::Auto => {
                    GroqToolChoice::String(GroqToolChoiceString::Auto)
                }
                ChatCompletionToolChoiceString::Required => {
                    GroqToolChoice::String(GroqToolChoiceString::Required)
                }
            },
            ChatCompletionToolChoice::Specific(specific) => {
                GroqToolChoice::Specific(SpecificToolChoice {
                    r#type: GroqToolType::Function,
                    function: SpecificToolFunction {
                        name: specific.function.name,
                    },
                })
            }
            ChatCompletionToolChoice::AllowedTools(allowed_tools) => {
                // Convert from chat_completions ChatCompletionAllowedToolsChoice to OpenAI AllowedToolsChoice
                GroqToolChoice::AllowedTools(OpenAIAllowedToolsChoice {
                    r#type: allowed_tools.r#type,
                    allowed_tools: OpenAIAllowedToolsConstraint {
                        mode: match allowed_tools.allowed_tools.mode {
                            ChatCompletionAllowedToolsMode::Auto => AllowedToolsMode::Auto,
                            ChatCompletionAllowedToolsMode::Required => AllowedToolsMode::Required,
                        },
                        tools: allowed_tools
                            .allowed_tools
                            .tools
                            .into_iter()
                            .map(|tool_ref| ToolReference::Function {
                                function: OpenAISpecificToolFunction {
                                    name: tool_ref.function.name,
                                },
                            })
                            .collect(),
                    },
                })
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct StreamOptions {
    pub(super) include_usage: bool,
}

/// This struct defines the supported parameters for the Groq API
/// See the [Groq API documentation](https://platform.groq.com/docs/api-reference/chat/create)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Default, Serialize)]
struct GroqRequest<'a> {
    messages: Vec<GroqRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<GroqResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GroqTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<GroqToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<ServiceTier>,
}

fn apply_inference_params(
    request: &mut GroqRequest,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() {
        request.reasoning_effort = reasoning_effort.clone();
    }

    // Groq supports auto and flex, but not priority and default
    if let Some(tier) = service_tier {
        match tier {
            ServiceTier::Auto | ServiceTier::Flex => {
                request.service_tier = Some(tier.clone());
            }
            ServiceTier::Priority | ServiceTier::Default => {
                warn_inference_parameter_not_supported(
                    PROVIDER_NAME,
                    &format!("service_tier ({tier})"),
                    None,
                );
            }
        }
    }

    if thinking_budget_tokens.is_some() {
        warn_inference_parameter_not_supported(
            PROVIDER_NAME,
            "thinking_budget_tokens",
            Some("Tip: You might want to use `reasoning_effort` for this provider."),
        );
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
}

impl<'a> GroqRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<GroqRequest<'a>, Error> {
        let response_format = Some(GroqResponseFormat::new(
            request.json_mode,
            request.output_schema,
        ));
        let stream_options = if request.stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        };
        let mut messages = prepare_groq_messages(request).await?;

        let (tools, tool_choice, parallel_tool_calls) = prepare_groq_tools(request)?;

        if model.to_lowercase().starts_with("o1-mini") {
            if let Some(GroqRequestMessage::System(_)) = messages.first() {
                if let GroqRequestMessage::System(system_msg) = messages.remove(0) {
                    let user_msg = GroqRequestMessage::User(GroqUserRequestMessage {
                        content: vec![GroqContentBlock::Text {
                            text: system_msg.content,
                        }],
                    });
                    messages.insert(0, user_msg);
                }
            }
        }

        let mut groq_request = GroqRequest {
            messages,
            model,
            temperature: request.temperature,
            max_completion_tokens: request.max_tokens,
            seed: request.seed,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            stream: request.stream,
            stream_options,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
            // allowed_tools is now part of tool_choice (AllowedToolsChoice variant)
            stop: request.borrow_stop_sequences(),
            reasoning_effort: None,
            service_tier: None, // handled below
        };

        apply_inference_params(&mut groq_request, &request.inference_params_v2);

        Ok(groq_request)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct GroqUsage {
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
}

impl From<GroqUsage> for Usage {
    fn from(usage: GroqUsage) -> Self {
        Usage {
            input_tokens: Some(usage.prompt_tokens),
            output_tokens: Some(usage.completion_tokens),
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct GroqResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub(super) struct GroqResponseToolCall {
    id: String,
    r#type: GroqToolType,
    function: GroqResponseFunctionCall,
}

impl From<GroqResponseToolCall> for ToolCall {
    fn from(groq_tool_call: GroqResponseToolCall) -> Self {
        ToolCall {
            id: groq_tool_call.id,
            name: groq_tool_call.function.name,
            arguments: groq_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct GroqResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_calls: Option<Vec<GroqResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum GroqFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

impl From<GroqFinishReason> for FinishReason {
    fn from(finish_reason: GroqFinishReason) -> Self {
        match finish_reason {
            GroqFinishReason::Stop => FinishReason::Stop,
            GroqFinishReason::Length => FinishReason::Length,
            GroqFinishReason::ContentFilter => FinishReason::ContentFilter,
            GroqFinishReason::ToolCalls => FinishReason::ToolCall,
            GroqFinishReason::FunctionCall => FinishReason::ToolCall,
            GroqFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct GroqResponseChoice {
    pub(super) index: u8,
    pub(super) message: GroqResponseMessage,
    pub(super) finish_reason: GroqFinishReason,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct GroqResponse {
    pub(super) choices: Vec<GroqResponseChoice>,
    pub(super) usage: GroqUsage,
}

struct GroqResponseWithMetadata<'a> {
    response: GroqResponse,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    raw_response: String,
}

impl<'a> TryFrom<GroqResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: GroqResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let GroqResponseWithMetadata {
            mut response,
            latency,
            raw_request,
            raw_response,
            generic_request,
        } = value;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                raw_request: Some(raw_request),
                raw_response: Some(serde_json::to_string(&response).unwrap_or_default()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }
        let GroqResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(serde_json::to_string(&response).unwrap_or_default()),
                provider_type: PROVIDER_TYPE.to_string(),
            }))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(text) = message.content {
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        };
        let usage = response.usage.into();
        let system = generic_request.system.clone();
        let messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages: messages,
                raw_request,
                raw_response: raw_response.clone(),
                usage,
                latency,
                finish_reason: Some(finish_reason.into()),
            },
        ))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct GroqFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct GroqToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: GroqFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct GroqDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<GroqToolCallChunk>>,
}

// Custom deserializer function for empty string to None
// This is required because SGLang (which depends on this code) returns "" in streaming chunks instead of null
fn empty_string_as_none<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    if let Some(s) = opt {
        if s.is_empty() {
            return Ok(None);
        }
        // Convert serde_json::Error to D::Error
        Ok(Some(
            T::deserialize(serde_json::Value::String(s).into_deserializer())
                .map_err(|e| serde::de::Error::custom(e.to_string()))?,
        ))
    } else {
        Ok(None)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct GroqChatChunkChoice {
    delta: GroqDelta,
    #[serde(default)]
    #[serde(deserialize_with = "empty_string_as_none")]
    finish_reason: Option<GroqFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct GroqChatChunk {
    choices: Vec<GroqChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<GroqUsage>,
}

/// Maps an Groq chunk to a TensorZero chunk for streaming inferences
fn groq_to_tensorzero_chunk(
    mut chunk: GroqChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
) -> Result<ProviderInferenceResponseChunk, Error> {
    let raw_message = serde_json::to_string(&chunk).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!(
                "Error parsing response from Groq: {}",
                DisplayOrDebugGateway::new(e)
            ),
            raw_request: None,
            raw_response: Some(serde_json::to_string(&chunk).unwrap_or_default()),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            raw_request: None,
            raw_response: Some(serde_json::to_string(&chunk).unwrap_or_default()),
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into());
    }
    let usage = chunk.usage.map(Into::into);
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
#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
    use base64::Engine;
    use serde_json::json;

    use super::*;
    use crate::inference::types::file::Detail;
    use crate::inference::types::resolved_input::LazyFile;
    use crate::inference::types::storage::{StorageKind, StoragePath};
    use crate::inference::types::{
        ContentBlock, FunctionType, ObjectStorageFile, ObjectStoragePointer,
        PendingObjectStoreFile, RequestMessage,
    };
    use crate::providers::test_helpers::{
        MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL, WEATHER_TOOL_CONFIG,
    };
    use crate::tool::ToolCallConfig;
    use crate::utils::testing::capture_logs;

    #[test]
    fn test_handle_groq_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_groq_error(
            StatusCode::UNAUTHORIZED,
            "Unauthorized access",
            PROVIDER_TYPE,
        );
        let details = unauthorized.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            status_code,
            message,
            raw_request,
            raw_response,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Unauthorized access");
            assert_eq!(*status_code, Some(StatusCode::UNAUTHORIZED));
            assert_eq!(provider, PROVIDER_TYPE);
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, Some("Unauthorized access".to_string()));
        }

        // Test forbidden error
        let forbidden = handle_groq_error(StatusCode::FORBIDDEN, "Forbidden access", PROVIDER_TYPE);
        let details = forbidden.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            raw_request,
            raw_response,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Forbidden access");
            assert_eq!(*status_code, Some(StatusCode::FORBIDDEN));
            assert_eq!(provider, PROVIDER_TYPE);
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, Some("Forbidden access".to_string()));
        }

        // Test rate limit error
        let rate_limit = handle_groq_error(
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded",
            PROVIDER_TYPE,
        );
        let details = rate_limit.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            raw_request,
            raw_response,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Rate limit exceeded");
            assert_eq!(*status_code, Some(StatusCode::TOO_MANY_REQUESTS));
            assert_eq!(provider, PROVIDER_TYPE);
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, Some("Rate limit exceeded".to_string()));
        }

        // Test server error
        let server_error = handle_groq_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Server error",
            PROVIDER_TYPE,
        );
        let details = server_error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        if let ErrorDetails::InferenceServer {
            message,
            raw_request,
            raw_response,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Server error");
            assert_eq!(provider, PROVIDER_TYPE);
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, Some("Server error".to_string()));
        }
    }

    #[tokio::test]
    async fn test_groq_request_new() {
        // Test basic request
        let basic_request = ModelInferenceRequest {
            inference_id: uuid::Uuid::now_v7(),
            messages: vec![
                RequestMessage {
                    role: Role::User,
                    content: vec!["Hello".to_string().into()],
                },
                RequestMessage {
                    role: Role::Assistant,
                    content: vec!["Hi there!".to_string().into()],
                },
            ],
            system: None,
            tool_config: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let groq_request =
            GroqRequest::new("meta-llama/llama-4-scout-17b-16e-instruct", &basic_request)
                .await
                .unwrap();

        assert_eq!(
            groq_request.model,
            "meta-llama/llama-4-scout-17b-16e-instruct"
        );
        assert_eq!(groq_request.messages.len(), 2);
        assert_eq!(groq_request.temperature, Some(0.7));
        assert_eq!(groq_request.max_completion_tokens, Some(100));
        assert_eq!(groq_request.seed, Some(69));
        assert_eq!(groq_request.top_p, Some(0.9));
        assert_eq!(groq_request.presence_penalty, Some(0.1));
        assert_eq!(groq_request.frequency_penalty, Some(0.2));
        assert!(groq_request.stream);
        assert_eq!(groq_request.response_format, Some(GroqResponseFormat::Text));
        assert!(groq_request.tools.is_none());
        assert_eq!(groq_request.tool_choice, None);
        assert!(groq_request.parallel_tool_calls.is_none());

        // Test request with tools and JSON mode
        let request_with_tools = ModelInferenceRequest {
            inference_id: uuid::Uuid::now_v7(),
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
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let groq_request = GroqRequest::new(
            "meta-llama/llama-4-scout-17b-16e-instruct",
            &request_with_tools,
        )
        .await
        .unwrap();

        assert_eq!(
            groq_request.model,
            "meta-llama/llama-4-scout-17b-16e-instruct"
        );
        assert_eq!(groq_request.messages.len(), 2); // We'll add a system message containing Json to fit Groq requirements
        assert_eq!(groq_request.temperature, None);
        assert_eq!(groq_request.max_completion_tokens, None);
        assert_eq!(groq_request.seed, None);
        assert_eq!(groq_request.top_p, None);
        assert_eq!(groq_request.presence_penalty, None);
        assert_eq!(groq_request.frequency_penalty, None);
        assert!(!groq_request.stream);
        assert_eq!(
            groq_request.response_format,
            Some(GroqResponseFormat::JsonObject)
        );
        assert!(groq_request.tools.is_some());
        let tools = groq_request.tools.as_ref().unwrap();
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            groq_request.tool_choice,
            Some(GroqToolChoice::Specific(SpecificToolChoice {
                r#type: GroqToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );

        // Test request with strict JSON mode with no output schema
        let request_with_tools = ModelInferenceRequest {
            inference_id: uuid::Uuid::now_v7(),
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
            json_mode: ModelInferenceRequestJsonMode::Strict,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let groq_request = GroqRequest::new(
            "meta-llama/llama-4-scout-17b-16e-instruct",
            &request_with_tools,
        )
        .await
        .unwrap();

        assert_eq!(
            groq_request.model,
            "meta-llama/llama-4-scout-17b-16e-instruct"
        );
        assert_eq!(groq_request.messages.len(), 1);
        assert_eq!(groq_request.temperature, None);
        assert_eq!(groq_request.max_completion_tokens, None);
        assert_eq!(groq_request.seed, None);
        assert!(!groq_request.stream);
        assert_eq!(groq_request.top_p, None);
        assert_eq!(groq_request.presence_penalty, None);
        assert_eq!(groq_request.frequency_penalty, None);
        // Resolves to normal JSON mode since no schema is provided (this shouldn't really happen in practice)
        assert_eq!(
            groq_request.response_format,
            Some(GroqResponseFormat::JsonObject)
        );

        // Test request with strict JSON mode with an output schema
        let output_schema = json!({});
        let request_with_tools = ModelInferenceRequest {
            inference_id: uuid::Uuid::now_v7(),
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
            json_mode: ModelInferenceRequestJsonMode::Strict,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
            extra_body: Default::default(),
            ..Default::default()
        };

        let groq_request = GroqRequest::new(
            "meta-llama/llama-4-scout-17b-16e-instruct",
            &request_with_tools,
        )
        .await
        .unwrap();

        assert_eq!(
            groq_request.model,
            "meta-llama/llama-4-scout-17b-16e-instruct"
        );
        assert_eq!(groq_request.messages.len(), 1);
        assert_eq!(groq_request.temperature, None);
        assert_eq!(groq_request.max_completion_tokens, None);
        assert_eq!(groq_request.seed, None);
        assert!(!groq_request.stream);
        assert_eq!(groq_request.top_p, None);
        assert_eq!(groq_request.presence_penalty, None);
        assert_eq!(groq_request.frequency_penalty, None);
        let expected_schema = serde_json::json!({});
        match groq_request.response_format {
            Some(GroqResponseFormat::JsonSchema { json_schema }) => {
                assert_eq!(json_schema["schema"], expected_schema);
                assert_eq!(json_schema["name"], "response");
                assert_eq!(json_schema["strict"], true);
            }
            _ => panic!("Expected JsonSchema variant"),
        }
    }

    #[test]
    fn test_try_from_groq_response() {
        // Test case 1: Valid response with content
        let valid_response = GroqResponse {
            choices: vec![GroqResponseChoice {
                index: 0,
                message: GroqResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: GroqFinishReason::Stop,
            }],
            usage: GroqUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: uuid::Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let request_body = GroqRequest {
            messages: vec![],
            model: "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(GroqResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            // allowed_tools is now part of tool_choice (AllowedToolsChoice variant)
            stop: None,
            reasoning_effort: None,
            service_tier: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test_response".to_string();
        let result = ProviderInferenceResponse::try_from(GroqResponseWithMetadata {
            response: valid_response,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec!["Hello, world!".to_string().into()]
        );
        assert_eq!(inference_response.usage.input_tokens, Some(10));
        assert_eq!(inference_response.usage.output_tokens, Some(20));
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(100)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.raw_response, raw_response);
        assert_eq!(inference_response.system, None);
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );
        // Test case 2: Valid response with tool calls
        let valid_response_with_tools = GroqResponse {
            choices: vec![GroqResponseChoice {
                index: 0,
                finish_reason: GroqFinishReason::ToolCalls,
                message: GroqResponseMessage {
                    content: None,
                    tool_calls: Some(vec![GroqResponseToolCall {
                        id: "call1".to_string(),
                        r#type: GroqToolType::Function,
                        function: GroqResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
            }],
            usage: GroqUsage {
                prompt_tokens: 15,
                completion_tokens: 25,
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: uuid::Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }],
            system: Some("test_system".to_string()),
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            seed: Some(69),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let request_body = GroqRequest {
            messages: vec![],
            model: "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(GroqResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            // allowed_tools is now part of tool_choice (AllowedToolsChoice variant)
            stop: None,
            reasoning_effort: None,
            service_tier: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(GroqResponseWithMetadata {
            response: valid_response_with_tools,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(110),
            },
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
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
            inference_response.finish_reason,
            Some(FinishReason::ToolCall)
        );
        assert_eq!(
            inference_response.latency,
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
        let invalid_response_no_choices = GroqResponse {
            choices: vec![],
            usage: GroqUsage {
                prompt_tokens: 5,
                completion_tokens: 0,
            },
        };
        let request_body = GroqRequest {
            messages: vec![],
            model: "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(GroqResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            // allowed_tools is now part of tool_choice (AllowedToolsChoice variant)
            stop: None,
            reasoning_effort: None,
            service_tier: None,
        };
        let result = ProviderInferenceResponse::try_from(GroqResponseWithMetadata {
            response: invalid_response_no_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(120),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        let details = err.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));

        // Test case 4: Invalid response with multiple choices
        let invalid_response_multiple_choices = GroqResponse {
            choices: vec![
                GroqResponseChoice {
                    index: 0,
                    message: GroqResponseMessage {
                        content: Some("Choice 1".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: GroqFinishReason::Stop,
                },
                GroqResponseChoice {
                    index: 1,
                    finish_reason: GroqFinishReason::Stop,
                    message: GroqResponseMessage {
                        content: Some("Choice 2".to_string()),
                        tool_calls: None,
                    },
                },
            ],
            usage: GroqUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
            },
        };

        let request_body = GroqRequest {
            messages: vec![],
            model: "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(GroqResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            // allowed_tools is now part of tool_choice (AllowedToolsChoice variant)
            stop: None,
            reasoning_effort: None,
            service_tier: None,
        };
        let result = ProviderInferenceResponse::try_from(GroqResponseWithMetadata {
            response: invalid_response_multiple_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(130),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        let details = err.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
    }

    #[test]
    fn test_prepare_groq_tools() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: uuid::Uuid::now_v7(),
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
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&MULTI_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let (tools, tool_choice, parallel_tool_calls) =
            prepare_groq_tools(&request_with_tools).unwrap();
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(tools[1].function.name, QUERY_TOOL.name());
        assert_eq!(tools[1].function.parameters, QUERY_TOOL.parameters());
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            GroqToolChoice::String(GroqToolChoiceString::Required)
        );
        let parallel_tool_calls = parallel_tool_calls.unwrap();
        assert!(parallel_tool_calls);
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: Some(true),
            ..Default::default()
        };

        // Test no tools but a tool choice and make sure tool choice output is None
        let request_without_tools = ModelInferenceRequest {
            inference_id: uuid::Uuid::now_v7(),
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
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let (tools, tool_choice, parallel_tool_calls) =
            prepare_groq_tools(&request_without_tools).unwrap();
        assert!(tools.is_none());
        assert!(tool_choice.is_none());
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_prepare_groq_tools_with_allowed_tools() {
        use crate::tool::{AllowedTools, AllowedToolsChoice};

        // Test with allowed_tools specified
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
            inference_id: uuid::Uuid::now_v7(),
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
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_groq_tools(&request).unwrap();

        // Verify tools are returned
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2);

        // Verify tool_choice - should now be AllowedTools variant with OpenAI spec structure
        let tool_choice = tool_choice.unwrap();
        match tool_choice {
            GroqToolChoice::AllowedTools(allowed_tools_choice) => {
                assert_eq!(allowed_tools_choice.r#type, "allowed_tools");
                assert_eq!(
                    allowed_tools_choice.allowed_tools.mode,
                    AllowedToolsMode::Auto
                );
                assert_eq!(allowed_tools_choice.allowed_tools.tools.len(), 1);
                match &allowed_tools_choice.allowed_tools.tools[0] {
                    ToolReference::Function { function } => {
                        assert_eq!(function.name, WEATHER_TOOL.name());
                    }
                    ToolReference::Custom { .. } => panic!("Expected Function variant"),
                }
            }
            _ => panic!("Expected AllowedTools variant"),
        }

        // Verify parallel_tool_calls
        let parallel_tool_calls = parallel_tool_calls.unwrap();
        assert!(!parallel_tool_calls);
    }

    #[tokio::test]
    async fn test_tensorzero_to_groq_messages() {
        let content_blocks = vec!["Hello".to_string().into()];
        let groq_messages = tensorzero_to_groq_user_messages(&content_blocks, true)
            .await
            .unwrap();
        assert_eq!(groq_messages.len(), 1);
        match &groq_messages[0] {
            GroqRequestMessage::User(content) => {
                assert_eq!(
                    content.content,
                    &[GroqContentBlock::Text {
                        text: "Hello".into()
                    }]
                );
            }
            _ => panic!("Expected a user message"),
        }

        // Message with multiple blocks
        let content_blocks = vec![
            "Hello".to_string().into(),
            "How are you?".to_string().into(),
        ];
        let groq_messages = tensorzero_to_groq_user_messages(&content_blocks, true)
            .await
            .unwrap();
        assert_eq!(groq_messages.len(), 1);
        match &groq_messages[0] {
            GroqRequestMessage::User(content) => {
                assert_eq!(
                    content.content,
                    vec![
                        GroqContentBlock::Text {
                            text: "Hello".into()
                        },
                        GroqContentBlock::Text {
                            text: "How are you?".into()
                        }
                    ]
                );
            }
            _ => panic!("Expected a user message"),
        }

        // User message with one string and one tool call block
        // Since user messages in Groq land can't contain tool calls (nor should they honestly),
        // We split the tool call out into a separate assistant message
        let tool_block = ContentBlock::ToolCall(ToolCall {
            id: "call1".to_string(),
            name: "test_function".to_string(),
            arguments: "{}".to_string(),
        });
        let content_blocks = vec!["Hello".to_string().into(), tool_block];
        let groq_messages = tensorzero_to_groq_assistant_messages(&content_blocks, true)
            .await
            .unwrap();
        assert_eq!(groq_messages.len(), 1);
        match &groq_messages[0] {
            GroqRequestMessage::Assistant(content) => {
                assert_eq!(
                    content.content,
                    Some(vec![GroqContentBlock::Text {
                        text: "Hello".into()
                    }])
                );
                let tool_calls = content.tool_calls.as_ref().unwrap();
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call1");
                assert_eq!(tool_calls[0].function.name, "test_function");
                assert_eq!(tool_calls[0].function.arguments, "{}");
            }
            _ => panic!("Expected an assistant message"),
        }
    }

    #[test]
    fn test_groq_to_tensorzero_chunk() {
        let chunk = GroqChatChunk {
            choices: vec![GroqChatChunkChoice {
                delta: GroqDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(GroqFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let message =
            groq_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
                .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "Hello".to_string(),
                id: "0".to_string(),
            })],
        );
        assert_eq!(message.finish_reason, Some(FinishReason::Stop));
        // Test what an intermediate tool chunk should look like
        let chunk = GroqChatChunk {
            choices: vec![GroqChatChunkChoice {
                finish_reason: Some(GroqFinishReason::ToolCalls),
                delta: GroqDelta {
                    content: None,
                    tool_calls: Some(vec![GroqToolCallChunk {
                        index: 0,
                        id: None,
                        function: GroqFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message =
            groq_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
                .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "id1".to_string(),
                raw_name: None,
                raw_arguments: "{\"hello\":\"world\"}".to_string(),
            })]
        );
        assert_eq!(message.finish_reason, Some(FinishReason::ToolCall));
        // Test what a bad tool chunk would do (new ID but no names)
        let chunk = GroqChatChunk {
            choices: vec![GroqChatChunkChoice {
                finish_reason: None,
                delta: GroqDelta {
                    content: None,
                    tool_calls: Some(vec![GroqToolCallChunk {
                        index: 1,
                        id: None,
                        function: GroqFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let error =
            groq_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
                .unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InferenceServer {
                message: "Tool call index out of bounds (meaning we haven't seen this many ids in the stream)".to_string(),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            }
        );
        // Test a correct new tool chunk
        let chunk = GroqChatChunk {
            choices: vec![GroqChatChunkChoice {
                finish_reason: Some(GroqFinishReason::Stop),
                delta: GroqDelta {
                    content: None,
                    tool_calls: Some(vec![GroqToolCallChunk {
                        index: 1,
                        id: Some("id2".to_string()),
                        function: GroqFunctionCallChunk {
                            name: Some("name2".to_string()),
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message =
            groq_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
                .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "id2".to_string(),
                raw_name: Some("name2".to_string()),
                raw_arguments: "{\"hello\":\"world\"}".to_string(),
            })]
        );
        assert_eq!(message.finish_reason, Some(FinishReason::Stop));
        // Check that the lists were updated
        assert_eq!(tool_call_ids, vec!["id1".to_string(), "id2".to_string()]);

        // Check a chunk with no choices and only usage
        // Test a correct new tool chunk
        let chunk = GroqChatChunk {
            choices: vec![],
            usage: Some(GroqUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
            }),
        };
        let message =
            groq_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
                .unwrap();
        assert_eq!(message.content, vec![]);
        assert_eq!(
            message.usage,
            Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            })
        );
    }

    #[test]
    fn test_new_groq_response_format() {
        // Test JSON mode On
        let json_mode = ModelInferenceRequestJsonMode::On;
        let output_schema = None;
        let format = GroqResponseFormat::new(json_mode, output_schema);
        assert_eq!(format, GroqResponseFormat::JsonObject);

        // Test JSON mode Off
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let format = GroqResponseFormat::new(json_mode, output_schema);
        assert_eq!(format, GroqResponseFormat::Text);

        // Test JSON mode Strict with no schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let format = GroqResponseFormat::new(json_mode, output_schema);
        assert_eq!(format, GroqResponseFormat::JsonObject);

        // Test JSON mode Strict with schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"type": "string"}
            }
        });
        let output_schema = Some(&schema);
        let format = GroqResponseFormat::new(json_mode, output_schema);
        match format {
            GroqResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema["schema"], schema);
                assert_eq!(json_schema["name"], "response");
                assert_eq!(json_schema["strict"], true);
            }
            _ => panic!("Expected JsonSchema variant"),
        }

        // Test serialization of JsonSchema format
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"type": "string"}
            }
        });
        let format = GroqResponseFormat::new(json_mode, Some(&schema));
        let serialized = serde_json::to_value(&format).unwrap();
        assert_eq!(serialized["type"], "json_schema");
        assert_eq!(serialized["json_schema"]["name"], "response");
        assert_eq!(serialized["json_schema"]["strict"], true);
        assert_eq!(serialized["json_schema"]["schema"], schema);
    }

    #[test]
    fn test_tensorzero_to_groq_system_message() {
        // Test Case 1: system is None, json_mode is Off
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages: Vec<GroqRequestMessage> = vec![];
        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert_eq!(result, None);

        // Test Case 2: system is Some, json_mode is On, messages contain "json"
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            GroqRequestMessage::User(GroqUserRequestMessage {
                content: vec![GroqContentBlock::Text {
                    text: "Please respond in JSON format.".into(),
                }],
            }),
            GroqRequestMessage::Assistant(GroqAssistantRequestMessage {
                content: Some(vec![GroqContentBlock::Text {
                    text: "Sure, here is the data.".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(GroqRequestMessage::System(GroqSystemRequestMessage {
            content: Cow::Borrowed("System instructions"),
        }));
        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 3: system is Some, json_mode is On, messages do not contain "json"
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            GroqRequestMessage::User(GroqUserRequestMessage {
                content: vec![GroqContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            GroqRequestMessage::Assistant(GroqAssistantRequestMessage {
                content: Some(vec![GroqContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected_content = "Respond using JSON.\n\nSystem instructions".to_string();
        let expected = Some(GroqRequestMessage::System(GroqSystemRequestMessage {
            content: Cow::Owned(expected_content),
        }));
        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 4: system is Some, json_mode is Off
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages = vec![
            GroqRequestMessage::User(GroqUserRequestMessage {
                content: vec![GroqContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            GroqRequestMessage::Assistant(GroqAssistantRequestMessage {
                content: Some(vec![GroqContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(GroqRequestMessage::System(GroqSystemRequestMessage {
            content: Cow::Borrowed("System instructions"),
        }));
        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 5: system is Some, json_mode is Strict
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let messages = vec![
            GroqRequestMessage::User(GroqUserRequestMessage {
                content: vec![GroqContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            GroqRequestMessage::Assistant(GroqAssistantRequestMessage {
                content: Some(vec![GroqContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(GroqRequestMessage::System(GroqSystemRequestMessage {
            content: Cow::Borrowed("System instructions"),
        }));
        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 6: system contains "json", json_mode is On
        let system = Some("Respond using JSON.\n\nSystem instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![GroqRequestMessage::User(GroqUserRequestMessage {
            content: vec![GroqContentBlock::Text {
                text: "Hello, how are you?".into(),
            }],
        })];
        let expected = Some(GroqRequestMessage::System(GroqSystemRequestMessage {
            content: Cow::Borrowed("Respond using JSON.\n\nSystem instructions"),
        }));
        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 7: system is None, json_mode is On
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            GroqRequestMessage::User(GroqUserRequestMessage {
                content: vec![GroqContentBlock::Text {
                    text: "Tell me a joke.".into(),
                }],
            }),
            GroqRequestMessage::Assistant(GroqAssistantRequestMessage {
                content: Some(vec![GroqContentBlock::Text {
                    text: "Sure, here's one for you.".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(GroqRequestMessage::System(GroqSystemRequestMessage {
            content: Cow::Owned("Respond using JSON.".to_string()),
        }));
        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 8: system is None, json_mode is Strict
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let messages = vec![
            GroqRequestMessage::User(GroqUserRequestMessage {
                content: vec![GroqContentBlock::Text {
                    text: "Provide a summary of the news.".into(),
                }],
            }),
            GroqRequestMessage::Assistant(GroqAssistantRequestMessage {
                content: Some(vec![GroqContentBlock::Text {
                    text: "Here's the summary.".into(),
                }]),
                tool_calls: None,
            }),
        ];

        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert!(result.is_none());

        // Test Case 9: system is None, json_mode is On, with empty messages
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages: Vec<GroqRequestMessage> = vec![];
        let expected = Some(GroqRequestMessage::System(GroqSystemRequestMessage {
            content: Cow::Owned("Respond using JSON.".to_string()),
        }));
        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 10: system is None, json_mode is Off, with messages containing "json"
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages = vec![GroqRequestMessage::User(GroqUserRequestMessage {
            content: vec![GroqContentBlock::Text {
                text: "Please include JSON in your response.".into(),
            }],
        })];
        let expected = None;
        let result = tensorzero_to_groq_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_try_from_groq_credentials() {
        // Test Static credentials
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = GroqCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, GroqCredentials::Static(_)));

        // Test Dynamic credentials
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = GroqCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, GroqCredentials::Dynamic(_)));

        // Test None credentials
        let generic = Credential::None;
        let creds = GroqCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, GroqCredentials::None));

        // Test Missing credentials
        let generic = Credential::Missing;
        let creds = GroqCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, GroqCredentials::None));

        // Test invalid credential type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = GroqCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_serialize_user_messages() {
        // Test that a single message is serialized as 'content: string'
        let message = GroqUserRequestMessage {
            content: vec![GroqContentBlock::Text {
                text: "My single message".into(),
            }],
        };
        let serialized = serde_json::to_string(&message).unwrap();
        assert_eq!(serialized, r#"{"content":"My single message"}"#);

        // Test that a multiple messages are serialized as an array of content blocks
        let message = GroqUserRequestMessage {
            content: vec![
                GroqContentBlock::Text {
                    text: "My first message".into(),
                },
                GroqContentBlock::Text {
                    text: "My second message".into(),
                },
            ],
        };
        let serialized = serde_json::to_string(&message).unwrap();
        assert_eq!(
            serialized,
            r#"{"content":[{"type":"text","text":"My first message"},{"type":"text","text":"My second message"}]}"#
        );
    }

    #[test]
    fn test_groq_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = GroqRequest {
            model: "test-model",
            ..Default::default()
        };

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort is applied correctly
        assert_eq!(request.reasoning_effort, Some("high".to_string()));

        // Test that thinking_budget_tokens warns with tip about reasoning_effort
        assert!(logs_contain(
            "Groq does not support the inference parameter `thinking_budget_tokens`, so it will be ignored. Tip: You might want to use `reasoning_effort` for this provider."
        ));

        // Test that verbosity warns
        assert!(logs_contain(
            "Groq does not support the inference parameter `verbosity`"
        ));
    }

    #[tokio::test]
    async fn test_groq_warns_on_detail() {
        let logs_contain = capture_logs();

        // Test with resolved file (base64 encoding path) with detail
        let dummy_storage_path = StoragePath {
            kind: StorageKind::Disabled,
            path: object_store::path::Path::parse("dummy-path").unwrap(),
        };
        let content_blocks = vec![ContentBlock::File(Box::new(LazyFile::Base64(
            PendingObjectStoreFile(ObjectStorageFile {
                file: ObjectStoragePointer {
                    source_url: None,
                    mime_type: mime::IMAGE_PNG,
                    storage_path: dummy_storage_path,
                    detail: Some(Detail::High),
                    filename: None,
                },
                data: BASE64_STANDARD.encode(b"fake image data"),
            }),
        )))];

        let _result = tensorzero_to_groq_user_messages(&content_blocks, true).await;

        // Should log a warning about detail not being supported
        assert!(logs_contain(
            "The image detail parameter is not supported by Groq"
        ));
    }
}
