use futures::{Stream, StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::de::IntoDeserializer;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::time::Instant;

use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{warn_discarded_thought_block, DisplayOrDebugGateway, Error, ErrorDetails};
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::resolved_input::FileWithPath;
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, ContentBlock, ContentBlockChunk,
    ContentBlockOutput, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseChunk, RequestMessage, Role, Text, TextChunk, Usage,
};
use crate::inference::types::{
    FinishReason, ProviderInferenceResponseArgs, ProviderInferenceResponseStreamInner,
};
use crate::inference::{InferenceProvider, TensorZeroEventError};
use crate::model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

use crate::providers::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("GROQ_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "Llama";
pub const PROVIDER_TYPE: &str = "llama";

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct LlamaProvider {
    model_name: String,
    #[serde(skip)]
    credentials: LlamaCredentials,
}

static DEFAULT_CREDENTIALS: OnceLock<LlamaCredentials> = OnceLock::new();

impl LlamaProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credentials = build_creds_caching_default(
            api_key_location,
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;
        Ok(LlamaProvider {
            model_name,
            credentials,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum LlamaCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for LlamaCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(LlamaCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(LlamaCredentials::Dynamic(key_name)),
            Credential::None => Ok(LlamaCredentials::None),
            Credential::Missing => Ok(LlamaCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Llama provider".to_string(),
            })),
        }
    }
}

impl LlamaCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            LlamaCredentials::Static(api_key) => Ok(Some(api_key)),
            LlamaCredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                }))
                .transpose()
            }
            LlamaCredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for LlamaProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_url = "https://api.llama.com/openai/v1/chat/completions".to_string();
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();

        let request_body =
            serde_json::to_value(LlamaRequest::new(&self.model_name, request.request)?).map_err(
                |e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing Llama request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                },
            )?;

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
            Ok(LlamaResponseWithMetadata {
                response,
                raw_response,
                latency,
                raw_request: raw_request.clone(),
                generic_request: request.request,
            }
            .try_into()?)
        } else {
            Err(handle_llama_error(
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
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(LlamaRequest::new(&self.model_name, request)?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Llama request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let request_url = "https://api.llama.com/openai/v1/chat/completions".to_string();
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
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

        let stream = stream_llama(
            PROVIDER_TYPE.to_string(),
            event_source.map_err(TensorZeroEventError::EventSource),
            start_time,
        )
        .peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
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
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}
pub async fn convert_stream_error(provider_type: String, e: reqwest_eventsource::Error) -> Error {
    let message = e.to_string();
    let mut raw_response = None;
    if let reqwest_eventsource::Error::InvalidStatusCode(_, resp) = e {
        raw_response = resp.text().await.ok();
    }
    ErrorDetails::InferenceServer {
        message,
        raw_request: None,
        raw_response,
        provider_type,
    }
    .into()
}

pub fn stream_llama(
    provider_type: String,
    event_source: impl Stream<Item = Result<Event, TensorZeroEventError>> + Send + 'static,
    start_time: Instant,
) -> ProviderInferenceResponseStreamInner {
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
                            yield Err(convert_stream_error(provider_type.clone(), e).await);
                        }
                    }
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<LlamaChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {e}",
                                    ),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                                provider_type: provider_type.clone(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            llama_to_tensorzero_chunk(d, latency, &mut tool_call_ids)
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

pub(super) fn handle_llama_error(
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
pub(super) struct LlamaSystemRequestMessage<'a> {
    pub content: Cow<'a, str>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct LlamaUserRequestMessage<'a> {
    #[serde(serialize_with = "serialize_text_content_vec")]
    pub(super) content: Vec<LlamaContentBlock<'a>>,
}

fn serialize_text_content_vec<S>(
    content: &Vec<LlamaContentBlock<'_>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // If we have a single text block, serialize it as a string
    // to stay compatible with older providers which may not support content blocks
    if let [LlamaContentBlock::Text { text }] = &content.as_slice() {
        text.serialize(serializer)
    } else {
        content.serialize(serializer)
    }
}

fn serialize_optional_text_content_vec<S>(
    content: &Option<Vec<LlamaContentBlock<'_>>>,
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
pub enum LlamaContentBlock<'a> {
    Text { text: Cow<'a, str> },
    ImageUrl { image_url: LlamaImageUrl },
    Unknown { data: Cow<'a, Value> },
}

impl Serialize for LlamaContentBlock<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Helper<'a> {
            Text { text: &'a str },
            ImageUrl { image_url: &'a LlamaImageUrl },
        }
        match self {
            LlamaContentBlock::Text { text } => Helper::Text { text }.serialize(serializer),
            LlamaContentBlock::ImageUrl { image_url } => {
                Helper::ImageUrl { image_url }.serialize(serializer)
            }
            LlamaContentBlock::Unknown { data } => data.serialize(serializer),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct LlamaImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LlamaRequestFunctionCall<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct LlamaRequestToolCall<'a> {
    pub id: &'a str,
    pub r#type: LlamaToolType,
    pub function: LlamaRequestFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for LlamaRequestToolCall<'a> {
    fn from(tool_call: &'a ToolCall) -> Self {
        LlamaRequestToolCall {
            id: &tool_call.id,
            r#type: LlamaToolType::Function,
            function: LlamaRequestFunctionCall {
                name: &tool_call.name,
                arguments: &tool_call.arguments,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct LlamaAssistantRequestMessage<'a> {
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_text_content_vec"
    )]
    pub content: Option<Vec<LlamaContentBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<LlamaRequestToolCall<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct LlamaToolRequestMessage<'a> {
    pub content: &'a str,
    pub tool_call_id: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub(super) enum LlamaRequestMessage<'a> {
    System(LlamaSystemRequestMessage<'a>),
    User(LlamaUserRequestMessage<'a>),
    Assistant(LlamaAssistantRequestMessage<'a>),
    Tool(LlamaToolRequestMessage<'a>),
}

impl LlamaRequestMessage<'_> {
    pub fn content_contains_case_insensitive(&self, value: &str) -> bool {
        match self {
            LlamaRequestMessage::System(msg) => msg.content.to_lowercase().contains(value),
            LlamaRequestMessage::User(msg) => msg.content.iter().any(|c| match c {
                LlamaContentBlock::Text { text } => text.to_lowercase().contains(value),
                LlamaContentBlock::ImageUrl { .. } => false,
                // Don't inspect the contents of 'unknown' blocks
                LlamaContentBlock::Unknown { data: _ } => false,
            }),
            LlamaRequestMessage::Assistant(msg) => {
                if let Some(content) = &msg.content {
                    content.iter().any(|c| match c {
                        LlamaContentBlock::Text { text } => text.to_lowercase().contains(value),
                        LlamaContentBlock::ImageUrl { .. } => false,
                        // Don't inspect the contents of 'unknown' blocks
                        LlamaContentBlock::Unknown { data: _ } => false,
                    })
                } else {
                    false
                }
            }
            LlamaRequestMessage::Tool(msg) => msg.content.to_lowercase().contains(value),
        }
    }
}

pub(super) fn prepare_llama_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Result<Vec<LlamaRequestMessage<'a>>, Error> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in request.messages.iter() {
        messages.extend(tensorzero_to_llama_messages(message)?);
    }
    if let Some(system_msg) =
        tensorzero_to_llama_system_message(request.system.as_deref(), &request.json_mode, &messages)
    {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to Llama format
/// NOTE: parallel tool calls are unreliable, and specific tool choice doesn't work
pub(super) fn prepare_llama_tools<'a>(
    request: &'a ModelInferenceRequest,
) -> (
    Option<Vec<LlamaTool<'a>>>,
    Option<LlamaToolChoice<'a>>,
    Option<bool>,
) {
    match &request.tool_config {
        None => (None, None, None),
        Some(tool_config) => {
            if tool_config.tools_available.is_empty() {
                return (None, None, None);
            }
            let tools = Some(tool_config.tools_available.iter().map(Into::into).collect());
            let tool_choice = Some((&tool_config.tool_choice).into());
            let parallel_tool_calls = tool_config.parallel_tool_calls;
            (tools, tool_choice, parallel_tool_calls)
        }
    }
}

/// If ModelInferenceRequestJsonMode::On and the system message or instructions does not contain "JSON"
/// the request will return an error.
/// So, we need to format the instructions to include "Respond using JSON." if it doesn't already.
pub(super) fn tensorzero_to_llama_system_message<'a>(
    system: Option<&'a str>,
    json_mode: &ModelInferenceRequestJsonMode,
    messages: &[LlamaRequestMessage<'a>],
) -> Option<LlamaRequestMessage<'a>> {
    match system {
        Some(system) => {
            match json_mode {
                ModelInferenceRequestJsonMode::On => {
                    if messages
                        .iter()
                        .any(|msg| msg.content_contains_case_insensitive("json"))
                        || system.to_lowercase().contains("json")
                    {
                        LlamaRequestMessage::System(LlamaSystemRequestMessage {
                            content: Cow::Borrowed(system),
                        })
                    } else {
                        let formatted_instructions = format!("Respond using JSON.\n\n{system}");
                        LlamaRequestMessage::System(LlamaSystemRequestMessage {
                            content: Cow::Owned(formatted_instructions),
                        })
                    }
                }

                // If JSON mode is either off or strict, we don't need to do anything special
                _ => LlamaRequestMessage::System(LlamaSystemRequestMessage {
                    content: Cow::Borrowed(system),
                }),
            }
            .into()
        }
        None => match *json_mode {
            ModelInferenceRequestJsonMode::On => {
                Some(LlamaRequestMessage::System(LlamaSystemRequestMessage {
                    content: Cow::Owned("Respond using JSON.".to_string()),
                }))
            }
            _ => None,
        },
    }
}

pub(super) fn tensorzero_to_llama_messages(
    message: &RequestMessage,
) -> Result<Vec<LlamaRequestMessage<'_>>, Error> {
    match message.role {
        Role::User => tensorzero_to_llama_user_messages(&message.content),
        Role::Assistant => tensorzero_to_llama_assistant_messages(&message.content),
    }
}

fn tensorzero_to_llama_user_messages(
    content_blocks: &[ContentBlock],
) -> Result<Vec<LlamaRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the user content blocks.

    let mut messages = Vec::new();
    let mut user_content_blocks = Vec::new();

    for block in content_blocks.iter() {
        match block {
            ContentBlock::Text(Text { text }) => {
                user_content_blocks.push(LlamaContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool calls are not supported in user messages".to_string(),
                }));
            }
            ContentBlock::ToolResult(tool_result) => {
                messages.push(LlamaRequestMessage::Tool(LlamaToolRequestMessage {
                    content: &tool_result.result,
                    tool_call_id: &tool_result.id,
                }));
            }
            ContentBlock::File(file) => {
                let FileWithPath {
                    file,
                    storage_path: _,
                } = &**file;
                user_content_blocks.push(LlamaContentBlock::ImageUrl {
                    image_url: LlamaImageUrl {
                        // This will only produce an error if we pass in a bad
                        // `Base64Image` (with missing image data)
                        url: format!("data:{};base64,{}", file.mime_type, file.data()?),
                    },
                });
            }
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(PROVIDER_TYPE, thought);
            }
            ContentBlock::Unknown {
                data,
                model_provider_name: _,
            } => {
                user_content_blocks.push(LlamaContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
        };
    }

    // If there are any user content blocks, combine them into a single user message.
    if !user_content_blocks.is_empty() {
        messages.push(LlamaRequestMessage::User(LlamaUserRequestMessage {
            content: user_content_blocks,
        }));
    }

    Ok(messages)
}

fn tensorzero_to_llama_assistant_messages(
    content_blocks: &[ContentBlock],
) -> Result<Vec<LlamaRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the assistant content blocks.
    let mut assistant_content_blocks = Vec::new();
    let mut assistant_tool_calls = Vec::new();

    for block in content_blocks.iter() {
        match block {
            ContentBlock::Text(Text { text }) => {
                assistant_content_blocks.push(LlamaContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(tool_call) => {
                let tool_call = LlamaRequestToolCall {
                    id: &tool_call.id,
                    r#type: LlamaToolType::Function,
                    function: LlamaRequestFunctionCall {
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
            ContentBlock::File(file) => {
                let FileWithPath {
                    file,
                    storage_path: _,
                } = &**file;
                assistant_content_blocks.push(LlamaContentBlock::ImageUrl {
                    image_url: LlamaImageUrl {
                        // This will only produce an error if we pass in a bad
                        // `Base64Image` (with missing image data)
                        url: format!("data:{};base64,{}", file.mime_type, file.data()?),
                    },
                });
            }
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(PROVIDER_TYPE, thought);
            }
            ContentBlock::Unknown {
                data,
                model_provider_name: _,
            } => {
                assistant_content_blocks.push(LlamaContentBlock::Unknown {
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

    let message = LlamaRequestMessage::Assistant(LlamaAssistantRequestMessage {
        content,
        tool_calls,
    });

    Ok(vec![message])
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum LlamaResponseFormat<'a> {
    #[default]
    Text,
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<&'a Value>, // the desired JSON schema
    },
}

impl<'a> LlamaResponseFormat<'a> {
    fn new(json_mode: &ModelInferenceRequestJsonMode, output_schema: Option<&'a Value>) -> Self {
        match json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                LlamaResponseFormat::JsonObject {
                    schema: output_schema,
                }
            }
            ModelInferenceRequestJsonMode::Off => LlamaResponseFormat::Text,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LlamaToolType {
    Function,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct LlamaFunction<'a> {
    pub(super) name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<&'a str>,
    pub parameters: &'a Value,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct LlamaTool<'a> {
    pub(super) r#type: LlamaToolType,
    pub(super) function: LlamaFunction<'a>,
    pub(super) strict: bool,
}

impl<'a> From<&'a ToolConfig> for LlamaTool<'a> {
    fn from(tool: &'a ToolConfig) -> Self {
        LlamaTool {
            r#type: LlamaToolType::Function,
            function: LlamaFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
            strict: tool.strict(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(super) enum LlamaToolChoice<'a> {
    String(LlamaToolChoiceString),
    Specific(SpecificToolChoice<'a>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum LlamaToolChoiceString {
    None,
    Auto,
    Required,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SpecificToolChoice<'a> {
    pub(super) r#type: LlamaToolType,
    pub(super) function: SpecificToolFunction<'a>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct SpecificToolFunction<'a> {
    pub(super) name: &'a str,
}

impl Default for LlamaToolChoice<'_> {
    fn default() -> Self {
        LlamaToolChoice::String(LlamaToolChoiceString::None)
    }
}

impl<'a> From<&'a ToolChoice> for LlamaToolChoice<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::None => LlamaToolChoice::String(LlamaToolChoiceString::None),
            ToolChoice::Auto => LlamaToolChoice::String(LlamaToolChoiceString::Auto),
            ToolChoice::Required => LlamaToolChoice::String(LlamaToolChoiceString::Required),
            ToolChoice::Specific(tool_name) => LlamaToolChoice::Specific(SpecificToolChoice {
                r#type: LlamaToolType::Function,
                function: SpecificToolFunction { name: tool_name },
            }),
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct StreamOptions {
    pub(super) include_usage: bool,
}

/// This struct defines the supported parameters for the Llama API
/// See the [Llama API documentation](https://platform.llama.com/docs/api-reference/chat/create)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Serialize)]
struct LlamaRequest<'a> {
    messages: Vec<LlamaRequestMessage<'a>>,
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
    response_format: Option<LlamaResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<LlamaTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<LlamaToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
}

impl<'a> LlamaRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<LlamaRequest<'a>, Error> {
        let response_format = Some(LlamaResponseFormat::new(
            &request.json_mode,
            request.output_schema,
        ));
        let stream_options = if request.stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        };
        let mut messages = prepare_llama_messages(request)?;

        let (tools, tool_choice, mut parallel_tool_calls) = prepare_llama_tools(request);
        if model.to_lowercase().starts_with("o1") && parallel_tool_calls == Some(false) {
            parallel_tool_calls = None;
        }

        if model.to_lowercase().starts_with("o1-mini") {
            if let Some(LlamaRequestMessage::System(_)) = messages.first() {
                if let LlamaRequestMessage::System(system_msg) = messages.remove(0) {
                    let user_msg = LlamaRequestMessage::User(LlamaUserRequestMessage {
                        content: vec![LlamaContentBlock::Text {
                            text: system_msg.content,
                        }],
                    });
                    messages.insert(0, user_msg);
                }
            }
        }

        Ok(LlamaRequest {
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
            stop: request.borrow_stop_sequences(),
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct LlamaUsage {
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<LlamaUsage> for Usage {
    fn from(usage: LlamaUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct LlamaResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub(super) struct LlamaResponseToolCall {
    id: String,
    r#type: LlamaToolType,
    function: LlamaResponseFunctionCall,
}

impl From<LlamaResponseToolCall> for ToolCall {
    fn from(llama_tool_call: LlamaResponseToolCall) -> Self {
        ToolCall {
            id: llama_tool_call.id,
            name: llama_tool_call.function.name,
            arguments: llama_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct LlamaResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_calls: Option<Vec<LlamaResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum LlamaFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

impl From<LlamaFinishReason> for FinishReason {
    fn from(finish_reason: LlamaFinishReason) -> Self {
        match finish_reason {
            LlamaFinishReason::Stop => FinishReason::Stop,
            LlamaFinishReason::Length => FinishReason::Length,
            LlamaFinishReason::ContentFilter => FinishReason::ContentFilter,
            LlamaFinishReason::ToolCalls => FinishReason::ToolCall,
            LlamaFinishReason::FunctionCall => FinishReason::ToolCall,
            LlamaFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct LlamaResponseChoice {
    pub(super) index: u8,
    pub(super) message: LlamaResponseMessage,
    pub(super) finish_reason: LlamaFinishReason,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct LlamaResponse {
    pub(super) choices: Vec<LlamaResponseChoice>,
    pub(super) usage: LlamaUsage,
}

struct LlamaResponseWithMetadata<'a> {
    response: LlamaResponse,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    raw_response: String,
}

impl<'a> TryFrom<LlamaResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: LlamaResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let LlamaResponseWithMetadata {
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
        let LlamaResponseChoice {
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
struct LlamaFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct LlamaToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: LlamaFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct LlamaDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<LlamaToolCallChunk>>,
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
struct LlamaChatChunkChoice {
    delta: LlamaDelta,
    #[serde(default)]
    #[serde(deserialize_with = "empty_string_as_none")]
    finish_reason: Option<LlamaFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct LlamaChatChunk {
    choices: Vec<LlamaChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<LlamaUsage>,
}

/// Maps an Llama chunk to a TensorZero chunk for streaming inferences
fn llama_to_tensorzero_chunk(
    mut chunk: LlamaChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
) -> Result<ProviderInferenceResponseChunk, Error> {
    let raw_message = serde_json::to_string(&chunk).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!(
                "Error parsing response from Llama: {}",
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

    use serde_json::json;

    use crate::{
        inference::types::{FunctionType, RequestMessage},
        providers::test_helpers::{
            MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL, WEATHER_TOOL_CONFIG,
        },
        tool::ToolCallConfig,
    };

    use super::*;

    #[test]
    fn test_handle_llama_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_llama_error(
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
        let forbidden = handle_llama_error(StatusCode::FORBIDDEN, "Forbidden access", PROVIDER_TYPE);
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
        let rate_limit = handle_llama_error(
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
        let server_error = handle_llama_error(
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

    #[test]
    fn test_llama_request_new() {
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

        let llama_request =
            LlamaRequest::new("meta-llama/llama-4-scout-17b-16e-instruct", &basic_request).unwrap();

        assert_eq!(
            llama_request.model,
            "meta-llama/llama-4-scout-17b-16e-instruct"
        );
        assert_eq!(llama_request.messages.len(), 2);
        assert_eq!(llama_request.temperature, Some(0.7));
        assert_eq!(llama_request.max_completion_tokens, Some(100));
        assert_eq!(llama_request.seed, Some(69));
        assert_eq!(llama_request.top_p, Some(0.9));
        assert_eq!(llama_request.presence_penalty, Some(0.1));
        assert_eq!(llama_request.frequency_penalty, Some(0.2));
        assert!(llama_request.stream);
        assert_eq!(llama_request.response_format, Some(LlamaResponseFormat::Text));
        assert!(llama_request.tools.is_none());
        assert_eq!(llama_request.tool_choice, None);
        assert!(llama_request.parallel_tool_calls.is_none());

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

        let llama_request = LlamaRequest::new(
            "meta-llama/llama-4-scout-17b-16e-instruct",
            &request_with_tools,
        )
        .unwrap();

        assert_eq!(
            llama_request.model,
            "meta-llama/llama-4-scout-17b-16e-instruct"
        );
        assert_eq!(llama_request.messages.len(), 2); // We'll add a system message containing Json to fit Llama requirements
        assert_eq!(llama_request.temperature, None);
        assert_eq!(llama_request.max_completion_tokens, None);
        assert_eq!(llama_request.seed, None);
        assert_eq!(llama_request.top_p, None);
        assert_eq!(llama_request.presence_penalty, None);
        assert_eq!(llama_request.frequency_penalty, None);
        assert!(!llama_request.stream);
        assert_eq!(
            llama_request.response_format,
            Some(LlamaResponseFormat::JsonObject { schema: None })
        );
        assert!(llama_request.tools.is_some());
        let tools = llama_request.tools.as_ref().unwrap();
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            llama_request.tool_choice,
            Some(LlamaToolChoice::Specific(SpecificToolChoice {
                r#type: LlamaToolType::Function,
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

        let llama_request = LlamaRequest::new(
            "meta-llama/llama-4-scout-17b-16e-instruct",
            &request_with_tools,
        )
        .unwrap();

        assert_eq!(
            llama_request.model,
            "meta-llama/llama-4-scout-17b-16e-instruct"
        );
        assert_eq!(llama_request.messages.len(), 1);
        assert_eq!(llama_request.temperature, None);
        assert_eq!(llama_request.max_completion_tokens, None);
        assert_eq!(llama_request.seed, None);
        assert!(!llama_request.stream);
        assert_eq!(llama_request.top_p, None);
        assert_eq!(llama_request.presence_penalty, None);
        assert_eq!(llama_request.frequency_penalty, None);
        // Resolves to normal JSON mode since no schema is provided (this shouldn't really happen in practice)
        assert_eq!(
            llama_request.response_format,
            Some(LlamaResponseFormat::JsonObject { schema: None })
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

        let llama_request = LlamaRequest::new(
            "meta-llama/llama-4-scout-17b-16e-instruct",
            &request_with_tools,
        )
        .unwrap();

        assert_eq!(
            llama_request.model,
            "meta-llama/llama-4-scout-17b-16e-instruct"
        );
        assert_eq!(llama_request.messages.len(), 1);
        assert_eq!(llama_request.temperature, None);
        assert_eq!(llama_request.max_completion_tokens, None);
        assert_eq!(llama_request.seed, None);
        assert!(!llama_request.stream);
        assert_eq!(llama_request.top_p, None);
        assert_eq!(llama_request.presence_penalty, None);
        assert_eq!(llama_request.frequency_penalty, None);
        let expected_schema = serde_json::json!({});
        assert_eq!(
            llama_request.response_format,
            Some(LlamaResponseFormat::JsonObject {
                schema: Some(&expected_schema),
            })
        );
    }

    #[test]
    fn test_try_from_llama_response() {
        // Test case 1: Valid response with content
        let valid_response = LlamaResponse {
            choices: vec![LlamaResponseChoice {
                index: 0,
                message: LlamaResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: LlamaFinishReason::Stop,
            }],
            usage: LlamaUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
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

        let request_body = LlamaRequest {
            messages: vec![],
            model: "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(LlamaResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test_response".to_string();
        let result = ProviderInferenceResponse::try_from(LlamaResponseWithMetadata {
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
        assert_eq!(inference_response.usage.input_tokens, 10);
        assert_eq!(inference_response.usage.output_tokens, 20);
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
        let valid_response_with_tools = LlamaResponse {
            choices: vec![LlamaResponseChoice {
                index: 0,
                finish_reason: LlamaFinishReason::ToolCalls,
                message: LlamaResponseMessage {
                    content: None,
                    tool_calls: Some(vec![LlamaResponseToolCall {
                        id: "call1".to_string(),
                        r#type: LlamaToolType::Function,
                        function: LlamaResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
            }],
            usage: LlamaUsage {
                prompt_tokens: 15,
                completion_tokens: 25,
                total_tokens: 40,
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

        let request_body = LlamaRequest {
            messages: vec![],
            model: "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(LlamaResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(LlamaResponseWithMetadata {
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
        assert_eq!(inference_response.usage.input_tokens, 15);
        assert_eq!(inference_response.usage.output_tokens, 25);
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
        let invalid_response_no_choices = LlamaResponse {
            choices: vec![],
            usage: LlamaUsage {
                prompt_tokens: 5,
                completion_tokens: 0,
                total_tokens: 5,
            },
        };
        let request_body = LlamaRequest {
            messages: vec![],
            model: "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(LlamaResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
        };
        let result = ProviderInferenceResponse::try_from(LlamaResponseWithMetadata {
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
        let invalid_response_multiple_choices = LlamaResponse {
            choices: vec![
                LlamaResponseChoice {
                    index: 0,
                    message: LlamaResponseMessage {
                        content: Some("Choice 1".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: LlamaFinishReason::Stop,
                },
                LlamaResponseChoice {
                    index: 1,
                    finish_reason: LlamaFinishReason::Stop,
                    message: LlamaResponseMessage {
                        content: Some("Choice 2".to_string()),
                        tool_calls: None,
                    },
                },
            ],
            usage: LlamaUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
                total_tokens: 20,
            },
        };

        let request_body = LlamaRequest {
            messages: vec![],
            model: "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(LlamaResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
        };
        let result = ProviderInferenceResponse::try_from(LlamaResponseWithMetadata {
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
    fn test_prepare_llama_tools() {
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
        let (tools, tool_choice, parallel_tool_calls) = prepare_llama_tools(&request_with_tools);
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(tools[1].function.name, QUERY_TOOL.name());
        assert_eq!(tools[1].function.parameters, QUERY_TOOL.parameters());
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            LlamaToolChoice::String(LlamaToolChoiceString::Required)
        );
        let parallel_tool_calls = parallel_tool_calls.unwrap();
        assert!(parallel_tool_calls);
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: Some(true),
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
        let (tools, tool_choice, parallel_tool_calls) = prepare_llama_tools(&request_without_tools);
        assert!(tools.is_none());
        assert!(tool_choice.is_none());
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_tensorzero_to_llama_messages() {
        let content_blocks = vec!["Hello".to_string().into()];
        let llama_messages = tensorzero_to_llama_user_messages(&content_blocks).unwrap();
        assert_eq!(llama_messages.len(), 1);
        match &llama_messages[0] {
            LlamaRequestMessage::User(content) => {
                assert_eq!(
                    content.content,
                    &[LlamaContentBlock::Text {
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
        let llama_messages = tensorzero_to_llama_user_messages(&content_blocks).unwrap();
        assert_eq!(llama_messages.len(), 1);
        match &llama_messages[0] {
            LlamaRequestMessage::User(content) => {
                assert_eq!(
                    content.content,
                    vec![
                        LlamaContentBlock::Text {
                            text: "Hello".into()
                        },
                        LlamaContentBlock::Text {
                            text: "How are you?".into()
                        }
                    ]
                );
            }
            _ => panic!("Expected a user message"),
        }

        // User message with one string and one tool call block
        // Since user messages in Llama land can't contain tool calls (nor should they honestly),
        // We split the tool call out into a separate assistant message
        let tool_block = ContentBlock::ToolCall(ToolCall {
            id: "call1".to_string(),
            name: "test_function".to_string(),
            arguments: "{}".to_string(),
        });
        let content_blocks = vec!["Hello".to_string().into(), tool_block];
        let llama_messages = tensorzero_to_llama_assistant_messages(&content_blocks).unwrap();
        assert_eq!(llama_messages.len(), 1);
        match &llama_messages[0] {
            LlamaRequestMessage::Assistant(content) => {
                assert_eq!(
                    content.content,
                    Some(vec![LlamaContentBlock::Text {
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
    fn test_llama_to_tensorzero_chunk() {
        let chunk = LlamaChatChunk {
            choices: vec![LlamaChatChunkChoice {
                delta: LlamaDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(LlamaFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let message =
            llama_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
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
        let chunk = LlamaChatChunk {
            choices: vec![LlamaChatChunkChoice {
                finish_reason: Some(LlamaFinishReason::ToolCalls),
                delta: LlamaDelta {
                    content: None,
                    tool_calls: Some(vec![LlamaToolCallChunk {
                        index: 0,
                        id: None,
                        function: LlamaFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message =
            llama_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
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
        let chunk = LlamaChatChunk {
            choices: vec![LlamaChatChunkChoice {
                finish_reason: None,
                delta: LlamaDelta {
                    content: None,
                    tool_calls: Some(vec![LlamaToolCallChunk {
                        index: 1,
                        id: None,
                        function: LlamaFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let error =
            llama_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
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
        let chunk = LlamaChatChunk {
            choices: vec![LlamaChatChunkChoice {
                finish_reason: Some(LlamaFinishReason::Stop),
                delta: LlamaDelta {
                    content: None,
                    tool_calls: Some(vec![LlamaToolCallChunk {
                        index: 1,
                        id: Some("id2".to_string()),
                        function: LlamaFunctionCallChunk {
                            name: Some("name2".to_string()),
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message =
            llama_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
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
        let chunk = LlamaChatChunk {
            choices: vec![],
            usage: Some(LlamaUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };
        let message =
            llama_to_tensorzero_chunk(chunk.clone(), Duration::from_millis(50), &mut tool_call_ids)
                .unwrap();
        assert_eq!(message.content, vec![]);
        assert_eq!(
            message.usage,
            Some(Usage {
                input_tokens: 10,
                output_tokens: 20,
            })
        );
    }

    #[test]
    fn test_new_llama_response_format() {
        // Test JSON mode On
        let json_mode = ModelInferenceRequestJsonMode::On;
        let output_schema = None;
        let format = LlamaResponseFormat::new(&json_mode, output_schema);
        assert_eq!(format, LlamaResponseFormat::JsonObject { schema: None });

        // Test JSON mode Off
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let format = LlamaResponseFormat::new(&json_mode, output_schema);
        assert_eq!(format, LlamaResponseFormat::Text);

        // Test JSON mode Strict with no schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let format = LlamaResponseFormat::new(&json_mode, output_schema);
        assert_eq!(format, LlamaResponseFormat::JsonObject { schema: None });

        // Test JSON mode Strict with schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"type": "string"}
            }
        });
        let output_schema = Some(&json_schema);
        let format = LlamaResponseFormat::new(&json_mode, output_schema);
        assert_eq!(
            format,
            LlamaResponseFormat::JsonObject {
                schema: output_schema
            }
        );
    }

    #[test]
    fn test_tensorzero_to_llama_system_message() {
        // Test Case 1: system is None, json_mode is Off
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages: Vec<LlamaRequestMessage> = vec![];
        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert_eq!(result, None);

        // Test Case 2: system is Some, json_mode is On, messages contain "json"
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            LlamaRequestMessage::User(LlamaUserRequestMessage {
                content: vec![LlamaContentBlock::Text {
                    text: "Please respond in JSON format.".into(),
                }],
            }),
            LlamaRequestMessage::Assistant(LlamaAssistantRequestMessage {
                content: Some(vec![LlamaContentBlock::Text {
                    text: "Sure, here is the data.".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(LlamaRequestMessage::System(LlamaSystemRequestMessage {
            content: Cow::Borrowed("System instructions"),
        }));
        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 3: system is Some, json_mode is On, messages do not contain "json"
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            LlamaRequestMessage::User(LlamaUserRequestMessage {
                content: vec![LlamaContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            LlamaRequestMessage::Assistant(LlamaAssistantRequestMessage {
                content: Some(vec![LlamaContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected_content = "Respond using JSON.\n\nSystem instructions".to_string();
        let expected = Some(LlamaRequestMessage::System(LlamaSystemRequestMessage {
            content: Cow::Owned(expected_content),
        }));
        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 4: system is Some, json_mode is Off
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages = vec![
            LlamaRequestMessage::User(LlamaUserRequestMessage {
                content: vec![LlamaContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            LlamaRequestMessage::Assistant(LlamaAssistantRequestMessage {
                content: Some(vec![LlamaContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(LlamaRequestMessage::System(LlamaSystemRequestMessage {
            content: Cow::Borrowed("System instructions"),
        }));
        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 5: system is Some, json_mode is Strict
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let messages = vec![
            LlamaRequestMessage::User(LlamaUserRequestMessage {
                content: vec![LlamaContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            LlamaRequestMessage::Assistant(LlamaAssistantRequestMessage {
                content: Some(vec![LlamaContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(LlamaRequestMessage::System(LlamaSystemRequestMessage {
            content: Cow::Borrowed("System instructions"),
        }));
        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 6: system contains "json", json_mode is On
        let system = Some("Respond using JSON.\n\nSystem instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![LlamaRequestMessage::User(LlamaUserRequestMessage {
            content: vec![LlamaContentBlock::Text {
                text: "Hello, how are you?".into(),
            }],
        })];
        let expected = Some(LlamaRequestMessage::System(LlamaSystemRequestMessage {
            content: Cow::Borrowed("Respond using JSON.\n\nSystem instructions"),
        }));
        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 7: system is None, json_mode is On
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            LlamaRequestMessage::User(LlamaUserRequestMessage {
                content: vec![LlamaContentBlock::Text {
                    text: "Tell me a joke.".into(),
                }],
            }),
            LlamaRequestMessage::Assistant(LlamaAssistantRequestMessage {
                content: Some(vec![LlamaContentBlock::Text {
                    text: "Sure, here's one for you.".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(LlamaRequestMessage::System(LlamaSystemRequestMessage {
            content: Cow::Owned("Respond using JSON.".to_string()),
        }));
        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 8: system is None, json_mode is Strict
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let messages = vec![
            LlamaRequestMessage::User(LlamaUserRequestMessage {
                content: vec![LlamaContentBlock::Text {
                    text: "Provide a summary of the news.".into(),
                }],
            }),
            LlamaRequestMessage::Assistant(LlamaAssistantRequestMessage {
                content: Some(vec![LlamaContentBlock::Text {
                    text: "Here's the summary.".into(),
                }]),
                tool_calls: None,
            }),
        ];

        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert!(result.is_none());

        // Test Case 9: system is None, json_mode is On, with empty messages
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages: Vec<LlamaRequestMessage> = vec![];
        let expected = Some(LlamaRequestMessage::System(LlamaSystemRequestMessage {
            content: Cow::Owned("Respond using JSON.".to_string()),
        }));
        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 10: system is None, json_mode is Off, with messages containing "json"
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages = vec![LlamaRequestMessage::User(LlamaUserRequestMessage {
            content: vec![LlamaContentBlock::Text {
                text: "Please include JSON in your response.".into(),
            }],
        })];
        let expected = None;
        let result = tensorzero_to_llama_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_try_from_llama_credentials() {
        // Test Static credentials
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = LlamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, LlamaCredentials::Static(_)));

        // Test Dynamic credentials
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = LlamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, LlamaCredentials::Dynamic(_)));

        // Test None credentials
        let generic = Credential::None;
        let creds = LlamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, LlamaCredentials::None));

        // Test Missing credentials
        let generic = Credential::Missing;
        let creds = LlamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, LlamaCredentials::None));

        // Test invalid credential type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = LlamaCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_serialize_user_messages() {
        // Test that a single message is serialized as 'content: string'
        let message = LlamaUserRequestMessage {
            content: vec![LlamaContentBlock::Text {
                text: "My single message".into(),
            }],
        };
        let serialized = serde_json::to_string(&message).unwrap();
        assert_eq!(serialized, r#"{"content":"My single message"}"#);

        // Test that a multiple messages are serialized as an array of content blocks
        let message = LlamaUserRequestMessage {
            content: vec![
                LlamaContentBlock::Text {
                    text: "My first message".into(),
                },
                LlamaContentBlock::Text {
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
}
