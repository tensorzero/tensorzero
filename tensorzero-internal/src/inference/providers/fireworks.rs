use std::{borrow::Cow, sync::OnceLock};

use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use crate::{
    cache::ModelProviderRequest,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    inference::types::{
        batch::{BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse},
        ContentBlockChunk, ContentBlockOutput, FinishReason, Latency, ModelInferenceRequest,
        ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
        ProviderInferenceResponse, ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
        ProviderInferenceResponseStreamInner, Text, TextChunk, Thought, ThoughtChunk,
    },
    model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider},
    tool::{ToolCall, ToolCallChunk},
};

use super::{
    helpers::inject_extra_request_data,
    helpers_thinking_block::{process_think_blocks, ThinkingState},
    openai::{
        get_chat_url, handle_openai_error, prepare_openai_tools, tensorzero_to_openai_messages,
        OpenAIFunction, OpenAIRequestMessage, OpenAISystemRequestMessage, OpenAITool,
        OpenAIToolChoice, OpenAIToolType, OpenAIUsage,
    },
    provider_trait::InferenceProvider,
};

lazy_static! {
    static ref FIREWORKS_API_BASE: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.fireworks.ai/inference/v1/")
            .expect("Failed to parse FIREWORKS_API_BASE")
    };
}

const PROVIDER_NAME: &str = "Fireworks";
const PROVIDER_TYPE: &str = "fireworks";

#[derive(Debug)]
pub struct FireworksProvider {
    model_name: String,
    credentials: FireworksCredentials,
    parse_think_blocks: bool,
}

static DEFAULT_CREDENTIALS: OnceLock<FireworksCredentials> = OnceLock::new();

impl FireworksProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
        parse_think_blocks: bool,
    ) -> Result<Self, Error> {
        let credentials = build_creds_caching_default(
            api_key_location,
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;
        Ok(FireworksProvider {
            model_name,
            credentials,
            parse_think_blocks,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

pub fn default_parse_think_blocks() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
pub enum FireworksCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for FireworksCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(FireworksCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(FireworksCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(FireworksCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Fireworks provider".to_string(),
            })),
        }
    }
}

impl FireworksCredentials {
    fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            FireworksCredentials::Static(api_key) => Ok(api_key),
            FireworksCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                })
            }
            &FireworksCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            }
            .into()),
        }
    }
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("FIREWORKS_API_KEY".to_string())
}

/// Key differences between Fireworks and OpenAI inference:
/// - Fireworks allows you to specify output format in JSON mode
/// - Fireworks automatically returns usage in streaming inference, we don't have to ask for it
/// - Fireworks allows you to auto-truncate requests that have too many tokens
///   (there are 2 ways to do it, we have the default of auto-truncation to the max window size)
impl InferenceProvider for FireworksProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        api_key: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let mut request_body =
            serde_json::to_value(FireworksRequest::new(&self.model_name, request)?).map_err(
                |e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing Fireworks request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                },
            )?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let request_url = get_chat_url(&FIREWORKS_API_BASE)?;
        let start_time = Instant::now();
        let api_key = self.credentials.get_api_key(api_key)?;
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .headers(headers)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!(
                        "Error sending request to Fireworks: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
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
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

            let response: FireworksResponse = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {raw_response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            Ok(FireworksResponseWithMetadata {
                response,
                latency,
                request: request_body,
                generic_request: request,
                raw_response,
                parse_think_blocks: self.parse_think_blocks,
            }
            .try_into()?)
        } else {
            Err(handle_openai_error(
                &serde_json::to_string(&request_body).unwrap_or_default(),
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                        raw_response: None,
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
        api_key: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let mut request_body =
            serde_json::to_value(FireworksRequest::new(&self.model_name, request)?).map_err(
                |e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing Fireworks request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                },
            )?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error serializing request body: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: None,
            })
        })?;
        let request_url = get_chat_url(&FIREWORKS_API_BASE)?;
        let api_key = self.credentials.get_api_key(api_key)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .headers(headers)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!(
                        "Error sending request to Fireworks: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
        // Use our own stream implementation to handle thinking blocks
        let stream = stream_fireworks(event_source, start_time, self.parse_think_blocks).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Fireworks".to_string(),
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

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum FireworksResponseFormat<'a> {
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<&'a Value>, // the desired JSON schema
    },
    #[default]
    Text,
}

/// This struct defines the supported parameters for the Fireworks inference API
/// See the [Fireworks API documentation](https://docs.fireworks.ai/api-reference/post-chatcompletions)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, service_tier, stop, user,
/// or context_length_exceeded_behavior.
/// NOTE: Fireworks does not support seed.
#[derive(Debug, Serialize)]
struct FireworksRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
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
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<FireworksResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<FireworksTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
}

impl<'a> FireworksRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<FireworksRequest<'a>, Error> {
        // NB: Fireworks will throw an error if you give FireworksResponseFormat::Text and then also include tools.
        // So we just don't include it as Text is the same as None anyway.
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(FireworksResponseFormat::JsonObject {
                    schema: request.output_schema,
                })
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_fireworks_messages(request)?;
        let (tools, tool_choice, _) = prepare_openai_tools(request);
        let tools = tools.map(|t| t.into_iter().map(|tool| tool.into()).collect());

        Ok(FireworksRequest {
            messages,
            model,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
        })
    }
}

fn prepare_fireworks_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in request.messages.iter() {
        messages.extend(tensorzero_to_openai_messages(message)?);
    }
    if let Some(system_msg) = tensorzero_to_fireworks_system_message(request.system.as_deref()) {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

fn tensorzero_to_fireworks_system_message(
    system: Option<&str>,
) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

#[derive(Debug, PartialEq, Serialize)]
struct FireworksTool<'a> {
    r#type: OpenAIToolType,
    function: OpenAIFunction<'a>,
}

impl<'a> From<OpenAITool<'a>> for FireworksTool<'a> {
    fn from(tool: OpenAITool<'a>) -> Self {
        FireworksTool {
            r#type: tool.r#type,
            function: tool.function,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct FireworksResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct FireworksResponseToolCall {
    id: String,
    r#type: OpenAIToolType,
    function: FireworksResponseFunctionCall,
}

impl From<FireworksResponseToolCall> for ToolCall {
    fn from(fireworks_tool_call: FireworksResponseToolCall) -> Self {
        ToolCall {
            id: fireworks_tool_call.id,
            name: fireworks_tool_call.function.name,
            arguments: fireworks_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct FireworksResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<FireworksResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
enum FireworksFinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    #[serde(other)]
    Unknown,
}

impl From<FireworksFinishReason> for FinishReason {
    fn from(reason: FireworksFinishReason) -> Self {
        match reason {
            FireworksFinishReason::Stop => FinishReason::Stop,
            FireworksFinishReason::Length => FinishReason::Length,
            FireworksFinishReason::ToolCalls => FinishReason::ToolCall,
            FireworksFinishReason::ContentFilter => FinishReason::ContentFilter,
            FireworksFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct FireworksResponseChoice {
    index: u8,
    message: FireworksResponseMessage,
    finish_reason: Option<FireworksFinishReason>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct FireworksResponse {
    choices: Vec<FireworksResponseChoice>,
    usage: OpenAIUsage,
}

// Streaming-specific structs
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct FireworksFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct FireworksToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    function: FireworksFunctionCallChunk,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct FireworksDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<FireworksToolCallChunk>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct FireworksChatChunkChoice {
    delta: FireworksDelta,
    #[serde(default)]
    finish_reason: Option<FireworksFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct FireworksChatChunk {
    choices: Vec<FireworksChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAIUsage>,
}

/// Streams the Fireworks response events and converts them into ProviderInferenceResponseChunks
/// This function handles parsing and processing of thinking blocks with proper state tracking
fn stream_fireworks(
    mut event_source: EventSource,
    start_time: Instant,
    parse_think_blocks: bool,
) -> ProviderInferenceResponseStreamInner {
    let mut tool_call_ids = Vec::new();
    let mut tool_call_names = Vec::new();
    let mut thinking_state = ThinkingState::Normal;
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    let message = e.to_string();
                    let mut raw_response = None;
                    if let reqwest_eventsource::Error::InvalidStatusCode(_, resp) = e {
                        raw_response = resp.text().await.ok();
                    }
                    yield Err(ErrorDetails::InferenceServer {
                        message,
                        raw_request: None,
                        raw_response,
                        provider_type: PROVIDER_TYPE.to_string(),
                    }.into());
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<FireworksChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!("Error parsing chunk. Error: {e}"),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            fireworks_to_tensorzero_chunk(message.data, d, latency, &mut tool_call_ids, &mut tool_call_names, &mut thinking_state, parse_think_blocks)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    })
}

/// Maps a Fireworks chunk to a TensorZero chunk for streaming inferences
///
/// This function handles the conversion of Fireworks chat chunks into TensorZero chunks.
/// It processes the content and tool calls from the Fireworks response, updating the tool call IDs and names.
/// If parsing think blocks is enabled, it also processes the thinking state and extracts reasoning.
fn fireworks_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: FireworksChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    tool_call_names: &mut Vec<String>,
    thinking_state: &mut ThinkingState,
    parse_think_blocks: bool,
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
    let usage = chunk.usage.map(|u| u.into());
    let mut finish_reason = None;
    let mut content = vec![];
    if let Some(choice) = chunk.choices.pop() {
        if let Some(reason) = choice.finish_reason {
            finish_reason = Some(reason.into());
        }
        if let Some(text) = choice.delta.content {
            if parse_think_blocks {
                if !thinking_state.update(&text, PROVIDER_TYPE)? {
                    match thinking_state {
                        ThinkingState::Normal | ThinkingState::Finished => {
                            content.push(ContentBlockChunk::Text(TextChunk {
                                text: text.to_string(),
                                id: thinking_state.get_id(),
                            }));
                        }
                        ThinkingState::Thinking => {
                            content.push(ContentBlockChunk::Thought(ThoughtChunk {
                                text: Some(text.to_string()),
                                signature: None,
                                id: thinking_state.get_id(),
                            }));
                        }
                    }
                }
            } else {
                // Just add the text verbatim if we're not parsing think blocks.
                content.push(ContentBlockChunk::Text(TextChunk {
                    text: text.to_string(),
                    id: "0".to_string(),
                }));
            }
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
                let name = match tool_call.function.name {
                    Some(name) => {
                        tool_call_names.push(name.clone());
                        name
                    }
                    None => {
                        tool_call_names
                            .get(index as usize)
                            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                                message: "Tool call index out of bounds (meaning we haven't seen this many names in the stream)".to_string(),
                                raw_request: None,
                                raw_response: None,
                                provider_type: PROVIDER_TYPE.to_string(),
                            }))?
                            .clone()
                    }
                };
                content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                    id,
                    raw_name: name,
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

struct FireworksResponseWithMetadata<'a> {
    response: FireworksResponse,
    raw_response: String,
    latency: Latency,
    request: serde_json::Value,
    generic_request: &'a ModelInferenceRequest<'a>,
    parse_think_blocks: bool,
}

impl<'a> TryFrom<FireworksResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: FireworksResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let FireworksResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
            raw_response,
            parse_think_blocks,
        } = value;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
            }
            .into());
        }
        let usage = response.usage.into();
        let FireworksResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
            }
            ))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(raw_text) = message.content {
            let (clean_text, extracted_reasoning) =
                process_think_blocks(&raw_text, parse_think_blocks, PROVIDER_TYPE)?;
            if let Some(reasoning) = extracted_reasoning {
                content.push(ContentBlockOutput::Thought(Thought {
                    text: reasoning,
                    signature: None,
                }));
            }
            if !clean_text.is_empty() {
                content.push(ContentBlockOutput::Text(Text { text: clean_text }));
            }
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        }
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing request body as JSON: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response,
                usage,
                latency,
                finish_reason: finish_reason.map(|r| r.into()),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;

    use crate::inference::providers::openai::{OpenAIToolType, OpenAIUsage};
    use crate::inference::providers::openai::{SpecificToolChoice, SpecificToolFunction};
    use crate::inference::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::types::{FunctionType, RequestMessage, Role, Usage};

    #[test]
    fn test_fireworks_response_with_thinking_blocks() {
        let test_response_with_thinking = "Hello <think>This is reasoning</think> world";
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

        // Create a valid response with thinking blocks in the content
        let valid_response = FireworksResponse {
            choices: vec![FireworksResponseChoice {
                index: 0,
                finish_reason: Some(FireworksFinishReason::Stop),
                message: FireworksResponseMessage {
                    content: Some(test_response_with_thinking.to_string()),
                    tool_calls: None,
                },
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        };

        // Test with parsing enabled
        let fireworks_response_with_metadata = FireworksResponseWithMetadata {
            response: valid_response.clone(),
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            request: serde_json::to_value(
                FireworksRequest::new("test-model", &generic_request).unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: true,
        };

        let inference_response: ProviderInferenceResponse =
            fireworks_response_with_metadata.try_into().unwrap();

        // Should have two content blocks: thought and text
        assert_eq!(inference_response.output.len(), 2);

        // First block should be a thought
        match &inference_response.output[0] {
            ContentBlockOutput::Thought(thought) => {
                assert_eq!(thought.text, "This is reasoning");
                assert_eq!(thought.signature, None);
            }
            _ => panic!("Expected a thought block"),
        }

        // Second block should be text
        match &inference_response.output[1] {
            ContentBlockOutput::Text(text) => {
                assert_eq!(text.text, "Hello  world");
            }
            _ => panic!("Expected a text block"),
        }

        // Test with parsing disabled
        let fireworks_response_with_metadata = FireworksResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            request: serde_json::to_value(
                FireworksRequest::new("test-model", &generic_request).unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: false,
        };

        let inference_response: ProviderInferenceResponse =
            fireworks_response_with_metadata.try_into().unwrap();

        // Should have only one content block with the original text
        assert_eq!(inference_response.output.len(), 1);

        // Block should be text with thinking tags preserved
        match &inference_response.output[0] {
            ContentBlockOutput::Text(text) => {
                assert_eq!(text.text, test_response_with_thinking);
            }
            _ => panic!("Expected a text block"),
        }
    }

    #[test]
    fn test_fireworks_request_new() {
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
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let fireworks_request =
            FireworksRequest::new("accounts/fireworks/models/llama-v3-8b", &request_with_tools)
                .unwrap();

        assert_eq!(
            fireworks_request.model,
            "accounts/fireworks/models/llama-v3-8b"
        );
        assert_eq!(fireworks_request.messages.len(), 1);
        assert_eq!(fireworks_request.temperature, Some(0.5));
        assert_eq!(fireworks_request.max_tokens, Some(100));
        assert_eq!(fireworks_request.top_p, Some(0.9));
        assert_eq!(fireworks_request.presence_penalty, Some(0.1));
        assert_eq!(fireworks_request.frequency_penalty, Some(0.2));
        assert!(!fireworks_request.stream);
        assert_eq!(
            fireworks_request.response_format,
            Some(FireworksResponseFormat::JsonObject {
                schema: request_with_tools.output_schema,
            })
        );
        assert!(fireworks_request.tools.is_some());
        let tools = fireworks_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            fireworks_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );
    }

    #[test]
    fn test_fireworks_api_base() {
        assert_eq!(
            FIREWORKS_API_BASE.as_str(),
            "https://api.fireworks.ai/inference/v1/"
        );
    }

    #[test]
    fn test_credential_to_fireworks_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = FireworksCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, FireworksCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = FireworksCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, FireworksCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = FireworksCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, FireworksCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = FireworksCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_fireworks_response_with_metadata_try_into() {
        let valid_response = FireworksResponse {
            choices: vec![FireworksResponseChoice {
                index: 0,
                finish_reason: Some(FireworksFinishReason::Stop),
                message: FireworksResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
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
        let fireworks_response_with_metadata = FireworksResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            request: serde_json::to_value(
                FireworksRequest::new("test-model", &generic_request).unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: false,
        };
        let inference_response: ProviderInferenceResponse =
            fireworks_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 1);
        assert_eq!(
            inference_response.output[0],
            "Hello, world!".to_string().into()
        );
        assert_eq!(inference_response.raw_response, "test_response");
        assert_eq!(inference_response.usage.input_tokens, 10);
        assert_eq!(inference_response.usage.output_tokens, 20);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_secs(0)
            }
        );
    }

    #[test]
    fn test_fireworks_to_tensorzero_chunk() {
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(FireworksFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let mut tool_call_names = vec!["name1".to_string()];
        let mut thinking_state = ThinkingState::Normal;
        let message = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();

        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "Hello".to_string(),
                id: "0".to_string(),
            })]
        );
        assert_eq!(message.finish_reason, Some(FinishReason::Stop));

        // Test what an intermediate tool chunk should look like
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: None,
                    tool_calls: Some(vec![FireworksToolCallChunk {
                        index: 0,
                        id: None,
                        function: FireworksFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(FireworksFinishReason::ToolCalls),
            }],
            usage: None,
        };
        let message = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();

        assert_eq!(
            message.content,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "id1".to_string(),
                raw_name: "name1".to_string(),
                raw_arguments: "{\"hello\":\"world\"}".to_string(),
            })]
        );
        assert_eq!(message.finish_reason, Some(FinishReason::ToolCall));

        // Test a chunk with no choices and only usage
        let chunk = FireworksChatChunk {
            choices: vec![],
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };
        let message = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
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
    fn test_fireworks_to_tensorzero_chunk_thinking() {
        // Test that the streaming function correctly handles thinking blocks
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("<think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let mut tool_call_ids = Vec::new();
        let mut tool_call_names = Vec::new();
        let mut thinking_state = ThinkingState::Normal;

        // With parsing enabled
        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should transition to Thinking state
        assert!(matches!(thinking_state, ThinkingState::Thinking));
        // No content should be added for the opening tag
        assert!(result.content.is_empty());

        // Now process some thinking content
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("reasoning".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should still be in Thinking state
        assert!(matches!(thinking_state, ThinkingState::Thinking));
        // Content should be added as thought
        assert_eq!(result.content.len(), 1);
        assert!(matches!(result.content[0], ContentBlockChunk::Thought(_)));
        if let ContentBlockChunk::Thought(thought) = &result.content[0] {
            assert_eq!(thought.text, Some("reasoning".to_string()));
            assert_eq!(thought.id, "1");
        }

        // Close the thinking block
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("</think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should transition to Finished state
        assert!(matches!(thinking_state, ThinkingState::Finished));
        // No content should be added for the closing tag
        assert!(result.content.is_empty());

        // After closing, regular text should be treated as text content
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("Final answer".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should remain in Finished state
        assert!(matches!(thinking_state, ThinkingState::Finished));
        // Content should be added as text
        assert_eq!(result.content.len(), 1);
        assert!(matches!(result.content[0], ContentBlockChunk::Text(_)));
        if let ContentBlockChunk::Text(text) = &result.content[0] {
            assert_eq!(text.text, "Final answer");
            assert_eq!(text.id, "2");
        }
    }

    #[test]
    fn test_fireworks_to_tensorzero_chunk_without_think_parsing() {
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("Hello <think>should not parse</think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(FireworksFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec![];
        let mut tool_call_names = vec![];
        let mut thinking_state = ThinkingState::Normal;
        let message = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            false,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "Hello <think>should not parse</think>".to_string(),
                id: "0".to_string(),
            })]
        );
    }

    #[test]
    fn test_fireworks_stream_tool_call_handling() {
        // Test new tool call with ID and name
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: None,
                    tool_calls: Some(vec![FireworksToolCallChunk {
                        index: 0,
                        id: Some("new_id".to_string()),
                        function: FireworksFunctionCallChunk {
                            name: Some("new_name".to_string()),
                            arguments: Some("{\"param\":\"value\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let mut tool_call_ids = Vec::new();
        let mut tool_call_names = Vec::new();
        let mut thinking_state = ThinkingState::Normal;

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should add the tool call to the state and the result
        assert_eq!(tool_call_ids, vec!["new_id"]);
        assert_eq!(tool_call_names, vec!["new_name"]);
        assert_eq!(result.content.len(), 1);

        if let ContentBlockChunk::ToolCall(tool_call) = &result.content[0] {
            assert_eq!(tool_call.id, "new_id");
            assert_eq!(tool_call.raw_name, "new_name");
            assert_eq!(tool_call.raw_arguments, "{\"param\":\"value\"}");
        } else {
            panic!("Expected a tool call chunk");
        }

        // Test continuation of a tool call (id and name already known)
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: None,
                    tool_calls: Some(vec![FireworksToolCallChunk {
                        index: 0,
                        id: None,
                        function: FireworksFunctionCallChunk {
                            name: None,
                            arguments: Some(",\"more\":\"data\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(FireworksFinishReason::ToolCalls),
            }],
            usage: None,
        };

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should reference the existing ID and name
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.finish_reason, Some(FinishReason::ToolCall));

        if let ContentBlockChunk::ToolCall(tool_call) = &result.content[0] {
            assert_eq!(tool_call.id, "new_id");
            assert_eq!(tool_call.raw_name, "new_name");
            assert_eq!(tool_call.raw_arguments, ",\"more\":\"data\"}");
        } else {
            panic!("Expected a tool call chunk");
        }
    }
}
