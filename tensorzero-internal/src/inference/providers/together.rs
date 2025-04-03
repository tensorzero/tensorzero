use std::{borrow::Cow, sync::OnceLock, time::Duration};

use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use crate::cache::ModelProviderRequest;
use crate::inference::types::{
    FinishReason, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseArgs,
};
use crate::model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider};
use crate::tool::ToolChoice;
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::types::{
        batch::{BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse},
        ContentBlockChunk, ContentBlockOutput, ProviderInferenceResponseChunk,
        ProviderInferenceResponseStreamInner, Text, TextChunk, Thought, ThoughtChunk,
    },
    tool::{ToolCall, ToolCallChunk},
};

use super::helpers::inject_extra_request_data;
use super::{
    openai::{
        get_chat_url, handle_openai_error, prepare_openai_tools, tensorzero_to_openai_messages,
        OpenAIRequestMessage, OpenAISystemRequestMessage, OpenAITool, OpenAIToolChoice,
        OpenAIToolType, OpenAIUsage,
    },
    provider_trait::InferenceProvider,
};

lazy_static! {
    static ref TOGETHER_API_BASE: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.together.xyz/v1").expect("Failed to parse TOGETHER_API_BASE")
    };
}

const PROVIDER_NAME: &str = "Together";
const PROVIDER_TYPE: &str = "together";

#[derive(Debug)]
pub struct TogetherProvider {
    model_name: String,
    credentials: TogetherCredentials,
    parse_think_blocks: bool,
}

pub fn default_parse_think_blocks() -> bool {
    true
}

static DEFAULT_CREDENTIALS: OnceLock<TogetherCredentials> = OnceLock::new();

impl TogetherProvider {
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
        Ok(TogetherProvider {
            model_name,
            credentials,
            parse_think_blocks,
        })
    }
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("TOGETHER_API_KEY".to_string())
}

#[derive(Clone, Debug)]
pub enum TogetherCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for TogetherCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(TogetherCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(TogetherCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(TogetherCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Together provider".to_string(),
            })),
        }
    }
}

impl TogetherCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Cow<'a, SecretString>, Error> {
        match self {
            TogetherCredentials::Static(api_key) => Ok(Cow::Owned(api_key.clone())),
            TogetherCredentials::Dynamic(key_name) => {
                Ok(Cow::Borrowed(dynamic_api_keys.get(key_name).ok_or_else(
                    || ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    },
                )?))
            }
            TogetherCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            })?,
        }
    }
}

impl InferenceProvider for TogetherProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let mut request_body =
            serde_json::to_value(TogetherRequest::new(&self.model_name, request)?).map_err(
                |e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Error serializing Together request: {e}"),
                    })
                },
            )?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let request_url = get_chat_url(&TOGETHER_API_BASE)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
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
                    message: format!("{e}"),
                    status_code: Some(e.status().unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            Ok(TogetherResponseWithMetadata {
                response,
                raw_response,
                latency: Latency::NonStreaming {
                    response_time: start_time.elapsed(),
                },
                request: request_body,
                generic_request: request,
                parse_think_blocks: self.parse_think_blocks,
            }
            .try_into()?)
        } else {
            let status = res.status();
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing error response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            Err(handle_openai_error(status, &raw_response, PROVIDER_TYPE))
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
        let mut request_body =
            serde_json::to_value(TogetherRequest::new(&self.model_name, request)?).map_err(
                |e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Error serializing request: {e}"),
                    })
                },
            )?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = get_chat_url(&TOGETHER_API_BASE)?;
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
                    message: format!("Error sending request to Together: {e}"),
                    status_code: None,
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
        let stream = stream_together(event_source, start_time, self.parse_think_blocks).peekable();
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
            provider_type: "GCP Vertex Gemini".to_string(),
        }
        .into())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum TogetherResponseFormat<'a> {
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<&'a Value>, // the desired JSON schema
    },
}

/// This struct defines the supported parameters for the Together inference API
/// See the [Together API documentation](https://docs.together.ai/docs/chat-overview)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, seed, service_tier, stop, user,
/// or context_length_exceeded_behavior
#[derive(Debug, Serialize)]
struct TogetherRequest<'a> {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<TogetherResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> TogetherRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<TogetherRequest<'a>, Error> {
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(TogetherResponseFormat::JsonObject {
                    schema: request.output_schema,
                })
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_together_messages(request)?;

        // NOTE: Together AI doesn't seem to support `tool_choice="none"`, so we simply don't include the `tools` field if that's the case
        let tool_choice = request
            .tool_config
            .as_ref()
            .map(|config| &config.tool_choice);

        let (tools, tool_choice, parallel_tool_calls) = match tool_choice {
            Some(&ToolChoice::None) => (None, None, None),
            _ => prepare_openai_tools(request),
        };

        Ok(TogetherRequest {
            messages,
            model,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            seed: request.seed,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
        })
    }
}

pub(super) fn prepare_together_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in request.messages.iter() {
        messages.extend(tensorzero_to_openai_messages(message)?);
    }

    if let Some(system_msg) = tensorzero_to_together_system_message(request.system.as_deref()) {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

fn tensorzero_to_together_system_message(system: Option<&str>) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct TogetherResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct TogetherResponseToolCall {
    id: String,
    r#type: OpenAIToolType,
    function: TogetherResponseFunctionCall,
}

impl From<TogetherResponseToolCall> for ToolCall {
    fn from(together_tool_call: TogetherResponseToolCall) -> Self {
        ToolCall {
            id: together_tool_call.id,
            name: together_tool_call.function.name,
            arguments: together_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<TogetherResponseToolCall>>,
}

const THINK_TAG: &str = "<think>";
const THINK_TAG_LEN: usize = THINK_TAG.len();
const END_THINK_TAG: &str = "</think>";
const END_THINK_TAG_LEN: usize = END_THINK_TAG.len();

/// Processes the thinking blocks from a span of text.
/// If parsing is disabled, the text is returned as is.
/// If parsing is enabled, the text is checked for a single thinking block.
/// If there is one, the text is cleaned of the thinking block and the reasoning is returned.
/// If there is no thinking block, the text is returned as is and the reasoning is None.
/// If there is more than one thinking block, an error is returned.
///
/// The function also validates that tags are properly matched - an error is returned
/// if there are mismatched opening/closing tags.
///
/// Returns a tuple of (cleaned_text, optional_reasoning).
/// The reasoning, if present, will have leading/trailing whitespace trimmed.
fn process_think_blocks(text: &str, parse: bool) -> Result<(String, Option<String>), Error> {
    if !parse {
        return Ok((text.to_string(), None));
    }
    let think_count = text.matches(THINK_TAG).count();
    if think_count > 1 {
        return Err(Error::new(ErrorDetails::InferenceServer {
            message: "Multiple thinking blocks found".to_string(),
            raw_request: None,
            raw_response: None,
            provider_type: PROVIDER_TYPE.to_string(),
        }));
    }

    if think_count != text.matches(END_THINK_TAG).count() {
        Err(Error::new(ErrorDetails::InferenceServer {
            message: "Mismatched thinking tags".to_string(),
            raw_request: None,
            raw_response: None,
            provider_type: PROVIDER_TYPE.to_string(),
        }))
    } else if let (Some(start), Some(end)) = (text.find(THINK_TAG), text.find(END_THINK_TAG)) {
        let reasoning = text[start + THINK_TAG_LEN..end].to_string();
        let cleaned = format!("{}{}", &text[..start], &text[end + END_THINK_TAG_LEN..]);
        Ok((cleaned, Some(reasoning)))
    } else {
        Ok((text.to_string(), None))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
enum TogetherFinishReason {
    Stop,
    Eos,
    Length,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

impl From<TogetherFinishReason> for FinishReason {
    fn from(finish_reason: TogetherFinishReason) -> Self {
        match finish_reason {
            TogetherFinishReason::Stop => FinishReason::Stop,
            TogetherFinishReason::Eos => FinishReason::Stop,
            TogetherFinishReason::Length => FinishReason::Length,
            TogetherFinishReason::ToolCalls => FinishReason::ToolCall,
            TogetherFinishReason::FunctionCall => FinishReason::ToolCall,
            TogetherFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct TogetherResponseChoice {
    index: u8,
    message: TogetherResponseMessage,
    finish_reason: Option<TogetherFinishReason>,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct TogetherResponse {
    choices: Vec<TogetherResponseChoice>,
    usage: OpenAIUsage,
}

struct TogetherResponseWithMetadata<'a> {
    response: TogetherResponse,
    latency: Latency,
    raw_response: String,
    request: serde_json::Value,
    generic_request: &'a ModelInferenceRequest<'a>,
    parse_think_blocks: bool,
}

impl<'a> TryFrom<TogetherResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: TogetherResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let TogetherResponseWithMetadata {
            mut response,
            latency,
            raw_response,
            request: request_body,
            generic_request,
            parse_think_blocks,
        } = value;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }
        let usage = response.usage.into();
        let TogetherResponseChoice {
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
            }))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(raw_text) = message.content {
            let (clean_text, extracted_reasoning) =
                process_think_blocks(&raw_text, parse_think_blocks)?;
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
                message: format!("Error serializing request body as JSON: {e}"),
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
                raw_response: raw_response.clone(),
                usage,
                latency,
                finish_reason: finish_reason.map(|r| r.into()),
            },
        ))
    }
}

enum ThinkingState {
    Normal,
    Thinking,
    Finished,
}

impl ThinkingState {
    fn get_id(&self) -> String {
        match self {
            ThinkingState::Normal => "0".to_string(),
            ThinkingState::Thinking => "1".to_string(),
            ThinkingState::Finished => "2".to_string(),
        }
    }

    // Returns true if an update was made to the thinking state
    // Returns false if the text is not a thinking block
    // Returns an error if the thinking state is invalid
    fn update(&mut self, text: &str) -> Result<bool, Error> {
        match (&mut *self, text) {
            (ThinkingState::Normal, "<think>") => {
                *self = ThinkingState::Thinking;
                Ok(true)
            }
            (ThinkingState::Normal, "</think>") => Err(Error::new(ErrorDetails::InferenceServer {
                message: "Found </think> while not thinking".to_string(),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })),
            (ThinkingState::Thinking, "</think>") => {
                *self = ThinkingState::Finished;
                Ok(true)
            }
            (ThinkingState::Thinking, "<think>") => {
                Err(Error::new(ErrorDetails::InferenceServer {
                    message: "Found <think> while already thinking".to_string(),
                    raw_request: None,
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                }))
            }
            (ThinkingState::Finished, "<think>") | (ThinkingState::Finished, "</think>") => {
                Err(Error::new(ErrorDetails::InferenceServer {
                    message: "Found thinking tags after thinking finished".to_string(),
                    raw_request: None,
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                }))
            }
            _ => Ok(false),
        }
    }
}

fn stream_together(
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
                        let data: Result<TogetherChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!("Error parsing chunk. Error: {}", e),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            together_to_tensorzero_chunk(d, latency, &mut tool_call_ids, &mut tool_call_names, &mut thinking_state, parse_think_blocks)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    })
}

/// Maps a Together chunk to a TensorZero chunk for streaming inferences
///
/// This function handles the conversion of Together chat chunks into TensorZero chunks.
/// It processes the content and tool calls from the Together response, updating the tool call IDs and names.
/// If parsing think blocks is enabled, it also processes the thinking state and extracts reasoning.
fn together_to_tensorzero_chunk(
    mut chunk: TogetherChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    tool_call_names: &mut Vec<String>,
    thinking_state: &mut ThinkingState,
    parse_think_blocks: bool,
) -> Result<ProviderInferenceResponseChunk, Error> {
    let raw_message = serde_json::to_string(&chunk).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error parsing response from Together: {e}"),
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
    let usage = chunk.usage.map(|u| u.into());
    let mut finish_reason = None;
    let mut content = vec![];
    if let Some(choice) = chunk.choices.pop() {
        if let Some(reason) = choice.finish_reason {
            finish_reason = Some(reason.into());
        }
        if let Some(text) = choice.delta.content {
            if parse_think_blocks {
                if !thinking_state.update(&text)? {
                    match thinking_state {
                        ThinkingState::Normal | ThinkingState::Finished => {
                            content.push(ContentBlockChunk::Text(TextChunk {
                                text,
                                id: thinking_state.get_id(),
                            }));
                        }
                        ThinkingState::Thinking => {
                            content.push(ContentBlockChunk::Thought(ThoughtChunk {
                                text: Some(text),
                                signature: None,
                                id: thinking_state.get_id(),
                            }));
                        }
                    }
                }
            } else {
                // Just add the text verbatim if we're not parsing think blocks.
                content.push(ContentBlockChunk::Text(TextChunk {
                    text,
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the type field
    function: TogetherFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<TogetherToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherChatChunkChoice {
    delta: TogetherDelta,
    finish_reason: Option<TogetherFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherChatChunk {
    choices: Vec<TogetherChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAIUsage>,
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;

    use crate::inference::providers::openai::{
        OpenAIToolType, OpenAIUsage, SpecificToolChoice, SpecificToolFunction,
    };
    use crate::inference::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::types::{FunctionType, RequestMessage, Role, Usage};

    #[test]
    fn test_together_request_new() {
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
            frequency_penalty: Some(0.1),
            max_tokens: Some(100),
            seed: Some(69),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let together_request =
            TogetherRequest::new("togethercomputer/llama-v3-8b", &request_with_tools).unwrap();

        assert_eq!(together_request.model, "togethercomputer/llama-v3-8b");
        assert_eq!(together_request.messages.len(), 1);
        assert_eq!(together_request.temperature, Some(0.5));
        assert_eq!(together_request.top_p, Some(0.9));
        assert_eq!(together_request.presence_penalty, Some(0.1));
        assert_eq!(together_request.frequency_penalty, Some(0.1));
        assert_eq!(together_request.max_tokens, Some(100));
        assert_eq!(together_request.seed, Some(69));
        assert!(!together_request.stream);
        let tools = together_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            together_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );
        assert_eq!(together_request.parallel_tool_calls, None);
    }

    #[test]
    fn test_together_api_base() {
        assert_eq!(TOGETHER_API_BASE.as_str(), "https://api.together.xyz/v1");
    }
    #[test]
    fn test_credential_to_together_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: TogetherCredentials = TogetherCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, TogetherCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = TogetherCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, TogetherCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = TogetherCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, TogetherCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = TogetherCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_together_response_with_metadata_try_into() {
        let valid_response = TogetherResponse {
            choices: vec![TogetherResponseChoice {
                index: 0,
                message: TogetherResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
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
        let together_response_with_metadata = TogetherResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            request: serde_json::to_value(
                TogetherRequest::new("test-model", &generic_request).unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: true,
        };
        let inference_response: ProviderInferenceResponse =
            together_response_with_metadata.try_into().unwrap();

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

        // Test case with thinking in the response
        let valid_response = TogetherResponse {
            choices: vec![TogetherResponseChoice {
                index: 0,
                message: TogetherResponseMessage {
                    content: Some("<think>hmmm</think>Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        };
        let together_response_with_metadata = TogetherResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            request: serde_json::to_value(
                TogetherRequest::new("test-model", &generic_request).unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: true,
        };
        let inference_response: ProviderInferenceResponse =
            together_response_with_metadata.try_into().unwrap();
        assert_eq!(inference_response.output.len(), 2);
        assert_eq!(
            inference_response.output[0],
            ContentBlockOutput::Thought(Thought {
                text: "hmmm".to_string(),
                signature: None,
            })
        );
        assert_eq!(
            inference_response.output[1],
            "Hello, world!".to_string().into()
        );
        assert_eq!(inference_response.raw_response, "test_response");

        // Test case with thinking in the middle of response
        let valid_response = TogetherResponse {
            choices: vec![TogetherResponseChoice {
                index: 0,
                message: TogetherResponseMessage {
                    content: Some("Hello <think>hmmm</think> world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        };
        let together_response_with_metadata = TogetherResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            request: serde_json::to_value(
                TogetherRequest::new("test-model", &generic_request).unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: true,
        };
        let inference_response: ProviderInferenceResponse =
            together_response_with_metadata.try_into().unwrap();
        assert_eq!(inference_response.output.len(), 2);
        assert_eq!(
            inference_response.output[0],
            ContentBlockOutput::Thought(Thought {
                text: "hmmm".to_string(),
                signature: None,
            })
        );
        assert_eq!(
            inference_response.output[1],
            "Hello  world!".to_string().into()
        );
        assert_eq!(inference_response.raw_response, "test_response");
    }

    #[test]
    fn test_single_thinking() {
        let text = "Hello <think>this is thinking</think> world";
        let (cleaned_text, reasoning) = process_think_blocks(text, true).unwrap();
        assert_eq!(cleaned_text, "Hello  world");
        assert_eq!(reasoning, Some("this is thinking".to_string()));

        let (cleaned_text, reasoning) = process_think_blocks(text, false).unwrap();
        assert_eq!(cleaned_text, text);
        assert_eq!(reasoning, None);
    }

    #[test]
    fn test_multiple_thinking_blocks() {
        let text = "Hello <think>thinking 1</think> middle <think>thinking 2</think>";
        let result = process_think_blocks(text, true);
        assert!(result.is_err());
        if let Err(err) = result {
            if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
                assert_eq!(message, "Multiple thinking blocks found");
            }
        }

        let (cleaned_text, reasoning) = process_think_blocks(text, false).unwrap();
        assert_eq!(cleaned_text, text);
        assert_eq!(reasoning, None);
    }

    #[test]
    fn test_extra_closing_tag() {
        let text = "Hello <think>Extra closing tag</think></think> world";
        let result = process_think_blocks(text, true);
        assert!(result.is_err());
        if let Err(err) = result {
            if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
                assert_eq!(message, "Mismatched thinking tags");
            }
        }

        let (cleaned_text, reasoning) = process_think_blocks(text, false).unwrap();
        assert_eq!(cleaned_text, text);
        assert_eq!(reasoning, None);
    }

    #[test]
    fn test_mismatched_tags() {
        let text = "Hello <think>thinking without end tag";
        let result = process_think_blocks(text, true);
        assert!(result.is_err());
        if let Err(err) = result {
            if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
                assert_eq!(message, "Mismatched thinking tags");
            }
        }

        let (cleaned_text, reasoning) = process_think_blocks(text, false).unwrap();
        assert_eq!(cleaned_text, text);
        assert_eq!(reasoning, None);
    }

    #[test]
    fn test_thinking_state() {
        // Test initial state and ID
        let mut state = ThinkingState::Normal;
        assert_eq!(state.get_id(), "0");

        // Test normal state transitions and IDs
        assert!(state.update("<think>").is_ok());
        assert!(matches!(state, ThinkingState::Thinking));
        assert_eq!(state.get_id(), "1");

        assert!(state.update("</think>").is_ok());
        assert!(matches!(state, ThinkingState::Finished));
        assert_eq!(state.get_id(), "2");

        // Test invalid transitions from Normal state
        let mut state = ThinkingState::Normal;
        let err = state.update("</think>").unwrap_err();
        assert!(matches!(
            err.get_details(),
            ErrorDetails::InferenceServer { .. }
        ));
        if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
            assert_eq!(message, "Found </think> while not thinking");
        }

        // Test invalid transitions from Thinking state
        let mut state = ThinkingState::Thinking;
        let err = state.update("<think>").unwrap_err();
        assert!(matches!(
            err.get_details(),
            ErrorDetails::InferenceServer { .. }
        ));
        if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
            assert_eq!(message, "Found <think> while already thinking");
        }

        // Test invalid transitions from Finished state
        let mut state = ThinkingState::Finished;
        let err = state.update("<think>").unwrap_err();
        assert!(matches!(
            err.get_details(),
            ErrorDetails::InferenceServer { .. }
        ));
        if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
            assert_eq!(message, "Found thinking tags after thinking finished");
        }

        let err = state.update("</think>").unwrap_err();
        assert!(matches!(
            err.get_details(),
            ErrorDetails::InferenceServer { .. }
        ));
        if let ErrorDetails::InferenceServer { message, .. } = err.get_owned_details() {
            assert_eq!(message, "Found thinking tags after thinking finished");
        }

        // Test that random text is allowed in any state
        let mut state = ThinkingState::Normal;
        assert!(state.update("random text").is_ok());

        state = ThinkingState::Thinking;
        assert!(state.update("random text").is_ok());

        state = ThinkingState::Finished;
        assert!(state.update("random text").is_ok());
    }

    #[test]
    fn test_together_to_tensorzero_chunk() {
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let mut tool_call_names = vec!["name1".to_string()];
        let mut thinking_state = ThinkingState::Normal;
        let message = together_to_tensorzero_chunk(
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
            })],
        );
        assert_eq!(message.finish_reason, Some(FinishReason::Stop));
        // Test what an intermediate tool chunk should look like
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: None,
                    tool_calls: Some(vec![TogetherToolCallChunk {
                        index: 0,
                        id: None,
                        function: TogetherFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(TogetherFinishReason::ToolCalls),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
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
        // Test what a bad tool chunk would do (new ID but no names)
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: None,
                    tool_calls: Some(vec![TogetherToolCallChunk {
                        index: 1,
                        id: None,
                        function: TogetherFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(TogetherFinishReason::ToolCalls),
            }],
            usage: None,
        };
        let error = together_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
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
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: None,
                    tool_calls: Some(vec![TogetherToolCallChunk {
                        index: 1,
                        id: Some("id2".to_string()),
                        function: TogetherFunctionCallChunk {
                            name: Some("name2".to_string()),
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(TogetherFinishReason::ToolCalls),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
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
                id: "id2".to_string(),
                raw_name: "name2".to_string(),
                raw_arguments: "{\"hello\":\"world\"}".to_string(),
            })]
        );
        // Check that the lists were updated
        assert_eq!(tool_call_ids, vec!["id1".to_string(), "id2".to_string()]);
        assert_eq!(
            tool_call_names,
            vec!["name1".to_string(), "name2".to_string()]
        );

        // Check a chunk with no choices and only usage
        let chunk = TogetherChatChunk {
            choices: vec![],
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };
        let message = together_to_tensorzero_chunk(
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

        // Test a thinking chunk
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("<think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();
        assert!(message.content.is_empty());
        assert!(matches!(thinking_state, ThinkingState::Thinking));

        // Test a thinking middle chunk
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("some thinking content".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
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
            vec![ContentBlockChunk::Thought(ThoughtChunk {
                text: Some("some thinking content".to_string()),
                signature: None,
                id: "1".to_string(),
            })]
        );
        assert!(matches!(thinking_state, ThinkingState::Thinking));

        // Test a thinking chunk end
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("</think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
            true,
        )
        .unwrap();
        assert!(message.content.is_empty());
        assert!(matches!(thinking_state, ThinkingState::Finished));
    }

    #[test]
    fn test_together_to_tensorzero_chunk_without_think_parsing() {
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("Hello <think>should not parse</think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec![];
        let mut tool_call_names = vec![];
        let mut thinking_state = ThinkingState::Normal;
        let message = together_to_tensorzero_chunk(
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
}
