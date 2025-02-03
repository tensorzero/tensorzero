use std::{borrow::Cow, sync::OnceLock, time::Duration};

use futures::{Stream, StreamExt};
use lazy_static::lazy_static;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::types::{
        batch::{BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse},
        ContentBlockChunk, ContentBlockOutput, Latency, ModelInferenceRequest,
        ModelInferenceRequestJsonMode, ProviderInferenceResponse, ProviderInferenceResponseChunk,
        ProviderInferenceResponseStream, TextChunk, Thought, ThoughtChunk,
    },
    model::{build_creds_caching_default, Credential, CredentialLocation},
    tool::{ToolCall, ToolCallChunk},
};

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
}

static DEFAULT_CREDENTIALS: OnceLock<TogetherCredentials> = OnceLock::new();

impl TogetherProvider {
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
        Ok(TogetherProvider {
            model_name,
            credentials,
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
    #[cfg(any(test, feature = "e2e_tests"))]
    None,
}

impl TryFrom<Credential> for TogetherCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(TogetherCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(TogetherCredentials::Dynamic(key_name)),
            #[cfg(any(test, feature = "e2e_tests"))]
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
            #[cfg(any(test, feature = "e2e_tests"))]
            TogetherCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            })?,
        }
    }
}

// TODO (#80): Add support for Llama 3.1 function calling as discussed [here](https://docs.together.ai/docs/llama-3-function-calling)

impl InferenceProvider for TogetherProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = TogetherRequest::new(&self.model_name, request);
        let request_url = get_chat_url(&TOGETHER_API_BASE)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
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
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
        let request_body = TogetherRequest::new(&self.model_name, request);
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
        let mut stream = Box::pin(stream_together(event_source, start_time));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::new(ErrorDetails::InferenceServer {
                    message: "Stream ended before first chunk".to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                }));
            }
        };
        Ok((chunk, stream, raw_request))
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
    pub fn new(model: &'a str, request: &'a ModelInferenceRequest) -> TogetherRequest<'a> {
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(TogetherResponseFormat::JsonObject {
                    schema: request.output_schema,
                })
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_together_messages(request);
        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(request);
        TogetherRequest {
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
        }
    }
}

pub(super) fn prepare_together_messages<'a>(
    request: &'a ModelInferenceRequest,
) -> Vec<OpenAIRequestMessage<'a>> {
    let mut messages: Vec<OpenAIRequestMessage> = request
        .messages
        .iter()
        .flat_map(tensorzero_to_openai_messages)
        .collect();
    if let Some(system_msg) = tensorzero_to_together_system_message(request.system.as_deref()) {
        messages.insert(0, system_msg);
    }
    messages
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

#[derive(Clone, Debug, PartialEq, Serialize)]
struct TogetherResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<TogetherResponseToolCall>>,
}

impl<'de> Deserialize<'de> for TogetherResponseMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        #[derive(Deserialize)]
        struct Helper {
            content: Option<String>,
            tool_calls: Option<Vec<TogetherResponseToolCall>>,
        }

        let helper = Helper::deserialize(deserializer)?;

        let (content, reasoning_content) = if let Some(text) = helper.content {
            // Count thinking blocks
            let think_count = text.matches("<think>").count();
            if think_count > 1 {
                return Err(D::Error::custom("Multiple thinking blocks found"));
            }

            if let (Some(start), Some(end)) = (text.find("<think>"), text.find("</think>")) {
                let reasoning = text[start + 7..end].trim().to_string();
                let mut cleaned = String::with_capacity(text.len());
                cleaned.push_str(&text[..start]);
                cleaned.push_str(&text[end + 8..]);
                (Some(cleaned.trim().to_string()), Some(reasoning))
            } else if think_count != text.matches("</think>").count() {
                // Check for mismatched tags
                return Err(D::Error::custom("Mismatched thinking tags"));
            } else {
                (Some(text), None)
            }
        } else {
            (None, None)
        };

        Ok(TogetherResponseMessage {
            content,
            reasoning_content,
            tool_calls: helper.tool_calls,
        })
    }
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct TogetherResponseChoice {
    index: u8,
    message: TogetherResponseMessage,
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
    request: TogetherRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
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
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }))?
            .message;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(reasoning) = message.reasoning_content {
            content.push(ContentBlockOutput::Thought(Thought { text: reasoning }));
        }
        if let Some(text) = message.content {
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        }
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request body as JSON: {e}"),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let system = generic_request.system.clone();
        let messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            content,
            system,
            messages,
            raw_request,
            raw_response,
            usage,
            latency,
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
) -> impl Stream<Item = Result<ProviderInferenceResponseChunk, Error>> {
    let mut tool_call_ids = Vec::new();
    let mut tool_call_names = Vec::new();
    let mut thinking_state = ThinkingState::Normal;
    async_stream::stream! {
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
                                message: format!(
                                    "Error parsing chunk. Error: {}",
                                    e,
                                ),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            together_to_tensorzero_chunk(d, latency, &mut tool_call_ids, &mut tool_call_names, &mut thinking_state)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    }
}

/// Maps a Together chunk to a TensorZero chunk for streaming inferences
fn together_to_tensorzero_chunk(
    mut chunk: TogetherChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    tool_names: &mut Vec<String>,
    thinking_state: &mut ThinkingState,
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
    let mut content = vec![];
    if let Some(choice) = chunk.choices.pop() {
        if let Some(text) = choice.delta.content {
            // If an update is made to the thinking state, we don't want to add a text chunk
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
                            text,
                            id: thinking_state.get_id(),
                        }));
                    }
                }
            };
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
                        tool_names.push(name.clone());
                        name
                    }
                    None => {
                        tool_names
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

    use crate::inference::providers::common::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::providers::openai::{
        OpenAIToolType, OpenAIUsage, SpecificToolChoice, SpecificToolFunction,
    };
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
        };

        let together_request =
            TogetherRequest::new("togethercomputer/llama-v3-8b", &request_with_tools);

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
        assert_eq!(together_request.parallel_tool_calls, Some(false));
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

        // Test Missing credential (test mode)
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            let generic = Credential::Missing;
            let creds = TogetherCredentials::try_from(generic).unwrap();
            assert!(matches!(creds, TogetherCredentials::None));
        }

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
                    reasoning_content: Some("Thinking...".to_string()),
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
        };
        let together_response_with_metadata = TogetherResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            request: TogetherRequest::new("test-model", &generic_request),
            generic_request: &generic_request,
        };
        let inference_response: ProviderInferenceResponse =
            together_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 2);
        assert_eq!(
            inference_response.output[0],
            ContentBlockOutput::Thought(Thought {
                text: "Thinking...".to_string()
            })
        );
        assert_eq!(
            inference_response.output[1],
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
    fn test_single_thinking() {
        let json = r#"{
            "content": "Hello <think>this is thinking</think> world",
            "tool_calls": null
        }"#;

        let msg: TogetherResponseMessage = serde_json::from_str(json).unwrap();
        // 2 spaces between Hello and world because of the thinking block and whitespace
        assert_eq!(msg.content, Some("Hello  world".to_string()));
        assert_eq!(msg.reasoning_content, Some("this is thinking".to_string()));
    }

    #[test]
    fn test_multiple_thinking_blocks() {
        let json = r#"{
            "content": "Hello <think>thinking 1</think> middle <think>thinking 2</think>",
            "tool_calls": null
        }"#;

        assert!(serde_json::from_str::<TogetherResponseMessage>(json).is_err());
    }

    #[test]
    fn test_mismatched_tags() {
        let json = r#"{
            "content": "Hello <think>thinking without end tag",
            "tool_calls": null
        }"#;

        assert!(serde_json::from_str::<TogetherResponseMessage>(json).is_err());
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
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "Hello".to_string(),
                id: "0".to_string(),
            })],
        );
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
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
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
            }],
            usage: None,
        };
        let error = together_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
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
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
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
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
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
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Thought(ThoughtChunk {
                text: "some thinking content".to_string(),
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
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
            &mut thinking_state,
        )
        .unwrap();
        assert!(message.content.is_empty());
        assert!(matches!(thinking_state, ThinkingState::Finished));
    }
}
