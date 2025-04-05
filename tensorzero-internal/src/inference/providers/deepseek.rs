use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, ProviderInferenceResponse,
};
use crate::inference::types::{
    ContentBlockChunk, ContentBlockOutput, ProviderInferenceResponseArgs,
    ProviderInferenceResponseStreamInner, TextChunk, Thought, ThoughtChunk,
};
use crate::inference::types::{
    PeekableProviderInferenceResponseStream, ProviderInferenceResponseChunk,
};
use crate::model::{Credential, CredentialLocation, ModelProvider};
use crate::tool::ToolCallChunk;

use super::helpers::inject_extra_request_data;
use super::openai::{
    convert_stream_error, get_chat_url, handle_openai_error, prepare_openai_tools,
    tensorzero_to_openai_messages, tensorzero_to_openai_system_message,
    OpenAIAssistantRequestMessage, OpenAIContentBlock, OpenAIFinishReason, OpenAIRequestMessage,
    OpenAIResponseToolCall, OpenAISystemRequestMessage, OpenAITool, OpenAIToolChoice, OpenAIUsage,
    OpenAIUserRequestMessage, StreamOptions,
};

lazy_static! {
    static ref DEEPSEEK_DEFAULT_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.deepseek.com/v1")
            .expect("Failed to parse DEEPSEEK_DEFAULT_BASE_URL")
    };
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("DEEPSEEK_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "DeepSeek";
const PROVIDER_TYPE: &str = "deepseek";

#[derive(Debug)]
pub enum DeepSeekCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for DeepSeekCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(DeepSeekCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(DeepSeekCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(DeepSeekCredentials::None),
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
    ) -> Result<&'a SecretString, Error> {
        match self {
            DeepSeekCredentials::Static(api_key) => Ok(api_key),
            DeepSeekCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                })
            }
            DeepSeekCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            }
            .into()),
        }
    }
}

#[derive(Debug)]
pub struct DeepSeekProvider {
    model_name: String,
    credentials: DeepSeekCredentials,
}

impl DeepSeekProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, PROVIDER_TYPE))?;
        let provider_credentials = DeepSeekCredentials::try_from(generic_credentials)?;

        Ok(DeepSeekProvider {
            model_name,
            credentials: provider_credentials,
        })
    }
}

impl InferenceProvider for DeepSeekProvider {
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
            serde_json::to_value(DeepSeekRequest::new(&self.model_name, request)?).map_err(
                |e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Error serializing DeepSeek request: {e}"),
                    })
                },
            )?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let request_url = get_chat_url(&DEEPSEEK_DEFAULT_BASE_URL)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .headers(headers);

        let res = request_builder
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to DeepSeek: {e}"),
                    status_code: e.status(),
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

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(DeepSeekResponseWithMetadata {
                response,
                raw_response,
                latency,
                request: request_body,
                generic_request: request,
            }
            .try_into()?)
        } else {
            let status = res.status();

            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing error response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            Err(handle_openai_error(status, &response, PROVIDER_TYPE))
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
            serde_json::to_value(DeepSeekRequest::new(&self.model_name, request)?).map_err(
                |e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Error serializing DeepSeek request: {e}"),
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
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request: {e}"),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let request_url = get_chat_url(&DEEPSEEK_DEFAULT_BASE_URL)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .headers(headers)
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request: {e}"),
                    status_code: None,
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

        let stream = stream_deepseek(event_source, start_time).peekable();
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

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum DeepSeekResponseFormat {
    #[default]
    Text,
    JsonObject,
}

impl DeepSeekResponseFormat {
    fn new(json_mode: &ModelInferenceRequestJsonMode) -> Self {
        match json_mode {
            ModelInferenceRequestJsonMode::On => DeepSeekResponseFormat::JsonObject,
            ModelInferenceRequestJsonMode::Off => DeepSeekResponseFormat::Text,
            ModelInferenceRequestJsonMode::Strict => DeepSeekResponseFormat::JsonObject,
        }
    }
}

#[derive(Debug, Serialize)]
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
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<DeepSeekResponseFormat>,
}

impl<'a> DeepSeekRequest<'a> {
    pub fn new(
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

        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };

        if request.json_mode == ModelInferenceRequestJsonMode::Strict {
            tracing::warn!("DeepSeek provider does not support strict JSON mode. Downgrading to normal JSON mode.");
        }

        let response_format = Some(DeepSeekResponseFormat::new(&request.json_mode));

        // NOTE: as mentioned by the DeepSeek team here: https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations
        // the R1 series of models does not perform well with the system prompt. As we move towards first-class support for reasoning models we should check
        // if a model is an R1 model and if so, remove the system prompt from the request and instead put it in the first user message.
        let messages = prepare_deepseek_messages(request, model)?;

        let (tools, tool_choice, _) = prepare_openai_tools(request);

        Ok(DeepSeekRequest {
            messages,
            model,
            temperature,
            max_tokens,
            seed,
            top_p,
            presence_penalty,
            frequency_penalty,
            stream,
            stream_options,
            response_format,
            tools,
            tool_choice,
        })
    }
}

fn stream_deepseek(
    mut event_source: EventSource,
    start_time: Instant,
) -> ProviderInferenceResponseStreamInner {
    let mut tool_call_ids = Vec::new();
    let mut tool_call_names = Vec::new();
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(PROVIDER_TYPE.to_string(), e).await);
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
                                    "Error parsing chunk. Error: {}",
                                    e,
                                ),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            deepseek_to_tensorzero_chunk(d, latency, &mut tool_call_ids, &mut tool_call_names)
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
    mut chunk: DeepSeekChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    tool_names: &mut Vec<String>,
) -> Result<ProviderInferenceResponseChunk, Error> {
    let raw_message = serde_json::to_string(&chunk).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error parsing response from DeepSeek: {e}"),
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

pub(super) fn prepare_deepseek_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
    model_name: &'a str,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in request.messages.iter() {
        messages.extend(tensorzero_to_openai_messages(message)?);
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
    } else if let Some(system_msg) = tensorzero_to_openai_system_message(
        request.system.as_deref(),
        &request.json_mode,
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
    request: serde_json::Value,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<DeepSeekResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: DeepSeekResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let DeepSeekResponseWithMetadata {
            mut response,
            raw_response,
            latency,
            request: request_body,
            generic_request,
        } = value;

        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices {}, Expected 1",
                    response.choices.len()
                ),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
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
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(reasoning) = message.reasoning_content {
            content.push(ContentBlockOutput::Thought(Thought {
                text: reasoning,
                signature: None,
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
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request body as JSON: {e}"),
            })
        })?;
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
                finish_reason: finish_reason.map(|r| r.into()),
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

    use crate::inference::providers::openai::{
        OpenAIRequestFunctionCall, OpenAIRequestToolCall, OpenAIToolRequestMessage, OpenAIToolType,
        OpenAIUsage, SpecificToolChoice, SpecificToolFunction,
    };
    use crate::inference::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::types::{
        FinishReason, FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };

    #[test]
    fn test_deepseek_request_new() {
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
            .expect("failed to create Deepseek Request during test");

        assert_eq!(deepseek_request.messages.len(), 1);
        assert_eq!(deepseek_request.temperature, Some(0.5));
        assert_eq!(deepseek_request.max_tokens, Some(100));
        assert!(!deepseek_request.stream);
        assert_eq!(deepseek_request.seed, Some(69));
        assert!(deepseek_request.tools.is_some());
        let tools = deepseek_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            deepseek_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
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

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            deepseek_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );

        let request_with_tools = ModelInferenceRequest {
            json_mode: ModelInferenceRequestJsonMode::Strict,
            ..request_with_tools
        };

        let deepseek_request = DeepSeekRequest::new("deepseek-chat", &request_with_tools);
        let deepseek_request = deepseek_request.unwrap();
        // We should downgrade the strict JSON mode to normal JSON mode for deepseek
        assert_eq!(
            deepseek_request.response_format,
            Some(DeepSeekResponseFormat::JsonObject)
        );
    }

    #[test]
    fn test_deepseek_api_base() {
        assert_eq!(
            DEEPSEEK_DEFAULT_BASE_URL.as_str(),
            "https://api.deepseek.com/v1"
        );
    }

    #[test]
    fn test_credential_to_deepseek_credentials() {
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
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
    #[test]
    fn test_deepseek_response_with_metadata_try_into() {
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
        let deepseek_response_with_metadata = DeepSeekResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            request: serde_json::to_value(
                DeepSeekRequest::new("deepseek-chat", &generic_request).unwrap(),
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
                text: "I'm thinking about the weather".to_string(),
                signature: None,
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
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_prepare_deepseek_messages() {
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

        let messages = prepare_deepseek_messages(&request, "deepseek-chat").unwrap();
        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[0], OpenAIRequestMessage::System(_)));
        assert!(matches!(messages[1], OpenAIRequestMessage::User(_)));

        // Test case 2: Reasoner model with system message
        let messages = prepare_deepseek_messages(&request, "deepseek-reasoner").unwrap();
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

        let messages = prepare_deepseek_messages(&request_no_system, "deepseek-chat").unwrap();
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

        let messages = prepare_deepseek_messages(&request_multiple, "deepseek-chat").unwrap();
        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[0], OpenAIRequestMessage::System(_)));
        assert!(matches!(messages[1], OpenAIRequestMessage::User(_)));
        assert!(matches!(messages[2], OpenAIRequestMessage::Assistant(_)));
        assert!(matches!(messages[3], OpenAIRequestMessage::User(_)));
    }

    // Helper constructors for test messages.
    fn system_message(content: &str) -> OpenAIRequestMessage {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(content),
        })
    }
    fn user_message(content: &str) -> OpenAIRequestMessage {
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
    fn test_coalesce_consecutive_messages() {
        // Create dummy tool calls to test assistant message merging.
        let tool_call1 = OpenAIRequestToolCall {
            id: "tc1",
            r#type: OpenAIToolType::Function,
            function: OpenAIRequestFunctionCall {
                name: "func1",
                arguments: "args1",
            },
        };
        let tool_call2 = OpenAIRequestToolCall {
            id: "tc2",
            r#type: OpenAIToolType::Function,
            function: OpenAIRequestFunctionCall {
                name: "func2",
                arguments: "args2",
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
