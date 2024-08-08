use futures::stream::Stream;
use futures::StreamExt;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, InferenceResponseStream, JSONMode, Latency,
    ModelInferenceRequest, ModelInferenceResponse, ModelInferenceResponseChunk, RequestMessage,
    Role, TextChunk, Tool, ToolCall, ToolCallChunk, ToolChoice, Usage,
};
use crate::model::ProviderConfig;

const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1/";

pub struct OpenAIProvider;

impl InferenceProvider for OpenAIProvider {
    async fn infer<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let (model_name, api_base, api_key) = match model {
            ProviderConfig::OpenAI {
                model_name,
                api_base,
                api_key,
            } => (
                model_name,
                api_base.as_deref(),
                api_key.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "OpenAI".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected OpenAI provider config".to_string(),
                })
            }
        };
        let request_body = OpenAIRequest::new(model_name, request);
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to OpenAI: {e}"),
            })?;
        if res.status().is_success() {
            let response_body =
                res.json::<OpenAIResponse>()
                    .await
                    .map_err(|e| Error::OpenAIServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(OpenAIResponseWithLatency {
                response: response_body,
                latency,
            }
            .try_into()?)
        } else {
            handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::OpenAIServer {
                    message: format!("Error parsing error response: {e}"),
                })?,
            )
        }
    }

    async fn infer_stream<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let (model_name, api_base, api_key) = match model {
            ProviderConfig::OpenAI {
                model_name,
                api_base,
                api_key,
            } => (
                model_name,
                api_base.as_deref(),
                api_key.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "OpenAI".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected OpenAI provider config".to_string(),
                })
            }
        };
        let request_body = OpenAIRequest::new(model_name, request);
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to OpenAI: {e}"),
            })?;

        let mut stream = Box::pin(stream_openai(event_source, start_time));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::OpenAIServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };
        Ok((chunk, stream))
    }
}

pub fn stream_openai(
    mut event_source: EventSource,
    start_time: Instant,
) -> impl Stream<Item = Result<ModelInferenceResponseChunk, Error>> {
    let mut tool_call_ids = Vec::new();
    let mut tool_call_names = Vec::new();
    async_stream::stream! {
        let inference_id = Uuid::now_v7();
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(Error::OpenAIServer {
                        message: e.to_string(),
                    });
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<OpenAIChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::OpenAIServer {
                                message: format!(
                                    "Error parsing chunk. Error: {}, Data: {}",
                                    e, message.data
                                ),
                            });
                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            openai_to_tensorzero_stream_message(d, inference_id, latency, &mut tool_call_ids, &mut tool_call_names)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    }
}

pub(super) fn get_chat_url(base_url: Option<&str>) -> Result<Url, Error> {
    let base_url = base_url.unwrap_or(OPENAI_DEFAULT_BASE_URL);
    let base_url = if base_url.ends_with('/') {
        base_url.to_string()
    } else {
        format!("{}/", base_url)
    };
    let url = Url::parse(&base_url)
        .map_err(|e| Error::InvalidBaseUrl {
            message: e.to_string(),
        })?
        .join("chat/completions")
        .map_err(|e| Error::InvalidBaseUrl {
            message: e.to_string(),
        })?;
    Ok(url)
}

pub(super) fn handle_openai_error(
    response_code: StatusCode,
    response_body: &str,
) -> Result<ModelInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::TOO_MANY_REQUESTS => {
            Err(Error::OpenAIClient {
                message: response_body.to_string(),
                status_code: response_code,
            })
        }
        _ => Err(Error::OpenAIServer {
            message: response_body.to_string(),
        }),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenAISystemRequestMessage<'a> {
    content: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenAIUserRequestMessage<'a> {
    content: &'a str, // NOTE: this could be an array including images and stuff according to API spec (not supported yet)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenAIRequestFunctionCall<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenAIRequestToolCall<'a> {
    id: &'a str,
    r#type: OpenAIToolType,
    function: OpenAIRequestFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for OpenAIRequestToolCall<'a> {
    fn from(tool_call: &'a ToolCall) -> Self {
        OpenAIRequestToolCall {
            id: &tool_call.id,
            r#type: OpenAIToolType::Function,
            function: OpenAIRequestFunctionCall {
                name: &tool_call.name,
                arguments: &tool_call.arguments,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenAIAssistantRequestMessage<'a> {
    content: Option<&'a str>,
    tool_calls: Option<Vec<OpenAIRequestToolCall<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenAIToolRequestMessage<'a> {
    content: &'a str,
    tool_call_id: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub(super) enum OpenAIRequestMessage<'a> {
    System(OpenAISystemRequestMessage<'a>),
    User(OpenAIUserRequestMessage<'a>),
    Assistant(OpenAIAssistantRequestMessage<'a>),
    Tool(OpenAIToolRequestMessage<'a>),
}

pub(super) fn prepare_openai_messages<'a>(
    request: &'a ModelInferenceRequest,
) -> Vec<OpenAIRequestMessage<'a>> {
    let mut messages: Vec<OpenAIRequestMessage> = request
        .messages
        .iter()
        .flat_map(|msg| tensorzero_to_openai_messages(msg))
        .collect();
    if let Some(system_msg) =
        tensorzero_to_openai_system_message(request.system_instructions.as_deref())
    {
        messages.insert(0, system_msg);
    }
    messages
}

// TODO(viraj): unit test this
pub(super) fn prepare_openai_tools<'a>(
    request: &'a ModelInferenceRequest,
) -> (Option<Vec<OpenAITool<'a>>>, Option<OpenAIToolChoice<'a>>) {
    let tools = request
        .tools_available
        .as_ref()
        .map(|t| t.iter().map(|tool| OpenAITool::from(tool)).collect());
    let tool_choice: Option<OpenAIToolChoice<'a>> = match tools {
        // TODO (Viraj): if this is an empty list, should we return None?
        Some(_) => Some((&request.tool_choice).into()),
        None => None,
    };
    (tools, tool_choice)
}

fn tensorzero_to_openai_system_message<'a>(
    system_instructions: Option<&'a str>,
) -> Option<OpenAIRequestMessage<'a>> {
    system_instructions.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: instructions,
        })
    })
}

fn tensorzero_to_openai_messages<'a>(message: &'a RequestMessage) -> Vec<OpenAIRequestMessage<'a>> {
    let mut messages = Vec::new();
    let mut tool_calls = Vec::new();
    let mut first_assistant_message_index: Option<usize> = None;
    for block in message.content.iter() {
        match block {
            ContentBlock::Text(text) => match message.role {
                Role::User => {
                    messages.push(OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                        content: text,
                    }));
                }
                Role::Assistant => {
                    messages.push(OpenAIRequestMessage::Assistant(
                        OpenAIAssistantRequestMessage {
                            content: Some(text),
                            tool_calls: None,
                        },
                    ));
                    if first_assistant_message_index.is_none() {
                        first_assistant_message_index = Some(messages.len() - 1);
                    }
                }
            },
            ContentBlock::ToolCall(tool_call) => {
                tool_calls.push(OpenAIRequestToolCall {
                    id: &tool_call.id,
                    r#type: OpenAIToolType::Function,
                    function: OpenAIRequestFunctionCall {
                        name: &tool_call.name,
                        arguments: &tool_call.arguments,
                    },
                });
            }
            ContentBlock::ToolResult(tool_result) => {
                let message = OpenAIRequestMessage::Tool(OpenAIToolRequestMessage {
                    content: &tool_result.result,
                    tool_call_id: &tool_result.id,
                });
                messages.push(message);
            }
        }
    }
    // TODO (viraj): test this explicitly
    if !tool_calls.is_empty() {
        match first_assistant_message_index {
            Some(index) => {
                if let Some(OpenAIRequestMessage::Assistant(msg)) = messages.get_mut(index) {
                    msg.tool_calls = Some(tool_calls);
                }
            }
            None => {
                messages.push(OpenAIRequestMessage::Assistant(
                    OpenAIAssistantRequestMessage {
                        content: None,
                        tool_calls: Some(tool_calls),
                    },
                ));
            }
        }
    }
    messages
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum OpenAIResponseFormat {
    JsonObject,
    #[default]
    Text,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum OpenAIToolType {
    Function,
}

#[derive(Debug, PartialEq, Serialize)]
struct OpenAIFunction<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    parameters: &'a Value,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct OpenAITool<'a> {
    r#type: OpenAIToolType,
    function: OpenAIFunction<'a>,
}

impl<'a> From<&'a Tool> for OpenAITool<'a> {
    fn from(tool: &'a Tool) -> Self {
        match tool {
            Tool::Function {
                description,
                name,
                parameters,
            } => OpenAITool {
                r#type: OpenAIToolType::Function,
                function: OpenAIFunction {
                    name: name,
                    description: description.as_deref(),
                    parameters: parameters,
                },
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(super) enum OpenAIToolChoice<'a> {
    String(OpenAIToolChoiceString),
    Specific(SpecificToolChoice<'a>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum OpenAIToolChoiceString {
    None,
    Auto,
    Required,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SpecificToolChoice<'a> {
    r#type: OpenAIToolType,
    function: SpecificToolFunction<'a>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct SpecificToolFunction<'a> {
    name: &'a str,
}

impl<'a> Default for OpenAIToolChoice<'a> {
    fn default() -> Self {
        OpenAIToolChoice::String(OpenAIToolChoiceString::None)
    }
}

impl<'a> From<&'a ToolChoice> for OpenAIToolChoice<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::None => OpenAIToolChoice::String(OpenAIToolChoiceString::None),
            ToolChoice::Auto => OpenAIToolChoice::String(OpenAIToolChoiceString::Auto),
            ToolChoice::Required => OpenAIToolChoice::String(OpenAIToolChoiceString::Required),
            ToolChoice::Tool(tool_name) => OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction { name: tool_name },
            }),
            ToolChoice::Implicit => OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction { name: "respond" },
            }),
        }
    }
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// This struct defines the supported parameters for the OpenAI API
/// See the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Serialize)]
struct OpenAIRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    response_format: OpenAIResponseFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> OpenAIRequest<'a> {
    pub fn new(model: &'a str, request: &'a ModelInferenceRequest) -> OpenAIRequest<'a> {
        let response_format = match request.json_mode {
            // TODO(#68): Implement structured output here
            JSONMode::On | JSONMode::Strict => OpenAIResponseFormat::JsonObject,
            JSONMode::Off => OpenAIResponseFormat::Text,
        };
        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };
        let messages = prepare_openai_messages(request);

        let (tools, tool_choice) = prepare_openai_tools(request);
        OpenAIRequest {
            messages,
            model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream,
            stream_options,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls: request.parallel_tool_calls,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<OpenAIUsage> for Usage {
    fn from(usage: OpenAIUsage) -> Self {
        Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenAIResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenAIResponseToolCall {
    id: String,
    r#type: OpenAIToolType,
    function: OpenAIResponseFunctionCall,
}

impl From<OpenAIResponseToolCall> for ToolCall {
    fn from(openai_tool_call: OpenAIResponseToolCall) -> Self {
        ToolCall {
            id: openai_tool_call.id,
            name: openai_tool_call.function.name,
            arguments: openai_tool_call.function.arguments,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct OpenAIResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct OpenAIResponseChoice {
    index: u8,
    message: OpenAIResponseMessage,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct OpenAIResponse {
    choices: Vec<OpenAIResponseChoice>,
    usage: OpenAIUsage,
}

pub(super) struct OpenAIResponseWithLatency {
    pub(super) response: OpenAIResponse,
    pub(super) latency: Latency,
}

impl TryFrom<OpenAIResponseWithLatency> for ModelInferenceResponse {
    type Error = Error;
    fn try_from(value: OpenAIResponseWithLatency) -> Result<Self, Self::Error> {
        let OpenAIResponseWithLatency {
            mut response,
            latency,
        } = value;
        let raw = serde_json::to_string(&response).map_err(|e| Error::OpenAIServer {
            message: format!("Error parsing response: {e}"),
        })?;
        if response.choices.len() != 1 {
            return Err(Error::OpenAIServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
            });
        }
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or(Error::OpenAIServer {
                message: "Response has no choices (this should never happen)".to_string(),
            })?
            .message;
        let mut content: Vec<ContentBlock> = Vec::new();
        if let Some(text) = message.content {
            content.push(ContentBlock::Text(text));
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlock::ToolCall(tool_call.into()));
            }
        }

        Ok(ModelInferenceResponse::new(content, raw, usage, latency))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct OpenAIFunctionCallChunk {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct OpenAIToolCallChunk {
    index: u8,
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: OpenAIFunctionCallChunk,
}

// This doesn't include role
#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct OpenAIDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct OpenAIChatChunkChoice {
    delta: OpenAIDelta,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct OpenAIChatChunk {
    choices: Vec<OpenAIChatChunkChoice>,
    usage: Option<OpenAIUsage>,
}

// TODO(Viraj): write a unit test for a chunk with no choices but only usage,
// since this case happens and we should behave well
fn openai_to_tensorzero_stream_message(
    mut chunk: OpenAIChatChunk,
    inference_id: Uuid,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    tool_names: &mut Vec<String>,
) -> Result<ModelInferenceResponseChunk, Error> {
    let raw_message = serde_json::to_string(&chunk).map_err(|e| Error::OpenAIServer {
        message: format!("Error parsing response from OpenAI: {e}"),
    })?;
    if chunk.choices.len() > 1 {
        return Err(Error::OpenAIServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
        });
    }
    let usage = chunk.usage.map(|u| u.into());
    let mut content = vec![];
    if let Some(choice) = chunk.choices.pop() {
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
                    Some(id) => {tool_call_ids.push(id.clone()); id}
                    None => {
                        tool_call_ids.get(index as usize).ok_or(Error::OpenAIServer {
                            message: "Tool call index out of bounds (meaning we haven't see this many ids in the stream)".to_string(),
                        })?.clone()
                    }
                };
                let name = match tool_call.function.name {
                    Some(name) => {tool_names.push(name.clone()); name}
                    None => {
                        tool_names.get(index as usize).ok_or(Error::OpenAIServer {
                            message: "Tool call index out of bounds (meaning we haven't see this many names in the stream)".to_string(),
                        })?.clone()
                    }
                };
                content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                    id,
                    name,
                    arguments: tool_call.function.arguments.unwrap_or_default(),
                }));
            }
        }
    }

    Ok(ModelInferenceResponseChunk::new(
        inference_id,
        content,
        usage,
        raw_message,
        latency,
    ))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::inference::types::{FunctionType, Tool};

    use super::*;

    #[test]
    fn test_get_chat_url() {
        // Test with default URL
        let default_url = get_chat_url(None).unwrap();
        assert_eq!(
            default_url.as_str(),
            "https://api.openai.com/v1/chat/completions"
        );

        // Test with custom base URL
        let custom_base = "https://custom.openai.com/api/";
        let custom_url = get_chat_url(Some(custom_base)).unwrap();
        assert_eq!(
            custom_url.as_str(),
            "https://custom.openai.com/api/chat/completions"
        );

        // Test with invalid URL
        let invalid_url = get_chat_url(Some("not a url"));
        assert!(invalid_url.is_err());

        // Test with URL without trailing slash
        let unjoinable_url = get_chat_url(Some("https://example.com"));
        assert!(unjoinable_url.is_ok());
        assert_eq!(
            unjoinable_url.unwrap().as_str(),
            "https://example.com/chat/completions"
        );
        // Test with URL that can't be joined
        let unjoinable_url = get_chat_url(Some("https://example.com/foo"));
        assert!(unjoinable_url.is_ok());
        assert_eq!(
            unjoinable_url.unwrap().as_str(),
            "https://example.com/foo/chat/completions"
        );
    }

    #[test]
    fn test_handle_openai_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_openai_error(StatusCode::UNAUTHORIZED, "Unauthorized access");
        assert!(matches!(unauthorized, Err(Error::OpenAIClient { .. })));
        if let Err(Error::OpenAIClient {
            message,
            status_code,
        }) = unauthorized
        {
            assert_eq!(message, "Unauthorized access");
            assert_eq!(status_code, StatusCode::UNAUTHORIZED);
        }

        // Test forbidden error
        let forbidden = handle_openai_error(StatusCode::FORBIDDEN, "Forbidden access");
        assert!(matches!(forbidden, Err(Error::OpenAIClient { .. })));
        if let Err(Error::OpenAIClient {
            message,
            status_code,
        }) = forbidden
        {
            assert_eq!(message, "Forbidden access");
            assert_eq!(status_code, StatusCode::FORBIDDEN);
        }

        // Test rate limit error
        let rate_limit = handle_openai_error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded");
        assert!(matches!(rate_limit, Err(Error::OpenAIClient { .. })));
        if let Err(Error::OpenAIClient {
            message,
            status_code,
        }) = rate_limit
        {
            assert_eq!(message, "Rate limit exceeded");
            assert_eq!(status_code, StatusCode::TOO_MANY_REQUESTS);
        }

        // Test server error
        let server_error = handle_openai_error(StatusCode::INTERNAL_SERVER_ERROR, "Server error");
        assert!(matches!(server_error, Err(Error::OpenAIServer { .. })));
        if let Err(Error::OpenAIServer { message }) = server_error {
            assert_eq!(message, "Server error");
        }
    }

    #[test]
    fn test_openai_request_new() {
        // Test basic request
        let basic_request = ModelInferenceRequest {
            messages: vec![
                RequestMessage {
                    role: Role::User,
                    content: vec![ContentBlock::Text("Hello".to_string())],
                },
                RequestMessage {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text("Hi there!".to_string())],
                },
            ],
            system_instructions: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: true,
            json_mode: JSONMode::Off,
            tools_available: None,
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let openai_request = OpenAIRequest::new("gpt-3.5-turbo", &basic_request);

        assert_eq!(openai_request.model, "gpt-3.5-turbo");
        assert_eq!(openai_request.messages.len(), 2);
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_tokens, Some(100));
        assert!(openai_request.stream);
        assert_eq!(openai_request.response_format, OpenAIResponseFormat::Text);
        assert!(openai_request.tools.is_none());
        assert_eq!(openai_request.tool_choice, None);
        assert!(openai_request.parallel_tool_calls.is_none());

        // Test request with tools and JSON mode
        let tool = Tool::Function {
            name: "get_weather".to_string(),
            description: Some("Get the current weather".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }),
        };

        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text("What's the weather?".to_string())],
            }],
            system_instructions: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: JSONMode::On,
            tools_available: Some(vec![tool]),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: Some(true),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools);

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 1);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_tokens, None);
        assert!(!openai_request.stream);
        assert_eq!(
            openai_request.response_format,
            OpenAIResponseFormat::JsonObject
        );
        assert!(openai_request.tools.is_some());
        assert_eq!(openai_request.tools.as_ref().unwrap().len(), 1);
        assert_eq!(
            openai_request.tool_choice,
            Some(OpenAIToolChoice::String(OpenAIToolChoiceString::Auto))
        );
        assert_eq!(openai_request.parallel_tool_calls, Some(true));
    }

    #[test]
    fn test_try_from_openai_response() {
        // Test case 1: Valid response with content
        let valid_response = OpenAIResponse {
            choices: vec![OpenAIResponseChoice {
                index: 0,
                message: OpenAIResponseMessage {
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

        let result = ModelInferenceResponse::try_from(OpenAIResponseWithLatency {
            response: valid_response,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.content,
            vec![ContentBlock::Text("Hello, world!".to_string())]
        );
        assert_eq!(inference_response.usage.prompt_tokens, 10);
        assert_eq!(inference_response.usage.completion_tokens, 20);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(100)
            }
        );

        // Test case 2: Valid response with tool calls
        let valid_response_with_tools = OpenAIResponse {
            choices: vec![OpenAIResponseChoice {
                index: 0,
                message: OpenAIResponseMessage {
                    content: None,
                    tool_calls: Some(vec![OpenAIResponseToolCall {
                        id: "call1".to_string(),
                        r#type: OpenAIToolType::Function,
                        function: OpenAIResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
            }],
            usage: OpenAIUsage {
                prompt_tokens: 15,
                completion_tokens: 25,
                total_tokens: 40,
            },
        };

        let result = ModelInferenceResponse::try_from(OpenAIResponseWithLatency {
            response: valid_response_with_tools,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(110),
            },
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.content,
            vec![ContentBlock::ToolCall(ToolCall {
                id: "call1".to_string(),
                name: "test_function".to_string(),
                arguments: "{}".to_string(),
            })]
        );
        assert_eq!(inference_response.usage.prompt_tokens, 15);
        assert_eq!(inference_response.usage.completion_tokens, 25);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(110)
            }
        );

        // Test case 3: Invalid response with no choices
        let invalid_response_no_choices = OpenAIResponse {
            choices: vec![],
            usage: OpenAIUsage {
                prompt_tokens: 5,
                completion_tokens: 0,
                total_tokens: 5,
            },
        };

        let result = ModelInferenceResponse::try_from(OpenAIResponseWithLatency {
            response: invalid_response_no_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(120),
            },
        });
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::OpenAIServer { .. }));

        // Test case 4: Invalid response with multiple choices
        let invalid_response_multiple_choices = OpenAIResponse {
            choices: vec![
                OpenAIResponseChoice {
                    index: 0,
                    message: OpenAIResponseMessage {
                        content: Some("Choice 1".to_string()),
                        tool_calls: None,
                    },
                },
                OpenAIResponseChoice {
                    index: 1,
                    message: OpenAIResponseMessage {
                        content: Some("Choice 2".to_string()),
                        tool_calls: None,
                    },
                },
            ],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
                total_tokens: 20,
            },
        };

        let result = ModelInferenceResponse::try_from(OpenAIResponseWithLatency {
            response: invalid_response_multiple_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(130),
            },
        });
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::OpenAIServer { .. }));
    }
}
