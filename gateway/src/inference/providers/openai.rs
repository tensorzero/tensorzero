use futures::stream::Stream;
use futures::StreamExt;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, JSONMode, Latency, ModelInferenceRequest,
    ModelInferenceResponseStream, ProviderInferenceResponse, ProviderInferenceResponseChunk,
    RequestMessage, Role, Text, TextChunk, Usage,
};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1/";

#[derive(Debug)]
pub struct OpenAIProvider {
    pub model_name: String,
    pub api_base: Option<String>,
    pub api_key: Option<SecretString>,
}

impl InferenceProvider for OpenAIProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<ProviderInferenceResponse, Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "OpenAI".to_string(),
        })?;
        let request_body = OpenAIRequest::new(&self.model_name, request);
        let request_url = get_chat_url(self.api_base.as_deref())?;
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
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<(ProviderInferenceResponseChunk, ModelInferenceResponseStream), Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "OpenAI".to_string(),
        })?;
        let request_body = OpenAIRequest::new(&self.model_name, request);
        let request_url = get_chat_url(self.api_base.as_deref())?;
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
) -> impl Stream<Item = Result<ProviderInferenceResponseChunk, Error>> {
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
                            openai_to_tensorzero_chunk(d, inference_id, latency, &mut tool_call_ids, &mut tool_call_names)
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
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => Err(Error::OpenAIClient {
            message: response_body.to_string(),
            status_code: response_code,
        }),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
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
        .flat_map(tensorzero_to_openai_messages)
        .collect();
    if let Some(system_msg) = tensorzero_to_openai_system_message(request.system.as_deref()) {
        messages.insert(0, system_msg);
    }
    messages
}

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to OpenAI format
pub(super) fn prepare_openai_tools<'a>(
    request: &'a ModelInferenceRequest,
) -> (
    Option<Vec<OpenAITool<'a>>>,
    Option<OpenAIToolChoice<'a>>,
    Option<bool>,
) {
    match request.tool_config {
        None => (None, None, None),
        Some(tool_config) => {
            if tool_config.tools_available.is_empty() {
                return (None, None, None);
            }
            let tools = Some(
                tool_config
                    .tools_available
                    .iter()
                    .map(|tool| tool.into())
                    .collect(),
            );
            let tool_choice = Some((&tool_config.tool_choice).into());
            let parallel_tool_calls = Some(tool_config.parallel_tool_calls);
            (tools, tool_choice, parallel_tool_calls)
        }
    }
}

fn tensorzero_to_openai_system_message(system: Option<&str>) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: instructions,
        })
    })
}

fn tensorzero_to_openai_messages(message: &RequestMessage) -> Vec<OpenAIRequestMessage<'_>> {
    let mut messages = Vec::new();

    for block in message.content.iter() {
        match block {
            ContentBlock::Text(Text { text }) => match message.role {
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
                }
            },
            ContentBlock::ToolCall(tool_call) => {
                let tool_call = OpenAIRequestToolCall {
                    id: &tool_call.id,
                    r#type: OpenAIToolType::Function,
                    function: OpenAIRequestFunctionCall {
                        name: &tool_call.name,
                        arguments: &tool_call.arguments,
                    },
                };

                messages.push(OpenAIRequestMessage::Assistant(
                    OpenAIAssistantRequestMessage {
                        content: None,
                        tool_calls: Some(vec![tool_call]),
                    },
                ));
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

    messages
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum OpenAIResponseFormat {
    JsonObject,
    #[default]
    Text,
    JsonSchema {
        json_schema: Value,
    },
}

impl OpenAIResponseFormat {
    fn new(json_mode: &JSONMode, output_schema: Option<&Value>) -> Self {
        match json_mode {
            JSONMode::On => OpenAIResponseFormat::JsonObject,
            JSONMode::Off => OpenAIResponseFormat::Text,
            JSONMode::Strict => match output_schema {
                Some(schema) => {
                    let json_schema = json!({"name": "response", "schema": schema.clone()});
                    OpenAIResponseFormat::JsonSchema { json_schema }
                }
                None => OpenAIResponseFormat::JsonObject,
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum OpenAIToolType {
    Function,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct OpenAIFunction<'a> {
    pub(super) name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<&'a str>,
    pub parameters: &'a Value,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct OpenAITool<'a> {
    pub(super) r#type: OpenAIToolType,
    pub(super) function: OpenAIFunction<'a>,
    pub(super) strict: bool,
}

impl<'a> From<&'a ToolConfig> for OpenAITool<'a> {
    fn from(tool: &'a ToolConfig) -> Self {
        OpenAITool {
            r#type: OpenAIToolType::Function,
            function: OpenAIFunction {
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
    pub(super) r#type: OpenAIToolType,
    pub(super) function: SpecificToolFunction<'a>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct SpecificToolFunction<'a> {
    pub(super) name: &'a str,
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

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// This struct defines the supported parameters for the OpenAI API
/// See the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Serialize)]
struct OpenAIRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
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
        let response_format = OpenAIResponseFormat::new(&request.json_mode, request.output_schema);
        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };
        let messages = prepare_openai_messages(request);

        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(request);
        OpenAIRequest {
            messages,
            model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            seed: request.seed,
            stream: request.stream,
            stream_options,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
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

impl TryFrom<OpenAIResponseWithLatency> for ProviderInferenceResponse {
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
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlock::ToolCall(tool_call.into()));
            }
        }

        Ok(ProviderInferenceResponse::new(content, raw, usage, latency))
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
#[derive(Debug, PartialEq, Deserialize, Serialize, Clone)]
struct OpenAIDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Debug, PartialEq, Deserialize, Serialize, Clone)]
struct OpenAIChatChunkChoice {
    delta: OpenAIDelta,
}

#[derive(Debug, PartialEq, Deserialize, Serialize, Clone)]
struct OpenAIChatChunk {
    choices: Vec<OpenAIChatChunkChoice>,
    usage: Option<OpenAIUsage>,
}

/// Maps an OpenAI chunk to a TensorZero chunk for streaming inferences
fn openai_to_tensorzero_chunk(
    mut chunk: OpenAIChatChunk,
    inference_id: Uuid,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    tool_names: &mut Vec<String>,
) -> Result<ProviderInferenceResponseChunk, Error> {
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
                            message: "Tool call index out of bounds (meaning we haven't seen this many ids in the stream)".to_string(),
                        })?.clone()
                    }
                };
                let name = match tool_call.function.name {
                    Some(name) => {tool_names.push(name.clone()); name}
                    None => {
                        tool_names.get(index as usize).ok_or(Error::OpenAIServer {
                            message: "Tool call index out of bounds (meaning we haven't seen this many names in the stream)".to_string(),
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

    Ok(ProviderInferenceResponseChunk::new(
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

    use crate::{
        inference::{
            providers::common::{MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL, WEATHER_TOOL_CONFIG},
            types::FunctionType,
        },
        tool::ToolCallConfig,
    };

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
            stream: true,
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let openai_request = OpenAIRequest::new("gpt-3.5-turbo", &basic_request);

        assert_eq!(openai_request.model, "gpt-3.5-turbo");
        assert_eq!(openai_request.messages.len(), 2);
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_tokens, Some(100));
        assert_eq!(openai_request.seed, Some(69));
        assert!(openai_request.stream);
        assert_eq!(openai_request.response_format, OpenAIResponseFormat::Text);
        assert!(openai_request.tools.is_none());
        assert_eq!(openai_request.tool_choice, None);
        assert!(openai_request.parallel_tool_calls.is_none());

        // Test request with tools and JSON mode
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: JSONMode::On,
            tool_config: Some(&WEATHER_TOOL_CONFIG),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools);

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 1);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_tokens, None);
        assert_eq!(openai_request.seed, None);
        assert!(!openai_request.stream);
        assert_eq!(
            openai_request.response_format,
            OpenAIResponseFormat::JsonObject
        );
        assert!(openai_request.tools.is_some());
        let tools = openai_request.tools.as_ref().unwrap();
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            openai_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );

        // Test request with strict JSON mode with no output schema
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: JSONMode::Strict,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools);

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 1);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_tokens, None);
        assert!(!openai_request.stream);
        assert_eq!(openai_request.seed, None);
        // Resolves to normal JSON mode since no schema is provided (this shouldn't really happen in practice)
        assert_eq!(
            openai_request.response_format,
            OpenAIResponseFormat::JsonObject
        );

        // Test request with strict JSON mode with an output schema
        let output_schema = json!({});
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: JSONMode::Strict,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
        };

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools);

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 1);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_tokens, None);
        assert_eq!(openai_request.seed, None);
        assert!(!openai_request.stream);
        let expected_schema = serde_json::json!({"name": "response", "schema": {}});
        assert_eq!(
            openai_request.response_format,
            OpenAIResponseFormat::JsonSchema {
                json_schema: expected_schema,
            }
        );
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

        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithLatency {
            response: valid_response,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.content,
            vec!["Hello, world!".to_string().into()]
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

        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithLatency {
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

        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithLatency {
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

        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithLatency {
            response: invalid_response_multiple_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(130),
            },
        });
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::OpenAIServer { .. }));
    }

    #[test]
    fn test_prepare_openai_tools() {
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: JSONMode::On,
            tool_config: Some(&MULTI_TOOL_CONFIG),
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(&request_with_tools);
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(tools[1].function.name, QUERY_TOOL.name());
        assert_eq!(tools[1].function.parameters, QUERY_TOOL.parameters());
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            OpenAIToolChoice::String(OpenAIToolChoiceString::Required)
        );
        let parallel_tool_calls = parallel_tool_calls.unwrap();
        assert!(parallel_tool_calls);
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: true,
        };

        // Test no tools but a tool choice and make sure tool choice output is None
        let request_without_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: JSONMode::On,
            tool_config: Some(&tool_config),
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let (tools, tool_choice, parallel_tool_calls) =
            prepare_openai_tools(&request_without_tools);
        assert!(tools.is_none());
        assert!(tool_choice.is_none());
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_tensorzero_to_openai_messages() {
        let simple_request_message = RequestMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into()],
        };
        let openai_messages = tensorzero_to_openai_messages(&simple_request_message);
        assert_eq!(openai_messages.len(), 1);
        match &openai_messages[0] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(content.content, "Hello");
            }
            _ => panic!("Expected a user message"),
        }

        // Message with multiple blocks
        let multi_block_message = RequestMessage {
            role: Role::User,
            content: vec![
                "Hello".to_string().into(),
                "How are you?".to_string().into(),
            ],
        };
        let openai_messages = tensorzero_to_openai_messages(&multi_block_message);
        assert_eq!(openai_messages.len(), 2);
        match &openai_messages[0] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(content.content, "Hello");
            }
            _ => panic!("Expected a user message"),
        }
        match &openai_messages[1] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(content.content, "How are you?");
            }
            _ => panic!("Expected a user message"),
        }

        // User message with one string and one tool call block
        // Since user messages in OpenAI land can't contain tool calls (nor should they honestly),
        // We split the tool call out into a separate assistant message
        let tool_block = ContentBlock::ToolCall(ToolCall {
            id: "call1".to_string(),
            name: "test_function".to_string(),
            arguments: "{}".to_string(),
        });
        let multi_block_message = RequestMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into(), tool_block],
        };
        let openai_messages = tensorzero_to_openai_messages(&multi_block_message);
        assert_eq!(openai_messages.len(), 2);
        match &openai_messages[0] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(content.content, "Hello");
            }
            _ => panic!("Expected a user message"),
        }
        match &openai_messages[1] {
            OpenAIRequestMessage::Assistant(content) => {
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
    fn test_openai_to_tensorzero_chunk() {
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
                delta: OpenAIDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let mut tool_call_names = vec!["name1".to_string()];
        let inference_id = Uuid::now_v7();
        let message = openai_to_tensorzero_chunk(
            chunk.clone(),
            inference_id,
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
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
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
                delta: OpenAIDelta {
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCallChunk {
                        index: 0,
                        id: None,
                        function: OpenAIFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message = openai_to_tensorzero_chunk(
            chunk.clone(),
            inference_id,
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "id1".to_string(),
                name: "name1".to_string(),
                arguments: "{\"hello\":\"world\"}".to_string(),
            })]
        );
        // Test what a bad tool chunk would do (new ID but no names)
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
                delta: OpenAIDelta {
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCallChunk {
                        index: 1,
                        id: None,
                        function: OpenAIFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let error = openai_to_tensorzero_chunk(
            chunk.clone(),
            inference_id,
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
        )
        .unwrap_err();
        assert_eq!(
            error,
            Error::OpenAIServer {
                message: "Tool call index out of bounds (meaning we haven't seen this many ids in the stream)".to_string(),
            }
        );
        // Test a correct new tool chunk
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
                delta: OpenAIDelta {
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCallChunk {
                        index: 1,
                        id: Some("id2".to_string()),
                        function: OpenAIFunctionCallChunk {
                            name: Some("name2".to_string()),
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message = openai_to_tensorzero_chunk(
            chunk.clone(),
            inference_id,
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "id2".to_string(),
                name: "name2".to_string(),
                arguments: "{\"hello\":\"world\"}".to_string(),
            })]
        );
        // Check that the lists were updated
        assert_eq!(tool_call_ids, vec!["id1".to_string(), "id2".to_string()]);
        assert_eq!(
            tool_call_names,
            vec!["name1".to_string(), "name2".to_string()]
        );

        // Check a chunk with no choices and only usage
        // Test a correct new tool chunk
        let chunk = OpenAIChatChunk {
            choices: vec![],
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };
        let message = openai_to_tensorzero_chunk(
            chunk.clone(),
            inference_id,
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
        )
        .unwrap();
        assert_eq!(message.content, vec![]);
        assert_eq!(
            message.usage,
            Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
            })
        );
    }

    #[test]
    fn test_new_openai_response_format() {
        // Test JSON mode On
        let json_mode = JSONMode::On;
        let output_schema = None;
        let format = OpenAIResponseFormat::new(&json_mode, output_schema);
        assert_eq!(format, OpenAIResponseFormat::JsonObject);

        // Test JSON mode Off
        let json_mode = JSONMode::Off;
        let format = OpenAIResponseFormat::new(&json_mode, output_schema);
        assert_eq!(format, OpenAIResponseFormat::Text);

        // Test JSON mode Strict with no schema
        let json_mode = JSONMode::Strict;
        let format = OpenAIResponseFormat::new(&json_mode, output_schema);
        assert_eq!(format, OpenAIResponseFormat::JsonObject);

        // Test JSON mode Strict with schema
        let json_mode = JSONMode::Strict;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"type": "string"}
            }
        });
        let output_schema = Some(&schema);
        let format = OpenAIResponseFormat::new(&json_mode, output_schema);
        match format {
            OpenAIResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema["schema"], schema);
                assert_eq!(json_schema["name"], "response");
            }
            _ => panic!("Expected JsonSchema format"),
        }
    }
}
