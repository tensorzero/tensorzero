use futures::{Stream, StreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tokio::time::Instant;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{ContentBlock, ContentBlockChunk, Latency, Role, Text};
use crate::inference::types::{
    ModelInferenceRequest, ModelInferenceResponse, ModelInferenceResponseChunk,
    ModelInferenceResponseStream, RequestMessage, TextChunk, Usage,
};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

#[derive(Debug)]
pub struct AnthropicProvider {
    pub model_name: String,
    pub api_key: Option<SecretString>,
}

impl InferenceProvider for AnthropicProvider {
    /// Anthropic non-streaming API request
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Anthropic".to_string(),
        })?;
        let request_body = AnthropicRequestBody::new(&self.model_name, request)?;
        let start_time = Instant::now();
        let res = http_client
            .post(ANTHROPIC_BASE_URL)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("x-api-key", api_key.expose_secret())
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request: {e}"),
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| Error::AnthropicServer {
                message: format!("Error parsing text response: {e}"),
            })?;

            let response = serde_json::from_str(&response).map_err(|e| Error::AnthropicServer {
                message: format!("Error parsing JSON response: {e}: {response}"),
            })?;

            let response_with_latency = AnthropicResponseWithLatency { response, latency };
            Ok(response_with_latency.try_into()?)
        } else {
            let response_code = res.status();
            let error_body =
                res.json::<AnthropicError>()
                    .await
                    .map_err(|e| Error::AnthropicServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            handle_anthropic_error(response_code, error_body.error)
        }
    }

    /// Anthropic streaming API request
    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, ModelInferenceResponseStream), Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Anthropic".to_string(),
        })?;
        let request_body = AnthropicRequestBody::new(&self.model_name, request)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(ANTHROPIC_BASE_URL)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("content-type", "application/json")
            .header("x-api-key", api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to Anthropic: {e}"),
            })?;
        let mut stream = Box::pin(stream_anthropic(event_source, start_time));
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::AnthropicServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };
        Ok((chunk, stream))
    }
}

/// Maps events from Anthropic into the TensorZero format
/// Modified from the example [here](https://github.com/64bit/async-openai/blob/5c9c817b095e3bacb2b6c9804864cdf8b15c795e/async-openai/src/client.rs#L433)
/// At a high level, this function is handling low-level EventSource details and mapping the objects returned by Anthropic into our `InferenceResultChunk` type

fn stream_anthropic(
    mut event_source: EventSource,
    start_time: Instant,
) -> impl Stream<Item = Result<ModelInferenceResponseChunk, Error>> {
    async_stream::stream! {
        let inference_id = Uuid::now_v7();
        let mut current_tool_id : Option<String> = None;
        let mut current_tool_name: Option<String> = None;
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(Error::AnthropicServer {
                        message: e.to_string(),
                    });
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<AnthropicStreamMessage, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::AnthropicServer {
                                message: format!(
                                    "Error parsing message: {}, Data: {}",
                                    e, message.data
                                ),
                            });
                        // Anthropic streaming API docs specify that this is the last message
                        if let Ok(AnthropicStreamMessage::MessageStop) = data {
                            break;
                        }

                        let response = data.and_then(|data| {
                            anthropic_to_tensorzero_stream_message(
                                data,
                                inference_id,
                                start_time.elapsed(),
                                &mut current_tool_id,
                                &mut current_tool_name,
                            )
                        });

                        match response {
                            Ok(None) => {},
                            Ok(Some(stream_message)) => yield Ok(stream_message),
                            Err(e) => yield Err(e),
                        }
                    }
                },
            }
        }

        event_source.close();
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
/// Anthropic doesn't handle the system message in this way
/// It's a field of the POST body instead
enum AnthropicRole {
    User,
    Assistant,
}

impl From<Role> for AnthropicRole {
    fn from(role: Role) -> Self {
        match role {
            Role::User => AnthropicRole::User,
            Role::Assistant => AnthropicRole::Assistant,
        }
    }
}

/// We can instruct Anthropic to use a particular tool,
/// any tool (but to use one), or to use a tool if needed.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum AnthropicToolChoice<'a> {
    Auto,
    Any,
    Tool { name: &'a str },
}

// We map our ToolChoice enum to the Anthropic one that serializes properly
impl<'a> TryFrom<&'a ToolChoice> for AnthropicToolChoice<'a> {
    type Error = Error;
    fn try_from(tool_choice: &'a ToolChoice) -> Result<Self, Error> {
        match tool_choice {
            ToolChoice::Auto => Ok(AnthropicToolChoice::Auto),
            ToolChoice::Required => Ok(AnthropicToolChoice::Any),
            ToolChoice::Specific(name) => Ok(AnthropicToolChoice::Tool { name }),
            ToolChoice::None => Err(Error::InvalidTool {
                message: "Tool choice is None. Anthropic does not support tool choice None."
                    .to_string(),
            }),
            ToolChoice::Implicit => Ok(AnthropicToolChoice::Tool { name: "respond" }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct AnthropicTool<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    input_schema: &'a Value,
}

impl<'a> From<&'a ToolConfig> for AnthropicTool<'a> {
    fn from(value: &'a ToolConfig) -> Self {
        // In case we add more tool types in the future, the compiler will complain here.
        AnthropicTool {
            name: value.name(),
            description: Some(value.description()),
            input_schema: value.parameters(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
// NB: Anthropic also supports Image blocks here but we won't for now
enum AnthropicMessageContent<'a> {
    Text {
        text: &'a str,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: Vec<AnthropicMessageContent<'a>>,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: Value,
    },
}

impl<'a> TryFrom<&'a ContentBlock> for AnthropicMessageContent<'a> {
    type Error = Error;

    fn try_from(block: &'a ContentBlock) -> Result<Self, Self::Error> {
        match block {
            ContentBlock::Text(Text { text }) => Ok(AnthropicMessageContent::Text { text }),
            ContentBlock::ToolCall(tool_call) => {
                // Convert the tool call arguments from String to JSON Value (Anthropic expects an object)
                let input: Value = serde_json::from_str(&tool_call.arguments).map_err(|e| {
                    Error::AnthropicClient {
                        status_code: StatusCode::BAD_REQUEST,
                        message: format!("Error parsing tool call arguments as JSON Value: {e}"),
                    }
                })?;

                if !input.is_object() {
                    return Err(Error::AnthropicClient {
                        status_code: StatusCode::BAD_REQUEST,
                        message: "Tool call arguments must be a JSON object".to_string(),
                    });
                }

                Ok(AnthropicMessageContent::ToolUse {
                    id: &tool_call.id,
                    name: &tool_call.name,
                    input,
                })
            }
            ContentBlock::ToolResult(tool_result) => Ok(AnthropicMessageContent::ToolResult {
                tool_use_id: &tool_result.id,
                content: vec![AnthropicMessageContent::Text {
                    text: &tool_result.result,
                }],
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct AnthropicMessage<'a> {
    role: AnthropicRole,
    content: Vec<AnthropicMessageContent<'a>>,
}

impl<'a> TryFrom<&'a RequestMessage> for AnthropicMessage<'a> {
    type Error = Error;

    fn try_from(
        inference_message: &'a RequestMessage,
    ) -> Result<AnthropicMessage<'a>, Self::Error> {
        let content: Vec<AnthropicMessageContent> = inference_message
            .content
            .iter()
            .map(|block| block.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(AnthropicMessage {
            role: inference_message.role.into(),
            content,
        })
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct AnthropicRequestBody<'a> {
    model: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    // This is the system message
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
}

impl<'a> AnthropicRequestBody<'a> {
    fn new(
        model_name: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<AnthropicRequestBody<'a>, Error> {
        if request.messages.is_empty() {
            return Err(Error::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            });
        }
        let system = request.system.as_deref();
        let request_messages: Vec<AnthropicMessage> = request
            .messages
            .iter()
            .map(AnthropicMessage::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let messages = prepare_messages(request_messages)?;
        let tools = request
            .tool_config
            .map(|c| &c.tools_available)
            .map(|tools| tools.iter().map(|tool| tool.into()).collect::<Vec<_>>());
        // `tool_choice` should only be set if tools are set and non-empty
        let tool_choice: Option<AnthropicToolChoice> = tools
            .as_ref()
            .filter(|t| !t.is_empty())
            .and(request.tool_config)
            .and_then(|c| (&c.tool_choice).try_into().ok());
        // NOTE: Anthropic does not support seed
        Ok(AnthropicRequestBody {
            model: model_name,
            messages,
            max_tokens: request.max_tokens.unwrap_or(4096),
            stream: Some(request.stream),
            system,
            temperature: request.temperature,
            tool_choice,
            tools,
        })
    }
}

/// Anthropic API doesn't support consecutive messages from the same role.
/// This function consolidates messages from the same role into a single message
/// so as to satisfy the API.
/// It also makes modifications to the messages to make Anthropic happy.
/// For example, it will prepend a default User message if the first message is an Assistant message.
/// It will also append a default User message if the last message is an Assistant message.
fn prepare_messages(messages: Vec<AnthropicMessage>) -> Result<Vec<AnthropicMessage>, Error> {
    let mut consolidated_messages: Vec<AnthropicMessage> = Vec::new();
    let mut last_role: Option<AnthropicRole> = None;
    for message in messages {
        let this_role = message.role.clone();
        match last_role {
            Some(role) => {
                if role == this_role {
                    let mut last_message =
                        consolidated_messages.pop().ok_or(Error::InvalidRequest {
                            message: "Last message is missing (this should never happen)"
                                .to_string(),
                        })?;
                    last_message.content.extend(message.content);
                    consolidated_messages.push(last_message);
                } else {
                    consolidated_messages.push(message);
                }
            }
            None => {
                consolidated_messages.push(message);
            }
        }
        last_role = Some(this_role)
    }
    // Anthropic also requires that there is at least one message and it is a User message.
    // If it's not we will prepend a default User message.
    match consolidated_messages.first() {
        Some(&AnthropicMessage {
            role: AnthropicRole::User,
            ..
        }) => {}
        _ => {
            consolidated_messages.insert(
                0,
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![AnthropicMessageContent::Text {
                        text: "[listening]",
                    }],
                },
            );
        }
    }
    // Anthropic will continue any assistant messages passed in.
    // Since we don't want to do that, we'll append a default User message in the case that the last message was
    // an assistant message
    if let Some(last_message) = consolidated_messages.last() {
        if last_message.role == AnthropicRole::Assistant {
            consolidated_messages.push(AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    text: "[listening]",
                }],
            });
        }
    }
    Ok(consolidated_messages)
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct AnthropicError {
    error: AnthropicErrorBody,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct AnthropicErrorBody {
    r#type: String,
    message: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

impl TryFrom<AnthropicContentBlock> for ContentBlock {
    type Error = Error;
    fn try_from(block: AnthropicContentBlock) -> Result<Self, Self::Error> {
        match block {
            AnthropicContentBlock::Text { text } => Ok(text.into()),
            AnthropicContentBlock::ToolUse { id, name, input } => {
                Ok(ContentBlock::ToolCall(ToolCall {
                    id,
                    name,
                    arguments: serde_json::to_string(&input).map_err(|e| {
                        Error::AnthropicServer {
                            message: format!("Error parsing input for tool call: {e}"),
                        }
                    })?,
                }))
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl From<AnthropicUsage> for Usage {
    fn from(value: AnthropicUsage) -> Self {
        Usage {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct AnthropicResponse {
    id: String,
    r#type: String, // this is always "message"
    role: String,   // this is always "assistant"
    content: Vec<AnthropicContentBlock>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct AnthropicResponseWithLatency {
    response: AnthropicResponse,
    latency: Latency,
}

impl TryFrom<AnthropicResponseWithLatency> for ModelInferenceResponse {
    type Error = Error;
    fn try_from(value: AnthropicResponseWithLatency) -> Result<Self, Self::Error> {
        let AnthropicResponseWithLatency { response, latency } = value;

        let raw_response =
            serde_json::to_string(&response).map_err(|e| Error::AnthropicServer {
                message: format!("Error parsing response from Anthropic: {e}"),
            })?;

        let content: Vec<ContentBlock> = response
            .content
            .into_iter()
            .map(|block| block.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ModelInferenceResponse::new(
            content,
            raw_response,
            response.usage.into(),
            latency,
        ))
    }
}

fn handle_anthropic_error(
    response_code: StatusCode,
    response_body: AnthropicErrorBody,
) -> Result<ModelInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED
        | StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::TOO_MANY_REQUESTS => Err(Error::AnthropicClient {
            message: response_body.message,
            status_code: response_code,
        }),
        // StatusCode::NOT_FOUND | StatusCode::FORBIDDEN | StatusCode::INTERNAL_SERVER_ERROR | 529: Overloaded
        // These are all captured in _ since they have the same error behavior
        _ => Err(Error::AnthropicServer {
            message: response_body.message,
        }),
    }
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicMessageBlock {
    Text {
        text: String,
    },
    TextDelta {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    InputJsonDelta {
        partial_json: String,
    },
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamMessage {
    ContentBlockDelta {
        delta: AnthropicMessageBlock,
        index: u32,
    },
    ContentBlockStart {
        content_block: AnthropicMessageBlock,
        index: u32,
    },
    ContentBlockStop {
        index: u32,
    },
    Error {
        error: Value,
    },
    MessageDelta {
        delta: Value,
        usage: Value,
    },
    MessageStart {
        message: Value,
    },
    MessageStop,
    Ping,
}

/// This function converts an Anthropic stream message to a TensorZero stream message.
/// It must keep track of the current tool ID and name in order to correctly handle ToolCallChunks (which we force to always contain the tool name and ID)
/// Anthropic only sends the tool ID and name in the ToolUse chunk so we need to keep the most recent ones as mutable references so
/// subsequent InputJSONDelta chunks can be initialized with this information as well.
/// There is no need to do the same bookkeeping for TextDelta chunks since they come with an index (which we use as an ID for a text chunk).
/// See the Anthropic [docs](https://docs.anthropic.com/en/api/messages-streaming) on streaming messages for details on the types of events and their semantics.
fn anthropic_to_tensorzero_stream_message(
    message: AnthropicStreamMessage,
    inference_id: Uuid,
    message_latency: Duration,
    current_tool_id: &mut Option<String>,
    current_tool_name: &mut Option<String>,
) -> Result<Option<ModelInferenceResponseChunk>, Error> {
    let raw_message = serde_json::to_string(&message).map_err(|e| Error::AnthropicServer {
        message: format!("Error parsing response from Anthropic: {e}"),
    })?;
    match message {
        AnthropicStreamMessage::ContentBlockDelta { delta, index } => match delta {
            AnthropicMessageBlock::TextDelta { text } => {
                Ok(Some(ModelInferenceResponseChunk::new(
                    inference_id,
                    vec![ContentBlockChunk::Text(TextChunk {
                        text,
                        id: index.to_string(),
                    })],
                    None,
                    raw_message,
                    message_latency,
                )))
            }
            AnthropicMessageBlock::InputJsonDelta { partial_json } => {
                Ok(Some(ModelInferenceResponseChunk::new(
                    inference_id,
                    // Take the current tool name and ID and use them to create a ToolCallChunk
                    // This is necessary because the ToolCallChunk must always contain the tool name and ID
                    // even though Anthropic only sends the tool ID and name in the ToolUse chunk and not InputJSONDelta
                    vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                        name: current_tool_name.clone().ok_or(Error::AnthropicServer {
                            message: "Got InputJsonDelta chunk from Anthropic without current tool name being set by a ToolUse".to_string(),
                        })?,
                        id: current_tool_id.clone().ok_or(Error::AnthropicServer {
                            message: "Got InputJsonDelta chunk from Anthropic without current tool id being set by a ToolUse".to_string(),
                        })?,
                        arguments: partial_json,
                    })],
                    None,
                    raw_message,
                    message_latency,
                )))
            }
            _ => Err(Error::AnthropicServer {
                message: "Unsupported content block type for ContentBlockDelta".to_string(),
            }),
        },
        AnthropicStreamMessage::ContentBlockStart {
            content_block,
            index,
        } => match content_block {
            AnthropicMessageBlock::Text { text } => {
                let text_chunk = ContentBlockChunk::Text(TextChunk {
                    text,
                    id: index.to_string(),
                });
                Ok(Some(ModelInferenceResponseChunk::new(
                    inference_id,
                    vec![text_chunk],
                    None,
                    raw_message,
                    message_latency,
                )))
            }
            AnthropicMessageBlock::ToolUse { id, name, .. } => {
                // This is a new tool call, update the ID for future chunks
                *current_tool_id = Some(id.clone());
                *current_tool_name = Some(name.clone());
                Ok(Some(ModelInferenceResponseChunk::new(
                    inference_id,
                    vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                        id,
                        name,
                        // As far as I can tell this is always {} so we ignore
                        arguments: "".to_string(),
                    })],
                    None,
                    raw_message,
                    message_latency,
                )))
            }
            _ => Err(Error::AnthropicServer {
                message: "Unsupported content block type for ContentBlockStart".to_string(),
            }),
        },
        AnthropicStreamMessage::ContentBlockStop { .. } => Ok(None),
        AnthropicStreamMessage::Error { error } => Err(Error::AnthropicServer {
            message: error.to_string(),
        }),
        AnthropicStreamMessage::MessageDelta { usage, .. } => {
            let usage = parse_usage_info(&usage);
            Ok(Some(ModelInferenceResponseChunk::new(
                inference_id,
                vec![],
                Some(usage.into()),
                raw_message,
                message_latency,
            )))
        }
        AnthropicStreamMessage::MessageStart { message } => {
            if let Some(usage_info) = message.get("usage") {
                let usage = parse_usage_info(usage_info);
                Ok(Some(ModelInferenceResponseChunk::new(
                    inference_id,
                    vec![],
                    Some(usage.into()),
                    raw_message,
                    message_latency,
                )))
            } else {
                Ok(None)
            }
        }
        AnthropicStreamMessage::MessageStop | AnthropicStreamMessage::Ping {} => Ok(None),
    }
}

fn parse_usage_info(usage_info: &Value) -> AnthropicUsage {
    let input_tokens = usage_info
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let output_tokens = usage_info
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    AnthropicUsage {
        input_tokens,
        output_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    use crate::inference::providers::common::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::types::{FunctionType, JSONMode};
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::tool::{DynamicToolConfig, ToolConfig, ToolResult};

    #[test]
    fn test_try_from_tool_choice() {
        // Need to cover all 4 cases
        let tool_choice = ToolChoice::None;
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_err());
        assert_eq!(
            anthropic_tool_choice.err().unwrap(),
            Error::InvalidTool {
                message: "Tool choice is None. Anthropic does not support tool choice None."
                    .to_string(),
            }
        );

        let tool_choice = ToolChoice::Auto;
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(anthropic_tool_choice.unwrap(), AnthropicToolChoice::Auto);

        let tool_choice = ToolChoice::Required;
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(anthropic_tool_choice.unwrap(), AnthropicToolChoice::Any);

        let tool_choice = ToolChoice::Specific("test".to_string());
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            AnthropicToolChoice::Tool { name: "test" }
        );

        let tool_choice = ToolChoice::Implicit;
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            AnthropicToolChoice::Tool { name: "respond" }
        );
    }

    #[tokio::test]
    async fn test_from_tool() {
        let parameters = json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            },
            "required": ["location", "unit"]
        });
        let tool = ToolConfig::Dynamic(DynamicToolConfig {
            name: "test".to_string(),
            description: "test".to_string(),
            parameters: DynamicJSONSchema::new(parameters.clone()),
            strict: false,
        });
        let anthropic_tool: AnthropicTool = (&tool).into();
        assert_eq!(
            anthropic_tool,
            AnthropicTool {
                name: "test",
                description: Some("test"),
                input_schema: &parameters,
            }
        );
    }

    #[test]
    fn test_try_from_content_block() {
        let text_content_block = "test".to_string().into();
        let anthropic_content_block =
            AnthropicMessageContent::try_from(&text_content_block).unwrap();
        assert_eq!(
            anthropic_content_block,
            AnthropicMessageContent::Text { text: "test" }
        );

        let tool_call_content_block = ContentBlock::ToolCall(ToolCall {
            id: "test_id".to_string(),
            name: "test_name".to_string(),
            arguments: serde_json::to_string(&json!({"type": "string"})).unwrap(),
        });
        let anthropic_content_block =
            AnthropicMessageContent::try_from(&tool_call_content_block).unwrap();
        assert_eq!(
            anthropic_content_block,
            AnthropicMessageContent::ToolUse {
                id: "test_id",
                name: "test_name",
                input: json!({"type": "string"})
            }
        );
    }

    #[test]
    fn test_try_from_request_message() {
        // Test a User message
        let inference_request_message = RequestMessage {
            role: Role::User,
            content: vec!["test".to_string().into()],
        };
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message).unwrap();
        assert_eq!(
            anthropic_message,
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "test" }],
            }
        );

        // Test an Assistant message
        let inference_request_message = RequestMessage {
            role: Role::Assistant,
            content: vec!["test_assistant".to_string().into()],
        };
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message).unwrap();
        assert_eq!(
            anthropic_message,
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    text: "test_assistant",
                }],
            }
        );

        // Test a Tool message
        let inference_request_message = RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "test_tool_call_id".to_string(),
                name: "test_tool_name".to_string(),
                result: "test_tool_response".to_string(),
            })],
        };
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message).unwrap();
        assert_eq!(
            anthropic_message,
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::ToolResult {
                    tool_use_id: "test_tool_call_id",
                    content: vec![AnthropicMessageContent::Text {
                        text: "test_tool_response"
                    }],
                }],
            }
        );
    }

    #[test]
    fn test_initialize_anthropic_request_body() {
        let model = "claude".to_string();
        let listening_message = AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text {
                text: "[listening]",
            }],
        };
        // Test Case 1: Empty message list
        let inference_request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        assert!(anthropic_request_body.is_err());
        assert_eq!(
            anthropic_request_body.err().unwrap(),
            Error::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            }
        );

        // Test Case 2: Messages with System message
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
        ];
        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: &model,
                messages: vec![
                    AnthropicMessage::try_from(&messages[0]).unwrap(),
                    AnthropicMessage::try_from(&messages[1]).unwrap(),
                    listening_message.clone(),
                ],
                max_tokens: 4096,
                stream: Some(false),
                system: Some("test_system"),
                temperature: None,
                tool_choice: None,
                tools: None,
            }
        );

        // Test case 3: Messages with system message that require consolidation
        // also some of the optional fields are tested
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::User,
                content: vec!["test_user2".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
        ];
        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: None,
            stream: true,
            json_mode: JSONMode::On,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: &model,
                messages: vec![
                    AnthropicMessage {
                        role: AnthropicRole::User,
                        content: vec![
                            AnthropicMessageContent::Text { text: "test_user" },
                            AnthropicMessageContent::Text { text: "test_user2" }
                        ],
                    },
                    AnthropicMessage::try_from(&messages[2]).unwrap(),
                    listening_message.clone(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some("test_system"),
                temperature: Some(0.5),
                tool_choice: None,
                tools: None,
            }
        );

        // Test case 4: Tool use & choice
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
            RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(ToolResult {
                    id: "tool_call_id".to_string(),
                    name: "test_tool_name".to_string(),
                    result: "tool_response".to_string(),
                })],
            },
        ];

        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: Some(&WEATHER_TOOL_CONFIG),
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: None,
            stream: true,
            json_mode: JSONMode::On,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: &model,
                messages: vec![
                    AnthropicMessage::try_from(&messages[0]).unwrap(),
                    AnthropicMessage::try_from(&messages[1]).unwrap(),
                    AnthropicMessage::try_from(&messages[2]).unwrap(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some("test_system"),
                temperature: Some(0.5),
                tool_choice: Some(AnthropicToolChoice::Tool {
                    name: "get_temperature",
                }),
                tools: Some(vec![AnthropicTool {
                    name: WEATHER_TOOL.name(),
                    description: Some(WEATHER_TOOL.description()),
                    input_schema: WEATHER_TOOL.parameters(),
                }]),
            }
        );
    }

    #[test]
    fn test_consolidate_messages() {
        let listening_message = AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text {
                text: "[listening]",
            }],
        };
        // Test case 1: No consolidation needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "Hello" }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text { text: "Hi" }],
            },
        ];
        let expected = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "Hello" }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text { text: "Hi" }],
            },
            listening_message.clone(),
        ];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 2: Consolidation needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "Hello" }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    text: "How are you?",
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text { text: "Hi" }],
            },
        ];
        let expected = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![
                    AnthropicMessageContent::Text { text: "Hello" },
                    AnthropicMessageContent::Text {
                        text: "How are you?",
                    },
                ],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text { text: "Hi" }],
            },
            listening_message.clone(),
        ];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 3: Multiple consolidations needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "Hello" }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    text: "How are you?",
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text { text: "Hi" }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    text: "I am here to help.",
                }],
            },
        ];
        let expected = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![
                    AnthropicMessageContent::Text { text: "Hello" },
                    AnthropicMessageContent::Text {
                        text: "How are you?",
                    },
                ],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![
                    AnthropicMessageContent::Text { text: "Hi" },
                    AnthropicMessageContent::Text {
                        text: "I am here to help.",
                    },
                ],
            },
            listening_message.clone(),
        ];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 4: No messages
        let messages: Vec<AnthropicMessage> = vec![];
        let expected: Vec<AnthropicMessage> = vec![listening_message.clone()];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 5: Single message
        let messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text { text: "Hello" }],
        }];
        let expected = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text { text: "Hello" }],
        }];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 6: Consolidate tool uses
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![AnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::ToolResult {
                    tool_use_id: "tool2",
                    content: vec![AnthropicMessageContent::Text {
                        text: "Tool call 2",
                    }],
                }],
            },
        ];
        let expected = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![
                AnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![AnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                },
                AnthropicMessageContent::ToolResult {
                    tool_use_id: "tool2",
                    content: vec![AnthropicMessageContent::Text {
                        text: "Tool call 2",
                    }],
                },
            ],
        }];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 7: Consolidate mixed text and tool use
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    text: "User message 1",
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![AnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    text: "User message 2",
                }],
            },
        ];
        let expected = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![
                AnthropicMessageContent::Text {
                    text: "User message 1",
                },
                AnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![AnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                },
                AnthropicMessageContent::Text {
                    text: "User message 2",
                },
            ],
        }];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);
    }

    #[test]
    fn test_handle_anthropic_error() {
        let error_body = AnthropicErrorBody {
            r#type: "error".to_string(),
            message: "test_message".to_string(),
        };
        let response_code = StatusCode::BAD_REQUEST;
        let result = handle_anthropic_error(response_code, error_body.clone());
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            Error::AnthropicClient {
                message: "test_message".to_string(),
                status_code: response_code,
            }
        );
        let response_code = StatusCode::UNAUTHORIZED;
        let result = handle_anthropic_error(response_code, error_body.clone());
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            Error::AnthropicClient {
                message: "test_message".to_string(),
                status_code: response_code,
            }
        );
        let response_code = StatusCode::TOO_MANY_REQUESTS;
        let result = handle_anthropic_error(response_code, error_body.clone());
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            Error::AnthropicClient {
                message: "test_message".to_string(),
                status_code: response_code,
            }
        );
        let response_code = StatusCode::NOT_FOUND;
        let result = handle_anthropic_error(response_code, error_body.clone());
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            Error::AnthropicServer {
                message: "test_message".to_string(),
            }
        );
        let response_code = StatusCode::INTERNAL_SERVER_ERROR;
        let result = handle_anthropic_error(response_code, error_body.clone());
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap(),
            Error::AnthropicServer {
                message: "test_message".to_string(),
            }
        );
    }

    #[test]
    fn test_anthropic_usage_to_usage() {
        let anthropic_usage = AnthropicUsage {
            input_tokens: 100,
            output_tokens: 50,
        };

        let usage: Usage = anthropic_usage.into();

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
    }

    #[test]
    fn test_anthropic_response_conversion() {
        // Test case 1: Text response
        let anthropic_response_body = AnthropicResponse {
            id: "1".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock::Text {
                text: "Response text".to_string(),
            }],
            model: "model-name".to_string(),
            stop_reason: Some("stop reason".to_string()),
            stop_sequence: Some("stop sequence".to_string()),
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let body_with_latency = AnthropicResponseWithLatency {
            response: anthropic_response_body.clone(),
            latency: latency.clone(),
        };

        let inference_response = ModelInferenceResponse::try_from(body_with_latency).unwrap();
        assert_eq!(
            inference_response.content,
            vec!["Response text".to_string().into()]
        );

        let raw_json = json!(anthropic_response_body).to_string();
        assert_eq!(raw_json, inference_response.raw_response);
        assert_eq!(inference_response.usage.input_tokens, 100);
        assert_eq!(inference_response.usage.output_tokens, 50);
        assert_eq!(inference_response.latency, latency);

        // Test case 2: Tool call response
        let anthropic_response_body = AnthropicResponse {
            id: "2".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock::ToolUse {
                id: "tool_call_1".to_string(),
                name: "get_temperature".to_string(),
                input: json!({"location": "New York"}),
            }],
            model: "model-name".to_string(),
            stop_reason: Some("tool_call".to_string()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let body_with_latency = AnthropicResponseWithLatency {
            response: anthropic_response_body.clone(),
            latency: latency.clone(),
        };

        let inference_response: ModelInferenceResponse = body_with_latency.try_into().unwrap();
        assert!(inference_response.content.len() == 1);
        assert_eq!(
            inference_response.content[0],
            ContentBlock::ToolCall(ToolCall {
                id: "tool_call_1".to_string(),
                name: "get_temperature".to_string(),
                arguments: r#"{"location":"New York"}"#.to_string(),
            })
        );

        let raw_json = json!(anthropic_response_body).to_string();
        assert_eq!(raw_json, inference_response.raw_response);
        assert_eq!(inference_response.usage.input_tokens, 100);
        assert_eq!(inference_response.usage.output_tokens, 50);
        assert_eq!(inference_response.latency, latency);

        // Test case 3: Mixed response (text and tool call)
        let anthropic_response_body = AnthropicResponse {
            id: "3".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                AnthropicContentBlock::Text {
                    text: "Here's the weather:".to_string(),
                },
                AnthropicContentBlock::ToolUse {
                    id: "tool_call_2".to_string(),
                    name: "get_temperature".to_string(),
                    input: json!({"location": "London"}),
                },
            ],
            model: "model-name".to_string(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let body_with_latency = AnthropicResponseWithLatency {
            response: anthropic_response_body.clone(),
            latency: latency.clone(),
        };
        let inference_response = ModelInferenceResponse::try_from(body_with_latency).unwrap();
        assert_eq!(
            inference_response.content[0],
            "Here's the weather:".to_string().into()
        );
        assert!(inference_response.content.len() == 2);
        assert_eq!(
            inference_response.content[1],
            ContentBlock::ToolCall(ToolCall {
                id: "tool_call_2".to_string(),
                name: "get_temperature".to_string(),
                arguments: r#"{"location":"London"}"#.to_string(),
            })
        );

        let raw_json = json!(anthropic_response_body).to_string();
        assert_eq!(raw_json, inference_response.raw_response);

        assert_eq!(inference_response.usage.input_tokens, 100);
        assert_eq!(inference_response.usage.output_tokens, 50);
        assert_eq!(inference_response.latency, latency);
    }

    #[test]
    fn test_anthropic_to_tensorzero_stream_message() {
        use serde_json::json;
        use uuid::Uuid;

        let inference_id = Uuid::now_v7();

        // Test ContentBlockDelta with TextDelta
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_delta = AnthropicStreamMessage::ContentBlockDelta {
            delta: AnthropicMessageBlock::TextDelta {
                text: "Hello".to_string(),
            },
            index: 0,
        };
        let latency = Duration::from_millis(100);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_delta,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::Text(text) => {
                assert_eq!(text.text, "Hello".to_string());
                assert_eq!(text.id, "0".to_string());
            }
            _ => panic!("Expected a text content block"),
        }
        assert_eq!(chunk.latency, latency);

        // Test ContentBlockDelta with InputJsonDelta but no previous tool info
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_delta = AnthropicStreamMessage::ContentBlockDelta {
            delta: AnthropicMessageBlock::InputJsonDelta {
                partial_json: "aaaa: bbbbb".to_string(),
            },
            index: 0,
        };
        let latency = Duration::from_millis(100);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_delta,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let error = result.unwrap_err();
        assert_eq!(
            error,
            Error::AnthropicServer {
                message: "Got InputJsonDelta chunk from Anthropic without current tool name being set by a ToolUse".to_string()
            }
        );

        // Test ContentBlockDelta with InputJsonDelta and previous tool info
        let mut current_tool_id = Some("tool_id".to_string());
        let mut current_tool_name = Some("tool_name".to_string());
        let content_block_delta = AnthropicStreamMessage::ContentBlockDelta {
            delta: AnthropicMessageBlock::InputJsonDelta {
                partial_json: "aaaa: bbbbb".to_string(),
            },
            index: 0,
        };
        let latency = Duration::from_millis(100);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_delta,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "tool_id".to_string());
                assert_eq!(tool_call.name, "tool_name".to_string());
                assert_eq!(tool_call.arguments, "aaaa: bbbbb".to_string());
            }
            _ => panic!("Expected a tool call content block"),
        }
        assert_eq!(chunk.latency, latency);

        // Test ContentBlockStart with ToolUse
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_start = AnthropicStreamMessage::ContentBlockStart {
            content_block: AnthropicMessageBlock::ToolUse {
                id: "tool1".to_string(),
                name: "calculator".to_string(),
                input: json!({}),
            },
            index: 1,
        };
        let latency = Duration::from_millis(110);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_start,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "tool1".to_string());
                assert_eq!(tool_call.name, "calculator".to_string());
                assert_eq!(tool_call.arguments, "".to_string());
            }
            _ => panic!("Expected a tool call content block"),
        }
        assert_eq!(chunk.latency, latency);
        assert_eq!(current_tool_id, Some("tool1".to_string()));
        assert_eq!(current_tool_name, Some("calculator".to_string()));

        // Test ContentBlockStart with Text
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_start = AnthropicStreamMessage::ContentBlockStart {
            content_block: AnthropicMessageBlock::Text {
                text: "Hello".to_string(),
            },
            index: 2,
        };
        let latency = Duration::from_millis(120);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_start,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::Text(text) => {
                assert_eq!(text.text, "Hello".to_string());
                assert_eq!(text.id, "2".to_string());
            }
            _ => panic!("Expected a text content block"),
        }
        assert_eq!(chunk.latency, latency);

        // Test ContentBlockStart with InputJsonDelta (should fail)
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_start = AnthropicStreamMessage::ContentBlockStart {
            content_block: AnthropicMessageBlock::InputJsonDelta {
                partial_json: "aaaa: bbbbb".to_string(),
            },
            index: 3,
        };
        let latency = Duration::from_millis(130);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_start,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let error = result.unwrap_err();
        assert_eq!(
            error,
            Error::AnthropicServer {
                message: "Unsupported content block type for ContentBlockStart".to_string()
            }
        );

        // Test ContentBlockStop
        let content_block_stop = AnthropicStreamMessage::ContentBlockStop { index: 2 };
        let latency = Duration::from_millis(120);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_stop,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test Error
        let error_message = AnthropicStreamMessage::Error {
            error: json!({"message": "Test error"}),
        };
        let latency = Duration::from_millis(130);
        let result = anthropic_to_tensorzero_stream_message(
            error_message,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            Error::AnthropicServer {
                message: r#"{"message":"Test error"}"#.to_string(),
            }
        );

        // Test MessageDelta with usage
        let message_delta = AnthropicStreamMessage::MessageDelta {
            delta: json!({}),
            usage: json!({"input_tokens": 10, "output_tokens": 20}),
        };
        let latency = Duration::from_millis(140);
        let result = anthropic_to_tensorzero_stream_message(
            message_delta,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 0);
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(chunk.latency, latency);

        // Test MessageStart with usage
        let message_start = AnthropicStreamMessage::MessageStart {
            message: json!({"usage": {"input_tokens": 5, "output_tokens": 15}}),
        };
        let latency = Duration::from_millis(150);
        let result = anthropic_to_tensorzero_stream_message(
            message_start,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 0);
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 15);
        assert_eq!(chunk.latency, latency);

        // Test MessageStop
        let message_stop = AnthropicStreamMessage::MessageStop;
        let latency = Duration::from_millis(160);
        let result = anthropic_to_tensorzero_stream_message(
            message_stop,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test Ping
        let ping = AnthropicStreamMessage::Ping {};
        let latency = Duration::from_millis(170);
        let result = anthropic_to_tensorzero_stream_message(
            ping,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_parse_usage_info() {
        // Test with valid input
        let usage_info = json!({
            "input_tokens": 100,
            "output_tokens": 200
        });
        let result = parse_usage_info(&usage_info);
        assert_eq!(result.input_tokens, 100);
        assert_eq!(result.output_tokens, 200);

        // Test with missing fields
        let usage_info = json!({
            "input_tokens": 50
        });
        let result = parse_usage_info(&usage_info);
        assert_eq!(result.input_tokens, 50);
        assert_eq!(result.output_tokens, 0);

        // Test with empty object
        let usage_info = json!({});
        let result = parse_usage_info(&usage_info);
        assert_eq!(result.input_tokens, 0);
        assert_eq!(result.output_tokens, 0);

        // Test with non-numeric values
        let usage_info = json!({
            "input_tokens": "not a number",
            "output_tokens": true
        });
        let result = parse_usage_info(&usage_info);
        assert_eq!(result.input_tokens, 0);
        assert_eq!(result.output_tokens, 0);
    }
}
