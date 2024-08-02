use futures::StreamExt;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::{
    inference::types::{
        InferenceRequestMessage, InferenceResponseStream, ModelInferenceRequest,
        ModelInferenceResponse, ModelInferenceResponseChunk, Tool, ToolCall, ToolCallChunk,
        ToolChoice, ToolType, Usage,
    },
    model::ProviderConfig,
};

const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider;

impl InferenceProvider for AnthropicProvider {
    /// Anthropic non-streaming API request
    async fn infer<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let (model_name, api_key) = match model {
            ProviderConfig::Anthropic {
                model_name,
                api_key,
            } => (
                model_name,
                api_key.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "Anthropic".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected Anthropic provider config".to_string(),
                })
            }
        };

        let request_body = AnthropicRequestBody::new(model_name, request)?;
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
        if res.status().is_success() {
            let response_body =
                res.json::<AnthropicResponseBody>()
                    .await
                    .map_err(|e| Error::AnthropicServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            response_body.try_into()
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
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let (model_name, api_key) = match model {
            ProviderConfig::Anthropic {
                model_name,
                api_key,
            } => (
                model_name,
                api_key.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "Anthropic".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected Anthropic provider config".to_string(),
                })
            }
        };
        let request_body = AnthropicRequestBody::new(model_name, request)?;
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
        let mut stream = stream_anthropic(event_source).await;
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
/// At a high level, this function is handling low-level EventSource details and mapping the objects returned by Anthropic into our `InferenceResponseChunk` type
async fn stream_anthropic(mut event_source: EventSource) -> InferenceResponseStream {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let inference_id = Uuid::now_v7();
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    if let Err(_e) = tx.send(Err(Error::AnthropicServer {
                        message: e.to_string(),
                    })) {
                        // rx dropped
                        break;
                    }
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<AnthropicStreamMessage, Error> =
                            serde_json::from_str(&message.data).map_err(|e| {
                                {
                                    Error::AnthropicServer {
                                        message: format!(
                                            "Error parsing message: {}, Data: {}",
                                            e, message.data
                                        ),
                                    }
                                }
                            });
                        let response = match data {
                            Err(e) => Err(e),
                            Ok(data) => {
                                // Anthropic streaming API docs specify that this is the last message
                                if let AnthropicStreamMessage::MessageStop = data {
                                    break;
                                }
                                anthropic_to_tensorzero_stream_message(data, inference_id)
                            }
                        }
                        .transpose();

                        if let Some(stream_message) = response {
                            if tx.send(stream_message).is_err() {
                                // rx dropped
                                break;
                            }
                        }
                    }
                },
            }
        }

        event_source.close();
    });

    Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}

#[derive(Serialize, PartialEq, Debug, Clone)]
#[serde(rename_all = "lowercase")]
/// Anthropic doesn't handle the system message in this way
/// It's a field of the POST body instead
enum AnthropicRole {
    User,
    Assistant,
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
            ToolChoice::Tool(name) => Ok(AnthropicToolChoice::Tool { name }),
            ToolChoice::None => Err(Error::InvalidTool {
                message: "Tool choice is None. Anthropic does not support tool choice None."
                    .to_string(),
            }),
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

impl<'a> TryFrom<&'a Tool> for AnthropicTool<'a> {
    type Error = Error;

    fn try_from(value: &'a Tool) -> Result<Self, Self::Error> {
        // In case we add more tool types in the future, the compiler will complain here.
        match value.r#type {
            ToolType::Function => Ok(AnthropicTool {
                name: &value.name,
                description: value.description.as_deref(),
                input_schema: &value.parameters,
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
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
        input: &'a str, // This
    }, // NB: Anthropic also supports Image blocks here but we won't for now
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct AnthropicMessage<'a> {
    role: AnthropicRole,
    content: Vec<AnthropicMessageContent<'a>>,
}

impl<'a> TryFrom<&'a InferenceRequestMessage> for AnthropicMessage<'a> {
    type Error = Error;
    fn try_from(
        inference_message: &'a InferenceRequestMessage,
    ) -> Result<AnthropicMessage<'a>, Error> {
        let (role, content) = match inference_message {
            InferenceRequestMessage::System(_) => Err(Error::InvalidMessage {
                message: "Can't convert System message to Anthropic message. Don't pass System message in except as the first message in the chat.".to_string(),
            }),
            InferenceRequestMessage::User(message) => Ok((AnthropicRole::User, vec![AnthropicMessageContent::Text { text: &message.content }])),
            InferenceRequestMessage::Assistant(message) => {
                let mut content = vec![];
                if let Some(text) = &message.content {
                    content.push(AnthropicMessageContent::Text { text });
                }
                for tool_call in message.tool_calls.as_ref().map(|v| v.iter()).unwrap_or_default() {
                    content.push(AnthropicMessageContent::ToolUse {
                        id: &tool_call.id,
                        name: &tool_call.name,
                        input: &tool_call.arguments,
                    });
                }
                Ok((AnthropicRole::Assistant, content))
            }
            InferenceRequestMessage::Tool(message) =>
                Ok((AnthropicRole::User, vec![AnthropicMessageContent::ToolResult {
                    tool_use_id: &message.tool_call_id,
                    content: vec![AnthropicMessageContent::Text {
                        text: &message.content,
                    }],
                }]))
        }?;
        Ok(AnthropicMessage { role, content })
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
        let first_message = &request.messages[0];
        let (system, request_messages) = match first_message {
            InferenceRequestMessage::System(message) => {
                (Some(message.content.as_str()), &request.messages[1..])
            }
            _ => (None, &request.messages[..]),
        };
        let messages: Vec<AnthropicMessage> = prepare_messages(
            request_messages
                .iter()
                .map(AnthropicMessage::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let tool_choice = request
            .tool_choice
            .as_ref()
            .map(AnthropicToolChoice::try_from)
            .transpose()?;
        let tools = request
            .tools_available
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .map(AnthropicTool::try_from)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl From<AnthropicUsage> for Usage {
    fn from(value: AnthropicUsage) -> Self {
        Usage {
            prompt_tokens: value.input_tokens,
            completion_tokens: value.output_tokens,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct AnthropicResponseBody {
    id: String,
    r#type: String, // this is always "message"
    role: String,   // this is always "assistant"
    content: Vec<AnthropicContentBlock>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

impl TryFrom<AnthropicResponseBody> for ModelInferenceResponse {
    type Error = Error;
    fn try_from(value: AnthropicResponseBody) -> Result<Self, Self::Error> {
        let raw = serde_json::to_string(&value).map_err(|e| Error::AnthropicServer {
            message: format!("Error parsing response from Anthropic: {e}"),
        })?;
        let mut message_text: Option<String> = None;
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        // Anthropic responses can in principle contain multiple content blocks.
        // We stack them into one response to match our response types.
        for block in value.content {
            match block {
                AnthropicContentBlock::Text { text } => match message_text {
                    Some(message) => message_text = Some(format!("{}\n{}", message, text)),
                    None => message_text = Some(text),
                },
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    let tool_call = ToolCall {
                        name,
                        arguments: input.to_string(),
                        id,
                    };
                    if let Some(calls) = tool_calls.as_mut() {
                        calls.push(tool_call);
                    } else {
                        tool_calls = Some(vec![tool_call]);
                    }
                }
            }
        }

        Ok(ModelInferenceResponse::new(
            message_text,
            tool_calls,
            raw,
            value.usage.into(),
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

struct StreamMessage {
    message: Option<String>,
    tool_calls: Option<Vec<ToolCallChunk>>,
}

impl From<AnthropicMessageBlock> for StreamMessage {
    fn from(block: AnthropicMessageBlock) -> Self {
        match block {
            AnthropicMessageBlock::Text { text } => StreamMessage {
                message: Some(text),
                tool_calls: None,
            },
            AnthropicMessageBlock::TextDelta { text } => StreamMessage {
                message: Some(text),
                tool_calls: None,
            },
            AnthropicMessageBlock::ToolUse { id, name, input } => StreamMessage {
                message: None,
                tool_calls: Some(vec![ToolCallChunk {
                    id: Some(id),
                    name: Some(name),
                    arguments: Some(input.to_string()),
                }]),
            },
            AnthropicMessageBlock::InputJsonDelta { partial_json } => StreamMessage {
                message: None,
                tool_calls: Some(vec![ToolCallChunk {
                    id: None,
                    name: None,
                    arguments: Some(partial_json),
                }]),
            },
        }
    }
}

#[derive(Deserialize, Debug, Serialize)]
#[allow(dead_code)]
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
    },
    MessageStart {
        message: Value,
    },
    MessageStop,
    Ping {},
}

fn anthropic_to_tensorzero_stream_message(
    message: AnthropicStreamMessage,
    inference_id: Uuid,
) -> Result<Option<ModelInferenceResponseChunk>, Error> {
    let raw_message = serde_json::to_string(&message).map_err(|e| Error::AnthropicServer {
        message: format!("Error parsing response from Anthropic: {e}"),
    })?;
    match message {
        AnthropicStreamMessage::ContentBlockDelta { delta, .. } => {
            let message: StreamMessage = delta.into();
            Ok(Some(ModelInferenceResponseChunk::new(
                inference_id,
                message.message,
                message.tool_calls,
                None,
                raw_message,
            )))
        }
        AnthropicStreamMessage::ContentBlockStart { content_block, .. } => {
            let message: StreamMessage = content_block.into();
            Ok(Some(ModelInferenceResponseChunk::new(
                inference_id,
                message.message,
                message.tool_calls,
                None,
                raw_message,
            )))
        }
        AnthropicStreamMessage::ContentBlockStop { .. } => Ok(None),
        AnthropicStreamMessage::Error { error } => Err(Error::AnthropicServer {
            message: error.to_string(),
        }),
        AnthropicStreamMessage::MessageDelta { delta } => {
            if let Some(usage_info) = delta.get("usage") {
                let usage = parse_usage_info(usage_info);
                Ok(Some(ModelInferenceResponseChunk::new(
                    inference_id,
                    None,
                    None,
                    Some(usage.into()),
                    raw_message,
                )))
            } else {
                Ok(None)
            }
        }
        AnthropicStreamMessage::MessageStart { message } => {
            if let Some(usage_info) = message.get("message").and_then(|m| m.get("usage")) {
                let usage = parse_usage_info(usage_info);
                Ok(Some(ModelInferenceResponseChunk::new(
                    inference_id,
                    None,
                    None,
                    Some(usage.into()),
                    raw_message,
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
    use serde_json::json;

    use crate::inference::types::{
        AssistantInferenceRequestMessage, FunctionType, SystemInferenceRequestMessage,
        ToolInferenceRequestMessage, UserInferenceRequestMessage,
    };

    use super::*;

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

        let tool_choice = ToolChoice::Tool("test".to_string());
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            AnthropicToolChoice::Tool { name: "test" }
        );
    }

    #[test]
    fn test_try_from_tool() {
        let tool = Tool {
            name: "test".to_string(),
            description: Some("test".to_string()),
            r#type: ToolType::Function,
            parameters: Value::Null,
        };
        let anthropic_tool = AnthropicTool::try_from(&tool);
        assert!(anthropic_tool.is_ok());
        assert_eq!(
            anthropic_tool.unwrap(),
            AnthropicTool {
                name: "test",
                description: Some("test"),
                input_schema: &Value::Null,
            }
        );
    }

    #[test]
    fn test_try_from_inference_request_message() {
        // Test a User message
        let inference_request_message =
            InferenceRequestMessage::User(UserInferenceRequestMessage {
                content: "test".to_string(),
            });
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message);
        assert!(anthropic_message.is_ok());
        assert_eq!(
            anthropic_message.unwrap(),
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "test" }],
            }
        );

        // Test an Assistant message
        let inference_request_message =
            InferenceRequestMessage::Assistant(AssistantInferenceRequestMessage {
                content: Some("test_assistant".to_string()),
                tool_calls: None,
            });
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message);
        assert!(anthropic_message.is_ok());
        assert_eq!(
            anthropic_message.unwrap(),
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    text: "test_assistant",
                }],
            }
        );

        // Test a Tool message
        let inference_request_message =
            InferenceRequestMessage::Tool(ToolInferenceRequestMessage {
                content: "test_tool_response".to_string(),
                tool_call_id: "test_tool_call_id".to_string(),
            });
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message);
        assert!(anthropic_message.is_ok());
        assert_eq!(
            anthropic_message.unwrap(),
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

        // Test a system message
        let inference_request_message =
            InferenceRequestMessage::System(SystemInferenceRequestMessage {
                content: "test_system".to_string(),
            });
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message);
        assert!(anthropic_message.is_err());
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
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: false,
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
            InferenceRequestMessage::System(SystemInferenceRequestMessage {
                content: "test_system".to_string(),
            }),
            InferenceRequestMessage::User(UserInferenceRequestMessage {
                content: "test_user".to_string(),
            }),
            InferenceRequestMessage::Assistant(AssistantInferenceRequestMessage {
                content: Some("test_assistant".to_string()),
                tool_calls: None,
            }),
        ];
        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: false,
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
                    AnthropicMessage::try_from(&messages[1]).unwrap(),
                    AnthropicMessage::try_from(&messages[2]).unwrap(),
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
            InferenceRequestMessage::System(SystemInferenceRequestMessage {
                content: "test_system".to_string(),
            }),
            InferenceRequestMessage::User(UserInferenceRequestMessage {
                content: "test_user".to_string(),
            }),
            InferenceRequestMessage::User(UserInferenceRequestMessage {
                content: "test_user2".to_string(),
            }),
            InferenceRequestMessage::Assistant(AssistantInferenceRequestMessage {
                content: Some("test_assistant".to_string()),
                tool_calls: None,
            }),
        ];
        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            stream: true,
            json_mode: true,
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
                    AnthropicMessage::try_from(&messages[3]).unwrap(),
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
            InferenceRequestMessage::System(SystemInferenceRequestMessage {
                content: "test_system".to_string(),
            }),
            InferenceRequestMessage::User(UserInferenceRequestMessage {
                content: "test_user".to_string(),
            }),
            InferenceRequestMessage::Assistant(AssistantInferenceRequestMessage {
                content: Some("test_assistant".to_string()),
                tool_calls: None,
            }),
            InferenceRequestMessage::Tool(ToolInferenceRequestMessage {
                tool_call_id: "tool_call_id".to_string(),
                content: "tool_response".to_string(),
            }),
        ];
        let tool = Tool {
            r#type: ToolType::Function,
            description: Some("test_description".to_string()),
            name: "test_name".to_string(),
            parameters: json!({"type": "string"}),
        };
        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            tools_available: Some(vec![tool.clone()]),
            tool_choice: Some(ToolChoice::Auto),
            parallel_tool_calls: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            stream: true,
            json_mode: true,
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
                    AnthropicMessage::try_from(&messages[1]).unwrap(),
                    AnthropicMessage::try_from(&messages[2]).unwrap(),
                    AnthropicMessage::try_from(&messages[3]).unwrap(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some("test_system"),
                temperature: Some(0.5),
                tool_choice: Some(AnthropicToolChoice::Auto),
                tools: Some(vec![AnthropicTool {
                    name: "test_name",
                    description: Some("test_description"),
                    input_schema: &json!({"type": "string"}),
                }]),
            }
        );

        // Test case 5: System message later in list
        let messages = vec![
            InferenceRequestMessage::User(UserInferenceRequestMessage {
                content: "test_user".to_string(),
            }),
            InferenceRequestMessage::System(SystemInferenceRequestMessage {
                content: "test_system".to_string(),
            }),
            InferenceRequestMessage::Assistant(AssistantInferenceRequestMessage {
                content: Some("test_assistant".to_string()),
                tool_calls: None,
            }),
        ];
        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: false,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        assert!(anthropic_request_body.is_err());
        assert_eq!(
            anthropic_request_body.err().unwrap(),
            Error::InvalidMessage {
                message: "Can't convert System message to Anthropic message. Don't pass System message in except as the first message in the chat.".to_string(),
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

        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
    }

    #[test]
    fn test_anthropic_response_conversion() {
        // Test case 1: Text response
        let anthropic_response_body = AnthropicResponseBody {
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

        let inference_response =
            ModelInferenceResponse::try_from(anthropic_response_body.clone()).unwrap();
        assert_eq!(
            inference_response.content.as_ref().unwrap(),
            "Response text"
        );
        assert!(inference_response.tool_calls.is_none());

        let raw_json = json!(anthropic_response_body).to_string();
        let parsed_raw: serde_json::Value = serde_json::from_str(&inference_response.raw).unwrap();
        assert_eq!(raw_json, serde_json::json!(parsed_raw).to_string());
        assert_eq!(inference_response.usage.prompt_tokens, 100);
        assert_eq!(inference_response.usage.completion_tokens, 50);

        // Test case 2: Tool call response
        let anthropic_response_body = AnthropicResponseBody {
            id: "2".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock::ToolUse {
                id: "tool_call_1".to_string(),
                name: "get_weather".to_string(),
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

        let inference_response: ModelInferenceResponse =
            anthropic_response_body.clone().try_into().unwrap();
        assert!(inference_response.content.is_none());
        assert!(inference_response.tool_calls.is_some());
        let tool_calls = inference_response.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].id, "tool_call_1");
        assert_eq!(tool_calls[0].arguments, r#"{"location":"New York"}"#);

        let raw_json = json!(anthropic_response_body).to_string();
        let parsed_raw: serde_json::Value = serde_json::from_str(&inference_response.raw).unwrap();
        assert_eq!(raw_json, serde_json::json!(parsed_raw).to_string());
        assert_eq!(inference_response.usage.prompt_tokens, 100);
        assert_eq!(inference_response.usage.completion_tokens, 50);

        // Test case 3: Mixed response (text and tool call)
        let anthropic_response_body = AnthropicResponseBody {
            id: "3".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                AnthropicContentBlock::Text {
                    text: "Here's the weather:".to_string(),
                },
                AnthropicContentBlock::ToolUse {
                    id: "tool_call_2".to_string(),
                    name: "get_weather".to_string(),
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

        let inference_response =
            ModelInferenceResponse::try_from(anthropic_response_body.clone()).unwrap();
        assert_eq!(
            inference_response.content.as_ref().unwrap(),
            "Here's the weather:"
        );
        assert!(inference_response.tool_calls.is_some());
        let tool_calls = inference_response.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].id, "tool_call_2");
        assert_eq!(tool_calls[0].arguments, r#"{"location":"London"}"#);

        let raw_json = json!(anthropic_response_body).to_string();
        let parsed_raw: serde_json::Value = serde_json::from_str(&inference_response.raw).unwrap();
        assert_eq!(raw_json, serde_json::json!(parsed_raw).to_string());

        assert_eq!(inference_response.usage.prompt_tokens, 100);
        assert_eq!(inference_response.usage.completion_tokens, 50);
    }

    #[test]
    fn test_anthropic_message_block_to_stream_message() {
        use serde_json::json;

        // Test Text block
        let text_block = AnthropicMessageBlock::Text {
            text: "Hello, world!".to_string(),
        };
        let stream_message: StreamMessage = text_block.into();
        assert_eq!(stream_message.message, Some("Hello, world!".to_string()));
        assert_eq!(stream_message.tool_calls, None);

        // Test TextDelta block
        let text_delta_block = AnthropicMessageBlock::TextDelta {
            text: "Delta text".to_string(),
        };
        let stream_message: StreamMessage = text_delta_block.into();
        assert_eq!(stream_message.message, Some("Delta text".to_string()));
        assert_eq!(stream_message.tool_calls, None);

        // Test ToolUse block
        let tool_input = json!({"operation": "add", "numbers": [1, 2]});
        let tool_use_block = AnthropicMessageBlock::ToolUse {
            id: "tool123".to_string(),
            name: "calculator".to_string(),
            input: tool_input.clone(),
        };
        let stream_message: StreamMessage = tool_use_block.into();
        assert_eq!(stream_message.message, None);
        assert!(stream_message.tool_calls.is_some());
        let tool_calls = stream_message.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, Some("tool123".to_string()));
        assert_eq!(tool_calls[0].name, Some("calculator".to_string()));
        assert_eq!(tool_calls[0].arguments, Some(tool_input.to_string()));

        // Test InputJsonDelta block
        let input_json_delta_block = AnthropicMessageBlock::InputJsonDelta {
            partial_json: r#"{"partial": "json"}"#.to_string(),
        };
        let stream_message: StreamMessage = input_json_delta_block.into();
        assert_eq!(stream_message.message, None);
        assert_eq!(
            stream_message.tool_calls,
            Some(vec![ToolCallChunk {
                id: None,
                name: None,
                arguments: Some(r#"{"partial": "json"}"#.to_string()),
            }])
        );
    }

    #[test]
    fn test_anthropic_to_tensorzero_stream_message() {
        use serde_json::json;
        use uuid::Uuid;

        let inference_id = Uuid::now_v7();

        // Test ContentBlockDelta
        let content_block_delta = AnthropicStreamMessage::ContentBlockDelta {
            delta: AnthropicMessageBlock::Text {
                text: "Hello".to_string(),
            },
            index: 0,
        };
        let result = anthropic_to_tensorzero_stream_message(content_block_delta, inference_id);
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content, Some("Hello".to_string()));
        assert_eq!(chunk.tool_calls, None);

        // Test ContentBlockStart
        let content_block_start = AnthropicStreamMessage::ContentBlockStart {
            content_block: AnthropicMessageBlock::ToolUse {
                id: "tool1".to_string(),
                name: "calculator".to_string(),
                input: json!({"operation": "add"}),
            },
            index: 1,
        };
        let result = anthropic_to_tensorzero_stream_message(content_block_start, inference_id);
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content, None);
        assert!(chunk.tool_calls.is_some());
        let tool_calls = chunk.tool_calls.unwrap();
        assert_eq!(tool_calls[0].id, Some("tool1".to_string()));
        assert_eq!(tool_calls[0].name, Some("calculator".to_string()));
        assert_eq!(
            tool_calls[0].arguments,
            Some(r#"{"operation":"add"}"#.to_string())
        );

        // Test ContentBlockStop
        let content_block_stop = AnthropicStreamMessage::ContentBlockStop { index: 2 };
        let result = anthropic_to_tensorzero_stream_message(content_block_stop, inference_id);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test Error
        let error_message = AnthropicStreamMessage::Error {
            error: json!({"message": "Test error"}),
        };
        let result = anthropic_to_tensorzero_stream_message(error_message, inference_id);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            Error::AnthropicServer {
                message: r#"{"message":"Test error"}"#.to_string(),
            }
        );

        // Test MessageDelta with usage
        let message_delta = AnthropicStreamMessage::MessageDelta {
            delta: json!({"usage": {"input_tokens": 10, "output_tokens": 20}}),
        };
        let result = anthropic_to_tensorzero_stream_message(message_delta, inference_id);
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content, None);
        assert_eq!(chunk.tool_calls, None);
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);

        // Test MessageStart with usage
        let message_start = AnthropicStreamMessage::MessageStart {
            message: json!({"message": {"usage": {"input_tokens": 5, "output_tokens": 15}}}),
        };
        let result = anthropic_to_tensorzero_stream_message(message_start, inference_id);
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content, None);
        assert_eq!(chunk.tool_calls, None);
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 15);

        // Test MessageStop
        let message_stop = AnthropicStreamMessage::MessageStop;
        let result = anthropic_to_tensorzero_stream_message(message_stop, inference_id);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test Ping
        let ping = AnthropicStreamMessage::Ping {};
        let result = anthropic_to_tensorzero_stream_message(ping, inference_id);
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
