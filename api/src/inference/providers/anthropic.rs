use std::time::{Duration, Instant};

use crate::inference::types::{
    InferenceRequestMessage, ModelInferenceRequest, ModelInferenceResponse, Role, Tool, ToolCall,
    ToolChoice, ToolType, Usage,
};
use reqwest::StatusCode;
use secrecy::{ExposeSecret, Secret};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::Error;

const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub async fn infer(
    request: ModelInferenceRequest,
    // TODO: use Gabe's model types
    model: &str,
    http_client: &reqwest::Client,
    api_key: &Secret<String>,
) -> Result<ModelInferenceResponse, Error> {
    let request_body = AnthropicRequestBody::new(model.to_string(), request)?;
    let start = Instant::now();
    let res = http_client
        .post(ANTHROPIC_BASE_URL)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("x-api-key", api_key.expose_secret())
        .json(&request_body)
        .send()
        .await
        .map_err(|e| Error::InferenceClient {
            message: format!("Error sending request to Anthropic: {e}"),
        })?;
    let latency = start.elapsed();
    match res.status().is_success() {
        true => {
            let response_body =
                res.json::<AnthropicResponseBody>()
                    .await
                    .map_err(|e| Error::AnthropicServer {
                        message: format!("Error parsing Anthropic response: {e}"),
                    })?;
            (response_body, latency).try_into()
        }
        false => {
            let response_code = res.status();
            let error_body =
                res.json::<AnthropicError>()
                    .await
                    .map_err(|e| Error::AnthropicServer {
                        message: format!("Error parsing Anthropic response: {e}"),
                    })?;
            handle_anthropic_error(response_code, error_body)
        }
    }
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
#[derive(Serialize, Debug, PartialEq)]
#[serde(tag = "type")]
enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

impl TryFrom<ToolChoice> for AnthropicToolChoice {
    type Error = Error;
    fn try_from(tool_choice: ToolChoice) -> Result<Self, Error> {
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

#[derive(Serialize, Debug, Clone, PartialEq)]
struct AnthropicTool {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: Value,
}

impl TryFrom<Tool> for AnthropicTool {
    type Error = Error;

    fn try_from(value: Tool) -> Result<Self, Self::Error> {
        // In case we add more tool types in the future, the compiler will complain here.
        match value.r#type {
            ToolType::Function => Ok(AnthropicTool {
                name: value.name,
                description: value.description,
                input_schema: value.parameters,
            }),
        }
    }
}

#[derive(Serialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "snake_case")]
enum AnthropicMessageContentType {
    Text,
    ToolCall,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
enum AnthropicMessageContent {
    Text {
        r#type: AnthropicMessageContentType,
        text: String,
    },
    ToolCall {
        r#type: AnthropicMessageContentType,
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize, Clone, PartialEq)]
struct AnthropicMessage {
    role: AnthropicRole,
    content: Vec<AnthropicMessageContent>,
}

impl TryFrom<InferenceRequestMessage> for AnthropicMessage {
    type Error = Error;
    fn try_from(inference_message: InferenceRequestMessage) -> Result<AnthropicMessage, Error> {
        let role = match inference_message.role {
            Role::User => Ok(AnthropicRole::User),
            Role::Assistant => Ok(AnthropicRole::Assistant),
            Role::System => Err(Error::InvalidMessage {
                message: "Can't convert System message to Anthropic message. Don't pass System message in except as the first message in the chat.".to_string(),
            }),
            Role::Tool => Ok(AnthropicRole::User),
        }?;
        let content = match inference_message.role {
            Role::User => Ok(vec![AnthropicMessageContent::Text {
                r#type: AnthropicMessageContentType::Text,
                text: inference_message.content,
            }]),
            Role::Assistant => Ok(vec![AnthropicMessageContent::Text {
                r#type: AnthropicMessageContentType::Text,
                text: inference_message.content,
            }]),
            Role::Tool => Ok(vec![AnthropicMessageContent::ToolCall {
                r#type: AnthropicMessageContentType::ToolCall,
                tool_use_id: inference_message.tool_call_id.ok_or(Error::InvalidMessage {
                    message: "Tool call ID is required for tool messages".to_string(),
                })?,
                content: inference_message.content,
            }]),
            Role::System => Err(Error::InvalidMessage {
                message: "Can't convert System message to Anthropic message. Don't pass System message in except as the first message in the chat.".to_string(),
            }),
        }?;
        Ok(AnthropicMessage { role, content })
    }
}

#[derive(Serialize, Debug, PartialEq)]
struct AnthropicRequestBody {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    // This is the system message
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
}

impl AnthropicRequestBody {
    fn new(model: String, request: ModelInferenceRequest) -> Result<AnthropicRequestBody, Error> {
        if request.messages.is_empty() {
            return Err(Error::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            });
        }
        let first_message = &request.messages[0];
        let (system, request_messages) = match first_message.role {
            Role::System => (
                Some(first_message.content.clone()),
                request.messages[1..].to_vec(),
            ),
            _ => (None, request.messages),
        };
        let messages: Vec<AnthropicMessage> = consolidate_messages(
            request_messages
                .into_iter()
                .map(AnthropicMessage::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let tool_choice = request
            .tool_choice
            .map(AnthropicToolChoice::try_from)
            .transpose()?;
        let tools = request
            .tools_available
            .map(|tools| {
                tools
                    .into_iter()
                    .map(AnthropicTool::try_from)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        Ok(AnthropicRequestBody {
            model,
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
fn consolidate_messages(messages: Vec<AnthropicMessage>) -> Result<Vec<AnthropicMessage>, Error> {
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
    Ok(consolidated_messages)
}

#[derive(Deserialize, Debug, PartialEq, Clone)]
struct AnthropicError {
    r#type: String,
    message: String,
}

#[derive(Deserialize, Debug, Serialize, Clone)]
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

#[derive(Deserialize, Serialize, Debug, Clone)]
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

#[derive(Deserialize, Serialize, Debug, Clone)]
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

impl TryFrom<(AnthropicResponseBody, Duration)> for ModelInferenceResponse {
    type Error = Error;
    fn try_from(value: (AnthropicResponseBody, Duration)) -> Result<Self, Self::Error> {
        let (value, duration) = value;
        let raw = json!(value);
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
                        tool_call_id: id,
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
            crate::inference::types::Latency::NonStreaming { ttd: duration },
        ))
    }
}

fn handle_anthropic_error(
    response_code: StatusCode,
    response_body: AnthropicError,
) -> Result<ModelInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED
        | StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::TOO_MANY_REQUESTS => Err(Error::AnthropicClient {
            message: response_body.message,
            status_code: response_code,
        }),
        // StatusCode::NOT_FOUND | StatusCode::FORBIDDEN | StatusCode::INTERNAL_SERVER_ERROR
        // These are all captured in _ since they have the same error behavior
        _ => Err(Error::AnthropicServer {
            message: response_body.message,
        }),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::inference::types::{FunctionType, Latency};

    use super::*;

    #[test]
    fn test_try_from_tool_choice() {
        // Need to cover all 4 cases
        let tool_choice = ToolChoice::None;
        let anthropic_tool_choice = AnthropicToolChoice::try_from(tool_choice);
        assert!(anthropic_tool_choice.is_err());
        assert_eq!(
            anthropic_tool_choice.err().unwrap(),
            Error::InvalidTool {
                message: "Tool choice is None. Anthropic does not support tool choice None."
                    .to_string(),
            }
        );

        let tool_choice = ToolChoice::Auto;
        let anthropic_tool_choice = AnthropicToolChoice::try_from(tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(anthropic_tool_choice.unwrap(), AnthropicToolChoice::Auto);

        let tool_choice = ToolChoice::Required;
        let anthropic_tool_choice = AnthropicToolChoice::try_from(tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(anthropic_tool_choice.unwrap(), AnthropicToolChoice::Any);

        let tool_choice = ToolChoice::Tool("test".to_string());
        let anthropic_tool_choice = AnthropicToolChoice::try_from(tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            AnthropicToolChoice::Tool {
                name: "test".to_string()
            }
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
        let anthropic_tool = AnthropicTool::try_from(tool);
        assert!(anthropic_tool.is_ok());
        assert_eq!(
            anthropic_tool.unwrap(),
            AnthropicTool {
                name: "test".to_string(),
                description: Some("test".to_string()),
                input_schema: Value::Null,
            }
        );
    }

    #[test]
    fn test_try_from_inference_request_message() {
        // Test a User message
        let inference_request_message = InferenceRequestMessage {
            role: Role::User,
            content: "test".to_string(),
            tool_call_id: None,
        };
        let anthropic_message = AnthropicMessage::try_from(inference_request_message);
        assert!(anthropic_message.is_ok());
        assert_eq!(
            anthropic_message.unwrap(),
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: "test".to_string(),
                }],
            }
        );

        // Test an Assistant message
        let inference_request_message = InferenceRequestMessage {
            role: Role::Assistant,
            content: "test_assistant".to_string(),
            tool_call_id: None,
        };
        let anthropic_message = AnthropicMessage::try_from(inference_request_message);
        assert!(anthropic_message.is_ok());
        assert_eq!(
            anthropic_message.unwrap(),
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: "test_assistant".to_string(),
                }],
            }
        );

        // Test a Tool message
        let inference_request_message = InferenceRequestMessage {
            role: Role::Tool,
            content: "test_tool_response".to_string(),
            tool_call_id: Some("test_tool_call_id".to_string()),
        };
        let anthropic_message = AnthropicMessage::try_from(inference_request_message);
        assert!(anthropic_message.is_ok());
        assert_eq!(
            anthropic_message.unwrap(),
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::ToolCall {
                    r#type: AnthropicMessageContentType::ToolCall,
                    tool_use_id: "test_tool_call_id".to_string(),
                    content: "test_tool_response".to_string(),
                }],
            }
        );

        // Test a tool message missing tool call ID
        let inference_request_message = InferenceRequestMessage {
            role: Role::Tool,
            content: "test_tool_response".to_string(),
            tool_call_id: None,
        };
        let anthropic_message = AnthropicMessage::try_from(inference_request_message);
        assert!(anthropic_message.is_err());
        assert_eq!(
            anthropic_message.err().unwrap(),
            Error::InvalidMessage {
                message: "Tool call ID is required for tool messages".to_string(),
            }
        );

        // Test a system message
        let inference_request_message = InferenceRequestMessage {
            role: Role::System,
            content: "test_system".to_string(),
            tool_call_id: None,
        };
        let anthropic_message = AnthropicMessage::try_from(inference_request_message);
        assert!(anthropic_message.is_err());
    }

    #[test]
    fn test_initialize_anthropic_request_body() {
        let model = "claude".to_string();
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
        let anthropic_request_body = AnthropicRequestBody::new(model.clone(), inference_request);
        assert!(anthropic_request_body.is_err());
        assert_eq!(
            anthropic_request_body.err().unwrap(),
            Error::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            }
        );

        // Test Case 2: Messages with System message
        let messages = vec![
            InferenceRequestMessage {
                role: Role::System,
                content: "test_system".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::User,
                content: "test_user".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::Assistant,
                content: "test_assistant".to_string(),
                tool_call_id: None,
            },
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
        let anthropic_request_body = AnthropicRequestBody::new(model.clone(), inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: model.clone(),
                messages: vec![
                    AnthropicMessage::try_from(messages[1].clone()).unwrap(),
                    AnthropicMessage::try_from(messages[2].clone()).unwrap(),
                ],
                max_tokens: 4096,
                stream: Some(false),
                system: Some("test_system".to_string()),
                temperature: None,
                tool_choice: None,
                tools: None,
            }
        );

        // Test case 3: Messages with system message that require consolidation
        // also some of the optional fields are tested
        let messages = vec![
            InferenceRequestMessage {
                role: Role::System,
                content: "test_system".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::User,
                content: "test_user".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::User,
                content: "test_user2".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::Assistant,
                content: "test_assistant".to_string(),
                tool_call_id: None,
            },
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
        let anthropic_request_body = AnthropicRequestBody::new(model.clone(), inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: model.clone(),
                messages: vec![
                    AnthropicMessage {
                        role: AnthropicRole::User,
                        content: vec![
                            AnthropicMessageContent::Text {
                                r#type: AnthropicMessageContentType::Text,
                                text: "test_user".to_string(),
                            },
                            AnthropicMessageContent::Text {
                                r#type: AnthropicMessageContentType::Text,
                                text: "test_user2".to_string(),
                            }
                        ],
                    },
                    AnthropicMessage::try_from(messages[3].clone()).unwrap(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some("test_system".to_string()),
                temperature: Some(0.5),
                tool_choice: None,
                tools: None,
            }
        );

        // Test case 4: Tool use & choice
        let messages = vec![
            InferenceRequestMessage {
                role: Role::System,
                content: "test_system".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::User,
                content: "test_user".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::Assistant,
                content: "test_assistant".to_string(),
                tool_call_id: None,
            },
            InferenceRequestMessage {
                role: Role::Tool,
                content: "tool_response".to_string(),
                tool_call_id: Some("tool_call_id".to_string()),
            },
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

        let anthropic_request_body = AnthropicRequestBody::new(model.clone(), inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: model.clone(),
                messages: vec![
                    AnthropicMessage::try_from(messages[1].clone()).unwrap(),
                    AnthropicMessage::try_from(messages[2].clone()).unwrap(),
                    AnthropicMessage::try_from(messages[3].clone()).unwrap(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some("test_system".to_string()),
                temperature: Some(0.5),
                tool_choice: Some(AnthropicToolChoice::Auto),
                tools: Some(vec![AnthropicTool {
                    name: "test_name".to_string(),
                    description: Some("test_description".to_string()),
                    input_schema: json!({"type": "string"}),
                }]),
            }
        );
    }

    #[test]
    fn test_consolidate_messages() {
        // Test case 1: No consolidation needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("Hello"),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("Hi"),
                }],
            },
        ];
        let expected = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("Hello"),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("Hi"),
                }],
            },
        ];
        assert_eq!(consolidate_messages(messages.clone()).unwrap(), expected);

        // Test case 2: Consolidation needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("Hello"),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("How are you?"),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("Hi"),
                }],
            },
        ];
        let expected = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![
                    AnthropicMessageContent::Text {
                        r#type: AnthropicMessageContentType::Text,
                        text: String::from("Hello"),
                    },
                    AnthropicMessageContent::Text {
                        r#type: AnthropicMessageContentType::Text,
                        text: String::from("How are you?"),
                    },
                ],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("Hi"),
                }],
            },
        ];
        assert_eq!(consolidate_messages(messages.clone()).unwrap(), expected);

        // Test case 3: Multiple consolidations needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("Hello"),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("How are you?"),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("Hi"),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: String::from("I am here to help."),
                }],
            },
        ];
        let expected = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![
                    AnthropicMessageContent::Text {
                        r#type: AnthropicMessageContentType::Text,
                        text: String::from("Hello"),
                    },
                    AnthropicMessageContent::Text {
                        r#type: AnthropicMessageContentType::Text,
                        text: String::from("How are you?"),
                    },
                ],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![
                    AnthropicMessageContent::Text {
                        r#type: AnthropicMessageContentType::Text,
                        text: String::from("Hi"),
                    },
                    AnthropicMessageContent::Text {
                        r#type: AnthropicMessageContentType::Text,
                        text: String::from("I am here to help."),
                    },
                ],
            },
        ];
        assert_eq!(consolidate_messages(messages.clone()).unwrap(), expected);

        // Test case 4: No messages
        let messages: Vec<AnthropicMessage> = vec![];
        let expected: Vec<AnthropicMessage> = vec![];
        assert_eq!(consolidate_messages(messages.clone()).unwrap(), expected);

        // Test case 5: Single message
        let messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text {
                r#type: AnthropicMessageContentType::Text,
                text: String::from("Hello"),
            }],
        }];
        let expected = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text {
                r#type: AnthropicMessageContentType::Text,
                text: String::from("Hello"),
            }],
        }];
        assert_eq!(consolidate_messages(messages.clone()).unwrap(), expected);

        // Test case 6: Consolidate tool uses
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::ToolCall {
                    r#type: AnthropicMessageContentType::ToolCall,
                    tool_use_id: "tool1".to_string(),
                    content: "Tool call 1".to_string(),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::ToolCall {
                    r#type: AnthropicMessageContentType::ToolCall,
                    tool_use_id: "tool2".to_string(),
                    content: "Tool call 2".to_string(),
                }],
            },
        ];
        let expected = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![
                AnthropicMessageContent::ToolCall {
                    r#type: AnthropicMessageContentType::ToolCall,
                    tool_use_id: "tool1".to_string(),
                    content: "Tool call 1".to_string(),
                },
                AnthropicMessageContent::ToolCall {
                    r#type: AnthropicMessageContentType::ToolCall,
                    tool_use_id: "tool2".to_string(),
                    content: "Tool call 2".to_string(),
                },
            ],
        }];
        assert_eq!(consolidate_messages(messages.clone()).unwrap(), expected);

        // Test case 7: Consolidate mixed text and tool use
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: "User message 1".to_string(),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::ToolCall {
                    r#type: AnthropicMessageContentType::ToolCall,
                    tool_use_id: "tool1".to_string(),
                    content: "Tool call 1".to_string(),
                }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: "User message 2".to_string(),
                }],
            },
        ];
        let expected = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![
                AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: "User message 1".to_string(),
                },
                AnthropicMessageContent::ToolCall {
                    r#type: AnthropicMessageContentType::ToolCall,
                    tool_use_id: "tool1".to_string(),
                    content: "Tool call 1".to_string(),
                },
                AnthropicMessageContent::Text {
                    r#type: AnthropicMessageContentType::Text,
                    text: "User message 2".to_string(),
                },
            ],
        }];
        assert_eq!(consolidate_messages(messages.clone()).unwrap(), expected);
    }

    #[test]
    fn test_handle_anthropic_error() {
        let error_body = AnthropicError {
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
        let latency = Duration::from_secs(1);

        let inference_response =
            ModelInferenceResponse::try_from((anthropic_response_body.clone(), latency)).unwrap();
        assert_eq!(
            inference_response.content.as_ref().unwrap(),
            "Response text"
        );
        assert!(inference_response.tool_calls.is_none());

        let raw_json = json!(anthropic_response_body);
        assert_eq!(inference_response.raw, raw_json);
        assert_eq!(inference_response.usage.prompt_tokens, 100);
        assert_eq!(inference_response.usage.completion_tokens, 50);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming { ttd: latency }
        );

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

        let inference_response =
            ModelInferenceResponse::try_from((anthropic_response_body.clone(), latency)).unwrap();
        assert!(inference_response.content.is_none());
        assert!(inference_response.tool_calls.is_some());
        let tool_calls = inference_response.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].tool_call_id, "tool_call_1");
        assert_eq!(tool_calls[0].arguments, r#"{"location":"New York"}"#);

        let raw_json = json!(anthropic_response_body);
        assert_eq!(inference_response.raw, raw_json);
        assert_eq!(inference_response.usage.prompt_tokens, 100);
        assert_eq!(inference_response.usage.completion_tokens, 50);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming { ttd: latency }
        );

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
            ModelInferenceResponse::try_from((anthropic_response_body.clone(), latency)).unwrap();
        assert_eq!(
            inference_response.content.as_ref().unwrap(),
            "Here's the weather:"
        );
        assert!(inference_response.tool_calls.is_some());
        let tool_calls = inference_response.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].tool_call_id, "tool_call_2");
        assert_eq!(tool_calls[0].arguments, r#"{"location":"London"}"#);

        let raw_json = json!(anthropic_response_body);
        assert_eq!(inference_response.raw, raw_json);

        assert_eq!(inference_response.usage.prompt_tokens, 100);
        assert_eq!(inference_response.usage.completion_tokens, 50);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming { ttd: latency }
        );
    }
}
