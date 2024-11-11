use std::collections::HashMap;

use axum::body::Body;
use axum::debug_handler;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::endpoints::inference::{
    inference, ChatCompletionInferenceParams, InferenceParams, Params,
};
use crate::error::Error;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::{
    current_timestamp, ContentBlockChunk, ContentBlockOutput, Input, InputMessage,
    InputMessageContent, Role, Usage,
};
use crate::tool::{
    DynamicToolParams, Tool, ToolCall, ToolCallChunk, ToolCallOutput, ToolChoice, ToolResult,
};

use super::inference::{
    InferenceCredentials, InferenceOutput, InferenceResponse, InferenceResponseChunk,
};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAICompatibleFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAICompatibleToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool. Currently, only `function` is supported.
    pub r#type: String,
    /// The function that the model called.
    pub function: OpenAICompatibleFunctionCall,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpenAICompatibleSystemMessage {
    content: Value,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpenAICompatibleUserMessage {
    content: Value,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpenAICompatibleAssistantMessage {
    content: Option<Value>,
    tool_calls: Option<Vec<OpenAICompatibleToolCall>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpenAICompatibleToolMessage {
    content: Option<Value>,
    tool_call_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
enum OpenAICompatibleMessage {
    System(OpenAICompatibleSystemMessage),
    User(OpenAICompatibleUserMessage),
    Assistant(OpenAICompatibleAssistantMessage),
    Tool(OpenAICompatibleToolMessage),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum OpenAICompatibleResponseFormat {
    Text,
    JsonObject { json_schema: Option<Value> },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", content = "function")]
#[serde(rename_all = "snake_case")]
enum OpenAICompatibleTool {
    Function {
        description: Option<String>,
        name: String,
        parameters: Value,
        #[serde(default)]
        strict: bool,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct FunctionName {
    name: String,
}

/// Specifies a tool the model should use. Use to force the model to call a specific function.
#[derive(Clone, Debug, Deserialize, PartialEq)]
struct OpenAICompatibleNamedToolChoice {
    /// The type of the tool. Currently, only `function` is supported.
    r#type: String,
    function: FunctionName,
}

/// Controls which (if any) tool is called by the model.
/// `none` means the model will not call any tool and instead generates a message.
/// `auto` means the model can pick between generating a message or calling one or more tools.
/// `required` means the model must call one or more tools.
/// Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.
///
/// `none` is the default when no tools are present. `auto` is the default if tools are present.present.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum ChatCompletionToolChoiceOption {
    #[default]
    None,
    Auto,
    Required,
    #[serde(untagged)]
    Named(OpenAICompatibleNamedToolChoice),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleParams {
    messages: Vec<OpenAICompatibleMessage>,
    model: String,
    frequency_penalty: Option<f32>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    presence_penalty: Option<f32>,
    response_format: Option<OpenAICompatibleResponseFormat>,
    seed: Option<u32>,
    stream: Option<bool>,
    temperature: Option<f32>,
    tools: Option<Vec<OpenAICompatibleTool>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    top_p: Option<f32>,
    parallel_tool_calls: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAICompatibleToolCall>>,
    role: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleChoice {
    index: u32,
    finish_reason: String,
    message: OpenAICompatibleResponseMessage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleResponse {
    id: String,
    choices: Vec<OpenAICompatibleChoice>,
    created: u32,
    model: String,
    system_fingerprint: String,
    object: String,
    usage: OpenAICompatibleUsage,
}

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn openai_compatible_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    headers: HeaderMap,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleParams>,
) -> Result<Response<Body>, Error> {
    let params = (headers, openai_compatible_params).try_into()?;
    let response = inference(config, http_client, clickhouse_connection_info, params).await?;
    match response {
        InferenceOutput::NonStreaming(response) => {
            let openai_compatible_response = OpenAICompatibleResponse::from(response);
            Ok(Json(openai_compatible_response).into_response())
        }
        InferenceOutput::Streaming(stream) => {
            let openai_compatible_stream = stream.map(prepare_serialized_openai_compatible_chunk);
            Ok(Sse::new(openai_compatible_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response())
        }
    }
}

impl TryFrom<(HeaderMap, OpenAICompatibleParams)> for Params<'static> {
    type Error = Error;
    fn try_from(
        (headers, openai_compatible_params): (HeaderMap, OpenAICompatibleParams),
    ) -> Result<Self, Self::Error> {
        let function_name = headers
            .get("function_name")
            .ok_or(Error::InvalidOpenAICompatibleRequest {
                message: "function_name header is required".to_string(),
            })?
            .to_str()
            .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                message: "function_name header is not valid UTF-8".to_string(),
            })?
            .to_string();
        let episode_id = headers
            .get("episode_id")
            .map(|h| {
                h.to_str()
                    .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                        message: "episode_id header is not valid UTF-8".to_string(),
                    })
                    .and_then(|s| {
                        Uuid::parse_str(s).map_err(|_| Error::InvalidEpisodeId {
                            message: "episode_id header is not a valid UUID".to_string(),
                        })
                    })
            })
            .transpose()?;
        let input = openai_compatible_params.messages.try_into()?;
        let chat_completion_inference_params = ChatCompletionInferenceParams {
            temperature: openai_compatible_params.temperature,
            max_tokens: openai_compatible_params.max_tokens,
            seed: openai_compatible_params.seed,
            top_p: openai_compatible_params.top_p,
            presence_penalty: openai_compatible_params.presence_penalty,
            frequency_penalty: openai_compatible_params.frequency_penalty,
        };
        let inference_params = InferenceParams {
            chat_completion: chat_completion_inference_params,
        };
        let variant_name = headers
            .get("variant_name")
            .map(|h| {
                h.to_str()
                    .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                        message: "variant_name header is not valid UTF-8".to_string(),
                    })
                    .map(|s| s.to_string())
            })
            .transpose()?;
        let dryrun = headers
            .get("dryrun")
            .map(|h| {
                h.to_str()
                    .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                        message: "dryrun header is not valid UTF-8".to_string(),
                    })
                    .and_then(|s| {
                        s.parse::<bool>()
                            .map_err(|_| Error::InvalidOpenAICompatibleRequest {
                                message: "dryrun header is not a valid boolean".to_string(),
                            })
                    })
            })
            .transpose()?;
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools: openai_compatible_params
                .tools
                .map(|tools| tools.into_iter().map(|tool| tool.into()).collect()),
            tool_choice: openai_compatible_params
                .tool_choice
                .map(|tool_choice| tool_choice.into()),
            parallel_tool_calls: openai_compatible_params.parallel_tool_calls,
        };
        let output_schema = match openai_compatible_params.response_format {
            Some(OpenAICompatibleResponseFormat::JsonObject { json_schema }) => json_schema,
            _ => None,
        };
        Ok(Params {
            function_name,
            episode_id,
            input,
            stream: openai_compatible_params.stream,
            params: inference_params,
            variant_name,
            dryrun,
            dynamic_tool_params,
            output_schema,
            credentials: InferenceCredentials::default(),
        })
    }
}

impl TryFrom<Vec<OpenAICompatibleMessage>> for Input {
    type Error = Error;
    fn try_from(
        openai_compatible_messages: Vec<OpenAICompatibleMessage>,
    ) -> Result<Self, Self::Error> {
        let mut system = None;
        let mut messages = Vec::new();
        let mut tool_call_id_to_name = HashMap::new();
        for message in openai_compatible_messages {
            match message {
                OpenAICompatibleMessage::System(msg) => {
                    if system.is_some() {
                        return Err(Error::InvalidOpenAICompatibleRequest {
                            message: "At most one system message is allowed".to_string(),
                        });
                    }
                    system = Some(msg.content);
                }
                OpenAICompatibleMessage::User(msg) => {
                    messages.push(InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::Text { value: msg.content }],
                    });
                }
                OpenAICompatibleMessage::Assistant(msg) => {
                    let mut message_content = Vec::new();
                    if let Some(content) = msg.content {
                        message_content.push(InputMessageContent::Text { value: content });
                    }
                    if let Some(tool_calls) = msg.tool_calls {
                        for tool_call in tool_calls {
                            tool_call_id_to_name
                                .insert(tool_call.id.clone(), tool_call.function.name.clone());
                            message_content.push(InputMessageContent::ToolCall(tool_call.into()));
                        }
                    }
                    messages.push(InputMessage {
                        role: Role::Assistant,
                        content: message_content,
                    });
                }
                OpenAICompatibleMessage::Tool(msg) => {
                    let name = tool_call_id_to_name
                        .get(&msg.tool_call_id)
                        .ok_or(Error::InvalidOpenAICompatibleRequest {
                            message: "tool call id not found".to_string(),
                        })?
                        .to_string();
                    messages.push(InputMessage {
                        role: Role::User,
                        content: vec![InputMessageContent::ToolResult(ToolResult {
                            id: msg.tool_call_id,
                            name,
                            result: msg.content.unwrap_or_default().to_string(),
                        })],
                    });
                }
            }
        }
        Ok(Input { system, messages })
    }
}

impl From<OpenAICompatibleTool> for Tool {
    fn from(tool: OpenAICompatibleTool) -> Self {
        match tool {
            OpenAICompatibleTool::Function {
                description,
                name,
                parameters,
                strict,
            } => Tool {
                description: description.unwrap_or_default(),
                parameters,
                name,
                strict,
            },
        }
    }
}

impl From<ChatCompletionToolChoiceOption> for ToolChoice {
    fn from(tool_choice: ChatCompletionToolChoiceOption) -> Self {
        match tool_choice {
            ChatCompletionToolChoiceOption::None => ToolChoice::None,
            ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
            ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
            ChatCompletionToolChoiceOption::Named(named) => {
                ToolChoice::Specific(named.function.name)
            }
        }
    }
}

impl From<OpenAICompatibleToolCall> for ToolCall {
    fn from(tool_call: OpenAICompatibleToolCall) -> Self {
        ToolCall {
            id: tool_call.id,
            name: tool_call.function.name,
            arguments: tool_call.function.arguments,
        }
    }
}

impl From<InferenceResponse> for OpenAICompatibleResponse {
    // TODO (Viraj, urgently): What do we do with episode_id?
    fn from(inference_response: InferenceResponse) -> Self {
        match inference_response {
            InferenceResponse::Chat(response) => {
                let (content, tool_calls) = process_chat_content(response.content);
                OpenAICompatibleResponse {
                    id: response.inference_id.to_string(),
                    choices: vec![OpenAICompatibleChoice {
                        index: 0,
                        finish_reason: "stop".to_string(),
                        message: OpenAICompatibleResponseMessage {
                            content,
                            tool_calls: Some(tool_calls),
                            role: "assistant".to_string(),
                        },
                    }],
                    created: current_timestamp() as u32,
                    model: response.variant_name,
                    system_fingerprint: "".to_string(),
                    object: "chat.completion".to_string(),
                    usage: response.usage.into(),
                }
            }
            InferenceResponse::Json(response) => OpenAICompatibleResponse {
                id: response.inference_id.to_string(),
                choices: vec![OpenAICompatibleChoice {
                    index: 0,
                    finish_reason: "stop".to_string(),
                    message: OpenAICompatibleResponseMessage {
                        content: Some(response.output.raw),
                        tool_calls: None,
                        role: "assistant".to_string(),
                    },
                }],
                created: current_timestamp() as u32,
                model: response.variant_name,
                system_fingerprint: "".to_string(),
                object: "chat.completion".to_string(),
                usage: OpenAICompatibleUsage {
                    prompt_tokens: response.usage.input_tokens,
                    completion_tokens: response.usage.output_tokens,
                    total_tokens: response.usage.input_tokens + response.usage.output_tokens,
                },
            },
        }
    }
}

fn process_chat_content(
    content: Vec<ContentBlockOutput>,
) -> (Option<String>, Vec<OpenAICompatibleToolCall>) {
    let mut content_str: Option<String> = None;
    let mut tool_calls = Vec::new();
    for block in content {
        match block {
            ContentBlockOutput::Text(text) => match content_str {
                Some(ref mut content) => content.push_str(&text.text),
                None => content_str = Some(text.text),
            },
            ContentBlockOutput::ToolCall(tool_call) => {
                tool_calls.push(tool_call.into());
            }
        }
    }
    (content_str, tool_calls)
}

impl From<ToolCallOutput> for OpenAICompatibleToolCall {
    fn from(tool_call: ToolCallOutput) -> Self {
        OpenAICompatibleToolCall {
            id: tool_call.id,
            r#type: "function".to_string(),
            function: OpenAICompatibleFunctionCall {
                name: tool_call.raw_name,
                arguments: tool_call.raw_arguments,
            },
        }
    }
}

impl From<Usage> for OpenAICompatibleUsage {
    fn from(usage: Usage) -> Self {
        OpenAICompatibleUsage {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: usage.input_tokens + usage.output_tokens,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleResponseChunk {
    id: String,
    choices: Vec<OpenAICompatibleChoiceChunk>,
    created: u32,
    model: String,
    system_fingerprint: String,
    object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAICompatibleUsage>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleChoiceChunk {
    index: u32,
    finish_reason: String,
    delta: OpenAICompatibleDelta,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAICompatibleToolCall>>,
}

impl From<InferenceResponseChunk> for OpenAICompatibleResponseChunk {
    fn from(chunk: InferenceResponseChunk) -> Self {
        match chunk {
            InferenceResponseChunk::Chat(c) => {
                let (content, tool_calls) = process_chat_content_chunk(c.content);
                OpenAICompatibleResponseChunk {
                    id: c.inference_id.to_string(),
                    choices: vec![OpenAICompatibleChoiceChunk {
                        index: 0,
                        finish_reason: "stop".to_string(),
                        delta: OpenAICompatibleDelta {
                            content,
                            tool_calls: Some(tool_calls),
                        },
                    }],
                    created: current_timestamp() as u32,
                    model: c.variant_name,
                    system_fingerprint: "".to_string(),
                    object: "chat.completion".to_string(),
                    usage: c.usage.map(|usage| usage.into()),
                }
            }
            InferenceResponseChunk::Json(c) => OpenAICompatibleResponseChunk {
                id: c.inference_id.to_string(),
                choices: vec![OpenAICompatibleChoiceChunk {
                    index: 0,
                    finish_reason: "stop".to_string(),
                    delta: OpenAICompatibleDelta {
                        content: Some(c.raw),
                        tool_calls: None,
                    },
                }],
                created: current_timestamp() as u32,
                model: c.variant_name,
                system_fingerprint: "".to_string(),
                object: "chat.completion".to_string(),
                usage: c.usage.map(|usage| usage.into()),
            },
        }
    }
}

fn process_chat_content_chunk(
    content: Vec<ContentBlockChunk>,
) -> (Option<String>, Vec<OpenAICompatibleToolCall>) {
    let mut content_str: Option<String> = None;
    let mut tool_calls = Vec::new();
    for block in content {
        match block {
            ContentBlockChunk::Text(text) => match content_str {
                Some(ref mut content) => content.push_str(&text.text),
                None => content_str = Some(text.text),
            },
            ContentBlockChunk::ToolCall(tool_call) => {
                tool_calls.push(tool_call.into());
            }
        }
    }
    (content_str, tool_calls)
}

fn prepare_serialized_openai_compatible_chunk(
    chunk: Option<InferenceResponseChunk>,
) -> Result<Event, Error> {
    if let Some(chunk) = chunk {
        let openai_compatible_chunk = OpenAICompatibleResponseChunk::from(chunk);
        let chunk_json =
            serde_json::to_value(openai_compatible_chunk).map_err(|e| Error::Inference {
                message: format!("Failed to convert chunk to JSON: {}", e),
            })?;
        Event::default()
            .json_data(chunk_json)
            .map_err(|e| Error::Inference {
                message: format!("Failed to convert Value to Event: {}", e),
            })
    } else {
        Ok(Event::default().data("[DONE]"))
    }
}

impl From<ToolCallChunk> for OpenAICompatibleToolCall {
    fn from(tool_call: ToolCallChunk) -> Self {
        OpenAICompatibleToolCall {
            id: tool_call.id,
            r#type: "function".to_string(),
            function: OpenAICompatibleFunctionCall {
                name: tool_call.raw_name,
                arguments: tool_call.raw_arguments,
            },
        }
    }
}
