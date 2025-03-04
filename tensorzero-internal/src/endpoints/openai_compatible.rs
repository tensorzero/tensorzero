//! OpenAI-compatible API endpoint implementation.
//!
//! This module provides compatibility with the OpenAI Chat Completions API format,
//! translating between OpenAI's request/response format and our internal types.
//! It implements request handling, parameter conversion, and response formatting
//! to match OpenAI's API specification.
//!
//! We convert the request into our internal types, call `endpoints::inference::inference` to perform the actual inference,
//! and then convert the response into the OpenAI-compatible format.

use std::collections::HashMap;

use axum::body::Body;
use axum::debug_handler;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::StreamExt;
use url::Url;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::endpoints::inference::{
    inference, ChatCompletionInferenceParams, InferenceParams, Params,
};
use crate::error::{Error, ErrorDetails};
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::{
    current_timestamp, ContentBlockChatOutput, ContentBlockChunk, Image, ImageKind, Input,
    InputMessage, InputMessageContent, Role, Usage,
};
use crate::tool::{
    DynamicToolParams, Tool, ToolCall, ToolCallChunk, ToolCallOutput, ToolChoice, ToolResult,
};
use crate::variant::JsonMode;

use super::inference::{
    InferenceCredentials, InferenceOutput, InferenceResponse, InferenceResponseChunk,
    InferenceStream,
};

/// A handler for the OpenAI-compatible inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    headers: HeaderMap,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleParams>,
) -> Result<Response<Body>, Error> {
    let params = (headers, openai_compatible_params).try_into()?;
    let response = inference(config, &http_client, clickhouse_connection_info, params).await?;
    match response {
        InferenceOutput::NonStreaming(response) => {
            let openai_compatible_response = OpenAICompatibleResponse::from(response);
            Ok(Json(openai_compatible_response).into_response())
        }
        InferenceOutput::Streaming(stream) => {
            let openai_compatible_stream = prepare_serialized_openai_compatible_events(stream);
            Ok(Sse::new(openai_compatible_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response())
        }
    }
}

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
    JsonSchema { json_schema: Value },
    JsonObject,
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
/// `none` is the default when no tools are present. `auto` is the default if tools are present.
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
    episode_id: String,
    choices: Vec<OpenAICompatibleChoice>,
    created: u32,
    model: String,
    system_fingerprint: String,
    object: String,
    usage: OpenAICompatibleUsage,
}

const TENSORZERO_FUNCTION_NAME_PREFIX: &str = "tensorzero::function_name::";
const TENSORZERO_MODEL_NAME_PREFIX: &str = "tensorzero::model_name::";

impl TryFrom<(HeaderMap, OpenAICompatibleParams)> for Params {
    type Error = Error;
    fn try_from(
        (headers, openai_compatible_params): (HeaderMap, OpenAICompatibleParams),
    ) -> Result<Self, Self::Error> {
        let (function_name, model_name) = if let Some(function_name) = openai_compatible_params
            .model
            .strip_prefix(TENSORZERO_FUNCTION_NAME_PREFIX)
        {
            (Some(function_name.to_string()), None)
        } else if let Some(model_name) = openai_compatible_params
            .model
            .strip_prefix(TENSORZERO_MODEL_NAME_PREFIX)
        {
            (None, Some(model_name.to_string()))
        } else if let Some(function_name) =
            openai_compatible_params.model.strip_prefix("tensorzero::")
        {
            tracing::warn!(
                function_name = function_name,
                "Deprecation Warning: Please set the `model` parameter to `tensorzero::function_name::your_function` instead of `tensorzero::your_function.` The latter will be removed in a future release."
            );
            (Some(function_name.to_string()), None)
        } else {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "`model` field must start with `tensorzero::function_name::` or `tensorzero::model_name::`. For example, `tensorzero::function_name::my_function` for a function `my_function` defined in your config, `tensorzero::model_name::my_model` for a model `my_model` defined in your config, or default functions like `tensorzero::model_name::openai::gpt-4o-mini`.".to_string(),
            }));
        };

        if let Some(function_name) = &function_name {
            if function_name.is_empty() {
                return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                message:
                    "function_name (passed in model field after \"tensorzero::function_name::\") cannot be empty"
                        .to_string(),
            }
            .into());
            }
        }

        if let Some(model_name) = &model_name {
            if model_name.is_empty() {
                return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: "model_name (passed in model field after \"tensorzero::model_name::\") cannot be empty".to_string(),
                }
                .into());
            }
        }

        let episode_id = headers
            .get("episode_id")
            .map(|h| {
                h.to_str()
                    .map_err(|_| {
                        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message: "episode_id header is not valid UTF-8".to_string(),
                        })
                    })
                    .and_then(|s| {
                        Uuid::parse_str(s).map_err(|_| {
                            Error::new(ErrorDetails::InvalidEpisodeId {
                                message: "episode_id header is not a valid UUID".to_string(),
                            })
                        })
                    })
            })
            .transpose()?;
        // If both max_tokens and max_completion_tokens are provided, we use the minimum of the two.
        // Otherwise, we use the provided value, or None if neither is provided.
        let max_tokens = match (
            openai_compatible_params.max_tokens,
            openai_compatible_params.max_completion_tokens,
        ) {
            (Some(max_tokens), Some(max_completion_tokens)) => {
                Some(max_tokens.min(max_completion_tokens))
            }
            (Some(max_tokens), None) => Some(max_tokens),
            (None, Some(max_completion_tokens)) => Some(max_completion_tokens),
            (None, None) => None,
        };
        let json_mode = match openai_compatible_params.response_format {
            Some(OpenAICompatibleResponseFormat::JsonSchema { json_schema: _ }) => {
                Some(JsonMode::Strict)
            }
            Some(OpenAICompatibleResponseFormat::JsonObject) => Some(JsonMode::On),
            Some(OpenAICompatibleResponseFormat::Text) => Some(JsonMode::Off),
            None => None,
        };
        let input = openai_compatible_params.messages.try_into()?;
        let chat_completion_inference_params = ChatCompletionInferenceParams {
            temperature: openai_compatible_params.temperature,
            max_tokens,
            seed: openai_compatible_params.seed,
            top_p: openai_compatible_params.top_p,
            presence_penalty: openai_compatible_params.presence_penalty,
            frequency_penalty: openai_compatible_params.frequency_penalty,
            json_mode,
        };
        let inference_params = InferenceParams {
            chat_completion: chat_completion_inference_params,
        };
        let variant_name = headers
            .get("variant_name")
            .map(|h| {
                h.to_str()
                    .map_err(|_| {
                        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message: "variant_name header is not valid UTF-8".to_string(),
                        })
                    })
                    .map(|s| s.to_string())
            })
            .transpose()?;
        let dryrun = headers
            .get("dryrun")
            .map(|h| {
                h.to_str()
                    .map_err(|_| {
                        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message: "dryrun header is not valid UTF-8".to_string(),
                        })
                    })
                    .and_then(|s| {
                        s.parse::<bool>().map_err(|_| {
                            Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                                message: "dryrun header is not a valid boolean".to_string(),
                            })
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
            Some(OpenAICompatibleResponseFormat::JsonSchema { json_schema }) => Some(json_schema),
            _ => None,
        };
        Ok(Params {
            function_name,
            model_name,
            episode_id,
            input,
            stream: openai_compatible_params.stream,
            params: inference_params,
            variant_name,
            dryrun,
            dynamic_tool_params,
            output_schema,
            // OpenAI compatible endpoint does not support dynamic credentials
            credentials: InferenceCredentials::default(),
            cache_options: CacheParamsOptions::default(),
            tags: HashMap::new(),
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
        for (index, message) in openai_compatible_messages.into_iter().enumerate() {
            match message {
                OpenAICompatibleMessage::System(msg) => {
                    if system.is_some() {
                        return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message: "At most one system message is allowed".to_string(),
                        }
                        .into());
                    }
                    if index != 0 {
                        return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message: "System message must be the first message".to_string(),
                        }
                        .into());
                    }
                    let mut system_content = convert_openai_message_content(msg.content.clone())?;
                    if system_content.len() != 1 {
                        return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message: "System message must be a single content block".to_string(),
                        }
                        .into());
                    }
                    system = Some(match system_content.remove(0) {
                        InputMessageContent::Text { value } => value,
                        InputMessageContent::RawText { value } => Value::String(value),
                        _ => {
                            return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                                message: "System message must be a text content block".to_string(),
                            }
                            .into())
                        }
                    });
                }
                OpenAICompatibleMessage::User(msg) => {
                    messages.push(InputMessage {
                        role: Role::User,
                        content: convert_openai_message_content(msg.content)?,
                    });
                }
                OpenAICompatibleMessage::Assistant(msg) => {
                    let mut message_content = Vec::new();
                    if let Some(content) = msg.content {
                        message_content.extend(convert_openai_message_content(content)?);
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
                        .ok_or_else(|| {
                            Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                                message: "tool call id not found".to_string(),
                            })
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

#[derive(Deserialize, Debug)]
#[serde(tag = "type", deny_unknown_fields, rename_all = "snake_case")]
#[allow(dead_code)]
enum OpenAICompatibleContentBlock {
    Text(TextContent),
    ImageUrl { image_url: OpenAICompatibleImageUrl },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", deny_unknown_fields, rename_all = "snake_case")]
#[allow(dead_code)]
struct OpenAICompatibleImageUrl {
    url: Url,
}

#[derive(Deserialize, Debug)]
#[serde(untagged, deny_unknown_fields, rename_all = "snake_case")]
// Two mutually exclusive modes - the standard OpenAI text, and our special TensorZero mode
pub enum TextContent {
    /// A normal openai text content block: `{"type": "text", "text": "Some content"}`. The `type` key comes from the parent `OpenAICompatibleContentBlock`
    RawText { text: String },
    /// A special TensorZero mode: `{"type": "text", "tensorzero::arguments": {"custom_key": "custom_val"}}`.
    TensorZeroArguments {
        #[serde(default, rename = "tensorzero::arguments")]
        tensorzero_arguments: Value,
    },
}

fn parse_base64_image_data_url(url: &str) -> Result<(ImageKind, &str), Error> {
    let Some(url) = url.strip_prefix("data:") else {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "Image data URL must start with `data:`".to_string(),
        }));
    };
    let Some((mime_type, data)) = url.split_once(";base64,") else {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "Image data URL must contain a base64-encoded data part".to_string(),
        }));
    };
    let image_type = match mime_type {
        "image/jpeg" => ImageKind::Jpeg,
        "image/png" => ImageKind::Png,
        "image/webp" => ImageKind::WebP,
        _ => {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: format!("Unsupported content type `{mime_type}`: - only `image/jpeg`, `image/png``, and `image/webp` image data URLs are supported"),
            }))
        }
    };
    Ok((image_type, data))
}

fn convert_openai_message_content(content: Value) -> Result<Vec<InputMessageContent>, Error> {
    match content {
        Value::String(s) => Ok(vec![InputMessageContent::Text {value: Value::String(s) }]),
        Value::Array(a) => {
            let mut outputs = Vec::with_capacity(a.len());
            for val in a {
                let block = serde_json::from_value::<OpenAICompatibleContentBlock>(val.clone());
                let output = match block {
                    Ok(OpenAICompatibleContentBlock::Text(TextContent::RawText { text })) => InputMessageContent::Text { value: Value::String(text) },
                    Ok(OpenAICompatibleContentBlock::Text(TextContent::TensorZeroArguments { tensorzero_arguments })) => InputMessageContent::Text { value: tensorzero_arguments },
                    Ok(OpenAICompatibleContentBlock::ImageUrl { image_url }) => {
                        if image_url.url.scheme() == "data" {
                            let url_str = image_url.url.to_string();
                            let (mime_type, data) = parse_base64_image_data_url(&url_str)?;
                            InputMessageContent::Image(Image::Base64 { mime_type, data: data.to_string() })
                        } else {
                            InputMessageContent::Image(Image::Url { url: image_url.url })
                        }
                    }
                    Err(e) => {
                        tracing::warn!(r#"Content block `{val}` was not a valid OpenAI content block. This is deprecated - please use `{{"type": "text", "tensorzero::arguments": {{"custom": "data"}}` to pass arbitrary JSON values to TensorZero: {e}"#);
                        InputMessageContent::Text { value: val }
                    }
                };
                outputs.push(output);
            }
            Ok(outputs)
        }
        _ => Err(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "message content must either be a string or an array of length 1 containing structured TensorZero inputs".to_string(),
        }.into()),
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
                    episode_id: response.episode_id.to_string(),
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
                episode_id: response.episode_id.to_string(),
            },
        }
    }
}

// Takes a vector of ContentBlockOutput and returns a tuple of (Option<String>, Vec<OpenAICompatibleToolCall>).
// This is useful since the OpenAI format separates text and tool calls in the response fields.
fn process_chat_content(
    content: Vec<ContentBlockChatOutput>,
) -> (Option<String>, Vec<OpenAICompatibleToolCall>) {
    let mut content_str: Option<String> = None;
    let mut tool_calls = Vec::new();
    for block in content {
        match block {
            ContentBlockChatOutput::Text(text) => match content_str {
                Some(ref mut content) => content.push_str(&text.text),
                None => content_str = Some(text.text),
            },
            ContentBlockChatOutput::ToolCall(tool_call) => {
                tool_calls.push(tool_call.into());
            }
            ContentBlockChatOutput::Thought(_thought) => {
                // OpenAI compatible endpoint does not support thought blocks
                // Users of this endpoint will need to check observability to see them
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
    episode_id: String,
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
                    episode_id: c.episode_id.to_string(),
                    choices: vec![OpenAICompatibleChoiceChunk {
                        index: 0,
                        finish_reason: "".to_string(),
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
                episode_id: c.episode_id.to_string(),
                choices: vec![OpenAICompatibleChoiceChunk {
                    index: 0,
                    finish_reason: "".to_string(),
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
            ContentBlockChunk::Thought(_thought) => {
                // OpenAI compatible endpoint does not support thought blocks
                // Users of this endpoint will need to check observability to see them
            }
        }
    }
    (content_str, tool_calls)
}

/// Prepares an Event for SSE on the way out of the gateway
/// When None is passed in, we send "[DONE]" to the client to signal the end of the stream
fn prepare_serialized_openai_compatible_events(
    mut stream: InferenceStream,
) -> impl Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        while let Some(chunk) = stream.next().await {
            // NOTE - in the future, we may want to end the stream early if we get an error
            // For now, we just ignore the error and try to get more chunks
            let Ok(chunk) = chunk else {
                continue;
            };
            let openai_compatible_chunk = OpenAICompatibleResponseChunk::from(chunk);
            let chunk_json = serde_json::to_value(openai_compatible_chunk).map_err(|e| {
                Error::new(ErrorDetails::Inference {
                    message: format!("Failed to convert chunk to JSON: {}", e),
                })
            })?;
            yield Event::default().json_data(chunk_json).map_err(|e| {
                Error::new(ErrorDetails::Inference {
                    message: format!("Failed to convert Value to Event: {}", e),
                })
            })
        }
        yield Ok(Event::default().data("[DONE]"));
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

#[cfg(test)]
mod tests {
    use crate::inference::types::{Text, TextChunk};

    use super::*;
    use axum::http::header::{HeaderName, HeaderValue};
    use serde_json::json;
    use tracing_test::traced_test;

    #[test]
    fn test_try_from_openai_compatible_params() {
        let episode_id = Uuid::now_v7();
        let headers = HeaderMap::from_iter(vec![
            (
                HeaderName::from_static("episode_id"),
                HeaderValue::from_str(&episode_id.to_string()).unwrap(),
            ),
            (
                HeaderName::from_static("variant_name"),
                HeaderValue::from_static("test_variant"),
            ),
        ]);
        let messages = vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
            content: Value::String("Hello, world!".to_string()),
        })];
        let params: Params = (
            headers,
            OpenAICompatibleParams {
                messages,
                model: "tensorzero::test_function".into(),
                frequency_penalty: Some(0.5),
                max_tokens: Some(100),
                max_completion_tokens: Some(50),
                presence_penalty: Some(0.5),
                response_format: None,
                seed: Some(23),
                stream: None,
                temperature: Some(0.5),
                tools: None,
                tool_choice: None,
                top_p: Some(0.5),
                parallel_tool_calls: None,
            },
        )
            .try_into()
            .unwrap();
        assert_eq!(params.function_name, Some("test_function".to_string()));
        assert_eq!(params.episode_id, Some(episode_id));
        assert_eq!(params.variant_name, Some("test_variant".to_string()));
        assert_eq!(params.input.messages.len(), 1);
        assert_eq!(params.input.messages[0].role, Role::User);
        assert_eq!(
            params.input.messages[0].content[0],
            InputMessageContent::Text {
                value: Value::String("Hello, world!".to_string()),
            }
        );
        assert_eq!(params.params.chat_completion.temperature, Some(0.5));
        assert_eq!(params.params.chat_completion.max_tokens, Some(50));
        assert_eq!(params.params.chat_completion.seed, Some(23));
        assert_eq!(params.params.chat_completion.top_p, Some(0.5));
        assert_eq!(params.params.chat_completion.presence_penalty, Some(0.5));
        assert_eq!(params.params.chat_completion.frequency_penalty, Some(0.5));
    }

    #[test]
    fn test_try_from_openai_compatible_messages() {
        let messages = vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
            content: Value::String("Hello, world!".to_string()),
        })];
        let input: Input = messages.try_into().unwrap();
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::User);
        assert_eq!(
            input.messages[0].content[0],
            InputMessageContent::Text {
                value: Value::String("Hello, world!".to_string()),
            }
        );
        // Now try a system message and a user message
        let messages = vec![
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("You are a helpful assistant".to_string()),
            }),
            OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                content: Value::String("Hello, world!".to_string()),
            }),
        ];
        let input: Input = messages.try_into().unwrap();
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::User);
        assert_eq!(
            input.system,
            Some(Value::String("You are a helpful assistant".to_string()))
        );
        // Now try some messages with structured content
        let messages = vec![
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("You are a helpful assistant".to_string()),
            }),
            OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                content: json!({
                    "country": "Japan",
                    "city": "Tokyo",
                }),
            }),
        ];
        let input: Result<Input, Error> = messages.try_into();
        let details = input.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "message content must either be a string or an array of length 1 containing structured TensorZero inputs".to_string(),
            }
        );

        // Try 2 system messages
        let messages = vec![
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("You are a helpful assistant".to_string()),
            }),
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("You are a helpful assistant".to_string()),
            }),
        ];
        let input: Result<Input, Error> = messages.try_into();
        let details = input.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "At most one system message is allowed".to_string(),
            }
        );

        // Try an assistant message with structured content
        let messages = vec![OpenAICompatibleMessage::Assistant(
            OpenAICompatibleAssistantMessage {
                content: Some(json!([{
                    "country": "Japan",
                    "city": "Tokyo",
                }])),
                tool_calls: None,
            },
        )];
        let input: Input = messages.try_into().unwrap();
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::Assistant);
        assert_eq!(
            input.messages[0].content[0],
            InputMessageContent::Text {
                value: json!({
                    "country": "Japan",
                    "city": "Tokyo",
                }),
            }
        );

        // Try an assistant message with text and tool calls
        let messages = vec![OpenAICompatibleMessage::Assistant(
            OpenAICompatibleAssistantMessage {
                content: Some(Value::String("Hello, world!".to_string())),
                tool_calls: Some(vec![OpenAICompatibleToolCall {
                    id: "1".to_string(),
                    r#type: "function".to_string(),
                    function: OpenAICompatibleFunctionCall {
                        name: "test_tool".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
            },
        )];
        let input: Input = messages.try_into().unwrap();
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::Assistant);
        assert_eq!(input.messages[0].content.len(), 2);

        let expected_text = InputMessageContent::Text {
            value: Value::String("Hello, world!".to_string()),
        };
        let expected_tool_call = InputMessageContent::ToolCall(ToolCall {
            id: "1".to_string(),
            name: "test_tool".to_string(),
            arguments: "{}".to_string(),
        });

        assert!(
            input.messages[0].content.contains(&expected_text),
            "Content does not contain the expected Text message."
        );
        assert!(
            input.messages[0].content.contains(&expected_tool_call),
            "Content does not contain the expected ToolCall."
        );

        let invalid_messages = vec![
            OpenAICompatibleMessage::Assistant(OpenAICompatibleAssistantMessage {
                content: Some(Value::String("Assistant message".to_string())),
                tool_calls: None,
            }),
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("System message".to_string()),
            }),
        ];
        let result: Result<Input, Error> = invalid_messages.try_into();
        assert!(
            result.is_err(),
            "Conversion should fail when a system message is after an assistant message."
        );
        if let Err(err) = result {
            let details = err.get_owned_details();
            match details {
                ErrorDetails::InvalidOpenAICompatibleRequest { message } => {
                    assert_eq!(
                        message, "System message must be the first message",
                        "Unexpected error message."
                    );
                }
                _ => panic!("Unexpected error type."),
            }
        }
    }

    #[test]
    fn test_convert_openai_message_content() {
        let content = json!([{
            "country": "Japan",
            "city": "Tokyo",
        }]);
        let value = convert_openai_message_content(content.clone()).unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::Text {
                value: json!({
                    "country": "Japan",
                    "city": "Tokyo",
                }),
            }]
        );
        let content = json!({
            "country": "Japan",
            "city": "Tokyo",
        });
        let error = convert_openai_message_content(content.clone()).unwrap_err();
        let details = error.get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "message content must either be a string or an array of length 1 containing structured TensorZero inputs".to_string(),
            }
        );
        let content = json!([]);
        let messages = convert_openai_message_content(content).unwrap();
        assert_eq!(messages, vec![]);

        let arguments_block = json!([{
            "type": "text",
            "tensorzero::arguments": {
                "custom_key": "custom_val"
            }
        }]);
        let value = convert_openai_message_content(arguments_block).unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::Text {
                value: json!({
                    "custom_key": "custom_val",
                }),
            }]
        );
    }

    #[test]
    #[traced_test]
    fn test_deprecated_custom_block() {
        let content = json!([{
            "country": "Japan",
            "city": "Tokyo",
        }]);
        let value = convert_openai_message_content(content.clone()).unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::Text {
                value: json!({
                    "country": "Japan",
                    "city": "Tokyo",
                }),
            }]
        );
        assert!(logs_contain(
            r#"Content block `{"country":"Japan","city":"Tokyo"}` was not a valid OpenAI content block."#
        ));

        let other_content = json!([{
            "type": "text",
            "my_custom_arg": 123
        }]);
        let value = convert_openai_message_content(other_content.clone()).unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::Text {
                value: json!({
                    "type": "text",
                    "my_custom_arg": 123
                }),
            }]
        );
        assert!(logs_contain(
            r#"Content block `{"type":"text","my_custom_arg":123}` was not a valid OpenAI content block."#
        ));
    }

    #[test]
    fn test_process_chat_content() {
        let content = vec![
            ContentBlockChatOutput::Text(Text {
                text: "Hello".to_string(),
            }),
            ContentBlockChatOutput::ToolCall(ToolCallOutput {
                arguments: None,
                name: Some("test_tool".to_string()),
                id: "1".to_string(),
                raw_name: "test_tool".to_string(),
                raw_arguments: "{}".to_string(),
            }),
            ContentBlockChatOutput::Text(Text {
                text: ", world!".to_string(),
            }),
        ];
        let (content_str, tool_calls) = process_chat_content(content);
        assert_eq!(content_str, Some("Hello, world!".to_string()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "1");
        assert_eq!(tool_calls[0].function.name, "test_tool");
        assert_eq!(tool_calls[0].function.arguments, "{}");
        let content: Vec<ContentBlockChatOutput> = vec![];
        let (content_str, tool_calls) = process_chat_content(content);
        assert_eq!(content_str, None);
        assert!(tool_calls.is_empty());

        let content = vec![
            ContentBlockChatOutput::Text(Text {
                text: "First part".to_string(),
            }),
            ContentBlockChatOutput::Text(Text {
                text: " second part".to_string(),
            }),
            ContentBlockChatOutput::ToolCall(ToolCallOutput {
                arguments: None,
                name: Some("middle_tool".to_string()),
                id: "123".to_string(),
                raw_name: "middle_tool".to_string(),
                raw_arguments: "{\"key\": \"value\"}".to_string(),
            }),
            ContentBlockChatOutput::Text(Text {
                text: " third part".to_string(),
            }),
            ContentBlockChatOutput::Text(Text {
                text: " fourth part".to_string(),
            }),
        ];
        let (content_str, tool_calls) = process_chat_content(content);
        assert_eq!(
            content_str,
            Some("First part second part third part fourth part".to_string())
        );
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "123");
        assert_eq!(tool_calls[0].function.name, "middle_tool");
        assert_eq!(tool_calls[0].function.arguments, "{\"key\": \"value\"}");
    }

    #[test]
    fn test_process_chat_content_chunk() {
        let content = vec![
            ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "Hello".to_string(),
            }),
            ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "1".to_string(),
                raw_name: "test_tool".to_string(),
                raw_arguments: "{}".to_string(),
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "2".to_string(),
                text: ", world!".to_string(),
            }),
        ];
        let (content_str, tool_calls) = process_chat_content_chunk(content);
        assert_eq!(content_str, Some("Hello, world!".to_string()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "1");
        assert_eq!(tool_calls[0].function.name, "test_tool");
        assert_eq!(tool_calls[0].function.arguments, "{}");

        let content: Vec<ContentBlockChunk> = vec![];
        let (content_str, tool_calls) = process_chat_content_chunk(content);
        assert_eq!(content_str, None);
        assert!(tool_calls.is_empty());

        let content = vec![
            ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "First part".to_string(),
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "2".to_string(),
                text: " second part".to_string(),
            }),
            ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "123".to_string(),
                raw_name: "middle_tool".to_string(),
                raw_arguments: "{\"key\": \"value\"}".to_string(),
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "3".to_string(),
                text: " third part".to_string(),
            }),
            ContentBlockChunk::Text(TextChunk {
                id: "4".to_string(),
                text: " fourth part".to_string(),
            }),
        ];
        let (content_str, tool_calls) = process_chat_content_chunk(content);
        assert_eq!(
            content_str,
            Some("First part second part third part fourth part".to_string())
        );
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "123");
        assert_eq!(tool_calls[0].function.name, "middle_tool");
        assert_eq!(tool_calls[0].function.arguments, "{\"key\": \"value\"}");
    }

    #[test]
    fn test_parse_base64() {
        assert_eq!(
            (ImageKind::Jpeg, "YWJjCg=="),
            parse_base64_image_data_url("data:image/jpeg;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            (ImageKind::Png, "YWJjCg=="),
            parse_base64_image_data_url("data:image/png;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            (ImageKind::WebP, "YWJjCg=="),
            parse_base64_image_data_url("data:image/webp;base64,YWJjCg==").unwrap()
        );
        let err = parse_base64_image_data_url("data:image/svg;base64,YWJjCg==")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Unsupported content type `image/svg`"),
            "Unexpected error message: {err}"
        );
    }
}
