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
use serde_json::{Map, Value};
use tokio_stream::StreamExt;
use url::Url;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::endpoints::inference::{
    inference, ChatCompletionInferenceParams, InferenceParams, Params,
};
use crate::error::{Error, ErrorDetails};
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::{
    current_timestamp, ContentBlockChatOutput, ContentBlockChunk, FinishReason, Image, ImageKind,
    Input, InputMessage, InputMessageContent, Role, TextKind, Usage,
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
    if !openai_compatible_params.unknown_fields.is_empty() {
        tracing::warn!(
            "Ignoring unknown fields in OpenAI-compatible request: {:?}",
            openai_compatible_params
                .unknown_fields
                .keys()
                .collect::<Vec<_>>()
        );
    }
    let params = Params::try_from_openai(headers, openai_compatible_params)?;

    // The prefix for the response's `model` field depends on the inference target
    // (We run this disambiguation deep in the `inference` call below but we don't get the decision out, so we duplicate it here)
    let response_model_prefix = match (&params.function_name, &params.model_name) {
        (Some(function_name), None) => Ok::<String, Error>(format!(
            "tensorzero::function_name::{}::variant_name::",
            function_name,
        )),
        (None, Some(_model_name)) => Ok("tensorzero::model_name::".to_string()),
        (Some(_), Some(_)) => Err(ErrorDetails::InvalidInferenceTarget {
            message: "Only one of `function_name` or `model_name` can be provided".to_string(),
        }
        .into()),
        (None, None) => Err(ErrorDetails::InvalidInferenceTarget {
            message: "Either `function_name` or `model_name` must be provided".to_string(),
        }
        .into()),
    }?;

    let response = inference(config, &http_client, clickhouse_connection_info, params).await?;

    match response {
        InferenceOutput::NonStreaming(response) => {
            let openai_compatible_response =
                OpenAICompatibleResponse::from((response, response_model_prefix));
            Ok(Json(openai_compatible_response).into_response())
        }
        InferenceOutput::Streaming(stream) => {
            let openai_compatible_stream =
                prepare_serialized_openai_compatible_events(stream, response_model_prefix);
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAICompatibleToolCallChunk {
    /// The ID of the tool call.
    pub id: Option<String>,
    /// The index of the tool call.
    pub index: usize,
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
    #[serde(alias = "developer")]
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
    #[serde(rename = "tensorzero::variant_name")]
    tensorzero_variant_name: Option<String>,
    #[serde(rename = "tensorzero::dryrun")]
    tensorzero_dryrun: Option<bool>,
    #[serde(rename = "tensorzero::episode_id")]
    tensorzero_episode_id: Option<Uuid>,
    #[serde(flatten)]
    unknown_fields: HashMap<String, Value>,
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
    finish_reason: OpenAICompatibleFinishReason,
    message: OpenAICompatibleResponseMessage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum OpenAICompatibleFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    // FunctionCall, we never generate this and it is deprecated
}

impl From<FinishReason> for OpenAICompatibleFinishReason {
    fn from(finish_reason: FinishReason) -> Self {
        match finish_reason {
            FinishReason::Stop => OpenAICompatibleFinishReason::Stop,
            FinishReason::Length => OpenAICompatibleFinishReason::Length,
            FinishReason::ContentFilter => OpenAICompatibleFinishReason::ContentFilter,
            FinishReason::ToolCall => OpenAICompatibleFinishReason::ToolCalls,
            FinishReason::Unknown => OpenAICompatibleFinishReason::Stop, // OpenAI doesn't have an unknown finish reason so we coerce
        }
    }
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

impl Params {
    fn try_from_openai(
        headers: HeaderMap,
        openai_compatible_params: OpenAICompatibleParams,
    ) -> Result<Self, Error> {
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

        let header_episode_id = headers
            .get("episode_id")
            .map(|h| {
                tracing::warn!("Deprecation warning: Please use the `tensorzero::episode_id` field instead of the `episode_id` header. The header will be removed in a future release.");
                h.to_str()
                    .map_err(|_| {
                        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message: "episode_id header is not valid UTF-8".to_string(),
                        })
                    })
                    .and_then(|s| {
                        Uuid::parse_str(s).map_err(|_| {
                            Error::new(ErrorDetails::InvalidTensorzeroUuid {
                                kind: "Episode".to_string(),
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
        let header_variant_name = headers
            .get("variant_name")
            .map(|h| {
                tracing::warn!("Deprecation warning: Please use the `tensorzero::variant_name` field instead of the `variant_name` header. The header will be removed in a future release.");
                h.to_str()
                    .map_err(|_| {
                        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message: "variant_name header is not valid UTF-8".to_string(),
                        })
                    })
                    .map(|s| s.to_string())
            })
            .transpose()?;
        let header_dryrun = headers
            .get("dryrun")
            .map(|h| {
                tracing::warn!("Deprecation warning: Please use the `tensorzero::dryrun` field instead of the `dryrun` header. The header will be removed in a future release.");
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
            episode_id: openai_compatible_params
                .tensorzero_episode_id
                .or(header_episode_id),
            input,
            stream: openai_compatible_params.stream,
            params: inference_params,
            variant_name: openai_compatible_params
                .tensorzero_variant_name
                .or(header_variant_name),
            dryrun: openai_compatible_params.tensorzero_dryrun.or(header_dryrun),
            dynamic_tool_params,
            output_schema,
            // OpenAI compatible endpoint does not support dynamic credentials
            credentials: InferenceCredentials::default(),
            cache_options: CacheParamsOptions::default(),
            // For now, we don't support internal inference for OpenAI compatible endpoint
            internal: false,
            tags: HashMap::new(),
            // OpenAI compatible endpoint does not support 'include_original_response'
            include_original_response: false,
            // OpenAI compatible endpoint does not support 'extra_body'
            extra_body: UnfilteredInferenceExtraBody::default(),
        })
    }
}

impl TryFrom<Vec<OpenAICompatibleMessage>> for Input {
    type Error = Error;
    fn try_from(
        openai_compatible_messages: Vec<OpenAICompatibleMessage>,
    ) -> Result<Self, Self::Error> {
        let mut system_messages = Vec::new();
        let mut messages = Vec::new();
        let mut tool_call_id_to_name = HashMap::new();
        let first_system = matches!(
            openai_compatible_messages.first(),
            Some(OpenAICompatibleMessage::System(_))
        );
        for message in openai_compatible_messages {
            match message {
                OpenAICompatibleMessage::System(msg) => {
                    let system_content = convert_openai_message_content(msg.content.clone())?;
                    for content in system_content {
                        system_messages.push(match content {
                            InputMessageContent::Text(TextKind::LegacyValue { value }) => value,
                            InputMessageContent::Text(TextKind::Text { text }) => {
                                Value::String(text)
                            }
                            InputMessageContent::Text(TextKind::Arguments { arguments }) => {
                                Value::Object(arguments)
                            }
                            InputMessageContent::RawText { value } => Value::String(value),
                            _ => {
                                return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                                    message: "System message must be a text content block"
                                        .to_string(),
                                }
                                .into())
                            }
                        });
                    }
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

        if system_messages.len() <= 1 {
            if system_messages.len() == 1 && !first_system {
                tracing::warn!("Moving system message to the start of the conversation");
            }
            Ok(Input {
                system: system_messages.pop(),
                messages,
            })
        } else {
            let mut output = String::new();
            for (i, system_message) in system_messages.iter().enumerate() {
                if let Value::String(msg) = system_message {
                    if i > 0 {
                        output.push('\n');
                    }
                    output.push_str(msg);
                } else {
                    return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                        message: "Multiple system messages provided, but not all were strings"
                            .to_string(),
                    }
                    .into());
                }
            }
            tracing::warn!("Multiple system messages provided - they will be concatenated and moved to the start of the conversation");
            Ok(Input {
                system: Some(Value::String(output)),
                messages,
            })
        }
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
        tensorzero_arguments: Map<String, Value>,
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
        Value::String(s) => Ok(vec![InputMessageContent::Text(TextKind::Text { text: s })]),
        Value::Array(a) => {
            let mut outputs = Vec::with_capacity(a.len());
            for val in a {
                let block = serde_json::from_value::<OpenAICompatibleContentBlock>(val.clone());
                let output = match block {
                    Ok(OpenAICompatibleContentBlock::Text(TextContent::RawText { text })) => InputMessageContent::Text(TextKind::Text {text }),
                    Ok(OpenAICompatibleContentBlock::Text(TextContent::TensorZeroArguments { tensorzero_arguments })) => InputMessageContent::Text(TextKind::Arguments { arguments: tensorzero_arguments }),
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
                        if let Value::Object(obj) = val {
                            InputMessageContent::Text(TextKind::Arguments { arguments: obj })
                        } else {
                            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                                message: format!("Content block `{val}` is not an object"),
                            }));
                        }
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

impl From<(InferenceResponse, String)> for OpenAICompatibleResponse {
    fn from((inference_response, response_model_prefix): (InferenceResponse, String)) -> Self {
        match inference_response {
            InferenceResponse::Chat(response) => {
                let (content, tool_calls) = process_chat_content(response.content);

                OpenAICompatibleResponse {
                    id: response.inference_id.to_string(),
                    choices: vec![OpenAICompatibleChoice {
                        index: 0,
                        finish_reason: response.finish_reason.unwrap_or(FinishReason::Stop).into(),
                        message: OpenAICompatibleResponseMessage {
                            content,
                            tool_calls: Some(tool_calls),
                            role: "assistant".to_string(),
                        },
                    }],
                    created: current_timestamp() as u32,
                    model: format!("{response_model_prefix}{}", response.variant_name),
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
                    finish_reason: response.finish_reason.unwrap_or(FinishReason::Stop).into(),
                    message: OpenAICompatibleResponseMessage {
                        content: Some(response.output.raw),
                        tool_calls: None,
                        role: "assistant".to_string(),
                    },
                }],
                created: current_timestamp() as u32,
                model: format!("{response_model_prefix}{}", response.variant_name),
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
                tracing::warn!(
                    "Ignoring 'thought' content block when constructing OpenAI-compatible response"
                );
            }
            ContentBlockChatOutput::Unknown {
                data: _,
                model_provider_name: _,
            } => {
                tracing::warn!(
                    "Ignoring 'unknown' content block when constructing OpenAI-compatible response"
                );
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
    finish_reason: Option<OpenAICompatibleFinishReason>,
    delta: OpenAICompatibleDelta,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAICompatibleToolCallChunk>>,
}

fn convert_inference_response_chunk_to_openai_compatible(
    chunk: InferenceResponseChunk,
    tool_id_to_index: &mut HashMap<String, usize>,
    response_model_prefix: &str,
) -> Vec<OpenAICompatibleResponseChunk> {
    let response_chunk = match chunk {
        InferenceResponseChunk::Chat(c) => {
            let (content, tool_calls) = process_chat_content_chunk(c.content, tool_id_to_index);
            OpenAICompatibleResponseChunk {
                id: c.inference_id.to_string(),
                episode_id: c.episode_id.to_string(),
                choices: vec![OpenAICompatibleChoiceChunk {
                    index: 0,
                    finish_reason: c.finish_reason.map(|finish_reason| finish_reason.into()),
                    delta: OpenAICompatibleDelta {
                        content,
                        tool_calls: Some(tool_calls),
                    },
                }],
                created: current_timestamp() as u32,
                model: format!("{response_model_prefix}{}", c.variant_name),
                system_fingerprint: "".to_string(),
                object: "chat.completion.chunk".to_string(),
                usage: c.usage.map(|usage| usage.into()),
            }
        }
        InferenceResponseChunk::Json(c) => OpenAICompatibleResponseChunk {
            id: c.inference_id.to_string(),
            episode_id: c.episode_id.to_string(),
            choices: vec![OpenAICompatibleChoiceChunk {
                index: 0,
                finish_reason: c.finish_reason.map(|finish_reason| finish_reason.into()),
                delta: OpenAICompatibleDelta {
                    content: Some(c.raw),
                    tool_calls: None,
                },
            }],
            created: current_timestamp() as u32,
            model: format!("{response_model_prefix}{}", c.variant_name),
            system_fingerprint: "".to_string(),
            object: "chat.completion.chunk".to_string(),
            usage: c.usage.map(|usage| usage.into()),
        },
    };

    vec![response_chunk]
}

fn process_chat_content_chunk(
    content: Vec<ContentBlockChunk>,
    tool_id_to_index: &mut HashMap<String, usize>,
) -> (Option<String>, Vec<OpenAICompatibleToolCallChunk>) {
    let mut content_str: Option<String> = None;
    let mut tool_calls = Vec::new();
    for block in content {
        match block {
            ContentBlockChunk::Text(text) => match content_str {
                Some(ref mut content) => content.push_str(&text.text),
                None => content_str = Some(text.text),
            },
            ContentBlockChunk::ToolCall(tool_call) => {
                let len = tool_id_to_index.len();
                let is_new = !tool_id_to_index.contains_key(&tool_call.id);
                let index = tool_id_to_index.entry(tool_call.id.clone()).or_insert(len);
                tool_calls.push(OpenAICompatibleToolCallChunk {
                    id: if is_new { Some(tool_call.id) } else { None },
                    index: *index,
                    r#type: "function".to_string(),
                    function: OpenAICompatibleFunctionCall {
                        name: tool_call.raw_name,
                        arguments: tool_call.raw_arguments,
                    },
                });
            }
            ContentBlockChunk::Thought(_thought) => {
                // OpenAI compatible endpoint does not support thought blocks
                // Users of this endpoint will need to check observability to see them
                tracing::warn!(
                    "Ignoring 'thought' content block chunk when constructing OpenAI-compatible response"
                );
            }
        }
    }
    (content_str, tool_calls)
}

/// Prepares an Event for SSE on the way out of the gateway
/// When None is passed in, we send "[DONE]" to the client to signal the end of the stream
fn prepare_serialized_openai_compatible_events(
    mut stream: InferenceStream,
    response_model_prefix: String,
) -> impl Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        let mut tool_id_to_index = HashMap::new();
        let mut is_first_chunk = true;
        while let Some(chunk) = stream.next().await {
            // NOTE - in the future, we may want to end the stream early if we get an error
            // For now, we just ignore the error and try to get more chunks
            let Ok(chunk) = chunk else {
                continue;
            };
            let openai_compatible_chunks = convert_inference_response_chunk_to_openai_compatible(chunk, &mut tool_id_to_index, &response_model_prefix);
            for chunk in openai_compatible_chunks {
                let mut chunk_json = serde_json::to_value(chunk).map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to convert chunk to JSON: {}", e),
                    })
                })?;
                if is_first_chunk {
                    // OpenAI includes "assistant" role in the first chunk but not in the subsequent chunks
                    chunk_json["choices"][0]["delta"]["role"] = Value::String("assistant".to_string());
                    is_first_chunk = false;
                }

                yield Event::default().json_data(chunk_json).map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to convert Value to Event: {}", e),
                    })
                })
            }
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
        let params = Params::try_from_openai(
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
                tensorzero_episode_id: None,
                tensorzero_variant_name: None,
                tensorzero_dryrun: None,
                unknown_fields: Default::default(),
            },
        )
        .unwrap();
        assert_eq!(params.function_name, Some("test_function".to_string()));
        assert_eq!(params.episode_id, Some(episode_id));
        assert_eq!(params.variant_name, Some("test_variant".to_string()));
        assert_eq!(params.input.messages.len(), 1);
        assert_eq!(params.input.messages[0].role, Role::User);
        assert_eq!(
            params.input.messages[0].content[0],
            InputMessageContent::Text(TextKind::Text {
                text: "Hello, world!".to_string(),
            })
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
            InputMessageContent::Text(TextKind::Text {
                text: "Hello, world!".to_string(),
            })
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
                content: Value::String("You are a helpful assistant 1.".to_string()),
            }),
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("You are a helpful assistant 2.".to_string()),
            }),
        ];
        let input: Input = messages.try_into().unwrap();
        assert_eq!(
            input.system,
            Some("You are a helpful assistant 1.\nYou are a helpful assistant 2.".into())
        );
        assert_eq!(input.messages.len(), 0);

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
            InputMessageContent::Text(TextKind::Arguments {
                arguments: json!({
                    "country": "Japan",
                    "city": "Tokyo",
                })
                .as_object()
                .unwrap()
                .clone(),
            })
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

        let expected_text = InputMessageContent::Text(TextKind::Text {
            text: "Hello, world!".to_string(),
        });
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

        let out_of_order_messages = vec![
            OpenAICompatibleMessage::Assistant(OpenAICompatibleAssistantMessage {
                content: Some(Value::String("Assistant message".to_string())),
                tool_calls: None,
            }),
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("System message".to_string()),
            }),
        ];
        let result: Input = out_of_order_messages.try_into().unwrap();
        assert_eq!(result.system, Some("System message".into()));
        assert_eq!(
            result.messages,
            vec![InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Text(TextKind::Text {
                    text: "Assistant message".to_string(),
                })],
            }]
        );
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
            vec![InputMessageContent::Text(TextKind::Arguments {
                arguments: json!({
                    "country": "Japan",
                    "city": "Tokyo",
                })
                .as_object()
                .unwrap()
                .clone(),
            })]
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
            vec![InputMessageContent::Text(TextKind::Arguments {
                arguments: json!({
                    "custom_key": "custom_val",
                })
                .as_object()
                .unwrap()
                .clone(),
            })]
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
            vec![InputMessageContent::Text(TextKind::Arguments {
                arguments: json!({
                    "country": "Japan",
                    "city": "Tokyo",
                })
                .as_object()
                .unwrap()
                .clone(),
            })]
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
            vec![InputMessageContent::Text(TextKind::Arguments {
                arguments: json!({
                    "type": "text",
                    "my_custom_arg": 123
                })
                .as_object()
                .unwrap()
                .clone(),
            })]
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
        let mut tool_id_to_index = HashMap::new();
        let (content_str, tool_calls) = process_chat_content_chunk(content, &mut tool_id_to_index);
        assert_eq!(content_str, Some("Hello, world!".to_string()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, Some("1".to_string()));
        assert_eq!(tool_calls[0].index, 0);
        assert_eq!(tool_calls[0].function.name, "test_tool");
        assert_eq!(tool_calls[0].function.arguments, "{}");

        let content: Vec<ContentBlockChunk> = vec![];
        let (content_str, tool_calls) = process_chat_content_chunk(content, &mut tool_id_to_index);
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
            ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "5".to_string(),
                raw_name: "last_tool".to_string(),
                raw_arguments: "{\"key\": \"value\"}".to_string(),
            }),
        ];
        let mut tool_id_to_index = HashMap::new();
        let (content_str, tool_calls) = process_chat_content_chunk(content, &mut tool_id_to_index);
        assert_eq!(
            content_str,
            Some("First part second part third part fourth part".to_string())
        );
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].id, Some("123".to_string()));
        assert_eq!(tool_calls[0].index, 0);
        assert_eq!(tool_calls[0].function.name, "middle_tool");
        assert_eq!(tool_calls[0].function.arguments, "{\"key\": \"value\"}");
        assert_eq!(tool_calls[1].id, Some("5".to_string()));
        assert_eq!(tool_calls[1].index, 1);
        assert_eq!(tool_calls[1].function.name, "last_tool");
        assert_eq!(tool_calls[1].function.arguments, "{\"key\": \"value\"}");
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
