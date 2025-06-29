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
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::{
    current_timestamp, ContentBlockChatOutput, ContentBlockChunk, File, FileKind, FinishReason,
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
use super::model_resolution;
use crate::embeddings::EmbeddingRequest;
use crate::moderation::ModerationProvider;

/// A handler for the OpenAI-compatible inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        kafka_connection_info,
        authentication_info: _,
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
    let stream_options = openai_compatible_params.stream_options;
    let logprobs_requested = matches!(openai_compatible_params.logprobs, Some(true));

    // Resolve the model name based on authentication state
    let model_resolution = model_resolution::resolve_model_name(
        &openai_compatible_params.model,
        &headers,
        false, // not for embedding
    )?;

    let original_model_name = model_resolution.original_model_name.to_string();

    // Create params with resolved model/function names
    let mut params = Params::try_from_openai_with_resolution(
        headers.clone(),
        openai_compatible_params.clone(),
        model_resolution,
    )?;

    // If the caller asked for logprobs, we need the raw provider response so we can
    // copy logprobs back out later.  That is enabled via `include_original_response`.
    if matches!(openai_compatible_params.logprobs, Some(true)) {
        params.include_original_response = true;
    }

    // Extract observability metadata from headers (set by auth middleware)
    if let (Some(project_id), Some(endpoint_id), Some(model_id)) = (
        headers
            .get("x-tensorzero-project-id")
            .and_then(|v| v.to_str().ok()),
        headers
            .get("x-tensorzero-endpoint-id")
            .and_then(|v| v.to_str().ok()),
        headers
            .get("x-tensorzero-model-id")
            .and_then(|v| v.to_str().ok()),
    ) {
        params.observability_metadata = Some(super::inference::ObservabilityMetadata {
            project_id: project_id.to_string(),
            endpoint_id: endpoint_id.to_string(),
            model_id: model_id.to_string(),
        });
    }

    let response = inference(
        config,
        &http_client,
        clickhouse_connection_info,
        kafka_connection_info,
        params,
    )
    .await?;

    match response {
        InferenceOutput::NonStreaming(response) => {
            let mut openai_compatible_response =
                OpenAICompatibleResponse::from((response.clone(), original_model_name.clone()));

            if logprobs_requested {
                // Try to fetch real logprobs from the original provider response (if available)
                if let Some(original_resp_json) = match &response {
                    InferenceResponse::Chat(chat) => chat
                        .original_response
                        .as_ref()
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok()),
                    InferenceResponse::Json(json) => json
                        .original_response
                        .as_ref()
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok()),
                } {
                    if let Some(provider_choices) =
                        original_resp_json.get("choices").and_then(|v| v.as_array())
                    {
                        for (idx, choice) in
                            openai_compatible_response.choices.iter_mut().enumerate()
                        {
                            if let Some(provider_choice) = provider_choices.get(idx) {
                                if let Some(lp) = provider_choice.get("logprobs") {
                                    choice.logprobs = Some(lp.clone());
                                } else {
                                    choice.logprobs = Some(serde_json::json!({"content": []}));
                                }
                            }
                        }
                    }
                } else {
                    // Fallback to empty array if provider didn't send or we failed to parse
                    for choice in &mut openai_compatible_response.choices {
                        choice.logprobs = Some(serde_json::json!({"content": []}));
                    }
                }
            }
            Ok(Json(openai_compatible_response).into_response())
        }
        InferenceOutput::Streaming(stream) => {
            let openai_compatible_stream = prepare_serialized_openai_compatible_events(
                stream,
                original_model_name,
                stream_options,
            );
            Ok(Sse::new(openai_compatible_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response())
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, Default)]
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
    JsonSchema { json_schema: JsonSchemaInfoOption },
    JsonObject,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
enum JsonSchemaInfoOption {
    JsonSchema(JsonSchemaInfo),
    DeprecatedJsonSchema(Value),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct JsonSchemaInfo {
    name: String,
    description: Option<String>,
    schema: Option<Value>,
    #[serde(default)]
    strict: bool,
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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
struct OpenAICompatibleStreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
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
    stream_options: Option<OpenAICompatibleStreamOptions>,
    temperature: Option<f32>,
    tools: Option<Vec<OpenAICompatibleTool>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    top_p: Option<f32>,
    parallel_tool_calls: Option<bool>,
    /// If set to `true`, the response should include per-token log-probabilities.
    logprobs: Option<bool>,
    // Guided decoding / template fields (TensorZero extensions)
    chat_template: Option<String>,
    chat_template_kwargs: Option<Value>,
    mm_processor_kwargs: Option<Value>,
    guided_json: Option<Value>,
    guided_regex: Option<String>,
    guided_choice: Option<Vec<String>>,
    guided_grammar: Option<String>,
    structural_tag: Option<String>,
    guided_decoding_backend: Option<String>,
    guided_whitespace_pattern: Option<String>,
    #[serde(rename = "tensorzero::variant_name")]
    tensorzero_variant_name: Option<String>,
    #[serde(rename = "tensorzero::dryrun")]
    tensorzero_dryrun: Option<bool>,
    #[serde(rename = "tensorzero::episode_id")]
    tensorzero_episode_id: Option<Uuid>,
    #[serde(rename = "tensorzero::cache_options")]
    tensorzero_cache_options: Option<CacheParamsOptions>,
    #[serde(default, rename = "tensorzero::extra_body")]
    tensorzero_extra_body: UnfilteredInferenceExtraBody,
    #[serde(default, rename = "tensorzero::extra_headers")]
    tensorzero_extra_headers: UnfilteredInferenceExtraHeaders,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleChoice {
    index: u32,
    finish_reason: OpenAICompatibleFinishReason,
    message: OpenAICompatibleResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<serde_json::Value>,
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
    service_tier: String,
    object: String,
    usage: OpenAICompatibleUsage,
}

impl Params {
    fn try_from_openai_with_resolution(
        headers: HeaderMap,
        openai_compatible_params: OpenAICompatibleParams,
        model_resolution: model_resolution::ModelResolution,
    ) -> Result<Self, Error> {
        let function_name = model_resolution.function_name;
        let model_name = model_resolution.model_name;

        if let Some(function_name) = &function_name {
            if function_name.is_empty() {
                return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: "function_name cannot be empty".to_string(),
                }
                .into());
            }
        }

        if let Some(model_name) = &model_name {
            if model_name.is_empty() {
                return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: "model_name cannot be empty".to_string(),
                }
                .into());
            }
        }

        Self::create_params(headers, openai_compatible_params, function_name, model_name)
    }

    #[cfg(test)]
    fn try_from_openai(
        headers: HeaderMap,
        openai_compatible_params: OpenAICompatibleParams,
    ) -> Result<Self, Error> {
        const TENSORZERO_FUNCTION_NAME_PREFIX: &str = "tensorzero::function_name::";
        const TENSORZERO_MODEL_NAME_PREFIX: &str = "tensorzero::model_name::";

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
                    message: "function_name (passed in model field after \"tensorzero::function_name::\") cannot be empty".to_string(),
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

        Self::create_params(headers, openai_compatible_params, function_name, model_name)
    }

    fn create_params(
        headers: HeaderMap,
        openai_compatible_params: OpenAICompatibleParams,
        function_name: Option<String>,
        model_name: Option<String>,
    ) -> Result<Self, Error> {
        let header_episode_id = headers
            .get("episode_id")
            .map(|h| {
                tracing::warn!("Deprecation Warning: Please use the `tensorzero::episode_id` field instead of the `episode_id` header. The header will be removed in a future release.");
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
            chat_template: openai_compatible_params.chat_template,
            chat_template_kwargs: openai_compatible_params.chat_template_kwargs,
            mm_processor_kwargs: openai_compatible_params.mm_processor_kwargs,
            guided_json: openai_compatible_params.guided_json,
            guided_regex: openai_compatible_params.guided_regex,
            guided_choice: openai_compatible_params.guided_choice,
            guided_grammar: openai_compatible_params.guided_grammar,
            structural_tag: openai_compatible_params.structural_tag,
            guided_decoding_backend: openai_compatible_params.guided_decoding_backend,
            guided_whitespace_pattern: openai_compatible_params.guided_whitespace_pattern,
            json_mode,
            logprobs: matches!(openai_compatible_params.logprobs, Some(true)),
        };
        let inference_params = InferenceParams {
            chat_completion: chat_completion_inference_params,
        };
        let header_variant_name = headers
            .get("variant_name")
            .map(|h| {
                tracing::warn!("Deprecation Warning: Please use the `tensorzero::variant_name` field instead of the `variant_name` header. The header will be removed in a future release.");
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
                tracing::warn!("Deprecation Warning: Please use the `tensorzero::dryrun` field instead of the `dryrun` header. The header will be removed in a future release.");
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
            Some(OpenAICompatibleResponseFormat::JsonSchema { json_schema }) => match json_schema {
                JsonSchemaInfoOption::JsonSchema(json_schema) => json_schema.schema,
                JsonSchemaInfoOption::DeprecatedJsonSchema(value) => {
                    tracing::warn!("Deprecation Warning: Please provide the correct `name`, `description`, `schema`, and `strict` fields in the `json_schema` field in the response format. Simply providing a JSON schema in this field will be rejected in a future TensorZero release.");
                    Some(value)
                }
            },
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
            cache_options: openai_compatible_params
                .tensorzero_cache_options
                .unwrap_or_default(),
            // For now, we don't support internal inference for OpenAI compatible endpoint
            internal: false,
            tags: HashMap::new(),
            // OpenAI compatible endpoint does not support 'include_original_response'
            include_original_response: false,
            extra_body: openai_compatible_params.tensorzero_extra_body,
            extra_headers: openai_compatible_params.tensorzero_extra_headers,
            observability_metadata: None, // Will be set in the handler
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
enum OpenAICompatibleContentBlock {
    Text(TextContent),
    ImageUrl { image_url: OpenAICompatibleImageUrl },
    File { file: OpenAICompatibleFile },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", deny_unknown_fields, rename_all = "snake_case")]
struct OpenAICompatibleImageUrl {
    url: Url,
}

#[derive(Deserialize, Debug)]
struct OpenAICompatibleFile {
    file_data: String,
    filename: String,
    // OpenAI supports file_id with their files API
    // We do not so we require these two fields
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

fn parse_base64_image_data_url(url: &str) -> Result<(FileKind, &str), Error> {
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
        "image/jpeg" => FileKind::Jpeg,
        "image/png" => FileKind::Png,
        "image/webp" => FileKind::WebP,
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
                            InputMessageContent::File(File::Base64 { mime_type, data: data.to_string() })
                        } else {
                            InputMessageContent::File(File::Url { url: image_url.url })
                        }
                    }
                    Ok(OpenAICompatibleContentBlock::File { file }) => {
                        InputMessageContent::File(File::Base64 { mime_type: file.filename.as_str().try_into()?, data: file.file_data })
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
    fn from((inference_response, model_name): (InferenceResponse, String)) -> Self {
        match inference_response {
            InferenceResponse::Chat(response) => {
                let (content, tool_calls, reasoning_content) =
                    process_chat_content(response.content);

                OpenAICompatibleResponse {
                    id: response.inference_id.to_string(),
                    choices: vec![OpenAICompatibleChoice {
                        index: 0,
                        finish_reason: response.finish_reason.unwrap_or(FinishReason::Stop).into(),
                        message: OpenAICompatibleResponseMessage {
                            content,
                            tool_calls: Some(tool_calls),
                            role: "assistant".to_string(),
                            logprobs: None,
                            reasoning_content,
                        },
                        logprobs: None,
                    }],
                    created: current_timestamp() as u32,
                    model: model_name.clone(),
                    service_tier: "".to_string(),
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
                        content: response.output.raw,
                        tool_calls: None,
                        role: "assistant".to_string(),
                        logprobs: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                }],
                created: current_timestamp() as u32,
                model: model_name,
                system_fingerprint: "".to_string(),
                service_tier: "".to_string(),
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

// Takes a vector of ContentBlockOutput and returns a tuple of (Option<String>, Vec<OpenAICompatibleToolCall>, Option<String>).
// This is useful since the OpenAI format separates text, tool calls, and reasoning content in the response fields.
fn process_chat_content(
    content: Vec<ContentBlockChatOutput>,
) -> (
    Option<String>,
    Vec<OpenAICompatibleToolCall>,
    Option<String>,
) {
    let mut content_str: Option<String> = None;
    let mut tool_calls = Vec::new();
    let mut reasoning_content: Option<String> = None;
    for block in content {
        match block {
            ContentBlockChatOutput::Text(text) => match content_str {
                Some(ref mut content) => content.push_str(&text.text),
                None => content_str = Some(text.text),
            },
            ContentBlockChatOutput::ToolCall(tool_call) => {
                tool_calls.push(tool_call.into());
            }
            ContentBlockChatOutput::Thought(thought) => {
                // Collect reasoning content from thought blocks
                match reasoning_content {
                    Some(ref mut content) => {
                        content.push('\n');
                        content.push_str(&thought.text);
                    }
                    None => reasoning_content = Some(thought.text),
                }
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
    (content_str, tool_calls, reasoning_content)
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
    service_tier: String,
    object: String,
    usage: Option<OpenAICompatibleUsage>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleChoiceChunk {
    index: u32,
    finish_reason: Option<OpenAICompatibleFinishReason>,
    logprobs: Option<()>, // This is always set to None for now
    delta: OpenAICompatibleDelta,
}

fn is_none_or_empty<T>(v: &Option<Vec<T>>) -> bool {
    // if it's None → skip, or if the Vec is empty → skip
    v.as_ref().is_none_or(|vec| vec.is_empty())
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "is_none_or_empty")]
    tool_calls: Option<Vec<OpenAICompatibleToolCallChunk>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

fn convert_inference_response_chunk_to_openai_compatible(
    chunk: InferenceResponseChunk,
    tool_id_to_index: &mut HashMap<String, usize>,
    model_name: &str,
) -> Vec<OpenAICompatibleResponseChunk> {
    let response_chunk = match chunk {
        InferenceResponseChunk::Chat(c) => {
            let (content, tool_calls, reasoning_content) =
                process_chat_content_chunk(c.content, tool_id_to_index);
            OpenAICompatibleResponseChunk {
                id: c.inference_id.to_string(),
                episode_id: c.episode_id.to_string(),
                choices: vec![OpenAICompatibleChoiceChunk {
                    index: 0,
                    finish_reason: c.finish_reason.map(|finish_reason| finish_reason.into()),
                    logprobs: None,
                    delta: OpenAICompatibleDelta {
                        content,
                        tool_calls: Some(tool_calls),
                        reasoning_content,
                    },
                }],
                created: current_timestamp() as u32,
                service_tier: "".to_string(),
                model: model_name.to_string(),
                system_fingerprint: "".to_string(),
                object: "chat.completion.chunk".to_string(),
                // We emit a single chunk containing 'usage' at the end of the stream
                usage: None,
            }
        }
        InferenceResponseChunk::Json(c) => OpenAICompatibleResponseChunk {
            id: c.inference_id.to_string(),
            episode_id: c.episode_id.to_string(),
            choices: vec![OpenAICompatibleChoiceChunk {
                index: 0,
                finish_reason: c.finish_reason.map(|finish_reason| finish_reason.into()),
                logprobs: None,
                delta: OpenAICompatibleDelta {
                    content: Some(c.raw),
                    tool_calls: None,
                    reasoning_content: None,
                },
            }],
            created: current_timestamp() as u32,
            service_tier: "".to_string(),
            model: model_name.to_string(),
            system_fingerprint: "".to_string(),
            object: "chat.completion.chunk".to_string(),
            // We emit a single chunk containing 'usage' at the end of the stream
            usage: None,
        },
    };

    vec![response_chunk]
}

fn process_chat_content_chunk(
    content: Vec<ContentBlockChunk>,
    tool_id_to_index: &mut HashMap<String, usize>,
) -> (
    Option<String>,
    Vec<OpenAICompatibleToolCallChunk>,
    Option<String>,
) {
    let mut content_str: Option<String> = None;
    let mut tool_calls = Vec::new();
    let mut reasoning_content: Option<String> = None;
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
            ContentBlockChunk::Thought(thought) => {
                // Collect reasoning content from thought chunks
                if let Some(thought_text) = thought.text {
                    match reasoning_content {
                        Some(ref mut content) => content.push_str(&thought_text),
                        None => reasoning_content = Some(thought_text),
                    }
                }
            }
        }
    }
    (content_str, tool_calls, reasoning_content)
}

/// Prepares an Event for SSE on the way out of the gateway
/// When None is passed in, we send "[DONE]" to the client to signal the end of the stream
fn prepare_serialized_openai_compatible_events(
    mut stream: InferenceStream,
    model_name: String,
    stream_options: Option<OpenAICompatibleStreamOptions>,
) -> impl Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        let mut tool_id_to_index = HashMap::new();
        let mut is_first_chunk = true;
        let mut total_usage = OpenAICompatibleUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };
        let mut inference_id = None;
        let mut episode_id = None;
        while let Some(chunk) = stream.next().await {
            // NOTE - in the future, we may want to end the stream early if we get an error
            // For now, we just ignore the error and try to get more chunks
            let Ok(chunk) = chunk else {
                continue;
            };
            inference_id = Some(chunk.inference_id());
            episode_id = Some(chunk.episode_id());
            let chunk_usage = match &chunk {
                InferenceResponseChunk::Chat(c) => {
                    &c.usage
                }
                InferenceResponseChunk::Json(c) => {
                    &c.usage
                }
            };
            if let Some(chunk_usage) = chunk_usage {
                total_usage.prompt_tokens += chunk_usage.input_tokens;
                total_usage.completion_tokens += chunk_usage.output_tokens;
                total_usage.total_tokens += chunk_usage.input_tokens + chunk_usage.output_tokens;
            }
            let openai_compatible_chunks = convert_inference_response_chunk_to_openai_compatible(chunk, &mut tool_id_to_index, &model_name);
            for chunk in openai_compatible_chunks {
                let mut chunk_json = serde_json::to_value(chunk).map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to convert chunk to JSON: {e}"),
                    })
                })?;
                if is_first_chunk {
                    // OpenAI includes "assistant" role in the first chunk but not in the subsequent chunks
                    chunk_json["choices"][0]["delta"]["role"] = Value::String("assistant".to_string());
                    is_first_chunk = false;
                }

                yield Event::default().json_data(chunk_json).map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to convert Value to Event: {e}"),
                    })
                })
            }
        }
        if stream_options.map(|s| s.include_usage).unwrap_or(false) {
            let episode_id = episode_id.ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: "Cannot find episode_id - no chunks were produced by TensorZero".to_string(),
                })
            })?;
            let inference_id = inference_id.ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: "Cannot find inference_id - no chunks were produced by TensorZero".to_string(),
                })
            })?;
            let usage_chunk = OpenAICompatibleResponseChunk {
                id: inference_id.to_string(),
                episode_id: episode_id.to_string(),
                choices: vec![],
                created: current_timestamp() as u32,
                model: model_name.clone(),
                system_fingerprint: "".to_string(),
                object: "chat.completion.chunk".to_string(),
                service_tier: "".to_string(),
                usage: Some(OpenAICompatibleUsage {
                    prompt_tokens: total_usage.prompt_tokens,
                    completion_tokens: total_usage.completion_tokens,
                    total_tokens: total_usage.total_tokens,
                }),
            };
            yield Event::default().json_data(
                usage_chunk)
                .map_err(|e| {
                    Error::new(ErrorDetails::Inference {
                        message: format!("Failed to convert usage chunk to JSON: {e}"),
                    })
                });
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

// OpenAI-compatible embedding types and handler

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleEmbeddingParams {
    input: OpenAICompatibleEmbeddingInput,
    model: String,
    #[serde(rename = "tensorzero::cache_options")]
    tensorzero_cache_options: Option<CacheParamsOptions>,
    #[serde(flatten)]
    unknown_fields: HashMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
enum OpenAICompatibleEmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleEmbeddingData {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleEmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleEmbeddingResponse {
    object: String,
    data: Vec<OpenAICompatibleEmbeddingData>,
    model: String,
    usage: OpenAICompatibleEmbeddingUsage,
}

/// A handler for the OpenAI-compatible embedding endpoint
#[debug_handler(state = AppStateData)]
pub async fn embedding_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        kafka_connection_info: _,
        authentication_info: _,
    }): AppState,
    headers: HeaderMap,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleEmbeddingParams>,
) -> Result<Response<Body>, Error> {
    let unknown_fields: Vec<&str> = openai_compatible_params
        .unknown_fields
        .keys()
        .filter(|k| k.as_str() != "encoding_format")
        .map(|k| k.as_str())
        .collect();

    if !unknown_fields.is_empty() {
        tracing::warn!(
            "Ignoring unknown fields in OpenAI-compatible embedding request: {:?}",
            unknown_fields
        );
    }

    // Resolve the model name based on authentication state
    let model_resolution = model_resolution::resolve_model_name(
        &openai_compatible_params.model,
        &headers,
        true, // for embedding
    )?;

    let model_id = model_resolution.model_name.ok_or_else(|| {
        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "Embedding requests must specify a model, not a function".to_string(),
        })
    })?;

    let original_model_name = model_resolution.original_model_name.to_string();

    // Convert OpenAI request to internal format
    let internal_input = match &openai_compatible_params.input {
        OpenAICompatibleEmbeddingInput::Single(text) => {
            crate::embeddings::EmbeddingInput::Single(text.clone())
        }
        OpenAICompatibleEmbeddingInput::Batch(texts) => {
            if texts.is_empty() {
                return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: "Batch embedding requests cannot be empty.".to_string(),
                }));
            }
            crate::embeddings::EmbeddingInput::Batch(texts.clone())
        }
    };

    let encoding_format = openai_compatible_params
        .unknown_fields
        .get("encoding_format")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let embedding_request = EmbeddingRequest {
        input: internal_input,
        encoding_format,
    };

    // Extract model configuration
    use crate::model::ModelTableExt;
    let models = config.models.read().await;
    let model = models
        .get_with_capability(
            &model_id,
            crate::endpoints::capability::EndpointCapability::Embedding,
        )
        .await?
        .ok_or_else(|| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "Model '{original_model_name}' not found or does not support embeddings"
                ),
            })
        })?;

    // Create credentials - empty for now as OpenAI compatible endpoint doesn't support dynamic credentials
    let credentials = InferenceCredentials::default();

    // Create inference clients
    let cache_options: crate::cache::CacheOptions = (
        openai_compatible_params
            .tensorzero_cache_options
            .unwrap_or_default(),
        false, // dryrun is false for now
    )
        .into();
    let clients = super::inference::InferenceClients {
        http_client: &http_client,
        credentials: &credentials,
        clickhouse_connection_info: &clickhouse_connection_info,
        cache_options: &cache_options,
    };

    // Call the model's embedding capability
    let response = model
        .embed(&embedding_request, &original_model_name, &clients)
        .await?;

    // Convert to OpenAI-compatible format
    let openai_response = OpenAICompatibleEmbeddingResponse {
        object: "list".to_string(),
        data: response
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| OpenAICompatibleEmbeddingData {
                object: "embedding".to_string(),
                embedding,
                index,
            })
            .collect(),
        model: original_model_name,
        usage: OpenAICompatibleEmbeddingUsage {
            prompt_tokens: response.usage.input_tokens,
            total_tokens: response.usage.input_tokens,
        },
    };

    Ok(Json(openai_response).into_response())
}

/// OpenAI-compatible moderation request structure
#[derive(Clone, Debug, Deserialize)]
pub struct OpenAICompatibleModerationParams {
    pub input: OpenAICompatibleModerationInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum OpenAICompatibleModerationInput {
    Single(String),
    Batch(Vec<String>),
}

/// OpenAI-compatible moderation response structure
#[derive(Clone, Debug, Serialize)]
pub struct OpenAICompatibleModerationResponse {
    pub id: String,
    pub model: String,
    pub results: Vec<crate::moderation::ModerationResult>,
}

/// A handler for the OpenAI-compatible moderation endpoint
#[debug_handler(state = AppStateData)]
pub async fn moderation_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        kafka_connection_info: _,
        authentication_info: _,
    }): AppState,
    headers: HeaderMap,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleModerationParams>,
) -> Result<Response<Body>, Error> {
    let unknown_fields: Vec<&str> = openai_compatible_params
        .unknown_fields
        .keys()
        .map(|k| k.as_str())
        .collect();

    if !unknown_fields.is_empty() {
        tracing::warn!(
            "Ignoring unknown fields in OpenAI-compatible moderation request: {:?}",
            unknown_fields
        );
    }

    // Default to omni-moderation-latest if no model specified
    let model_name = openai_compatible_params
        .model
        .clone()
        .unwrap_or_else(|| "omni-moderation-latest".to_string());

    // Resolve the model name based on authentication state
    let model_resolution = model_resolution::resolve_model_name(&model_name, &headers, false)?;

    let resolved_model_name = model_resolution
        .model_name
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "Moderation requests must specify a model, not a function".to_string(),
            })
        })?
        .to_string();

    // Convert OpenAI request to internal format
    let internal_input = match &openai_compatible_params.input {
        OpenAICompatibleModerationInput::Single(text) => {
            crate::moderation::ModerationInput::Single(text.clone())
        }
        OpenAICompatibleModerationInput::Batch(texts) => {
            if texts.is_empty() {
                return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: "Batch moderation requests cannot be empty.".to_string(),
                }));
            }
            crate::moderation::ModerationInput::Batch(texts.clone())
        }
    };

    let moderation_request = crate::moderation::ModerationRequest {
        input: internal_input,
        model: None, // Let the provider set the appropriate model name
    };

    // Get the regular model table
    let models = config.models.read().await;

    // Check if the model exists and has moderation capability
    let model_config = models.get(&resolved_model_name).await?.ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!("Model '{resolved_model_name}' not found"),
        })
    })?;

    // Verify the model supports moderation
    if !model_config
        .endpoints
        .contains(&crate::endpoints::capability::EndpointCapability::Moderation)
    {
        return Err(Error::new(ErrorDetails::Config {
            message: format!("Model '{resolved_model_name}' does not support moderation"),
        }));
    }

    // Create credentials - empty for now as OpenAI compatible endpoint doesn't support dynamic credentials
    let credentials = InferenceCredentials::default();

    // Create inference clients with no caching for moderation
    let cache_options = crate::cache::CacheOptions {
        enabled: crate::cache::CacheEnabledMode::Off,
        max_age_s: None,
    };
    let clients = super::inference::InferenceClients {
        http_client: &http_client,
        credentials: &credentials,
        clickhouse_connection_info: &clickhouse_connection_info,
        cache_options: &cache_options,
    };

    // For now, we'll use the first provider in routing that supports moderation
    // This is a temporary solution until we fully integrate moderation into the regular model system
    let mut provider_errors = HashMap::new();
    let mut response = None;

    for provider_name in &model_config.routing {
        let provider = match model_config.providers.get(provider_name) {
            Some(p) => &p.config,
            None => {
                tracing::warn!("Provider {} not found in model config", provider_name);
                continue;
            }
        };

        // Check if this provider supports moderation
        tracing::info!("Checking provider {} for moderation support", provider_name);
        match provider {
            crate::model::ProviderConfig::OpenAI(openai_provider) => {
                tracing::info!("Found OpenAI provider for moderation");
                // For OpenAI, we need to use the provider's configured model name
                let mut provider_request = moderation_request.clone();
                provider_request.model = Some(openai_provider.model_name().to_string());
                // Use the OpenAI provider's moderation capability
                match openai_provider
                    .moderate(&provider_request, clients.http_client, &credentials)
                    .await
                {
                    Ok(provider_response) => {
                        response = Some(crate::moderation::ModerationResponse::new(
                            provider_response,
                            provider_name.clone(),
                        ));
                        break;
                    }
                    Err(e) => {
                        provider_errors.insert(provider_name.to_string(), e);
                        continue;
                    }
                }
            }
            _ => {
                // Other providers don't support moderation yet
                continue;
            }
        }
    }

    let response = response.ok_or_else(|| {
        Error::new(ErrorDetails::Config {
            message: format!("No providers available for moderation model '{resolved_model_name}'"),
        })
    })?;

    // Convert to OpenAI-compatible format
    let openai_response = OpenAICompatibleModerationResponse {
        id: format!("modr-{}", Uuid::now_v7()),
        model: model_resolution.original_model_name.to_string(),
        results: response.results,
    };

    Ok(Json(openai_response).into_response())
}

#[cfg(test)]
mod tests {

    use super::*;
    use axum::http::header::{HeaderName, HeaderValue};
    use serde_json::json;
    use tracing_test::traced_test;

    use crate::cache::CacheEnabledMode;
    use crate::inference::types::{Text, TextChunk};

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
                seed: Some(23),
                temperature: Some(0.5),
                top_p: Some(0.5),
                ..Default::default()
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
        let (content_str, tool_calls, reasoning_content) = process_chat_content(content);
        assert_eq!(content_str, Some("Hello, world!".to_string()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "1");
        assert_eq!(tool_calls[0].function.name, "test_tool");
        assert_eq!(tool_calls[0].function.arguments, "{}");
        assert_eq!(reasoning_content, None);
        let content: Vec<ContentBlockChatOutput> = vec![];
        let (content_str, tool_calls, _reasoning_content) = process_chat_content(content);
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
        let (content_str, tool_calls, reasoning_content) = process_chat_content(content);
        assert_eq!(
            content_str,
            Some("First part second part third part fourth part".to_string())
        );
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(reasoning_content, None);
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
        let (content_str, tool_calls, reasoning_content) =
            process_chat_content_chunk(content, &mut tool_id_to_index);
        assert_eq!(content_str, Some("Hello, world!".to_string()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, Some("1".to_string()));
        assert_eq!(tool_calls[0].index, 0);
        assert_eq!(tool_calls[0].function.name, "test_tool");
        assert_eq!(tool_calls[0].function.arguments, "{}");
        assert_eq!(reasoning_content, None);

        let content: Vec<ContentBlockChunk> = vec![];
        let (content_str, tool_calls, _reasoning_content) =
            process_chat_content_chunk(content, &mut tool_id_to_index);
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
        let (content_str, tool_calls, reasoning_content) =
            process_chat_content_chunk(content, &mut tool_id_to_index);
        assert_eq!(
            content_str,
            Some("First part second part third part fourth part".to_string())
        );
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(reasoning_content, None);
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
            (FileKind::Jpeg, "YWJjCg=="),
            parse_base64_image_data_url("data:image/jpeg;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            (FileKind::Png, "YWJjCg=="),
            parse_base64_image_data_url("data:image/png;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            (FileKind::WebP, "YWJjCg=="),
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

    #[test]
    fn test_cache_options() {
        let headers = HeaderMap::new();

        // Test default cache options (should be write-only)
        let params = Params::try_from_openai(
            headers.clone(),
            OpenAICompatibleParams {
                messages: vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                    content: Value::String("test".to_string()),
                })],
                model: "tensorzero::function_name::test_function".into(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(params.cache_options, CacheParamsOptions::default());

        // Test explicit cache options
        let params = Params::try_from_openai(
            headers.clone(),
            OpenAICompatibleParams {
                messages: vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                    content: Value::String("test".to_string()),
                })],
                model: "tensorzero::function_name::test_function".into(),
                tensorzero_cache_options: Some(CacheParamsOptions {
                    max_age_s: Some(3600),
                    enabled: CacheEnabledMode::On,
                }),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            params.cache_options,
            CacheParamsOptions {
                max_age_s: Some(3600),
                enabled: CacheEnabledMode::On
            }
        );

        // Test interaction with dryrun
        let params = Params::try_from_openai(
            headers.clone(),
            OpenAICompatibleParams {
                messages: vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                    content: Value::String("test".to_string()),
                })],
                model: "tensorzero::function_name::test_function".into(),
                tensorzero_dryrun: Some(true),
                tensorzero_cache_options: Some(CacheParamsOptions {
                    max_age_s: Some(3600),
                    enabled: CacheEnabledMode::On,
                }),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            params.cache_options,
            CacheParamsOptions {
                max_age_s: Some(3600),
                enabled: CacheEnabledMode::On,
            }
        );

        // Test write-only with dryrun (should become Off)
        let params = Params::try_from_openai(
            headers,
            OpenAICompatibleParams {
                messages: vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                    content: Value::String("test".to_string()),
                })],
                model: "tensorzero::function_name::test_function".into(),
                tensorzero_dryrun: Some(true),
                tensorzero_cache_options: Some(CacheParamsOptions {
                    max_age_s: None,
                    enabled: CacheEnabledMode::WriteOnly,
                }),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            params.cache_options,
            CacheParamsOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly
            }
        );
    }

    // Tests for OpenAI-compatible embeddings endpoint

    #[test]
    fn test_openai_compatible_embedding_params_deserialization() {
        // Test single string input
        let json_single = json!({
            "input": "Hello, world!",
            "model": "text-embedding-ada-002"
        });

        let params: OpenAICompatibleEmbeddingParams = serde_json::from_value(json_single).unwrap();
        assert_eq!(params.model, "text-embedding-ada-002");
        match params.input {
            OpenAICompatibleEmbeddingInput::Single(text) => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected single input"),
        }
        assert!(params.tensorzero_cache_options.is_none());

        // Test batch input
        let json_batch = json!({
            "input": ["Hello", "World", "Test"],
            "model": "text-embedding-ada-002"
        });

        let params: OpenAICompatibleEmbeddingParams = serde_json::from_value(json_batch).unwrap();
        match params.input {
            OpenAICompatibleEmbeddingInput::Batch(texts) => {
                assert_eq!(texts, vec!["Hello", "World", "Test"]);
            }
            _ => panic!("Expected batch input"),
        }

        // Test with TensorZero cache options
        let json_with_cache = json!({
            "input": "Test input",
            "model": "embedding-model",
            "tensorzero::cache_options": {
                "max_age_s": 3600,
                "enabled": "on"
            }
        });

        let params: OpenAICompatibleEmbeddingParams =
            serde_json::from_value(json_with_cache).unwrap();
        assert!(params.tensorzero_cache_options.is_some());
        let cache_options = params.tensorzero_cache_options.unwrap();
        assert_eq!(cache_options.max_age_s, Some(3600));
        assert_eq!(cache_options.enabled, CacheEnabledMode::On);

        // Test with unknown fields (should be captured)
        let json_unknown = json!({
            "input": "Test",
            "model": "embedding-model",
            "encoding_format": "float",
            "dimensions": 1536,
            "user": "test-user"
        });

        let params: OpenAICompatibleEmbeddingParams = serde_json::from_value(json_unknown).unwrap();
        assert!(!params.unknown_fields.is_empty());
        assert!(params.unknown_fields.contains_key("encoding_format"));
        assert!(params.unknown_fields.contains_key("dimensions"));
        assert!(params.unknown_fields.contains_key("user"));
    }

    #[test]
    fn test_openai_compatible_embedding_response_serialization() {
        let response = OpenAICompatibleEmbeddingResponse {
            object: "list".to_string(),
            data: vec![OpenAICompatibleEmbeddingData {
                object: "embedding".to_string(),
                embedding: vec![0.1, 0.2, 0.3, -0.4],
                index: 0,
            }],
            model: "text-embedding-ada-002".to_string(),
            usage: OpenAICompatibleEmbeddingUsage {
                prompt_tokens: 5,
                total_tokens: 5,
            },
        };

        let json_value = serde_json::to_value(&response).unwrap();

        // Verify structure matches OpenAI API spec
        assert_eq!(json_value["object"], "list");
        assert_eq!(json_value["model"], "text-embedding-ada-002");

        let data = &json_value["data"].as_array().unwrap()[0];
        assert_eq!(data["object"], "embedding");
        assert_eq!(data["index"], 0);
        assert_eq!(data["embedding"].as_array().unwrap().len(), 4);
        assert!((data["embedding"][0].as_f64().unwrap() - 0.1).abs() < 1e-6);
        assert!((data["embedding"][3].as_f64().unwrap() - (-0.4)).abs() < 1e-6);

        let usage = &json_value["usage"];
        assert_eq!(usage["prompt_tokens"], 5);
        assert_eq!(usage["total_tokens"], 5);
    }

    #[test]
    fn test_openai_compatible_embedding_input_variants() {
        // Test single string
        let single_json = json!("Hello, world!");
        let single_input: OpenAICompatibleEmbeddingInput =
            serde_json::from_value(single_json).unwrap();
        match single_input {
            OpenAICompatibleEmbeddingInput::Single(text) => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected single input"),
        }

        // Test array of strings
        let batch_json = json!(["Hello", "World", "Test"]);
        let batch_input: OpenAICompatibleEmbeddingInput =
            serde_json::from_value(batch_json).unwrap();
        match batch_input {
            OpenAICompatibleEmbeddingInput::Batch(texts) => {
                assert_eq!(texts.len(), 3);
                assert_eq!(texts[0], "Hello");
                assert_eq!(texts[1], "World");
                assert_eq!(texts[2], "Test");
            }
            _ => panic!("Expected batch input"),
        }

        // Test empty array
        let empty_json = json!([]);
        let empty_input: OpenAICompatibleEmbeddingInput =
            serde_json::from_value(empty_json).unwrap();
        match empty_input {
            OpenAICompatibleEmbeddingInput::Batch(texts) => assert_eq!(texts.len(), 0),
            _ => panic!("Expected batch input"),
        }
    }

    #[test]
    fn test_embedding_data_structure() {
        let embedding_data = OpenAICompatibleEmbeddingData {
            object: "embedding".to_string(),
            embedding: vec![1.0, -0.5, 0.0, 0.7],
            index: 42,
        };

        let json = serde_json::to_value(&embedding_data).unwrap();
        assert_eq!(json["object"], "embedding");
        assert_eq!(json["index"], 42);

        let embedding_array = json["embedding"].as_array().unwrap();
        assert_eq!(embedding_array.len(), 4);
        assert!((embedding_array[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((embedding_array[1].as_f64().unwrap() - (-0.5)).abs() < 1e-6);
        assert!((embedding_array[2].as_f64().unwrap() - 0.0).abs() < 1e-6);
        assert!((embedding_array[3].as_f64().unwrap() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_usage_structure() {
        let usage = OpenAICompatibleEmbeddingUsage {
            prompt_tokens: 100,
            total_tokens: 100,
        };

        let json = serde_json::to_value(&usage).unwrap();
        assert_eq!(json["prompt_tokens"], 100);
        assert_eq!(json["total_tokens"], 100);

        // Verify that completion_tokens is not included (embeddings don't have completion tokens)
        assert!(!json.as_object().unwrap().contains_key("completion_tokens"));
    }

    #[test]
    fn test_embedding_response_multiple_embeddings() {
        // Test response with multiple embeddings (though we don't support batch yet)
        let response = OpenAICompatibleEmbeddingResponse {
            object: "list".to_string(),
            data: vec![
                OpenAICompatibleEmbeddingData {
                    object: "embedding".to_string(),
                    embedding: vec![0.1, 0.2],
                    index: 0,
                },
                OpenAICompatibleEmbeddingData {
                    object: "embedding".to_string(),
                    embedding: vec![0.3, 0.4],
                    index: 1,
                },
            ],
            model: "test-model".to_string(),
            usage: OpenAICompatibleEmbeddingUsage {
                prompt_tokens: 10,
                total_tokens: 10,
            },
        };

        let json = serde_json::to_value(&response).unwrap();
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["index"], 0);
        assert_eq!(data[1]["index"], 1);
        let embedding0 = data[0]["embedding"].as_array().unwrap();
        assert!((embedding0[0].as_f64().unwrap() - 0.1).abs() < 1e-6);
        assert!((embedding0[1].as_f64().unwrap() - 0.2).abs() < 1e-6);

        let embedding1 = data[1]["embedding"].as_array().unwrap();
        assert!((embedding1[0].as_f64().unwrap() - 0.3).abs() < 1e-6);
        assert!((embedding1[1].as_f64().unwrap() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_params_model_name_extraction() {
        // Test model name prefixes that might be used in the handler
        let test_cases = vec![
            ("tensorzero::embedding_model_name::my-model", "my-model"),
            ("tensorzero::model_name::my-model", "my-model"),
            ("plain-model-name", "plain-model-name"),
            (
                "tensorzero::function_name::my-function",
                "tensorzero::function_name::my-function",
            ), // Should not be stripped
        ];

        for (input_model, expected_output) in test_cases {
            // This tests the logic that would be used in the embedding_handler
            let extracted = if let Some(model_name) =
                input_model.strip_prefix("tensorzero::embedding_model_name::")
            {
                model_name.to_string()
            } else if let Some(model_name) = input_model.strip_prefix("tensorzero::model_name::") {
                model_name.to_string()
            } else {
                input_model.to_string()
            };

            assert_eq!(
                extracted, expected_output,
                "Failed for input: {input_model}"
            );
        }
    }
}
