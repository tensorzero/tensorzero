//! Chat completion types and conversion logic for OpenAI-compatible API.
//!
//! This module contains all request/response types for the chat completion endpoint,
//! including message structures, parameter types, and conversion logic between
//! OpenAI-compatible formats and TensorZero's internal representations.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use tensorzero_derive::TensorZeroDeserialize;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::config::UninitializedVariantInfo;
use crate::endpoints::inference::{
    ChatCompletionInferenceParams, InferenceCredentials, InferenceParams, InferenceResponse, Params,
};
use crate::endpoints::openai_compatible::types::input_files::{
    OpenAICompatibleFile, OpenAICompatibleImageUrl, OpenAICompatibleInputAudio,
    convert_file_to_base64, convert_image_url_to_file, convert_input_audio_to_file,
};
use crate::endpoints::openai_compatible::types::is_none_or_empty;
use crate::endpoints::openai_compatible::types::tool::{
    ChatCompletionToolChoiceOption, OpenAICompatibleTool, OpenAICompatibleToolCall,
    OpenAICompatibleToolChoiceParams, OpenAICompatibleToolMessage,
};
use crate::endpoints::openai_compatible::types::usage::OpenAICompatibleUsage;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::chat_completion_inference_params::ServiceTier;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::usage::{RawResponseEntry, RawUsageEntry};
use crate::inference::types::{
    Arguments, ContentBlockChatOutput, FinishReason, Input, InputMessage, InputMessageContent,
    RawText, Role, System, Template, Text, Thought, Unknown, current_timestamp,
};
use crate::tool::{DynamicToolParams, ProviderTool, ToolResult};
use crate::variant::JsonMode;

// ============================================================================
// Message Types
// ============================================================================

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleSystemMessage {
    pub content: Value,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleUserMessage {
    pub content: Value,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAICompatibleAssistantMessage {
    pub content: Option<Value>,
    pub tool_calls: Option<Vec<OpenAICompatibleToolCall>>,
    #[serde(default)]
    pub tensorzero_extra_content_experimental: Option<Vec<InputExtraContentBlock>>,
}

#[derive(Clone, Debug, PartialEq, TensorZeroDeserialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum OpenAICompatibleMessage {
    #[serde(alias = "developer")]
    System(OpenAICompatibleSystemMessage),
    User(OpenAICompatibleUserMessage),
    Assistant(OpenAICompatibleAssistantMessage),
    Tool(OpenAICompatibleToolMessage),
}

// ============================================================================
// Request Parameter Types
// ============================================================================

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAICompatibleResponseFormat {
    Text,
    JsonSchema { json_schema: JsonSchemaInfo },
    JsonObject,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct JsonSchemaInfo {
    pub name: String,
    pub description: Option<String>,
    pub schema: Option<Value>,
    #[serde(default)]
    pub strict: bool,
}

impl std::fmt::Display for JsonSchemaInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
pub struct OpenAICompatibleStreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct OpenAICompatibleParams {
    pub messages: Vec<OpenAICompatibleMessage>,
    pub model: String,
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub response_format: Option<OpenAICompatibleResponseFormat>,
    pub seed: Option<u32>,
    pub stream: Option<bool>,
    pub stream_options: Option<OpenAICompatibleStreamOptions>,
    pub temperature: Option<f32>,
    pub tools: Option<Vec<OpenAICompatibleTool>>,
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    pub top_p: Option<f32>,
    pub parallel_tool_calls: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<ServiceTier>,
    pub verbosity: Option<String>,
    pub n: Option<u32>,
    #[serde(rename = "tensorzero::variant_name")]
    pub tensorzero_variant_name: Option<String>,
    #[serde(rename = "tensorzero::dryrun")]
    pub tensorzero_dryrun: Option<bool>,
    #[serde(rename = "tensorzero::episode_id")]
    pub tensorzero_episode_id: Option<Uuid>,
    #[serde(rename = "tensorzero::cache_options")]
    pub tensorzero_cache_options: Option<CacheParamsOptions>,
    #[serde(default, rename = "tensorzero::extra_body")]
    pub tensorzero_extra_body: UnfilteredInferenceExtraBody,
    #[serde(default, rename = "tensorzero::extra_headers")]
    pub tensorzero_extra_headers: UnfilteredInferenceExtraHeaders,
    #[serde(default, rename = "tensorzero::tags")]
    pub tensorzero_tags: HashMap<String, String>,
    #[serde(default, rename = "tensorzero::deny_unknown_fields")]
    pub tensorzero_deny_unknown_fields: bool,
    #[serde(default, rename = "tensorzero::credentials")]
    pub tensorzero_credentials: InferenceCredentials,
    #[serde(rename = "tensorzero::internal_dynamic_variant_config")]
    pub tensorzero_internal_dynamic_variant_config: Option<UninitializedVariantInfo>,
    #[serde(default, rename = "tensorzero::provider_tools")]
    pub tensorzero_provider_tools: Vec<ProviderTool>,
    #[serde(default, rename = "tensorzero::params")]
    pub tensorzero_params: Option<InferenceParams>,
    #[serde(default, rename = "tensorzero::include_raw_usage")]
    pub tensorzero_include_raw_usage: bool,
    /// DEPRECATED (#5697 / 2026.4+): Use `tensorzero::include_raw_response` instead.
    #[serde(default, rename = "tensorzero::include_original_response")]
    pub tensorzero_include_original_response: bool,
    #[serde(default, rename = "tensorzero::include_raw_response")]
    pub tensorzero_include_raw_response: bool,
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, Value>,
}

// ============================================================================
// Response Types
// ============================================================================

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleResponseMessage {
    pub content: Option<String>,
    #[serde(skip_serializing_if = "is_none_or_empty")]
    pub tool_calls: Option<Vec<OpenAICompatibleToolCall>>,
    pub role: String,
    #[serde(skip_serializing_if = "is_none_or_empty")]
    pub tensorzero_extra_content_experimental: Option<Vec<ExtraContentBlock>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleChoice {
    pub index: u32,
    pub finish_reason: OpenAICompatibleFinishReason,
    pub message: OpenAICompatibleResponseMessage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAICompatibleFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    // FunctionCall: we never generate this and it is deprecated
}

impl From<FinishReason> for OpenAICompatibleFinishReason {
    fn from(finish_reason: FinishReason) -> Self {
        match finish_reason {
            FinishReason::Stop => OpenAICompatibleFinishReason::Stop,
            FinishReason::StopSequence => OpenAICompatibleFinishReason::Stop,
            FinishReason::Length => OpenAICompatibleFinishReason::Length,
            FinishReason::ContentFilter => OpenAICompatibleFinishReason::ContentFilter,
            FinishReason::ToolCall => OpenAICompatibleFinishReason::ToolCalls,
            FinishReason::Unknown => OpenAICompatibleFinishReason::Stop, // OpenAI doesn't have an unknown finish reason so we coerce
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleResponse {
    pub id: String,
    pub episode_id: String,
    pub choices: Vec<OpenAICompatibleChoice>,
    pub created: u32,
    pub model: String,
    pub system_fingerprint: String,
    pub service_tier: Option<String>,
    pub object: String,
    pub usage: OpenAICompatibleUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_usage: Option<Vec<RawUsageEntry>>,
    /// DEPRECATED (#5697 / 2026.4+): Use `tensorzero_raw_response` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_original_response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_response: Option<Vec<RawResponseEntry>>,
}

/// Extra content block for OpenAI-compatible responses (Thought or Unknown)
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExtraContentBlock {
    Thought {
        insert_index: usize,
        #[serde(flatten)]
        thought: Thought,
    },
    Unknown {
        insert_index: usize,
        #[serde(flatten)]
        unknown: Unknown,
    },
}

/// Extra content block for OpenAI-compatible input (with optional insert_index)
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputExtraContentBlock {
    Thought {
        insert_index: Option<usize>,
        #[serde(flatten)]
        thought: Thought,
    },
    Unknown {
        insert_index: Option<usize>,
        #[serde(flatten)]
        unknown: Unknown,
    },
}

// ============================================================================
// Content Block Types
// ============================================================================

#[derive(Debug, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum OpenAICompatibleContentBlock {
    Text(TextContent),
    ImageUrl {
        image_url: OpenAICompatibleImageUrl,
    },
    File {
        file: OpenAICompatibleFile,
    },
    InputAudio {
        input_audio: OpenAICompatibleInputAudio,
    },
    #[serde(rename = "tensorzero::raw_text")]
    RawText(RawText),
    #[serde(rename = "tensorzero::template")]
    Template(Template),
}

#[derive(Debug)]
// Two mutually exclusive modes - the standard OpenAI text, and our special TensorZero mode
pub enum TextContent {
    /// A normal openai text content block: `{"type": "text", "text": "Some content"}`. The `type` key comes from the parent `OpenAICompatibleContentBlock`
    Text { text: String },
    /// A special TensorZero mode: `{"type": "text", "tensorzero::arguments": {"custom_key": "custom_val"}}`.
    TensorZeroArguments { tensorzero_arguments: Arguments },
}

impl<'de> Deserialize<'de> for TextContent {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let mut object: Map<String, Value> = Map::deserialize(de)?;
        let text = object.remove("text");
        let arguments = object.remove("tensorzero::arguments");
        match (text, arguments) {
            (Some(text), None) => Ok(TextContent::Text {
                text: match text {
                    Value::String(text) => text,
                    _ => {
                        return Err(serde::de::Error::custom(
                            "`text` must be a string when using `\"type\": \"text\"`",
                        ));
                    }
                },
            }),
            (None, Some(arguments)) => Ok(TextContent::TensorZeroArguments {
                tensorzero_arguments: match arguments {
                    Value::Object(arguments) => Arguments(arguments),
                    _ => {
                        return Err(serde::de::Error::custom(
                            "`tensorzero::arguments` must be an object when using `\"type\": \"text\"`",
                        ));
                    }
                },
            }),
            (Some(_), Some(_)) => Err(serde::de::Error::custom(
                "Only one of `text` or `tensorzero::arguments` can be set when using `\"type\": \"text\"`",
            )),
            (None, None) => Err(serde::de::Error::custom(
                "Either `text` or `tensorzero::arguments` must be set when using `\"type\": \"text\"`",
            )),
        }
    }
}

// ============================================================================
// Conversion Implementations
// ============================================================================

const TENSORZERO_FUNCTION_NAME_PREFIX: &str = "tensorzero::function_name::";
const TENSORZERO_MODEL_NAME_PREFIX: &str = "tensorzero::model_name::";

impl Params {
    pub fn try_from_openai(
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
        } else {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "`model` field must start with `tensorzero::function_name::` or `tensorzero::model_name::`. For example, `tensorzero::function_name::my_function` for a function `my_function` defined in your config, `tensorzero::model_name::my_model` for a model `my_model` defined in your config, or default functions like `tensorzero::model_name::openai::gpt-4o-mini`.".to_string(),
            }));
        };

        if let Some(function_name) = &function_name
            && function_name.is_empty()
        {
            return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                message:
                    "function_name (passed in model field after \"tensorzero::function_name::\") cannot be empty"
                        .to_string(),
            }
            .into());
        }

        if let Some(model_name) = &model_name
            && model_name.is_empty()
        {
            return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: "model_name (passed in model field after \"tensorzero::model_name::\") cannot be empty".to_string(),
                }
                .into());
        }

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
        let input = openai_messages_to_input(openai_compatible_params.messages)?;

        let mut inference_params = openai_compatible_params
            .tensorzero_params
            .unwrap_or_default();

        // Collect the inference parameters.
        // `tensorzero::params` takes precedence over inferred values from OpenAI-compatible parameters.
        // This is necessary so that users can set things like `json_mode="tool"` dynamically; otherwise it'll always be `"strict"`.
        // TODO (GabrielBianconi): Should we warn if we override parameters that are already set?
        inference_params.chat_completion = ChatCompletionInferenceParams {
            frequency_penalty: inference_params
                .chat_completion
                .frequency_penalty
                .or(openai_compatible_params.frequency_penalty),
            json_mode: inference_params.chat_completion.json_mode.or(json_mode),
            max_tokens: inference_params.chat_completion.max_tokens.or(max_tokens),
            presence_penalty: inference_params
                .chat_completion
                .presence_penalty
                .or(openai_compatible_params.presence_penalty),
            reasoning_effort: inference_params
                .chat_completion
                .reasoning_effort
                .or(openai_compatible_params.reasoning_effort),
            service_tier: inference_params
                .chat_completion
                .service_tier
                .or(openai_compatible_params.service_tier),
            seed: inference_params
                .chat_completion
                .seed
                .or(openai_compatible_params.seed),
            stop_sequences: inference_params
                .chat_completion
                .stop_sequences
                .or(openai_compatible_params.stop),
            temperature: inference_params
                .chat_completion
                .temperature
                .or(openai_compatible_params.temperature),
            thinking_budget_tokens: inference_params.chat_completion.thinking_budget_tokens,
            top_p: inference_params
                .chat_completion
                .top_p
                .or(openai_compatible_params.top_p),
            verbosity: inference_params
                .chat_completion
                .verbosity
                .or(openai_compatible_params.verbosity),
        };

        let OpenAICompatibleToolChoiceParams {
            allowed_tools,
            tool_choice,
        } = openai_compatible_params
            .tool_choice
            .map(ChatCompletionToolChoiceOption::into_tool_params)
            .unwrap_or_default();
        let dynamic_tool_params = DynamicToolParams {
            allowed_tools,
            additional_tools: openai_compatible_params
                .tools
                .map(|tools| tools.into_iter().map(|t| t.into()).collect()),
            tool_choice,
            parallel_tool_calls: openai_compatible_params.parallel_tool_calls,
            provider_tools: openai_compatible_params.tensorzero_provider_tools,
        };
        let output_schema = match openai_compatible_params.response_format {
            Some(OpenAICompatibleResponseFormat::JsonSchema { json_schema }) => json_schema.schema,
            _ => None,
        };
        Ok(Params {
            function_name,
            model_name,
            episode_id: openai_compatible_params.tensorzero_episode_id,
            input,
            stream: openai_compatible_params.stream,
            params: inference_params,
            variant_name: openai_compatible_params.tensorzero_variant_name,
            dryrun: openai_compatible_params.tensorzero_dryrun,
            dynamic_tool_params,
            output_schema,
            credentials: openai_compatible_params.tensorzero_credentials,
            cache_options: openai_compatible_params
                .tensorzero_cache_options
                .unwrap_or_default(),
            // For now, we don't support internal inference for OpenAI compatible endpoint
            internal: false,
            tags: openai_compatible_params.tensorzero_tags,
            include_original_response: openai_compatible_params
                .tensorzero_include_original_response,
            include_raw_response: openai_compatible_params.tensorzero_include_raw_response,
            include_raw_usage: openai_compatible_params.tensorzero_include_raw_usage,
            extra_body: openai_compatible_params.tensorzero_extra_body,
            extra_headers: openai_compatible_params.tensorzero_extra_headers,
            internal_dynamic_variant_config: openai_compatible_params
                .tensorzero_internal_dynamic_variant_config,
        })
    }
}

pub fn openai_messages_to_input(
    openai_compatible_messages: Vec<OpenAICompatibleMessage>,
) -> Result<Input, Error> {
    let mut system_message = None;
    let mut messages = Vec::new();
    let mut tool_call_id_to_name = HashMap::new();
    let first_system = matches!(
        openai_compatible_messages.first(),
        Some(OpenAICompatibleMessage::System(_))
    );
    let mut iter = openai_compatible_messages.into_iter().peekable();
    while let Some(message) = iter.next() {
        match message {
            OpenAICompatibleMessage::System(msg) => {
                let had_prior_system = system_message.is_some();
                let system_content =
                    convert_openai_message_content("system".to_string(), msg.content.clone())?;

                for content in system_content {
                    let text = match content {
                        InputMessageContent::Text(t) => Some(t.text),
                        InputMessageContent::RawText(rt) => Some(rt.value),
                        InputMessageContent::Template(t) => {
                            if system_message.is_some() {
                                return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                                    message:
                                        "System message cannot contain template with other content"
                                            .to_string(),
                                }
                                .into());
                            }
                            system_message = Some(System::Template(t.arguments));
                            None
                        }
                        _ => return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message:
                                "System message must contain only text or template content blocks"
                                    .to_string(),
                        }
                        .into()),
                    };

                    if let Some(text) = text {
                        match &mut system_message {
                            None => system_message = Some(System::Text(text)),
                            Some(System::Text(s)) => {
                                s.push('\n');
                                s.push_str(&text);
                            }
                            Some(System::Template(_)) => {
                                return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                                    message: "Cannot add text after template system message"
                                        .to_string(),
                                }
                                .into());
                            }
                        }
                    }
                }

                if had_prior_system {
                    tracing::warn!(
                        "Multiple system messages provided. They will be concatenated and moved to the start of the conversation."
                    );
                } else if !first_system {
                    tracing::warn!("Moving system message to the start of the conversation.");
                }
            }
            OpenAICompatibleMessage::User(msg) => {
                messages.push(InputMessage {
                    role: Role::User,
                    content: convert_openai_message_content("user".to_string(), msg.content)?,
                });
            }
            OpenAICompatibleMessage::Assistant(msg) => {
                let mut message_content = Vec::new();
                if let Some(content) = msg.content {
                    message_content.extend(convert_openai_message_content(
                        "assistant".to_string(),
                        content,
                    )?);
                }
                if let Some(tool_calls) = msg.tool_calls {
                    for tool_call in tool_calls {
                        tool_call_id_to_name
                            .insert(tool_call.id.clone(), tool_call.function.name.clone());
                        message_content.push(InputMessageContent::ToolCall(tool_call.into()));
                    }
                }
                // Process extra content with index-based insertion
                if let Some(extra_content) = msg.tensorzero_extra_content_experimental {
                    // First pass: items with insert_index (in input order)
                    for block in &extra_content {
                        match block {
                            InputExtraContentBlock::Thought {
                                insert_index: Some(idx),
                                thought,
                            } => {
                                let idx = (*idx).min(message_content.len());
                                message_content
                                    .insert(idx, InputMessageContent::Thought(thought.clone()));
                            }
                            InputExtraContentBlock::Unknown {
                                insert_index: Some(idx),
                                unknown,
                            } => {
                                let idx = (*idx).min(message_content.len());
                                message_content
                                    .insert(idx, InputMessageContent::Unknown(unknown.clone()));
                            }
                            _ => {} // Handle in second pass
                        }
                    }

                    // Second pass: items without insert_index (prepend to beginning)
                    // We iterate in reverse so the first unindexed item ends up first
                    for block in extra_content.into_iter().rev() {
                        match block {
                            InputExtraContentBlock::Thought {
                                insert_index: None,
                                thought,
                            } => {
                                message_content.insert(0, InputMessageContent::Thought(thought));
                            }
                            InputExtraContentBlock::Unknown {
                                insert_index: None,
                                unknown,
                            } => {
                                message_content.insert(0, InputMessageContent::Unknown(unknown));
                            }
                            _ => {} // Already handled
                        }
                    }
                }
                messages.push(InputMessage {
                    role: Role::Assistant,
                    content: message_content,
                });
            }
            OpenAICompatibleMessage::Tool(msg) => {
                // When we encounter a tool result, coalesce all subsequent tool results into a single
                // `Role::User` message.
                // This ensures that parallel tool call results can be passed through properly to providers
                // (e.g. AWS Bedrock) that require parallel tool call results to occur in the same
                // `Role::User` message.
                let mut tool_results = Vec::new();
                let name = tool_call_id_to_name
                    .get(&msg.tool_call_id)
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                            message: "tool call id not found".to_string(),
                        })
                    })?
                    .to_string();
                tool_results.push(InputMessageContent::ToolResult(ToolResult {
                    id: msg.tool_call_id,
                    name,
                    result: msg.content.unwrap_or_default().to_string(),
                }));

                while let Some(message) = iter.peek() {
                    if let OpenAICompatibleMessage::Tool(tool_result) = message {
                        let name = tool_call_id_to_name
                            .get(&tool_result.tool_call_id)
                            .ok_or_else(|| {
                                Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                                    message: "tool call id not found".to_string(),
                                })
                            })?
                            .to_string();
                        tool_results.push(InputMessageContent::ToolResult(ToolResult {
                            id: tool_result.tool_call_id.clone(),
                            name,
                            result: tool_result.content.clone().unwrap_or_default().to_string(),
                        }));
                        // Consume the tool result that we just peeked
                        iter.next();
                    } else {
                        break;
                    }
                }
                messages.push(InputMessage {
                    role: Role::User,
                    content: tool_results,
                });
            }
        }
    }

    Ok(Input {
        system: system_message,
        messages,
    })
}

pub fn convert_openai_message_content(
    role: String,
    content: Value,
) -> Result<Vec<InputMessageContent>, Error> {
    match content {
        Value::String(s) => Ok(vec![InputMessageContent::Text(Text { text: s })]),
        Value::Array(a) => {
            let mut outputs = Vec::with_capacity(a.len());
            for val in a {
                let block = serde_json::from_value::<OpenAICompatibleContentBlock>(val.clone());
                let output = match block {
                    Ok(OpenAICompatibleContentBlock::RawText(raw_text)) => InputMessageContent::RawText(raw_text),
                    Ok(OpenAICompatibleContentBlock::Template(template)) => InputMessageContent::Template(template),
                    Ok(OpenAICompatibleContentBlock::Text(TextContent::Text { text })) => InputMessageContent::Text(Text { text }),
                    Ok(OpenAICompatibleContentBlock::Text(TextContent::TensorZeroArguments { tensorzero_arguments })) => {
                        crate::utils::deprecation_warning("Using `tensorzero::arguments` in text content blocks is deprecated. Please use `{{\"type\": \"tensorzero::template\", \"name\": \"role\", \"arguments\": {{...}}}}` instead.");
                        InputMessageContent::Template(Template { name: role.clone(), arguments: tensorzero_arguments })
                    }
                    Ok(OpenAICompatibleContentBlock::ImageUrl { image_url }) => {
                        InputMessageContent::File(convert_image_url_to_file(image_url)?)
                    }
                    Ok(OpenAICompatibleContentBlock::File { file }) => {
                        InputMessageContent::File(convert_file_to_base64(file)?)
                    }
                    Ok(OpenAICompatibleContentBlock::InputAudio { input_audio }) => {
                        InputMessageContent::File(convert_input_audio_to_file(input_audio)?)
                    }
                    Err(e) => {
                        if let Some(obj) = val.as_object() {
                            // If the user tried using any 'tensorzero::' fields, we assume that they were deliberately trying to use TensorZero,
                            // and weren't passing in some other OpenAI-compatible content block type that we don't know about.
                            // We emit an error in this case, since the user incorrectly used TensorZero-specific values
                            if obj.keys().any(|k| k.starts_with("tensorzero::")) {
                                return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                                    message: format!("Invalid TensorZero content block: {e}"),
                                }));
                            } else if obj.keys().any(|k| *k == "type") {
                                // If the 'type' key is set, assume that the user was trying to specify an OpenAI-compatible content block,
                                // (rather than using the deprecated behavior of directly passing a JSON object for the TensorZero function arguments),
                                // Since we encountered a parse error, we reject this as an invalid OpenAI-compatible content block
                                return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                                    message: format!("Invalid content block: {e}"),
                                }));
                            }
                        }
                        crate::utils::deprecation_warning(&format!(r#"Content block `{val}` was not a valid OpenAI content block. Please use `{{"type": "tensorzero::template", "name": "role", "arguments": {{"custom": "data"}}}}` to pass arbitrary JSON values to TensorZero: {e}"#));
                        if let Value::Object(obj) = val {
                            InputMessageContent::Template(Template { name: role.clone(), arguments: Arguments(obj) })
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

impl From<(InferenceResponse, String, bool, bool)> for OpenAICompatibleResponse {
    fn from(
        (
            inference_response,
            response_model_prefix,
            include_original_response,
            include_raw_response,
        ): (InferenceResponse, String, bool, bool),
    ) -> Self {
        match inference_response {
            InferenceResponse::Chat(response) => {
                let (content, tool_calls, extra_content) = process_chat_content(response.content);
                let tensorzero_original_response = if include_original_response {
                    response.original_response
                } else {
                    None
                };
                let tensorzero_raw_response = if include_raw_response {
                    response.raw_response
                } else {
                    None
                };

                OpenAICompatibleResponse {
                    id: response.inference_id.to_string(),
                    choices: vec![OpenAICompatibleChoice {
                        index: 0,
                        finish_reason: response.finish_reason.unwrap_or(FinishReason::Stop).into(),
                        message: OpenAICompatibleResponseMessage {
                            content,
                            tool_calls: if tool_calls.is_empty() {
                                None
                            } else {
                                Some(tool_calls)
                            },
                            role: "assistant".to_string(),
                            tensorzero_extra_content_experimental: if extra_content.is_empty() {
                                None
                            } else {
                                Some(extra_content)
                            },
                        },
                    }],
                    created: current_timestamp() as u32,
                    model: format!("{response_model_prefix}{}", response.variant_name),
                    service_tier: None,
                    system_fingerprint: String::new(),
                    object: "chat.completion".to_string(),
                    usage: response.usage.into(),
                    tensorzero_raw_usage: response.raw_usage,
                    tensorzero_original_response,
                    tensorzero_raw_response,
                    episode_id: response.episode_id.to_string(),
                }
            }
            InferenceResponse::Json(response) => {
                let tensorzero_original_response = if include_original_response {
                    response.original_response
                } else {
                    None
                };
                let tensorzero_raw_response = if include_raw_response {
                    response.raw_response
                } else {
                    None
                };

                OpenAICompatibleResponse {
                    id: response.inference_id.to_string(),
                    choices: vec![OpenAICompatibleChoice {
                        index: 0,
                        finish_reason: response.finish_reason.unwrap_or(FinishReason::Stop).into(),
                        message: OpenAICompatibleResponseMessage {
                            content: response.output.raw,
                            tool_calls: None,
                            role: "assistant".to_string(),
                            tensorzero_extra_content_experimental: None,
                        },
                    }],
                    created: current_timestamp() as u32,
                    model: format!("{response_model_prefix}{}", response.variant_name),
                    system_fingerprint: String::new(),
                    service_tier: None,
                    object: "chat.completion".to_string(),
                    usage: response.usage.into(),
                    tensorzero_raw_usage: response.raw_usage,
                    tensorzero_original_response,
                    tensorzero_raw_response,
                    episode_id: response.episode_id.to_string(),
                }
            }
        }
    }
}

/// Takes a vector of ContentBlockOutput and returns a tuple of
/// (Option<String>, Vec<OpenAICompatibleToolCall>, Vec<ExtraContentBlock>).
/// This is useful since the OpenAI format separates text, tool calls, and extra content in the response fields.
pub fn process_chat_content(
    content: Vec<ContentBlockChatOutput>,
) -> (
    Option<String>,
    Vec<OpenAICompatibleToolCall>,
    Vec<ExtraContentBlock>,
) {
    let mut content_str: Option<String> = None;
    let mut tool_calls = Vec::new();
    let mut extra_content = Vec::new();
    for (insert_index, block) in content.into_iter().enumerate() {
        match block {
            ContentBlockChatOutput::Text(text) => match content_str {
                Some(ref mut content) => content.push_str(&text.text),
                None => content_str = Some(text.text),
            },
            ContentBlockChatOutput::ToolCall(tool_call) => {
                tool_calls.push(tool_call.into());
            }
            ContentBlockChatOutput::Thought(thought) => {
                extra_content.push(ExtraContentBlock::Thought {
                    insert_index,
                    thought,
                });
            }
            ContentBlockChatOutput::Unknown(unknown) => {
                extra_content.push(ExtraContentBlock::Unknown {
                    insert_index,
                    unknown,
                });
            }
        }
    }
    (content_str, tool_calls, extra_content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::cache::CacheEnabledMode;
    use crate::endpoints::openai_compatible::types::tool::OpenAICompatibleFunctionCall;
    use crate::tool::{InferenceResponseToolCall, ToolCallWrapper};

    #[test]
    fn test_try_from_openai_compatible_params() {
        let episode_id = Uuid::now_v7();
        let messages = vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
            content: Value::String("Hello, world!".to_string()),
        })];
        let tensorzero_tags = HashMap::from([("test".to_string(), "test".to_string())]);
        let params = Params::try_from_openai(OpenAICompatibleParams {
            messages,
            model: "tensorzero::function_name::test_function".into(),
            frequency_penalty: Some(0.5),
            max_tokens: Some(100),
            max_completion_tokens: Some(50),
            presence_penalty: Some(0.5),
            seed: Some(23),
            temperature: Some(0.5),
            top_p: Some(0.5),
            tensorzero_episode_id: Some(episode_id),
            tensorzero_variant_name: Some("test_variant".to_string()),
            tensorzero_tags: tensorzero_tags.clone(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(params.function_name, Some("test_function".to_string()));
        assert_eq!(params.episode_id, Some(episode_id));
        assert_eq!(params.variant_name, Some("test_variant".to_string()));
        assert_eq!(params.input.messages.len(), 1);
        assert_eq!(params.input.messages[0].role, Role::User);
        assert_eq!(
            params.input.messages[0].content[0],
            InputMessageContent::Text(Text {
                text: "Hello, world!".to_string(),
            })
        );
        assert_eq!(params.params.chat_completion.temperature, Some(0.5));
        assert_eq!(params.params.chat_completion.max_tokens, Some(50));
        assert_eq!(params.params.chat_completion.seed, Some(23));
        assert_eq!(params.params.chat_completion.top_p, Some(0.5));
        assert_eq!(params.params.chat_completion.presence_penalty, Some(0.5));
        assert_eq!(params.params.chat_completion.frequency_penalty, Some(0.5));
        assert_eq!(params.tags, tensorzero_tags);
    }

    #[test]
    fn test_try_from_parallel_tool_calls() {
        // Try an assistant message with text and tool calls
        let messages = vec![
            OpenAICompatibleMessage::Assistant(OpenAICompatibleAssistantMessage {
                content: Some(Value::String("Hello, world!".to_string())),
                tool_calls: Some(vec![OpenAICompatibleToolCall {
                    id: "1".to_string(),
                    r#type: "function".to_string(),
                    function: OpenAICompatibleFunctionCall {
                        name: "test_tool".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tensorzero_extra_content_experimental: None,
            }),
            OpenAICompatibleMessage::Assistant(OpenAICompatibleAssistantMessage {
                content: None,
                tool_calls: Some(vec![OpenAICompatibleToolCall {
                    id: "2".to_string(),
                    r#type: "function".to_string(),
                    function: OpenAICompatibleFunctionCall {
                        name: "test_tool".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tensorzero_extra_content_experimental: None,
            }),
            OpenAICompatibleMessage::Assistant(OpenAICompatibleAssistantMessage {
                content: None,
                tool_calls: Some(vec![OpenAICompatibleToolCall {
                    id: "3".to_string(),
                    r#type: "function".to_string(),
                    function: OpenAICompatibleFunctionCall {
                        name: "test_tool".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tensorzero_extra_content_experimental: None,
            }),
            OpenAICompatibleMessage::Tool(OpenAICompatibleToolMessage {
                content: Some(Value::String("Tool result 1".to_string())),
                tool_call_id: "1".to_string(),
            }),
            OpenAICompatibleMessage::Tool(OpenAICompatibleToolMessage {
                content: Some(Value::String("Tool result 2".to_string())),
                tool_call_id: "2".to_string(),
            }),
            OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                content: Value::String("First message".to_string()),
            }),
            OpenAICompatibleMessage::Tool(OpenAICompatibleToolMessage {
                content: Some(Value::String("Tool result 3".to_string())),
                tool_call_id: "3".to_string(),
            }),
            OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                content: Value::String("Second message".to_string()),
            }),
        ];
        let input: Input = openai_messages_to_input(messages).unwrap();

        assert_eq!(
            input.messages,
            vec![
                InputMessage {
                    role: Role::Assistant,
                    content: vec![
                        InputMessageContent::Text(Text {
                            text: "Hello, world!".to_string(),
                        }),
                        InputMessageContent::ToolCall(ToolCallWrapper::InferenceResponseToolCall(
                            InferenceResponseToolCall {
                                id: "1".to_string(),
                                raw_name: "test_tool".to_string(),
                                raw_arguments: "{}".to_string(),
                                name: None,
                                arguments: None
                            }
                        )),
                    ]
                },
                InputMessage {
                    role: Role::Assistant,
                    content: vec![InputMessageContent::ToolCall(
                        ToolCallWrapper::InferenceResponseToolCall(InferenceResponseToolCall {
                            id: "2".to_string(),
                            raw_name: "test_tool".to_string(),
                            raw_arguments: "{}".to_string(),
                            name: None,
                            arguments: None
                        })
                    ),]
                },
                InputMessage {
                    role: Role::Assistant,
                    content: vec![InputMessageContent::ToolCall(
                        ToolCallWrapper::InferenceResponseToolCall(InferenceResponseToolCall {
                            id: "3".to_string(),
                            raw_name: "test_tool".to_string(),
                            raw_arguments: "{}".to_string(),
                            name: None,
                            arguments: None
                        })
                    )]
                },
                InputMessage {
                    role: Role::User,
                    content: vec![
                        InputMessageContent::ToolResult(ToolResult {
                            name: "test_tool".to_string(),
                            result: "\"Tool result 1\"".to_string(),
                            id: "1".to_string()
                        }),
                        InputMessageContent::ToolResult(ToolResult {
                            name: "test_tool".to_string(),
                            result: "\"Tool result 2\"".to_string(),
                            id: "2".to_string()
                        })
                    ]
                },
                InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "First message".to_string(),
                    })]
                },
                InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::ToolResult(ToolResult {
                        name: "test_tool".to_string(),
                        result: "\"Tool result 3\"".to_string(),
                        id: "3".to_string()
                    })]
                },
                InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Second message".to_string(),
                    })]
                }
            ]
        );
    }

    #[test]
    fn test_try_from_openai_compatible_messages() {
        let messages = vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
            content: Value::String("Hello, world!".to_string()),
        })];
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::User);
        assert_eq!(
            input.messages[0].content[0],
            InputMessageContent::Text(Text {
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
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::User);
        assert_eq!(
            input.system,
            Some(System::Text("You are a helpful assistant".to_string()))
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
        let input: Result<Input, Error> = openai_messages_to_input(messages);
        let error = input.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
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
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(
            input.system,
            Some(System::Text(
                "You are a helpful assistant 1.\nYou are a helpful assistant 2.".to_string()
            ))
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
                tensorzero_extra_content_experimental: None,
            },
        )];
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::Assistant);
        assert_eq!(
            input.messages[0].content[0],
            InputMessageContent::Template(Template {
                name: "assistant".to_string(),
                arguments: Arguments(
                    json!({
                        "country": "Japan",
                        "city": "Tokyo",
                    })
                    .as_object()
                    .unwrap()
                    .clone()
                ),
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
                tensorzero_extra_content_experimental: None,
            },
        )];
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::Assistant);
        assert_eq!(input.messages[0].content.len(), 2);

        let expected_text = InputMessageContent::Text(Text {
            text: "Hello, world!".to_string(),
        });
        let expected_tool_call = InputMessageContent::ToolCall(
            ToolCallWrapper::InferenceResponseToolCall(InferenceResponseToolCall {
                id: "1".to_string(),
                raw_name: "test_tool".to_string(),
                raw_arguments: "{}".to_string(),
                name: None,
                arguments: None,
            }),
        );

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
                tensorzero_extra_content_experimental: None,
            }),
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("System message".to_string()),
            }),
        ];
        let result: Input = openai_messages_to_input(out_of_order_messages).unwrap();
        assert_eq!(
            result.system,
            Some(System::Text("System message".to_string()))
        );
        assert_eq!(
            result.messages,
            vec![InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Text(Text {
                    text: "Assistant message".to_string(),
                })],
            }]
        );

        // Try a system message with legacy template format (tensorzero::arguments)
        let messages = vec![OpenAICompatibleMessage::System(
            OpenAICompatibleSystemMessage {
                content: json!([{
                    "type": "text",
                    "tensorzero::arguments": {
                        "assistant_name": "Alfred Pennyworth"
                    }
                }]),
            },
        )];
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(
            input.system,
            Some(System::Template(Arguments(
                json!({"assistant_name": "Alfred Pennyworth"})
                    .as_object()
                    .unwrap()
                    .clone()
            )))
        );
        assert_eq!(input.messages.len(), 0);

        // Try a system message with new template format
        let messages = vec![OpenAICompatibleMessage::System(
            OpenAICompatibleSystemMessage {
                content: json!([{
                    "type": "tensorzero::template",
                    "name": "system",
                    "arguments": {
                        "assistant_name": "Jarvis"
                    }
                }]),
            },
        )];
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(
            input.system,
            Some(System::Template(Arguments(
                json!({"assistant_name": "Jarvis"})
                    .as_object()
                    .unwrap()
                    .clone()
            )))
        );

        // Error: system message with template and text content (multiple content blocks)
        let messages = vec![OpenAICompatibleMessage::System(
            OpenAICompatibleSystemMessage {
                content: json!([
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "text",
                        "tensorzero::arguments": {
                            "assistant_name": "Alfred"
                        }
                    }
                ]),
            },
        )];
        let result: Result<Input, Error> = openai_messages_to_input(messages);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            *error.get_details(),
            ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "System message cannot contain template with other content".to_string(),
            }
        );

        // Error: text system message followed by template system message
        let messages = vec![
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("You are helpful.".to_string()),
            }),
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: json!([{
                    "type": "text",
                    "tensorzero::arguments": {
                        "assistant_name": "Alfred"
                    }
                }]),
            }),
        ];
        let result: Result<Input, Error> = openai_messages_to_input(messages);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            *error.get_details(),
            ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "System message cannot contain template with other content".to_string(),
            }
        );

        // Error: template system message followed by text system message
        let messages = vec![
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: json!([{
                    "type": "text",
                    "tensorzero::arguments": {
                        "assistant_name": "Alfred"
                    }
                }]),
            }),
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("You are helpful.".to_string()),
            }),
        ];
        let result: Result<Input, Error> = openai_messages_to_input(messages);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            *error.get_details(),
            ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "Cannot add text after template system message".to_string(),
            }
        );

        // Error: multiple template system messages
        let messages = vec![
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: json!([{
                    "type": "text",
                    "tensorzero::arguments": {
                        "assistant_name": "Alfred"
                    }
                }]),
            }),
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: json!([{
                    "type": "tensorzero::template",
                    "name": "system",
                    "arguments": {
                        "assistant_name": "Jarvis"
                    }
                }]),
            }),
        ];
        let result: Result<Input, Error> = openai_messages_to_input(messages);
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            *error.get_details(),
            ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "System message cannot contain template with other content".to_string(),
            }
        );

        // Success: system message with multiple text content blocks (should concatenate within single message)
        let messages = vec![OpenAICompatibleMessage::System(
            OpenAICompatibleSystemMessage {
                content: json!([
                    {"type": "text", "text": "You are helpful."},
                    {"type": "text", "text": "You are concise."}
                ]),
            },
        )];
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(
            input.system,
            Some(System::Text(
                "You are helpful.\nYou are concise.".to_string()
            ))
        );

        // Success: system message with tensorzero::raw_text
        let messages = vec![OpenAICompatibleMessage::System(
            OpenAICompatibleSystemMessage {
                content: json!([{
                    "type": "tensorzero::raw_text",
                    "value": "Raw system text"
                }]),
            },
        )];
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(
            input.system,
            Some(System::Text("Raw system text".to_string()))
        );
    }

    #[test]
    fn test_convert_openai_message_content() {
        // text content
        let content = "Hello, world!".to_string();
        let value =
            convert_openai_message_content("user".to_string(), Value::String(content.clone()))
                .unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::Text(Text { text: content })]
        );
        // tensorzero::raw_text
        let content = json!([{
            "type": "tensorzero::raw_text",
            "value": "This is raw text"
        }]);
        let value = convert_openai_message_content("user".to_string(), content.clone()).unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::RawText(RawText {
                value: "This is raw text".to_string()
            })]
        );
        // tensorzero::arguments
        let content = json!([{
            "country": "Japan",
            "city": "Tokyo",
        }]);
        let value = convert_openai_message_content("user".to_string(), content.clone()).unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(
                    json!({
                        "country": "Japan",
                        "city": "Tokyo",
                    })
                    .as_object()
                    .unwrap()
                    .clone()
                ),
            })]
        );
        let content = json!({
            "country": "Japan",
            "city": "Tokyo",
        });
        let error =
            convert_openai_message_content("user".to_string(), content.clone()).unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "message content must either be a string or an array of length 1 containing structured TensorZero inputs".to_string(),
            }
        );
        let content = json!([]);
        let messages = convert_openai_message_content("user".to_string(), content).unwrap();
        assert_eq!(messages, vec![]);

        let arguments_block = json!([{
            "type": "text",
            "tensorzero::arguments": {
                "custom_key": "custom_val"
            }
        }]);
        let value = convert_openai_message_content("user".to_string(), arguments_block).unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(
                    json!({
                        "custom_key": "custom_val",
                    })
                    .as_object()
                    .unwrap()
                    .clone()
                ),
            })]
        );

        let template_block = json!([{
            "type": "tensorzero::template",
            "name": "my_template",
            "arguments": {
                "custom_key": "custom_val",
            }
        }]);
        let value = convert_openai_message_content("user".to_string(), template_block).unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::Template(Template {
                name: "my_template".to_string(),
                arguments: Arguments(
                    json!({ "custom_key": "custom_val" })
                        .as_object()
                        .unwrap()
                        .clone()
                )
            })]
        );
    }

    #[test]
    fn test_deprecated_custom_block() {
        let logs_contain = crate::utils::testing::capture_logs();
        let content = json!([{
            "country": "Japan",
            "city": "Tokyo",
        }]);
        let value = convert_openai_message_content("user".to_string(), content.clone()).unwrap();
        assert_eq!(
            value,
            vec![InputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(
                    json!({
                        "country": "Japan",
                        "city": "Tokyo",
                    })
                    .as_object()
                    .unwrap()
                    .clone()
                ),
            })]
        );
        assert!(logs_contain(
            r#"Content block `{"country":"Japan","city":"Tokyo"}` was not a valid OpenAI content block."#
        ));

        let other_content = json!([{
            "type": "text",
            "my_custom_arg": 123
        }]);
        let err = convert_openai_message_content("user".to_string(), other_content.clone())
            .expect_err("Should not accept invalid block");
        assert_eq!(
            err.to_string(),
            "Invalid request to OpenAI-compatible endpoint: Invalid content block: Either `text` or `tensorzero::arguments` must be set when using `\"type\": \"text\"`"
        );
    }

    #[test]
    fn test_process_chat_content() {
        let content = vec![
            ContentBlockChatOutput::Text(Text {
                text: "Hello".to_string(),
            }),
            ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
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
        let (content_str, tool_calls, thoughts) = process_chat_content(content);
        assert_eq!(content_str, Some("Hello, world!".to_string()));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "1");
        assert_eq!(tool_calls[0].function.name, "test_tool");
        assert_eq!(tool_calls[0].function.arguments, "{}");
        assert!(thoughts.is_empty());

        let content: Vec<ContentBlockChatOutput> = vec![];
        let (content_str, tool_calls, thoughts) = process_chat_content(content);
        assert_eq!(content_str, None);
        assert!(tool_calls.is_empty());
        assert!(thoughts.is_empty());

        let content = vec![
            ContentBlockChatOutput::Text(Text {
                text: "First part".to_string(),
            }),
            ContentBlockChatOutput::Text(Text {
                text: " second part".to_string(),
            }),
            ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
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
        let (content_str, tool_calls, thoughts) = process_chat_content(content);
        assert_eq!(
            content_str,
            Some("First part second part third part fourth part".to_string())
        );
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "123");
        assert_eq!(tool_calls[0].function.name, "middle_tool");
        assert_eq!(tool_calls[0].function.arguments, "{\"key\": \"value\"}");
        assert!(thoughts.is_empty());
    }

    #[test]
    fn test_cache_options() {
        // Test default cache options (should be write-only)
        let params = Params::try_from_openai(OpenAICompatibleParams {
            messages: vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                content: Value::String("test".to_string()),
            })],
            model: "tensorzero::function_name::test_function".into(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(params.cache_options, CacheParamsOptions::default());

        // Test explicit cache options
        let params = Params::try_from_openai(OpenAICompatibleParams {
            messages: vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                content: Value::String("test".to_string()),
            })],
            model: "tensorzero::function_name::test_function".into(),
            tensorzero_cache_options: Some(CacheParamsOptions {
                max_age_s: Some(3600),
                enabled: CacheEnabledMode::On,
            }),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            params.cache_options,
            CacheParamsOptions {
                max_age_s: Some(3600),
                enabled: CacheEnabledMode::On
            }
        );

        // Test interaction with dryrun
        let params = Params::try_from_openai(OpenAICompatibleParams {
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
        })
        .unwrap();
        assert_eq!(
            params.cache_options,
            CacheParamsOptions {
                max_age_s: Some(3600),
                enabled: CacheEnabledMode::On,
            }
        );

        // Test write-only with dryrun (should become Off)
        let params = Params::try_from_openai(OpenAICompatibleParams {
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
        })
        .unwrap();
        assert_eq!(
            params.cache_options,
            CacheParamsOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly
            }
        );
    }

    #[test]
    fn test_process_chat_content_with_extra_content() {
        let content = vec![
            ContentBlockChatOutput::Thought(Thought {
                text: Some("Let me think about this...".to_string()),
                signature: Some("sig123".to_string()),
                summary: None,
                provider_type: Some("anthropic".to_string()),
                extra_data: None,
            }),
            ContentBlockChatOutput::Text(Text {
                text: "Hello".to_string(),
            }),
            ContentBlockChatOutput::Unknown(Unknown {
                data: serde_json::json!({"custom": "data"}),
                model_name: Some("test_model".to_string()),
                provider_name: None,
            }),
            ContentBlockChatOutput::Thought(Thought {
                text: Some("Another thought".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
                extra_data: None,
            }),
        ];
        let (content_str, tool_calls, extra_content) = process_chat_content(content);
        assert_eq!(
            content_str,
            Some("Hello".to_string()),
            "Text content should be extracted"
        );
        assert!(tool_calls.is_empty(), "No tool calls expected");
        assert_eq!(
            extra_content.len(),
            3,
            "Should have three extra content blocks"
        );

        // First: Thought at insert_index 0
        match &extra_content[0] {
            ExtraContentBlock::Thought {
                insert_index,
                thought,
            } => {
                assert_eq!(*insert_index, 0, "First thought should have insert_index 0");
                assert_eq!(thought.text, Some("Let me think about this...".to_string()));
                assert_eq!(thought.signature, Some("sig123".to_string()));
                assert_eq!(thought.provider_type, Some("anthropic".to_string()));
            }
            ExtraContentBlock::Unknown { .. } => panic!("Expected Thought at position 0"),
        }

        // Second: Unknown at insert_index 2
        match &extra_content[1] {
            ExtraContentBlock::Unknown {
                insert_index,
                unknown,
            } => {
                assert_eq!(*insert_index, 2, "Unknown should have insert_index 2");
                assert_eq!(unknown.data, serde_json::json!({"custom": "data"}));
                assert_eq!(unknown.model_name, Some("test_model".to_string()));
            }
            ExtraContentBlock::Thought { .. } => panic!("Expected Unknown at position 1"),
        }

        // Third: Thought at insert_index 3
        match &extra_content[2] {
            ExtraContentBlock::Thought {
                insert_index,
                thought,
            } => {
                assert_eq!(
                    *insert_index, 3,
                    "Second thought should have insert_index 3"
                );
                assert_eq!(thought.text, Some("Another thought".to_string()));
                assert_eq!(thought.signature, None);
            }
            ExtraContentBlock::Unknown { .. } => panic!("Expected Thought at position 2"),
        }
    }

    #[test]
    fn test_input_extra_content_ordering() {
        // Test with indexed and unindexed extra content blocks
        let messages = vec![OpenAICompatibleMessage::Assistant(
            OpenAICompatibleAssistantMessage {
                content: Some(Value::String("Response text".to_string())),
                tool_calls: None,
                tensorzero_extra_content_experimental: Some(vec![
                    // Unindexed thought - should be appended
                    InputExtraContentBlock::Thought {
                        insert_index: None,
                        thought: Thought {
                            text: Some("Unindexed thought 1".to_string()),
                            signature: None,
                            summary: None,
                            provider_type: None,
                            extra_data: None,
                        },
                    },
                    // Indexed thought at position 2
                    InputExtraContentBlock::Thought {
                        insert_index: Some(2),
                        thought: Thought {
                            text: Some("Indexed thought at 2".to_string()),
                            signature: Some("sig456".to_string()),
                            summary: None,
                            provider_type: Some("anthropic".to_string()),
                            extra_data: None,
                        },
                    },
                    // Unindexed unknown - should be appended
                    InputExtraContentBlock::Unknown {
                        insert_index: None,
                        unknown: Unknown {
                            data: serde_json::json!({"test": "data"}),
                            model_name: None,
                            provider_name: None,
                        },
                    },
                    // Indexed thought at position 0
                    InputExtraContentBlock::Thought {
                        insert_index: Some(0),
                        thought: Thought {
                            text: Some("Indexed thought at 0".to_string()),
                            signature: None,
                            summary: None,
                            provider_type: None,
                            extra_data: None,
                        },
                    },
                ]),
            },
        )];
        let input: Input = openai_messages_to_input(messages).unwrap();
        assert_eq!(input.messages.len(), 1, "Should have one message");
        let content = &input.messages[0].content;

        // Expected order based on the algorithm:
        // 1. Start with text content: ["Response text"]
        // 2. First pass - items with insert_index (in input order):
        //    - "Indexed thought at 2" insert at min(2, 1)=1: ["Response text", "Indexed at 2"]
        //    - "Indexed thought at 0" insert at min(0, 2)=0: ["Indexed at 0", "Response text", "Indexed at 2"]
        // 3. Second pass - items without insert_index (prepend to beginning, in reverse order):
        //    - Insert Unknown at 0: [Unknown, "Indexed at 0", "Response text", "Indexed at 2"]
        //    - Insert "Unindexed thought 1" at 0: ["Unindexed 1", Unknown, "Indexed at 0", "Response text", "Indexed at 2"]
        assert_eq!(content.len(), 5, "Should have 5 content blocks");

        // Verify content order
        match &content[0] {
            InputMessageContent::Thought(t) => {
                assert_eq!(
                    t.text,
                    Some("Unindexed thought 1".to_string()),
                    "First should be unindexed thought 1 (prepended)"
                );
            }
            _ => panic!("Expected Thought at position 0"),
        }
        match &content[1] {
            InputMessageContent::Unknown(u) => {
                assert_eq!(
                    u.data,
                    serde_json::json!({"test": "data"}),
                    "Second should be the unknown block (prepended)"
                );
            }
            _ => panic!("Expected Unknown at position 1"),
        }
        match &content[2] {
            InputMessageContent::Thought(t) => {
                assert_eq!(
                    t.text,
                    Some("Indexed thought at 0".to_string()),
                    "Third should be indexed thought at 0"
                );
            }
            _ => panic!("Expected Thought at position 2"),
        }
        match &content[3] {
            InputMessageContent::Text(t) => {
                assert_eq!(t.text, "Response text", "Fourth should be the text content");
            }
            _ => panic!("Expected Text at position 3"),
        }
        match &content[4] {
            InputMessageContent::Thought(t) => {
                assert_eq!(
                    t.text,
                    Some("Indexed thought at 2".to_string()),
                    "Fifth should be indexed thought at 2"
                );
                assert_eq!(
                    t.signature,
                    Some("sig456".to_string()),
                    "Should preserve signature"
                );
                assert_eq!(
                    t.provider_type,
                    Some("anthropic".to_string()),
                    "Should preserve provider_type"
                );
            }
            _ => panic!("Expected Thought at position 4"),
        }
    }

    #[test]
    fn test_input_extra_content_block_deserialization() {
        // Test Thought variant deserialization with insert_index
        let json_thought_with_index = json!({
            "type": "thought",
            "insert_index": 5,
            "text": "My thought",
            "signature": "sig789",
            "provider_type": "anthropic"
        });
        let block: InputExtraContentBlock =
            serde_json::from_value(json_thought_with_index).unwrap();
        match block {
            InputExtraContentBlock::Thought {
                insert_index,
                thought,
            } => {
                assert_eq!(insert_index, Some(5));
                assert_eq!(thought.text, Some("My thought".to_string()));
                assert_eq!(thought.signature, Some("sig789".to_string()));
                assert_eq!(thought.provider_type, Some("anthropic".to_string()));
            }
            InputExtraContentBlock::Unknown { .. } => panic!("Expected Thought variant"),
        }

        // Test Thought variant deserialization without insert_index
        let json_thought_without_index = json!({
            "type": "thought",
            "text": "Another thought"
        });
        let block: InputExtraContentBlock =
            serde_json::from_value(json_thought_without_index).unwrap();
        match block {
            InputExtraContentBlock::Thought {
                insert_index,
                thought,
            } => {
                assert_eq!(insert_index, None);
                assert_eq!(thought.text, Some("Another thought".to_string()));
                assert_eq!(thought.signature, None);
            }
            InputExtraContentBlock::Unknown { .. } => panic!("Expected Thought variant"),
        }

        // Test Unknown variant deserialization
        let json_unknown = json!({
            "type": "unknown",
            "insert_index": 2,
            "data": {"custom": "value"},
            "model_name": "test_model"
        });
        let block: InputExtraContentBlock = serde_json::from_value(json_unknown).unwrap();
        match block {
            InputExtraContentBlock::Unknown {
                insert_index,
                unknown,
            } => {
                assert_eq!(insert_index, Some(2));
                assert_eq!(unknown.data, json!({"custom": "value"}));
                assert_eq!(unknown.model_name, Some("test_model".to_string()));
            }
            InputExtraContentBlock::Thought { .. } => panic!("Expected Unknown variant"),
        }
    }

    #[test]
    fn test_extra_content_block_serialization() {
        // Test Thought variant serialization
        let thought_block = ExtraContentBlock::Thought {
            insert_index: 3,
            thought: Thought {
                text: Some("Thinking...".to_string()),
                signature: Some("sig_abc".to_string()),
                summary: None,
                provider_type: Some("anthropic".to_string()),
                extra_data: None,
            },
        };
        let json = serde_json::to_value(&thought_block).unwrap();

        // With tagged enum + flatten, type tag and flattened fields at top level
        assert_eq!(json["type"], "thought");
        assert_eq!(json["insert_index"], 3);
        assert_eq!(json["text"], "Thinking...");
        assert_eq!(json["signature"], "sig_abc");
        assert_eq!(json["provider_type"], "anthropic");

        // Test Unknown variant serialization
        let unknown_block = ExtraContentBlock::Unknown {
            insert_index: 5,
            unknown: Unknown {
                data: json!({"custom": "data"}),
                model_name: Some("test_model".to_string()),
                provider_name: None,
            },
        };
        let json = serde_json::to_value(&unknown_block).unwrap();

        assert_eq!(json["type"], "unknown");
        assert_eq!(json["insert_index"], 5);
        assert_eq!(json["data"], json!({"custom": "data"}));
        assert_eq!(json["model_name"], "test_model");
    }

    // ==========================================================================
    // Insert Index Round-Trip Tests
    // ==========================================================================
    //
    // These tests verify that the insert_index algorithm correctly preserves
    // content ordering through a split→recombine cycle:
    // 1. Start with an original array of content blocks
    // 2. Split into main content (text/tool_calls) and extra content (thoughts/unknowns with insert_index)
    // 3. Recombine using the insertion algorithm
    // 4. Verify the result matches the original

    /// Represents content blocks for testing (simplified from ContentBlockChatOutput)
    #[derive(Clone, Debug, PartialEq)]
    enum TestContentBlock {
        Text(String),
        Thought(String),
        Unknown(String),
    }

    /// Splits an array of content blocks into main content and extra content with insert_index.
    /// This simulates what `process_chat_content` does on the output side.
    fn split_content(
        original: Vec<TestContentBlock>,
    ) -> (Vec<TestContentBlock>, Vec<(usize, TestContentBlock)>) {
        let mut main_content = vec![];
        let mut extra_content = vec![];

        for (idx, block) in original.into_iter().enumerate() {
            match block {
                TestContentBlock::Text(_) => main_content.push(block),
                TestContentBlock::Thought(_) | TestContentBlock::Unknown(_) => {
                    extra_content.push((idx, block));
                }
            }
        }

        (main_content, extra_content)
    }

    /// Recombines main content and extra content using the insert_index algorithm.
    /// This simulates what `openai_messages_to_input` does on the input side.
    fn recombine_content(
        main_content: Vec<TestContentBlock>,
        extra_content: Vec<(usize, TestContentBlock)>,
    ) -> Vec<TestContentBlock> {
        let mut result = main_content;

        // Insert items at their insert_index positions (in input order)
        for (insert_index, block) in extra_content {
            let idx = insert_index.min(result.len());
            result.insert(idx, block);
        }

        result
    }

    /// Helper to run a round-trip test case
    fn assert_roundtrip(original: Vec<TestContentBlock>, description: &str) {
        let original_clone = original.clone();
        let (main_content, extra_content) = split_content(original);
        let recombined = recombine_content(main_content, extra_content);
        assert_eq!(
            recombined, original_clone,
            "Round-trip failed for case: {description}"
        );
    }

    #[test]
    fn test_insert_index_roundtrip_simple_interleaving() {
        // Original: [Thought, Text, Thought]
        // Split: main=[Text], extra=[(0, Thought), (2, Thought)]
        // Recombine: insert Thought@0 → [Thought], insert Thought@2 → [Thought, Text, Thought]
        // (But wait - after first insert, Text is at index 1, so insert at 2 goes after Text)
        let original = vec![
            TestContentBlock::Thought("T1".into()),
            TestContentBlock::Text("Hello".into()),
            TestContentBlock::Thought("T2".into()),
        ];
        assert_roundtrip(original, "simple interleaving");
    }

    #[test]
    fn test_insert_index_roundtrip_multiple_thoughts_at_start() {
        // Original: [Thought, Thought, Text]
        let original = vec![
            TestContentBlock::Thought("T1".into()),
            TestContentBlock::Thought("T2".into()),
            TestContentBlock::Text("Hello".into()),
        ];
        assert_roundtrip(original, "multiple thoughts at start");
    }

    #[test]
    fn test_insert_index_roundtrip_thoughts_at_end() {
        // Original: [Text, Thought, Thought]
        let original = vec![
            TestContentBlock::Text("Hello".into()),
            TestContentBlock::Thought("T1".into()),
            TestContentBlock::Thought("T2".into()),
        ];
        assert_roundtrip(original, "thoughts at end");
    }

    #[test]
    fn test_insert_index_roundtrip_mixed_thoughts_and_unknowns() {
        // Original: [Thought, Text, Unknown, Text, Thought]
        let original = vec![
            TestContentBlock::Thought("T1".into()),
            TestContentBlock::Text("Hello".into()),
            TestContentBlock::Unknown("U1".into()),
            TestContentBlock::Text("World".into()),
            TestContentBlock::Thought("T2".into()),
        ];
        assert_roundtrip(original, "mixed thoughts and unknowns");
    }

    #[test]
    fn test_insert_index_roundtrip_all_extra_content() {
        // Original: [Thought, Unknown, Thought]
        let original = vec![
            TestContentBlock::Thought("T1".into()),
            TestContentBlock::Unknown("U1".into()),
            TestContentBlock::Thought("T2".into()),
        ];
        assert_roundtrip(original, "all extra content, no main content");
    }

    #[test]
    fn test_insert_index_roundtrip_only_main_content() {
        // Original: [Text, Text]
        let original = vec![
            TestContentBlock::Text("Hello".into()),
            TestContentBlock::Text("World".into()),
        ];
        assert_roundtrip(original, "only main content");
    }

    #[test]
    fn test_insert_index_roundtrip_alternating() {
        // Original: [Text, Thought, Text, Thought, Text]
        let original = vec![
            TestContentBlock::Text("A".into()),
            TestContentBlock::Thought("T1".into()),
            TestContentBlock::Text("B".into()),
            TestContentBlock::Thought("T2".into()),
            TestContentBlock::Text("C".into()),
        ];
        assert_roundtrip(original, "alternating text and thoughts");
    }

    #[test]
    fn test_insert_index_roundtrip_complex_sequence() {
        // Original: [Thought, Text, Thought, Unknown, Text, Thought, Text, Unknown, Thought]
        let original = vec![
            TestContentBlock::Thought("T1".into()),
            TestContentBlock::Text("A".into()),
            TestContentBlock::Thought("T2".into()),
            TestContentBlock::Unknown("U1".into()),
            TestContentBlock::Text("B".into()),
            TestContentBlock::Thought("T3".into()),
            TestContentBlock::Text("C".into()),
            TestContentBlock::Unknown("U2".into()),
            TestContentBlock::Thought("T4".into()),
        ];
        assert_roundtrip(original, "complex mixed sequence");
    }

    #[test]
    fn test_insert_index_roundtrip_empty() {
        // Original: []
        let original: Vec<TestContentBlock> = vec![];
        assert_roundtrip(original, "empty array");
    }

    #[test]
    fn test_insert_index_roundtrip_single_thought() {
        // Original: [Thought]
        let original = vec![TestContentBlock::Thought("T1".into())];
        assert_roundtrip(original, "single thought");
    }

    #[test]
    fn test_insert_index_roundtrip_single_text() {
        // Original: [Text]
        let original = vec![TestContentBlock::Text("Hello".into())];
        assert_roundtrip(original, "single text");
    }

    #[test]
    fn test_insert_index_roundtrip_many_thoughts_between_texts() {
        // Original: [Text, Thought, Thought, Thought, Text]
        let original = vec![
            TestContentBlock::Text("Start".into()),
            TestContentBlock::Thought("T1".into()),
            TestContentBlock::Thought("T2".into()),
            TestContentBlock::Thought("T3".into()),
            TestContentBlock::Text("End".into()),
        ];
        assert_roundtrip(original, "many thoughts between two texts");
    }

    // ==========================================================================
    // Edge Case Tests for insert_index
    // ==========================================================================

    #[test]
    fn test_insert_index_out_of_bounds_clamped() {
        // When insert_index > len, it gets clamped to len (append)
        let messages = vec![OpenAICompatibleMessage::Assistant(
            OpenAICompatibleAssistantMessage {
                content: Some(Value::String("Hello".to_string())),
                tool_calls: None,
                tensorzero_extra_content_experimental: Some(vec![
                    InputExtraContentBlock::Thought {
                        insert_index: Some(999), // Way out of bounds
                        thought: Thought {
                            text: Some("Out of bounds thought".to_string()),
                            signature: None,
                            summary: None,
                            provider_type: None,
                            extra_data: None,
                        },
                    },
                ]),
            },
        )];
        let input = openai_messages_to_input(messages).unwrap();
        let content = &input.messages[0].content;

        // Should be clamped to position 1 (after the text)
        assert_eq!(content.len(), 2, "Should have 2 content blocks");
        assert!(
            matches!(&content[0], InputMessageContent::Text(_)),
            "Text should be first"
        );
        assert!(
            matches!(&content[1], InputMessageContent::Thought(_)),
            "Thought should be second (clamped to end)"
        );
    }

    #[test]
    fn test_insert_index_empty_main_content() {
        // insert_index with no main content
        let messages = vec![OpenAICompatibleMessage::Assistant(
            OpenAICompatibleAssistantMessage {
                content: None,
                tool_calls: None,
                tensorzero_extra_content_experimental: Some(vec![
                    InputExtraContentBlock::Thought {
                        insert_index: Some(0),
                        thought: Thought {
                            text: Some("First".to_string()),
                            signature: None,
                            summary: None,
                            provider_type: None,
                            extra_data: None,
                        },
                    },
                    InputExtraContentBlock::Thought {
                        insert_index: Some(1),
                        thought: Thought {
                            text: Some("Second".to_string()),
                            signature: None,
                            summary: None,
                            provider_type: None,
                            extra_data: None,
                        },
                    },
                ]),
            },
        )];
        let input = openai_messages_to_input(messages).unwrap();
        let content = &input.messages[0].content;

        assert_eq!(content.len(), 2, "Should have 2 thoughts");
        match (&content[0], &content[1]) {
            (InputMessageContent::Thought(t1), InputMessageContent::Thought(t2)) => {
                assert_eq!(
                    t1.text,
                    Some("First".to_string()),
                    "First thought should be at position 0"
                );
                assert_eq!(
                    t2.text,
                    Some("Second".to_string()),
                    "Second thought should be at position 1"
                );
            }
            _ => panic!("Expected two Thought content blocks"),
        }
    }

    #[test]
    fn test_insert_index_duplicate_indices() {
        // Multiple items with the same insert_index - should be inserted in input order
        let messages = vec![OpenAICompatibleMessage::Assistant(
            OpenAICompatibleAssistantMessage {
                content: Some(Value::String("Hello".to_string())),
                tool_calls: None,
                tensorzero_extra_content_experimental: Some(vec![
                    InputExtraContentBlock::Thought {
                        insert_index: Some(0),
                        thought: Thought {
                            text: Some("First at 0".to_string()),
                            signature: None,
                            summary: None,
                            provider_type: None,
                            extra_data: None,
                        },
                    },
                    InputExtraContentBlock::Unknown {
                        insert_index: Some(0),
                        unknown: Unknown {
                            data: json!({"id": "second at 0"}),
                            model_name: None,
                            provider_name: None,
                        },
                    },
                ]),
            },
        )];
        let input = openai_messages_to_input(messages).unwrap();
        let content = &input.messages[0].content;

        // With duplicate indices, they're inserted in input order at the same position
        // First insert at 0: [Thought, "Hello"]
        // Second insert at 0: [Unknown, Thought, "Hello"]
        assert_eq!(content.len(), 3, "Should have 3 content blocks");
        assert!(
            matches!(&content[0], InputMessageContent::Unknown(_)),
            "Unknown should be first (inserted second at index 0)"
        );
        assert!(
            matches!(&content[1], InputMessageContent::Thought(_)),
            "Thought should be second (inserted first at index 0)"
        );
        assert!(
            matches!(&content[2], InputMessageContent::Text(_)),
            "Text should be third"
        );
    }

    #[test]
    fn test_insert_index_only_unindexed() {
        // All items have insert_index: None - should be prepended in order
        let messages = vec![OpenAICompatibleMessage::Assistant(
            OpenAICompatibleAssistantMessage {
                content: Some(Value::String("Hello".to_string())),
                tool_calls: None,
                tensorzero_extra_content_experimental: Some(vec![
                    InputExtraContentBlock::Thought {
                        insert_index: None,
                        thought: Thought {
                            text: Some("Unindexed 1".to_string()),
                            signature: None,
                            summary: None,
                            provider_type: None,
                            extra_data: None,
                        },
                    },
                    InputExtraContentBlock::Unknown {
                        insert_index: None,
                        unknown: Unknown {
                            data: json!({"id": "unindexed 2"}),
                            model_name: None,
                            provider_name: None,
                        },
                    },
                ]),
            },
        )];
        let input = openai_messages_to_input(messages).unwrap();
        let content = &input.messages[0].content;

        assert_eq!(content.len(), 3, "Should have 3 content blocks");
        assert!(
            matches!(&content[0], InputMessageContent::Thought(_)),
            "Unindexed Thought should be first (prepended)"
        );
        assert!(
            matches!(&content[1], InputMessageContent::Unknown(_)),
            "Unindexed Unknown should be second (prepended)"
        );
        assert!(
            matches!(&content[2], InputMessageContent::Text(_)),
            "Text should be last"
        );
    }

    #[test]
    fn test_insert_index_with_tool_calls() {
        // Tool calls present before extra_content insertion
        let messages = vec![OpenAICompatibleMessage::Assistant(
            OpenAICompatibleAssistantMessage {
                content: Some(Value::String("Hello".to_string())),
                tool_calls: Some(vec![OpenAICompatibleToolCall {
                    id: "call_1".to_string(),
                    r#type: "function".to_string(),
                    function: OpenAICompatibleFunctionCall {
                        name: "test_tool".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tensorzero_extra_content_experimental: Some(vec![
                    InputExtraContentBlock::Thought {
                        insert_index: Some(0),
                        thought: Thought {
                            text: Some("Thought at 0".to_string()),
                            signature: None,
                            summary: None,
                            provider_type: None,
                            extra_data: None,
                        },
                    },
                ]),
            },
        )];
        let input = openai_messages_to_input(messages).unwrap();
        let content = &input.messages[0].content;

        // Content order: text first, then tool_call, then extra content inserted
        // Before extra: [Text, ToolCall]
        // After insert at 0: [Thought, Text, ToolCall]
        assert_eq!(content.len(), 3, "Should have 3 content blocks");
        assert!(
            matches!(&content[0], InputMessageContent::Thought(_)),
            "Thought should be first (inserted at 0)"
        );
        assert!(
            matches!(&content[1], InputMessageContent::Text(_)),
            "Text should be second"
        );
        assert!(
            matches!(&content[2], InputMessageContent::ToolCall(_)),
            "ToolCall should be third"
        );
    }
}
