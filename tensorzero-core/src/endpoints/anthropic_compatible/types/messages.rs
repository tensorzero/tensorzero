//! Message types and conversion logic for Anthropic-compatible API.
//!
//! This module provides type definitions and conversion functions for translating
//! between Anthropic's Messages API format and TensorZero's internal representations.
//!
//! # Key Types
//!
//! - [`AnthropicMessagesParams`]: Request parameters matching Anthropic's API
//! - [`AnthropicMessageResponse`]: Response format matching Anthropic's API
//! - [`AnthropicContentBlock`]: Content blocks for requests (Text, ToolUse, ToolResult)
//!
//! # Conversions
//!
//! - [`Params::try_from_anthropic()`]: Converts Anthropic params to TensorZero Params
//! - [`AnthropicMessageResponse::from()`]: Converts TensorZero response to Anthropic format
//! - [`finish_reason_to_anthropic()`]: Maps TensorZero finish reasons to Anthropic's format
//!
//! # Example
//!
//! ```rust
//! use tensorzero_core::endpoints::anthropic_compatible::types::messages::AnthropicMessagesParams;
//! use serde_json::json;
//!
//! // Basic request
//! let params = AnthropicMessagesParams {
//!     model: "tensorzero::function_name::my_function".to_string(),
//!     max_tokens: 100,
//!     messages: vec![
//!         AnthropicMessage::User(AnthropicUserMessage {
//!             content: serde_json::Value::String("Hello, world!".to_string()),
//!         }),
//!     ],
//!     ..Default::default()
//! };
//!
//! // With system prompt
//! let params_with_system = AnthropicMessagesParams {
//!     model: "tensorzero::function_name::my_function".to_string(),
//!     max_tokens: 100,
//!     system: Some(json!("You are a helpful assistant")),
//!     messages: vec![...],
//!     ..Default::default()
//! };
//!
//! // With tools
//! let params_with_tools = AnthropicMessagesParams {
//!     model: "tensorzero::function_name::my_function".to_string(),
//!     max_tokens: 1000,
//!     messages: vec![...],
//!     tools: Some(vec![AnthropicTool {
//!         name: "get_weather".to_string(),
//!         description: "Get current weather".to_string(),
//!         input_schema: AnthropicInputSchema {
//!             schema_type: "object".to_string(),
//!             properties: Some(HashMap::from_iter([
//!                 ("location".to_string(), json!({
//!                     "type": "string",
//!                     "description": "City name"
//!                 })),
//!             ])),
//!             required: Some(vec!["location".to_string()]),
//!             additional_properties: Some(false),
//!         },
//!     }]),
//!     tool_choice: Some(AnthropicToolChoice::Auto),
//!     ..Default::default()
//! };
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tensorzero_derive::TensorZeroDeserialize;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::config::UninitializedVariantInfo;
use crate::endpoints::anthropic_compatible::types::tool::{
    AnthropicTool, AnthropicToolChoice, AnthropicToolChoiceParams,
};
use crate::endpoints::anthropic_compatible::types::usage::AnthropicUsage;
use crate::endpoints::inference::{
    ChatCompletionInferenceParams, InferenceCredentials, InferenceParams, InferenceResponse, Params,
};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::usage::{RawResponseEntry, RawUsageEntry};
use crate::inference::types::{
    ContentBlockChatOutput, FinishReason, Input, InputMessage, InputMessageContent, RawText, Role,
    System, Template, Text,
};
use crate::tool::{DynamicToolParams, ProviderTool, ToolResult};

// ============================================================================
// Message Types
// ============================================================================

/// Anthropic message (user or assistant - system is a separate field)
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum AnthropicMessage {
    User(AnthropicUserMessage),
    Assistant(AnthropicAssistantMessage),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicUserMessage {
    pub content: AnthropicMessageContent,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicAssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<AnthropicMessageContent>,
}

/// Content can be a string or an array of content blocks
pub type AnthropicMessageContent = Value;

/// Content block types for requests
#[derive(Clone, Debug, PartialEq, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    #[serde(rename = "tensorzero::raw_text")]
    RawText(RawText),
    #[serde(rename = "tensorzero::template")]
    Template(Template),
}

// ============================================================================
// Request Parameter Types
// ============================================================================

#[derive(Clone, Debug, Default, Deserialize)]
pub struct AnthropicMessagesParams {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<Value>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    // TensorZero-specific parameters
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
    #[serde(default, rename = "tensorzero::include_raw_response")]
    pub tensorzero_include_raw_response: bool,
}

// ============================================================================
// Response Types
// ============================================================================

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnthropicMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub content: Vec<AnthropicOutputContentBlock>,
    pub model: String,
    pub stop_reason: AnthropicStopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_usage: Option<Vec<RawUsageEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_response: Option<Vec<RawResponseEntry>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum AnthropicOutputContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

impl From<FinishReason> for AnthropicStopReason {
    fn from(finish_reason: FinishReason) -> Self {
        match finish_reason {
            FinishReason::Stop => AnthropicStopReason::EndTurn,
            FinishReason::StopSequence => AnthropicStopReason::StopSequence,
            FinishReason::Length => AnthropicStopReason::MaxTokens,
            FinishReason::ToolCall => AnthropicStopReason::ToolUse,
            FinishReason::ContentFilter => AnthropicStopReason::EndTurn, // Coerce to end_turn
            FinishReason::Unknown => AnthropicStopReason::EndTurn,
        }
    }
}

// ============================================================================
// Conversion Implementations
// ============================================================================

const TENSORZERO_FUNCTION_NAME_PREFIX: &str = "tensorzero::function_name::";
const TENSORZERO_MODEL_NAME_PREFIX: &str = "tensorzero::model_name::";
const ANTHROPIC_MESSAGE_TYPE: &str = "message";
const ANTHROPIC_ROLE_ASSISTANT: &str = "assistant";

impl Params {
    pub fn try_from_anthropic(anthropic_params: AnthropicMessagesParams) -> Result<Self, Error> {
        let (function_name, model_name) = if let Some(function_name) = anthropic_params
            .model
            .strip_prefix(TENSORZERO_FUNCTION_NAME_PREFIX)
        {
            (Some(function_name.to_string()), None)
        } else if let Some(model_name) = anthropic_params
            .model
            .strip_prefix(TENSORZERO_MODEL_NAME_PREFIX)
        {
            (None, Some(model_name.to_string()))
        } else {
            return Err(Error::new(ErrorDetails::InvalidAnthropicCompatibleRequest {
                message: "`model` field must start with `tensorzero::function_name::` or `tensorzero::model_name::`. For example, `tensorzero::function_name::my_function` for a function `my_function` defined in your config, or `tensorzero::model_name::my_model` for a model `my_model` defined in your config.".to_string(),
            }));
        };

        if let Some(function_name) = &function_name
            && function_name.is_empty()
        {
            return Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
                message: "function_name (passed in model field after \"tensorzero::function_name::\") cannot be empty".to_string(),
            }.into());
        }

        if let Some(model_name) = &model_name
            && model_name.is_empty()
        {
            return Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
                message: "model_name (passed in model field after \"tensorzero::model_name::\") cannot be empty".to_string(),
            }.into());
        }

        let input =
            anthropic_messages_to_input(anthropic_params.system, anthropic_params.messages)?;

        let mut inference_params = anthropic_params.tensorzero_params.unwrap_or_default();

        inference_params.chat_completion = ChatCompletionInferenceParams {
            frequency_penalty: inference_params.chat_completion.frequency_penalty,
            json_mode: inference_params.chat_completion.json_mode,
            max_tokens: inference_params
                .chat_completion
                .max_tokens
                .or(Some(anthropic_params.max_tokens)),
            presence_penalty: inference_params.chat_completion.presence_penalty,
            reasoning_effort: inference_params.chat_completion.reasoning_effort,
            service_tier: inference_params.chat_completion.service_tier,
            seed: inference_params.chat_completion.seed,
            stop_sequences: inference_params
                .chat_completion
                .stop_sequences
                .or(anthropic_params.stop_sequences),
            temperature: inference_params
                .chat_completion
                .temperature
                .or(anthropic_params.temperature),
            thinking_budget_tokens: inference_params.chat_completion.thinking_budget_tokens,
            top_p: inference_params
                .chat_completion
                .top_p
                .or(anthropic_params.top_p),
            verbosity: inference_params.chat_completion.verbosity.clone(),
        };

        let AnthropicToolChoiceParams {
            allowed_tools,
            tool_choice,
        } = anthropic_params
            .tool_choice
            .map(|tc| tc.into_tool_params())
            .unwrap_or_default();

        let dynamic_tool_params = DynamicToolParams {
            allowed_tools,
            additional_tools: anthropic_params
                .tools
                .map(|tools| tools.into_iter().map(|t| t.into()).collect()),
            tool_choice,
            parallel_tool_calls: Some(true), // Anthropic supports parallel tool calls
            provider_tools: anthropic_params.tensorzero_provider_tools,
        };

        Ok(Params {
            function_name,
            model_name,
            episode_id: anthropic_params.tensorzero_episode_id,
            input,
            stream: anthropic_params.stream,
            params: inference_params,
            variant_name: anthropic_params.tensorzero_variant_name,
            dryrun: anthropic_params.tensorzero_dryrun,
            dynamic_tool_params,
            output_schema: None, // Anthropic doesn't have a direct response_format equivalent
            credentials: anthropic_params.tensorzero_credentials,
            cache_options: anthropic_params
                .tensorzero_cache_options
                .unwrap_or_default(),
            internal: false,
            tags: anthropic_params.tensorzero_tags,
            include_original_response: false, // Deprecated
            include_raw_response: anthropic_params.tensorzero_include_raw_response,
            include_raw_usage: anthropic_params.tensorzero_include_raw_usage,
            extra_body: anthropic_params.tensorzero_extra_body,
            extra_headers: anthropic_params.tensorzero_extra_headers,
            internal_dynamic_variant_config: anthropic_params
                .tensorzero_internal_dynamic_variant_config,
        })
    }
}

/// Convert Anthropic messages to TensorZero Input format.
///
/// This function handles the translation from Anthropic's message format to TensorZero's internal format,
/// including merging consecutive tool result messages (for parallel tool calls).
///
/// # Arguments
/// * `system` - Optional system prompt (string or array of blocks)
/// * `messages` - Vector of Anthropic messages (User and Assistant)
///
/// # Returns
/// A TensorZero `Input` containing the converted messages and system prompt
///
/// # Tool Result Merging
/// When multiple consecutive user messages contain tool results, they are merged into a single
/// user message containing all tool results. This handles Anthropic's format where parallel
/// tool calls are represented as consecutive tool result messages.
///
/// # Errors
/// Returns an error if:
/// - The system prompt is invalid (not a string or array)
/// - Any message content is invalid
pub fn anthropic_messages_to_input(
    system: Option<Value>,
    messages: Vec<AnthropicMessage>,
) -> Result<Input, Error> {
    // Convert system prompt (can be string or array of blocks)
    let system_message = if let Some(system) = system {
        Some(convert_system_prompt(system)?)
    } else {
        None
    };

    let mut converted_messages = Vec::new();

    for message in messages {
        match message {
            AnthropicMessage::User(msg) => {
                let content = convert_anthropic_message_content(msg.content)?;
                converted_messages.push(InputMessage {
                    role: Role::User,
                    content,
                });
            }
            AnthropicMessage::Assistant(msg) => {
                let mut message_content = Vec::new();
                if let Some(content) = msg.content {
                    message_content.extend(convert_anthropic_message_content(content)?);
                }
                converted_messages.push(InputMessage {
                    role: Role::Assistant,
                    content: message_content,
                });
            }
        }
    }

    // Merge consecutive tool results (similar to OpenAI endpoint)
    // This ensures that parallel tool call results can be passed through properly
    let mut final_messages = Vec::new();
    let mut i = 0;
    while i < converted_messages.len() {
        let message = &converted_messages[i];
        if message.role == Role::User {
            // Check if this is a tool result message
            let has_tool_result = message
                .content
                .iter()
                .any(|c| matches!(c, InputMessageContent::ToolResult(_)));

            if has_tool_result {
                // Collect all consecutive tool results
                let mut tool_results = Vec::new();
                while i < converted_messages.len()
                    && converted_messages[i].role == Role::User
                    && converted_messages[i]
                        .content
                        .iter()
                        .any(|c| matches!(c, InputMessageContent::ToolResult(_)))
                {
                    for content in &converted_messages[i].content {
                        if let InputMessageContent::ToolResult(_) = content {
                            tool_results.push(content.clone());
                        }
                    }
                    i += 1;
                }
                // Add any text content from the next user message
                if i < converted_messages.len() && converted_messages[i].role == Role::User {
                    for content in &converted_messages[i].content {
                        if let InputMessageContent::Text(_) = content {
                            tool_results.push(content.clone());
                            break;
                        }
                    }
                }
                final_messages.push(InputMessage {
                    role: Role::User,
                    content: tool_results,
                });
                continue;
            }
        }
        final_messages.push(message.clone());
        i += 1;
    }

    Ok(Input {
        system: system_message,
        messages: final_messages,
    })
}

/// Convert system prompt from Anthropic format to TensorZero format.
///
/// # Arguments
/// * `system` - Either a string or an array of content blocks (e.g., for cache_control)
///
/// # Returns
/// A `System` instance containing the converted system prompt
///
/// # Errors
/// Returns an error if the system value is not a string or array
fn convert_system_prompt(system: Value) -> Result<System, Error> {
    match system {
        Value::String(s) => Ok(System::Text(s)),
        Value::Array(blocks) => {
            // System as array of blocks (e.g., for cache_control)
            // Concatenate text blocks with newline separators
            let text_blocks: Vec<&str> = blocks
                .iter()
                .filter_map(|block| {
                    block
                        .as_object()
                        .and_then(|obj| obj.get("type"))
                        .and_then(|t| t.as_str())
                        .filter(|&t| t == "text")
                        .and_then(|_| {
                            block
                                .as_object()
                                .and_then(|obj| obj.get("text"))
                                .and_then(|text| text.as_str())
                        })
                })
                .collect();

            Ok(System::Text(text_blocks.join("\n")))
        }
        _ => Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
            message: "system must be a string or array of blocks".to_string(),
        }
        .into()),
    }
}

/// Convert Anthropic message content to TensorZero format.
///
/// Handles both simple string content and structured content blocks.
///
/// # Arguments
/// * `content` - JSON value containing either:
///   - A string (simple text message)
///   - An array of content blocks (Text, ToolUse, ToolResult, etc.)
///
/// # Returns
/// Vector of TensorZero input message contents
///
/// # Errors
/// Returns an error if:
/// - The content is not a string or array
/// - Any content block is malformed or invalid
///
/// # Supported Conversions
/// - `string` → `InputMessageContent::Text`
/// - `{"type": "text", "text": "..."}` → `InputMessageContent::Text`
/// - `{"type": "tool_use", ...}` → `InputMessageContent::ToolCall`
/// - `{"type": "tool_result", ...}` → `InputMessageContent::ToolResult`
/// - `{"type": "raw_text", ...}` → `InputMessageContent::RawText`
/// - `{"type": "template", ...}` → `InputMessageContent::Template`
fn convert_anthropic_message_content(content: Value) -> Result<Vec<InputMessageContent>, Error> {
    match content {
        Value::String(s) => Ok(vec![InputMessageContent::Text(Text { text: s })]),
        Value::Array(blocks) => {
            blocks
                .into_iter()
                .map(|block| {
                    let content_block =
                        serde_json::from_value::<AnthropicContentBlock>(block.clone());

                    match content_block {
                        Ok(AnthropicContentBlock::Text { text }) => {
                            Ok(InputMessageContent::Text(Text { text }))
                        }
                        Ok(AnthropicContentBlock::ToolUse { id, name, input }) => {
                            Ok(InputMessageContent::ToolCall(
                                crate::tool::ToolCallWrapper::InferenceResponseToolCall(
                                    crate::tool::InferenceResponseToolCall {
                                        id,
                                        raw_name: name.clone(),
                                        raw_arguments: serde_json::to_string(&input)
                                            .unwrap_or_default(),
                                        name: None,
                                        arguments: None,
                                    },
                                ),
                            ))
                        }
                        Ok(AnthropicContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        }) => Ok(InputMessageContent::ToolResult(ToolResult {
                            id: tool_use_id,
                            name: String::new(), // Will be filled in from tool_use_id mapping
                            result: content,
                        })),
                        Ok(AnthropicContentBlock::RawText(raw_text)) => {
                            Ok(InputMessageContent::RawText(raw_text))
                        }
                        Ok(AnthropicContentBlock::Template(t)) => {
                            Ok(InputMessageContent::Template(t))
                        }
                        Err(e) => Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
                            message: format!("Invalid content block: {e}"),
                        }
                        .into()),
                    }
                })
                .collect()
        }
        _ => Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
            message: "message content must be a string or array of content blocks".to_string(),
        }
        .into()),
    }
}

impl From<(InferenceResponse, String)> for AnthropicMessageResponse {
    fn from((inference_response, response_model_prefix): (InferenceResponse, String)) -> Self {
        match inference_response {
            InferenceResponse::Chat(response) => {
                let content = process_chat_content(response.content);
                AnthropicMessageResponse {
                    id: response.inference_id.to_string(),
                    message_type: ANTHROPIC_MESSAGE_TYPE.to_string(),
                    role: ANTHROPIC_ROLE_ASSISTANT.to_string(),
                    content,
                    model: format!("{response_model_prefix}{}", response.variant_name),
                    stop_reason: response.finish_reason.unwrap_or(FinishReason::Stop).into(),
                    stop_sequence: None,
                    usage: AnthropicUsage {
                        input_tokens: response.usage.input_tokens.unwrap_or(0),
                        output_tokens: response.usage.output_tokens.unwrap_or(0),
                    },
                    tensorzero_raw_usage: response.raw_usage,
                    tensorzero_raw_response: response.raw_response,
                }
            }
            InferenceResponse::Json(response) => AnthropicMessageResponse {
                id: response.inference_id.to_string(),
                message_type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![AnthropicOutputContentBlock::Text {
                    text: response.output.raw.unwrap_or_default(),
                }],
                model: format!("{response_model_prefix}{}", response.variant_name),
                stop_reason: response.finish_reason.unwrap_or(FinishReason::Stop).into(),
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: response.usage.input_tokens.unwrap_or(0),
                    output_tokens: response.usage.output_tokens.unwrap_or(0),
                },
                tensorzero_raw_usage: response.raw_usage,
                tensorzero_raw_response: response.raw_response,
            },
        }
    }
}

/// Process chat content blocks and convert to Anthropic-compatible format.
///
/// Filters out unsupported block types (Thought, Unknown) with warnings,
/// and converts Text and ToolCall blocks to Anthropic format.
///
/// # Arguments
/// * `content` - Vector of content blocks from TensorZero chat response
///
/// # Returns
/// Vector of Anthropic-compatible content blocks (Text and ToolUse only)
///
/// # Filtering
/// - `Text` blocks → `AnthropicOutputContentBlock::Text`
/// - `ToolCall` blocks → `AnthropicOutputContentBlock::ToolUse`
/// - `Thought` blocks → Logged and filtered out (not supported by Anthropic)
/// - `Unknown` blocks → Logged and filtered out
fn process_chat_content(content: Vec<ContentBlockChatOutput>) -> Vec<AnthropicOutputContentBlock> {
    content
        .into_iter()
        .filter_map(|block| match block {
            ContentBlockChatOutput::Text(text) => {
                Some(AnthropicOutputContentBlock::Text { text: text.text })
            }
            ContentBlockChatOutput::ToolCall(tool_call) => {
                Some(AnthropicOutputContentBlock::ToolUse {
                    id: tool_call.id,
                    name: tool_call.raw_name,
                    input: serde_json::from_str(&tool_call.raw_arguments).unwrap_or_default(),
                })
            }
            ContentBlockChatOutput::Thought(_) => {
                tracing::warn!(
                    "Ignoring 'thought' content block when constructing Anthropic-compatible response"
                );
                None
            }
            ContentBlockChatOutput::Unknown(_) => {
                tracing::warn!(
                    "Ignoring 'unknown' content block when constructing Anthropic-compatible response"
                );
                None
            }
        })
        .collect()
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert TensorZero FinishReason to Anthropic-compatible stop reason string
/// This is used by both streaming and non-streaming responses
pub fn finish_reason_to_anthropic(finish_reason: FinishReason) -> String {
    match finish_reason {
        FinishReason::Stop => "end_turn".to_string(),
        FinishReason::StopSequence => "stop_sequence".to_string(),
        FinishReason::Length => "max_tokens".to_string(),
        FinishReason::ToolCall => "tool_use".to_string(),
        FinishReason::ContentFilter => "end_turn".to_string(),
        FinishReason::Unknown => "end_turn".to_string(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_anthropic_messages_to_input_basic() {
        let messages = vec![AnthropicMessage::User(AnthropicUserMessage {
            content: Value::String("Hello, world!".to_string()),
        })];

        let input = anthropic_messages_to_input(None, messages).unwrap();
        assert_eq!(input.messages.len(), 1);
        assert_eq!(input.messages[0].role, Role::User);
        assert_eq!(
            input.messages[0].content[0],
            InputMessageContent::Text(Text {
                text: "Hello, world!".to_string(),
            })
        );
    }

    #[test]
    fn test_anthropic_messages_to_input_with_system() {
        let system = json!("You are a helpful assistant");
        let messages = vec![AnthropicMessage::User(AnthropicUserMessage {
            content: Value::String("Hello".to_string()),
        })];

        let input = anthropic_messages_to_input(Some(system), messages).unwrap();
        assert_eq!(
            input.system,
            Some(System::Text("You are a helpful assistant".to_string()))
        );
    }

    #[test]
    fn test_anthropic_messages_to_input_with_system_array() {
        let system = json!([
            {"type": "text", "text": "You are helpful"}
        ]);
        let messages = vec![AnthropicMessage::User(AnthropicUserMessage {
            content: Value::String("Hello".to_string()),
        })];

        let input = anthropic_messages_to_input(Some(system), messages).unwrap();
        assert_eq!(
            input.system,
            Some(System::Text("You are helpful".to_string()))
        );
    }

    #[test]
    fn test_anthropic_messages_to_input_tool_use() {
        let messages = vec![
            AnthropicMessage::User(AnthropicUserMessage {
                content: Value::String("What's the weather?".to_string()),
            }),
            AnthropicMessage::Assistant(AnthropicAssistantMessage {
                content: Some(Value::Array(vec![json!({
                    "type": "tool_use",
                    "id": "toolu_0123",
                    "name": "get_weather",
                    "input": {"location": "SF"}
                })])),
            }),
        ];

        let input = anthropic_messages_to_input(None, messages).unwrap();
        assert_eq!(input.messages.len(), 2);
        assert_eq!(input.messages[0].role, Role::User);
        assert_eq!(input.messages[1].role, Role::Assistant);
        assert!(matches!(
            input.messages[1].content[0],
            InputMessageContent::ToolCall(_)
        ));
    }

    #[test]
    fn test_anthropic_messages_to_input_with_tool_result() {
        let messages = vec![
            AnthropicMessage::User(AnthropicUserMessage {
                content: Value::String("What's the weather?".to_string()),
            }),
            AnthropicMessage::Assistant(AnthropicAssistantMessage {
                content: Some(Value::Array(vec![json!({
                    "type": "tool_use",
                    "id": "toolu_0123",
                    "name": "get_weather",
                    "input": {"location": "SF"}
                })])),
            }),
            AnthropicMessage::User(AnthropicUserMessage {
                content: Value::Array(vec![json!({
                    "type": "tool_result",
                    "tool_use_id": "toolu_0123",
                    "content": "68 degrees"
                })]),
            }),
        ];

        let input = anthropic_messages_to_input(None, messages).unwrap();
        // Should have 3 messages: user question, assistant tool use, user tool result
        assert_eq!(input.messages.len(), 3);
        assert_eq!(input.messages[0].role, Role::User);
        assert_eq!(input.messages[1].role, Role::Assistant);
        assert_eq!(input.messages[2].role, Role::User);
        // Last message should contain the tool result
        assert!(
            input.messages[2]
                .content
                .iter()
                .any(|c| matches!(c, InputMessageContent::ToolResult(_)))
        );
    }

    #[test]
    fn test_params_try_from_anthropic_basic() {
        let params = Params::try_from_anthropic(AnthropicMessagesParams {
            model: "tensorzero::function_name::test_function".to_string(),
            max_tokens: 100,
            messages: vec![AnthropicMessage::User(AnthropicUserMessage {
                content: Value::String("test".to_string()),
            })],
            ..Default::default()
        })
        .unwrap();

        assert_eq!(params.function_name, Some("test_function".to_string()));
        assert_eq!(params.params.chat_completion.max_tokens, Some(100));
    }

    #[test]
    fn test_params_try_from_anthropic_invalid_prefix() {
        let result = Params::try_from_anthropic(AnthropicMessagesParams {
            model: "gpt-4".to_string(),
            max_tokens: 100,
            messages: vec![],
            ..Default::default()
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_params_try_from_anthropic_empty_function_name() {
        let result = Params::try_from_anthropic(AnthropicMessagesParams {
            model: "tensorzero::function_name::".to_string(),
            max_tokens: 100,
            messages: vec![],
            ..Default::default()
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_params_try_from_anthropic_with_temperature() {
        let params = Params::try_from_anthropic(AnthropicMessagesParams {
            model: "tensorzero::model_name::test_model".to_string(),
            max_tokens: 100,
            temperature: Some(0.7),
            top_p: Some(0.9),
            messages: vec![AnthropicMessage::User(AnthropicUserMessage {
                content: Value::String("test".to_string()),
            })],
            ..Default::default()
        })
        .unwrap();

        assert_eq!(params.model_name, Some("test_model".to_string()));
        assert_eq!(params.params.chat_completion.temperature, Some(0.7));
        assert_eq!(params.params.chat_completion.top_p, Some(0.9));
    }

    #[test]
    fn test_params_try_from_anthropic_with_stop_sequences() {
        let params = Params::try_from_anthropic(AnthropicMessagesParams {
            model: "tensorzero::model_name::test_model".to_string(),
            max_tokens: 100,
            stop_sequences: Some(vec!["STOP".to_string(), "END".to_string()]),
            messages: vec![AnthropicMessage::User(AnthropicUserMessage {
                content: Value::String("test".to_string()),
            })],
            ..Default::default()
        })
        .unwrap();

        assert_eq!(
            params.params.chat_completion.stop_sequences,
            Some(vec!["STOP".to_string(), "END".to_string()])
        );
    }

    #[test]
    fn test_params_try_from_anthropic_tool_choice_auto() {
        let params = Params::try_from_anthropic(AnthropicMessagesParams {
            model: "tensorzero::function_name::test_function".to_string(),
            max_tokens: 100,
            tool_choice: Some(AnthropicToolChoice::Auto),
            messages: vec![],
            ..Default::default()
        })
        .unwrap();

        assert_eq!(
            params.dynamic_tool_params.tool_choice,
            Some(tensorzero_types::ToolChoice::Auto)
        );
    }

    #[test]
    fn test_params_try_from_anthropic_tool_choice_specific() {
        let params = Params::try_from_anthropic(AnthropicMessagesParams {
            model: "tensorzero::function_name::test_function".to_string(),
            max_tokens: 100,
            tool_choice: Some(AnthropicToolChoice::Tool {
                name: "my_tool".to_string(),
            }),
            messages: vec![],
            ..Default::default()
        })
        .unwrap();

        assert_eq!(
            params.dynamic_tool_params.tool_choice,
            Some(tensorzero_types::ToolChoice::Specific(
                "my_tool".to_string()
            ))
        );
        assert_eq!(
            params.dynamic_tool_params.allowed_tools,
            Some(vec!["my_tool".to_string()])
        );
    }

    #[test]
    fn test_response_conversion_chat() {
        use crate::endpoints::inference::ChatInferenceResponse;
        use crate::inference::types::Usage;
        use uuid::Uuid;

        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![],
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            },
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: Some(crate::inference::types::FinishReason::Stop),
        });

        let anthropic_response =
            AnthropicMessageResponse::from((response, "test_prefix::".to_string()));

        assert_eq!(anthropic_response.message_type, "message");
        assert_eq!(anthropic_response.role, "assistant");
        assert_eq!(anthropic_response.model, "test_prefix::test_variant");
        assert_eq!(anthropic_response.stop_reason, AnthropicStopReason::EndTurn);
        assert_eq!(anthropic_response.usage.input_tokens, 10);
        assert_eq!(anthropic_response.usage.output_tokens, 20);
    }

    #[test]
    fn test_response_conversion_with_text() {
        use crate::endpoints::inference::ChatInferenceResponse;
        use crate::inference::types::{Text, Usage};
        use uuid::Uuid;

        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![crate::inference::types::ContentBlockChatOutput::Text(
                Text {
                    text: "Hello, world!".to_string(),
                },
            )],
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            },
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: None,
        });

        let anthropic_response = AnthropicMessageResponse::from((response, "prefix::".to_string()));

        assert_eq!(anthropic_response.content.len(), 1);
        assert_eq!(
            anthropic_response.content[0],
            AnthropicOutputContentBlock::Text {
                text: "Hello, world!".to_string()
            }
        );
    }

    #[test]
    fn test_response_conversion_with_tool() {
        use crate::endpoints::inference::ChatInferenceResponse;
        use crate::inference::types::{ContentBlockChatOutput, Usage};
        use crate::tool::InferenceResponseToolCall;
        use uuid::Uuid;

        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();
        let response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id,
            episode_id,
            variant_name: "test_variant".to_string(),
            content: vec![ContentBlockChatOutput::ToolCall(
                InferenceResponseToolCall {
                    id: "tool_123".to_string(),
                    raw_name: "my_tool".to_string(),
                    raw_arguments: "{\"arg\": \"value\"}".to_string(),
                    name: None,
                    arguments: None,
                },
            )],
            usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            },
            raw_usage: None,
            original_response: None,
            raw_response: None,
            finish_reason: Some(crate::inference::types::FinishReason::ToolCall),
        });

        let anthropic_response = AnthropicMessageResponse::from((response, "prefix::".to_string()));

        assert_eq!(anthropic_response.content.len(), 1);
        assert_eq!(
            anthropic_response.content[0],
            AnthropicOutputContentBlock::ToolUse {
                id: "tool_123".to_string(),
                name: "my_tool".to_string(),
                input: serde_json::json!({"arg": "value"})
            }
        );
        assert_eq!(anthropic_response.stop_reason, AnthropicStopReason::ToolUse);
    }

    #[test]
    fn test_stop_reason_conversion() {
        assert_eq!(
            AnthropicStopReason::from(crate::inference::types::FinishReason::Stop),
            AnthropicStopReason::EndTurn
        );
        assert_eq!(
            AnthropicStopReason::from(crate::inference::types::FinishReason::Length),
            AnthropicStopReason::MaxTokens
        );
        assert_eq!(
            AnthropicStopReason::from(crate::inference::types::FinishReason::ToolCall),
            AnthropicStopReason::ToolUse
        );
        assert_eq!(
            AnthropicStopReason::from(crate::inference::types::FinishReason::StopSequence),
            AnthropicStopReason::StopSequence
        );
        assert_eq!(
            AnthropicStopReason::from(crate::inference::types::FinishReason::ContentFilter),
            AnthropicStopReason::EndTurn
        );
    }

    #[test]
    fn test_content_block_with_text() {
        let block = AnthropicContentBlock::Text {
            text: "Hello".to_string(),
        };
        assert_eq!(
            block,
            AnthropicContentBlock::Text {
                text: "Hello".to_string()
            }
        );
    }

    #[test]
    fn test_content_block_with_tool_use() {
        let block = AnthropicContentBlock::ToolUse {
            id: "123".to_string(),
            name: "my_tool".to_string(),
            input: json!({"arg": "value"}),
        };
        assert!(matches!(block, AnthropicContentBlock::ToolUse { .. }));
        if let AnthropicContentBlock::ToolUse { id, name, .. } = block {
            assert_eq!(id, "123");
            assert_eq!(name, "my_tool");
        }
    }

    #[test]
    fn test_content_block_with_tool_result() {
        let block = AnthropicContentBlock::ToolResult {
            tool_use_id: "123".to_string(),
            content: "result".to_string(),
            is_error: false,
        };
        assert!(matches!(block, AnthropicContentBlock::ToolResult { .. }));
        if let AnthropicContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } = block
        {
            assert_eq!(tool_use_id, "123");
            assert_eq!(content, "result");
            assert!(!is_error);
        }
    }

    #[test]
    fn test_content_block_serialization() {
        let json = json!({
            "type": "text",
            "text": "Hello"
        });
        let block: AnthropicContentBlock = serde_json::from_value(json.clone()).unwrap();
        assert!(matches!(block, AnthropicContentBlock::Text { .. }));

        let json = json!({
            "type": "tool_use",
            "id": "123",
            "name": "my_tool",
            "input": {"arg": "value"}
        });
        let block: AnthropicContentBlock = serde_json::from_value(json.clone()).unwrap();
        assert!(matches!(block, AnthropicContentBlock::ToolUse { .. }));

        let json = json!({
            "type": "tool_result",
            "tool_use_id": "123",
            "content": "result",
            "is_error": false
        });
        let block: AnthropicContentBlock = serde_json::from_value(json.clone()).unwrap();
        assert!(matches!(block, AnthropicContentBlock::ToolResult { .. }));
    }
}
