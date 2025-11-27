//! Chat completion types and conversion logic for OpenAI-compatible API.
//!
//! This module contains all request/response types for the chat completion endpoint,
//! including message structures, parameter types, and conversion logic between
//! OpenAI-compatible formats and TensorZero's internal representations.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::config::UninitializedVariantInfo;
use crate::endpoints::inference::{
    ChatCompletionInferenceParams, InferenceCredentials, InferenceParams, InferenceResponse, Params,
};
use crate::endpoints::openai_compatible::types::input_files::{
    convert_file_to_base64, convert_image_url_to_file, convert_input_audio_to_file,
    OpenAICompatibleFile, OpenAICompatibleImageUrl, OpenAICompatibleInputAudio,
};
use crate::endpoints::openai_compatible::types::tool::{
    ChatCompletionToolChoiceOption, OpenAICompatibleTool, OpenAICompatibleToolCall,
    OpenAICompatibleToolChoiceParams, OpenAICompatibleToolMessage,
};
use crate::endpoints::openai_compatible::types::usage::OpenAICompatibleUsage;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::chat_completion_inference_params::ServiceTier;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::{
    current_timestamp, Arguments, ContentBlockChatOutput, FinishReason, Input, InputMessage,
    InputMessageContent, RawText, Role, System, Template, Text,
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
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
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
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum OpenAICompatibleResponseFormat {
    Text,
    JsonSchema { json_schema: JsonSchemaInfo },
    JsonObject,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
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
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, Value>,
}

// ============================================================================
// Response Types
// ============================================================================

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAICompatibleResponseMessage {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAICompatibleToolCall>>,
    pub role: String,
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
}

// ============================================================================
// Content Block Types
// ============================================================================

#[derive(Deserialize, Debug)]
#[serde(tag = "type", deny_unknown_fields, rename_all = "snake_case")]
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
                    _ => return Err(serde::de::Error::custom(
                        "`text` must be a string when using `\"type\": \"text\"`",
                    )),
                },
            }),
            (None, Some(arguments)) => Ok(TextContent::TensorZeroArguments {
                tensorzero_arguments: match arguments {
                    Value::Object(arguments) => Arguments(arguments),
                    _ => return Err(serde::de::Error::custom(
                        "`tensorzero::arguments` must be an object when using `\"type\": \"text\"`",
                    )),
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
            // OpenAI compatible endpoint does not support 'include_original_response'
            include_original_response: false,
            extra_body: openai_compatible_params.tensorzero_extra_body,
            extra_headers: openai_compatible_params.tensorzero_extra_headers,
            internal_dynamic_variant_config: openai_compatible_params
                .tensorzero_internal_dynamic_variant_config,
        })
    }
}

impl TryFrom<Vec<OpenAICompatibleMessage>> for Input {
    type Error = Error;
    fn try_from(
        openai_compatible_messages: Vec<OpenAICompatibleMessage>,
    ) -> Result<Self, Self::Error> {
        let mut system_message = None;
        let mut messages = Vec::new();
        let mut tool_call_id_to_name = HashMap::new();
        let first_system = matches!(
            openai_compatible_messages.first(),
            Some(OpenAICompatibleMessage::System(_))
        );
        for message in openai_compatible_messages {
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
                                        message: "System message cannot contain template with other content".to_string(),
                                    }.into());
                                }
                                system_message = Some(System::Template(t.arguments));
                                None
                            }
                            _ => {
                                return Err(ErrorDetails::InvalidOpenAICompatibleRequest {
                                    message: "System message must contain only text or template content blocks".to_string(),
                                }.into())
                            }
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
                        tracing::warn!("Multiple system messages provided. They will be concatenated and moved to the start of the conversation.");
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

        Ok(Input {
            system: system_message,
            messages,
        })
    }
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
                    service_tier: None,
                    system_fingerprint: String::new(),
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
                    },
                }],
                created: current_timestamp() as u32,
                model: format!("{response_model_prefix}{}", response.variant_name),
                system_fingerprint: String::new(),
                service_tier: None,
                object: "chat.completion".to_string(),
                usage: OpenAICompatibleUsage {
                    prompt_tokens: response.usage.input_tokens,
                    completion_tokens: response.usage.output_tokens,
                    total_tokens: response.usage.total_tokens(),
                },
                episode_id: response.episode_id.to_string(),
            },
        }
    }
}

/// Takes a vector of ContentBlockOutput and returns a tuple of (Option<String>, Vec<OpenAICompatibleToolCall>).
/// This is useful since the OpenAI format separates text and tool calls in the response fields.
pub fn process_chat_content(
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
            ContentBlockChatOutput::Unknown(_) => {
                tracing::warn!(
                    "Ignoring 'unknown' content block when constructing OpenAI-compatible response"
                );
            }
        }
    }
    (content_str, tool_calls)
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
    fn test_try_from_openai_compatible_messages() {
        let messages = vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
            content: Value::String("Hello, world!".to_string()),
        })];
        let input: Input = messages.try_into().unwrap();
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
        let input: Input = messages.try_into().unwrap();
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
        let input: Result<Input, Error> = messages.try_into();
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
        let input: Input = messages.try_into().unwrap();
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
            },
        )];
        let input: Input = messages.try_into().unwrap();
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
            },
        )];
        let input: Input = messages.try_into().unwrap();
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
            }),
            OpenAICompatibleMessage::System(OpenAICompatibleSystemMessage {
                content: Value::String("System message".to_string()),
            }),
        ];
        let result: Input = out_of_order_messages.try_into().unwrap();
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
        let input: Input = messages.try_into().unwrap();
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
        let input: Input = messages.try_into().unwrap();
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
        let result: Result<Input, Error> = messages.try_into();
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
        let result: Result<Input, Error> = messages.try_into();
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
        let result: Result<Input, Error> = messages.try_into();
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
        let result: Result<Input, Error> = messages.try_into();
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
        let input: Input = messages.try_into().unwrap();
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
        let input: Input = messages.try_into().unwrap();
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
        assert_eq!(err.to_string(), "Invalid request to OpenAI-compatible endpoint: Invalid content block: Either `text` or `tensorzero::arguments` must be set when using `\"type\": \"text\"`");
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
}
