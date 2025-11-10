//! OpenAI-compatible API endpoint implementation.
//!
//! This module provides compatibility with the OpenAI Chat Completions API format,
//! translating between OpenAI's request/response format and our internal types.
//! It implements request handling, parameter conversion, and response formatting
//! to match OpenAI's API specification.
//!
//! We convert the request into our internal types, call `endpoints::inference::inference` to perform the actual inference,
//! and then convert the response into the OpenAI-compatible format.

use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use axum::{debug_handler, Extension};
use futures::Stream;
use mime::MediaType;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use tokio_stream::StreamExt;
use url::Url;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::config::UninitializedVariantInfo;
use crate::embeddings::{Embedding, EmbeddingInput};
use crate::endpoints::embeddings::Params as EmbeddingParams;
use crate::endpoints::inference::{
    inference, ChatCompletionInferenceParams, InferenceParams, Params,
};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::chat_completion_inference_params::ServiceTier;
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::file::Detail;
use crate::inference::types::{
    current_timestamp, Arguments, Base64File, ContentBlockChatOutput, ContentBlockChunk, File,
    FinishReason, Input, InputMessage, InputMessageContent, RawText, Role, System, Template, Text,
    UrlFile, Usage,
};

use crate::tool::{
    DynamicToolParams, InferenceResponseToolCall, ProviderTool, Tool, ToolCallWrapper, ToolChoice,
    ToolResult,
};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};
use crate::variant::JsonMode;
use serde::Deserializer;

use super::embeddings::{embeddings, EmbeddingResponse};
use super::inference::{
    InferenceCredentials, InferenceOutput, InferenceResponse, InferenceResponseChunk,
    InferenceStream,
};
use crate::embeddings::EmbeddingEncodingFormat;
use crate::endpoints::{RequestApiKeyExtension, RouteHandlers};
use axum::routing::post;
use axum::Router;

/// Constructs (but does not register) all of our OpenAI-compatible endpoints.
/// The `RouterExt::register_openai_compatible_routes` is a convenience method
/// to register all of the routes on a router.
///
/// Alternatively, the returned `RouteHandlers` can be inspected (e.g. to allow middleware to see the route paths)
/// and then manually registered on a router.
pub fn build_openai_compatible_routes() -> RouteHandlers {
    RouteHandlers {
        routes: vec![
            ("/openai/v1/chat/completions", post(inference_handler)),
            ("/openai/v1/embeddings", post(embeddings_handler)),
        ],
    }
}

pub trait RouterExt {
    /// Applies our OpenAI-compatible endpoints to the router.
    /// This is used by the the gateway for the patched OpenAI python client (`start_openai_compatible_gateway`),
    /// as well as the normal standalone TensorZero gateway.
    fn register_openai_compatible_routes(self) -> Self;
}

impl RouterExt for Router<AppStateData> {
    fn register_openai_compatible_routes(mut self) -> Self {
        for (path, handler) in build_openai_compatible_routes().routes {
            self = self.route(path, handler);
        }
        self
    }
}

/// A handler for the OpenAI-compatible inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleParams>,
) -> Result<Response<Body>, Error> {
    if !openai_compatible_params.unknown_fields.is_empty() {
        if openai_compatible_params.tensorzero_deny_unknown_fields {
            let mut unknown_field_names = openai_compatible_params
                .unknown_fields
                .keys()
                .cloned()
                .collect::<Vec<_>>();

            unknown_field_names.sort();
            let unknown_field_names = unknown_field_names.join(", ");

            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: format!("`tensorzero::deny_unknown_fields` is set to true, but found unknown fields in the request: [{unknown_field_names}]")
            }));
        }
        tracing::warn!(
            "Ignoring unknown fields in OpenAI-compatible request: {:?}",
            openai_compatible_params
                .unknown_fields
                .keys()
                .collect::<Vec<_>>()
        );
    }
    let stream_options = openai_compatible_params.stream_options;
    let params = Params::try_from_openai(openai_compatible_params)?;

    // The prefix for the response's `model` field depends on the inference target
    // (We run this disambiguation deep in the `inference` call below but we don't get the decision out, so we duplicate it here)
    let response_model_prefix = match (&params.function_name, &params.model_name) {
        (Some(function_name), None) => Ok::<String, Error>(format!(
            "tensorzero::function_name::{function_name}::variant_name::",
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

    let response = inference(
        config,
        &http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        params,
        api_key_ext,
    )
    .await?;

    match response {
        InferenceOutput::NonStreaming(response) => {
            let openai_compatible_response =
                OpenAICompatibleResponse::from((response, response_model_prefix));
            Ok(Json(openai_compatible_response).into_response())
        }
        InferenceOutput::Streaming(stream) => {
            let openai_compatible_stream = prepare_serialized_openai_compatible_events(
                stream,
                response_model_prefix,
                stream_options,
            );
            Ok(Sse::new(openai_compatible_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response())
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct OpenAICompatibleEmbeddingParams {
    input: EmbeddingInput,
    model: String,
    dimensions: Option<u32>,
    #[serde(default)]
    encoding_format: EmbeddingEncodingFormat,
    #[serde(default, rename = "tensorzero::credentials")]
    tensorzero_credentials: InferenceCredentials,
    #[serde(rename = "tensorzero::dryrun")]
    tensorzero_dryrun: Option<bool>,
    #[serde(rename = "tensorzero::cache_options")]
    tensorzero_cache_options: Option<CacheParamsOptions>,
}

impl TryFrom<OpenAICompatibleEmbeddingParams> for EmbeddingParams {
    type Error = Error;
    fn try_from(params: OpenAICompatibleEmbeddingParams) -> Result<Self, Self::Error> {
        let model_name = match params
            .model
            .strip_prefix(TENSORZERO_EMBEDDING_MODEL_NAME_PREFIX)
        {
            Some(model_name) => model_name.to_string(),
            None => {
                tracing::warn!("Deprecation Warning: Model names in the OpenAI-compatible embeddings endpoint should be prefixed with 'tensorzero::embedding_model_name::'");
                params.model
            }
        };
        Ok(EmbeddingParams {
            input: params.input,
            model_name,
            dimensions: params.dimensions,
            encoding_format: params.encoding_format,
            credentials: params.tensorzero_credentials,
            dryrun: params.tensorzero_dryrun,
            cache_options: params.tensorzero_cache_options.unwrap_or_default(),
        })
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "object", rename_all = "lowercase")]
pub enum OpenAIEmbeddingResponse {
    List {
        data: Vec<OpenAIEmbedding>,
        model: String,
        usage: OpenAIEmbeddingUsage,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "object", rename_all = "lowercase")]
pub enum OpenAIEmbedding {
    Embedding { embedding: Embedding, index: usize },
}

#[derive(Debug, Serialize)]
pub struct OpenAIEmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

impl From<EmbeddingResponse> for OpenAIEmbeddingResponse {
    fn from(response: EmbeddingResponse) -> Self {
        OpenAIEmbeddingResponse::List {
            data: response
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(i, embedding)| OpenAIEmbedding::Embedding {
                    embedding,
                    index: i,
                })
                .collect(),
            model: format!("{TENSORZERO_EMBEDDING_MODEL_NAME_PREFIX}{}", response.model),
            usage: OpenAIEmbeddingUsage {
                prompt_tokens: response.usage.input_tokens,
                total_tokens: response.usage.input_tokens,
            },
        }
    }
}

pub async fn embeddings_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleEmbeddingParams>,
) -> Result<Json<OpenAIEmbeddingResponse>, Error> {
    let embedding_params = openai_compatible_params.try_into()?;
    let response = embeddings(
        config,
        &http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        embedding_params,
        api_key_ext,
    )
    .await?;
    Ok(Json(response.into()))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAICompatibleFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAICompatibleToolCallDelta {
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
    pub function: OpenAICompatibleToolCallDelta,
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum OpenAICompatibleResponseFormat {
    Text,
    JsonSchema { json_schema: JsonSchemaInfo },
    JsonObject,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct JsonSchemaInfo {
    name: String,
    description: Option<String>,
    schema: Option<Value>,
    #[serde(default)]
    strict: bool,
}

impl std::fmt::Display for JsonSchemaInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAICompatibleAllowedToolsMode {
    Auto,
    Required,
}

impl From<OpenAICompatibleAllowedToolsMode> for ToolChoice {
    fn from(mode: OpenAICompatibleAllowedToolsMode) -> Self {
        match mode {
            OpenAICompatibleAllowedToolsMode::Auto => ToolChoice::Auto,
            OpenAICompatibleAllowedToolsMode::Required => ToolChoice::Required,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct OpenAICompatibleAllowedTools {
    tools: Vec<OpenAICompatibleNamedToolChoice>,
    mode: OpenAICompatibleAllowedToolsMode,
}

/// Controls which (if any) tool is called by the model.
/// `none` means the model will not call any tool and instead generates a message.
/// `auto` means the model can pick between generating a message or calling one or more tools.
/// `required` means the model must call one or more tools.
/// Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.
///
/// `none` is the default when no tools are present. `auto` is the default if tools are present.
#[derive(Clone, Debug, Default, PartialEq)]
enum ChatCompletionToolChoiceOption {
    #[default]
    None,
    Auto,
    Required,
    AllowedTools(OpenAICompatibleAllowedTools),
    Named(OpenAICompatibleNamedToolChoice),
}

impl<'de> Deserialize<'de> for ChatCompletionToolChoiceOption {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        use serde_json::Value;

        let value = Value::deserialize(deserializer)?;

        match &value {
            Value::String(s) => match s.as_str() {
                "none" => Ok(ChatCompletionToolChoiceOption::None),
                "auto" => Ok(ChatCompletionToolChoiceOption::Auto),
                "required" => Ok(ChatCompletionToolChoiceOption::Required),
                _ => Err(D::Error::custom(format!("Invalid tool choice string: {s}"))),
            },
            Value::Object(obj) => {
                if let Some(type_value) = obj.get("type") {
                    if let Some(type_str) = type_value.as_str() {
                        match type_str {
                            "function" => {
                                // This is a named tool choice
                                let named: OpenAICompatibleNamedToolChoice =
                                    serde_json::from_value(value).map_err(D::Error::custom)?;
                                Ok(ChatCompletionToolChoiceOption::Named(named))
                            }
                            "allowed_tools" => {
                                // This is an allowed tools choice - extract the allowed_tools field
                                if let Some(allowed_tools_value) = obj.get("allowed_tools") {
                                    let allowed_tools: OpenAICompatibleAllowedTools =
                                        serde_json::from_value(allowed_tools_value.clone())
                                            .map_err(D::Error::custom)?;
                                    Ok(ChatCompletionToolChoiceOption::AllowedTools(allowed_tools))
                                } else {
                                    Err(D::Error::custom(
                                        "Missing 'allowed_tools' field in allowed_tools type",
                                    ))
                                }
                            }
                            _ => Err(D::Error::custom(format!(
                                "Invalid tool choice type: {type_str}",
                            ))),
                        }
                    } else {
                        Err(D::Error::custom(
                            "Tool choice 'type' field must be a string",
                        ))
                    }
                } else {
                    Err(D::Error::custom(
                        "Tool choice field must have a 'type' field if it is an object",
                    ))
                }
            }
            _ => Err(D::Error::custom("Tool choice must be a string or object")),
        }
    }
}

impl ChatCompletionToolChoiceOption {
    fn into_tool_params(self) -> OpenAICompatibleToolChoiceParams {
        match self {
            ChatCompletionToolChoiceOption::None => OpenAICompatibleToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::None),
            },
            ChatCompletionToolChoiceOption::Auto => OpenAICompatibleToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::Auto),
            },
            ChatCompletionToolChoiceOption::Required => OpenAICompatibleToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::Required),
            },
            ChatCompletionToolChoiceOption::AllowedTools(allowed_tool_info) => {
                OpenAICompatibleToolChoiceParams {
                    allowed_tools: Some(
                        allowed_tool_info
                            .tools
                            .into_iter()
                            .map(|tool| tool.function.name)
                            .collect(),
                    ),
                    tool_choice: Some(allowed_tool_info.mode.into()),
                }
            }
            ChatCompletionToolChoiceOption::Named(named_tool) => OpenAICompatibleToolChoiceParams {
                allowed_tools: None,
                tool_choice: Some(ToolChoice::Specific(named_tool.function.name)),
            },
        }
    }
}

#[derive(Default)]
struct OpenAICompatibleToolChoiceParams {
    pub allowed_tools: Option<Vec<String>>,
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
struct OpenAICompatibleStreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
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
    stop: Option<Vec<String>>,
    reasoning_effort: Option<String>,
    service_tier: Option<ServiceTier>,
    verbosity: Option<String>,
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
    #[serde(default, rename = "tensorzero::tags")]
    tensorzero_tags: HashMap<String, String>,
    #[serde(default, rename = "tensorzero::deny_unknown_fields")]
    tensorzero_deny_unknown_fields: bool,
    #[serde(default, rename = "tensorzero::credentials")]
    tensorzero_credentials: InferenceCredentials,
    #[serde(rename = "tensorzero::internal_dynamic_variant_config")]
    tensorzero_internal_dynamic_variant_config: Option<UninitializedVariantInfo>,
    #[serde(default, rename = "tensorzero::provider_tools")]
    tensorzero_provider_tools: Option<Vec<ProviderTool>>,
    #[serde(default, rename = "tensorzero::params")]
    tensorzero_params: Option<InferenceParams>,
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
struct OpenAICompatibleResponse {
    id: String,
    episode_id: String,
    choices: Vec<OpenAICompatibleChoice>,
    created: u32,
    model: String,
    system_fingerprint: String,
    service_tier: Option<String>,
    object: String,
    usage: OpenAICompatibleUsage,
}

const TENSORZERO_FUNCTION_NAME_PREFIX: &str = "tensorzero::function_name::";
const TENSORZERO_MODEL_NAME_PREFIX: &str = "tensorzero::model_name::";
const TENSORZERO_EMBEDDING_MODEL_NAME_PREFIX: &str = "tensorzero::embedding_model_name::";

impl Params {
    fn try_from_openai(openai_compatible_params: OpenAICompatibleParams) -> Result<Self, Error> {
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

        // Override the inference parameters with OpenAI-compatible parameters
        // TODO (GabrielBianconi): Should we warn if we override parameters that are already set?
        // Currently: OpenAI-compatible parameters take precedence over TensorZero parameters
        inference_params.chat_completion = ChatCompletionInferenceParams {
            frequency_penalty: openai_compatible_params
                .frequency_penalty
                .or(inference_params.chat_completion.frequency_penalty),
            json_mode: json_mode.or(inference_params.chat_completion.json_mode),
            max_tokens: max_tokens.or(inference_params.chat_completion.max_tokens),
            presence_penalty: openai_compatible_params
                .presence_penalty
                .or(inference_params.chat_completion.presence_penalty),
            reasoning_effort: openai_compatible_params
                .reasoning_effort
                .or(inference_params.chat_completion.reasoning_effort),
            service_tier: openai_compatible_params
                .service_tier
                .or(inference_params.chat_completion.service_tier),
            seed: openai_compatible_params
                .seed
                .or(inference_params.chat_completion.seed),
            stop_sequences: openai_compatible_params
                .stop
                .or(inference_params.chat_completion.stop_sequences),
            temperature: openai_compatible_params
                .temperature
                .or(inference_params.chat_completion.temperature),
            thinking_budget_tokens: inference_params.chat_completion.thinking_budget_tokens,
            top_p: openai_compatible_params
                .top_p
                .or(inference_params.chat_completion.top_p),
            verbosity: openai_compatible_params
                .verbosity
                .or(inference_params.chat_completion.verbosity),
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
                .map(|tools| tools.into_iter().map(OpenAICompatibleTool::into).collect()),
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

#[derive(Deserialize, Debug)]
#[serde(tag = "type", deny_unknown_fields, rename_all = "snake_case")]
enum OpenAICompatibleContentBlock {
    Text(TextContent),
    ImageUrl {
        image_url: OpenAICompatibleImageUrl,
    },
    File {
        file: OpenAICompatibleFile,
    },
    #[serde(rename = "tensorzero::raw_text")]
    RawText(RawText),
    #[serde(rename = "tensorzero::template")]
    Template(Template),
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", deny_unknown_fields, rename_all = "snake_case")]
struct OpenAICompatibleImageUrl {
    url: Url,
    #[serde(rename = "tensorzero::mime_type")]
    mime_type: Option<MediaType>,
    #[serde(default)]
    detail: Option<Detail>,
}

#[derive(Deserialize, Debug)]
struct OpenAICompatibleFile {
    file_data: String,
    // TODO (#4478): collect and store filename
    // filename: String,
    // OpenAI supports file_id with their files API
    // We do not so we require these two fields
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

fn parse_base64_file_data_url(data_url: &str) -> Result<(MediaType, &str), Error> {
    let Some(url) = data_url.strip_prefix("data:") else {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "Expected a base64-encoded data URL with MIME type (e.g. `data:image/png;base64,SGVsbG8sIFdvcmxkIQ==`), but got a value without the `data:` prefix.".to_string(),
        }));
    };
    let Some((mime_type, data)) = url.split_once(";base64,") else {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "Expected a base64-encoded data URL with MIME type (e.g. `data:image/png;base64,SGVsbG8sIFdvcmxkIQ==`), but got a value without the `;base64,` separator.".to_string(),
        }));
    };
    let file_type: MediaType = mime_type.parse().map_err(|_| {
        Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: format!("Unknown MIME type `{mime_type}` in data URL"),
        })
    })?;
    Ok((file_type, data))
}

fn convert_openai_message_content(
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
                        tracing::warn!("Deprecation Warning: Using `tensorzero::arguments` in text content blocks is deprecated. Please use `{{\"type\": \"tensorzero::template\", \"name\": \"role\", \"arguments\": {{...}}}}` instead.");
                        InputMessageContent::Template(Template { name: role.clone(), arguments: tensorzero_arguments })
                    }
                    Ok(OpenAICompatibleContentBlock::ImageUrl { image_url }) => {
                        if image_url.url.scheme() == "data" {
                            let image_url_str = image_url.url.to_string();
                            let (mime_type, data) = parse_base64_file_data_url(&image_url_str)?;
                            let base64_file = Base64File::new(None, mime_type, data.to_string(), image_url.detail)?;
                            InputMessageContent::File(File::Base64(base64_file))
                        } else {
                            InputMessageContent::File(File::Url(UrlFile { url: image_url.url, mime_type: image_url.mime_type, detail: image_url.detail }))
                        }
                    }
                    Ok(OpenAICompatibleContentBlock::File { file }) => {
                        let (mime_type, data) = parse_base64_file_data_url(&file.file_data)?;
                        let base64_file = Base64File::new(None, mime_type, data.to_string(), None)?;
                        InputMessageContent::File(File::Base64(base64_file))
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
                        tracing::warn!(r#"Deprecation Warning: Content block `{val}` was not a valid OpenAI content block. Please use `{{"type": "tensorzero::template", "name": "role", "arguments": {{"custom": "data"}}}}` to pass arbitrary JSON values to TensorZero: {e}"#);
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

impl From<OpenAICompatibleToolCall> for ToolCallWrapper {
    fn from(tool_call: OpenAICompatibleToolCall) -> Self {
        ToolCallWrapper::InferenceResponseToolCall(InferenceResponseToolCall {
            id: tool_call.id,
            raw_name: tool_call.function.name,
            raw_arguments: tool_call.function.arguments,
            name: None,
            arguments: None,
        })
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

impl From<InferenceResponseToolCall> for OpenAICompatibleToolCall {
    fn from(tool_call: InferenceResponseToolCall) -> Self {
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
    service_tier: Option<String>,
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
    // if it’s None → skip, or if the Vec is empty → skip
    v.as_ref().is_none_or(Vec::is_empty)
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct OpenAICompatibleDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "is_none_or_empty")]
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
                    finish_reason: c.finish_reason.map(FinishReason::into),
                    logprobs: None,
                    delta: OpenAICompatibleDelta {
                        content,
                        tool_calls: Some(tool_calls),
                    },
                }],
                created: current_timestamp() as u32,
                service_tier: None,
                model: format!("{response_model_prefix}{}", c.variant_name),
                system_fingerprint: String::new(),
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
                finish_reason: c.finish_reason.map(FinishReason::into),
                logprobs: None,
                delta: OpenAICompatibleDelta {
                    content: Some(c.raw),
                    tool_calls: None,
                },
            }],
            created: current_timestamp() as u32,
            service_tier: None,
            model: format!("{response_model_prefix}{}", c.variant_name),
            system_fingerprint: String::new(),
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
                    function: OpenAICompatibleToolCallDelta {
                        name: tool_call.raw_name.unwrap_or_default(),
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
            ContentBlockChunk::Unknown(_) => {
                // OpenAI compatible endpoint does not support unknown blocks
                // Users of this endpoint will need to check observability to see them
                tracing::warn!(
                    "Ignoring 'unknown' content block chunk when constructing OpenAI-compatible response"
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
        let mut variant_name = None;
        while let Some(chunk) = stream.next().await {
            // NOTE - in the future, we may want to end the stream early if we get an error
            // For now, we just ignore the error and try to get more chunks
            let Ok(chunk) = chunk else {
                continue;
            };
            inference_id = Some(chunk.inference_id());
            episode_id = Some(chunk.episode_id());
            variant_name = Some(chunk.variant_name().to_string());
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
            let openai_compatible_chunks = convert_inference_response_chunk_to_openai_compatible(chunk, &mut tool_id_to_index, &response_model_prefix);
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
            let variant_name = variant_name.ok_or_else(|| {
                Error::new(ErrorDetails::Inference {
                    message: "Cannot find variant_name - no chunks were produced by TensorZero".to_string(),
                })
            })?;
            let usage_chunk = OpenAICompatibleResponseChunk {
                id: inference_id.to_string(),
                episode_id: episode_id.to_string(),
                choices: vec![],
                created: current_timestamp() as u32,
                model: format!("{response_model_prefix}{variant_name}"),
                system_fingerprint: String::new(),
                object: "chat.completion.chunk".to_string(),
                service_tier: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::cache::CacheEnabledMode;
    use crate::inference::types::file::Detail;
    use crate::inference::types::{System, Text, TextChunk};
    use crate::tool::ToolCallChunk;

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
            response_format: None,
            seed: Some(23),
            stream: None,
            temperature: Some(0.5),
            tools: None,
            tool_choice: None,
            top_p: Some(0.5),
            parallel_tool_calls: None,
            tensorzero_episode_id: Some(episode_id),
            tensorzero_variant_name: Some("test_variant".to_string()),
            tensorzero_dryrun: None,
            tensorzero_cache_options: None,
            tensorzero_extra_body: UnfilteredInferenceExtraBody::default(),
            tensorzero_extra_headers: UnfilteredInferenceExtraHeaders::default(),
            tensorzero_tags: tensorzero_tags.clone(),
            tensorzero_deny_unknown_fields: false,
            tensorzero_credentials: InferenceCredentials::default(),
            unknown_fields: Default::default(),
            stream_options: None,
            stop: None,
            tensorzero_internal_dynamic_variant_config: None,
            tensorzero_provider_tools: None,
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
    fn test_process_chat_content_chunk() {
        let content = vec![
            ContentBlockChunk::Text(TextChunk {
                id: "1".to_string(),
                text: "Hello".to_string(),
            }),
            ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "1".to_string(),
                raw_name: Some("test_tool".to_string()),
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
        assert_eq!(tool_calls[0].function.name, "test_tool".to_string());
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
                raw_name: Some("middle_tool".to_string()),
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
                raw_name: Some("last_tool".to_string()),
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
        assert_eq!(tool_calls[0].function.name, "middle_tool".to_string());
        assert_eq!(tool_calls[0].function.arguments, "{\"key\": \"value\"}");
        assert_eq!(tool_calls[1].id, Some("5".to_string()));
        assert_eq!(tool_calls[1].index, 1);
        assert_eq!(tool_calls[1].function.name, "last_tool".to_string());
        assert_eq!(tool_calls[1].function.arguments, "{\"key\": \"value\"}");
    }

    #[test]
    fn test_parse_base64_file_data_url() {
        assert_eq!(
            (mime::IMAGE_JPEG, "YWJjCg=="),
            parse_base64_file_data_url("data:image/jpeg;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            (mime::IMAGE_PNG, "YWJjCg=="),
            parse_base64_file_data_url("data:image/png;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            ("image/webp".parse().unwrap(), "YWJjCg=="),
            parse_base64_file_data_url("data:image/webp;base64,YWJjCg==").unwrap()
        );
        assert_eq!(
            ("application/pdf".parse().unwrap(), "JVBERi0xLjQK"),
            parse_base64_file_data_url("data:application/pdf;base64,JVBERi0xLjQK").unwrap()
        );

        // Test error when prefix is missing
        let result = parse_base64_file_data_url("YWJjCg==");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("without the `data:` prefix"));

        // Test error when base64 separator is missing
        let result = parse_base64_file_data_url("data:image/png,YWJjCg==");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("without the `;base64,` separator"));
    }

    #[test]
    fn test_cache_options() {
        // Test default cache options (should be write-only)
        let params = Params::try_from_openai(OpenAICompatibleParams {
            messages: vec![OpenAICompatibleMessage::User(OpenAICompatibleUserMessage {
                content: Value::String("test".to_string()),
            })],
            model: "tensorzero::function_name::test_function".into(),
            frequency_penalty: None,
            max_tokens: None,
            max_completion_tokens: None,
            presence_penalty: None,
            response_format: None,
            seed: None,
            stream: None,
            temperature: None,
            tools: None,
            tool_choice: None,
            top_p: None,
            parallel_tool_calls: None,
            tensorzero_variant_name: None,
            tensorzero_dryrun: None,
            tensorzero_episode_id: None,
            tensorzero_cache_options: None,
            tensorzero_extra_body: UnfilteredInferenceExtraBody::default(),
            tensorzero_extra_headers: UnfilteredInferenceExtraHeaders::default(),
            tensorzero_tags: HashMap::new(),
            tensorzero_credentials: InferenceCredentials::default(),
            unknown_fields: Default::default(),
            stream_options: None,
            stop: None,
            tensorzero_deny_unknown_fields: false,
            tensorzero_internal_dynamic_variant_config: None,
            tensorzero_provider_tools: None,
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
            frequency_penalty: None,
            max_tokens: None,
            max_completion_tokens: None,
            presence_penalty: None,
            response_format: None,
            seed: None,
            stream: None,
            temperature: None,
            tools: None,
            tool_choice: None,
            top_p: None,
            parallel_tool_calls: None,
            tensorzero_variant_name: None,
            tensorzero_dryrun: None,
            tensorzero_episode_id: None,
            tensorzero_cache_options: Some(CacheParamsOptions {
                max_age_s: Some(3600),
                enabled: CacheEnabledMode::On,
            }),
            tensorzero_extra_body: UnfilteredInferenceExtraBody::default(),
            tensorzero_extra_headers: UnfilteredInferenceExtraHeaders::default(),
            tensorzero_tags: HashMap::new(),
            tensorzero_credentials: InferenceCredentials::default(),
            unknown_fields: Default::default(),
            stream_options: None,
            stop: None,
            tensorzero_deny_unknown_fields: false,
            tensorzero_internal_dynamic_variant_config: None,
            tensorzero_provider_tools: None,
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
            frequency_penalty: None,
            max_tokens: None,
            max_completion_tokens: None,
            presence_penalty: None,
            response_format: None,
            seed: None,
            stream: None,
            temperature: None,
            tools: None,
            tool_choice: None,
            top_p: None,
            parallel_tool_calls: None,
            tensorzero_variant_name: None,
            tensorzero_dryrun: Some(true),
            tensorzero_episode_id: None,
            tensorzero_cache_options: Some(CacheParamsOptions {
                max_age_s: Some(3600),
                enabled: CacheEnabledMode::On,
            }),
            tensorzero_extra_body: UnfilteredInferenceExtraBody::default(),
            tensorzero_extra_headers: UnfilteredInferenceExtraHeaders::default(),
            tensorzero_tags: HashMap::new(),
            tensorzero_credentials: InferenceCredentials::default(),
            unknown_fields: Default::default(),
            stream_options: None,
            stop: None,
            tensorzero_deny_unknown_fields: false,
            tensorzero_internal_dynamic_variant_config: None,
            tensorzero_provider_tools: None,
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
            frequency_penalty: None,
            max_tokens: None,
            max_completion_tokens: None,
            presence_penalty: None,
            response_format: None,
            seed: None,
            stream: None,
            temperature: None,
            tools: None,
            tool_choice: None,
            top_p: None,
            parallel_tool_calls: None,
            tensorzero_variant_name: None,
            tensorzero_dryrun: Some(true),
            tensorzero_episode_id: None,
            tensorzero_cache_options: Some(CacheParamsOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            }),
            tensorzero_extra_body: UnfilteredInferenceExtraBody::default(),
            tensorzero_extra_headers: UnfilteredInferenceExtraHeaders::default(),
            tensorzero_tags: HashMap::new(),
            tensorzero_credentials: InferenceCredentials::default(),
            unknown_fields: Default::default(),
            stream_options: None,
            stop: None,
            tensorzero_deny_unknown_fields: false,
            tensorzero_internal_dynamic_variant_config: None,
            tensorzero_provider_tools: None,
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
    fn test_try_from_embedding_params_deprecated() {
        let logs_contain = crate::utils::testing::capture_logs();
        let openai_embedding_params = OpenAICompatibleEmbeddingParams {
            input: EmbeddingInput::Single("foo".to_string()),
            model: "text-embedding-ada-002".to_string(),
            dimensions: Some(15),
            encoding_format: EmbeddingEncodingFormat::Float,
            tensorzero_credentials: InferenceCredentials::default(),
            tensorzero_dryrun: None,
            tensorzero_cache_options: None,
        };
        let param: EmbeddingParams = openai_embedding_params.try_into().unwrap();
        assert_eq!(param.model_name, "text-embedding-ada-002");
        assert_eq!(param.dimensions, Some(15));
        assert_eq!(param.encoding_format, EmbeddingEncodingFormat::Float);
        assert!(logs_contain("Deprecation Warning: Model names in the OpenAI-compatible embeddings endpoint should be prefixed with 'tensorzero::embedding_model_name::'"));
    }
    #[test]
    fn test_try_from_embedding_params_strip() {
        let logs_contain = crate::utils::testing::capture_logs();
        let openai_embedding_params = OpenAICompatibleEmbeddingParams {
            input: EmbeddingInput::Single("foo".to_string()),
            model: "tensorzero::embedding_model_name::text-embedding-ada-002".to_string(),
            dimensions: Some(15),
            encoding_format: EmbeddingEncodingFormat::Float,
            tensorzero_credentials: InferenceCredentials::default(),
            tensorzero_dryrun: None,
            tensorzero_cache_options: None,
        };
        let param: EmbeddingParams = openai_embedding_params.try_into().unwrap();
        assert_eq!(param.model_name, "text-embedding-ada-002");
        assert_eq!(param.dimensions, Some(15));
        assert_eq!(param.encoding_format, EmbeddingEncodingFormat::Float);
        assert!(!logs_contain("Deprecation Warning: Model names in the OpenAI-compatible embeddings endpoint should be prefixed with 'tensorzero::embedding_model_name::'"));
    }

    #[test]
    fn test_chat_completion_tool_choice_option_deserialization_and_conversion() {
        // Test deserialization from JSON and conversion to OpenAICompatibleToolChoiceParams

        // Test None variant
        let json_none = json!("none");
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_none).unwrap();
        assert_eq!(tool_choice, ChatCompletionToolChoiceOption::None);
        let params = tool_choice.into_tool_params();
        assert_eq!(params.allowed_tools, None);
        assert_eq!(params.tool_choice, Some(ToolChoice::None));

        // Test Auto variant
        let json_auto = json!("auto");
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_auto).unwrap();
        assert_eq!(tool_choice, ChatCompletionToolChoiceOption::Auto);
        let params = tool_choice.into_tool_params();
        assert_eq!(params.allowed_tools, None);
        assert_eq!(params.tool_choice, Some(ToolChoice::Auto));

        // Test Required variant
        let json_required = json!("required");
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_required).unwrap();
        assert_eq!(tool_choice, ChatCompletionToolChoiceOption::Required);
        let params = tool_choice.into_tool_params();
        assert_eq!(params.allowed_tools, None);
        assert_eq!(params.tool_choice, Some(ToolChoice::Required));

        // Test Named variant (specific tool)
        let json_named = json!({
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        });
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_named).unwrap();
        assert_eq!(
            tool_choice,
            ChatCompletionToolChoiceOption::Named(OpenAICompatibleNamedToolChoice {
                r#type: "function".to_string(),
                function: FunctionName {
                    name: "get_weather".to_string()
                }
            })
        );
        let params = tool_choice.into_tool_params();
        assert_eq!(params.allowed_tools, None);
        assert_eq!(
            params.tool_choice,
            Some(ToolChoice::Specific("get_weather".to_string()))
        );

        // Test AllowedTools variant with auto mode
        let json_allowed_auto = json!({
            "type": "allowed_tools",
            "allowed_tools": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email"
                    }
                }
            ],
            "mode": "auto"
        }});
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_allowed_auto).unwrap();
        assert_eq!(
            tool_choice,
            ChatCompletionToolChoiceOption::AllowedTools(OpenAICompatibleAllowedTools {
                tools: vec![
                    OpenAICompatibleNamedToolChoice {
                        r#type: "function".to_string(),
                        function: FunctionName {
                            name: "get_weather".to_string()
                        }
                    },
                    OpenAICompatibleNamedToolChoice {
                        r#type: "function".to_string(),
                        function: FunctionName {
                            name: "send_email".to_string()
                        }
                    }
                ],
                mode: OpenAICompatibleAllowedToolsMode::Auto
            })
        );
        let params = tool_choice.into_tool_params();
        assert_eq!(
            params.allowed_tools,
            Some(vec!["get_weather".to_string(), "send_email".to_string()])
        );
        assert_eq!(params.tool_choice, Some(ToolChoice::Auto));

        // Test AllowedTools variant with required mode
        let json_allowed_required = json!({
            "type": "allowed_tools",
            "allowed_tools": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email"
                    }
                }
            ],
            "mode": "required"
        }});
        let tool_choice: ChatCompletionToolChoiceOption =
            serde_json::from_value(json_allowed_required).unwrap();
        assert_eq!(
            tool_choice,
            ChatCompletionToolChoiceOption::AllowedTools(OpenAICompatibleAllowedTools {
                tools: vec![
                    OpenAICompatibleNamedToolChoice {
                        r#type: "function".to_string(),
                        function: FunctionName {
                            name: "get_weather".to_string()
                        }
                    },
                    OpenAICompatibleNamedToolChoice {
                        r#type: "function".to_string(),
                        function: FunctionName {
                            name: "send_email".to_string()
                        }
                    }
                ],
                mode: OpenAICompatibleAllowedToolsMode::Required
            })
        );
        let params = tool_choice.into_tool_params();
        assert_eq!(
            params.allowed_tools,
            Some(vec!["get_weather".to_string(), "send_email".to_string()])
        );
        assert_eq!(params.tool_choice, Some(ToolChoice::Required));

        // Test default value (should be None)
        let tool_choice_default = ChatCompletionToolChoiceOption::default();
        assert_eq!(tool_choice_default, ChatCompletionToolChoiceOption::None);
        let params_default = tool_choice_default.into_tool_params();
        assert_eq!(params_default.allowed_tools, None);
        assert_eq!(params_default.tool_choice, Some(ToolChoice::None));
    }

    #[test]
    fn test_chat_completion_tool_choice_option_invalid_deserialization() {
        // Test invalid JSON values that should fail to deserialize

        // Invalid string value
        let json_invalid = json!("invalid_choice");
        let result: Result<ChatCompletionToolChoiceOption, _> =
            serde_json::from_value(json_invalid);
        assert!(result.is_err());

        // Invalid object structure for named tool choice
        let json_invalid_named = json!({
            "type": "invalid_type",
            "function": {
                "name": "test"
            }
        });
        let result: Result<ChatCompletionToolChoiceOption, _> =
            serde_json::from_value(json_invalid_named);
        assert!(result.is_err());

        // Missing function name in named tool choice
        let json_missing_name = json!({
            "type": "function",
            "function": {}
        });
        let result: Result<ChatCompletionToolChoiceOption, _> =
            serde_json::from_value(json_missing_name);
        assert!(result.is_err());

        // Invalid mode in allowed tools
        let json_invalid_mode = json!({
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test"
                    }
                }
            ],
            "mode": "invalid_mode"
        });
        let result: Result<ChatCompletionToolChoiceOption, _> =
            serde_json::from_value(json_invalid_mode);
        assert!(result.is_err());

        // Test AllowedTools variant with no type
        let json_allowed_required = json!({
            "allowed_tools": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email"
                    }
                }
            ],
            "mode": "required"
        }});
        let err = serde_json::from_value::<ChatCompletionToolChoiceOption>(json_allowed_required)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Tool choice field must have a 'type' field if it is an object"
        );
    }

    #[test]
    fn test_openai_compatible_allowed_tools_mode_conversion() {
        // Test conversion from OpenAICompatibleAllowedToolsMode to ToolChoice
        let auto_mode = OpenAICompatibleAllowedToolsMode::Auto;
        let tool_choice: ToolChoice = auto_mode.into();
        assert_eq!(tool_choice, ToolChoice::Auto);

        let required_mode = OpenAICompatibleAllowedToolsMode::Required;
        let tool_choice: ToolChoice = required_mode.into();
        assert_eq!(tool_choice, ToolChoice::Required);
    }

    #[test]
    fn test_deserialize_image_url_with_detail() {
        // Test deserialization with detail: low
        let json_low = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "low"
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json_low).unwrap();
        match block {
            OpenAICompatibleContentBlock::ImageUrl { image_url } => {
                assert_eq!(image_url.url.as_str(), "https://example.com/image.png");
                assert_eq!(image_url.detail, Some(Detail::Low));
            }
            _ => panic!("Expected ImageUrl variant"),
        }

        // Test deserialization with detail: high
        let json_high = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "high"
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json_high).unwrap();
        match block {
            OpenAICompatibleContentBlock::ImageUrl { image_url } => {
                assert_eq!(image_url.detail, Some(Detail::High));
            }
            _ => panic!("Expected ImageUrl variant"),
        }

        // Test deserialization with detail: auto
        let json_auto = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "auto"
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json_auto).unwrap();
        match block {
            OpenAICompatibleContentBlock::ImageUrl { image_url } => {
                assert_eq!(image_url.detail, Some(Detail::Auto));
            }
            _ => panic!("Expected ImageUrl variant"),
        }

        // Test deserialization without detail (should default to None)
        let json_none = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png"
            }
        });
        let block: OpenAICompatibleContentBlock = serde_json::from_value(json_none).unwrap();
        match block {
            OpenAICompatibleContentBlock::ImageUrl { image_url } => {
                assert_eq!(image_url.detail, None);
            }
            _ => panic!("Expected ImageUrl variant"),
        }

        // Test deserialization with invalid detail should fail
        let json_invalid = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "invalid"
            }
        });
        let result: Result<OpenAICompatibleContentBlock, _> = serde_json::from_value(json_invalid);
        assert!(result.is_err());
    }
}
