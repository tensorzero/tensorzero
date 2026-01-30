use std::borrow::Cow;

use futures::{StreamExt, TryStreamExt};
use lazy_static::lazy_static;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use serde_json::{Value, json};
use tensorzero_types_providers::xai::{
    XAIAssistantRequestMessage, XAIRequestMessage, XAIResponse as XAIResponseGeneric,
    XAISystemRequestMessage, XAIToolRequestMessage, XAIUsage, XAIUserRequestMessage,
};
use tokio::time::Instant;
use url::Url;

use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{
    DelayedError, DisplayOrDebugGateway, Error, ErrorDetails, warn_discarded_thought_block,
};
use crate::http::TensorzeroHttpClient;
use crate::inference::InferenceProvider;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported,
};
use crate::inference::types::usage::raw_usage_entries_from_value;
use crate::inference::types::{
    ApiType, ContentBlock, ContentBlockOutput, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, RequestMessage, Role, Text, Thought, Unknown, Usage,
    batch::StartBatchProviderInferenceResponse,
};
use crate::model::{Credential, ModelProvider};
use crate::providers::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};
use crate::providers::openai::{
    OpenAIMessagesConfig, OpenAIResponseChoice, StreamOptions, get_chat_url, handle_openai_error,
    prepare_file_message, stream_openai,
};
use uuid::Uuid;

use crate::inference::TensorZeroEventError;
use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::chat_completions::{ChatCompletionTool, ChatCompletionToolChoice};

lazy_static! {
    static ref XAI_DEFAULT_BASE_URL: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.x.ai/v1").expect("Failed to parse XAI_DEFAULT_BASE_URL")
    };
}

const PROVIDER_NAME: &str = "xAI";
pub const PROVIDER_TYPE: &str = "xai";

type XAIResponse = XAIResponseGeneric<OpenAIResponseChoice>;

impl From<XAIUsage> for Usage {
    fn from(usage: XAIUsage) -> Self {
        // Add `reasoning_tokens` to `completion_tokens` for total output tokens
        let output_tokens = match (usage.completion_tokens, usage.completion_tokens_details) {
            (Some(completion), Some(details)) => {
                Some(completion + details.reasoning_tokens.unwrap_or(0))
            }
            (Some(completion), None) => Some(completion),
            (None, Some(details)) => details.reasoning_tokens,
            (None, None) => None,
        };
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens,
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct XAIProvider {
    model_name: String,
    #[serde(skip)]
    credentials: XAICredentials,
}

impl XAIProvider {
    pub fn new(model_name: String, credentials: XAICredentials) -> Self {
        XAIProvider {
            model_name,
            credentials,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum XAICredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<XAICredentials>,
        fallback: Box<XAICredentials>,
    },
}

impl TryFrom<Credential> for XAICredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(XAICredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(XAICredentials::Dynamic(key_name)),
            Credential::None => Ok(XAICredentials::None),
            Credential::Missing => Ok(XAICredentials::None),
            Credential::WithFallback { default, fallback } => Ok(XAICredentials::WithFallback {
                default: Box::new((*default).try_into()?),
                fallback: Box::new((*fallback).try_into()?),
            }),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for xAI provider".to_string(),
            })),
        }
    }
}

impl XAICredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            XAICredentials::Static(api_key) => Ok(api_key),
            XAICredentials::Dynamic(key_name) => dynamic_api_keys.get(key_name).ok_or_else(|| {
                DelayedError::new(ErrorDetails::ApiKeyMissing {
                    provider_name: PROVIDER_NAME.to_string(),
                    message: format!("Dynamic api key `{key_name}` is missing"),
                })
            }),
            XAICredentials::WithFallback { default, fallback } => {
                // Try default first, fall back to fallback if it fails
                match default.get_api_key(dynamic_api_keys) {
                    Ok(key) => Ok(key),
                    Err(e) => {
                        e.log_at_level(
                            "Using fallback credential, as default credential is unavailable: ",
                            tracing::Level::WARN,
                        );
                        fallback.get_api_key(dynamic_api_keys)
                    }
                }
            }
            XAICredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
        }
    }
}

impl InferenceProvider for XAIProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
            model_inference_id,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(XAIRequest::new(&self.model_name, request).await?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing xAI request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let request_url = get_chat_url(&XAI_DEFAULT_BASE_URL)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let request_builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;

        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(XAIResponseWithMetadata {
                response,
                raw_response,
                latency,
                raw_request,
                generic_request: request,
                model_inference_id,
            }
            .try_into()?)
        } else {
            let status = res.status();

            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            Err(handle_openai_error(
                &raw_request,
                status,
                &response,
                PROVIDER_TYPE,
                None,
            ))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
            model_inference_id,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(XAIRequest::new(&self.model_name, request).await?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing xAI request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;

        let request_url = get_chat_url(&XAI_DEFAULT_BASE_URL)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let request_builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (event_source, raw_request) = inject_extra_request_data_and_send_eventsource(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;

        let inner_stream = stream_openai(
            PROVIDER_TYPE.to_string(),
            model_inference_id,
            event_source.map_err(TensorZeroEventError::EventSource),
            start_time,
            None,
            &raw_request,
        );
        // Wrap with xAI-specific stream to add reasoning tokens to usage
        let stream = stream_xai(inner_stream).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "xAI".to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

// =============================================================================
// Message Preparation
// =============================================================================

/// Prepares xAI-specific messages from TensorZero request messages.
/// This is similar to `prepare_openai_messages` but handles Thought blocks
/// by extracting reasoning_content for assistant messages.
async fn prepare_xai_messages<'a>(
    system: Option<&'a str>,
    messages: &'a [RequestMessage],
    messages_config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<XAIRequestMessage<'a>>, Error> {
    let mut xai_messages = Vec::new();

    // Add system message if present
    if let Some(system_content) = system {
        // Check if we need to prepend JSON mode instruction
        // Only add if neither system content nor messages contain "json"
        let system_content = if messages_config.json_mode
            == Some(&ModelInferenceRequestJsonMode::On)
            && !system_content.to_lowercase().contains("json")
            && !messages_contain_json_hint(messages)
        {
            Cow::Owned(format!("Respond using JSON.\n\n{system_content}"))
        } else {
            Cow::Borrowed(system_content)
        };
        xai_messages.push(XAIRequestMessage::System(XAISystemRequestMessage {
            content: system_content,
        }));
    } else if messages_config.json_mode == Some(&ModelInferenceRequestJsonMode::On)
        && !messages_contain_json_hint(messages)
    {
        // Add JSON mode instruction as system message if no system message present
        xai_messages.push(XAIRequestMessage::System(XAISystemRequestMessage {
            content: Cow::Borrowed("Respond using JSON."),
        }));
    }

    // Convert each message
    for message in messages {
        let converted = tensorzero_to_xai_message(message, messages_config).await?;
        xai_messages.extend(converted);
    }

    Ok(xai_messages)
}

/// Checks if messages contain the word "json" (case-insensitive)
fn messages_contain_json_hint(messages: &[RequestMessage]) -> bool {
    messages.iter().any(|msg| {
        msg.content.iter().any(|block| match block {
            ContentBlock::Text(Text { text }) => text.to_lowercase().contains("json"),
            _ => false,
        })
    })
}

/// Converts a single TensorZero RequestMessage to xAI format
async fn tensorzero_to_xai_message<'a>(
    message: &'a RequestMessage,
    messages_config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<XAIRequestMessage<'a>>, Error> {
    match message.role {
        Role::User => tensorzero_to_xai_user_messages(&message.content, messages_config).await,
        Role::Assistant => {
            let msg =
                tensorzero_to_xai_assistant_message(&message.content, messages_config).await?;
            // Skip assistant messages without content or tool_calls.
            // xAI requires at least one content element, so reasoning_content alone is not enough.
            if msg.content.is_none() && msg.tool_calls.is_none() {
                Ok(vec![])
            } else {
                Ok(vec![XAIRequestMessage::Assistant(msg)])
            }
        }
    }
}

/// Converts user content blocks to xAI user messages
async fn tensorzero_to_xai_user_messages<'a>(
    content_blocks: &'a [ContentBlock],
    messages_config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<XAIRequestMessage<'a>>, Error> {
    let mut messages = Vec::new();
    let mut user_content_blocks = Vec::new();

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                let content_block = serde_json::json!({
                    "type": "text",
                    "text": text
                });
                user_content_blocks.push(content_block);
            }
            ContentBlock::ToolCall(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool calls are not supported in user messages".to_string(),
                }));
            }
            ContentBlock::ToolResult(tool_result) => {
                // First, flush any accumulated user content
                if !user_content_blocks.is_empty() {
                    messages.push(XAIRequestMessage::User(XAIUserRequestMessage {
                        content: Cow::Owned(user_content_blocks),
                    }));
                    user_content_blocks = Vec::new();
                }
                // Add tool result as a separate message
                messages.push(XAIRequestMessage::Tool(XAIToolRequestMessage {
                    content: Cow::Borrowed(&tool_result.result),
                    tool_call_id: Cow::Borrowed(&tool_result.id),
                }));
            }
            ContentBlock::File(file) => {
                let openai_block = prepare_file_message(file, messages_config).await?;
                let content_block = serde_json::to_value(&openai_block).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Error serializing file content block: {e}"),
                    })
                })?;
                user_content_blocks.push(content_block);
            }
            ContentBlock::Thought(thought) => {
                // Thought blocks in user messages are always discarded
                warn_discarded_thought_block(PROVIDER_TYPE, thought);
            }
            ContentBlock::Unknown(Unknown { data, .. }) => {
                user_content_blocks.push(data.clone());
            }
        }
    }

    // Flush remaining user content
    if !user_content_blocks.is_empty() {
        messages.push(XAIRequestMessage::User(XAIUserRequestMessage {
            content: Cow::Owned(user_content_blocks),
        }));
    }

    Ok(messages)
}

/// Converts assistant content blocks to xAI assistant message with reasoning_content support
async fn tensorzero_to_xai_assistant_message<'a>(
    content_blocks: &'a [ContentBlock],
    messages_config: OpenAIMessagesConfig<'a>,
) -> Result<XAIAssistantRequestMessage<'a>, Error> {
    let mut content_values = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning_content: Option<Cow<'a, str>> = None;

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                let content_block = serde_json::json!({
                    "type": "text",
                    "text": text
                });
                content_values.push(content_block);
            }
            ContentBlock::ToolCall(tool_call) => {
                let tool_call_json = serde_json::json!({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                });
                tool_calls.push(tool_call_json);
            }
            ContentBlock::ToolResult(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool results are not supported in assistant messages".to_string(),
                }));
            }
            ContentBlock::File(file) => {
                let openai_block = prepare_file_message(file, messages_config).await?;
                let content_block = serde_json::to_value(&openai_block).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Error serializing file content block: {e}"),
                    })
                })?;
                content_values.push(content_block);
            }
            ContentBlock::Thought(thought) => {
                // Thought blocks from other providers are already filtered at the model layer.
                // Extract reasoning text for multi-turn reasoning support.

                // INVARIANT: We should've filtered thought content blocks from other providers.
                debug_assert!(matches!(
                    thought.provider_type.as_deref(),
                    None | Some(PROVIDER_TYPE)
                ));

                if let Some(text) = &thought.text {
                    // Concatenate multiple thought blocks if present
                    match &mut reasoning_content {
                        Some(existing) => {
                            let combined = format!("{}\n\n{}", existing.as_ref(), text);
                            reasoning_content = Some(Cow::Owned(combined));
                        }
                        None => {
                            reasoning_content = Some(Cow::Borrowed(text));
                        }
                    }
                }
            }
            ContentBlock::Unknown(Unknown { data, .. }) => {
                content_values.push(data.clone());
            }
        }
    }

    let content = if content_values.is_empty() {
        None
    } else {
        Some(Cow::Owned(content_values))
    };

    let tool_calls = if tool_calls.is_empty() {
        None
    } else {
        Some(Cow::Owned(tool_calls))
    };

    Ok(XAIAssistantRequestMessage {
        content,
        tool_calls,
        reasoning_content,
    })
}

// =============================================================================
// Request Types
// =============================================================================

/// This struct defines the supported parameters for the xAI API
/// See the [xAI API documentation](https://docs.x.ai/api/endpoints#chat-completions)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// logit_bias, seed, service_tier, stop, user or response_format.
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(Default))]
struct XAIRequest<'a> {
    messages: Vec<XAIRequestMessage<'a>>,
    model: &'a str,

    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<XAIResponseFormat>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ChatCompletionTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ChatCompletionToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

fn apply_inference_params(
    request: &mut XAIRequest,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() {
        request.reasoning_effort.clone_from(reasoning_effort);
    }

    if service_tier.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier", None);
    }

    if thinking_budget_tokens.is_some() {
        warn_inference_parameter_not_supported(
            PROVIDER_NAME,
            "thinking_budget_tokens",
            Some("Tip: You might want to use `reasoning_effort` for this provider."),
        );
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
}

impl<'a> XAIRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<XAIRequest<'a>, Error> {
        let ModelInferenceRequest {
            temperature,
            max_tokens,
            seed,
            top_p,
            presence_penalty,
            frequency_penalty,
            stream,
            ..
        } = *request;

        let stream_options = if request.stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        };

        let response_format = XAIResponseFormat::new(request.json_mode, request.output_schema);

        let messages = prepare_xai_messages(
            request.system.as_deref(),
            &request.messages,
            OpenAIMessagesConfig {
                json_mode: Some(&request.json_mode),
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: request
                    .fetch_and_encode_input_files_before_inference,
            },
        )
        .await?;

        let (tools, tool_choice, parallel_tool_calls) =
            prepare_chat_completion_tools(request, false)?;
        let mut xai_request = XAIRequest {
            messages,
            model,
            temperature,
            max_tokens,
            seed,
            top_p,
            response_format,
            presence_penalty,
            frequency_penalty,
            stream,
            stream_options,
            tools,
            parallel_tool_calls,
            tool_choice,
            stop: request.borrow_stop_sequences(),
            reasoning_effort: None,
        };

        apply_inference_params(&mut xai_request, &request.inference_params_v2);

        Ok(xai_request)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum XAIResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: Value,
    },
}

impl XAIResponseFormat {
    fn new(
        json_mode: ModelInferenceRequestJsonMode,
        output_schema: Option<&Value>,
    ) -> Option<Self> {
        match json_mode {
            ModelInferenceRequestJsonMode::On => Some(XAIResponseFormat::JsonObject),
            // For now, we never explicitly send `XAIResponseFormat::Text`
            ModelInferenceRequestJsonMode::Off => None,
            ModelInferenceRequestJsonMode::Strict => match output_schema {
                Some(schema) => {
                    let json_schema = json!({"name": "response", "strict": true, "schema": schema});
                    Some(XAIResponseFormat::JsonSchema { json_schema })
                }
                None => Some(XAIResponseFormat::JsonObject),
            },
        }
    }
}

struct XAIResponseWithMetadata<'a> {
    response: XAIResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    model_inference_id: Uuid,
}

impl<'a> TryFrom<XAIResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: XAIResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let XAIResponseWithMetadata {
            mut response,
            latency,
            raw_request,
            generic_request,
            raw_response,
            model_inference_id,
        } = value;

        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices {}, Expected 1",
                    response.choices.len()
                ),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }

        let OpenAIResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
            }))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(reasoning) = message.reasoning_content {
            content.push(ContentBlockOutput::Thought(Thought {
                text: Some(reasoning),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
                extra_data: None,
            }));
        }
        if let Some(text) = message.content {
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        }

        let raw_usage = xai_usage_from_raw_response(&raw_response).map(|usage| {
            raw_usage_entries_from_value(
                model_inference_id,
                PROVIDER_TYPE,
                ApiType::ChatCompletions,
                usage,
            )
        });
        let usage = response.usage.into();
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response: raw_response.clone(),
                usage,
                raw_usage,
                relay_raw_response: None,
                provider_latency: latency,
                finish_reason: Some(finish_reason.into()),
                id: model_inference_id,
            },
        ))
    }
}

fn xai_usage_from_raw_response(raw_response: &str) -> Option<Value> {
    serde_json::from_str::<Value>(raw_response)
        .ok()
        .and_then(|value| value.get("usage").filter(|v| !v.is_null()).cloned())
}

/// Extracts reasoning_tokens from a raw JSON response's completion_tokens_details
fn extract_reasoning_tokens_from_raw(raw_response: &str) -> Option<u32> {
    serde_json::from_str::<Value>(raw_response)
        .ok()
        .and_then(|value| {
            value
                .get("usage")
                .and_then(|u| u.get("completion_tokens_details"))
                .and_then(|d| d.get("reasoning_tokens"))
                .and_then(|r| r.as_u64())
                .map(|r| r as u32)
        })
}

/// Wraps the OpenAI stream to add reasoning tokens to usage for xAI.
/// xAI reports reasoning_tokens separately in completion_tokens_details,
/// so we need to add them to output_tokens for accurate token counting.
fn stream_xai(
    inner_stream: ProviderInferenceResponseStreamInner,
) -> ProviderInferenceResponseStreamInner {
    Box::pin(
        inner_stream.map(|result: Result<ProviderInferenceResponseChunk, Error>| {
            result.map(|mut chunk| {
                // If this chunk has usage, check for reasoning tokens in raw_response
                if let Some(ref mut usage) = chunk.usage
                    && let Some(reasoning_tokens) =
                        extract_reasoning_tokens_from_raw(&chunk.raw_response)
                {
                    // Add reasoning tokens to output_tokens
                    usage.output_tokens = Some(usage.output_tokens.unwrap_or(0) + reasoning_tokens);
                }
                chunk
            })
        }),
    )
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;
    use uuid::Uuid;

    use super::*;

    use crate::inference::types::{
        FinishReason, FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };
    use crate::providers::chat_completions::{
        ChatCompletionSpecificToolChoice, ChatCompletionSpecificToolFunction,
        ChatCompletionToolChoice, ChatCompletionToolType,
    };
    use crate::providers::openai::{
        OpenAIFinishReason, OpenAIResponseChoice, OpenAIResponseMessage,
    };
    use crate::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};

    #[tokio::test]
    async fn test_xai_request_new() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let xai_request = XAIRequest::new("grok-beta", &request_with_tools)
            .await
            .expect("failed to create xAI Request during test");

        assert_eq!(xai_request.messages.len(), 1);
        assert_eq!(xai_request.temperature, Some(0.5));
        assert_eq!(xai_request.max_tokens, Some(100));
        assert!(!xai_request.stream);
        assert_eq!(xai_request.seed, Some(69));
        assert!(xai_request.tools.is_some());
        let tools = xai_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool = &tools[0];
        assert_eq!(tool.function.name, WEATHER_TOOL.name());
        assert_eq!(tool.function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            xai_request.tool_choice,
            Some(ChatCompletionToolChoice::Specific(
                ChatCompletionSpecificToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionSpecificToolFunction {
                        name: WEATHER_TOOL.name(),
                    }
                }
            ))
        );

        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Json,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let xai_request = XAIRequest::new("grok-beta", &request_with_tools)
            .await
            .expect("failed to create xAI Request");

        assert_eq!(xai_request.messages.len(), 2);
        assert_eq!(xai_request.temperature, Some(0.5));
        assert_eq!(xai_request.max_tokens, Some(100));
        assert_eq!(xai_request.top_p, Some(0.9));
        assert_eq!(xai_request.presence_penalty, Some(0.1));
        assert_eq!(xai_request.frequency_penalty, Some(0.2));
        assert!(!xai_request.stream);
        assert_eq!(xai_request.seed, Some(69));

        assert!(xai_request.tools.is_some());
        let tools = xai_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool = &tools[0];
        assert_eq!(tool.function.name, WEATHER_TOOL.name());
        assert_eq!(tool.function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            xai_request.tool_choice,
            Some(ChatCompletionToolChoice::Specific(
                ChatCompletionSpecificToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionSpecificToolFunction {
                        name: WEATHER_TOOL.name(),
                    }
                }
            ))
        );
    }

    #[test]
    fn test_xai_api_base() {
        assert_eq!(XAI_DEFAULT_BASE_URL.as_str(), "https://api.x.ai/v1");
    }

    #[test]
    fn test_credential_to_xai_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: XAICredentials = XAICredentials::try_from(generic).unwrap();
        assert!(matches!(creds, XAICredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = XAICredentials::try_from(generic).unwrap();
        assert!(matches!(creds, XAICredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = XAICredentials::try_from(generic).unwrap();
        assert!(matches!(creds, XAICredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = XAICredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
    #[tokio::test]
    async fn test_xai_response_with_metadata_try_into() {
        let valid_response = XAIResponse {
            choices: vec![OpenAIResponseChoice {
                index: 0,
                message: OpenAIResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: OpenAIFinishReason::Stop,
            }],
            usage: XAIUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                completion_tokens_details: None,
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let xai_response_with_metadata = XAIResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &XAIRequest::new("grok-beta", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            model_inference_id: Uuid::now_v7(),
        };
        let inference_response: ProviderInferenceResponse =
            xai_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 1);
        assert_eq!(
            inference_response.output[0],
            "Hello, world!".to_string().into()
        );
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(inference_response.raw_response, "test_response");
        assert_eq!(inference_response.usage.input_tokens, Some(10));
        assert_eq!(inference_response.usage.output_tokens, Some(20));
        assert_eq!(
            inference_response.provider_latency,
            Latency::NonStreaming {
                response_time: Duration::from_secs(0)
            }
        );
    }

    #[test]
    fn test_xai_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = XAIRequest::default();

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort is applied correctly
        assert_eq!(request.reasoning_effort, Some("high".to_string()));

        // Test that thinking_budget_tokens warns with tip about reasoning_effort
        assert!(logs_contain(
            "xAI does not support the inference parameter `thinking_budget_tokens`, so it will be ignored. Tip: You might want to use `reasoning_effort` for this provider."
        ));

        // Test that verbosity warns
        assert!(logs_contain(
            "xAI does not support the inference parameter `verbosity`, so it will be ignored."
        ));
    }

    #[tokio::test]
    async fn test_xai_assistant_message_with_reasoning_content() {
        // Test that Thought blocks with provider_type = "xai" are included as reasoning_content
        let content_blocks = vec![
            ContentBlock::Text(Text {
                text: "Hello, world!".to_string(),
            }),
            ContentBlock::Thought(Thought {
                text: Some("I'm thinking about this...".to_string()),
                signature: None,
                summary: None,
                provider_type: Some("xai".to_string()),
                extra_data: None,
            }),
        ];

        let messages_config = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };

        let result = tensorzero_to_xai_assistant_message(&content_blocks, messages_config)
            .await
            .expect("failed to convert assistant message");

        assert!(result.content.is_some(), "content should be present");
        assert!(
            result.reasoning_content.is_some(),
            "reasoning_content should be present"
        );
        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("I'm thinking about this...")
        );
    }

    #[tokio::test]
    async fn test_xai_assistant_message_with_no_provider_type_thought() {
        // Test that Thought blocks with provider_type = None are included as reasoning_content
        let content_blocks = vec![
            ContentBlock::Text(Text {
                text: "Response text".to_string(),
            }),
            ContentBlock::Thought(Thought {
                text: Some("Generic reasoning".to_string()),
                signature: None,
                summary: None,
                provider_type: None, // No provider type specified
                extra_data: None,
            }),
        ];

        let messages_config = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };

        let result = tensorzero_to_xai_assistant_message(&content_blocks, messages_config)
            .await
            .expect("failed to convert assistant message");

        assert!(
            result.reasoning_content.is_some(),
            "reasoning_content should be included for None provider_type"
        );
        assert_eq!(
            result.reasoning_content.as_deref(),
            Some("Generic reasoning")
        );
    }

    #[tokio::test]
    async fn test_xai_assistant_message_includes_all_thought_blocks() {
        // Thought blocks from other providers are filtered at the model layer,
        // so any Thought block that reaches the provider should be included.
        // This test verifies the provider includes all Thought blocks it receives.
        let content_blocks = vec![
            ContentBlock::Text(Text {
                text: "Response text".to_string(),
            }),
            ContentBlock::Thought(Thought {
                text: Some("Some reasoning".to_string()),
                signature: None,
                summary: None,
                provider_type: Some("xai".to_string()),
                extra_data: None,
            }),
        ];

        let messages_config = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };

        let result = tensorzero_to_xai_assistant_message(&content_blocks, messages_config)
            .await
            .expect("failed to convert assistant message");

        assert!(
            result.reasoning_content.is_some(),
            "reasoning_content should be included"
        );
        assert_eq!(result.reasoning_content.as_deref(), Some("Some reasoning"));
    }

    #[tokio::test]
    async fn test_xai_assistant_message_concatenates_multiple_thoughts() {
        // Test that multiple Thought blocks are concatenated
        let content_blocks = vec![
            ContentBlock::Thought(Thought {
                text: Some("First thought".to_string()),
                signature: None,
                summary: None,
                provider_type: Some("xai".to_string()),
                extra_data: None,
            }),
            ContentBlock::Thought(Thought {
                text: Some("Second thought".to_string()),
                signature: None,
                summary: None,
                provider_type: None, // Also should be included
                extra_data: None,
            }),
        ];

        let messages_config = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };

        let result = tensorzero_to_xai_assistant_message(&content_blocks, messages_config)
            .await
            .expect("failed to convert assistant message");

        assert!(
            result.reasoning_content.is_some(),
            "reasoning_content should be present"
        );
        let reasoning = result.reasoning_content.as_deref().unwrap();
        assert!(
            reasoning.contains("First thought"),
            "should contain first thought"
        );
        assert!(
            reasoning.contains("Second thought"),
            "should contain second thought"
        );
        assert!(
            reasoning.contains("\n\n"),
            "thoughts should be separated by double newline"
        );
    }

    #[test]
    fn test_xai_request_message_serialization() {
        // Test that XAIRequestMessage serializes correctly with reasoning_content
        let assistant_msg = XAIRequestMessage::Assistant(XAIAssistantRequestMessage {
            content: Some(Cow::Owned(vec![serde_json::json!({
                "type": "text",
                "text": "Hello"
            })])),
            tool_calls: None,
            reasoning_content: Some(Cow::Borrowed("I'm thinking...")),
        });

        let serialized =
            serde_json::to_string(&assistant_msg).expect("failed to serialize message");

        // Verify the serialization includes reasoning_content
        assert!(
            serialized.contains("reasoning_content"),
            "serialized message should contain reasoning_content"
        );
        assert!(
            serialized.contains("I'm thinking..."),
            "serialized message should contain reasoning text"
        );
        assert!(
            serialized.contains("\"role\":\"assistant\""),
            "serialized message should have assistant role"
        );
    }

    #[test]
    fn test_xai_request_message_serialization_without_reasoning() {
        // Test that XAIRequestMessage omits reasoning_content when None
        let assistant_msg = XAIRequestMessage::Assistant(XAIAssistantRequestMessage {
            content: Some(Cow::Owned(vec![serde_json::json!({
                "type": "text",
                "text": "Hello"
            })])),
            tool_calls: None,
            reasoning_content: None,
        });

        let serialized =
            serde_json::to_string(&assistant_msg).expect("failed to serialize message");

        // Verify reasoning_content is omitted when None
        assert!(
            !serialized.contains("reasoning_content"),
            "serialized message should not contain reasoning_content when None"
        );
    }

    #[tokio::test]
    async fn test_prepare_xai_messages_json_hint_in_system_not_duplicated() {
        // Test that JSON instruction is NOT prepended when system already contains "json"
        let system = Some("Output as JSON format");
        let messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["What's the weather?".to_string().into()],
        }];

        let messages_config = OpenAIMessagesConfig {
            json_mode: Some(&ModelInferenceRequestJsonMode::On),
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };

        let result = prepare_xai_messages(system, &messages, messages_config)
            .await
            .expect("failed to prepare messages");

        // Find the system message
        let system_msg = result
            .iter()
            .find_map(|msg| {
                if let XAIRequestMessage::System(sys) = msg {
                    Some(sys.content.as_ref())
                } else {
                    None
                }
            })
            .expect("system message should be present");

        // Should NOT have the "Respond using JSON." prefix since system already mentions JSON
        assert!(
            !system_msg.starts_with("Respond using JSON."),
            "JSON instruction should not be prepended when system already contains 'json'. Got: {system_msg}"
        );
        assert_eq!(
            system_msg, "Output as JSON format",
            "system content should be unchanged"
        );
    }

    #[tokio::test]
    async fn test_prepare_xai_messages_json_hint_added_when_missing() {
        // Test that JSON instruction IS prepended when neither system nor messages contain "json"
        let system = Some("You are a helpful assistant.");
        let messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["What's the weather?".to_string().into()],
        }];

        let messages_config = OpenAIMessagesConfig {
            json_mode: Some(&ModelInferenceRequestJsonMode::On),
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };

        let result = prepare_xai_messages(system, &messages, messages_config)
            .await
            .expect("failed to prepare messages");

        // Find the system message
        let system_msg = result
            .iter()
            .find_map(|msg| {
                if let XAIRequestMessage::System(sys) = msg {
                    Some(sys.content.as_ref())
                } else {
                    None
                }
            })
            .expect("system message should be present");

        // Should have the "Respond using JSON." prefix since no JSON hint exists
        assert!(
            system_msg.starts_with("Respond using JSON."),
            "JSON instruction should be prepended when no JSON hint exists. Got: {system_msg}"
        );
    }
}
