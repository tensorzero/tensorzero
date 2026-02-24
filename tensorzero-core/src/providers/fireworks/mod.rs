use std::borrow::Cow;

use crate::error::warn_discarded_thought_block;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported,
};
use crate::inference::types::{ContentBlock, Role};
use crate::providers::openai::OpenAIMessagesConfig;
use crate::{
    http::TensorZeroEventSource, providers::helpers_thinking_block::REASONING_FIELD_CHUNK_ID,
    tool::FunctionTool,
};
use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_sse_stream::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tensorzero_types_providers::fireworks::*;
use tokio::time::Instant;
use url::Url;

use super::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};
use crate::inference::types::usage::raw_usage_entries_from_value;
use crate::{
    cache::ModelProviderRequest,
    endpoints::inference::InferenceCredentials,
    error::{DelayedError, DisplayOrDebugGateway, Error, ErrorDetails},
    inference::{
        InferenceProvider,
        types::{
            ApiType, ContentBlockChunk, ContentBlockOutput, FinishReason, Latency,
            ModelInferenceRequest, ModelInferenceRequestJsonMode,
            PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
            ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
            ProviderInferenceResponseStreamInner, RequestMessage, Text, TextChunk, Thought,
            ThoughtChunk,
            batch::{
                BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
            },
        },
    },
    model::{Credential, ModelProvider},
    tool::{FunctionToolConfig, ToolCall, ToolCallChunk},
};
use uuid::Uuid;

use super::{
    helpers_thinking_block::{ThinkingState, process_think_blocks},
    openai::{
        OpenAIAssistantRequestMessage, OpenAIContentBlock, OpenAIFunction,
        OpenAIRequestFunctionCall, OpenAIRequestMessage, OpenAIRequestToolCall,
        OpenAISystemRequestMessage, OpenAIToolChoice, OpenAIToolType, get_chat_url,
        handle_openai_error, tensorzero_to_openai_messages,
    },
};

lazy_static! {
    pub static ref FIREWORKS_API_BASE: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.fireworks.ai/").expect("Failed to parse FIREWORKS_API_BASE")
    };
    pub static ref FIREWORKS_API_INFERENCE_BASE: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.fireworks.ai/inference/v1/")
            .expect("Failed to parse FIREWORKS_API_INFERENCE_BASE")
    };
}

pub const PROVIDER_NAME: &str = "Fireworks";
pub const PROVIDER_TYPE: &str = "fireworks";

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct FireworksProvider {
    model_name: String,
    #[serde(skip)]
    credentials: FireworksCredentials,
}

impl FireworksProvider {
    pub fn new(model_name: String, credentials: FireworksCredentials) -> Self {
        FireworksProvider {
            model_name,
            credentials,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum FireworksCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<FireworksCredentials>,
        fallback: Box<FireworksCredentials>,
    },
}

impl TryFrom<Credential> for FireworksCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(FireworksCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(FireworksCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(FireworksCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(FireworksCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Fireworks provider".to_string(),
            })),
        }
    }
}

impl FireworksCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            FireworksCredentials::Static(api_key) => Ok(api_key),
            FireworksCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            FireworksCredentials::WithFallback { default, fallback } => {
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
            &FireworksCredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
        }
    }
}

/// Key differences between Fireworks and OpenAI inference:
/// - Fireworks allows you to specify output format in JSON mode
/// - Fireworks automatically returns usage in streaming inference, we don't have to ask for it
/// - Fireworks allows you to auto-truncate requests that have too many tokens
///   (there are 2 ways to do it, we have the default of auto-truncation to the max window size)
impl InferenceProvider for FireworksProvider {
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
        api_key: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(
            FireworksRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Fireworks request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let request_url = get_chat_url(&FIREWORKS_API_INFERENCE_BASE)?;
        let start_time = Instant::now();
        let api_key = self.credentials.get_api_key(api_key).map_err(|e| e.log())?;
        let builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());
        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            ApiType::ChatCompletions,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            builder,
        )
        .await?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    api_type: ApiType::ChatCompletions,
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

            let response: FireworksResponse = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {raw_response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    api_type: ApiType::ChatCompletions,
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            Ok(FireworksResponseWithMetadata {
                response,
                latency,
                raw_request,
                generic_request: request,
                raw_response,
                model_inference_id,
            }
            .try_into()?)
        } else {
            Err(handle_openai_error(
                &raw_request,
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: PROVIDER_TYPE.to_string(),
                        api_type: ApiType::ChatCompletions,
                        raw_request: Some(raw_request.clone()),
                        raw_response: None,
                    })
                })?,
                PROVIDER_TYPE,
                None,
                ApiType::ChatCompletions,
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
        api_key: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(
            FireworksRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Fireworks request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let request_url = get_chat_url(&FIREWORKS_API_INFERENCE_BASE)?;
        let api_key = self.credentials.get_api_key(api_key).map_err(|e| e.log())?;
        let start_time = Instant::now();
        let builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());
        let (event_source, raw_request) = inject_extra_request_data_and_send_eventsource(
            PROVIDER_TYPE,
            ApiType::ChatCompletions,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            builder,
        )
        .await?;
        // Use our own stream implementation to handle thinking blocks
        let stream = stream_fireworks(event_source, start_time, model_inference_id).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Fireworks".to_string(),
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

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum FireworksResponseFormat<'a> {
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<&'a Value>, // the desired JSON schema
    },
    #[default]
    Text,
}

/// This struct defines the supported parameters for the Fireworks inference API
/// See the [Fireworks API documentation](https://docs.fireworks.ai/api-reference/post-chatcompletions)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, service_tier, stop, user,
/// or context_length_exceeded_behavior.
/// NOTE: Fireworks does not support seed.
#[derive(Debug, Default, Serialize)]
struct FireworksRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<FireworksResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<FireworksTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

type PreparedFireworksToolsResult<'a> = (
    Option<Vec<FireworksTool<'a>>>,
    Option<OpenAIToolChoice<'a>>,
    Option<bool>,
);

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to Fireworks format
pub(super) fn prepare_fireworks_tools<'a>(
    request: &'a ModelInferenceRequest,
) -> Result<PreparedFireworksToolsResult<'a>, Error> {
    match &request.tool_config {
        None => Ok((None, None, None)),
        Some(tool_config) => {
            if !tool_config.any_tools_available() {
                return Ok((None, None, None));
            }
            let tools = Some(
                tool_config
                    .strict_tools_available()?
                    .map(Into::into)
                    .collect(),
            );
            let parallel_tool_calls = tool_config.parallel_tool_calls;

            // Fireworks does not support allowed_tools constraint, use regular tool_choice
            let tool_choice = Some((&tool_config.tool_choice).into());
            Ok((tools, tool_choice, parallel_tool_calls))
        }
    }
}

fn apply_inference_params(
    request: &mut FireworksRequest,
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

impl<'a> FireworksRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<FireworksRequest<'a>, Error> {
        // NB: Fireworks will throw an error if you give FireworksResponseFormat::Text and then also include tools.
        // So we just don't include it as Text is the same as None anyway.
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(FireworksResponseFormat::JsonObject {
                    schema: request.output_schema,
                })
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_fireworks_messages(
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
        let (tools, tool_choice, _) = prepare_fireworks_tools(request)?;

        let mut fireworks_request = FireworksRequest {
            messages,
            model,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            stop: request.borrow_stop_sequences(),
            max_tokens: request.max_tokens,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
            reasoning_effort: None,
        };

        apply_inference_params(&mut fireworks_request, &request.inference_params_v2);

        Ok(fireworks_request)
    }
}

pub async fn prepare_fireworks_messages<'a>(
    system: Option<&'a str>,
    messages: &'a [RequestMessage],
    config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    // Convert all messages concurrently, then assemble in order.
    enum ConvertedMessage<'a> {
        Assistant(OpenAIRequestMessage<'a>),
        Other(Vec<OpenAIRequestMessage<'a>>),
    }

    let conversion_futures: Vec<_> = messages
        .iter()
        .map(|msg| async move {
            match msg.role {
                Role::Assistant => {
                    let m = tensorzero_to_fireworks_assistant_message(
                        Cow::Borrowed(&msg.content),
                        config,
                    )
                    .await?;
                    Ok(ConvertedMessage::Assistant(m))
                }
                Role::User => {
                    let openai_msgs = tensorzero_to_openai_messages(msg, config).await?;
                    Ok(ConvertedMessage::Other(openai_msgs))
                }
            }
        })
        .collect();

    let converted: Vec<Result<ConvertedMessage<'a>, Error>> =
        futures::future::join_all(conversion_futures).await;

    let mut output_messages: Vec<OpenAIRequestMessage<'a>> = Vec::new();

    if let Some(system_msg) = tensorzero_to_fireworks_system_message(system) {
        output_messages.push(system_msg);
    }

    for result in converted {
        match result? {
            ConvertedMessage::Assistant(msg) => {
                if !msg.no_content() {
                    output_messages.push(msg);
                }
            }
            ConvertedMessage::Other(msgs) => {
                output_messages.extend(msgs);
            }
        }
    }

    Ok(output_messages)
}

/// Converts assistant content blocks to a Fireworks-compatible request message.
///
/// Dispatches thought blocks based on `extra_data.reasoning_format`:
/// - `"reasoning_field"` or `None` → accumulated into the `reasoning_content` field
/// - `"think_tags"` → wrapped in `<think>…</think>` and prepended to content blocks
pub async fn tensorzero_to_fireworks_assistant_message<'a>(
    content_blocks: Cow<'a, [ContentBlock]>,
    messages_config: OpenAIMessagesConfig<'a>,
) -> Result<OpenAIRequestMessage<'a>, Error> {
    let content_block_cows: Vec<Cow<'_, ContentBlock>> = match content_blocks {
        Cow::Borrowed(content_blocks) => content_blocks.iter().map(Cow::Borrowed).collect(),
        Cow::Owned(content_blocks) => content_blocks.into_iter().map(Cow::Owned).collect(),
    };

    let mut assistant_content_blocks = Vec::new();
    let mut assistant_tool_calls = Vec::new();
    let mut reasoning_content: Option<Cow<'_, str>> = None;
    let mut think_tag_texts: Vec<String> = Vec::new();
    let mut thought_blocks_for_discard: Vec<Cow<'_, Thought>> = Vec::new();

    for block in content_block_cows {
        match block {
            Cow::Borrowed(ContentBlock::Text(tensorzero_types::content::Text { text })) => {
                assistant_content_blocks.push(OpenAIContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            Cow::Owned(ContentBlock::Text(tensorzero_types::content::Text { text })) => {
                assistant_content_blocks.push(OpenAIContentBlock::Text {
                    text: Cow::Owned(text),
                });
            }
            Cow::Borrowed(ContentBlock::ToolCall(tool_call)) => {
                assistant_tool_calls.push(OpenAIRequestToolCall {
                    id: Cow::Borrowed(&tool_call.id),
                    r#type: OpenAIToolType::Function,
                    function: OpenAIRequestFunctionCall {
                        name: Cow::Borrowed(&tool_call.name),
                        arguments: Cow::Borrowed(&tool_call.arguments),
                    },
                });
            }
            Cow::Owned(ContentBlock::ToolCall(tool_call)) => {
                assistant_tool_calls.push(OpenAIRequestToolCall {
                    id: Cow::Owned(tool_call.id),
                    r#type: OpenAIToolType::Function,
                    function: OpenAIRequestFunctionCall {
                        name: Cow::Owned(tool_call.name),
                        arguments: Cow::Owned(tool_call.arguments),
                    },
                });
            }
            Cow::Borrowed(ContentBlock::ToolResult(_))
            | Cow::Owned(ContentBlock::ToolResult(_)) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool results are not supported in assistant messages".to_string(),
                }));
            }
            Cow::Borrowed(ContentBlock::File(file)) => {
                assistant_content_blocks
                    .push(super::openai::prepare_file_message(file, messages_config).await?);
            }
            Cow::Owned(ContentBlock::File(ref file)) => {
                assistant_content_blocks
                    .push(super::openai::prepare_file_message(file, messages_config).await?);
            }
            Cow::Borrowed(ContentBlock::Thought(thought)) => {
                debug_assert!(
                    thought.provider_type.as_deref().is_none()
                        || thought.provider_type.as_deref() == Some(messages_config.provider_type)
                );

                let format = thought
                    .extra_data
                    .as_ref()
                    .and_then(|d| d.get("reasoning_format"))
                    .and_then(|v| v.as_str());

                match format {
                    Some("think_tags") => {
                        if let Some(text) = &thought.text {
                            think_tag_texts.push(text.clone());
                        }
                    }
                    Some("reasoning_field") | None => {
                        if let Some(text) = &thought.text {
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
                    Some(other) => {
                        return Err(Error::new(ErrorDetails::InvalidMessage {
                            message: format!(
                                "Unknown `reasoning_format` value in thought `extra_data`: `{other}`"
                            ),
                        }));
                    }
                }

                thought_blocks_for_discard.push(Cow::Borrowed(thought));
            }
            Cow::Owned(ContentBlock::Thought(thought)) => {
                debug_assert!(
                    thought.provider_type.as_deref().is_none()
                        || thought.provider_type.as_deref() == Some(messages_config.provider_type)
                );

                let format = thought
                    .extra_data
                    .as_ref()
                    .and_then(|d| d.get("reasoning_format"))
                    .and_then(|v| v.as_str());

                match format {
                    Some("think_tags") => {
                        if let Some(text) = &thought.text {
                            think_tag_texts.push(text.clone());
                        }
                    }
                    Some("reasoning_field") | None => {
                        if let Some(text) = &thought.text {
                            match &mut reasoning_content {
                                Some(existing) => {
                                    let combined = format!("{}\n\n{}", existing.as_ref(), text);
                                    reasoning_content = Some(Cow::Owned(combined));
                                }
                                None => {
                                    reasoning_content = Some(Cow::Owned(text.clone()));
                                }
                            }
                        }
                    }
                    Some(other) => {
                        return Err(Error::new(ErrorDetails::InvalidMessage {
                            message: format!(
                                "Unknown `reasoning_format` value in thought `extra_data`: `{other}`"
                            ),
                        }));
                    }
                }

                thought_blocks_for_discard.push(Cow::Owned(thought));
            }
            Cow::Borrowed(ContentBlock::Unknown(tensorzero_types::content::Unknown {
                data,
                ..
            })) => {
                assistant_content_blocks.push(OpenAIContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
            Cow::Owned(ContentBlock::Unknown(tensorzero_types::content::Unknown {
                data, ..
            })) => {
                assistant_content_blocks.push(OpenAIContentBlock::Unknown {
                    data: Cow::Owned(data),
                });
            }
        }
    }

    // Prepend think-tag reasoning to content blocks
    if !think_tag_texts.is_empty() {
        let combined = think_tag_texts.join("\n\n");
        let think_block = format!("<think>{combined}</think>");
        assistant_content_blocks.insert(
            0,
            OpenAIContentBlock::Text {
                text: Cow::Owned(think_block),
            },
        );
    }

    let content = match assistant_content_blocks.len() {
        0 => None,
        _ => Some(assistant_content_blocks),
    };

    let tool_calls = match assistant_tool_calls.len() {
        0 => None,
        _ => Some(assistant_tool_calls),
    };

    // Most providers require at least one content element or tool call in
    // assistant messages, so reasoning_content alone is not enough.
    // Drop reasoning_content and warn when there's nothing else to carry it.
    if content.is_none() && tool_calls.is_none() {
        for thought in &thought_blocks_for_discard {
            warn_discarded_thought_block(messages_config.provider_type, thought);
        }
    }
    let reasoning_content = if content.is_some() || tool_calls.is_some() {
        reasoning_content
    } else {
        None
    };

    Ok(OpenAIRequestMessage::Assistant(
        OpenAIAssistantRequestMessage {
            content,
            tool_calls,
            reasoning_content,
        },
    ))
}

fn tensorzero_to_fireworks_system_message(
    system: Option<&str>,
) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

#[derive(Debug, PartialEq, Serialize)]
pub struct FireworksTool<'a> {
    r#type: OpenAIToolType,
    function: OpenAIFunction<'a>,
}

impl<'a> From<&'a FunctionTool> for FireworksTool<'a> {
    fn from(tool: &'a FunctionTool) -> Self {
        FireworksTool {
            r#type: OpenAIToolType::Function,
            function: OpenAIFunction {
                name: &tool.name,
                description: Some(&tool.description),
                parameters: &tool.parameters,
            },
        }
    }
}

impl<'a> From<&'a FunctionToolConfig> for FireworksTool<'a> {
    fn from(tool: &'a FunctionToolConfig) -> Self {
        FireworksTool {
            r#type: OpenAIToolType::Function,
            function: OpenAIFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
        }
    }
}

fn fireworks_tool_call_to_tensorzero(fireworks_tool_call: FireworksResponseToolCall) -> ToolCall {
    ToolCall {
        id: fireworks_tool_call.id,
        name: fireworks_tool_call.function.name,
        arguments: fireworks_tool_call.function.arguments,
    }
}

impl From<FireworksFinishReason> for FinishReason {
    fn from(reason: FireworksFinishReason) -> Self {
        match reason {
            FireworksFinishReason::Stop => FinishReason::Stop,
            FireworksFinishReason::Length => FinishReason::Length,
            FireworksFinishReason::ToolCalls => FinishReason::ToolCall,
            FireworksFinishReason::ContentFilter => FinishReason::ContentFilter,
            FireworksFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

/// Streams the Fireworks response events and converts them into ProviderInferenceResponseChunks
/// This function handles parsing and processing of thinking blocks with proper state tracking
fn stream_fireworks(
    mut event_source: TensorZeroEventSource,
    start_time: Instant,
    model_inference_id: Uuid,
) -> ProviderInferenceResponseStreamInner {
    let mut tool_call_ids = Vec::new();
    let mut thinking_state = ThinkingState::Normal;
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    let message = e.to_string();
                    let mut raw_response = None;
                    if let reqwest_sse_stream::ReqwestSseStreamError::InvalidStatusCode(_, resp) = *e {
                        raw_response = resp.text().await.ok();
                    }
                    yield Err(ErrorDetails::InferenceServer {
                        message,
                        raw_request: None,
                        raw_response,
                        provider_type: PROVIDER_TYPE.to_string(),
                        api_type: ApiType::ChatCompletions,
                    }.into());
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<FireworksChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!("Error parsing chunk. Error: {e}"),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                                api_type: ApiType::ChatCompletions,
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            fireworks_to_tensorzero_chunk(
                                message.data,
                                d,
                                latency,
                                &mut tool_call_ids,
                                &mut thinking_state,
                                model_inference_id,
                                PROVIDER_TYPE,
                            )
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

/// Maps a Fireworks chunk to a TensorZero chunk for streaming inferences
///
/// This function handles the conversion of Fireworks chat chunks into TensorZero chunks.
/// It processes the content and tool calls from the Fireworks response, updating the tool call IDs and names.
/// Think blocks in content are always parsed; reasoning from the `reasoning_content` field
/// is tagged with `reasoning_format: "reasoning_field"`, while `<think>` tags are tagged
/// with `reasoning_format: "think_tags"`.
fn fireworks_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: FireworksChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    thinking_state: &mut ThinkingState,
    model_inference_id: Uuid,
    provider_type: &str,
) -> Result<ProviderInferenceResponseChunk, Error> {
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            raw_request: None,
            raw_response: Some(serde_json::to_string(&chunk).unwrap_or_default()),
            provider_type: PROVIDER_TYPE.to_string(),
            api_type: ApiType::ChatCompletions,
        }
        .into());
    }
    let raw_usage = fireworks_usage_from_raw_response(&raw_message).map(|usage| {
        raw_usage_entries_from_value(
            model_inference_id,
            provider_type,
            ApiType::ChatCompletions,
            usage,
        )
    });
    let usage = chunk.usage.map(|u| u.into());
    let mut finish_reason = None;
    let mut content = vec![];
    if let Some(choice) = chunk.choices.pop() {
        if let Some(reason) = choice.finish_reason {
            finish_reason = Some(reason.into());
        }
        if let Some(reasoning) = choice.delta.reasoning_content {
            content.push(ContentBlockChunk::Thought(ThoughtChunk {
                text: Some(reasoning),
                signature: None,
                summary_id: None,
                summary_text: None,
                id: REASONING_FIELD_CHUNK_ID.to_string(),
                provider_type: Some(PROVIDER_TYPE.to_string()),
                extra_data: Some(serde_json::json!({"reasoning_format": "reasoning_field"})),
            }));
        }
        if let Some(text) = choice.delta.content
            && !thinking_state.update(&text, PROVIDER_TYPE, ApiType::ChatCompletions)?
        {
            match thinking_state {
                ThinkingState::Normal | ThinkingState::Finished => {
                    content.push(ContentBlockChunk::Text(TextChunk {
                        text: text.to_string(),
                        id: thinking_state.get_id(),
                    }));
                }
                ThinkingState::Thinking => {
                    content.push(ContentBlockChunk::Thought(ThoughtChunk {
                        text: Some(text.to_string()),
                        signature: None,
                        summary_id: None,
                        summary_text: None,
                        id: thinking_state.get_id(),
                        provider_type: Some(PROVIDER_TYPE.to_string()),
                        extra_data: Some(serde_json::json!({"reasoning_format": "think_tags"})),
                    }));
                }
            }
        }
        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                let index = tool_call.index;
                let id = match tool_call.id {
                    Some(id) => {
                        tool_call_ids.push(id.clone());
                        id
                    }
                    None => {
                        tool_call_ids
                            .get(index as usize)
                            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                                message: "Tool call index out of bounds (meaning we haven't seen this many ids in the stream)".to_string(),
                                raw_request: None,
                                raw_response: None,
                                provider_type: PROVIDER_TYPE.to_string(),
                                api_type: ApiType::ChatCompletions,
                            }))?
                            .clone()
                    }
                };

                content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                    id,
                    raw_name: tool_call.function.name,
                    raw_arguments: tool_call.function.arguments.unwrap_or_default(),
                }));
            }
        }
    }

    Ok(match raw_usage {
        Some(entries) => ProviderInferenceResponseChunk::new_with_raw_usage(
            content,
            usage,
            raw_message,
            latency,
            finish_reason,
            Some(entries),
        ),
        None => {
            ProviderInferenceResponseChunk::new(content, usage, raw_message, latency, finish_reason)
        }
    })
}

struct FireworksResponseWithMetadata<'a> {
    response: FireworksResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    model_inference_id: Uuid,
}

impl<'a> TryFrom<FireworksResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: FireworksResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let FireworksResponseWithMetadata {
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
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                api_type: ApiType::ChatCompletions,
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
            }
            .into());
        }
        let FireworksResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                api_type: ApiType::ChatCompletions,
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
            }
            ))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(reasoning) = message.reasoning_content {
            content.push(ContentBlockOutput::Thought(Thought {
                text: Some(reasoning),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
                extra_data: Some(serde_json::json!({"reasoning_format": "reasoning_field"})),
            }));
        }
        if let Some(raw_text) = message.content {
            let (clean_text, extracted_reasoning) =
                process_think_blocks(&raw_text, true, PROVIDER_TYPE, ApiType::ChatCompletions)?;
            if let Some(reasoning) = extracted_reasoning {
                content.push(ContentBlockOutput::Thought(Thought {
                    text: Some(reasoning),
                    signature: None,
                    summary: None,
                    provider_type: Some(PROVIDER_TYPE.to_string()),
                    extra_data: Some(serde_json::json!({"reasoning_format": "think_tags"})),
                }));
            }
            if !clean_text.is_empty() {
                content.push(ContentBlockOutput::Text(Text { text: clean_text }));
            }
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(
                    fireworks_tool_call_to_tensorzero(tool_call),
                ));
            }
        }
        let raw_usage = fireworks_usage_from_raw_response(&raw_response).map(|usage| {
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
                raw_response,
                usage,
                raw_usage,
                relay_raw_response: None,
                provider_latency: latency,
                finish_reason: finish_reason.map(FireworksFinishReason::into),
                id: model_inference_id,
            },
        ))
    }
}

fn fireworks_usage_from_raw_response(raw_response: &str) -> Option<Value> {
    serde_json::from_str::<Value>(raw_response)
        .ok()
        .and_then(|value| value.get("usage").filter(|v| !v.is_null()).cloned())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;
    use tensorzero_types_providers::openai::OpenAIUsage;
    use uuid::Uuid;

    use super::*;

    use crate::inference::types::{FunctionType, RequestMessage, Role, Usage};
    use crate::providers::openai::OpenAIToolType;
    use crate::providers::openai::{SpecificToolChoice, SpecificToolFunction};
    use crate::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};

    #[tokio::test]
    async fn test_fireworks_response_with_thinking_blocks() {
        let test_response_with_thinking = "Hello <think>This is reasoning</think> world";
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

        // Create a valid response with thinking blocks in the content
        let valid_response = FireworksResponse {
            choices: vec![FireworksResponseChoice {
                index: 0,
                finish_reason: Some(FireworksFinishReason::Stop),
                message: FireworksResponseMessage {
                    reasoning_content: None,
                    content: Some(test_response_with_thinking.to_string()),
                    tool_calls: None,
                },
            }],
            usage: OpenAIUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
            },
        };

        // Think blocks are always parsed now
        let fireworks_response_with_metadata = FireworksResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &FireworksRequest::new("test-model", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            model_inference_id: Uuid::now_v7(),
        };

        let inference_response: ProviderInferenceResponse =
            fireworks_response_with_metadata.try_into().unwrap();

        // Should have two content blocks: thought and text
        assert_eq!(
            inference_response.output.len(),
            2,
            "expected thought + text blocks"
        );

        // First block should be a thought with think_tags format
        match &inference_response.output[0] {
            ContentBlockOutput::Thought(thought) => {
                assert_eq!(thought.text, Some("This is reasoning".to_string()));
                assert_eq!(thought.signature, None);
                assert_eq!(
                    thought.extra_data,
                    Some(serde_json::json!({"reasoning_format": "think_tags"})),
                    "expected think_tags reasoning_format"
                );
            }
            _ => panic!("Expected a thought block"),
        }

        // Second block should be text
        match &inference_response.output[1] {
            ContentBlockOutput::Text(text) => {
                assert_eq!(text.text, "Hello  world".to_string());
            }
            _ => panic!("Expected a text block"),
        }
    }

    #[tokio::test]
    async fn test_fireworks_request_new() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let fireworks_request =
            FireworksRequest::new("accounts/fireworks/models/llama-v3-8b", &request_with_tools)
                .await
                .unwrap();

        assert_eq!(
            fireworks_request.model,
            "accounts/fireworks/models/llama-v3-8b"
        );
        assert_eq!(fireworks_request.messages.len(), 1);
        assert_eq!(fireworks_request.temperature, Some(0.5));
        assert_eq!(fireworks_request.max_tokens, Some(100));
        assert_eq!(fireworks_request.top_p, Some(0.9));
        assert_eq!(fireworks_request.presence_penalty, Some(0.1));
        assert_eq!(fireworks_request.frequency_penalty, Some(0.2));
        assert!(!fireworks_request.stream);
        assert_eq!(
            fireworks_request.response_format,
            Some(FireworksResponseFormat::JsonObject {
                schema: request_with_tools.output_schema,
            })
        );
        assert!(fireworks_request.tools.is_some());
        let tools = fireworks_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            fireworks_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );
    }

    #[tokio::test]
    async fn test_fireworks_api_base() {
        assert_eq!(
            FIREWORKS_API_INFERENCE_BASE.as_str(),
            "https://api.fireworks.ai/inference/v1/"
        );
    }

    #[tokio::test]
    async fn test_credential_to_fireworks_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = FireworksCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, FireworksCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = FireworksCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, FireworksCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = FireworksCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, FireworksCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = FireworksCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[tokio::test]
    async fn test_fireworks_response_with_metadata_try_into() {
        let valid_response = FireworksResponse {
            choices: vec![FireworksResponseChoice {
                index: 0,
                finish_reason: Some(FireworksFinishReason::Stop),
                message: FireworksResponseMessage {
                    reasoning_content: None,
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
            }],
            usage: OpenAIUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
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
        let fireworks_response_with_metadata = FireworksResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &FireworksRequest::new("test-model", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            model_inference_id: Uuid::now_v7(),
        };
        let inference_response: ProviderInferenceResponse =
            fireworks_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 1);
        assert_eq!(
            inference_response.output[0],
            "Hello, world!".to_string().into()
        );
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

    #[tokio::test]
    async fn test_fireworks_to_tensorzero_chunk() {
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("Hello".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: Some(FireworksFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let mut thinking_state = ThinkingState::Normal;
        let message = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        )
        .unwrap();

        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "Hello".to_string(),
                id: "0".to_string(),
            })]
        );
        assert_eq!(message.finish_reason, Some(FinishReason::Stop));

        // Test what an intermediate tool chunk should look like
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![FireworksToolCallChunk {
                        index: 0,
                        id: None,
                        function: FireworksFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(FireworksFinishReason::ToolCalls),
            }],
            usage: None,
        };
        let message = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        )
        .unwrap();

        assert_eq!(
            message.content,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "id1".to_string(),
                raw_name: None,
                raw_arguments: "{\"hello\":\"world\"}".to_string(),
            })]
        );
        assert_eq!(message.finish_reason, Some(FinishReason::ToolCall));

        // Test a chunk with no choices and only usage
        let usage = OpenAIUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
        };
        let chunk = FireworksChatChunk {
            choices: vec![],
            usage: Some(usage.clone()),
        };
        let model_inference_id = Uuid::now_v7();
        let raw_message = serde_json::json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_details": {
                    "cached_tokens": 2
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 1
                }
            }
        })
        .to_string();
        let message = fireworks_to_tensorzero_chunk(
            raw_message,
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            model_inference_id,
            PROVIDER_TYPE,
        )
        .unwrap();

        let expected_raw_usage = Some(raw_usage_entries_from_value(
            model_inference_id,
            PROVIDER_TYPE,
            ApiType::ChatCompletions,
            serde_json::json!({
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_details": {
                    "cached_tokens": 2
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 1
                }
            }),
        ));
        assert_eq!(message.content, vec![]);
        assert_eq!(
            message.usage,
            Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
                cost: None,
            }),
            "expected usage to include provider raw_usage entries"
        );
        assert_eq!(
            message.raw_usage, expected_raw_usage,
            "expected raw_usage to include provider raw_usage entries"
        );
    }

    #[tokio::test]
    async fn test_fireworks_to_tensorzero_chunk_thinking() {
        // Test that the streaming function correctly handles thinking blocks
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("<think>".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let mut tool_call_ids = Vec::new();
        let mut thinking_state = ThinkingState::Normal;

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        )
        .unwrap();

        // Should transition to Thinking state
        assert!(matches!(thinking_state, ThinkingState::Thinking));
        // No content should be added for the opening tag
        assert!(result.content.is_empty());

        // Now process some thinking content
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("reasoning".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        )
        .unwrap();

        // Should still be in Thinking state
        assert!(matches!(thinking_state, ThinkingState::Thinking));
        // Content should be added as thought with think_tags format
        assert_eq!(result.content.len(), 1, "expected one thought chunk");
        if let ContentBlockChunk::Thought(thought) = &result.content[0] {
            assert_eq!(thought.text, Some("reasoning".to_string()));
            assert_eq!(thought.id, "1");
            assert_eq!(
                thought.extra_data,
                Some(serde_json::json!({"reasoning_format": "think_tags"})),
                "expected think_tags reasoning_format on streaming thought"
            );
        } else {
            panic!("Expected a thought chunk");
        }

        // Close the thinking block
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("</think>".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        )
        .unwrap();

        // Should transition to Finished state
        assert!(matches!(thinking_state, ThinkingState::Finished));
        // No content should be added for the closing tag
        assert!(result.content.is_empty());

        // After closing, regular text should be treated as text content
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: Some("Final answer".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        )
        .unwrap();

        // Should remain in Finished state
        assert!(matches!(thinking_state, ThinkingState::Finished));
        // Content should be added as text
        assert_eq!(result.content.len(), 1);
        if let ContentBlockChunk::Text(text) = &result.content[0] {
            assert_eq!(text.text, "Final answer");
            assert_eq!(text.id, "2");
        } else {
            panic!("Expected a text chunk");
        }
    }

    #[tokio::test]
    async fn test_fireworks_to_tensorzero_chunk_reasoning_content() {
        // Test that streaming reasoning_content delta uses REASONING_FIELD_CHUNK_ID
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: None,
                    reasoning_content: Some("step-by-step reasoning".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let mut tool_call_ids = Vec::new();
        let mut thinking_state = ThinkingState::Normal;

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        )
        .unwrap();

        assert_eq!(result.content.len(), 1, "expected one thought chunk");
        if let ContentBlockChunk::Thought(thought) = &result.content[0] {
            assert_eq!(thought.text, Some("step-by-step reasoning".to_string()));
            assert_eq!(
                thought.id,
                REASONING_FIELD_CHUNK_ID.to_string(),
                "expected REASONING_FIELD_CHUNK_ID for reasoning_content"
            );
            assert_eq!(
                thought.extra_data,
                Some(serde_json::json!({"reasoning_format": "reasoning_field"})),
                "expected reasoning_field format"
            );
        } else {
            panic!("Expected a thought chunk");
        }
    }

    #[tokio::test]
    async fn test_fireworks_response_with_reasoning_content_field() {
        // Test non-streaming response with reasoning_content field
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            stream: false,
            seed: None,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let response = FireworksResponse {
            choices: vec![FireworksResponseChoice {
                index: 0,
                finish_reason: Some(FireworksFinishReason::Stop),
                message: FireworksResponseMessage {
                    reasoning_content: Some("I need to think about this".to_string()),
                    content: Some("The answer is 42".to_string()),
                    tool_calls: None,
                },
            }],
            usage: OpenAIUsage {
                prompt_tokens: Some(5),
                completion_tokens: Some(15),
            },
        };

        let fireworks_response_with_metadata = FireworksResponseWithMetadata {
            response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: "test_request".to_string(),
            generic_request: &generic_request,
            model_inference_id: Uuid::now_v7(),
        };

        let result: ProviderInferenceResponse =
            fireworks_response_with_metadata.try_into().unwrap();

        assert_eq!(result.output.len(), 2, "expected thought + text blocks");

        // First block: thought from reasoning_content field
        match &result.output[0] {
            ContentBlockOutput::Thought(thought) => {
                assert_eq!(thought.text, Some("I need to think about this".to_string()));
                assert_eq!(
                    thought.extra_data,
                    Some(serde_json::json!({"reasoning_format": "reasoning_field"})),
                    "expected reasoning_field format for reasoning_content"
                );
            }
            _ => panic!("Expected a thought block"),
        }

        // Second block: text content
        match &result.output[1] {
            ContentBlockOutput::Text(text) => {
                assert_eq!(text.text, "The answer is 42");
            }
            _ => panic!("Expected a text block"),
        }
    }

    #[tokio::test]
    async fn test_fireworks_response_with_both_reasoning_sources() {
        // Test non-streaming response with both reasoning_content field AND <think> tags
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            stream: false,
            seed: None,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let response = FireworksResponse {
            choices: vec![FireworksResponseChoice {
                index: 0,
                finish_reason: Some(FireworksFinishReason::Stop),
                message: FireworksResponseMessage {
                    reasoning_content: Some("field reasoning".to_string()),
                    content: Some("<think>tag reasoning</think>The final answer".to_string()),
                    tool_calls: None,
                },
            }],
            usage: OpenAIUsage {
                prompt_tokens: Some(5),
                completion_tokens: Some(15),
            },
        };

        let fireworks_response_with_metadata = FireworksResponseWithMetadata {
            response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: "test_request".to_string(),
            generic_request: &generic_request,
            model_inference_id: Uuid::now_v7(),
        };

        let result: ProviderInferenceResponse =
            fireworks_response_with_metadata.try_into().unwrap();

        assert_eq!(
            result.output.len(),
            3,
            "expected reasoning_field thought + think_tags thought + text"
        );

        // First: reasoning_content field thought
        match &result.output[0] {
            ContentBlockOutput::Thought(thought) => {
                assert_eq!(thought.text, Some("field reasoning".to_string()));
                assert_eq!(
                    thought.extra_data,
                    Some(serde_json::json!({"reasoning_format": "reasoning_field"})),
                );
            }
            _ => panic!("Expected a thought block from reasoning_content"),
        }

        // Second: think tag thought
        match &result.output[1] {
            ContentBlockOutput::Thought(thought) => {
                assert_eq!(thought.text, Some("tag reasoning".to_string()));
                assert_eq!(
                    thought.extra_data,
                    Some(serde_json::json!({"reasoning_format": "think_tags"})),
                );
            }
            _ => panic!("Expected a thought block from think tags"),
        }

        // Third: text content
        match &result.output[2] {
            ContentBlockOutput::Text(text) => {
                assert_eq!(text.text, "The final answer");
            }
            _ => panic!("Expected a text block"),
        }
    }

    #[tokio::test]
    async fn test_tensorzero_to_fireworks_assistant_message_reasoning_field() {
        // Thought with reasoning_field format should go to reasoning_content
        let content_blocks = vec![
            ContentBlock::Thought(Thought {
                text: Some("my reasoning".to_string()),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
                extra_data: Some(serde_json::json!({"reasoning_format": "reasoning_field"})),
            }),
            ContentBlock::Text(tensorzero_types::content::Text {
                text: "the answer".to_string(),
            }),
        ];
        let config = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };
        let msg = tensorzero_to_fireworks_assistant_message(Cow::Borrowed(&content_blocks), config)
            .await
            .unwrap();
        if let OpenAIRequestMessage::Assistant(assistant) = msg {
            assert_eq!(
                assistant.reasoning_content,
                Some(Cow::Borrowed("my reasoning")),
                "expected reasoning_content to contain the thought"
            );
            assert!(
                assistant.content.is_some(),
                "expected content to be present"
            );
        } else {
            panic!("Expected an Assistant message");
        }
    }

    #[tokio::test]
    async fn test_tensorzero_to_fireworks_assistant_message_think_tags() {
        // Thought with think_tags format should be wrapped in <think> tags in content
        let content_blocks = vec![
            ContentBlock::Thought(Thought {
                text: Some("tag reasoning".to_string()),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
                extra_data: Some(serde_json::json!({"reasoning_format": "think_tags"})),
            }),
            ContentBlock::Text(tensorzero_types::content::Text {
                text: "the answer".to_string(),
            }),
        ];
        let config = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };
        let msg = tensorzero_to_fireworks_assistant_message(Cow::Borrowed(&content_blocks), config)
            .await
            .unwrap();
        if let OpenAIRequestMessage::Assistant(assistant) = msg {
            assert_eq!(
                assistant.reasoning_content, None,
                "reasoning_content should be None for think_tags"
            );
            let content = assistant.content.unwrap();
            // First block should be the think tag wrapper
            assert_eq!(content.len(), 2, "expected think tag block + text block");
            match &content[0] {
                OpenAIContentBlock::Text { text } => {
                    assert_eq!(text.as_ref(), "<think>tag reasoning</think>");
                }
                _ => panic!("Expected text content block with think tags"),
            }
        } else {
            panic!("Expected an Assistant message");
        }
    }

    #[tokio::test]
    async fn test_tensorzero_to_fireworks_assistant_message_invalid_reasoning_format() {
        // Thought with an unknown reasoning_format should return an error
        let content_blocks = vec![
            ContentBlock::Thought(Thought {
                text: Some("reasoning".to_string()),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
                extra_data: Some(serde_json::json!({"reasoning_format": "unknown_format"})),
            }),
            ContentBlock::Text(tensorzero_types::content::Text {
                text: "the answer".to_string(),
            }),
        ];
        let config = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };
        let result =
            tensorzero_to_fireworks_assistant_message(Cow::Owned(content_blocks), config).await;
        assert!(result.is_err(), "should error on unknown reasoning_format");
        let err = result.unwrap_err();
        let details = err.get_details();
        assert!(
            matches!(details, ErrorDetails::InvalidMessage { message } if message.contains("unknown_format")),
            "error should mention the unknown format value"
        );
    }

    #[tokio::test]
    async fn test_fireworks_stream_tool_call_handling() {
        // Test new tool call with ID and name
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![FireworksToolCallChunk {
                        index: 0,
                        id: Some("new_id".to_string()),
                        function: FireworksFunctionCallChunk {
                            name: Some("new_name".to_string()),
                            arguments: Some("{\"param\":\"value\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let mut tool_call_ids = Vec::new();
        let mut thinking_state = ThinkingState::Normal;

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        )
        .unwrap();

        // Should add the tool call to the state and the result
        assert_eq!(tool_call_ids, vec!["new_id"]);
        assert_eq!(result.content.len(), 1);

        if let ContentBlockChunk::ToolCall(tool_call) = &result.content[0] {
            assert_eq!(tool_call.id, "new_id");
            assert_eq!(tool_call.raw_name, Some("new_name".to_string()));
            assert_eq!(tool_call.raw_arguments, "{\"param\":\"value\"}");
        } else {
            panic!("Expected a tool call chunk");
        }

        // Test continuation of a tool call (id and name already known)
        let chunk = FireworksChatChunk {
            choices: vec![FireworksChatChunkChoice {
                delta: FireworksDelta {
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![FireworksToolCallChunk {
                        index: 0,
                        id: None,
                        function: FireworksFunctionCallChunk {
                            name: None,
                            arguments: Some(",\"more\":\"data\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(FireworksFinishReason::ToolCalls),
            }],
            usage: None,
        };

        let result = fireworks_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            Uuid::now_v7(),
            PROVIDER_TYPE,
        )
        .unwrap();

        // Should reference the existing ID and name
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.finish_reason, Some(FinishReason::ToolCall));

        if let ContentBlockChunk::ToolCall(tool_call) = &result.content[0] {
            assert_eq!(tool_call.id, "new_id");
            assert_eq!(tool_call.raw_name, None); // We don't add the tool name if it isn't explicitly set
            assert_eq!(tool_call.raw_arguments, ",\"more\":\"data\"}");
        } else {
            panic!("Expected a tool call chunk");
        }
    }

    #[test]
    fn test_fireworks_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = FireworksRequest::default();

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort is applied correctly
        assert_eq!(request.reasoning_effort, Some("high".to_string()));

        // Test that thinking_budget_tokens warns with tip about reasoning_effort
        assert!(logs_contain(
            "Fireworks does not support the inference parameter `thinking_budget_tokens`, so it will be ignored. Tip: You might want to use `reasoning_effort` for this provider."
        ));

        // Test that verbosity warns
        assert!(logs_contain(
            "Fireworks does not support the inference parameter `verbosity`"
        ));
    }
}
