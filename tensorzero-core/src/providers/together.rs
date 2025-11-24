use std::{borrow::Cow, time::Duration};

use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2,
};
use crate::inference::types::RequestMessage;
use crate::providers::openai::OpenAIMessagesConfig;
use futures::{future::try_join_all, StreamExt};
use lazy_static::lazy_static;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use crate::cache::ModelProviderRequest;
use crate::error::DisplayOrDebugGateway;
use crate::http::{TensorZeroEventSource, TensorzeroHttpClient};
use crate::inference::types::{
    FinishReason, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseArgs,
};
use crate::inference::InferenceProvider;
use crate::model::{Credential, ModelProvider};
use crate::providers::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};
use crate::tool::ToolChoice;
use crate::{
    endpoints::inference::InferenceCredentials,
    error::{DelayedError, Error, ErrorDetails},
    inference::types::{
        batch::{BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse},
        ContentBlockChunk, ContentBlockOutput, ProviderInferenceResponseChunk,
        ProviderInferenceResponseStreamInner, Text, TextChunk, Thought, ThoughtChunk,
    },
    tool::{ToolCall, ToolCallChunk},
};

use super::helpers_thinking_block::{process_think_blocks, ThinkingState};
use super::openai::{
    get_chat_url, handle_openai_error, tensorzero_to_openai_messages, OpenAIRequestMessage,
    OpenAISystemRequestMessage, OpenAIToolType, OpenAIUsage,
};
use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::chat_completions::{
    ChatCompletionTool, ChatCompletionToolChoice, ChatCompletionToolChoiceString,
};

lazy_static! {
    pub static ref TOGETHER_API_BASE: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.together.xyz/v1/").expect("Failed to parse TOGETHER_API_BASE")
    };
}

pub const PROVIDER_NAME: &str = "Together";
pub const PROVIDER_TYPE: &str = "together";

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TogetherProvider {
    model_name: String,
    #[serde(skip)]
    credentials: TogetherCredentials,
    parse_think_blocks: bool,
}

pub fn default_parse_think_blocks() -> bool {
    true
}

impl TogetherProvider {
    pub fn new(
        model_name: String,
        credentials: TogetherCredentials,
        parse_think_blocks: bool,
    ) -> Self {
        TogetherProvider {
            model_name,
            credentials,
            parse_think_blocks,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum TogetherCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<TogetherCredentials>,
        fallback: Box<TogetherCredentials>,
    },
}

impl TryFrom<Credential> for TogetherCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(TogetherCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(TogetherCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(TogetherCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(TogetherCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Together provider".to_string(),
            })),
        }
    }
}

impl TogetherCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Cow<'a, SecretString>, DelayedError> {
        match self {
            TogetherCredentials::Static(api_key) => Ok(Cow::Owned(api_key.clone())),
            TogetherCredentials::Dynamic(key_name) => Ok(Cow::Borrowed(
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })?,
            )),
            TogetherCredentials::WithFallback { default, fallback } => {
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
            TogetherCredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
        }
    }
}

impl InferenceProvider for TogetherProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(
            TogetherRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Together request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let request_url = get_chat_url(&TOGETHER_API_BASE)?;
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

            Ok(TogetherResponseWithMetadata {
                response,
                raw_response,
                latency: Latency::NonStreaming {
                    response_time: start_time.elapsed(),
                },
                raw_request: raw_request.clone(),
                generic_request: request,
                parse_think_blocks: self.parse_think_blocks,
            }
            .try_into()?)
        } else {
            let status = res.status();
            let raw_response = res.text().await.map_err(|e| {
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
                &raw_response,
                PROVIDER_TYPE,
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
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(
            TogetherRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let request_url = get_chat_url(&TOGETHER_API_BASE)?;
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
        let stream = stream_together(event_source, start_time, self.parse_think_blocks).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
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
            provider_type: "GCP Vertex Gemini".to_string(),
        }
        .into())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum TogetherResponseFormat<'a> {
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<&'a Value>, // the desired JSON schema
    },
}

/// This struct defines the supported parameters for the Together inference API
/// See the [Together API documentation](https://docs.together.ai/docs/chat-overview)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, seed, service_tier, stop, user,
/// or context_length_exceeded_behavior
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(Default))]
struct TogetherRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    #[cfg_attr(test, serde(default))]
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
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<TogetherResponseFormat<'a>>,
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
    request: &mut TogetherRequest,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() {
        request.reasoning_effort = reasoning_effort.clone();
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

impl<'a> TogetherRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<TogetherRequest<'a>, Error> {
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(TogetherResponseFormat::JsonObject {
                    schema: request.output_schema,
                })
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_together_messages(
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

        // NOTE: Together AI doesn't seem to support `tool_choice="none"`, so we simply don't include the `tools` field if that's the case
        let tool_choice = request
            .tool_config
            .as_ref()
            .map(|config| &config.tool_choice);

        let (tools, mut tool_choice, parallel_tool_calls) = match tool_choice {
            Some(&ToolChoice::None) => (None, None, None),
            _ => prepare_chat_completion_tools(request, false)?,
        };
        // Together AI doesn't seem to support `tool_choice="required"`, so we convert it to `tool_choice="auto"`
        if let Some(ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Required)) =
            tool_choice
        {
            tool_choice = Some(ChatCompletionToolChoice::String(
                ChatCompletionToolChoiceString::Auto,
            ));
        }

        let mut together_request = TogetherRequest {
            messages,
            model,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            seed: request.seed,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
            stop: request.borrow_stop_sequences(),
            reasoning_effort: None,
        };

        apply_inference_params(&mut together_request, &request.inference_params_v2);

        Ok(together_request)
    }
}

pub async fn prepare_together_messages<'a>(
    system: Option<&'a str>,
    request_messages: &'a [RequestMessage],
    config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let mut messages: Vec<_> = try_join_all(
        request_messages
            .iter()
            .map(|msg| tensorzero_to_openai_messages(msg, config)),
    )
    .await?
    .into_iter()
    .flatten()
    .collect();

    if let Some(system_msg) = tensorzero_to_together_system_message(system) {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

fn tensorzero_to_together_system_message(system: Option<&str>) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct TogetherResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct TogetherResponseToolCall {
    id: String,
    r#type: OpenAIToolType,
    function: TogetherResponseFunctionCall,
}

impl From<TogetherResponseToolCall> for ToolCall {
    fn from(together_tool_call: TogetherResponseToolCall) -> Self {
        ToolCall {
            id: together_tool_call.id,
            name: together_tool_call.function.name,
            arguments: together_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<TogetherResponseToolCall>>,
}

// The thinking block processing has been moved to helpers_thinking_block.rs

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
enum TogetherFinishReason {
    Stop,
    Eos,
    Length,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

impl From<TogetherFinishReason> for FinishReason {
    fn from(finish_reason: TogetherFinishReason) -> Self {
        match finish_reason {
            TogetherFinishReason::Stop => FinishReason::Stop,
            TogetherFinishReason::Eos => FinishReason::Stop,
            TogetherFinishReason::Length => FinishReason::Length,
            TogetherFinishReason::ToolCalls => FinishReason::ToolCall,
            TogetherFinishReason::FunctionCall => FinishReason::ToolCall,
            TogetherFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct TogetherResponseChoice {
    index: u8,
    message: TogetherResponseMessage,
    finish_reason: Option<TogetherFinishReason>,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct TogetherResponse {
    choices: Vec<TogetherResponseChoice>,
    usage: OpenAIUsage,
}

struct TogetherResponseWithMetadata<'a> {
    response: TogetherResponse,
    latency: Latency,
    raw_response: String,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    parse_think_blocks: bool,
}

impl<'a> TryFrom<TogetherResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: TogetherResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let TogetherResponseWithMetadata {
            mut response,
            latency,
            raw_response,
            raw_request,
            generic_request,
            parse_think_blocks,
        } = value;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }
        let usage = response.usage.into();
        let TogetherResponseChoice {
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
        if let Some(raw_text) = message.content {
            let (clean_text, extracted_reasoning) =
                process_think_blocks(&raw_text, parse_think_blocks, PROVIDER_TYPE)?;
            if let Some(reasoning) = extracted_reasoning {
                content.push(ContentBlockOutput::Thought(Thought {
                    text: Some(reasoning),
                    signature: None,
                    summary: None,
                    provider_type: Some(PROVIDER_TYPE.to_string()),
                }));
            }
            if !clean_text.is_empty() {
                content.push(ContentBlockOutput::Text(Text { text: clean_text }));
            }
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        }
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
                latency,
                finish_reason: finish_reason.map(Into::into),
            },
        ))
    }
}

// ThinkingState has been moved to helpers_thinking_block.rs

fn stream_together(
    mut event_source: TensorZeroEventSource,
    start_time: Instant,
    parse_think_blocks: bool,
) -> ProviderInferenceResponseStreamInner {
    let mut tool_call_ids = Vec::new();
    let mut thinking_state = ThinkingState::Normal;
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    let message = e.to_string();
                    let mut raw_response = None;
                    if let reqwest_eventsource::Error::InvalidStatusCode(_, resp) = e {
                        raw_response = resp.text().await.ok();
                    }
                    yield Err(ErrorDetails::InferenceServer {
                        message,
                        raw_request: None,
                        raw_response,
                        provider_type: PROVIDER_TYPE.to_string(),
                    }.into());
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<TogetherChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!("Error parsing chunk. Error: {e}"),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            together_to_tensorzero_chunk(message.data, d, latency, &mut tool_call_ids, &mut thinking_state, parse_think_blocks)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    })
}

/// Maps a Together chunk to a TensorZero chunk for streaming inferences
///
/// This function handles the conversion of Together chat chunks into TensorZero chunks.
/// It processes the content and tool calls from the Together response, updating the tool call IDs and names.
/// If parsing think blocks is enabled, it also processes the thinking state and extracts reasoning.
fn together_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: TogetherChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    thinking_state: &mut ThinkingState,
    parse_think_blocks: bool,
) -> Result<ProviderInferenceResponseChunk, Error> {
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            raw_request: None,
            raw_response: Some(serde_json::to_string(&chunk).unwrap_or_default()),
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into());
    }
    let usage = chunk.usage.map(Into::into);
    let mut finish_reason = None;
    let mut content = vec![];
    if let Some(choice) = chunk.choices.pop() {
        if let Some(reason) = choice.finish_reason {
            finish_reason = Some(reason.into());
        }
        if let Some(text) = choice.delta.content {
            if parse_think_blocks {
                if !thinking_state.update(&text, PROVIDER_TYPE)? {
                    match thinking_state {
                        ThinkingState::Normal | ThinkingState::Finished => {
                            content.push(ContentBlockChunk::Text(TextChunk {
                                text,
                                id: thinking_state.get_id(),
                            }));
                        }
                        ThinkingState::Thinking => {
                            content.push(ContentBlockChunk::Thought(ThoughtChunk {
                                text: Some(text),
                                signature: None,
                                summary_id: None,
                                summary_text: None,
                                id: thinking_state.get_id(),
                                provider_type: Some(PROVIDER_TYPE.to_string()),
                            }));
                        }
                    }
                }
            } else {
                // Just add the text verbatim if we're not parsing think blocks.
                content.push(ContentBlockChunk::Text(TextChunk {
                    text,
                    id: "0".to_string(),
                }));
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

    Ok(ProviderInferenceResponseChunk::new(
        content,
        usage,
        raw_message,
        latency,
        finish_reason,
    ))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the type field
    function: TogetherFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<TogetherToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherChatChunkChoice {
    delta: TogetherDelta,
    finish_reason: Option<TogetherFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TogetherChatChunk {
    choices: Vec<TogetherChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAIUsage>,
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;
    use uuid::Uuid;

    use super::*;

    use crate::inference::types::{FunctionType, RequestMessage, Role, Usage};
    use crate::providers::chat_completions::{
        ChatCompletionSpecificToolChoice, ChatCompletionSpecificToolFunction,
        ChatCompletionToolChoice, ChatCompletionToolType,
    };
    use crate::providers::openai::OpenAIUsage;
    use crate::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};

    #[tokio::test]
    async fn test_together_request_new() {
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
            frequency_penalty: Some(0.1),
            max_tokens: Some(100),
            seed: Some(69),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let together_request =
            TogetherRequest::new("togethercomputer/llama-v3-8b", &request_with_tools)
                .await
                .unwrap();

        assert_eq!(together_request.model, "togethercomputer/llama-v3-8b");
        assert_eq!(together_request.messages.len(), 1);
        assert_eq!(together_request.temperature, Some(0.5));
        assert_eq!(together_request.top_p, Some(0.9));
        assert_eq!(together_request.presence_penalty, Some(0.1));
        assert_eq!(together_request.frequency_penalty, Some(0.1));
        assert_eq!(together_request.max_tokens, Some(100));
        assert_eq!(together_request.seed, Some(69));
        assert!(!together_request.stream);
        let tools = together_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        let tool = &tools[0];
        assert_eq!(tool.function.name, WEATHER_TOOL.name());
        assert_eq!(tool.function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            together_request.tool_choice,
            Some(ChatCompletionToolChoice::Specific(
                ChatCompletionSpecificToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionSpecificToolFunction {
                        name: WEATHER_TOOL.name(),
                    }
                }
            ))
        );
        assert_eq!(together_request.parallel_tool_calls, None);
    }

    #[test]
    fn test_together_api_base() {
        assert_eq!(TOGETHER_API_BASE.as_str(), "https://api.together.xyz/v1/");
    }
    #[test]
    fn test_credential_to_together_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: TogetherCredentials = TogetherCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, TogetherCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = TogetherCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, TogetherCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = TogetherCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, TogetherCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = TogetherCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[tokio::test]
    async fn test_together_response_with_metadata_try_into() {
        let valid_response = TogetherResponse {
            choices: vec![TogetherResponseChoice {
                index: 0,
                message: TogetherResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
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
        let together_response_with_metadata = TogetherResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &TogetherRequest::new("test-model", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: true,
        };
        let inference_response: ProviderInferenceResponse =
            together_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 1);
        assert_eq!(
            inference_response.output[0],
            "Hello, world!".to_string().into()
        );
        assert_eq!(inference_response.raw_response, "test_response");
        assert_eq!(inference_response.usage.input_tokens, Some(10));
        assert_eq!(inference_response.usage.output_tokens, Some(20));
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_secs(0)
            }
        );

        // Test case with thinking in the response
        let valid_response = TogetherResponse {
            choices: vec![TogetherResponseChoice {
                index: 0,
                message: TogetherResponseMessage {
                    content: Some("<think>hmmm</think>Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: OpenAIUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
            },
        };
        let together_response_with_metadata = TogetherResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &TogetherRequest::new("test-model", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: true,
        };
        let inference_response: ProviderInferenceResponse =
            together_response_with_metadata.try_into().unwrap();
        assert_eq!(inference_response.output.len(), 2);
        assert_eq!(
            inference_response.output[0],
            ContentBlockOutput::Thought(Thought {
                text: Some("hmmm".to_string()),
                signature: None,
                summary: None,
                provider_type: Some("together".to_string()),
            })
        );
        assert_eq!(
            inference_response.output[1],
            "Hello, world!".to_string().into()
        );
        assert_eq!(inference_response.raw_response, "test_response");

        // Test case with thinking in the middle of response
        let valid_response = TogetherResponse {
            choices: vec![TogetherResponseChoice {
                index: 0,
                message: TogetherResponseMessage {
                    content: Some("Hello <think>hmmm</think> world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: OpenAIUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
            },
        };
        let together_response_with_metadata = TogetherResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &TogetherRequest::new("test-model", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: true,
        };
        let inference_response: ProviderInferenceResponse =
            together_response_with_metadata.try_into().unwrap();
        assert_eq!(inference_response.output.len(), 2);
        assert_eq!(
            inference_response.output[0],
            ContentBlockOutput::Thought(Thought {
                text: Some("hmmm".to_string()),
                signature: None,
                summary: None,
                provider_type: Some("together".to_string()),
            })
        );
        assert_eq!(
            inference_response.output[1],
            "Hello  world!".to_string().into()
        );
        assert_eq!(inference_response.raw_response, "test_response");
    }

    #[tokio::test]
    async fn test_together_think_block_parsing_in_response() {
        // Test how TogetherAI integration works with think blocks in response parsing
        let response_with_thinking = TogetherResponse {
            choices: vec![TogetherResponseChoice {
                index: 0,
                message: TogetherResponseMessage {
                    content: Some(
                        "<think>This is the reasoning process</think>This is the answer"
                            .to_string(),
                    ),
                    tool_calls: None,
                },
                finish_reason: None,
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

        // With parsing enabled
        let metadata = TogetherResponseWithMetadata {
            response: response_with_thinking.clone(),
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &TogetherRequest::new("test-model", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: true,
        };

        let inference_response: ProviderInferenceResponse = metadata.try_into().unwrap();

        // We should have two content blocks - one thought and one text
        assert_eq!(inference_response.output.len(), 2);

        // First block should be the thought
        assert!(matches!(
            inference_response.output[0],
            ContentBlockOutput::Thought(_)
        ));
        if let ContentBlockOutput::Thought(thought) = &inference_response.output[0] {
            assert_eq!(
                thought.text,
                Some("This is the reasoning process".to_string())
            );
            assert_eq!(thought.signature, None);
        }

        // Second block should be the text
        assert!(matches!(
            inference_response.output[1],
            ContentBlockOutput::Text(_)
        ));
        if let ContentBlockOutput::Text(text) = &inference_response.output[1] {
            assert_eq!(text.text, "This is the answer".to_string());
        }

        // With parsing disabled
        let metadata = TogetherResponseWithMetadata {
            response: response_with_thinking,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &TogetherRequest::new("test-model", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            parse_think_blocks: false,
        };

        let inference_response: ProviderInferenceResponse = metadata.try_into().unwrap();

        // We should have one content block with the raw text
        assert_eq!(inference_response.output.len(), 1);
        assert!(matches!(
            inference_response.output[0],
            ContentBlockOutput::Text(_)
        ));
        if let ContentBlockOutput::Text(text) = &inference_response.output[0] {
            assert_eq!(
                text.text,
                "<think>This is the reasoning process</think>This is the answer"
            );
        }
    }

    #[tokio::test]
    async fn test_together_to_tensorzero_chunk_thinking() {
        // Test that the streaming function correctly handles thinking blocks
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("<think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let mut tool_call_ids = Vec::new();
        let mut thinking_state = ThinkingState::Normal;

        // With parsing enabled
        let result = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should transition to Thinking state
        assert!(matches!(thinking_state, ThinkingState::Thinking));
        // No content should be added for the opening tag
        assert!(result.content.is_empty());

        // Now process some thinking content
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("reasoning".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should still be in Thinking state
        assert!(matches!(thinking_state, ThinkingState::Thinking));
        // Content should be added as thought
        assert_eq!(result.content.len(), 1);
        assert!(matches!(result.content[0], ContentBlockChunk::Thought(_)));
        if let ContentBlockChunk::Thought(thought) = &result.content[0] {
            assert_eq!(thought.text, Some("reasoning".to_string()));
            assert_eq!(thought.id, "1");
        }

        // Close the thinking block
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("</think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should transition to Finished state
        assert!(matches!(thinking_state, ThinkingState::Finished));
        // No content should be added for the closing tag
        assert!(result.content.is_empty());

        // After closing, regular text should be treated as text content
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("Final answer".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let result = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(100),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();

        // Should remain in Finished state
        assert!(matches!(thinking_state, ThinkingState::Finished));
        // Content should be added as text
        assert_eq!(result.content.len(), 1);
        assert!(matches!(result.content[0], ContentBlockChunk::Text(_)));
        if let ContentBlockChunk::Text(text) = &result.content[0] {
            assert_eq!(text.text, "Final answer");
            assert_eq!(text.id, "2");
        }
    }

    #[tokio::test]
    async fn test_together_to_tensorzero_chunk() {
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let mut thinking_state = ThinkingState::Normal;
        let message = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "Hello".to_string(),
                id: "0".to_string(),
            })],
        );
        assert_eq!(message.finish_reason, Some(FinishReason::Stop));
        // Test what an intermediate tool chunk should look like
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: None,
                    tool_calls: Some(vec![TogetherToolCallChunk {
                        index: 0,
                        id: None,
                        function: TogetherFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(TogetherFinishReason::ToolCalls),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
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
        // Test what a bad tool chunk would do (new ID but no names)
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: None,
                    tool_calls: Some(vec![TogetherToolCallChunk {
                        index: 1,
                        id: None,
                        function: TogetherFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(TogetherFinishReason::ToolCalls),
            }],
            usage: None,
        };
        let error = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InferenceServer {
                message: "Tool call index out of bounds (meaning we haven't seen this many ids in the stream)".to_string(),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            }
        );
        // Test a correct new tool chunk
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: None,
                    tool_calls: Some(vec![TogetherToolCallChunk {
                        index: 1,
                        id: Some("id2".to_string()),
                        function: TogetherFunctionCallChunk {
                            name: Some("name2".to_string()),
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
                finish_reason: Some(TogetherFinishReason::ToolCalls),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "id2".to_string(),
                raw_name: Some("name2".to_string()),
                raw_arguments: "{\"hello\":\"world\"}".to_string(),
            })]
        );
        // Check that the lists were updated
        assert_eq!(tool_call_ids, vec!["id1".to_string(), "id2".to_string()]);

        // Check a chunk with no choices and only usage
        let chunk = TogetherChatChunk {
            choices: vec![],
            usage: Some(OpenAIUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
            }),
        };
        let message = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();
        assert_eq!(message.content, vec![]);
        assert_eq!(
            message.usage,
            Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            })
        );

        // Test a thinking chunk
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("<think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();
        assert!(message.content.is_empty());
        assert!(matches!(thinking_state, ThinkingState::Thinking));

        // Test a thinking middle chunk
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("some thinking content".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Thought(ThoughtChunk {
                text: Some("some thinking content".to_string()),
                signature: None,
                summary_id: None,
                summary_text: None,
                id: "1".to_string(),
                provider_type: Some("together".to_string()),
            })]
        );
        assert!(matches!(thinking_state, ThinkingState::Thinking));

        // Test a thinking chunk end
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("</think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let message = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            true,
        )
        .unwrap();
        assert!(message.content.is_empty());
        assert!(matches!(thinking_state, ThinkingState::Finished));
    }

    #[test]
    fn test_together_to_tensorzero_chunk_without_think_parsing() {
        let chunk = TogetherChatChunk {
            choices: vec![TogetherChatChunkChoice {
                delta: TogetherDelta {
                    content: Some("Hello <think>should not parse</think>".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(TogetherFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec![];
        let mut thinking_state = ThinkingState::Normal;
        let message = together_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk,
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut thinking_state,
            false,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "Hello <think>should not parse</think>".to_string(),
                id: "0".to_string(),
            })]
        );
    }

    #[test]
    fn test_together_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = TogetherRequest::default();

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort is applied correctly
        assert_eq!(request.reasoning_effort, Some("high".to_string()));

        // Test that thinking_budget_tokens warns with tip about reasoning_effort
        assert!(logs_contain(
            "Together does not support the inference parameter `thinking_budget_tokens`, so it will be ignored. Tip: You might want to use `reasoning_effort` for this provider."
        ));

        // Test that verbosity warns
        assert!(logs_contain(
            "Together does not support the inference parameter `verbosity`"
        ));
    }
}
