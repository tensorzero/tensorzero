use std::{borrow::Cow, time::Duration};

use crate::{
    http::{TensorZeroEventSource, TensorzeroHttpClient},
    providers::openai::OpenAIMessagesConfig,
};
use futures::{future::try_join_all, StreamExt};
use lazy_static::lazy_static;
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use url::Url;

use crate::{
    cache::ModelProviderRequest,
    endpoints::inference::InferenceCredentials,
    error::{DelayedError, DisplayOrDebugGateway, Error, ErrorDetails},
    inference::{
        types::{
            batch::{
                BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
            },
            chat_completion_inference_params::{
                warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2,
            },
            ContentBlockChunk, ContentBlockOutput, FinishReason, Latency, ModelInferenceRequest,
            ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
            ProviderInferenceResponse, ProviderInferenceResponseArgs,
            ProviderInferenceResponseChunk, ProviderInferenceResponseStreamInner, TextChunk, Usage,
        },
        InferenceProvider,
    },
    model::{Credential, ModelProvider},
    providers::helpers::{
        check_new_tool_call_name, convert_stream_error, inject_extra_request_data_and_send,
        inject_extra_request_data_and_send_eventsource,
    },
    tool::{FunctionToolConfig, ToolCall, ToolCallChunk, ToolChoice},
};

use super::openai::{
    get_chat_url, tensorzero_to_openai_messages, OpenAIFunction, OpenAIRequestMessage,
    OpenAISystemRequestMessage, OpenAIToolType,
};

lazy_static! {
    static ref MISTRAL_API_BASE: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.mistral.ai/v1/").expect("Failed to parse MISTRAL_API_BASE")
    };
}

const PROVIDER_NAME: &str = "Mistral";
pub const PROVIDER_TYPE: &str = "mistral";

type PreparedMistralToolsResult<'a> = (
    Option<Vec<MistralTool<'a>>>,
    Option<MistralToolChoice<'a>>,
    Option<bool>,
);

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct MistralProvider {
    model_name: String,
    #[serde(skip)]
    credentials: MistralCredentials,
}

impl MistralProvider {
    pub fn new(model_name: String, credentials: MistralCredentials) -> Self {
        MistralProvider {
            model_name,
            credentials,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum MistralCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<MistralCredentials>,
        fallback: Box<MistralCredentials>,
    },
}

impl TryFrom<Credential> for MistralCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(MistralCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(MistralCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(MistralCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(MistralCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Mistral provider".to_string(),
            })),
        }
    }
}

impl MistralCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            MistralCredentials::Static(api_key) => Ok(api_key),
            MistralCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            MistralCredentials::WithFallback { default, fallback } => {
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
            MistralCredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
        }
    }
}

impl InferenceProvider for MistralProvider {
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
            MistralRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Mistral request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let request_url = get_chat_url(&MISTRAL_API_BASE)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
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
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            MistralResponseWithMetadata {
                response,
                latency,
                raw_response,
                raw_request,
                generic_request: request,
            }
            .try_into()
        } else {
            handle_mistral_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: Some(raw_request),
                        raw_response: None,
                    })
                })?,
            )
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
            MistralRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Mistral request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let request_url = get_chat_url(&MISTRAL_API_BASE)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let builder = http_client
            .post(request_url)
            .bearer_auth(api_key.expose_secret());

        let (event_source, raw_request) = inject_extra_request_data_and_send_eventsource(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            builder,
        )
        .await?;
        let stream = stream_mistral(event_source, start_time, &raw_request).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Mistral".to_string(),
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

fn handle_mistral_error(
    response_code: StatusCode,
    response_body: &str,
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => Err(ErrorDetails::InferenceClient {
            message: response_body.to_string(),
            status_code: Some(response_code),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into()),
        _ => Err(ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into()),
    }
}

pub fn stream_mistral(
    mut event_source: TensorZeroEventSource,
    start_time: Instant,
    raw_request: &str,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            let mut last_tool_name = None;
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(raw_request.clone(), PROVIDER_TYPE.to_string(), e).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<MistralChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {}, Data: {}",
                                    e, message.data
                                ),
                                provider_type: PROVIDER_TYPE.to_string(),
                                raw_request: Some(raw_request.clone()),
                                raw_response: None,
                            }.into());
                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            mistral_to_tensorzero_chunk(message.data, d, latency, &mut last_tool_name)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    })
}

pub(super) async fn prepare_mistral_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
    config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let mut messages: Vec<_> = try_join_all(
        request
            .messages
            .iter()
            .map(|msg| tensorzero_to_openai_messages(msg, config)),
    )
    .await?
    .into_iter()
    .flatten()
    .collect();
    if let Some(system_msg) = tensorzero_to_mistral_system_message(request.system.as_deref()) {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

fn tensorzero_to_mistral_system_message(system: Option<&str>) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum MistralResponseFormat {
    JsonObject,
    #[default]
    Text,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(untagged)]
pub(super) enum MistralToolChoice<'a> {
    String(MistralToolChoiceString),
    Specific(MistralSpecificToolChoice<'a>),
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum MistralToolChoiceString {
    Auto,
    None,
    Any,
}

#[derive(Debug, Serialize, PartialEq)]
pub(super) struct MistralSpecificToolChoice<'a> {
    r#type: &'static str,
    function: MistralSpecificToolFunction<'a>,
}

#[derive(Debug, Serialize, PartialEq)]
struct MistralSpecificToolFunction<'a> {
    name: &'a str,
}

impl<'a> From<&'a ToolChoice> for MistralToolChoice<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::Auto => MistralToolChoice::String(MistralToolChoiceString::Auto),
            ToolChoice::Required => MistralToolChoice::String(MistralToolChoiceString::Any),
            ToolChoice::None => MistralToolChoice::String(MistralToolChoiceString::None),
            ToolChoice::Specific(tool_name) => {
                MistralToolChoice::Specific(MistralSpecificToolChoice {
                    r#type: "function",
                    function: MistralSpecificToolFunction { name: tool_name },
                })
            }
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct MistralTool<'a> {
    r#type: OpenAIToolType,
    function: OpenAIFunction<'a>,
}

impl<'a> From<&'a FunctionToolConfig> for MistralTool<'a> {
    fn from(tool: &'a FunctionToolConfig) -> Self {
        MistralTool {
            r#type: OpenAIToolType::Function,
            function: OpenAIFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
        }
    }
}

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to Mistral format
pub(super) fn prepare_mistral_tools<'a>(
    request: &'a ModelInferenceRequest<'a>,
) -> Result<PreparedMistralToolsResult<'a>, Error> {
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

            // Mistral does not support allowed_tools constraint, use regular tool_choice
            let tool_choice = Some((&tool_config.tool_choice).into());
            Ok((tools, tool_choice, parallel_tool_calls))
        }
    }
}

/// This struct defines the supported parameters for the Mistral inference API
/// See the [Mistral API documentation](https://docs.mistral.ai/api/#tag/chat)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, service_tier, stop, user,
/// or context_length_exceeded_behavior.
/// NOTE: Mistral does not support seed.
#[derive(Debug, Serialize)]
struct MistralRequest<'a> {
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
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    random_seed: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<MistralResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MistralTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<MistralToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
}

fn apply_inference_params(
    _request: &mut MistralRequest,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "reasoning_effort", None);
    }

    if service_tier.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier", None);
    }

    if thinking_budget_tokens.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "thinking_budget_tokens", None);
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
}

impl<'a> MistralRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<MistralRequest<'a>, Error> {
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(MistralResponseFormat::JsonObject)
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_mistral_messages(
            request,
            OpenAIMessagesConfig {
                json_mode: Some(&request.json_mode),
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: request
                    .fetch_and_encode_input_files_before_inference,
            },
        )
        .await?;
        let (tools, tool_choice, _) = prepare_mistral_tools(request)?;

        let mut mistral_request = MistralRequest {
            messages,
            model,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            random_seed: request.seed,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
            stop: request.borrow_stop_sequences(),
        };

        apply_inference_params(&mut mistral_request, &request.inference_params_v2);

        Ok(mistral_request)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct MistralUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl From<MistralUsage> for Usage {
    fn from(usage: MistralUsage) -> Self {
        Usage {
            input_tokens: Some(usage.prompt_tokens),
            output_tokens: Some(usage.completion_tokens),
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct MistralResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct MistralResponseToolCall {
    id: String,
    function: MistralResponseFunctionCall,
}

impl From<MistralResponseToolCall> for ToolCall {
    fn from(mistral_tool_call: MistralResponseToolCall) -> Self {
        ToolCall {
            id: mistral_tool_call.id,
            name: mistral_tool_call.function.name,
            arguments: mistral_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
enum MistralFinishReason {
    Stop,
    Length,
    ModelLength,
    Error,
    ToolCalls,
    #[serde(other)]
    Unknown,
}

impl From<MistralFinishReason> for FinishReason {
    fn from(reason: MistralFinishReason) -> Self {
        match reason {
            MistralFinishReason::Stop => FinishReason::Stop,
            MistralFinishReason::Length => FinishReason::Length,
            MistralFinishReason::ModelLength => FinishReason::Length,
            MistralFinishReason::Error => FinishReason::Unknown,
            MistralFinishReason::ToolCalls => FinishReason::ToolCall,
            MistralFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct MistralResponseChoice {
    index: u8,
    message: MistralResponseMessage,
    finish_reason: MistralFinishReason,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct MistralResponse {
    choices: Vec<MistralResponseChoice>,
    usage: MistralUsage,
}

struct MistralResponseWithMetadata<'a> {
    response: MistralResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<MistralResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: MistralResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let MistralResponseWithMetadata {
            mut response,
            raw_response,
            latency,
            raw_request,
            generic_request,
        } = value;
        if response.choices.len() != 1 {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_response.clone()),
            }));
        }
        let usage = response.usage.into();
        let MistralResponseChoice {
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
        if let Some(text) = message.content {
            if !text.is_empty() {
                content.push(text.into());
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
                finish_reason: Some(finish_reason.into()),
            },
        ))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralFunctionCallChunk {
    name: String,
    arguments: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralToolCallChunk {
    // #[serde(skip_serializing_if = "Option::is_none")]
    id: String,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: MistralFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralChatChunkChoice {
    delta: MistralDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<MistralFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralChatChunk {
    choices: Vec<MistralChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<MistralUsage>,
}

/// Maps a Mistral chunk to a TensorZero chunk for streaming inferences
fn mistral_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: MistralChatChunk,
    latency: Duration,
    last_tool_name: &mut Option<String>,
) -> Result<ProviderInferenceResponseChunk, Error> {
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: Some(raw_message.clone()),
        }
        .into());
    }
    let usage = chunk.usage.map(Into::into);
    let mut content = vec![];
    let mut finish_reason = None;
    if let Some(choice) = chunk.choices.pop() {
        if let Some(choice_finish_reason) = choice.finish_reason {
            finish_reason = Some(choice_finish_reason.into());
        }
        if let Some(text) = choice.delta.content {
            if !text.is_empty() {
                content.push(ContentBlockChunk::Text(TextChunk {
                    text,
                    id: "0".to_string(),
                }));
            }
        }
        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: tool_call.id,
                    raw_name: check_new_tool_call_name(tool_call.function.name, last_tool_name),
                    raw_arguments: tool_call.function.arguments,
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;

    use crate::inference::types::{FunctionType, RequestMessage, Role};
    use crate::providers::test_helpers::{QUERY_TOOL, WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::tool::{AllowedTools, ToolCallConfig};
    #[tokio::test]
    async fn test_mistral_request_new() {
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
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let mistral_request = MistralRequest::new("mistral-small-latest", &request_with_tools)
            .await
            .unwrap();

        assert_eq!(mistral_request.model, "mistral-small-latest");
        assert_eq!(mistral_request.messages.len(), 1);
        assert_eq!(mistral_request.temperature, Some(0.5));
        assert_eq!(mistral_request.max_tokens, Some(100));
        assert!(!mistral_request.stream);
        assert_eq!(
            mistral_request.response_format,
            Some(MistralResponseFormat::JsonObject)
        );
        assert!(mistral_request.tools.is_some());
        let tools = mistral_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            mistral_request.tool_choice,
            Some(MistralToolChoice::Specific(MistralSpecificToolChoice {
                r#type: "function",
                function: MistralSpecificToolFunction {
                    name: "get_temperature"
                },
            }))
        );
    }

    #[test]
    fn test_prepare_mistral_tools_with_allowed_tools() {
        use crate::tool::{AllowedTools, AllowedToolsChoice};

        // Test with allowed_tools specified - Mistral doesn't support allowed_tools constraint
        let tool_config = ToolCallConfig {
            static_tools_available: vec![WEATHER_TOOL.clone(), QUERY_TOOL.clone()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: Some(false),
            allowed_tools: AllowedTools {
                tools: vec![WEATHER_TOOL.name().to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify only allowed tools are returned (strict_tools_available respects allowed_tools)
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());

        // Verify tool_choice
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::String(MistralToolChoiceString::Auto)
        );

        // Verify parallel_tool_calls
        assert_eq!(parallel_tool_calls, Some(false));
    }

    #[test]
    fn test_prepare_mistral_tools_auto_mode() {
        // Test Auto mode with default allowed_tools
        let tool_config = ToolCallConfig {
            static_tools_available: vec![WEATHER_TOOL.clone(), QUERY_TOOL.clone()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(),
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify tools
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2);

        // Verify tool_choice is Auto
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::String(MistralToolChoiceString::Auto)
        );

        // Verify parallel_tool_calls is None (default behavior)
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_prepare_mistral_tools_required_mode() {
        use crate::tool::AllowedTools;

        let tool_config = ToolCallConfig {
            static_tools_available: vec![WEATHER_TOOL.clone()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(),
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify tools
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);

        // Verify tool_choice is Any (Required maps to Any)
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::String(MistralToolChoiceString::Any)
        );

        // Verify parallel_tool_calls is None
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_prepare_mistral_tools_none_mode() {
        let tool_config = ToolCallConfig {
            static_tools_available: vec![WEATHER_TOOL.clone()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(),
        };

        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify tools are still returned
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);

        // Verify tool_choice is None
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::String(MistralToolChoiceString::None)
        );

        // Verify parallel_tool_calls is None
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_prepare_mistral_tools_specific_mode() {
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_mistral_tools(&request).unwrap();

        // Verify tools
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());

        // Verify tool_choice is Specific
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            MistralToolChoice::Specific(MistralSpecificToolChoice {
                r#type: "function",
                function: MistralSpecificToolFunction {
                    name: "get_temperature"
                },
            })
        );

        // Verify parallel_tool_calls is None (WEATHER_TOOL_CONFIG doesn't set parallel_tool_calls)
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_try_from_mistral_response() {
        // Test case 1: Valid response with content
        let valid_response = MistralResponse {
            choices: vec![MistralResponseChoice {
                index: 0,
                message: MistralResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: MistralFinishReason::Stop,
            }],
            usage: MistralUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
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
            seed: Some(69),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test_response".to_string();
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: valid_response,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec!["Hello, world!".to_string().into()]
        );
        assert_eq!(inference_response.usage.input_tokens, Some(10));
        assert_eq!(inference_response.usage.output_tokens, Some(20));
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(100)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.raw_response, raw_response);
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(inference_response.system, None);
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );

        // Test case 2: Valid response with tool calls
        let valid_response_with_tools = MistralResponse {
            choices: vec![MistralResponseChoice {
                index: 0,
                message: MistralResponseMessage {
                    content: None,
                    tool_calls: Some(vec![MistralResponseToolCall {
                        id: "call1".to_string(),
                        function: MistralResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
                finish_reason: MistralFinishReason::ToolCalls,
            }],
            usage: MistralUsage {
                prompt_tokens: 15,
                completion_tokens: 25,
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }],
            system: Some("test_system".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: valid_response_with_tools,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(110),
            },
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec![ContentBlockOutput::ToolCall(ToolCall {
                id: "call1".to_string(),
                name: "test_function".to_string(),
                arguments: "{}".to_string(),
            })]
        );
        assert_eq!(inference_response.usage.input_tokens, Some(15));
        assert_eq!(inference_response.usage.output_tokens, Some(25));
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(110)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.raw_response, raw_response);
        assert_eq!(inference_response.system, Some("test_system".to_string()));
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]
        );
        // Test case 3: Invalid response with no choices
        let invalid_response_no_choices = MistralResponse {
            choices: vec![],
            usage: MistralUsage {
                prompt_tokens: 5,
                completion_tokens: 0,
            },
        };
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop: None,
        };
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: invalid_response_no_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(120),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        let error = result.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        // Test case 4: Invalid response with multiple choices
        let invalid_response_multiple_choices = MistralResponse {
            choices: vec![
                MistralResponseChoice {
                    index: 0,
                    message: MistralResponseMessage {
                        content: Some("Choice 1".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: MistralFinishReason::Stop,
                },
                MistralResponseChoice {
                    index: 1,
                    message: MistralResponseMessage {
                        content: Some("Choice 2".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: MistralFinishReason::Stop,
                },
            ],
            usage: MistralUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
            },
        };
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
            stop: None,
        };
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: invalid_response_multiple_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(130),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        let error = result.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
    }

    #[test]
    fn test_handle_mistral_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_mistral_error(StatusCode::UNAUTHORIZED, "Unauthorized access");
        let error = unauthorized.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(*message, "Unauthorized access");
            assert_eq!(*status_code, Some(StatusCode::UNAUTHORIZED));
            assert_eq!(*provider, PROVIDER_TYPE.to_string());
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, None);
        }

        // Test forbidden error
        let forbidden = handle_mistral_error(StatusCode::FORBIDDEN, "Forbidden access");
        let error = forbidden.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(*message, "Forbidden access");
            assert_eq!(*status_code, Some(StatusCode::FORBIDDEN));
            assert_eq!(*provider, PROVIDER_TYPE.to_string());
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, None);
        }

        // Test rate limit error
        let rate_limit = handle_mistral_error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded");
        let error = rate_limit.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(*message, "Rate limit exceeded");
            assert_eq!(*status_code, Some(StatusCode::TOO_MANY_REQUESTS));
            assert_eq!(*provider, PROVIDER_TYPE.to_string());
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, None);
        }

        // Test server error
        let server_error = handle_mistral_error(StatusCode::INTERNAL_SERVER_ERROR, "Server error");
        let error = server_error.unwrap_err();
        let details = error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        if let ErrorDetails::InferenceServer {
            message,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(*message, "Server error");
            assert_eq!(*provider, PROVIDER_TYPE.to_string());
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, None);
        }
    }

    #[test]
    fn test_mistral_api_base() {
        assert_eq!(MISTRAL_API_BASE.as_str(), "https://api.mistral.ai/v1/");
    }

    #[test]
    fn test_credential_to_mistral_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: MistralCredentials = MistralCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MistralCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = MistralCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MistralCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = MistralCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MistralCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = MistralCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_mistral_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = MistralRequest {
            messages: vec![],
            model: "test-model",
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            random_seed: None,
            stream: false,
            response_format: None,
            tools: None,
            tool_choice: None,
            stop: None,
        };

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort warns
        assert!(logs_contain(
            "Mistral does not support the inference parameter `reasoning_effort`"
        ));

        // Test that thinking_budget_tokens warns
        assert!(logs_contain(
            "Mistral does not support the inference parameter `thinking_budget_tokens`"
        ));

        // Test that verbosity warns
        assert!(logs_contain(
            "Mistral does not support the inference parameter `verbosity`"
        ));
    }
}
