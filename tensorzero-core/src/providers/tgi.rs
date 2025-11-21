use crate::http::TensorzeroHttpClient;
use async_trait::async_trait;
/// TGI integration for TensorZero
///
/// Here, we list known limitations of TGI in our experience
///
/// First, TGI doesn't handle streaming tool calls correctly. It mangles the tool name field entirely and also doesn't catch EOS tokens properly.
/// Second, TGI doesn't handle multiple tools in a single request, because it doesn't return correct tool names. See the docs [here](https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/using_guidance#the-tools-parameter)
/// for an example.
/// Third, TGI doesn't support tool responses being sent back at all.
/// Fourth, TGI only supports JSON mode through a tool call. Luckily, we do this out of the box with `tool` as the json mode
///
/// In light of this, we have decided to not explicitly support tool calling for TGI and only support JSON mode via `tool`.
/// Our implementation currently allows you to use a tool in TGI (nonstreaming), but YMMV.
use futures::{Stream, StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::pin::Pin;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use super::helpers::convert_stream_error;
use super::openai::{
    get_chat_url, prepare_openai_messages, OpenAIRequestMessage, OpenAIToolType, StreamOptions,
    SystemOrDeveloper,
};
use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::DisplayOrDebugGateway;
use crate::error::{DelayedError, Error, ErrorDetails};
use crate::inference::types::batch::{
    BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
};
use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2,
};
use crate::inference::types::{
    ContentBlockChunk, ContentBlockOutput, FinishReason, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, TextChunk, Usage,
};
use crate::inference::InferenceProvider;
use crate::inference::TensorZeroEventError;
use crate::inference::WrappedProvider;
use crate::model::{Credential, ModelProvider};
use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::chat_completions::{ChatCompletionTool, ChatCompletionToolChoice};
use crate::providers::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};
use crate::providers::openai::{check_api_base_suffix, OpenAIMessagesConfig};
use crate::tool::ToolCall;

const PROVIDER_NAME: &str = "TGI";
pub const PROVIDER_TYPE: &str = "tgi";

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TGIProvider {
    api_base: Url,
    #[serde(skip)]
    credentials: TGICredentials,
}

impl TGIProvider {
    pub fn new(api_base: Url, credentials: TGICredentials) -> Self {
        // Check if the api_base has the `/chat/completions` suffix and warn if it does
        check_api_base_suffix(&api_base);

        TGIProvider {
            api_base,
            credentials,
        }
    }
}

#[derive(Clone, Debug)]
pub enum TGICredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<TGICredentials>,
        fallback: Box<TGICredentials>,
    },
}

impl TryFrom<Credential> for TGICredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(TGICredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(TGICredentials::Dynamic(key_name)),
            Credential::None => Ok(TGICredentials::None),
            Credential::Missing => Ok(TGICredentials::None),
            Credential::WithFallback { default, fallback } => Ok(TGICredentials::WithFallback {
                default: Box::new((*default).try_into()?),
                fallback: Box::new((*fallback).try_into()?),
            }),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for TGI provider".to_string(),
            })),
        }
    }
}

impl TGICredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, DelayedError> {
        match self {
            TGICredentials::Static(api_key) => Ok(Some(api_key)),
            TGICredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                }))
                .transpose()
            }
            TGICredentials::WithFallback { default, fallback } => {
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
            TGICredentials::None => Ok(None),
        }
    }
}

#[async_trait]
impl WrappedProvider for TGIProvider {
    fn thought_block_provider_type_suffix(&self) -> Cow<'static, str> {
        Cow::Borrowed("tgi")
    }

    async fn make_body<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name: _,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
    ) -> Result<serde_json::Value, Error> {
        // TGI doesn't care about the `model_name` field, so we can hardcode it to "tgi"

        serde_json::to_value(TGIRequest::new(PROVIDER_TYPE, request).await?).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing TGI request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })
    }

    fn parse_response(
        &self,
        request: &ModelInferenceRequest,
        raw_request: String,
        raw_response: String,
        latency: Latency,
        _model_name: &str,
        _provider_name: &str,
    ) -> Result<ProviderInferenceResponse, Error> {
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

        TGIResponseWithMetadata {
            response,
            latency,
            raw_response,
            raw_request,
            generic_request: request,
        }
        .try_into()
    }

    fn stream_events(
        &self,
        event_source: Pin<
            Box<dyn Stream<Item = Result<Event, TensorZeroEventError>> + Send + 'static>,
        >,
        start_time: Instant,
        raw_request: &str,
    ) -> ProviderInferenceResponseStreamInner {
        stream_tgi(event_source, start_time, raw_request)
    }
}

impl InferenceProvider for TGIProvider {
    async fn infer<'a>(
        &'a self,
        model_provider_request: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = self.make_body(model_provider_request).await?;
        let request_url = get_chat_url(&self.api_base)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();

        let mut request_builder = http_client.post(request_url);

        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &model_provider_request.request.extra_body,
            &model_provider_request.request.extra_headers,
            model_provider,
            model_provider_request.model_name,
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
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            self.parse_response(
                model_provider_request.request,
                raw_request,
                raw_response,
                latency,
                model_provider_request.model_name,
                model_provider_request.provider_name,
            )
        } else {
            Err(handle_tgi_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: Some(raw_request.clone()),
                        raw_response: None,
                    })
                })?,
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
        let request_body = serde_json::to_value(TGIRequest::new(PROVIDER_NAME, request).await?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing TGI request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;

        // TGI integration does not support tools in streaming mode
        if request_body.get("tools").is_some() {
            return Err(ErrorDetails::InvalidTool {
                message: "TGI does not support tools in streaming mode".to_string(),
            }
            .into());
        }
        let request_url = get_chat_url(&self.api_base)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let mut request_builder = http_client.post(request_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
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

        let stream = stream_tgi(
            event_source.map_err(TensorZeroEventError::EventSource),
            start_time,
            &raw_request,
        )
        .peekable();
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
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

fn stream_tgi(
    event_source: impl Stream<Item = Result<Event, TensorZeroEventError>> + Send + 'static,
    start_time: Instant,
    raw_request: &str,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    Box::pin(async_stream::stream! {
        futures::pin_mut!(event_source);
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    match e {
                        TensorZeroEventError::TensorZero(e) => {
                            yield Err(e);
                        }
                        TensorZeroEventError::EventSource(e) => {
                            yield Err(convert_stream_error(raw_request.clone(), PROVIDER_TYPE.to_string(), e).await);
                        }
                    }
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<TGIChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {e}",
                                ),
                                raw_request: Some(raw_request.clone()),
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            tgi_to_tensorzero_chunk(message.data, d, latency)
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

/// This struct defines the supported parameters for the TGI API
/// See the [TGI documentation](https://huggingface.co/docs/text-generation-inference/en/reference/api_reference#openai-messages-api)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(Default))]
struct TGIRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    #[cfg_attr(test, serde(default))]
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
}

fn apply_inference_params(
    _request: &mut TGIRequest,
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

impl<'a> TGIRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<TGIRequest<'a>, Error> {
        // TGI doesn't support JSON mode at all (only through tools [https://huggingface.co/docs/text-generation-inference/en/conceptual/guidance])
        // So we log a warning and ignore the JSON mode
        // You can get JSON mode through `tool` instead.
        if request.json_mode != ModelInferenceRequestJsonMode::Off {
            tracing::warn!("TGI does not support JSON mode. Ignoring JSON mode. Consider using `json_mode = \"tool\"` instead.");
        }

        let stream_options = if request.stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        };

        let messages = prepare_openai_messages(
            request
                .system
                .as_deref()
                .map(|m| SystemOrDeveloper::System(Cow::Borrowed(m))),
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

        let mut tgi_request = TGIRequest {
            messages,
            model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            seed: request.seed,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            stream: request.stream,
            stream_options,
            tools,
            tool_choice,
            parallel_tool_calls,
            stop: request.borrow_stop_sequences(),
        };

        apply_inference_params(&mut tgi_request, &request.inference_params_v2);

        Ok(tgi_request)
    }
}

struct TGIResponseWithMetadata<'a> {
    response: TGIResponse,
    latency: Latency,
    raw_response: String,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<TGIResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: TGIResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let TGIResponseWithMetadata {
            mut response,
            latency,
            raw_response,
            raw_request,
            generic_request,
        } = value;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response),
            }
            .into());
        }
        let usage = response.usage.into();
        let TGIResponseChoice {
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
            content.push(text.into());
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

pub(super) fn handle_tgi_error(response_code: StatusCode, response_body: &str) -> Error {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => ErrorDetails::InferenceClient {
            message: response_body.to_string(),
            status_code: Some(response_code),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into(),
        _ => ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into(),
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct TGIUsage {
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

impl From<TGIUsage> for Usage {
    fn from(usage: TGIUsage) -> Self {
        Usage {
            input_tokens: Some(usage.prompt_tokens),
            output_tokens: Some(usage.completion_tokens),
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct TGIResponseFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct TGIResponseToolCall {
    id: String,
    r#type: OpenAIToolType,
    function: TGIResponseFunctionCall,
}

impl From<TGIResponseToolCall> for ToolCall {
    fn from(tgi_tool_call: TGIResponseToolCall) -> Self {
        ToolCall {
            id: tgi_tool_call.id,
            name: tgi_tool_call.function.name,
            arguments: serde_json::to_string(&tgi_tool_call.function.arguments).unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<TGIResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum TGIFinishReason {
    Stop,
    StopSequence,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

impl From<TGIFinishReason> for FinishReason {
    fn from(finish_reason: TGIFinishReason) -> Self {
        match finish_reason {
            TGIFinishReason::Stop => FinishReason::Stop,
            TGIFinishReason::StopSequence => FinishReason::StopSequence,
            TGIFinishReason::Length => FinishReason::Length,
            TGIFinishReason::ContentFilter => FinishReason::ContentFilter,
            TGIFinishReason::ToolCalls => FinishReason::ToolCall,
            TGIFinishReason::FunctionCall => FinishReason::ToolCall,
            TGIFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct TGIResponseChoice {
    index: u8,
    message: TGIResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<TGIFinishReason>,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct TGIResponse {
    choices: Vec<TGIResponseChoice>,
    usage: TGIUsage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: TGIFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<TGIToolCallChunk>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIChatChunkChoice {
    delta: TGIDelta,
    finish_reason: Option<TGIFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIChatChunk {
    choices: Vec<TGIChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<TGIUsage>,
}

/// Maps an TGI chunk to a TensorZero chunk for streaming inferences
fn tgi_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: TGIChatChunk,
    latency: Duration,
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
    let mut content = vec![];
    let mut finish_reason = None;
    if let Some(choice) = chunk.choices.pop() {
        if let Some(reason) = choice.finish_reason {
            finish_reason = Some(reason.into());
        }
        if let Some(text) = choice.delta.content {
            content.push(ContentBlockChunk::Text(TextChunk {
                text,
                id: "0".to_string(),
            }));
        }
        if let Some(_tool_calls) = choice.delta.tool_calls {
            return Err(ErrorDetails::InferenceServer {
                message: "TGI returned a tool call but we don't make streaming tool call requests for this provider"
                    .to_string(),
                raw_request: None,
                raw_response: Some(raw_message),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
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

    use serde_json::json;
    use uuid::Uuid;

    use crate::{
        inference::types::{FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role},
        providers::{
            chat_completions::{
                ChatCompletionSpecificToolChoice, ChatCompletionSpecificToolFunction,
                ChatCompletionToolChoice, ChatCompletionToolType,
            },
            test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG},
        },
    };

    use super::*;

    #[tokio::test]
    async fn test_tgi_request_new() {
        let model_name = PROVIDER_TYPE.to_string();
        let basic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![
                RequestMessage {
                    role: Role::User,
                    content: vec!["Hello".to_string().into()],
                },
                RequestMessage {
                    role: Role::Assistant,
                    content: vec!["Hi there!".to_string().into()],
                },
            ],
            system: None,
            tool_config: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let tgi_request = TGIRequest::new(&model_name, &basic_request).await.unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 2);
        assert_eq!(tgi_request.temperature, Some(0.7));
        assert_eq!(tgi_request.max_tokens, Some(100));
        assert_eq!(tgi_request.seed, Some(69));
        assert_eq!(tgi_request.top_p, Some(0.9));
        assert_eq!(tgi_request.presence_penalty, Some(0.1));
        assert_eq!(tgi_request.frequency_penalty, Some(0.2));
        assert!(tgi_request.stream);
        assert!(tgi_request.tools.is_none());
        assert_eq!(tgi_request.tool_choice, None);
        assert!(tgi_request.parallel_tool_calls.is_none());

        // Test request with tools and JSON mode
        let request_with_tools = ModelInferenceRequest {
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
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let tgi_request = TGIRequest::new(&model_name, &request_with_tools)
            .await
            .unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 2);
        assert_eq!(tgi_request.temperature, None);
        assert_eq!(tgi_request.max_tokens, None);
        assert_eq!(tgi_request.seed, None);
        assert_eq!(tgi_request.top_p, None);
        assert_eq!(tgi_request.presence_penalty, None);
        assert_eq!(tgi_request.frequency_penalty, None);
        assert!(!tgi_request.stream);
        assert!(tgi_request.tools.is_some());
        let tools = tgi_request.tools.as_ref().unwrap();
        let tool = &tools[0];
        assert_eq!(tool.function.name, WEATHER_TOOL.name());
        assert_eq!(tool.function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            tgi_request.tool_choice,
            Some(ChatCompletionToolChoice::Specific(
                ChatCompletionSpecificToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionSpecificToolFunction {
                        name: WEATHER_TOOL.name(),
                    }
                }
            ))
        );

        // Test request with strict JSON mode with no output schema
        let request_with_tools = ModelInferenceRequest {
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
            json_mode: ModelInferenceRequestJsonMode::Strict,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let tgi_request = TGIRequest::new(&model_name, &request_with_tools)
            .await
            .unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 1);
        assert_eq!(tgi_request.temperature, None);
        assert_eq!(tgi_request.max_tokens, None);
        assert_eq!(tgi_request.seed, None);
        assert!(!tgi_request.stream);
        assert_eq!(tgi_request.top_p, None);
        assert_eq!(tgi_request.presence_penalty, None);
        assert_eq!(tgi_request.frequency_penalty, None);

        // Test request with strict JSON mode with an output schema
        let output_schema = json!({});
        let request_with_tools = ModelInferenceRequest {
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
            json_mode: ModelInferenceRequestJsonMode::Strict,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
            extra_body: Default::default(),
            ..Default::default()
        };

        let tgi_request = TGIRequest::new(&model_name, &request_with_tools)
            .await
            .unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 1);
        assert_eq!(tgi_request.temperature, None);
        assert_eq!(tgi_request.max_tokens, None);
        assert_eq!(tgi_request.seed, None);
        assert!(!tgi_request.stream);
        assert_eq!(tgi_request.top_p, None);
        assert_eq!(tgi_request.presence_penalty, None);
        assert_eq!(tgi_request.frequency_penalty, None);
    }

    #[test]
    fn test_tgi_to_tensorzero_chunk() {
        let chunk = TGIChatChunk {
            choices: vec![TGIChatChunkChoice {
                finish_reason: Some(TGIFinishReason::Stop),
                delta: TGIDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
            }],
            usage: None,
        };
        let message = tgi_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
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
    }
    #[tokio::test]
    async fn test_tgi_response_with_metadata_try_into() {
        let valid_response = TGIResponse {
            choices: vec![TGIResponseChoice {
                index: 0,
                finish_reason: Some(TGIFinishReason::Stop),
                message: TGIResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
            }],
            usage: TGIUsage {
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
        let tgi_response_with_metadata = TGIResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_value(
                TGIRequest::new("test-model", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap()
            .to_string(),
            generic_request: &generic_request,
        };
        let inference_response: ProviderInferenceResponse =
            tgi_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 1);
        assert_eq!(
            inference_response.output[0],
            "Hello, world!".to_string().into()
        );
        assert_eq!(inference_response.raw_response, "test_response");
        assert_eq!(inference_response.usage.input_tokens, Some(10));
        assert_eq!(inference_response.usage.output_tokens, Some(20));
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_secs(0)
            }
        );
    }

    #[test]
    fn test_tgi_provider_new_api_base_check() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Valid cases (should not warn)
        let _ = TGIProvider::new(
            Url::parse("http://localhost:1234/v1/").unwrap(),
            TGICredentials::None,
        );

        let _ = TGIProvider::new(
            Url::parse("http://localhost:1234/v1").unwrap(),
            TGICredentials::None,
        );

        // Invalid cases (should warn)
        let invalid_url_1 = Url::parse("http://localhost:1234/chat/completions").unwrap();
        let _ = TGIProvider::new(invalid_url_1.clone(), TGICredentials::None);
        assert!(logs_contain("automatically appends `/chat/completions`"));
        assert!(logs_contain(invalid_url_1.as_ref()));

        let invalid_url_2 = Url::parse("http://localhost:1234/v1/chat/completions/").unwrap();
        let _ = TGIProvider::new(invalid_url_2.clone(), TGICredentials::None);
        assert!(logs_contain("automatically appends `/chat/completions`"));
        assert!(logs_contain(invalid_url_2.as_ref()));
    }

    #[test]
    fn test_tgi_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = TGIRequest::default();

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort warns
        assert!(logs_contain(
            "TGI does not support the inference parameter `reasoning_effort`"
        ));

        // Test that thinking_budget_tokens warns
        assert!(logs_contain(
            "TGI does not support the inference parameter `thinking_budget_tokens`"
        ));

        // Test that verbosity warns
        assert!(logs_contain(
            "TGI does not support the inference parameter `verbosity`"
        ));
    }
}
