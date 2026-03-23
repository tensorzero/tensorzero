use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_sse_stream::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use std::borrow::Cow;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use super::helpers::{
    convert_stream_error, inject_extra_request_data_and_send,
    inject_extra_request_data_and_send_eventsource,
};
use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DelayedError, DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::{TensorZeroEventSource, TensorzeroHttpClient};
use crate::inference::InferenceProvider;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported,
};
use crate::inference::types::usage::raw_usage_entries_from_value;
use crate::inference::types::{
    ApiType, ContentBlockChunk, ContentBlockOutput, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, TextChunk,
    batch::StartBatchProviderInferenceResponse,
};
use crate::model::{Credential, ModelProvider};
use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::chat_completions::{ChatCompletionTool, ChatCompletionToolChoice};
use crate::providers::openai::OpenAIMessagesConfig;
use crate::providers::openai::{
    OpenAIFinishReason, OpenAIRequestMessage, OpenAIUsage, StreamOptions, SystemOrDeveloper,
    get_chat_url, handle_openai_error, openai_response_tool_call_to_tensorzero_tool_call,
    prepare_system_or_developer_message, tensorzero_to_openai_messages,
};
use crate::tool::ToolCallChunk;
use serde_json::Value;
use tensorzero_types_providers::minimax::{
    MiniMaxChatChunk, MiniMaxResponse, MiniMaxResponseChoice, MiniMaxResponseFormat,
};
use uuid::Uuid;

lazy_static! {
    static ref MINIMAX_DEFAULT_BASE_URL: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.minimax.io/v1")
            .expect("Failed to parse MINIMAX_DEFAULT_BASE_URL")
    };
}

const PROVIDER_NAME: &str = "MiniMax";
pub const PROVIDER_TYPE: &str = "minimax";

#[derive(Clone, Debug)]
pub enum MiniMaxCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<MiniMaxCredentials>,
        fallback: Box<MiniMaxCredentials>,
    },
}

impl TryFrom<Credential> for MiniMaxCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(MiniMaxCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(MiniMaxCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(MiniMaxCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(MiniMaxCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for MiniMax provider".to_string(),
            })),
        }
    }
}

impl MiniMaxCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            MiniMaxCredentials::Static(api_key) => Ok(api_key),
            MiniMaxCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            MiniMaxCredentials::WithFallback { default, fallback } => {
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
            MiniMaxCredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
        }
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MiniMaxProvider {
    model_name: String,
    #[serde(skip)]
    credentials: MiniMaxCredentials,
}

impl MiniMaxProvider {
    pub fn new(model_name: String, credentials: MiniMaxCredentials) -> Self {
        MiniMaxProvider {
            model_name,
            credentials,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl InferenceProvider for MiniMaxProvider {
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
        let request_body = serde_json::to_value(
            MiniMaxRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing MiniMax request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let request_url = get_chat_url(&MINIMAX_DEFAULT_BASE_URL)?;
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
            ApiType::ChatCompletions,
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
                    api_type: ApiType::ChatCompletions,
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
                    api_type: ApiType::ChatCompletions,
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(MiniMaxResponseWithMetadata {
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
                    api_type: ApiType::ChatCompletions,
                })
            })?;
            Err(handle_openai_error(
                &raw_request,
                status,
                &response,
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
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(
            MiniMaxRequest::new(&self.model_name, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing MiniMax request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let request_url = get_chat_url(&MINIMAX_DEFAULT_BASE_URL)?;
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
            ApiType::ChatCompletions,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;

        let stream =
            stream_minimax(event_source, start_time, &raw_request, model_inference_id).peekable();
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

fn minimax_response_format(
    json_mode: ModelInferenceRequestJsonMode,
) -> Option<MiniMaxResponseFormat> {
    match json_mode {
        ModelInferenceRequestJsonMode::On => Some(MiniMaxResponseFormat::JsonObject),
        ModelInferenceRequestJsonMode::Off => None,
        ModelInferenceRequestJsonMode::Strict => Some(MiniMaxResponseFormat::JsonObject),
    }
}

#[derive(Debug, Default, Serialize)]
struct MiniMaxRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
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
    stop: Option<Cow<'a, [String]>>,
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
    response_format: Option<MiniMaxResponseFormat>,
}

fn apply_inference_params(
    _request: &mut MiniMaxRequest,
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

impl<'a> MiniMaxRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<MiniMaxRequest<'a>, Error> {
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

        if request.json_mode == ModelInferenceRequestJsonMode::Strict {
            tracing::warn!(
                "MiniMax provider does not support strict JSON mode. Downgrading to normal JSON mode."
            );
        }

        let response_format = minimax_response_format(request.json_mode);

        let messages = prepare_minimax_messages(request).await?;

        let (tools, tool_choice, _) = prepare_chat_completion_tools(request, false)?;

        let mut minimax_request = MiniMaxRequest {
            messages,
            model,
            temperature,
            max_tokens,
            seed,
            top_p,
            stop: request.borrow_stop_sequences(),
            presence_penalty,
            frequency_penalty,
            stream,
            stream_options,
            response_format,
            tools,
            tool_choice,
        };

        apply_inference_params(&mut minimax_request, &request.inference_params_v2);

        Ok(minimax_request)
    }
}

fn stream_minimax(
    mut event_source: TensorZeroEventSource,
    start_time: Instant,
    raw_request: &str,
    model_inference_id: Uuid,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    let mut tool_call_ids = Vec::new();
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(raw_request.clone(), PROVIDER_TYPE.to_string(), ApiType::ChatCompletions, *e, None).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<MiniMaxChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {e}",
                                ),
                                raw_request: Some(raw_request.clone()),
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                                api_type: ApiType::ChatCompletions,
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            minimax_to_tensorzero_chunk(
                                message.data,
                                d,
                                latency,
                                &mut tool_call_ids,
                                model_inference_id,
                            )
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

/// Maps a MiniMax chunk to a TensorZero chunk for streaming inferences
fn minimax_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: MiniMaxChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    model_inference_id: Uuid,
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
    let raw_usage = minimax_usage_from_raw_response(&raw_message).map(|usage| {
        raw_usage_entries_from_value(
            model_inference_id,
            PROVIDER_TYPE,
            ApiType::ChatCompletions,
            usage,
        )
    });
    let usage = chunk.usage.map(OpenAIUsage::into);
    let mut content = vec![];
    let mut finish_reason = None;
    if let Some(choice) = chunk.choices.pop() {
        if let Some(choice_finish_reason) = choice.finish_reason {
            finish_reason = Some(choice_finish_reason.into());
        }
        if let Some(text) = choice.delta.content {
            content.push(ContentBlockChunk::Text(TextChunk {
                text,
                id: "0".to_string(),
            }));
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

    Ok(ProviderInferenceResponseChunk::new_with_raw_usage(
        content,
        usage,
        raw_message,
        latency,
        finish_reason,
        raw_usage,
    ))
}

async fn prepare_minimax_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let config = OpenAIMessagesConfig {
        json_mode: Some(&request.json_mode),
        provider_type: PROVIDER_TYPE,
        fetch_and_encode_input_files_before_inference: request
            .fetch_and_encode_input_files_before_inference,
    };
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        messages.extend(tensorzero_to_openai_messages(message, config).await?);
    }
    if let Some(system_msg) = prepare_system_or_developer_message(
        request
            .system
            .as_deref()
            .map(|m| SystemOrDeveloper::System(Cow::Borrowed(m))),
        Some(&request.json_mode),
        &messages,
    ) {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

struct MiniMaxResponseWithMetadata<'a> {
    response: MiniMaxResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    model_inference_id: Uuid,
}

impl<'a> TryFrom<MiniMaxResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: MiniMaxResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let MiniMaxResponseWithMetadata {
            mut response,
            raw_response,
            latency,
            raw_request,
            generic_request,
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
                api_type: ApiType::ChatCompletions,
            }
            .into());
        }

        let MiniMaxResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
                api_type: ApiType::ChatCompletions,
            }))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(text) = message.content {
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(
                    openai_response_tool_call_to_tensorzero_tool_call(tool_call),
                ));
            }
        }
        let raw_usage = minimax_usage_from_raw_response(&raw_response).map(|usage| {
            raw_usage_entries_from_value(
                model_inference_id,
                PROVIDER_TYPE,
                ApiType::ChatCompletions,
                usage,
            )
        });
        let usage = response.usage.into();
        let system = generic_request.system.clone();
        let messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages: messages,
                raw_request,
                raw_response,
                usage,
                raw_usage,
                relay_raw_response: None,
                provider_latency: latency,
                finish_reason: finish_reason.map(OpenAIFinishReason::into),
                id: model_inference_id,
            },
        ))
    }
}

fn minimax_usage_from_raw_response(raw_response: &str) -> Option<Value> {
    serde_json::from_str::<Value>(raw_response)
        .ok()
        .and_then(|value| value.get("usage").filter(|v| !v.is_null()).cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;
    use std::time::Duration;
    use uuid::Uuid;

    use crate::inference::types::{
        FinishReason, FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };
    use crate::providers::chat_completions::{
        ChatCompletionSpecificToolChoice, ChatCompletionSpecificToolFunction,
        ChatCompletionToolChoice, ChatCompletionToolType,
    };
    use crate::providers::openai::OpenAIUsage;
    use crate::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use tensorzero_types_providers::minimax::MiniMaxResponseMessage;

    #[tokio::test]
    async fn test_minimax_request_new() {
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

        let minimax_request = MiniMaxRequest::new("MiniMax-M2.7", &request_with_tools)
            .await
            .expect("failed to create MiniMax Request during test");

        assert_eq!(minimax_request.messages.len(), 1);
        assert_eq!(minimax_request.temperature, Some(0.5));
        assert_eq!(minimax_request.max_tokens, Some(100));
        assert!(!minimax_request.stream);
        assert_eq!(minimax_request.seed, Some(69));
        assert!(minimax_request.tools.is_some());
        let tools = minimax_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool = &tools[0];
        assert_eq!(tool.function.name, WEATHER_TOOL.name());
        assert_eq!(tool.function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            minimax_request.tool_choice,
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

        let minimax_request = MiniMaxRequest::new("MiniMax-M2.7", &request_with_tools)
            .await
            .expect("failed to create MiniMax Request");

        assert_eq!(minimax_request.messages.len(), 2);
        assert_eq!(minimax_request.temperature, Some(0.5));
        assert_eq!(minimax_request.max_tokens, Some(100));
        assert_eq!(minimax_request.top_p, Some(0.9));
        assert_eq!(minimax_request.presence_penalty, Some(0.1));
        assert_eq!(minimax_request.frequency_penalty, Some(0.2));
        assert!(!minimax_request.stream);
        assert_eq!(minimax_request.seed, Some(69));

        assert!(minimax_request.tools.is_some());
        let tools = minimax_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool = &tools[0];
        assert_eq!(tool.function.name, WEATHER_TOOL.name());
        assert_eq!(tool.function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            minimax_request.tool_choice,
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
            json_mode: ModelInferenceRequestJsonMode::Strict,
            ..request_with_tools
        };

        let minimax_request = MiniMaxRequest::new("MiniMax-M2.7", &request_with_tools).await;
        let minimax_request = minimax_request.unwrap();
        // We should downgrade the strict JSON mode to normal JSON mode for MiniMax
        assert_eq!(
            minimax_request.response_format,
            Some(MiniMaxResponseFormat::JsonObject)
        );
    }

    #[tokio::test]
    async fn test_minimax_api_base() {
        assert_eq!(
            MINIMAX_DEFAULT_BASE_URL.as_str(),
            "https://api.minimax.io/v1"
        );
    }

    #[tokio::test]
    async fn test_credential_to_minimax_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: MiniMaxCredentials = MiniMaxCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MiniMaxCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = MiniMaxCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MiniMaxCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = MiniMaxCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MiniMaxCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = MiniMaxCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[tokio::test]
    async fn test_minimax_response_with_metadata_try_into() {
        let valid_response = MiniMaxResponse {
            choices: vec![MiniMaxResponseChoice {
                index: 0,
                message: MiniMaxResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(OpenAIFinishReason::Stop),
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
        let minimax_response_with_metadata = MiniMaxResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(
                &MiniMaxRequest::new("MiniMax-M2.7", &generic_request)
                    .await
                    .unwrap(),
            )
            .unwrap(),
            generic_request: &generic_request,
            model_inference_id: Uuid::now_v7(),
        };
        let inference_response: ProviderInferenceResponse =
            minimax_response_with_metadata.try_into().unwrap();

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
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_minimax_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = MiniMaxRequest::default();

        apply_inference_params(&mut request, &inference_params);

        assert!(logs_contain(
            "MiniMax does not support the inference parameter `reasoning_effort`"
        ));

        assert!(logs_contain(
            "MiniMax does not support the inference parameter `thinking_budget_tokens`"
        ));

        assert!(logs_contain(
            "MiniMax does not support the inference parameter `verbosity`"
        ));
    }
}
