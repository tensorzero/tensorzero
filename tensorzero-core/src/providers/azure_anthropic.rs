use std::borrow::Cow;

use futures::StreamExt;
use futures::future::try_join_all;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use super::anthropic::{
    AnthropicMessage, AnthropicMessageContent, AnthropicMessagesConfig, AnthropicRole,
    AnthropicStopReason, AnthropicStreamMessage, AnthropicSystemBlock, AnthropicTool,
    AnthropicToolChoice, anthropic_to_tensorzero_stream_message, handle_anthropic_error,
    prefill_json_chunk_response, prefill_json_response,
};
use super::helpers::{convert_stream_error, peek_first_chunk};
use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::{TensorZeroEventSource, TensorzeroHttpClient};
use crate::inference::InferenceProvider;
use crate::inference::types::FunctionType;
use crate::inference::types::batch::BatchRequestRow;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported,
};
use crate::inference::types::{
    ContentBlockOutput, FlattenUnknown, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseArgs, ProviderInferenceResponseStreamInner,
    Thought, Usage, batch::StartBatchProviderInferenceResponse,
};
use crate::model::{Credential, EndpointLocation, ModelProvider};
use crate::providers::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};
use crate::tool::{ToolCall, ToolChoice};

const PROVIDER_NAME: &str = "Azure Anthropic";
pub const PROVIDER_TYPE: &str = "azure_anthropic";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AzureAnthropicProvider {
    deployment_id: String,
    #[serde(skip)]
    endpoint: AzureAnthropicEndpoint,
    #[serde(skip)]
    credentials: AzureAnthropicCredentials,
}

#[derive(Clone, Debug)]
pub enum AzureAnthropicEndpoint {
    Static(Url),
    Dynamic(String),
}

impl AzureAnthropicEndpoint {
    fn get_endpoint(&self, dynamic_endpoints: &InferenceCredentials) -> Result<Url, Error> {
        match self {
            AzureAnthropicEndpoint::Static(url) => Ok(url.clone()),
            AzureAnthropicEndpoint::Dynamic(key_name) => {
                let endpoint_str = dynamic_endpoints.get(key_name).ok_or_else(|| {
                    Error::new(ErrorDetails::DynamicEndpointNotFound {
                        key_name: key_name.clone(),
                    })
                })?;
                Url::parse(endpoint_str.expose_secret()).map_err(|_| {
                    Error::new(ErrorDetails::InvalidDynamicEndpoint {
                        url: endpoint_str.expose_secret().to_string(),
                    })
                })
            }
        }
    }
}

impl AzureAnthropicProvider {
    pub fn new(
        deployment_id: String,
        endpoint_location: EndpointLocation,
        credentials: AzureAnthropicCredentials,
    ) -> Result<Self, Error> {
        let endpoint = match endpoint_location {
            EndpointLocation::Static(url_str) => {
                let url = Url::parse(&url_str).map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Invalid endpoint URL '{url_str}': {e}"),
                    })
                })?;
                AzureAnthropicEndpoint::Static(url)
            }
            EndpointLocation::Env(env_var) => {
                let url_str = std::env::var(&env_var).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Environment variable '{env_var}' not found for Azure Anthropic endpoint"
                        ),
                    })
                })?;
                let url = Url::parse(&url_str).map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Invalid endpoint URL from env var '{env_var}': {e}"),
                    })
                })?;
                AzureAnthropicEndpoint::Static(url)
            }
            EndpointLocation::Dynamic(key_name) => AzureAnthropicEndpoint::Dynamic(key_name),
        };

        Ok(AzureAnthropicProvider {
            deployment_id,
            endpoint,
            credentials,
        })
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    fn get_request_url(&self, dynamic_credentials: &InferenceCredentials) -> Result<Url, Error> {
        let mut url = self.endpoint.get_endpoint(dynamic_credentials)?;
        url.path_segments_mut()
            .map_err(|()| {
                Error::new(ErrorDetails::Config {
                    message: "Invalid Azure Anthropic endpoint URL (cannot-be-a-base)".to_string(),
                })
            })?
            .push("anthropic")
            .push("v1")
            .push("messages");
        Ok(url)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum AzureAnthropicCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<AzureAnthropicCredentials>,
        fallback: Box<AzureAnthropicCredentials>,
    },
}

impl TryFrom<Credential> for AzureAnthropicCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(api_key) => Ok(AzureAnthropicCredentials::Static(api_key)),
            Credential::Dynamic(key_name) => Ok(AzureAnthropicCredentials::Dynamic(key_name)),
            Credential::None | Credential::Missing => Ok(AzureAnthropicCredentials::None),
            Credential::WithFallback { default, fallback } => {
                let default = AzureAnthropicCredentials::try_from(*default)?;
                let fallback = AzureAnthropicCredentials::try_from(*fallback)?;
                Ok(AzureAnthropicCredentials::WithFallback {
                    default: Box::new(default),
                    fallback: Box::new(fallback),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Azure Anthropic provider".to_string(),
            })),
        }
    }
}

impl AzureAnthropicCredentials {
    fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            AzureAnthropicCredentials::Static(api_key) => Ok(api_key),
            AzureAnthropicCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            AzureAnthropicCredentials::None => Err(Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "API key is not configured for Azure Anthropic provider".to_string(),
            })),
            AzureAnthropicCredentials::WithFallback { default, fallback } => default
                .get_api_key(dynamic_api_keys)
                .or_else(|_| fallback.get_api_key(dynamic_api_keys)),
        }
    }
}

impl InferenceProvider for AzureAnthropicProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(
            AzureAnthropicRequestBody::new(&self.deployment_id, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Azure Anthropic request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = self.get_request_url(dynamic_api_keys)?;
        let start_time = Instant::now();
        let builder = http_client
            .post(request_url)
            .header("x-api-key", api_key.expose_secret())
            .header("anthropic-version", ANTHROPIC_API_VERSION);

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

        let response_status = res.status();

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

        if response_status.is_success() {
            let response: AzureAnthropicResponse =
                serde_json::from_str(&raw_response).map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing JSON response: {e}: {raw_response}"),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: Some(raw_request.clone()),
                        raw_response: Some(raw_response.clone()),
                    })
                })?;

            let response_with_metadata = AzureAnthropicResponseWithMetadata {
                response,
                raw_response,
                latency,
                raw_request,
                function_type: &request.function_type,
                json_mode: &request.json_mode,
                generic_request: request,
                model_name,
                provider_name,
            };

            Ok(response_with_metadata.try_into()?)
        } else {
            handle_anthropic_error(response_status, raw_request, raw_response)
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(
            AzureAnthropicRequestBody::new(&self.deployment_id, request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Azure Anthropic request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = self.get_request_url(dynamic_api_keys)?;
        let start_time = Instant::now();
        let builder = http_client
            .post(request_url)
            .header("x-api-key", api_key.expose_secret())
            .header("anthropic-version", ANTHROPIC_API_VERSION);

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

        let mut stream = stream_anthropic(
            event_source,
            start_time,
            model_provider,
            model_name,
            provider_name,
            &raw_request,
        )
        .peekable();
        let chunk = peek_first_chunk(&mut stream, &raw_request, PROVIDER_TYPE).await?;

        if matches!(
            request.json_mode,
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict
        ) && matches!(request.function_type, FunctionType::Json)
        {
            prefill_json_chunk_response(chunk);
        }

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

/// Azure Anthropic requires that the user provides `max_tokens`, but the value depends on the model.
/// We maintain a library of known maximum values, and ask the user to hardcode it if it's unknown.
///
/// Azure Anthropic currently supports only 4 models (all in Preview):
/// - claude-haiku-4-5: 64,000 max tokens
/// - claude-opus-4-1: 32,000 max tokens
/// - claude-sonnet-4-5: 64,000 max tokens
/// - claude-opus-4-5: 64,000 max tokens
///
/// All models support thinking, tool calling, and have a 200,000 token context window.
fn get_default_max_tokens(deployment_id: &str) -> Result<u32, Error> {
    if deployment_id.contains("claude-haiku-4-5")
        || deployment_id.contains("claude-sonnet-4-5")
        || deployment_id.contains("claude-opus-4-5")
    {
        Ok(64_000)
    } else if deployment_id.contains("claude-opus-4-1") {
        Ok(32_000)
    } else {
        Err(ErrorDetails::Config {
            message: format!(
                "Unknown Azure Anthropic model '{deployment_id}'. Azure Anthropic currently supports: claude-haiku-4-5, claude-opus-4-1, claude-sonnet-4-5, claude-opus-4-5. Please set `max_tokens` explicitly in the variant config if using a different model."
            ),
        }
        .into())
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct AzureAnthropicThinkingConfig {
    r#type: &'static str,
    budget_tokens: i32,
}

#[derive(Debug, Default, PartialEq, Serialize)]
struct AzureAnthropicRequestBody<'a> {
    model: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<AnthropicSystemBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AzureAnthropicThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
}

fn needs_json_prefill(request: &ModelInferenceRequest<'_>) -> bool {
    matches!(
        request.json_mode,
        ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict
    ) && matches!(request.function_type, FunctionType::Json)
}

fn prefill_json_message(mut messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    messages.push(AnthropicMessage {
        role: AnthropicRole::Assistant,
        content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
            text: "{",
        })],
    });
    messages
}

impl<'a> AzureAnthropicRequestBody<'a> {
    async fn new(
        deployment_id: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<AzureAnthropicRequestBody<'a>, Error> {
        if request.messages.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            }
            .into());
        }

        let messages_config = AnthropicMessagesConfig {
            fetch_and_encode_input_files_before_inference: request
                .fetch_and_encode_input_files_before_inference,
        };

        let system = match request.system.as_deref() {
            Some(text) => Some(vec![AnthropicSystemBlock::Text { text }]),
            None => None,
        };

        let messages: Vec<AnthropicMessage> =
            try_join_all(request.messages.iter().map(|m| {
                AnthropicMessage::from_request_message(m, messages_config, PROVIDER_TYPE)
            }))
            .await?;

        let messages = if needs_json_prefill(request) {
            prefill_json_message(messages)
        } else {
            messages
        };

        // Workaround for Anthropic API limitation: they don't support explicitly specifying "none"
        // for tool choice. When ToolChoice::None is specified, we don't send any tools in the
        // request payload to achieve the same effect.
        let tools = match &request.tool_config {
            Some(c) if !matches!(c.tool_choice, ToolChoice::None) => Some(
                c.strict_tools_available()?
                    // Azure Anthropic does not support structured outputs
                    .map(|tool| AnthropicTool::new(tool, false))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        };

        // `tool_choice` should only be set if tools are set and non-empty
        let tool_choice: Option<AnthropicToolChoice> = tools
            .as_ref()
            .filter(|t| !t.is_empty())
            .and(request.tool_config.as_ref())
            .and_then(|c| c.as_ref().try_into().ok());

        let max_tokens = match request.max_tokens {
            Some(max_tokens) => Ok(max_tokens),
            None => get_default_max_tokens(deployment_id),
        }?;

        let mut azure_anthropic_request = AzureAnthropicRequestBody {
            model: deployment_id,
            messages,
            max_tokens,
            stream: Some(request.stream),
            system,
            temperature: request.temperature,
            thinking: None,
            top_p: request.top_p,
            stop_sequences: request.borrow_stop_sequences(),
            tool_choice,
            tools,
        };

        apply_inference_params(&mut azure_anthropic_request, &request.inference_params_v2);

        Ok(azure_anthropic_request)
    }
}

fn apply_inference_params(
    request: &mut AzureAnthropicRequestBody,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity: _,
    } = inference_params;

    if let Some(reasoning_effort) = reasoning_effort {
        warn_inference_parameter_not_supported(
            PROVIDER_TYPE,
            "reasoning_effort",
            Some(&reasoning_effort.to_string()),
        );
    }

    if let Some(service_tier) = service_tier {
        warn_inference_parameter_not_supported(
            PROVIDER_TYPE,
            "service_tier",
            Some(&service_tier.to_string()),
        );
    }

    if let Some(thinking_budget_tokens) = thinking_budget_tokens {
        request.thinking = Some(AzureAnthropicThinkingConfig {
            r#type: "enabled",
            budget_tokens: *thinking_budget_tokens,
        });
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct AzureAnthropicResponse {
    id: String,
    r#type: String,
    role: String,
    content: Vec<AzureAnthropicContentBlock>,
    model: String,
    stop_reason: Option<AnthropicStopReason>,
    #[serde(default)]
    usage: AzureAnthropicUsage,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum AzureAnthropicContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct AzureAnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl From<AzureAnthropicUsage> for Usage {
    fn from(usage: AzureAnthropicUsage) -> Self {
        Usage {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
        }
    }
}

struct AzureAnthropicResponseWithMetadata<'a> {
    response: AzureAnthropicResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    function_type: &'a FunctionType,
    json_mode: &'a ModelInferenceRequestJsonMode,
    generic_request: &'a ModelInferenceRequest<'a>,
    model_name: &'a str,
    provider_name: &'a str,
}

fn convert_to_output(
    model_name: &str,
    provider_name: &str,
    block: AzureAnthropicContentBlock,
) -> Result<ContentBlockOutput, Error> {
    use crate::inference::types::Text;
    match block {
        AzureAnthropicContentBlock::Text { text } => Ok(ContentBlockOutput::Text(Text { text })),
        AzureAnthropicContentBlock::Thinking { thinking } => {
            Ok(ContentBlockOutput::Thought(Thought {
                text: Some(thinking),
                signature: None,
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
            }))
        }
        AzureAnthropicContentBlock::ToolUse { id, name, input } => {
            let arguments = serde_json::to_string(&input).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error serializing tool call arguments for provider {provider_name} and model {model_name}: {e}"
                    ),
                    raw_request: None,
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            Ok(ContentBlockOutput::ToolCall(ToolCall {
                id,
                name,
                arguments,
            }))
        }
    }
}

impl<'a> TryFrom<AzureAnthropicResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;

    fn try_from(value: AzureAnthropicResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let AzureAnthropicResponseWithMetadata {
            response,
            raw_response,
            latency,
            raw_request,
            function_type,
            json_mode,
            generic_request,
            model_name,
            provider_name,
        } = value;

        let content: Vec<ContentBlockOutput> = response
            .content
            .into_iter()
            .map(|block| convert_to_output(model_name, provider_name, block))
            .collect::<Result<Vec<_>, _>>()?;

        let content = if matches!(
            json_mode,
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict
        ) && matches!(function_type, FunctionType::Json)
        {
            prefill_json_response(content)?
        } else {
            content
        };

        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response,
                usage: response.usage.into(),
                latency,
                finish_reason: response.stop_reason.map(AnthropicStopReason::into),
            },
        ))
    }
}

/// Maps events from Anthropic into the TensorZero format
fn stream_anthropic(
    mut event_source: TensorZeroEventSource,
    start_time: Instant,
    model_provider: &ModelProvider,
    model_name: &str,
    provider_name: &str,
    raw_request: &str,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    let discard_unknown_chunks = model_provider.discard_unknown_chunks;
    let model_name = model_name.to_string();
    let provider_name = provider_name.to_string();
    Box::pin(async_stream::stream! {
        let mut current_tool_id: Option<String> = None;
        let mut current_tool_name: Option<String> = None;
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(raw_request.clone(), PROVIDER_TYPE.to_string(), e).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<AnthropicStreamMessage, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing message: {}, Data: {}",
                                    e, message.data
                                ),
                                provider_type: PROVIDER_TYPE.to_string(),
                                raw_request: Some(raw_request.clone()),
                                raw_response: None,
                            }));
                        // Anthropic streaming API docs specify that this is the last message
                        if let Ok(AnthropicStreamMessage::MessageStop) = data {
                            break;
                        }

                        let response = data.and_then(|data| {
                            anthropic_to_tensorzero_stream_message(
                                message.data.clone(),
                                data,
                                start_time.elapsed(),
                                &mut current_tool_id,
                                &mut current_tool_name,
                                discard_unknown_chunks,
                                &model_name,
                                &provider_name,
                                PROVIDER_TYPE,
                            )
                        });

                        match response {
                            Ok(None) => {},
                            Ok(Some(stream_message)) => yield Ok(stream_message),
                            Err(e) => yield Err(e),
                        }
                    }
                },
            }
        }

        event_source.close();
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::RequestMessage;
    use crate::inference::types::Role;

    #[tokio::test]
    async fn test_get_default_max_tokens() {
        // Azure Anthropic supports only 4 models (all in Preview):
        // claude-haiku-4-5, claude-opus-4-1, claude-sonnet-4-5, claude-opus-4-5

        // 64,000 max tokens models
        assert_eq!(
            get_default_max_tokens("claude-haiku-4-5").unwrap(),
            64_000,
            "claude-haiku-4-5 should have max_tokens of 64000"
        );
        assert_eq!(
            get_default_max_tokens("claude-sonnet-4-5").unwrap(),
            64_000,
            "claude-sonnet-4-5 should have max_tokens of 64000"
        );
        assert_eq!(
            get_default_max_tokens("claude-opus-4-5").unwrap(),
            64_000,
            "claude-opus-4-5 should have max_tokens of 64000"
        );

        // 32,000 max tokens models
        assert_eq!(
            get_default_max_tokens("claude-opus-4-1").unwrap(),
            32_000,
            "claude-opus-4-1 should have max_tokens of 32000"
        );

        // Test with deployment ID suffixes (Azure may append version suffixes)
        assert_eq!(
            get_default_max_tokens("claude-haiku-4-5-20250514").unwrap(),
            64_000,
            "claude-haiku-4-5 with date suffix should have max_tokens of 64000"
        );

        // Unsupported models should return error
        let result = get_default_max_tokens("claude-3-5-sonnet");
        assert!(
            result.is_err(),
            "claude-3-5-sonnet is not supported on Azure Anthropic"
        );

        let result = get_default_max_tokens("claude-sonnet-4");
        assert!(
            result.is_err(),
            "claude-sonnet-4 (without -5) is not supported on Azure Anthropic"
        );

        let result = get_default_max_tokens("unknown-model");
        assert!(result.is_err(), "Unknown model should return an error");
    }

    #[tokio::test]
    async fn test_azure_anthropic_request_body_new() {
        let messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into()],
        }];

        let request = ModelInferenceRequest {
            messages: messages.clone(),
            ..Default::default()
        };

        let body = AzureAnthropicRequestBody::new("claude-sonnet-4-5", &request)
            .await
            .unwrap();

        assert_eq!(body.model, "claude-sonnet-4-5");
        assert_eq!(body.max_tokens, 64_000);
        assert_eq!(body.stream, Some(false));
    }

    #[tokio::test]
    async fn test_azure_anthropic_request_body_with_explicit_max_tokens() {
        let messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into()],
        }];

        let request = ModelInferenceRequest {
            messages,
            max_tokens: Some(100),
            ..Default::default()
        };

        let body = AzureAnthropicRequestBody::new("claude-sonnet-4-5", &request)
            .await
            .unwrap();

        assert_eq!(body.max_tokens, 100, "Explicit max_tokens should be used");
    }

    #[tokio::test]
    async fn test_azure_anthropic_request_body_empty_messages() {
        let request = ModelInferenceRequest {
            messages: vec![],
            ..Default::default()
        };

        let result = AzureAnthropicRequestBody::new("claude-sonnet-4-5", &request).await;
        assert!(result.is_err(), "Empty messages should return an error");
    }

    // TODO: Implement test_from_tool
    // This test should verify that AnthropicTool is correctly created from a FunctionToolConfig
    // See gcp_vertex_anthropic.rs for reference implementation
    #[tokio::test]
    async fn test_from_tool() {
        // TODO: Test tool conversion from FunctionToolConfig to AnthropicTool
        // Should test both dynamic and static tool configs
    }

    // TODO: Implement test_try_from_content_block
    // This test should verify content block conversions (text, tool calls, tool results)
    // See gcp_vertex_anthropic.rs for reference implementation
    #[tokio::test]
    async fn test_try_from_content_block() {
        // TODO: Test text content block conversion
        // TODO: Test tool call content block conversion
        // TODO: Test tool result content block conversion
    }

    // TODO: Implement test_initialize_anthropic_request_body
    // This test should verify request body construction with various configurations
    // See gcp_vertex_anthropic.rs for reference implementation
    #[tokio::test]
    async fn test_initialize_anthropic_request_body() {
        // TODO: Test case 1: Empty message list (should error)
        // TODO: Test case 2: Messages with system message
        // TODO: Test case 3: Messages with temperature, top_p, max_tokens
        // TODO: Test case 4: Tool use & choice
    }

    // TODO: Implement test_azure_anthropic_usage_to_usage
    // This test should verify Usage conversion from AzureAnthropicUsage
    #[test]
    fn test_azure_anthropic_usage_to_usage() {
        let azure_anthropic_usage = AzureAnthropicUsage {
            input_tokens: 100,
            output_tokens: 50,
        };

        let usage: Usage = azure_anthropic_usage.into();

        assert_eq!(usage.input_tokens, Some(100), "input_tokens should be 100");
        assert_eq!(usage.output_tokens, Some(50), "output_tokens should be 50");
    }

    // TODO: Implement test_azure_anthropic_response_conversion
    // This test should verify full response conversion from AzureAnthropicResponse
    // See gcp_vertex_anthropic.rs for reference implementation
    #[test]
    fn test_azure_anthropic_response_conversion() {
        // TODO: Test case 1: Text response
        // TODO: Test case 2: Tool call response
        // TODO: Test case 3: Thinking block response
        // TODO: Test case 4: Mixed content response
    }

    // TODO: Implement test_prefill_json_message
    // This test should verify JSON prefill functionality
    #[test]
    fn test_prefill_json_message() {
        // TODO: Test that prefill_json_message adds an assistant message with "{"
    }

    // TODO: Implement test_azure_anthropic_apply_inference_params
    // This test should verify inference params are applied correctly
    #[test]
    fn test_azure_anthropic_apply_inference_params() {
        // TODO: Test thinking_budget_tokens is applied
        // TODO: Test that reasoning_effort and service_tier emit warnings
    }
}
