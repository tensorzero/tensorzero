use std::borrow::Cow;

use futures::{StreamExt, TryStreamExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::time::Instant;
use url::Url;

use crate::cache::ModelProviderRequest;
use crate::embeddings::{
    Embedding, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingProvider,
    EmbeddingProviderRequestInfo, EmbeddingProviderResponse, EmbeddingRequest,
};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DelayedError, DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::batch::BatchRequestRow;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2, ServiceTier,
};
use crate::inference::types::extra_body::FullExtraBodyConfig;
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse,
};
use crate::inference::types::{ContentBlockOutput, ProviderInferenceResponseArgs};
use crate::model::{Credential, EndpointLocation, ModelProvider};
use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};
use crate::providers::openai::OpenAIMessagesConfig;

use super::chat_completions::{
    ChatCompletionAllowedToolsChoice, ChatCompletionSpecificToolChoice, ChatCompletionTool,
    ChatCompletionToolChoice, ChatCompletionToolChoiceString,
};
use super::openai::{
    handle_openai_error, prepare_openai_messages, stream_openai, OpenAIEmbeddingUsage,
    OpenAIRequestMessage, OpenAIResponse, OpenAIResponseChoice, SystemOrDeveloper,
};
use crate::inference::{InferenceProvider, TensorZeroEventError};

const PROVIDER_NAME: &str = "Azure";
pub const PROVIDER_TYPE: &str = "azure";
const AZURE_INFERENCE_API_VERSION: &str = "2025-04-01-preview";

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AzureProvider {
    deployment_id: String,
    #[serde(skip)]
    endpoint: AzureEndpoint,
    #[serde(skip)]
    credentials: AzureCredentials,
}

#[derive(Clone, Debug)]
pub enum AzureEndpoint {
    Static(Url),
    Dynamic(String),
}

impl AzureEndpoint {
    fn get_endpoint<'a>(
        &'a self,
        dynamic_endpoints: &'a InferenceCredentials,
    ) -> Result<Url, Error> {
        match self {
            AzureEndpoint::Static(url) => Ok(url.clone()),
            AzureEndpoint::Dynamic(key_name) => {
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

impl AzureProvider {
    pub fn new(
        deployment_id: String,
        endpoint_location: EndpointLocation,
        credentials: AzureCredentials,
    ) -> Result<Self, Error> {
        let endpoint = match endpoint_location {
            EndpointLocation::Static(url_str) => {
                let url = Url::parse(&url_str).map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Invalid endpoint URL '{url_str}': {e}"),
                    })
                })?;
                AzureEndpoint::Static(url)
            }
            EndpointLocation::Env(env_var) => {
                let url_str = std::env::var(&env_var).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: format!(
                            "Environment variable '{env_var}' not found for Azure endpoint"
                        ),
                    })
                })?;
                let url = Url::parse(&url_str).map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Invalid endpoint URL from env var '{env_var}': {e}"),
                    })
                })?;
                AzureEndpoint::Static(url)
            }
            EndpointLocation::Dynamic(key_name) => AzureEndpoint::Dynamic(key_name),
        };

        Ok(AzureProvider {
            deployment_id,
            endpoint,
            credentials,
        })
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum AzureCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<AzureCredentials>,
        fallback: Box<AzureCredentials>,
    },
}

impl TryFrom<Credential> for AzureCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(AzureCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(AzureCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(AzureCredentials::None),
            Credential::WithFallback { default, fallback } => Ok(AzureCredentials::WithFallback {
                default: Box::new((*default).try_into()?),
                fallback: Box::new((*fallback).try_into()?),
            }),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Azure provider".to_string(),
            })),
        }
    }
}

impl AzureCredentials {
    fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            AzureCredentials::Static(api_key) => Ok(api_key),
            AzureCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            AzureCredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
            AzureCredentials::WithFallback { default, fallback } => {
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
        }
    }
}

impl InferenceProvider for AzureProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        api_key: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body =
            serde_json::to_value(AzureRequest::new(request).await?).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Azure request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let endpoint = self.endpoint.get_endpoint(api_key)?;
        let request_url = get_azure_chat_url(&endpoint, &self.deployment_id)?;
        let start_time = Instant::now();
        let api_key = self.credentials.get_api_key(api_key).map_err(|e| e.log())?;
        let builder = http_client
            .post(request_url)
            .header("api-key", api_key.expose_secret());

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
        if res.status().is_success() {
            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

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
                    message: format!("Error parsing JSON response: {e}: {raw_response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            Ok(AzureResponseWithMetadata {
                response,
                latency,
                raw_request,
                generic_request: request,
                raw_response,
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
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
            Err(handle_openai_error(
                &raw_request,
                status,
                &response,
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
        let request_body =
            serde_json::to_value(AzureRequest::new(request).await?).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Azure request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let endpoint = self.endpoint.get_endpoint(dynamic_api_keys)?;
        let request_url = get_azure_chat_url(&endpoint, &self.deployment_id)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let builder = http_client
            .post(request_url)
            .header("api-key", api_key.expose_secret());
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
        let stream = stream_openai(
            PROVIDER_TYPE.to_string(),
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
            provider_type: "Azure".to_string(),
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

impl EmbeddingProvider for AzureProvider {
    async fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
        model_provider_data: &EmbeddingProviderRequestInfo,
    ) -> Result<EmbeddingProviderResponse, Error> {
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let endpoint = self.endpoint.get_endpoint(dynamic_api_keys)?;
        let request_url = get_azure_embedding_url(&endpoint, &self.deployment_id)?;
        let request_body = AzureEmbeddingRequest::new(request);

        let request_builder = client
            .post(request_url)
            .header("api-key", api_key.expose_secret());
        let start_time = Instant::now();

        let request_body_value = serde_json::to_value(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Azure embedding request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let (response, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &FullExtraBodyConfig::default(), // No overrides supported
            &Default::default(),             // No extra headers for embeddings yet
            model_provider_data,
            &self.deployment_id,
            request_body_value,
            request_builder,
        )
        .await?;
        if response.status().is_success() {
            let raw_response = response.text().await.map_err(|e| {
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
            let response: AzureEmbeddingResponse =
                serde_json::from_str(&raw_response).map_err(|e| {
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
            Ok(into_embedding_provider_response(
                response,
                request_body,
                latency,
                raw_response,
            )?)
        } else {
            let status = response.status();
            let response_text = response.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
            Err(handle_openai_error(
                &raw_request,
                status,
                &response_text,
                PROVIDER_TYPE,
            ))
        }
    }
}

fn get_azure_chat_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing URL: {e:?}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?
        .push("openai")
        .push("deployments")
        .push(deployment_id)
        .push("chat")
        .push("completions");
    url.query_pairs_mut()
        .append_pair("api-version", AZURE_INFERENCE_API_VERSION);
    Ok(url)
}

fn get_azure_embedding_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();

    url.path_segments_mut()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing URL: {e:?}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?
        .push("openai")
        .push("deployments")
        .push(deployment_id)
        .push("embeddings");
    url.query_pairs_mut()
        .append_pair("api-version", AZURE_INFERENCE_API_VERSION);
    Ok(url)
}

#[derive(Debug, Deserialize)]
struct AzureEmbeddingResponse {
    data: Vec<AzureEmbeddingData>,
    usage: OpenAIEmbeddingUsage,
}

#[derive(Debug, Deserialize)]
struct AzureEmbeddingData {
    embedding: Embedding,
}

fn into_embedding_provider_response(
    response: AzureEmbeddingResponse,
    request_body: AzureEmbeddingRequest,
    latency: Latency,
    raw_response: String,
) -> Result<EmbeddingProviderResponse, Error> {
    let raw_request = serde_json::to_string(&request_body).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!(
                "Error serializing request body as JSON: {}",
                DisplayOrDebugGateway::new(e)
            ),
            raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
            raw_response: None,
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;
    let embeddings = response
        .data
        .into_iter()
        .map(|data| data.embedding)
        .collect();
    Ok(EmbeddingProviderResponse::new(
        embeddings,
        request_body.input.clone(),
        raw_request,
        raw_response,
        response.usage.into(),
        latency,
    ))
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(untagged)]
enum AzureToolChoice<'a> {
    String(AzureToolChoiceString),
    Specific(ChatCompletionSpecificToolChoice<'a>),
    AllowedTools(ChatCompletionAllowedToolsChoice<'a>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum AzureToolChoiceString {
    None,
    Auto,
    // Note: Azure doesn't support required tool choice.
}

impl<'a> From<ChatCompletionToolChoice<'a>> for AzureToolChoice<'a> {
    fn from(tool_choice: ChatCompletionToolChoice<'a>) -> Self {
        match tool_choice {
            ChatCompletionToolChoice::String(tool_choice) => match tool_choice {
                ChatCompletionToolChoiceString::None => {
                    AzureToolChoice::String(AzureToolChoiceString::None)
                }
                ChatCompletionToolChoiceString::Auto => {
                    AzureToolChoice::String(AzureToolChoiceString::Auto)
                }
                ChatCompletionToolChoiceString::Required => {
                    AzureToolChoice::String(AzureToolChoiceString::Auto)
                } // Azure doesn't support required
            },
            ChatCompletionToolChoice::Specific(tool_choice) => {
                AzureToolChoice::Specific(tool_choice)
            }
            ChatCompletionToolChoice::AllowedTools(allowed_tools_choice) => {
                // Convert from common ChatCompletionAllowedToolsChoice to Azure/OpenAI AllowedToolsChoice
                AzureToolChoice::AllowedTools(allowed_tools_choice)
            }
        }
    }
}

/// This struct defines the supported parameters for the Azure OpenAI inference API
/// See the [API documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, seed, service_tier, stop, user,
/// or context_length_exceeded_behavior
#[derive(Debug, Serialize)]
struct AzureRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<AzureResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ChatCompletionTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AzureToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<String>,
}

fn apply_inference_params(
    request: &mut AzureRequest,
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

    // Azure supports auto and default, but not flex and priority
    if let Some(tier) = service_tier {
        match tier {
            ServiceTier::Auto | ServiceTier::Default => {
                request.service_tier = Some(tier.clone());
            }
            ServiceTier::Flex | ServiceTier::Priority => {
                warn_inference_parameter_not_supported(
                    PROVIDER_NAME,
                    &format!("service_tier ({tier})"),
                    None,
                );
            }
        }
    }

    if thinking_budget_tokens.is_some() {
        warn_inference_parameter_not_supported(
            PROVIDER_NAME,
            "thinking_budget_tokens",
            Some("Tip: You might want to use `reasoning_effort` for this provider."),
        );
    }

    if verbosity.is_some() {
        request.verbosity = verbosity.clone();
    }
}

impl<'a> AzureRequest<'a> {
    pub async fn new(request: &'a ModelInferenceRequest<'_>) -> Result<AzureRequest<'a>, Error> {
        let response_format = AzureResponseFormat::new(request.json_mode, request.output_schema);
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
        let (tools, tool_choice, _) = prepare_chat_completion_tools(request, true)?;
        let mut azure_request = AzureRequest {
            messages,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.borrow_stop_sequences(),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_completion_tokens: request.max_tokens,
            stream: request.stream,
            response_format,
            seed: request.seed,
            tools,
            // allowed_tools is now part of tool_choice (AllowedToolsChoice variant)
            tool_choice: tool_choice.map(AzureToolChoice::from),
            reasoning_effort: None,
            service_tier: None, // handled below
            verbosity: None,
        };

        apply_inference_params(&mut azure_request, &request.inference_params_v2);

        Ok(azure_request)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum AzureResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: Value,
    },
}

impl AzureResponseFormat {
    fn new(
        json_mode: ModelInferenceRequestJsonMode,
        output_schema: Option<&Value>,
    ) -> Option<Self> {
        // Note: Some models on Azure won't support strict JSON mode.
        // Azure will 400 if you try to use it for those.
        // See these docs: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs
        match json_mode {
            ModelInferenceRequestJsonMode::On => Some(AzureResponseFormat::JsonObject),
            // For now, we never explicitly send `AzureResponseFormat::Text`
            ModelInferenceRequestJsonMode::Off => None,
            ModelInferenceRequestJsonMode::Strict => match output_schema {
                Some(schema) => {
                    let json_schema = json!({"name": "response", "strict": true, "schema": schema});
                    Some(AzureResponseFormat::JsonSchema { json_schema })
                }
                None => Some(AzureResponseFormat::JsonObject),
            },
        }
    }
}

struct AzureResponseWithMetadata<'a> {
    response: OpenAIResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<AzureResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: AzureResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let AzureResponseWithMetadata {
            mut response,
            latency,
            raw_request,
            generic_request,
            raw_response,
        } = value;

        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_response.clone()),
            }
            .into());
        }
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();
        let usage = response.usage.into();
        let OpenAIResponseChoice {
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

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response,
                usage,
                latency,
                finish_reason: Some(finish_reason.into()),
            },
        ))
    }
}

#[derive(Debug, Serialize)]
struct AzureEmbeddingRequest<'a> {
    input: &'a EmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    encoding_format: EmbeddingEncodingFormat,
}

impl<'a> AzureEmbeddingRequest<'a> {
    fn new(request: &'a EmbeddingRequest) -> Self {
        Self {
            input: &request.input,
            dimensions: request.dimensions,
            encoding_format: request.encoding_format,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::SecretString;
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::time::Duration;
    use uuid::Uuid;

    use crate::config::SKIP_CREDENTIAL_VALIDATION;
    use crate::inference::types::{
        FinishReason, FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };
    use crate::model::EndpointLocation;
    use crate::providers::chat_completions::{
        ChatCompletionSpecificToolChoice, ChatCompletionSpecificToolFunction,
        ChatCompletionToolChoice, ChatCompletionToolChoiceString, ChatCompletionToolType,
    };
    use crate::providers::openai::{
        OpenAIFinishReason, OpenAIResponseChoice, OpenAIResponseMessage, OpenAIUsage,
    };
    use crate::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};

    #[tokio::test]
    async fn test_azure_request_new() {
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

        let azure_request = AzureRequest::new(&request_with_tools).await.unwrap();

        assert_eq!(azure_request.messages.len(), 1);
        assert_eq!(azure_request.temperature, Some(0.5));
        assert_eq!(azure_request.max_completion_tokens, Some(100));
        assert!(!azure_request.stream);
        assert_eq!(azure_request.seed, Some(69));
        assert_eq!(azure_request.response_format, None);
        assert!(azure_request.tools.is_some());
        let tools = azure_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            azure_request.tool_choice,
            Some(AzureToolChoice::Specific(
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

        let azure_request = AzureRequest::new(&request_with_tools).await.unwrap();

        assert_eq!(azure_request.messages.len(), 2);
        assert_eq!(azure_request.temperature, Some(0.5));
        assert_eq!(azure_request.max_completion_tokens, Some(100));
        assert_eq!(azure_request.top_p, Some(0.9));
        assert_eq!(azure_request.presence_penalty, Some(0.1));
        assert_eq!(azure_request.frequency_penalty, Some(0.2));
        assert!(!azure_request.stream);
        assert_eq!(azure_request.seed, Some(69));
        assert_eq!(
            azure_request.response_format,
            Some(AzureResponseFormat::JsonObject)
        );
        assert!(azure_request.tools.is_some());
        let tools = azure_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            azure_request.tool_choice,
            Some(AzureToolChoice::Specific(
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
    fn test_azure_tool_choice_from() {
        // Required is converted to Auto
        let tool_choice =
            ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Required);
        let azure_tool_choice = AzureToolChoice::from(tool_choice);
        assert_eq!(
            azure_tool_choice,
            AzureToolChoice::String(AzureToolChoiceString::Auto)
        );

        // Specific tool choice is converted to Specific
        let specific_tool_choice =
            ChatCompletionToolChoice::Specific(ChatCompletionSpecificToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: ChatCompletionSpecificToolFunction {
                    name: "test_function",
                },
            });
        let azure_specific_tool_choice = AzureToolChoice::from(specific_tool_choice);
        assert_eq!(
            azure_specific_tool_choice,
            AzureToolChoice::Specific(ChatCompletionSpecificToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: ChatCompletionSpecificToolFunction {
                    name: "test_function",
                }
            })
        );

        // None is converted to None
        let none_tool_choice =
            ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::None);
        let azure_none_tool_choice = AzureToolChoice::from(none_tool_choice);
        assert_eq!(
            azure_none_tool_choice,
            AzureToolChoice::String(AzureToolChoiceString::None)
        );

        // Auto is converted to Auto
        let auto_tool_choice =
            ChatCompletionToolChoice::String(ChatCompletionToolChoiceString::Auto);
        let azure_auto_tool_choice = AzureToolChoice::from(auto_tool_choice);
        assert_eq!(
            azure_auto_tool_choice,
            AzureToolChoice::String(AzureToolChoiceString::Auto)
        );
    }

    #[test]
    fn test_credential_to_azure_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = AzureCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, AzureCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = AzureCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, AzureCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = AzureCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, AzureCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = AzureCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[tokio::test]
    async fn test_azure_response_with_metadata_try_into() {
        let valid_response = OpenAIResponse {
            choices: vec![OpenAIResponseChoice {
                index: 0,
                message: OpenAIResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: OpenAIFinishReason::Stop,
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
        let azure_response_with_metadata = AzureResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: serde_json::to_string(&AzureRequest::new(&generic_request).await.unwrap())
                .unwrap(),
            generic_request: &generic_request,
        };
        let inference_response: ProviderInferenceResponse =
            azure_response_with_metadata.try_into().unwrap();

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

    #[tokio::test]
    async fn test_azure_provider_with_static_endpoint() {
        // Run in credential validation skip context to avoid API key requirement
        let provider = SKIP_CREDENTIAL_VALIDATION
            .scope((), async {
                AzureProvider::new(
                    "gpt-4.1-mini".to_string(),
                    EndpointLocation::Static("https://test.openai.azure.com".to_string()),
                    AzureCredentials::None,
                )
            })
            .await
            .unwrap();

        assert_eq!(provider.deployment_id(), "gpt-4.1-mini");
        match provider.endpoint {
            AzureEndpoint::Static(url) => {
                assert_eq!(url.as_str(), "https://test.openai.azure.com/");
            }
            AzureEndpoint::Dynamic(_) => panic!("Expected static endpoint"),
        }
    }

    #[tokio::test]
    async fn test_azure_provider_with_dynamic_endpoint() {
        // Run in credential validation skip context to avoid API key requirement
        let provider = SKIP_CREDENTIAL_VALIDATION
            .scope((), async {
                AzureProvider::new(
                    "gpt-4.1-mini".to_string(),
                    EndpointLocation::Dynamic("azure_endpoint".to_string()),
                    AzureCredentials::None,
                )
            })
            .await
            .unwrap();

        assert_eq!(provider.deployment_id(), "gpt-4.1-mini");
        match provider.endpoint {
            AzureEndpoint::Dynamic(key) => {
                assert_eq!(key, "azure_endpoint");
            }
            AzureEndpoint::Static(_) => panic!("Expected dynamic endpoint"),
        }
    }

    #[test]
    fn test_azure_endpoint_resolution() {
        let endpoint = AzureEndpoint::Dynamic("test_endpoint".to_string());
        let mut credentials = HashMap::new();
        credentials.insert(
            "test_endpoint".to_string(),
            SecretString::from("https://dynamic.openai.azure.com"),
        );

        let resolved = endpoint.get_endpoint(&credentials).unwrap();
        assert_eq!(resolved.as_str(), "https://dynamic.openai.azure.com/");
    }

    #[test]
    fn test_azure_endpoint_resolution_missing_key() {
        let endpoint = AzureEndpoint::Dynamic("missing_endpoint".to_string());
        let credentials = HashMap::new();

        let result = endpoint.get_endpoint(&credentials);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Dynamic endpoint 'missing_endpoint' not found"));
    }

    #[test]
    fn test_azure_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = AzureRequest {
            messages: vec![],
            temperature: None,
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_completion_tokens: None,
            seed: None,
            stream: false,
            response_format: None,
            tools: None,
            // allowed_tools is now part of tool_choice (AllowedToolsChoice variant)
            tool_choice: None,
            reasoning_effort: None,
            service_tier: None,
            verbosity: None,
        };

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort is applied correctly
        assert_eq!(request.reasoning_effort, Some("high".to_string()));

        // Test that thinking_budget_tokens warns with tip about reasoning_effort
        assert!(logs_contain(
            "Azure does not support the inference parameter `thinking_budget_tokens`, so it will be ignored. Tip: You might want to use `reasoning_effort` for this provider."
        ));

        // Test that verbosity is applied correctly
        assert_eq!(request.verbosity, Some("low".to_string()));
    }
}
