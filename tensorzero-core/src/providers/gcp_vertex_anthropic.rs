use std::borrow::Cow;
use std::fmt::Display;

use futures::future::try_join_all;
use futures::StreamExt;
use reqwest_eventsource::Event;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use tokio::time::Instant;

use super::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};
use crate::cache::ModelProviderRequest;
use crate::config::skip_credential_validation;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::{TensorZeroEventSource, TensorzeroHttpClient};
use crate::inference::types::batch::BatchRequestRow;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2,
};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, FunctionType, Latency,
    ModelInferenceRequestJsonMode,
};
use crate::inference::types::{
    ContentBlockOutput, FlattenUnknown, ModelInferenceRequest,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseArgs, ProviderInferenceResponseStreamInner, Thought, Unknown, Usage,
};
use crate::inference::InferenceProvider;
use crate::model::CredentialLocationWithFallback;
use crate::model::ModelProvider;
use crate::model_table::{GCPVertexAnthropicKind, ProviderType, ProviderTypeDefaultCredentials};
use crate::providers::anthropic::{
    anthropic_to_tensorzero_stream_message, handle_anthropic_error, AnthropicStreamMessage,
    AnthropicToolChoice,
};
use crate::providers::gcp_vertex_gemini::location_subdomain_prefix;
use crate::tool::{ToolCall, ToolChoice};

use super::anthropic::{
    prefill_json_chunk_response, prefill_json_response, AnthropicMessage, AnthropicMessageContent,
    AnthropicMessagesConfig, AnthropicRole, AnthropicStopReason, AnthropicSystemBlock,
    AnthropicTool,
};
use super::gcp_vertex_gemini::{parse_shorthand_url, GCPVertexCredentials, ShorthandUrl};
use super::helpers::{convert_stream_error, peek_first_chunk};

/// Implements a subset of the GCP Vertex Gemini API as documented [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/generateContent) for non-streaming
/// and [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/streamGenerateContent) for streaming
const PROVIDER_NAME: &str = "GCP Vertex Anthropic";
pub const PROVIDER_TYPE: &str = "gcp_vertex_anthropic";

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GCPVertexAnthropicProvider {
    model_id: String,
    request_url: String,
    streaming_request_url: String,
    audience: String,
    #[serde(skip)]
    credentials: GCPVertexCredentials,
}

fn handle_gcp_error(
    // This is only used in test mode
    #[cfg_attr(not(any(test, feature = "e2e_tests")), expect(unused_variables))]
    provider_type: ProviderType,
    e: impl Display + Debug,
) -> Result<GCPVertexCredentials, Error> {
    if skip_credential_validation() {
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            tracing::warn!(
                "Failed to get GCP SDK credentials for a model provider of type `{provider_type}`, so the associated tests will likely fail: {e}",
            );
        }
        Ok(GCPVertexCredentials::None)
    } else {
        Err(Error::new(ErrorDetails::GCPCredentials {
            message: format!(
                "Failed to create GCP Vertex credentials from SDK: {}",
                DisplayOrDebugGateway::new(e)
            ),
        }))
    }
}

pub async fn make_gcp_sdk_credentials(
    provider_type: ProviderType,
) -> Result<GCPVertexCredentials, Error> {
    let creds_result = google_cloud_auth::credentials::Builder::default().build();

    let creds = match creds_result {
        Ok(creds) => creds,
        Err(e) => {
            return handle_gcp_error(provider_type, e);
        }
    };
    // Test that the credentials are valid by getting headers
    match creds.headers(http::Extensions::default()).await {
        Ok(_) => Ok(GCPVertexCredentials::Sdk(creds)),
        Err(e) => handle_gcp_error(provider_type, e),
    }
}

impl GCPVertexAnthropicProvider {
    // Constructs a provider from a shorthand string of the form:
    // * 'projects/<project_id>/locations/<location>/publishers/anthropic/models/XXX'
    // * 'projects/<project_id>/locations/<location>/endpoints/XXX'
    //
    // This is *not* a full url - we append ':generateContent' or ':streamGenerateContent' to the end of the path as needed.
    pub async fn new_shorthand(
        project_url_path: String,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self, Error> {
        let credentials = GCPVertexAnthropicKind
            .get_defaulted_credential(None, default_credentials)
            .await?;

        // We only support model urls with the publisher 'anthropic'
        let shorthand_url = parse_shorthand_url(&project_url_path, "anthropic")?;
        let (location, model_id) = match shorthand_url {
            ShorthandUrl::Publisher { location, model_id } => (location, model_id.to_string()),
            ShorthandUrl::Endpoint {
                location,
                endpoint_id,
            } => (location, format!("endpoints/{endpoint_id}")),
        };

        let location_prefix = location_subdomain_prefix(location);

        let request_url = format!(
            "https://{location_prefix}aiplatform.googleapis.com/v1/{project_url_path}:rawPredict"
        );
        let streaming_request_url = format!(
            "https://{location_prefix}aiplatform.googleapis.com/v1/{project_url_path}:streamRawPredict"
        );
        let audience = format!("https://{location_prefix}aiplatform.googleapis.com/");

        Ok(GCPVertexAnthropicProvider {
            model_id,
            request_url,
            streaming_request_url,
            audience,
            credentials,
        })
    }

    pub async fn new(
        model_id: String,
        location: String,
        project_id: String,
        api_key_location: Option<CredentialLocationWithFallback>,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<Self, Error> {
        let credentials = GCPVertexAnthropicKind
            .get_defaulted_credential(api_key_location.as_ref(), default_credentials)
            .await?;

        let location_prefix = location_subdomain_prefix(&location);

        let request_url = format!("https://{location_prefix}aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:rawPredict");
        let streaming_request_url = format!("https://{location_prefix}aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:streamRawPredict");
        let audience = format!("https://{location_prefix}aiplatform.googleapis.com/");

        Ok(GCPVertexAnthropicProvider {
            model_id,
            request_url,
            streaming_request_url,
            audience,
            credentials,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

const ANTHROPIC_API_VERSION: &str = "vertex-2023-10-16";

impl InferenceProvider for GCPVertexAnthropicProvider {
    /// Anthropic non-streaming API request
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
            GCPVertexAnthropicRequestBody::new(self.model_id(), request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing GCP Vertex Anthropic request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let auth_headers = self
            .credentials
            .get_auth_headers(&self.audience, dynamic_api_keys)
            .await
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let builder = http_client.post(&self.request_url).headers(auth_headers);

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
            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {raw_response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            let response_with_latency = GCPVertexAnthropicResponseWithMetadata {
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

            Ok(response_with_latency.try_into()?)
        } else {
            handle_anthropic_error(response_status, raw_request, raw_response)
        }
    }

    /// Anthropic streaming API request
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
            GCPVertexAnthropicRequestBody::new(self.model_id(), request).await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing GCP Vertex Anthropic request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let auth_headers = self
            .credentials
            .get_auth_headers(&self.audience, dynamic_api_keys)
            .await
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let builder = http_client
            .post(&self.streaming_request_url)
            .headers(auth_headers);
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
            provider_type: "GCP Vertex Anthropic".to_string(),
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

/// Maps events from Anthropic into the TensorZero format
/// Modified from the example [here](https://github.com/64bit/async-openai/blob/5c9c817b095e3bacb2b6c9804864cdf8b15c795e/async-openai/src/client.rs#L433)
/// At a high level, this function is handling low-level EventSource details and mapping the objects returned by Anthropic into our `InferenceResultChunk` type
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
        let mut current_tool_id : Option<String> = None;
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
                                message.data,
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

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexAnthropicThinkingConfig {
    r#type: &'static str,
    budget_tokens: i32,
}

#[derive(Debug, Default, PartialEq, Serialize)]
struct GCPVertexAnthropicRequestBody<'a> {
    anthropic_version: &'static str,
    messages: Vec<AnthropicMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    // This is the system message
    system: Option<Vec<AnthropicSystemBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<GCPVertexAnthropicThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
}

fn apply_inference_params(
    request: &mut GCPVertexAnthropicRequestBody,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() {
        warn_inference_parameter_not_supported(
            PROVIDER_NAME,
            "reasoning_effort",
            Some("Tip: You might want to use `thinking_budget_tokens` for this provider."),
        );
    }

    if let Some(budget_tokens) = thinking_budget_tokens {
        request.thinking = Some(GCPVertexAnthropicThinkingConfig {
            r#type: "enabled",
            budget_tokens: *budget_tokens,
        });
    }

    if service_tier.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier", None);
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
}

impl<'a> GCPVertexAnthropicRequestBody<'a> {
    async fn new(
        model_id: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<GCPVertexAnthropicRequestBody<'a>, Error> {
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
        // We use the content block form rather than string so people can use
        // extra_body for cache control.
        let system = match request.system.as_deref() {
            Some(text) => Some(vec![AnthropicSystemBlock::Text { text }]),
            None => None,
        };
        let request_messages: Vec<AnthropicMessage> =
            try_join_all(request.messages.iter().map(|m| {
                AnthropicMessage::from_request_message(m, messages_config, PROVIDER_TYPE)
            }))
            .await?
            .into_iter()
            .filter(|m| !m.content.is_empty())
            .collect::<Vec<_>>();
        let mut messages = prepare_messages(request_messages)?;
        if matches!(
            request.json_mode,
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict
        ) && matches!(request.function_type, FunctionType::Json)
        {
            prefill_json_message(&mut messages);
        }

        // Workaround for GCP Vertex AI Anthropic API limitation: they don't support explicitly specifying "none"
        // for tool choice. When ToolChoice::None is specified, we don't send any tools in the
        // request payload to achieve the same effect.
        let tools = match request.tool_config.as_ref() {
            Some(c) if !matches!(c.tool_choice, ToolChoice::None) => Some(
                c.strict_tools_available()?
                    // GCP Vertex Anthropic does not support structured outputs
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
            None => get_default_max_tokens(model_id),
        }?;

        // NOTE: Anthropic does not support seed
        let mut gcp_vertex_anthropic_request = GCPVertexAnthropicRequestBody {
            anthropic_version: ANTHROPIC_API_VERSION,
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

        apply_inference_params(
            &mut gcp_vertex_anthropic_request,
            &request.inference_params_v2,
        );

        Ok(gcp_vertex_anthropic_request)
    }
}

/// Returns the default max_tokens for a given GCP Anthropic model name, or an error if unknown.
///
/// GCP Anthropic requires that the user provides `max_tokens`, but the value depends on the model.
/// We maintain a library of known maximum values, and ask the user to hardcode it if it's unknown.
fn get_default_max_tokens(model_id: &str) -> Result<u32, Error> {
    if model_id.starts_with("claude-3-haiku@") {
        // GCP docs say 8k but that causes `max_tokens: XXX > 8192, which is the maximum allowed number of output tokens for claude-3-haiku-20250219`
        Ok(4_096)
    } else if model_id.starts_with("claude-3-5-haiku@")
        || model_id.starts_with("claude-3-5-sonnet@")
        || model_id.starts_with("claude-3-5-sonnet-v2@")
    {
        Ok(8_192)
    } else if model_id.starts_with("claude-3-7-sonnet@") {
        // GCP docs say 128k but that causes `max_tokens: XXX > 64000, which is the maximum allowed number of output tokens for claude-3-7-sonnet-20250219`
        Ok(64_000)
    } else if model_id.starts_with("claude-sonnet-4@")
        || model_id.starts_with("claude-haiku-4-5@")
        || model_id.starts_with("claude-sonnet-4-5@")
        || model_id.starts_with("claude-opus-4-5@")
    {
        Ok(64_000)
    } else if model_id.starts_with("claude-opus-4@") || model_id.starts_with("claude-opus-4-1@") {
        Ok(32_000)
    } else {
        Err(Error::new(ErrorDetails::InferenceClient {
            message: format!("The TensorZero Gateway doesn't know the output token limit for `{model_id}` and GCP Vertex AI Anthropic requires you to provide a `max_tokens` value. Please set `max_tokens` in your configuration or inference request."),
            status_code: None,
            provider_type: PROVIDER_TYPE.into(),
            raw_request: None,
            raw_response: None,
        }))
    }
}

/// Anthropic API doesn't support consecutive messages from the same role.
/// This function consolidates messages from the same role into a single message
/// so as to satisfy the API.
/// It also makes modifications to the messages to make Anthropic happy.
/// For example, it will prepend a default User message if the first message is an Assistant message.
/// It will also append a default User message if the last message is an Assistant message.
fn prepare_messages(messages: Vec<AnthropicMessage>) -> Result<Vec<AnthropicMessage>, Error> {
    let mut consolidated_messages: Vec<AnthropicMessage> = Vec::new();
    let mut last_role: Option<AnthropicRole> = None;
    for message in messages {
        let this_role = message.role.clone();
        match last_role {
            Some(role) => {
                if role == this_role {
                    let mut last_message =
                        consolidated_messages.pop().ok_or_else(|| Error::new(ErrorDetails::InvalidRequest {
                            message: "Last message is missing (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new"
                                .to_string(),
                        }))?;
                    last_message.content.extend(message.content);
                    consolidated_messages.push(last_message);
                } else {
                    consolidated_messages.push(message);
                }
            }
            None => {
                consolidated_messages.push(message);
            }
        }
        last_role = Some(this_role);
    }
    // Anthropic also requires that there is at least one message and it is a User message.
    // If it's not we will prepend a default User message.
    match consolidated_messages.first() {
        Some(&AnthropicMessage {
            role: AnthropicRole::User,
            ..
        }) => {}
        _ => {
            consolidated_messages.insert(
                0,
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "[listening]",
                    })],
                },
            );
        }
    }
    // Anthropic will continue any assistant messages passed in.
    // Since we don't want to do that, we'll append a default User message in the case that the last message was
    // an assistant message
    if let Some(last_message) = consolidated_messages.last() {
        if last_message.role == AnthropicRole::Assistant {
            consolidated_messages.push(AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "[listening]",
                })],
            });
        }
    }
    Ok(consolidated_messages)
}

fn prefill_json_message(messages: &mut Vec<AnthropicMessage>) {
    // Add a JSON-prefill message for Anthropic's JSON mode
    messages.push(AnthropicMessage {
        role: AnthropicRole::Assistant,
        content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
            text: "Here is the JSON requested:\n{",
        })],
    });
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GCPVertexAnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
}

fn convert_to_output(
    model_name: &str,
    provider_name: &str,
    block: FlattenUnknown<'static, GCPVertexAnthropicContentBlock>,
) -> Result<ContentBlockOutput, Error> {
    match block {
        FlattenUnknown::Normal(GCPVertexAnthropicContentBlock::Text { text }) => Ok(text.into()),
        FlattenUnknown::Normal(GCPVertexAnthropicContentBlock::ToolUse { id, name, input }) => {
            Ok(ContentBlockOutput::ToolCall(ToolCall {
                id,
                name,
                arguments: serde_json::to_string(&input).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error parsing input for tool call: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?,
            }))
        }
        FlattenUnknown::Normal(GCPVertexAnthropicContentBlock::Thinking {
            thinking,
            signature,
        }) => Ok(ContentBlockOutput::Thought(Thought {
            text: Some(thinking),
            signature: Some(signature),
            summary: None,
            provider_type: Some(PROVIDER_TYPE.to_string()),
        })),
        FlattenUnknown::Normal(GCPVertexAnthropicContentBlock::RedactedThinking { data }) => {
            Ok(ContentBlockOutput::Thought(Thought {
                text: None,
                signature: Some(data),
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
            }))
        }
        FlattenUnknown::Unknown(obj) => Ok(ContentBlockOutput::Unknown(Unknown {
            data: obj.into_owned(),
            model_name: Some(model_name.to_string()),
            provider_name: Some(provider_name.to_string()),
        })),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GCPVertexAnthropic {
    input_tokens: u32,
    output_tokens: u32,
}

impl From<GCPVertexAnthropic> for Usage {
    fn from(value: GCPVertexAnthropic) -> Self {
        Usage {
            input_tokens: Some(value.input_tokens),
            output_tokens: Some(value.output_tokens),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct GCPVertexAnthropicResponse {
    id: String,
    r#type: String, // this is always "message"
    role: String,   // this is always "assistant"
    content: Vec<FlattenUnknown<'static, GCPVertexAnthropicContentBlock>>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<AnthropicStopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<String>,
    usage: GCPVertexAnthropic,
}

#[derive(Debug)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
struct GCPVertexAnthropicResponseWithMetadata<'a> {
    response: GCPVertexAnthropicResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    function_type: &'a FunctionType,
    json_mode: &'a ModelInferenceRequestJsonMode,
    generic_request: &'a ModelInferenceRequest<'a>,
    model_name: &'a str,
    provider_name: &'a str,
}

impl<'a> TryFrom<GCPVertexAnthropicResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: GCPVertexAnthropicResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let GCPVertexAnthropicResponseWithMetadata {
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

#[cfg(test)]
mod tests {
    use crate::inference::types::FlattenUnknown;
    use std::borrow::Cow;

    use super::*;

    use serde_json::{json, Value};
    use std::time::Duration;
    use uuid::Uuid;

    use crate::inference::types::{
        ContentBlock, FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::tool::{DynamicToolConfig, FunctionToolConfig, ToolResult};

    fn parse_usage_info(usage_info: &Value) -> GCPVertexAnthropic {
        let input_tokens = usage_info
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let output_tokens = usage_info
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        GCPVertexAnthropic {
            input_tokens,
            output_tokens,
        }
    }

    #[tokio::test]
    async fn test_from_tool() {
        let parameters = json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"}
            },
            "required": ["location", "unit"]
        });
        let tool = FunctionToolConfig::Dynamic(DynamicToolConfig {
            name: "test".to_string(),
            description: "test".to_string(),
            parameters: DynamicJSONSchema::new(parameters.clone()),
            strict: false,
        });
        let anthropic_tool: AnthropicTool = AnthropicTool::new(&tool, false);
        assert_eq!(
            anthropic_tool,
            AnthropicTool {
                name: "test",
                description: Some("test"),
                input_schema: &parameters,
                strict: None,
            }
        );
    }

    #[tokio::test]
    async fn test_try_from_content_block() {
        let text_content_block = "test".to_string().into();
        let message_config = AnthropicMessagesConfig {
            fetch_and_encode_input_files_before_inference: false,
        };
        let anthropic_content_block = AnthropicMessageContent::from_content_block(
            &text_content_block,
            message_config,
            PROVIDER_TYPE,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            anthropic_content_block,
            FlattenUnknown::Normal(AnthropicMessageContent::Text { text: "test" })
        );

        let tool_call_content_block = ContentBlock::ToolCall(ToolCall {
            id: "test_id".to_string(),
            name: "test_name".to_string(),
            arguments: serde_json::to_string(&json!({"type": "string"})).unwrap(),
        });
        let anthropic_content_block = AnthropicMessageContent::from_content_block(
            &tool_call_content_block,
            message_config,
            PROVIDER_TYPE,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            anthropic_content_block,
            FlattenUnknown::Normal(AnthropicMessageContent::ToolUse {
                id: "test_id",
                name: "test_name",
                input: json!({"type": "string"})
            })
        );
    }

    #[tokio::test]
    async fn test_initialize_anthropic_request_body() {
        let listening_message = AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "[listening]",
            })],
        };
        // Test Case 1: Empty message list
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let anthropic_request_body =
            GCPVertexAnthropicRequestBody::new("claude-opus-4@20250514", &inference_request).await;
        let error = anthropic_request_body.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            }
        );

        // Test Case 2: Messages with System message
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
        ];
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let message_config = AnthropicMessagesConfig {
            fetch_and_encode_input_files_before_inference: false,
        };
        let anthropic_request_body =
            GCPVertexAnthropicRequestBody::new("claude-opus-4@20250514", &inference_request).await;
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            GCPVertexAnthropicRequestBody {
                anthropic_version: ANTHROPIC_API_VERSION,
                messages: vec![
                    AnthropicMessage::from_request_message(
                        &messages[0],
                        message_config,
                        PROVIDER_TYPE
                    )
                    .await
                    .unwrap(),
                    AnthropicMessage::from_request_message(
                        &messages[1],
                        message_config,
                        PROVIDER_TYPE
                    )
                    .await
                    .unwrap(),
                    listening_message.clone(),
                ],
                max_tokens: 32_000,
                stream: Some(false),
                system: Some(vec![AnthropicSystemBlock::Text {
                    text: "test_system"
                }]),
                ..Default::default()
            }
        );

        // Test case 3: Messages with system message that require consolidation
        // also some of the optional fields are tested
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::User,
                content: vec!["test_user2".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
        ];
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: None,
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::On,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let anthropic_request_body =
            GCPVertexAnthropicRequestBody::new("claude-opus-4@20250514", &inference_request).await;
        let message_config = AnthropicMessagesConfig {
            fetch_and_encode_input_files_before_inference: false,
        };
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            GCPVertexAnthropicRequestBody {
                anthropic_version: ANTHROPIC_API_VERSION,
                messages: vec![
                    AnthropicMessage {
                        role: AnthropicRole::User,
                        content: vec![
                            FlattenUnknown::Normal(AnthropicMessageContent::Text {
                                text: "test_user"
                            }),
                            FlattenUnknown::Normal(AnthropicMessageContent::Text {
                                text: "test_user2"
                            })
                        ],
                    },
                    AnthropicMessage::from_request_message(
                        &messages[2],
                        message_config,
                        PROVIDER_TYPE
                    )
                    .await
                    .unwrap(),
                    listening_message.clone(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some(vec![AnthropicSystemBlock::Text {
                    text: "test_system"
                }]),
                temperature: Some(0.5),
                top_p: Some(0.9),
                ..Default::default()
            }
        );

        // Test case 4: Tool use & choice
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
            RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(ToolResult {
                    id: "tool_call_id".to_string(),
                    name: "test_tool_name".to_string(),
                    result: "tool_response".to_string(),
                })],
            },
        ];

        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::On,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let anthropic_request_body =
            GCPVertexAnthropicRequestBody::new("claude-opus-4@20250514", &inference_request).await;
        let message_config = AnthropicMessagesConfig {
            fetch_and_encode_input_files_before_inference: false,
        };
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            GCPVertexAnthropicRequestBody {
                anthropic_version: ANTHROPIC_API_VERSION,
                messages: vec![
                    AnthropicMessage::from_request_message(
                        &messages[0],
                        message_config,
                        PROVIDER_TYPE
                    )
                    .await
                    .unwrap(),
                    AnthropicMessage::from_request_message(
                        &messages[1],
                        message_config,
                        PROVIDER_TYPE
                    )
                    .await
                    .unwrap(),
                    AnthropicMessage::from_request_message(
                        &messages[2],
                        message_config,
                        PROVIDER_TYPE
                    )
                    .await
                    .unwrap(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some(vec![AnthropicSystemBlock::Text {
                    text: "test_system"
                }]),
                temperature: Some(0.5),
                top_p: Some(0.9),
                tool_choice: Some(AnthropicToolChoice::Tool {
                    name: "get_temperature",
                    disable_parallel_tool_use: Some(false),
                }),
                tools: Some(vec![AnthropicTool {
                    name: WEATHER_TOOL.name(),
                    description: Some(WEATHER_TOOL.description()),
                    input_schema: WEATHER_TOOL.parameters(),
                    strict: None,
                }]),
                ..Default::default()
            }
        );
    }

    #[tokio::test]
    async fn test_get_default_max_tokens_in_new_gcp_anthropic_request_body() {
        let messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into()],
        }];

        let request = ModelInferenceRequest {
            messages: messages.clone(),
            ..Default::default()
        };

        let request_with_max_tokens = ModelInferenceRequest {
            messages,
            max_tokens: Some(100),
            ..Default::default()
        };

        let model = "claude-opus-4-1@20250805".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 32_000);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-opus-4@20250514".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 32_000);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-sonnet-4@20250514".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-7-sonnet@20250219".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-5-sonnet-v2@20240222".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 8_192);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-5-sonnet@20240229".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 8_192);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-5-haiku@20240307".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 8_192);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-haiku@20240307".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 4_096);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-haiku-4-5@20251001".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-sonnet-4-5@20250929".to_string();
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-sonnet-4".to_string(); // fake model
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert!(body.is_err());
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-4-5-ballad@20260101".to_string(); // fake model
        let body = GCPVertexAnthropicRequestBody::new(&model, &request).await;
        assert!(body.is_err());
        let body = GCPVertexAnthropicRequestBody::new(&model, &request_with_max_tokens).await;
        assert_eq!(body.unwrap().max_tokens, 100);
    }

    #[test]
    fn test_consolidate_messages() {
        let listening_message = AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "[listening]",
            })],
        };
        // Test case 1: No consolidation needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hello",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hi",
                })],
            },
        ];
        let expected = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hello",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hi",
                })],
            },
            listening_message.clone(),
        ];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 2: Consolidation needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hello",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "How are you?",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hi",
                })],
            },
        ];
        let expected = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![
                    FlattenUnknown::Normal(AnthropicMessageContent::Text { text: "Hello" }),
                    FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "How are you?",
                    }),
                ],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hi",
                })],
            },
            listening_message.clone(),
        ];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 3: Multiple consolidations needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hello",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "How are you?",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hi",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "I am here to help.",
                })],
            },
        ];
        let expected = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![
                    FlattenUnknown::Normal(AnthropicMessageContent::Text { text: "Hello" }),
                    FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "How are you?",
                    }),
                ],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![
                    FlattenUnknown::Normal(AnthropicMessageContent::Text { text: "Hi" }),
                    FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "I am here to help.",
                    }),
                ],
            },
            listening_message.clone(),
        ];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 4: No messages
        let messages: Vec<AnthropicMessage> = vec![];
        let expected: Vec<AnthropicMessage> = vec![listening_message.clone()];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 5: Single message
        let messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Hello",
            })],
        }];
        let expected = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Hello",
            })],
        }];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 6: Consolidate tool uses
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(
                    AnthropicMessageContent::ToolResult {
                        tool_use_id: "tool1",
                        content: vec![AnthropicMessageContent::Text {
                            text: "Tool call 1",
                        }],
                    },
                )],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(
                    AnthropicMessageContent::ToolResult {
                        tool_use_id: "tool2",
                        content: vec![AnthropicMessageContent::Text {
                            text: "Tool call 2",
                        }],
                    },
                )],
            },
        ];
        let expected = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![
                FlattenUnknown::Normal(AnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![AnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                }),
                FlattenUnknown::Normal(AnthropicMessageContent::ToolResult {
                    tool_use_id: "tool2",
                    content: vec![AnthropicMessageContent::Text {
                        text: "Tool call 2",
                    }],
                }),
            ],
        }];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 7: Consolidate mixed text and tool use
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "User message 1",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(
                    AnthropicMessageContent::ToolResult {
                        tool_use_id: "tool1",
                        content: vec![AnthropicMessageContent::Text {
                            text: "Tool call 1",
                        }],
                    },
                )],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "User message 2",
                })],
            },
        ];
        let expected = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![
                FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "User message 1",
                }),
                FlattenUnknown::Normal(AnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![AnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                }),
                FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "User message 2",
                }),
            ],
        }];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);
    }

    #[test]
    fn test_anthropic_usage_to_usage() {
        let anthropic_usage = GCPVertexAnthropic {
            input_tokens: 100,
            output_tokens: 50,
        };

        let usage: Usage = anthropic_usage.into();

        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(50));
    }

    #[test]
    fn test_anthropic_response_conversion() {
        // Test case 1: Text response and unknown content
        let anthropic_response_body = GCPVertexAnthropicResponse {
            id: "1".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                FlattenUnknown::Normal(GCPVertexAnthropicContentBlock::Text {
                    text: "Response text".to_string(),
                }),
                FlattenUnknown::Unknown(Cow::Owned(json!({"my_custom": "content"}))),
            ],
            model: "model-name".into(),
            stop_reason: Some(AnthropicStopReason::EndTurn),
            stop_sequence: Some("stop sequence".to_string()),
            usage: GCPVertexAnthropic {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            system: Some("system".to_string()),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["Hello".to_string().into()],
            }],
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = GCPVertexAnthropicRequestBody {
            anthropic_version: "1.0",
            messages: vec![],
            stream: Some(false),
            max_tokens: 1000,
            ..Default::default()
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test response".to_string();
        let body_with_latency = GCPVertexAnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            raw_response: raw_response.clone(),
            latency: latency.clone(),
            raw_request: raw_request.clone(),
            function_type: &FunctionType::Chat,
            json_mode: &ModelInferenceRequestJsonMode::Off,
            generic_request: &generic_request,
            model_name: "my-model",
            provider_name: "my-provider",
        };

        let inference_response = ProviderInferenceResponse::try_from(body_with_latency).unwrap();
        assert_eq!(
            inference_response.output,
            vec![
                "Response text".to_string().into(),
                ContentBlockOutput::Unknown(Unknown {
                    data: serde_json::json!({"my_custom": "content"}),
                    model_name: Some("my-model".to_string()),
                    provider_name: Some("my-provider".to_string()),
                })
            ]
        );

        assert_eq!(raw_response, inference_response.raw_response);
        assert_eq!(inference_response.usage.input_tokens, Some(100));
        assert_eq!(inference_response.usage.output_tokens, Some(50));
        assert_eq!(inference_response.latency, latency);
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.system, Some("system".to_string()));
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["Hello".to_string().into(),],
            }]
        );
        // Test case 2: Tool call response
        let anthropic_response_body = GCPVertexAnthropicResponse {
            id: "2".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![FlattenUnknown::Normal(
                GCPVertexAnthropicContentBlock::ToolUse {
                    id: "tool_call_1".to_string(),
                    name: "get_temperature".to_string(),
                    input: json!({"location": "New York"}),
                },
            )],
            model: "model-name".into(),
            stop_reason: Some(AnthropicStopReason::ToolUse),
            stop_sequence: None,
            usage: GCPVertexAnthropic {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            system: None,
            messages: vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["Hello2".to_string().into()],
            }],
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = GCPVertexAnthropicRequestBody {
            anthropic_version: "1.0",
            messages: vec![],
            max_tokens: 1000,
            ..Default::default()
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = GCPVertexAnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            raw_response: raw_response.clone(),
            latency: latency.clone(),
            raw_request: raw_request.clone(),
            function_type: &FunctionType::Chat,
            json_mode: &ModelInferenceRequestJsonMode::Off,
            generic_request: &generic_request,
            model_name: "model-name",
            provider_name: "provider-name",
        };

        let inference_response: ProviderInferenceResponse = body_with_latency.try_into().unwrap();
        assert!(inference_response.output.len() == 1);
        assert_eq!(
            inference_response.output[0],
            ContentBlockOutput::ToolCall(ToolCall {
                id: "tool_call_1".to_string(),
                name: "get_temperature".to_string(),
                arguments: r#"{"location":"New York"}"#.to_string(),
            })
        );

        assert_eq!(raw_response, inference_response.raw_response);
        assert_eq!(inference_response.usage.input_tokens, Some(100));
        assert_eq!(inference_response.usage.output_tokens, Some(50));
        assert_eq!(inference_response.latency, latency);
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.system, None);
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["Hello2".to_string().into()],
            }]
        );
        // Test case 3: Mixed response (text and tool call)
        let anthropic_response_body = GCPVertexAnthropicResponse {
            id: "3".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                FlattenUnknown::Normal(GCPVertexAnthropicContentBlock::Text {
                    text: "Here's the weather:".to_string(),
                }),
                FlattenUnknown::Normal(GCPVertexAnthropicContentBlock::ToolUse {
                    id: "tool_call_2".to_string(),
                    name: "get_temperature".to_string(),
                    input: json!({"location": "London"}),
                }),
            ],
            model: "model-name".into(),
            stop_reason: None,
            stop_sequence: None,
            usage: GCPVertexAnthropic {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            system: None,
            messages: vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["Hello3".to_string().into()],
            }],
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = GCPVertexAnthropicRequestBody {
            anthropic_version: "1.0",
            messages: vec![],
            max_tokens: 1000,
            ..Default::default()
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = GCPVertexAnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            raw_response: raw_response.clone(),
            latency: latency.clone(),
            raw_request: raw_request.clone(),
            function_type: &FunctionType::Chat,
            json_mode: &ModelInferenceRequestJsonMode::Off,
            generic_request: &generic_request,
            model_name: "model-name",
            provider_name: "provider-name",
        };
        let inference_response = ProviderInferenceResponse::try_from(body_with_latency).unwrap();
        assert_eq!(
            inference_response.output[0],
            "Here's the weather:".to_string().into()
        );
        assert!(inference_response.output.len() == 2);
        assert_eq!(
            inference_response.output[1],
            ContentBlockOutput::ToolCall(ToolCall {
                id: "tool_call_2".to_string(),
                name: "get_temperature".to_string(),
                arguments: r#"{"location":"London"}"#.to_string(),
            })
        );

        assert_eq!(raw_response, inference_response.raw_response);

        assert_eq!(inference_response.usage.input_tokens, Some(100));
        assert_eq!(inference_response.usage.output_tokens, Some(50));
        assert_eq!(inference_response.latency, latency);
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.system, None);
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["Hello3".to_string().into()],
            }]
        );
    }

    #[test]
    fn test_parse_usage_info() {
        // Test with valid input
        let usage_info = json!({
            "input_tokens": 100,
            "output_tokens": 200
        });
        let result = parse_usage_info(&usage_info);
        assert_eq!(result.input_tokens, 100);
        assert_eq!(result.output_tokens, 200);

        // Test with missing fields
        let usage_info = json!({
            "input_tokens": 50
        });
        let result = parse_usage_info(&usage_info);
        assert_eq!(result.input_tokens, 50);
        assert_eq!(result.output_tokens, 0);

        // Test with empty object
        let usage_info = json!({});
        let result = parse_usage_info(&usage_info);
        assert_eq!(result.input_tokens, 0);
        assert_eq!(result.output_tokens, 0);

        // Test with non-numeric values
        let usage_info = json!({
            "input_tokens": "not a number",
            "output_tokens": true
        });
        let result = parse_usage_info(&usage_info);
        assert_eq!(result.input_tokens, 0);
        assert_eq!(result.output_tokens, 0);
    }

    #[test]
    fn test_prefill_json_message() {
        let input_messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Generate some JSON",
            })],
        }];

        let mut result = input_messages.clone();
        prefill_json_message(&mut result);

        assert_eq!(result.len(), 2);

        assert_eq!(result[0].role, AnthropicRole::User);
        assert_eq!(
            result[0].content,
            vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Generate some JSON",
            })]
        );

        assert_eq!(result[1].role, AnthropicRole::Assistant);
        assert_eq!(
            result[1].content,
            vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Here is the JSON requested:\n{",
            })]
        );
    }

    #[test]
    fn test_gcp_vertex_anthropic_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = GCPVertexAnthropicRequestBody::default();

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort warns with tip about thinking_budget_tokens
        assert!(logs_contain(
            "GCP Vertex Anthropic does not support the inference parameter `reasoning_effort`, so it will be ignored. Tip: You might want to use `thinking_budget_tokens` for this provider."
        ));

        // Test that thinking_budget_tokens is applied correctly
        assert_eq!(
            request.thinking,
            Some(GCPVertexAnthropicThinkingConfig {
                r#type: "enabled",
                budget_tokens: 1024,
            })
        );

        // Test that verbosity warns
        assert!(logs_contain(
            "GCP Vertex Anthropic does not support the inference parameter `verbosity`, so it will be ignored."
        ));
    }
}
