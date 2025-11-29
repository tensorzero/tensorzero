use futures::future::try_join_all;
use futures::StreamExt;
use lazy_static::lazy_static;
use mime::MediaType;
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;

use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{
    warn_discarded_unknown_chunk, DelayedError, DisplayOrDebugGateway, Error, ErrorDetails,
};
use crate::http::{TensorZeroEventSource, TensorzeroHttpClient};
use crate::inference::types::batch::BatchRequestRow;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2, ServiceTier,
};
use crate::inference::types::resolved_input::{FileUrl, LazyFile};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, ContentBlock, ContentBlockChunk, FinishReason,
    FunctionType, Latency, ModelInferenceRequestJsonMode, ObjectStorageFile, Role, Text, Unknown,
};
use crate::inference::types::{
    ContentBlockOutput, FlattenUnknown, ModelInferenceRequest,
    PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, RequestMessage, TextChunk, Thought, ThoughtChunk,
    UnknownChunk, Usage,
};
use crate::inference::InferenceProvider;
use crate::model::{Credential, ModelProvider};
use crate::providers;
use crate::providers::helpers::{
    inject_extra_request_data_and_send, inject_extra_request_data_and_send_eventsource,
};
use crate::tool::{FunctionToolConfig, ToolCall, ToolCallChunk, ToolCallConfig, ToolChoice};

use super::helpers::convert_stream_error;
use super::helpers::{peek_first_chunk, warn_cannot_forward_url_if_missing_mime_type};

lazy_static! {
    static ref ANTHROPIC_DEFAULT_BASE_URL: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.anthropic.com/v1/messages")
            .expect("Failed to parse ANTHROPIC_DEFAULT_BASE_URL")
    };
}
const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const PROVIDER_NAME: &str = "Anthropic";
pub const PROVIDER_TYPE: &str = "anthropic";

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct AnthropicProvider {
    model_name: String,
    api_base: Option<Url>,
    #[serde(skip)]
    credentials: AnthropicCredentials,
    beta_structured_outputs: bool,
}

impl AnthropicProvider {
    pub fn new(
        model_name: String,
        api_base: Option<Url>,
        credentials: AnthropicCredentials,
        beta_structured_outputs: bool,
    ) -> Self {
        AnthropicProvider {
            model_name,
            api_base,
            credentials,
            beta_structured_outputs,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    fn base_url(&self) -> &Url {
        self.api_base
            .as_ref()
            .unwrap_or(&ANTHROPIC_DEFAULT_BASE_URL)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum AnthropicCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<AnthropicCredentials>,
        fallback: Box<AnthropicCredentials>,
    },
}

impl TryFrom<Credential> for AnthropicCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(AnthropicCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(AnthropicCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(AnthropicCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(AnthropicCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Anthropic provider".to_string(),
            })),
        }
    }
}

impl AnthropicCredentials {
    fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            AnthropicCredentials::Static(api_key) => Ok(api_key),
            AnthropicCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            AnthropicCredentials::WithFallback { default, fallback } => {
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
            AnthropicCredentials::None => Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
                message: "No credentials are set".to_string(),
            })),
        }
    }
}

impl InferenceProvider for AnthropicProvider {
    /// Anthropic non-streaming API request
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name: tensorzero_model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = serde_json::to_value(
            AnthropicRequestBody::new(&self.model_name, request, self.beta_structured_outputs)
                .await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Anthropic request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let mut builder = http_client
            .post(self.base_url().as_ref())
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("x-api-key", api_key.expose_secret());

        if self.beta_structured_outputs {
            builder = builder.header("anthropic-beta", "structured-outputs-2025-11-13");
        }

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            tensorzero_model_name,
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
                        "Error parsing JSON response: {}: {raw_response}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            let response_with_latency = AnthropicResponseWithMetadata {
                response,
                latency,
                raw_request,
                generic_request: request,
                input_messages: request.messages.clone(),
                raw_response,
                model_name: tensorzero_model_name,
                provider_name: &model_provider.name,
                beta_structured_outputs: self.beta_structured_outputs,
            };
            Ok(response_with_latency.try_into()?)
        } else {
            let response_code = res.status();
            let response_text = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error fetching response: {}", DisplayOrDebugGateway::new(e)),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
            handle_anthropic_error(response_code, raw_request, response_text)
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
        api_key: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body = serde_json::to_value(
            AnthropicRequestBody::new(&self.model_name, request, self.beta_structured_outputs)
                .await?,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Anthropic request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let start_time = Instant::now();
        let api_key = self.credentials.get_api_key(api_key).map_err(|e| e.log())?;
        let mut builder = http_client
            .post(self.base_url().as_ref())
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("x-api-key", api_key.expose_secret());

        if self.beta_structured_outputs {
            builder = builder.header("anthropic-beta", "structured-outputs-2025-11-13");
        }

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
        if needs_json_prefill(request, self.beta_structured_outputs) {
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
            provider_type: "Anthropic".to_string(),
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
                                raw_request: Some(raw_request.to_string()),
                                raw_response: Some(message.data.clone()),
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

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
/// Anthropic doesn't handle the system message in this way
/// It's a field of the POST body instead
pub(super) enum AnthropicRole {
    User,
    Assistant,
}

impl From<Role> for AnthropicRole {
    fn from(role: Role) -> Self {
        match role {
            Role::User => AnthropicRole::User,
            Role::Assistant => AnthropicRole::Assistant,
        }
    }
}

/// We can instruct Anthropic to use a particular tool,
/// any tool (but to use one), or to use a tool if needed.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum AnthropicToolChoice<'a> {
    Auto {
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: &'a str,
        disable_parallel_tool_use: Option<bool>,
    },
}

// We map our ToolCallConfig struct to the AnthropicToolChoice that serializes properly
impl<'a> TryFrom<&'a ToolCallConfig> for AnthropicToolChoice<'a> {
    type Error = Error;

    fn try_from(tool_call_config: &'a ToolCallConfig) -> Result<Self, Error> {
        let disable_parallel_tool_use = Some(tool_call_config.parallel_tool_calls == Some(false));
        let tool_choice = &tool_call_config.tool_choice;

        match tool_choice {
            ToolChoice::Auto => Ok(AnthropicToolChoice::Auto {
                disable_parallel_tool_use,
            }),
            ToolChoice::Required => Ok(AnthropicToolChoice::Any {
                disable_parallel_tool_use,
            }),
            ToolChoice::Specific(name) => Ok(AnthropicToolChoice::Tool {
                name,
                disable_parallel_tool_use,
            }),
            ToolChoice::None => Ok(AnthropicToolChoice::Auto {
                disable_parallel_tool_use,
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct AnthropicTool<'a> {
    pub(super) name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<&'a str>,
    pub(super) input_schema: &'a Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) strict: Option<bool>,
}

impl<'a> AnthropicTool<'a> {
    pub fn new(tool: &'a FunctionToolConfig, beta_structured_outputs: bool) -> Self {
        // In case we add more tool types in the future, the compiler will complain here.
        Self {
            name: tool.name(),
            description: Some(tool.description()),
            input_schema: tool.parameters(),
            strict: beta_structured_outputs.then_some(tool.strict()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub(super) enum AnthropicMessageContent<'a> {
    Text {
        text: &'a str,
    },
    Image {
        source: AnthropicDocumentSource,
    },
    Document {
        source: AnthropicDocumentSource,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: Vec<AnthropicMessageContent<'a>>,
    },
    Thinking {
        thinking: Option<&'a str>,
        signature: Option<&'a str>,
    },
    RedactedThinking {
        data: &'a str,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: Value,
    },
}

/// This is used by Anthropic for both images and documents -
/// the only different is the outer `AnthropicMessageContent`
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum AnthropicDocumentSource {
    Base64 { media_type: MediaType, data: String },
    Url { url: String },
}

impl<'a> AnthropicMessageContent<'a> {
    pub(super) async fn from_content_block(
        block: &'a ContentBlock,
        messages_config: AnthropicMessagesConfig,
        provider_type: &str,
    ) -> Result<Option<FlattenUnknown<'a, AnthropicMessageContent<'a>>>, Error> {
        match block {
            ContentBlock::Text(Text { text }) => Ok(Some(FlattenUnknown::Normal(
                AnthropicMessageContent::Text { text },
            ))),
            ContentBlock::ToolCall(tool_call) => {
                // Convert the tool call arguments from String to JSON Value (Anthropic expects an object)
                let input: Value = serde_json::from_str(&tool_call.arguments).map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: format!(
                            "Error parsing tool call arguments as JSON Value: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: provider_type.to_string(),
                        raw_request: None,
                        raw_response: Some(tool_call.arguments.clone()),
                    })
                })?;

                if !input.is_object() {
                    return Err(Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: "Tool call arguments must be a JSON object".to_string(),
                        provider_type: provider_type.to_string(),
                        raw_request: None,
                        raw_response: Some(tool_call.arguments.clone()),
                    }));
                }

                Ok(Some(FlattenUnknown::Normal(
                    AnthropicMessageContent::ToolUse {
                        id: &tool_call.id,
                        name: &tool_call.name,
                        input,
                    },
                )))
            }
            ContentBlock::ToolResult(tool_result) => Ok(Some(FlattenUnknown::Normal(
                AnthropicMessageContent::ToolResult {
                    tool_use_id: &tool_result.id,
                    content: vec![AnthropicMessageContent::Text {
                        text: &tool_result.result,
                    }],
                },
            ))),
            ContentBlock::File(file) => match &**file {
                LazyFile::Url {
                    file_url:
                        FileUrl {
                            mime_type: Some(mime_type),
                            url,
                            detail,
                        },
                    future: _,
                } if !messages_config.fetch_and_encode_input_files_before_inference => {
                    // If the user provided a url, and we're not configured to fetch the file beforehand,
                    // then forward the url directly to Anthropic.
                    if detail.is_some() {
                        tracing::warn!(
                            "The image detail parameter is not supported by Anthropic. The `detail` field will be ignored."
                        );
                    }
                    if mime_type.type_() == mime::IMAGE {
                        Ok(Some(FlattenUnknown::Normal(
                            AnthropicMessageContent::Image {
                                source: AnthropicDocumentSource::Url {
                                    url: url.to_string(),
                                },
                            },
                        )))
                    } else {
                        Ok(Some(FlattenUnknown::Normal(
                            AnthropicMessageContent::Document {
                                source: AnthropicDocumentSource::Url {
                                    url: url.to_string(),
                                },
                            },
                        )))
                    }
                }
                _ => {
                    warn_cannot_forward_url_if_missing_mime_type(
                        file,
                        messages_config.fetch_and_encode_input_files_before_inference,
                        provider_type,
                    );
                    // Otherwise, fetch the file, encode it as base64, and send it to Anthropic
                    let resolved_file = file.resolve().await?;
                    let ObjectStorageFile { file, data } = &*resolved_file;
                    if file.detail.is_some() {
                        tracing::warn!(
                            "The image detail parameter is not supported by Anthropic. The `detail` field will be ignored."
                        );
                    }
                    let document = AnthropicDocumentSource::Base64 {
                        media_type: file.mime_type.clone(),
                        data: data.clone(),
                    };
                    if file.mime_type.type_() == mime::IMAGE {
                        Ok(Some(FlattenUnknown::Normal(
                            AnthropicMessageContent::Image { source: document },
                        )))
                    } else {
                        Ok(Some(FlattenUnknown::Normal(
                            AnthropicMessageContent::Document { source: document },
                        )))
                    }
                }
            },
            ContentBlock::Thought(thought) => {
                if let Some(text) = thought.text.as_deref() {
                    Ok(Some(FlattenUnknown::Normal(
                        AnthropicMessageContent::Thinking {
                            thinking: Some(text),
                            signature: thought.signature.as_deref(),
                        },
                    )))
                } else if let Some(signature) = thought.signature.as_deref() {
                    Ok(Some(FlattenUnknown::Normal(
                        AnthropicMessageContent::RedactedThinking { data: signature },
                    )))
                } else {
                    Ok(None)
                }
            }
            ContentBlock::Unknown(Unknown { data, .. }) => {
                Ok(Some(FlattenUnknown::Unknown(Cow::Borrowed(data))))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct AnthropicMessage<'a> {
    pub(super) role: AnthropicRole,
    pub(super) content: Vec<FlattenUnknown<'a, AnthropicMessageContent<'a>>>,
}

impl<'a> AnthropicMessage<'a> {
    pub(super) async fn from_request_message(
        message: &'a RequestMessage,
        messages_config: AnthropicMessagesConfig,
        provider_type: &str,
    ) -> Result<Self, Error> {
        let content: Vec<FlattenUnknown<AnthropicMessageContent>> =
            try_join_all(message.content.iter().map(|c| {
                AnthropicMessageContent::from_content_block(c, messages_config, provider_type)
            }))
            .await?
            .into_iter()
            .flatten()
            .collect();

        Ok(AnthropicMessage {
            role: message.role.into(),
            content,
        })
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum AnthropicSystemBlock<'a> {
    Text {
        text: &'a str,
        // This also contains cache control and citations but we will ignore these for now.
    },
}

#[derive(Debug, PartialEq, Serialize)]
struct AnthropicThinkingConfig {
    r#type: &'static str,
    budget_tokens: i32,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum AnthropicOutputFormat {
    JsonSchema { schema: Value },
}

#[derive(Debug, Default, PartialEq, Serialize)]
struct AnthropicRequestBody<'a> {
    model: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<AnthropicOutputFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    // This is the system message
    system: Option<Vec<AnthropicSystemBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
}

#[derive(Copy, Clone, Debug)]
pub(super) struct AnthropicMessagesConfig {
    pub(super) fetch_and_encode_input_files_before_inference: bool,
}

fn needs_json_prefill(request: &ModelInferenceRequest<'_>, beta_structured_outputs: bool) -> bool {
    matches!(
        request.json_mode,
        ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict
    ) && matches!(request.function_type, FunctionType::Json)
        // Anthropic rejects prefill when 'output_format' is specified
        && !(beta_structured_outputs
            && matches!(request.json_mode, ModelInferenceRequestJsonMode::Strict))
}

impl<'a> AnthropicRequestBody<'a> {
    async fn new(
        model_name: &'a str,
        request: &'a ModelInferenceRequest<'_>,
        beta_structured_outputs: bool,
    ) -> Result<AnthropicRequestBody<'a>, Error> {
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
            .await?;
        let messages = prepare_messages(request_messages);
        let messages = if needs_json_prefill(request, beta_structured_outputs) {
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
                    .map(|tool| AnthropicTool::new(tool, beta_structured_outputs))
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
            None => get_default_max_tokens(model_name),
        }?;

        // NOTE: Anthropic does not support seed
        let mut anthropic_request = AnthropicRequestBody {
            model: model_name,
            messages,
            max_tokens,
            stream: Some(request.stream),
            system,
            temperature: request.temperature,
            thinking: None,
            top_p: request.top_p,
            service_tier: None, // handled below
            tool_choice,
            tools,
            output_format: if beta_structured_outputs {
                match request.json_mode {
                    ModelInferenceRequestJsonMode::Strict => {
                        request
                            .output_schema
                            .map(|schema| AnthropicOutputFormat::JsonSchema {
                                schema: schema.clone(),
                            })
                    }
                    ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Off => None,
                }
            } else {
                None
            },
            stop_sequences: request.borrow_stop_sequences(),
        };

        apply_inference_params(&mut anthropic_request, &request.inference_params_v2);

        Ok(anthropic_request)
    }
}

fn apply_inference_params(
    request: &mut AnthropicRequestBody,
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

    // Map service_tier values to Anthropic-compatible values
    if let Some(tier) = service_tier {
        match tier {
            ServiceTier::Auto | ServiceTier::Priority => {
                request.service_tier = Some("auto".to_string());
            }
            ServiceTier::Default => {
                request.service_tier = Some("standard_only".to_string());
            }
            ServiceTier::Flex => {
                warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier (flex)", None);
            }
        }
    }

    if let Some(budget_tokens) = thinking_budget_tokens {
        request.thinking = Some(AnthropicThinkingConfig {
            r#type: "enabled",
            budget_tokens: *budget_tokens,
        });
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
}

/// Returns the default max_tokens for a given Anthropic model name, or an error if unknown.
///
/// Anthropic requires that the user provides `max_tokens`, but the value depends on the model.
/// We maintain a library of known maximum values, and ask the user to hardcode it if it's unknown.
fn get_default_max_tokens(model_name: &str) -> Result<u32, Error> {
    if model_name.starts_with("claude-3-haiku") || model_name.starts_with("claude-3-opus") {
        Ok(4_096)
    } else if model_name.starts_with("claude-3-5-haiku")
        || model_name.starts_with("claude-3-5-sonnet")
    {
        Ok(8_192)
    } else if model_name.starts_with("claude-3-7-sonnet")
        || model_name.starts_with("claude-sonnet-4-202")
        || model_name == "claude-sonnet-4-0"
        || model_name.starts_with("claude-haiku-4-5")
        || model_name.starts_with("claude-sonnet-4-5")
        || model_name.starts_with("claude-opus-4-5")
    {
        Ok(64_000)
    } else if model_name.starts_with("claude-opus-4-202")
        || model_name == "claude-opus-4-0"
        || model_name.starts_with("claude-opus-4-1-202")
        || model_name == "claude-opus-4-1"
    {
        Ok(32_000)
    } else {
        Err(Error::new(ErrorDetails::InferenceClient {
            message: format!(
                "The TensorZero Gateway doesn't know the output token limit for `{model_name}` and Anthropic requires you to provide a `max_tokens` value. Please set `max_tokens` in your configuration or inference request."
            ),
            status_code: None,
            provider_type: PROVIDER_TYPE.into(),
            raw_request: None,
            raw_response: None,
        }))
    }
}

/// Modifies the message array to satisfy Anthropic API requirements by:
/// - Prepending a default User message with "[listening]" if the first message is not from a User
/// - Appending a default User message with "[listening]" if the last message is from an Assistant
fn prepare_messages(
    mut messages: Vec<AnthropicMessage<'_>>,
) -> std::vec::Vec<providers::anthropic::AnthropicMessage<'_>> {
    // Anthropic also requires that there is at least one message and it is a User message.
    // If it's not we will prepend a default User message.
    match messages.first() {
        Some(&AnthropicMessage {
            role: AnthropicRole::User,
            ..
        }) => {}
        _ => {
            messages.insert(
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
    if let Some(last_message) = messages.last() {
        if last_message.role == AnthropicRole::Assistant {
            messages.push(AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "[listening]",
                })],
            });
        }
    }
    messages
}

fn prefill_json_message(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    let mut messages = messages;
    // Add a JSON-prefill message for Anthropic's JSON mode
    messages.push(AnthropicMessage {
        role: AnthropicRole::Assistant,
        content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
            text: "Here is the JSON requested:\n{",
        })],
    });
    messages
}

pub(crate) fn prefill_json_response(
    content: Vec<ContentBlockOutput>,
) -> Result<Vec<ContentBlockOutput>, Error> {
    // Check if the content is a single text block
    if content.len() == 1 {
        if let ContentBlockOutput::Text(text) = &content[0] {
            // If it's a single text block, add a "{" to the beginning
            return Ok(vec![ContentBlockOutput::Text(Text {
                text: format!("{{{}", text.text.trim()),
            })]);
        }
    }
    // If it's not a single text block, return content as-is but log an error
    Error::new(ErrorDetails::OutputParsing {
        message: "Expected a single text block in the response from Anthropic".to_string(),
        raw_output: serde_json::to_string(&content).map_err(|e| Error::new(ErrorDetails::Inference {
            message: format!("Error serializing content as JSON: {}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new", DisplayOrDebugGateway::new(e)),
        }))?,
    });
    Ok(content)
}

pub(crate) fn prefill_json_chunk_response(chunk: &mut ProviderInferenceResponseChunk) {
    if chunk.content.is_empty() {
        chunk.content = vec![ContentBlockChunk::Text(TextChunk {
            text: "{".to_string(),
            id: "0".to_string(),
        })];
    } else if chunk.content.len() == 1 {
        if let ContentBlockChunk::Text(TextChunk { text, .. }) = &chunk.content[0] {
            // Add a "{" to the beginning of the text
            chunk.content = vec![ContentBlockChunk::Text(TextChunk {
                text: format!("{{{}", text.trim_start()),
                id: "0".to_string(),
            })];
        }
    } else {
        Error::new(ErrorDetails::OutputParsing {
            message: "Expected a single text block in the response from Anthropic".to_string(),
            raw_output: serde_json::to_string(&chunk.content).map_err(|e| Error::new(ErrorDetails::Inference {
                message: format!("Error serializing content as JSON: {}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new", DisplayOrDebugGateway::new(e)),
            })).unwrap_or_default()
        });
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

fn convert_to_output(
    model_name: &str,
    provider_name: &str,
    block: FlattenUnknown<'static, AnthropicContentBlock>,
) -> Result<ContentBlockOutput, Error> {
    match block {
        FlattenUnknown::Normal(AnthropicContentBlock::Text { text }) => Ok(text.into()),
        FlattenUnknown::Normal(AnthropicContentBlock::ToolUse { id, name, input }) => {
            Ok(ContentBlockOutput::ToolCall(ToolCall {
                id,
                name,
                arguments: serde_json::to_string(&input).map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing input for tool call: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: Some(serde_json::to_string(&input).unwrap_or_default()),
                    })
                })?,
            }))
        }
        FlattenUnknown::Normal(AnthropicContentBlock::Thinking {
            thinking,
            signature,
        }) => Ok(ContentBlockOutput::Thought(Thought {
            text: Some(thinking),
            signature: Some(signature),
            summary: None,
            provider_type: Some(PROVIDER_TYPE.to_string()),
        })),
        FlattenUnknown::Normal(AnthropicContentBlock::RedactedThinking { data }) => {
            Ok(ContentBlockOutput::Thought(Thought {
                text: None,
                signature: Some(data),
                summary: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
            }))
        }
        FlattenUnknown::Unknown(data) => Ok(ContentBlockOutput::Unknown(Unknown {
            data: data.into_owned(),
            model_name: Some(model_name.to_string()),
            provider_name: Some(provider_name.to_string()),
        })),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl From<AnthropicUsage> for Usage {
    fn from(value: AnthropicUsage) -> Self {
        Usage {
            input_tokens: Some(value.input_tokens),
            output_tokens: Some(value.output_tokens),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct AnthropicResponse {
    id: String,
    r#type: String, // this is always "message"
    role: String,   // this is always "assistant"
    content: Vec<FlattenUnknown<'static, AnthropicContentBlock>>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<AnthropicStopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    #[serde(other)]
    Unknown,
}

impl From<AnthropicStopReason> for FinishReason {
    fn from(value: AnthropicStopReason) -> Self {
        match value {
            AnthropicStopReason::EndTurn => FinishReason::Stop,
            AnthropicStopReason::MaxTokens => FinishReason::Length,
            AnthropicStopReason::StopSequence => FinishReason::StopSequence,
            AnthropicStopReason::ToolUse => FinishReason::ToolCall,
            AnthropicStopReason::Unknown => FinishReason::Unknown,
        }
    }
}

#[derive(Debug)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
struct AnthropicResponseWithMetadata<'a> {
    response: AnthropicResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    input_messages: Vec<RequestMessage>,
    model_name: &'a str,
    provider_name: &'a str,
    beta_structured_outputs: bool,
}

impl<'a> TryFrom<AnthropicResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: AnthropicResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let AnthropicResponseWithMetadata {
            response,
            raw_response,
            latency,
            raw_request,
            generic_request,
            input_messages,
            model_name,
            provider_name,
            beta_structured_outputs,
        } = value;
        let output: Vec<ContentBlockOutput> = response
            .content
            .into_iter()
            .map(|block| convert_to_output(model_name, provider_name, block))
            .collect::<Result<Vec<_>, _>>()?;
        let content = if needs_json_prefill(generic_request, beta_structured_outputs) {
            prefill_json_response(output)?
        } else {
            output
        };

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system: generic_request.system.clone(),
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

pub(super) fn handle_anthropic_error(
    response_code: StatusCode,
    raw_request: String,
    raw_response: String,
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED
        | StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::TOO_MANY_REQUESTS => Err(ErrorDetails::InferenceClient {
            status_code: Some(response_code),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(raw_request),
            raw_response: Some(raw_response.clone()),
            message: raw_response,
        }
        .into()),
        // StatusCode::NOT_FOUND | StatusCode::FORBIDDEN | StatusCode::INTERNAL_SERVER_ERROR | 529: Overloaded
        // These are all captured in _ since they have the same error behavior
        _ => Err(ErrorDetails::InferenceServer {
            raw_response: Some(raw_response.clone()),
            message: raw_response,
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(raw_request),
        }
        .into()),
    }
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    SignatureDelta { signature: String },
    ThinkingDelta { thinking: String },
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub struct AnthropicMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<AnthropicStopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamMessage {
    ContentBlockDelta {
        delta: FlattenUnknown<'static, AnthropicContentBlockDelta>,
        index: u32,
    },
    ContentBlockStart {
        content_block: FlattenUnknown<'static, AnthropicContentBlock>,
        index: u32,
    },
    ContentBlockStop {
        index: u32,
    },
    Error {
        error: Value,
    },
    MessageDelta {
        delta: FlattenUnknown<'static, AnthropicMessageDelta>,
        usage: Value,
    },
    MessageStart {
        message: Value,
    },
    MessageStop,
    Ping,
}

/// This function converts an Anthropic stream message to a TensorZero stream message.
/// It must keep track of the current tool ID and name in order to correctly handle ToolCallChunks (which we force to always contain the tool name and ID)
/// Anthropic only sends the tool ID and name in the ToolUse chunk so we need to keep the most recent ones as mutable references so
/// subsequent InputJSONDelta chunks can be initialized with this information as well.
/// There is no need to do the same bookkeeping for TextDelta chunks since they come with an index (which we use as an ID for a text chunk).
/// See the Anthropic [docs](https://docs.anthropic.com/en/api/messages-streaming) on streaming messages for details on the types of events and their semantics.
#[expect(clippy::too_many_arguments)]
pub(super) fn anthropic_to_tensorzero_stream_message(
    raw_message: String,
    message: AnthropicStreamMessage,
    message_latency: Duration,
    current_tool_id: &mut Option<String>,
    current_tool_name: &mut Option<String>,
    discard_unknown_chunks: bool,
    model_name: &str,
    provider_name: &str,
    provider_type: &str,
) -> Result<Option<ProviderInferenceResponseChunk>, Error> {
    match message {
        AnthropicStreamMessage::ContentBlockDelta {
            delta: FlattenUnknown::Normal(delta),
            index,
        } => match delta {
            AnthropicContentBlockDelta::TextDelta { text } => {
                Ok(Some(ProviderInferenceResponseChunk::new(
                    vec![ContentBlockChunk::Text(TextChunk {
                        text,
                        id: index.to_string(),
                    })],
                    None,
                    raw_message,
                    message_latency,
                    None,
                )))
            }
            AnthropicContentBlockDelta::InputJsonDelta { partial_json } => {
                Ok(Some(ProviderInferenceResponseChunk::new(
                    // Take the current tool name and ID and use them to create a ToolCallChunk
                    // This is necessary because the ToolCallChunk must always contain the tool name and ID
                    // even though Anthropic only sends the tool ID and name in the ToolUse chunk and not InputJSONDelta
                    vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                        raw_name: None,
                        id: current_tool_id.clone().ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                            message: "Got InputJsonDelta chunk from Anthropic without current tool id being set by a ToolUse".to_string(),
                            provider_type: provider_type.to_string(),
                            raw_request: None,
                            raw_response: None,
                        }))?,
                        raw_arguments: partial_json,
                    })],
                    None,
                    raw_message,
                    message_latency,
                    None,
                )))
            }
            AnthropicContentBlockDelta::ThinkingDelta { thinking } => {
                Ok(Some(ProviderInferenceResponseChunk::new(
                    vec![ContentBlockChunk::Thought(ThoughtChunk {
                        text: Some(thinking),
                        signature: None,
                        id: index.to_string(),
                        summary_id: None,
                        summary_text: None,
                        provider_type: Some(provider_type.to_string()),
                    })],
                    None,
                    raw_message,
                    message_latency,
                    None,
                )))
            }
            AnthropicContentBlockDelta::SignatureDelta { signature } => {
                Ok(Some(ProviderInferenceResponseChunk::new(
                    vec![ContentBlockChunk::Thought(ThoughtChunk {
                        text: None,
                        signature: Some(signature),
                        id: index.to_string(),
                        summary_id: None,
                        summary_text: None,
                        provider_type: Some(provider_type.to_string()),
                    })],
                    None,
                    raw_message,
                    message_latency,
                    None,
                )))
            }
        },
        AnthropicStreamMessage::ContentBlockStart {
            content_block: FlattenUnknown::Normal(content_block),
            index,
        } => match content_block {
            AnthropicContentBlock::Text { text } => {
                let text_chunk = ContentBlockChunk::Text(TextChunk {
                    text,
                    id: index.to_string(),
                });
                Ok(Some(ProviderInferenceResponseChunk::new(
                    vec![text_chunk],
                    None,
                    raw_message,
                    message_latency,
                    None,
                )))
            }
            AnthropicContentBlock::ToolUse { id, name, .. } => {
                // This is a new tool call, update the ID for future chunks
                *current_tool_id = Some(id.clone());
                *current_tool_name = Some(name.clone());
                Ok(Some(ProviderInferenceResponseChunk::new(
                    vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                        id,
                        raw_name: Some(name),
                        // As far as I can tell this is always {} so we ignore
                        raw_arguments: String::new(),
                    })],
                    None,
                    raw_message,
                    message_latency,
                    None,
                )))
            }
            AnthropicContentBlock::Thinking {
                thinking,
                signature,
            } => Ok(Some(ProviderInferenceResponseChunk::new(
                vec![ContentBlockChunk::Thought(ThoughtChunk {
                    text: Some(thinking),
                    signature: Some(signature),
                    id: index.to_string(),
                    summary_id: None,
                    summary_text: None,
                    provider_type: Some(provider_type.to_string()),
                })],
                None,
                raw_message,
                message_latency,
                None,
            ))),
            AnthropicContentBlock::RedactedThinking { data } => {
                Ok(Some(ProviderInferenceResponseChunk::new(
                    vec![ContentBlockChunk::Thought(ThoughtChunk {
                        text: None,
                        signature: Some(data),
                        id: index.to_string(),
                        summary_id: None,
                        summary_text: None,
                        provider_type: Some(provider_type.to_string()),
                    })],
                    None,
                    raw_message,
                    message_latency,
                    None,
                )))
            }
        },
        AnthropicStreamMessage::ContentBlockStop { .. } => Ok(None),
        AnthropicStreamMessage::Error { error } => Err(ErrorDetails::InferenceServer {
            message: error.to_string(),
            provider_type: provider_type.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into()),
        AnthropicStreamMessage::MessageDelta {
            usage,
            delta: FlattenUnknown::Normal(delta),
        } => {
            let usage = parse_usage_info(&usage);
            Ok(Some(ProviderInferenceResponseChunk::new(
                vec![],
                Some(usage.into()),
                raw_message,
                message_latency,
                delta.stop_reason.map(AnthropicStopReason::into),
            )))
        }
        AnthropicStreamMessage::MessageStart { message } => {
            if let Some(usage_info) = message.get("usage") {
                let usage = parse_usage_info(usage_info);
                Ok(Some(ProviderInferenceResponseChunk::new(
                    vec![],
                    Some(usage.into()),
                    raw_message,
                    message_latency,
                    None,
                )))
            } else {
                Ok(None)
            }
        }
        AnthropicStreamMessage::MessageStop | AnthropicStreamMessage::Ping => Ok(None),
        AnthropicStreamMessage::ContentBlockDelta {
            delta: FlattenUnknown::Unknown(delta),
            index,
        } => {
            if discard_unknown_chunks {
                warn_discarded_unknown_chunk(provider_type, &delta.to_string());
                return Ok(None);
            }
            Ok(Some(ProviderInferenceResponseChunk::new(
                vec![ContentBlockChunk::Unknown(UnknownChunk {
                    id: index.to_string(),
                    data: delta.into_owned(),
                    model_name: Some(model_name.to_string()),
                    provider_name: Some(provider_name.to_string()),
                })],
                None,
                raw_message,
                message_latency,
                None,
            )))
        }
        AnthropicStreamMessage::ContentBlockStart {
            content_block: FlattenUnknown::Unknown(content_block),
            index,
        } => {
            if discard_unknown_chunks {
                warn_discarded_unknown_chunk(provider_type, &content_block.to_string());
                return Ok(None);
            }
            Ok(Some(ProviderInferenceResponseChunk::new(
                vec![ContentBlockChunk::Unknown(UnknownChunk {
                    id: index.to_string(),
                    data: content_block.into_owned(),
                    model_name: Some(model_name.to_string()),
                    provider_name: Some(provider_name.to_string()),
                })],
                None,
                raw_message,
                message_latency,
                None,
            )))
        }
        AnthropicStreamMessage::MessageDelta {
            usage: _,
            delta: FlattenUnknown::Unknown(delta),
        } => {
            if discard_unknown_chunks {
                warn_discarded_unknown_chunk(provider_type, &delta.to_string());
                return Ok(None);
            }
            Ok(Some(ProviderInferenceResponseChunk::new(
                vec![ContentBlockChunk::Unknown(UnknownChunk {
                    id: "message_delta".to_string(),
                    data: delta.into_owned(),
                    model_name: Some(model_name.to_string()),
                    provider_name: Some(provider_name.to_string()),
                })],
                None,
                raw_message,
                message_latency,
                None,
            )))
        }
    }
}

fn parse_usage_info(usage_info: &Value) -> AnthropicUsage {
    let input_tokens = usage_info
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let output_tokens = usage_info
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    AnthropicUsage {
        input_tokens,
        output_tokens,
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use futures::FutureExt;
    use serde_json::json;
    use url::Url;
    use uuid::Uuid;

    use super::*;
    use crate::inference::types::file::Detail;
    use crate::inference::types::resolved_input::{FileUrl, LazyFile};
    use crate::inference::types::{ContentBlock, FunctionType, ModelInferenceRequestJsonMode};
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::providers::test_helpers::WEATHER_TOOL_CONFIG;
    use crate::tool::{DynamicToolConfig, ToolResult};
    use crate::utils::testing::capture_logs;

    #[test]
    fn test_try_from_tool_call_config() {
        // Need to cover all 4 cases
        let tool_call_config = ToolCallConfig {
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_call_config);
        assert!(matches!(
            anthropic_tool_choice.unwrap(),
            AnthropicToolChoice::Auto {
                disable_parallel_tool_use: Some(true)
            }
        ));

        let tool_call_config = ToolCallConfig {
            parallel_tool_calls: Some(true),
            ..Default::default()
        };
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_call_config);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            AnthropicToolChoice::Auto {
                disable_parallel_tool_use: Some(false)
            }
        );

        let tool_call_config = ToolCallConfig {
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: Some(true),
            ..Default::default()
        };
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_call_config);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            AnthropicToolChoice::Any {
                disable_parallel_tool_use: Some(false)
            }
        );

        let tool_call_config = ToolCallConfig {
            tool_choice: ToolChoice::Specific("test".to_string()),
            parallel_tool_calls: Some(false),
            ..Default::default()
        };
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_call_config);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            AnthropicToolChoice::Tool {
                name: "test",
                disable_parallel_tool_use: Some(true)
            }
        );
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
        let text_content_block: ContentBlock = "test".to_string().into();
        let anthropic_content_block = AnthropicMessageContent::from_content_block(
            &text_content_block,
            AnthropicMessagesConfig {
                fetch_and_encode_input_files_before_inference: false,
            },
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
            AnthropicMessagesConfig {
                fetch_and_encode_input_files_before_inference: false,
            },
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
    async fn test_try_from_request_message() {
        // Test a User message
        let inference_request_message = RequestMessage {
            role: Role::User,
            content: vec!["test".to_string().into()],
        };
        let anthropic_message = AnthropicMessage::from_request_message(
            &inference_request_message,
            AnthropicMessagesConfig {
                fetch_and_encode_input_files_before_inference: false,
            },
            PROVIDER_TYPE,
        )
        .await
        .unwrap();
        assert_eq!(
            anthropic_message,
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "test"
                })],
            }
        );

        // Test an Assistant message
        let inference_request_message = RequestMessage {
            role: Role::Assistant,
            content: vec!["test_assistant".to_string().into()],
        };
        let anthropic_message = AnthropicMessage::from_request_message(
            &inference_request_message,
            AnthropicMessagesConfig {
                fetch_and_encode_input_files_before_inference: false,
            },
            PROVIDER_TYPE,
        )
        .await
        .unwrap();
        assert_eq!(
            anthropic_message,
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "test_assistant",
                })],
            }
        );

        // Test a Tool message
        let inference_request_message = RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "test_tool_call_id".to_string(),
                name: "test_tool_name".to_string(),
                result: "test_tool_response".to_string(),
            })],
        };
        let anthropic_message = AnthropicMessage::from_request_message(
            &inference_request_message,
            AnthropicMessagesConfig {
                fetch_and_encode_input_files_before_inference: false,
            },
            PROVIDER_TYPE,
        )
        .await
        .unwrap();
        assert_eq!(
            anthropic_message,
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(
                    AnthropicMessageContent::ToolResult {
                        tool_use_id: "test_tool_call_id",
                        content: vec![AnthropicMessageContent::Text {
                            text: "test_tool_response"
                        }],
                    }
                )],
            }
        );
    }

    #[tokio::test]
    async fn test_initialize_anthropic_request_body() {
        let model = "claude-3-7-sonnet-latest".to_string();
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
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let anthropic_request_body =
            AnthropicRequestBody::new(&model, &inference_request, false).await;
        let error = anthropic_request_body.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            }
        );

        // Test Case 2: Messages starting with Assistant - should prepend and append listening message
        let messages = vec![RequestMessage {
            role: Role::Assistant,
            content: vec!["test_assistant".to_string().into()],
        }];
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages,
            system: Some("test_system".to_string()),
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let anthropic_request_body =
            AnthropicRequestBody::new(&model, &inference_request, false).await;
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: &model,
                messages: vec![
                    listening_message.clone(),
                    AnthropicMessage::from_request_message(
                        &inference_request.messages[0],
                        AnthropicMessagesConfig {
                            fetch_and_encode_input_files_before_inference: false,
                        },
                        PROVIDER_TYPE,
                    )
                    .await
                    .unwrap(),
                    listening_message.clone(),
                ],
                max_tokens: 64_000,
                stream: Some(false),
                system: Some(vec![AnthropicSystemBlock::Text {
                    text: "test_system"
                }]),
                ..Default::default()
            }
        );

        // Test Case 3: Messages ending with Assistant - should append listening message
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
            messages,
            system: Some("test_system".to_string()),
            tool_config: None,
            temperature: Some(0.5),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            seed: None,
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let anthropic_request_body =
            AnthropicRequestBody::new(&model, &inference_request, false).await;
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: &model,
                messages: vec![
                    AnthropicMessage::from_request_message(
                        &inference_request.messages[0],
                        AnthropicMessagesConfig {
                            fetch_and_encode_input_files_before_inference: false,
                        },
                        PROVIDER_TYPE,
                    )
                    .await
                    .unwrap(),
                    AnthropicMessage::from_request_message(
                        &inference_request.messages[1],
                        AnthropicMessagesConfig {
                            fetch_and_encode_input_files_before_inference: false,
                        },
                        PROVIDER_TYPE,
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
                ..Default::default()
            }
        );

        // Test Case 4: Valid message sequence - no changes needed
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
                content: vec!["test_user2".to_string().into()],
            },
        ];
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages,
            system: None,
            tool_config: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let anthropic_request_body =
            AnthropicRequestBody::new(&model, &inference_request, false).await;
        assert!(anthropic_request_body.is_ok());
        // Convert messages asynchronously
        let expected_messages = try_join_all(inference_request.messages.iter().map(|m| {
            AnthropicMessage::from_request_message(
                m,
                AnthropicMessagesConfig {
                    fetch_and_encode_input_files_before_inference: false,
                },
                PROVIDER_TYPE,
            )
        }))
        .await
        .unwrap();

        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: &model,
                messages: expected_messages,
                max_tokens: 64_000,
                stream: Some(false),
                ..Default::default()
            }
        );

        // Test Case 5: Tool use with JSON mode
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "test_id".to_string(),
                    name: "get_temperature".to_string(),
                    arguments: r#"{"location":"London"}"#.to_string(),
                })],
            },
        ];
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages,
            system: None,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            function_type: FunctionType::Json,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let anthropic_request_body =
            AnthropicRequestBody::new(&model, &inference_request, false).await;
        assert!(anthropic_request_body.is_ok());
        let result = anthropic_request_body.unwrap();
        assert_eq!(result.messages.len(), 4); // Original 2 messages + listening message + JSON prefill
        assert_eq!(
            result.messages[0],
            AnthropicMessage::from_request_message(
                &inference_request.messages[0],
                AnthropicMessagesConfig {
                    fetch_and_encode_input_files_before_inference: false,
                },
                PROVIDER_TYPE,
            )
            .await
            .unwrap()
        );
        assert_eq!(
            result.messages[1],
            AnthropicMessage::from_request_message(
                &inference_request.messages[1],
                AnthropicMessagesConfig {
                    fetch_and_encode_input_files_before_inference: false,
                },
                PROVIDER_TYPE,
            )
            .await
            .unwrap()
        );
        assert_eq!(result.messages[2], listening_message);
        assert_eq!(
            result.messages[3],
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Here is the JSON requested:\n{",
                })],
            }
        );
    }

    #[tokio::test]
    async fn test_get_default_max_tokens_in_new_anthropic_request_body() {
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

        let model = "claude-opus-4-1-20250805".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 32_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-opus-4-20250514".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 32_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-sonnet-4-20250514".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-7-sonnet-20250219".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-5-sonnet-20241022".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 8_192);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-5-haiku-20241022".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 8_192);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-opus-4-1".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 32_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-opus-4-0".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 32_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-sonnet-4-0".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-7-sonnet-latest".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-5-sonnet-latest".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 8_192);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-5-haiku-latest".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 8_192);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-haiku-20240307".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 4_096);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-haiku-4-5-20251001".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-sonnet-4-5-20250929".to_string();
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert_eq!(body.unwrap().max_tokens, 64_000);
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-3-5-ballad-latest".to_string(); // fake model
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert!(body.is_err());
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);

        let model = "claude-4-5-haiku-20260101".to_string(); // fake model
        let body = AnthropicRequestBody::new(&model, &request, false).await;
        assert!(body.is_err());
        let body = AnthropicRequestBody::new(&model, &request_with_max_tokens, false).await;
        assert_eq!(body.unwrap().max_tokens, 100);
    }

    #[tokio::test]
    async fn test_prepare_messages() {
        let listening_message = AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "[listening]",
            })],
        };

        // Test case 1: Empty messages - should add listening message
        let messages = vec![];
        let result = prepare_messages(messages);
        assert_eq!(result, vec![listening_message.clone()]);

        // Test case 2: First message is Assistant - should prepend listening message
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hi",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hello",
                })],
            },
        ];
        let result = prepare_messages(messages);
        assert_eq!(
            result,
            vec![
                listening_message.clone(),
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "Hi"
                    })],
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "Hello"
                    })],
                },
            ]
        );

        // Test case 3: Last message is Assistant - should append listening message
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
        let result = prepare_messages(messages);
        assert_eq!(
            result,
            vec![
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "Hello"
                    })],
                },
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "Hi"
                    })],
                },
                listening_message.clone(),
            ]
        );

        // Test case 4: Valid message sequence - no changes needed
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
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "How are you?",
                })],
            },
        ];
        let result = prepare_messages(messages.clone());
        assert_eq!(result, messages);

        // Test case 5: Both first Assistant and last Assistant - should add listening messages at both ends
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hi",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "Hello",
                })],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                    text: "How can I help?",
                })],
            },
        ];
        let result = prepare_messages(messages);
        assert_eq!(
            result,
            vec![
                listening_message.clone(),
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "Hi"
                    })],
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "Hello"
                    })],
                },
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "How can I help?"
                    })],
                },
                listening_message.clone(),
            ]
        );

        // Test case 6: Single Assistant message - should add listening messages at both ends
        let messages = vec![AnthropicMessage {
            role: AnthropicRole::Assistant,
            content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Hi",
            })],
        }];
        let result = prepare_messages(messages);
        assert_eq!(
            result,
            vec![
                listening_message.clone(),
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                        text: "Hi"
                    })],
                },
                listening_message.clone(),
            ]
        );

        // Test case 7: Single User message - no changes needed
        let messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Hello",
            })],
        }];
        let result = prepare_messages(messages.clone());
        assert_eq!(result, messages);
    }

    #[test]
    fn test_handle_anthropic_error() {
        let response_code = StatusCode::BAD_REQUEST;
        let result = handle_anthropic_error(
            response_code,
            "raw request".to_string(),
            "raw response".to_string(),
        );
        let error = result.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InferenceClient {
                message: "raw response".to_string(),
                status_code: Some(response_code),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("raw response".to_string()),
            }
        );
        let response_code = StatusCode::UNAUTHORIZED;
        let result = handle_anthropic_error(
            response_code,
            "raw request".to_string(),
            "raw response".to_string(),
        );
        let error = result.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InferenceClient {
                message: "raw response".to_string(),
                status_code: Some(response_code),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("raw response".to_string()),
            }
        );
        let response_code = StatusCode::TOO_MANY_REQUESTS;
        let result = handle_anthropic_error(
            response_code,
            "raw request".to_string(),
            "raw response".to_string(),
        );
        let error = result.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InferenceClient {
                message: "raw response".to_string(),
                status_code: Some(response_code),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("raw response".to_string()),
            }
        );
        let response_code = StatusCode::NOT_FOUND;
        let result = handle_anthropic_error(
            response_code,
            "raw request".to_string(),
            "raw response".to_string(),
        );
        let error = result.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InferenceServer {
                message: "raw response".to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("raw response".to_string()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
        );
        let response_code = StatusCode::INTERNAL_SERVER_ERROR;
        let result = handle_anthropic_error(
            response_code,
            "raw request".to_string(),
            "raw response".to_string(),
        );
        let error = result.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InferenceServer {
                message: "raw response".to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("raw response".to_string()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
        );
    }

    #[test]
    fn test_anthropic_usage_to_usage() {
        let anthropic_usage = AnthropicUsage {
            input_tokens: 100,
            output_tokens: 50,
        };

        let usage: Usage = anthropic_usage.into();

        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(50));
    }

    #[test]
    fn test_anthropic_response_conversion() {
        // Test case 1: Text response
        let anthropic_response_body = AnthropicResponse {
            id: "1".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![FlattenUnknown::Normal(AnthropicContentBlock::Text {
                text: "Response text".to_string(),
            })],
            model: "model-name".into(),
            stop_reason: Some(AnthropicStopReason::EndTurn),
            stop_sequence: Some("stop sequence".to_string()),
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
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
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let request_body = AnthropicRequestBody {
            model: "model-name",
            messages: vec![],
            max_tokens: 100,
            stream: Some(false),
            system: None,
            top_p: Some(0.5),
            ..Default::default()
        };
        let raw_response = "{\"foo\": \"bar\"}".to_string();
        let input_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into()],
        }];
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = AnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            raw_response: raw_response.clone(),
            latency: latency.clone(),
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            input_messages: input_messages.clone(),
            model_name: "model-name",
            provider_name: "dummy",
            beta_structured_outputs: false,
        };

        let inference_response = ProviderInferenceResponse::try_from(body_with_latency).unwrap();
        assert_eq!(
            inference_response.output,
            vec!["Response text".to_string().into()]
        );

        assert_eq!(raw_response, inference_response.raw_response);
        assert_eq!(inference_response.usage.input_tokens, Some(100));
        assert_eq!(inference_response.usage.output_tokens, Some(50));
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(inference_response.latency, latency);
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.input_messages, input_messages);

        // Test case 2: Tool call response
        let anthropic_response_body = AnthropicResponse {
            id: "2".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![FlattenUnknown::Normal(AnthropicContentBlock::ToolUse {
                id: "tool_call_1".to_string(),
                name: "get_temperature".to_string(),
                input: json!({"location": "New York"}),
            })],
            model: "model-name".into(),
            stop_reason: Some(AnthropicStopReason::ToolUse),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let request_body = AnthropicRequestBody {
            model: "model-name",
            messages: vec![],
            max_tokens: 100,
            stream: Some(false),
            top_p: Some(0.5),
            ..Default::default()
        };
        let input_messages = vec![RequestMessage {
            role: Role::Assistant,
            content: vec!["Hello".to_string().into()],
        }];
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = AnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            raw_response: raw_response.clone(),
            latency: latency.clone(),
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            input_messages: input_messages.clone(),
            model_name: "model-name",
            provider_name: "dummy",
            beta_structured_outputs: false,
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
        assert_eq!(
            inference_response.finish_reason,
            Some(FinishReason::ToolCall)
        );
        assert_eq!(inference_response.input_messages, input_messages);

        // Test case 3: Mixed response (text and tool call)
        let anthropic_response_body = AnthropicResponse {
            id: "3".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                FlattenUnknown::Normal(AnthropicContentBlock::Text {
                    text: "Here's the weather:".to_string(),
                }),
                FlattenUnknown::Normal(AnthropicContentBlock::ToolUse {
                    id: "tool_call_2".to_string(),
                    name: "get_temperature".to_string(),
                    input: json!({"location": "London"}),
                }),
            ],
            model: "model-name".into(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let request_body = AnthropicRequestBody {
            model: "model-name",
            messages: vec![],
            max_tokens: 100,
            stream: Some(false),
            top_p: Some(0.5),
            ..Default::default()
        };
        let input_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Helloooo".to_string().into()],
        }];
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = AnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            raw_response: raw_response.clone(),
            latency: latency.clone(),
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            input_messages: input_messages.clone(),
            model_name: "model-name",
            provider_name: "dummy",
            beta_structured_outputs: false,
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
        assert_eq!(inference_response.finish_reason, None);
        assert_eq!(inference_response.latency, latency);
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.input_messages, input_messages);
    }

    #[test]
    fn test_anthropic_to_tensorzero_stream_message() {
        use serde_json::json;

        // Test ContentBlockDelta with TextDelta
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_delta = AnthropicStreamMessage::ContentBlockDelta {
            delta: FlattenUnknown::Normal(AnthropicContentBlockDelta::TextDelta {
                text: "Hello".to_string(),
            }),
            index: 0,
        };
        let latency = Duration::from_millis(100);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            content_block_delta,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::Text(text) => {
                assert_eq!(text.text, "Hello".to_string());
                assert_eq!(text.id, "0".to_string());
            }
            _ => panic!("Expected a text content block"),
        }
        assert_eq!(chunk.latency, latency);

        // Test ContentBlockDelta with InputJsonDelta but no previous tool info
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_delta = AnthropicStreamMessage::ContentBlockDelta {
            delta: FlattenUnknown::Normal(AnthropicContentBlockDelta::InputJsonDelta {
                partial_json: "aaaa: bbbbb".to_string(),
            }),
            index: 0,
        };
        let latency = Duration::from_millis(100);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            content_block_delta,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        let error = result.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InferenceServer {
                message: "Got InputJsonDelta chunk from Anthropic without current tool id being set by a ToolUse".to_string(),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            }
        );

        // Test ContentBlockDelta with InputJsonDelta and previous tool info
        let mut current_tool_id = Some("tool_id".to_string());
        let mut current_tool_name = Some("tool_name".to_string());
        let content_block_delta = AnthropicStreamMessage::ContentBlockDelta {
            delta: FlattenUnknown::Normal(AnthropicContentBlockDelta::InputJsonDelta {
                partial_json: "aaaa: bbbbb".to_string(),
            }),
            index: 0,
        };
        let latency = Duration::from_millis(100);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            content_block_delta,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "tool_id".to_string());
                assert_eq!(tool_call.raw_name, None); // We don't add the tool name if it isn't in the contentBlockDelta
                assert_eq!(tool_call.raw_arguments, "aaaa: bbbbb".to_string());
            }
            _ => panic!("Expected a tool call content block"),
        }
        assert_eq!(chunk.latency, latency);

        // Test ContentBlockStart with ToolUse
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_start = AnthropicStreamMessage::ContentBlockStart {
            content_block: FlattenUnknown::Normal(AnthropicContentBlock::ToolUse {
                id: "tool1".to_string(),
                name: "calculator".to_string(),
                input: json!({}),
            }),
            index: 1,
        };
        let latency = Duration::from_millis(110);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            content_block_start,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        )
        .unwrap();
        let chunk = result.unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "tool1".to_string());
                assert_eq!(tool_call.raw_name, Some("calculator".to_string()));
                assert_eq!(tool_call.raw_arguments, String::new());
            }
            _ => panic!("Expected a tool call content block"),
        }
        assert_eq!(chunk.latency, latency);
        assert_eq!(current_tool_id, Some("tool1".to_string()));
        assert_eq!(current_tool_name, Some("calculator".to_string()));

        // Test ContentBlockStart with Text
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_start = AnthropicStreamMessage::ContentBlockStart {
            content_block: FlattenUnknown::Normal(AnthropicContentBlock::Text {
                text: "Hello".to_string(),
            }),
            index: 2,
        };
        let latency = Duration::from_millis(120);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            content_block_start,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::Text(text) => {
                assert_eq!(text.text, "Hello".to_string());
                assert_eq!(text.id, "2".to_string());
            }
            _ => panic!("Expected a text content block"),
        }
        assert_eq!(chunk.latency, latency);

        // Test ContentBlockStop
        let content_block_stop = AnthropicStreamMessage::ContentBlockStop { index: 2 };
        let latency = Duration::from_millis(120);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            content_block_stop,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test Error
        let error_message = AnthropicStreamMessage::Error {
            error: json!({"message": "Test error"}),
        };
        let latency = Duration::from_millis(130);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            error_message,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        let error = result.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InferenceServer {
                message: r#"{"message":"Test error"}"#.to_string(),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            }
        );

        // Test MessageDelta with usage
        let message_delta = AnthropicStreamMessage::MessageDelta {
            delta: FlattenUnknown::Normal(AnthropicMessageDelta {
                stop_reason: Some(AnthropicStopReason::EndTurn),
                stop_sequence: None,
            }),
            usage: json!({"input_tokens": 10, "output_tokens": 20}),
        };
        let latency = Duration::from_millis(140);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            message_delta,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 0);
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(chunk.latency, latency);
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));

        // Test MessageStart with usage
        let message_start = AnthropicStreamMessage::MessageStart {
            message: json!({"usage": {"input_tokens": 5, "output_tokens": 15}}),
        };
        let latency = Duration::from_millis(150);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            message_start,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 0);
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(5));
        assert_eq!(usage.output_tokens, Some(15));
        assert_eq!(chunk.latency, latency);

        // Test MessageStop
        let message_stop = AnthropicStreamMessage::MessageStop;
        let latency = Duration::from_millis(160);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            message_stop,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test Ping
        let ping = AnthropicStreamMessage::Ping {};
        let latency = Duration::from_millis(170);
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            ping,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
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
    fn test_anthropic_base_url() {
        assert_eq!(
            ANTHROPIC_DEFAULT_BASE_URL.as_str(),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn test_anthropic_provider_custom_api_base() {
        let custom_url = Url::parse("https://example.com/custom").unwrap();
        let provider = AnthropicProvider::new(
            "claude".to_string(),
            Some(custom_url.clone()),
            AnthropicCredentials::None,
            false,
        );

        assert_eq!(provider.base_url(), &custom_url);
    }

    #[test]
    fn test_anthropic_provider_default_api_base() {
        let provider = AnthropicProvider::new(
            "claude".to_string(),
            None,
            AnthropicCredentials::None,
            false,
        );

        assert_eq!(
            provider.base_url().as_str(),
            ANTHROPIC_DEFAULT_BASE_URL.as_str()
        );
    }

    #[test]
    fn test_prefill_json_message() {
        // Create a sample input message
        let input_messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Generate some JSON",
            })],
        }];

        // Call the function
        let result = prefill_json_message(input_messages);

        // Assert that the result has one more message than the input
        assert_eq!(result.len(), 2);

        // Check the original message is unchanged
        assert_eq!(result[0].role, AnthropicRole::User);
        assert_eq!(
            result[0].content,
            vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Generate some JSON",
            })]
        );

        // Check the new message is correct
        assert_eq!(result[1].role, AnthropicRole::Assistant);
        assert_eq!(
            result[1].content,
            vec![FlattenUnknown::Normal(AnthropicMessageContent::Text {
                text: "Here is the JSON requested:\n{",
            })]
        );
    }

    #[test]
    fn test_prefill_json_response() {
        // Test case 1: Single text block
        let input = vec![ContentBlockOutput::Text(Text {
            text: "  \"key\": \"value\"}".to_string(),
        })];
        let result = prefill_json_response(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0],
            ContentBlockOutput::Text(Text {
                text: "{\"key\": \"value\"}".to_string(),
            })
        );

        // Test case 2: Multiple blocks
        let input = vec![
            ContentBlockOutput::Text(Text {
                text: "Block 1".to_string(),
            }),
            ContentBlockOutput::Text(Text {
                text: "Block 2".to_string(),
            }),
        ];
        let result = prefill_json_response(input.clone()).unwrap();
        assert_eq!(result, input);

        // Test case 3: Empty input
        let input = vec![];
        let result = prefill_json_response(input.clone()).unwrap();
        assert_eq!(result, input);

        // Test case 4: Non-text block
        let input = vec![ContentBlockOutput::ToolCall(ToolCall {
            id: "1".to_string(),
            name: "test_tool".to_string(),
            arguments: "{}".to_string(),
        })];
        let result = prefill_json_response(input.clone()).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn test_prefill_json_chunk_response() {
        // Test case 1: Empty content
        let chunk = ProviderInferenceResponseChunk {
            content: vec![],
            created: 0,
            usage: None,
            raw_response: String::new(),
            latency: Duration::from_millis(0),
            finish_reason: None,
        };
        let mut result = chunk.clone();
        prefill_json_chunk_response(&mut result);
        assert_eq!(
            result.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "{".to_string(),
                id: "0".to_string()
            })]
        );
        // Test case 2: Single text block
        let chunk = ProviderInferenceResponseChunk {
            created: 0,
            usage: None,
            raw_response: String::new(),
            latency: Duration::from_millis(0),
            finish_reason: None,
            content: vec![ContentBlockChunk::Text(TextChunk {
                text: "\"key\": \"value ".to_string(),
                id: "0".to_string(),
            })],
        };
        let mut result = chunk.clone();
        prefill_json_chunk_response(&mut result);
        assert_eq!(
            result.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "{\"key\": \"value ".to_string(),
                id: "0".to_string()
            })]
        );

        // Test case 3: Multiple blocks (should remain unchanged)
        let chunk = ProviderInferenceResponseChunk {
            created: 0,
            usage: None,
            raw_response: String::new(),
            latency: Duration::from_millis(0),
            finish_reason: None,
            content: vec![
                ContentBlockChunk::Text(TextChunk {
                    text: "Block 1".to_string(),
                    id: "test_id".to_string(),
                }),
                ContentBlockChunk::Text(TextChunk {
                    text: "Block 2".to_string(),
                    id: "test_id".to_string(),
                }),
            ],
        };
        let mut result = chunk.clone();
        prefill_json_chunk_response(&mut result);
        assert_eq!(result, chunk);

        // Test case 4: Non-text block (should remain unchanged)
        let chunk = ProviderInferenceResponseChunk {
            created: 0,
            usage: None,
            raw_response: String::new(),
            latency: Duration::from_millis(0),
            finish_reason: None,
            content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "1".to_string(),
                raw_name: Some("test_tool".to_string()),
                raw_arguments: "{}".to_string(),
            })],
        };
        let mut result = chunk.clone();
        prefill_json_chunk_response(&mut result);
        assert_eq!(result, chunk);
    }

    #[test]
    fn test_credential_to_anthropic_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = AnthropicCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, AnthropicCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = AnthropicCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, AnthropicCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = AnthropicCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, AnthropicCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = AnthropicCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_convert_unknown_chunk_returns_chunk() {
        let result = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            AnthropicStreamMessage::ContentBlockStart {
                content_block: FlattenUnknown::Unknown(Cow::Owned(
                    serde_json::json!({"my_unknown": "content_block"}),
                )),
                index: 0,
            },
            Duration::from_secs(0),
            &mut Default::default(),
            &mut Default::default(),
            false,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlockChunk::Unknown(UnknownChunk { id, data, .. }) => {
                assert_eq!(id, "0");
                assert_eq!(
                    data.get("my_unknown").and_then(|v| v.as_str()),
                    Some("content_block")
                );
            }
            _ => panic!("Expected Unknown chunk"),
        }
    }

    #[test]
    fn test_convert_unknown_chunk_warn() {
        let logs_contain = crate::utils::testing::capture_logs();
        let res = anthropic_to_tensorzero_stream_message(
            "my_raw_chunk".to_string(),
            AnthropicStreamMessage::ContentBlockStart {
                content_block: FlattenUnknown::Unknown(Cow::Owned(
                    serde_json::json!({"my_unknown": "content_block"}),
                )),
                index: 0,
            },
            Duration::from_secs(0),
            &mut Default::default(),
            &mut Default::default(),
            true,
            "test_model",
            "test_provider",
            PROVIDER_TYPE,
        )
        .unwrap();
        assert_eq!(res, None);
        assert!(logs_contain("Discarding unknown chunk"));
    }

    #[test]
    fn test_anthropic_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = AnthropicRequestBody {
            model: "claude-3-5-sonnet-20241022",
            messages: vec![],
            max_tokens: 1024,
            ..Default::default()
        };

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort warns with tip about thinking_budget_tokens
        assert!(logs_contain(
            "Anthropic does not support the inference parameter `reasoning_effort`, so it will be ignored. Tip: You might want to use `thinking_budget_tokens` for this provider."
        ));

        // Test that thinking_budget_tokens is applied correctly
        assert_eq!(
            request.thinking,
            Some(AnthropicThinkingConfig {
                r#type: "enabled",
                budget_tokens: 1024,
            })
        );

        // Test that verbosity warns
        assert!(logs_contain(
            "Anthropic does not support the inference parameter `verbosity`"
        ));
    }

    #[tokio::test]
    async fn test_anthropic_warns_on_detail() {
        let logs_contain = capture_logs();

        // Test URL forwarding path with detail
        let url = Url::parse("https://example.com/image.png").unwrap();
        let content_block = ContentBlock::File(Box::new(LazyFile::Url {
            file_url: FileUrl {
                url: url.clone(),
                mime_type: Some(mime::IMAGE_PNG),
                detail: Some(Detail::Low),
            },
            future: async { panic!("Should not resolve") }.boxed().shared(),
        }));

        let config = AnthropicMessagesConfig {
            fetch_and_encode_input_files_before_inference: false,
        };

        let _result =
            AnthropicMessageContent::from_content_block(&content_block, config, PROVIDER_TYPE)
                .await;

        // Should log a warning about detail not being supported
        assert!(logs_contain(
            "The image detail parameter is not supported by Anthropic"
        ));
    }

    #[test]
    fn test_anthropic_respects_allowed_tools() {
        use crate::providers::test_helpers::{QUERY_TOOL, WEATHER_TOOL};
        use crate::tool::{AllowedTools, AllowedToolsChoice};

        // Create a ToolCallConfig with two tools but only allow one
        let tool_config = ToolCallConfig {
            static_tools_available: vec![WEATHER_TOOL.clone(), QUERY_TOOL.clone()],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools {
                tools: vec!["get_temperature".to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
        };

        // Convert to Anthropic tools
        let tools: Vec<AnthropicTool> = tool_config
            .strict_tools_available()
            .unwrap()
            .map(|tool| AnthropicTool::new(tool, false))
            .collect();

        // Verify only the allowed tool is included
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_temperature");
    }
}
