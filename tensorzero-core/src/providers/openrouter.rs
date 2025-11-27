use futures::future::try_join_all;
use futures::{Stream, StreamExt, TryStreamExt};
use lazy_static::lazy_static;
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::de::IntoDeserializer;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow;

use crate::http::TensorzeroHttpClient;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;
#[cfg(test)]
use uuid::Uuid;

use crate::cache::ModelProviderRequest;
use crate::embeddings::EmbeddingEncodingFormat;
use crate::embeddings::{
    Embedding, EmbeddingInput, EmbeddingProvider, EmbeddingProviderRequestInfo,
    EmbeddingProviderResponse, EmbeddingRequest,
};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{
    warn_discarded_thought_block, DelayedError, DisplayOrDebugGateway, Error, ErrorDetails,
};
use crate::inference::types::batch::StartBatchProviderInferenceResponse;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2,
};
use crate::inference::types::file::{mime_type_to_audio_format, mime_type_to_ext};
use crate::inference::types::ObjectStorageFile;
use crate::inference::types::{
    resolved_input::{FileUrl, LazyFile},
    ContentBlock, ContentBlockChunk, ContentBlockOutput, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, RequestMessage, Role, Text,
    TextChunk, Unknown, Usage,
};
use crate::inference::types::{
    FinishReason, ProviderInferenceResponseArgs, ProviderInferenceResponseStreamInner,
};
use crate::inference::InferenceProvider;
use crate::model::{Credential, ModelProvider};
use crate::tool::{FunctionToolConfig, ToolCall, ToolCallChunk, ToolChoice};

use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::helpers::{
    convert_stream_error, inject_extra_request_data_and_send,
    inject_extra_request_data_and_send_eventsource, warn_cannot_forward_url_if_missing_mime_type,
};

use super::chat_completions::{
    ChatCompletionAllowedToolsMode, ChatCompletionTool, ChatCompletionToolChoice,
    ChatCompletionToolChoiceString,
};
// Import unified OpenAI types for allowed_tools support
use super::openai::{
    AllowedToolsChoice as OpenAIAllowedToolsChoice,
    AllowedToolsConstraint as OpenAIAllowedToolsConstraint, AllowedToolsMode,
    SpecificToolFunction as OpenAISpecificToolFunction, ToolReference,
};

use crate::inference::types::extra_body::FullExtraBodyConfig;
use crate::inference::TensorZeroEventError;
use crate::providers::openai::OpenAIEmbeddingUsage;

lazy_static! {
    static ref OPENROUTER_DEFAULT_BASE_URL: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://openrouter.ai/api/v1")
            .expect("Failed to parse OPENROUTER_DEFAULT_BASE_URL")
    };
}

const PROVIDER_NAME: &str = "OpenRouter";
pub const PROVIDER_TYPE: &str = "openrouter";

type PreparedOpenRouterToolsResult<'a> = (
    Option<Vec<OpenRouterTool<'a>>>,
    Option<OpenRouterToolChoice<'a>>,
    Option<bool>,
);

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenRouterProvider {
    model_name: String,
    #[serde(skip)]
    credentials: OpenRouterCredentials,
}

impl OpenRouterProvider {
    pub fn new(model_name: String, credentials: OpenRouterCredentials) -> Self {
        OpenRouterProvider {
            model_name,
            credentials,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum OpenRouterCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<OpenRouterCredentials>,
        fallback: Box<OpenRouterCredentials>,
    },
}

impl TryFrom<Credential> for OpenRouterCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(OpenRouterCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(OpenRouterCredentials::Dynamic(key_name)),
            Credential::None => Ok(OpenRouterCredentials::None),
            Credential::Missing => Ok(OpenRouterCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(OpenRouterCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for OpenRouter provider".to_string(),
            })),
        }
    }
}

impl OpenRouterCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, DelayedError> {
        match self {
            OpenRouterCredentials::Static(api_key) => Ok(Some(api_key)),
            OpenRouterCredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                }))
                .transpose()
            }
            OpenRouterCredentials::None => Ok(None),
            OpenRouterCredentials::WithFallback { default, fallback } => {
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

impl InferenceProvider for OpenRouterProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_url = get_chat_url(&OPENROUTER_DEFAULT_BASE_URL)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let request_body_obj = OpenRouterRequest::new(&self.model_name, request.request).await?;
        let request_body = serde_json::to_value(request_body_obj).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let mut request_builder = http_client
            .post(request_url)
            .header("X-Title", "TensorZero")
            .header("HTTP-Referer", "https://www.tensorzero.com/");

        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.request.extra_body,
            &request.request.extra_headers,
            model_provider,
            request.model_name,
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
            Ok(OpenRouterResponseWithMetadata {
                response,
                raw_response,
                latency,
                raw_request: raw_request.clone(),
                generic_request: request.request,
            }
            .try_into()?)
        } else {
            Err(handle_openrouter_error(
                &raw_request.clone(),
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        raw_request: Some(raw_request),
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?,
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
            serde_json::to_value(OpenRouterRequest::new(&self.model_name, request).await?)
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing OpenRouter request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;
        let request_url = get_chat_url(&OPENROUTER_DEFAULT_BASE_URL)?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let mut request_builder = http_client
            .post(request_url)
            .header("X-Title", "TensorZero")
            .header("HTTP-Referer", "https://www.tensorzero.com/");
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

        let stream = stream_openrouter(
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

impl EmbeddingProvider for OpenRouterProvider {
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
        let request_body = OpenRouterEmbeddingRequest::new(
            &self.model_name,
            &request.input,
            request.encoding_format,
        );
        let request_url = get_embedding_url(&OPENROUTER_DEFAULT_BASE_URL)?;
        let start_time = Instant::now();
        let mut request_builder = client
            .post(request_url)
            .header("X-Title", "TensorZero")
            .header("HTTP-Referer", "https://www.tensorzero.com/");
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }

        let request_body_value = serde_json::to_value(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing OpenRouter embedding request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &FullExtraBodyConfig::default(), // No overrides supported
            &Default::default(),             // No extra headers for embeddings yet
            model_provider_data,
            &self.model_name,
            request_body_value,
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

            let response: OpenRouterEmbeddingResponse = serde_json::from_str(&raw_response)
                .map_err(|e| {
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

            Ok(OpenRouterEmbeddingResponseWithMetadata {
                response,
                latency,
                request: request_body,
                raw_response,
            }
            .try_into()?)
        } else {
            Err(handle_openrouter_error(
                &raw_request,
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        raw_request: Some(raw_request.clone()),
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?,
                PROVIDER_TYPE,
            ))
        }
    }
}

pub fn stream_openrouter(
    provider_type: String,
    event_source: impl Stream<Item = Result<Event, TensorZeroEventError>> + Send + 'static,
    start_time: Instant,
    raw_request: &str,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    let mut tool_call_ids = Vec::new();
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
                            yield Err(convert_stream_error(raw_request.clone(), provider_type.clone(), e).await);
                        }
                    }
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<OpenRouterChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {e}",
                                ),
                                raw_request: Some(raw_request.clone()),
                                raw_response: Some(message.data.clone()),
                                provider_type: provider_type.clone(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            openrouter_to_tensorzero_chunk(message.data, d, latency, &mut tool_call_ids)
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

pub(super) fn get_chat_url(base_url: &Url) -> Result<Url, Error> {
    let mut url = base_url.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    url.join("chat/completions").map_err(|e| {
        Error::new(ErrorDetails::InvalidBaseUrl {
            message: e.to_string(),
        })
    })
}

fn get_embedding_url(base_url: &Url) -> Result<Url, Error> {
    let mut url = base_url.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    url.join("embeddings").map_err(|e| {
        Error::new(ErrorDetails::InvalidBaseUrl {
            message: e.to_string(),
        })
    })
}

// Functions only needed for tests
#[cfg(test)]
fn get_file_url(base_url: &Url, file_id: Option<&str>) -> Result<Url, Error> {
    let mut url = base_url.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    let path = if let Some(id) = file_id {
        format!("files/{id}/content")
    } else {
        "files".to_string()
    };
    url.join(&path).map_err(|e| {
        Error::new(ErrorDetails::InvalidBaseUrl {
            message: e.to_string(),
        })
    })
}

pub(super) fn handle_openrouter_error(
    raw_request: &str,
    response_code: StatusCode,
    response_body: &str,
    provider_type: &str,
) -> Error {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => ErrorDetails::InferenceClient {
            status_code: Some(response_code),
            message: response_body.to_string(),
            raw_request: Some(raw_request.to_string()),
            raw_response: Some(response_body.to_string()),
            provider_type: provider_type.to_string(),
        }
        .into(),
        _ => ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            raw_request: Some(raw_request.to_string()),
            raw_response: Some(response_body.to_string()),
            provider_type: provider_type.to_string(),
        }
        .into(),
    }
}

// Embedding-related structures
#[derive(Debug, Serialize)]
struct OpenRouterEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a EmbeddingInput,
    // Note: OpenRouter doesn't support the dimensions parameter
    encoding_format: EmbeddingEncodingFormat,
}

impl<'a> OpenRouterEmbeddingRequest<'a> {
    fn new(
        model: &'a str,
        input: &'a EmbeddingInput,
        encoding_format: EmbeddingEncodingFormat,
    ) -> Self {
        Self {
            model,
            input,
            encoding_format,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenRouterEmbeddingResponse {
    data: Vec<OpenRouterEmbeddingData>,
    usage: Option<OpenAIEmbeddingUsage>,
}

struct OpenRouterEmbeddingResponseWithMetadata<'a> {
    response: OpenRouterEmbeddingResponse,
    latency: Latency,
    request: OpenRouterEmbeddingRequest<'a>,
    raw_response: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenRouterEmbeddingData {
    embedding: Embedding,
}

impl<'a> TryFrom<OpenRouterEmbeddingResponseWithMetadata<'a>> for EmbeddingProviderResponse {
    type Error = Error;
    fn try_from(
        response: OpenRouterEmbeddingResponseWithMetadata<'a>,
    ) -> Result<Self, Self::Error> {
        let OpenRouterEmbeddingResponseWithMetadata {
            response,
            latency,
            request,
            raw_response,
        } = response;
        let raw_request = serde_json::to_string(&request).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error serializing request body as JSON: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                raw_request: Some(serde_json::to_string(&request).unwrap_or_default()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        let embeddings = response
            .data
            .into_iter()
            .map(|embedding| embedding.embedding)
            .collect();

        Ok(EmbeddingProviderResponse::new(
            embeddings,
            request.input.clone(),
            raw_request,
            raw_response,
            response.usage.map(|usage| usage.into()).unwrap_or_default(),
            latency,
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenRouterSystemRequestMessage<'a> {
    pub content: Cow<'a, str>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenRouterUserRequestMessage<'a> {
    #[serde(serialize_with = "serialize_text_content_vec")]
    pub(super) content: Vec<OpenRouterContentBlock<'a>>,
}

fn serialize_text_content_vec<S>(
    content: &Vec<OpenRouterContentBlock<'_>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // If we have a single text block, serialize it as a string
    // to stay compatible with older providers which may not support content blocks
    if let [OpenRouterContentBlock::Text { text }] = &content.as_slice() {
        text.serialize(serializer)
    } else {
        content.serialize(serializer)
    }
}

fn serialize_optional_text_content_vec<S>(
    content: &Option<Vec<OpenRouterContentBlock<'_>>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match content {
        Some(vec) => serialize_text_content_vec(vec, serializer),
        None => serializer.serialize_none(),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum OpenRouterContentBlock<'a> {
    Text {
        text: Cow<'a, str>,
    },
    ImageUrl {
        image_url: OpenRouterImageUrl,
    },
    File {
        file: OpenRouterFile<'a>,
    },
    InputAudio {
        input_audio: OpenRouterInputAudio<'a>,
    },
    Unknown {
        data: Cow<'a, Value>,
    },
}

impl Serialize for OpenRouterContentBlock<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Helper<'a> {
            Text {
                text: &'a str,
            },
            ImageUrl {
                image_url: &'a OpenRouterImageUrl,
            },
            File {
                file: &'a OpenRouterFile<'a>,
            },
            InputAudio {
                input_audio: &'a OpenRouterInputAudio<'a>,
            },
        }
        match self {
            OpenRouterContentBlock::Text { text } => Helper::Text { text }.serialize(serializer),
            OpenRouterContentBlock::ImageUrl { image_url } => {
                Helper::ImageUrl { image_url }.serialize(serializer)
            }
            OpenRouterContentBlock::File { file } => Helper::File { file }.serialize(serializer),
            OpenRouterContentBlock::InputAudio { input_audio } => {
                Helper::InputAudio { input_audio }.serialize(serializer)
            }
            OpenRouterContentBlock::Unknown { data } => data.serialize(serializer),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct OpenRouterImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct OpenRouterFile<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<Cow<'a, str>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct OpenRouterInputAudio<'a> {
    pub data: Cow<'a, str>,
    pub format: Cow<'a, str>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenRouterRequestFunctionCall<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct OpenRouterRequestToolCall<'a> {
    pub id: &'a str,
    pub r#type: OpenRouterToolType,
    pub function: OpenRouterRequestFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for OpenRouterRequestToolCall<'a> {
    fn from(tool_call: &'a ToolCall) -> Self {
        OpenRouterRequestToolCall {
            id: &tool_call.id,
            r#type: OpenRouterToolType::Function,
            function: OpenRouterRequestFunctionCall {
                name: &tool_call.name,
                arguments: &tool_call.arguments,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenRouterAssistantRequestMessage<'a> {
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_text_content_vec"
    )]
    pub content: Option<Vec<OpenRouterContentBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenRouterRequestToolCall<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenRouterToolRequestMessage<'a> {
    pub content: &'a str,
    pub tool_call_id: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub(super) enum OpenRouterRequestMessage<'a> {
    System(OpenRouterSystemRequestMessage<'a>),
    User(OpenRouterUserRequestMessage<'a>),
    Assistant(OpenRouterAssistantRequestMessage<'a>),
    Tool(OpenRouterToolRequestMessage<'a>),
}

impl OpenRouterRequestMessage<'_> {
    pub fn content_contains_case_insensitive(&self, value: &str) -> bool {
        match self {
            OpenRouterRequestMessage::System(msg) => msg.content.to_lowercase().contains(value),
            OpenRouterRequestMessage::User(msg) => msg.content.iter().any(|c| match c {
                OpenRouterContentBlock::Text { text } => text.to_lowercase().contains(value),
                OpenRouterContentBlock::ImageUrl { .. } => false,
                OpenRouterContentBlock::File { .. } => false,
                OpenRouterContentBlock::InputAudio { .. } => false,
                // Don't inspect the contents of 'unknown' blocks
                OpenRouterContentBlock::Unknown { data: _ } => false,
            }),
            OpenRouterRequestMessage::Assistant(msg) => {
                if let Some(content) = &msg.content {
                    content.iter().any(|c| match c {
                        OpenRouterContentBlock::Text { text } => {
                            text.to_lowercase().contains(value)
                        }
                        OpenRouterContentBlock::ImageUrl { .. } => false,
                        OpenRouterContentBlock::File { .. } => false,
                        OpenRouterContentBlock::InputAudio { .. } => false,
                        // Don't inspect the contents of 'unknown' blocks
                        OpenRouterContentBlock::Unknown { data: _ } => false,
                    })
                } else {
                    false
                }
            }
            OpenRouterRequestMessage::Tool(msg) => msg.content.to_lowercase().contains(value),
        }
    }
}

pub(super) async fn prepare_openrouter_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Result<Vec<OpenRouterRequestMessage<'a>>, Error> {
    let fetch_and_encode = request.fetch_and_encode_input_files_before_inference;
    let mut messages: Vec<_> = try_join_all(
        request
            .messages
            .iter()
            .map(|msg| tensorzero_to_openrouter_messages(msg, fetch_and_encode)),
    )
    .await?
    .into_iter()
    .flatten()
    .collect();
    if let Some(system_msg) = tensorzero_to_openrouter_system_message(
        request.system.as_deref(),
        request.json_mode,
        &messages,
    ) {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to OpenRouter format
pub(super) fn prepare_openrouter_tools<'a>(
    request: &'a ModelInferenceRequest,
) -> Result<PreparedOpenRouterToolsResult<'a>, Error> {
    let (tools, tool_choice, parallel_tool_calls) = prepare_chat_completion_tools(request, true)?;

    // Convert from ChatCompletionTool to OpenRouterTool
    let openrouter_tools = tools.map(|t| t.into_iter().map(OpenRouterTool::from).collect());

    // Convert from ChatCompletionToolChoice to OpenRouterToolChoice
    let openrouter_tool_choice = tool_choice.map(OpenRouterToolChoice::from);

    Ok((
        openrouter_tools,
        openrouter_tool_choice,
        parallel_tool_calls,
    ))
}

/// This function is complicated only by the fact that OpenRouter and Azure require
/// different instructions depending on the json mode and the content of the messages.
///
/// If ModelInferenceRequestJsonMode::On and the system message or instructions does not contain "JSON"
/// the request will return an error.
/// So, we need to format the instructions to include "Respond using JSON." if it doesn't already.
pub(super) fn tensorzero_to_openrouter_system_message<'a>(
    system: Option<&'a str>,
    json_mode: ModelInferenceRequestJsonMode,
    messages: &[OpenRouterRequestMessage<'a>],
) -> Option<OpenRouterRequestMessage<'a>> {
    match system {
        Some(system) => {
            match json_mode {
                ModelInferenceRequestJsonMode::On => {
                    if messages
                        .iter()
                        .any(|msg| msg.content_contains_case_insensitive("json"))
                        || system.to_lowercase().contains("json")
                    {
                        OpenRouterRequestMessage::System(OpenRouterSystemRequestMessage {
                            content: Cow::Borrowed(system),
                        })
                    } else {
                        let formatted_instructions = format!("Respond using JSON.\n\n{system}");
                        OpenRouterRequestMessage::System(OpenRouterSystemRequestMessage {
                            content: Cow::Owned(formatted_instructions),
                        })
                    }
                }

                // If JSON mode is either off or strict, we don't need to do anything special
                _ => OpenRouterRequestMessage::System(OpenRouterSystemRequestMessage {
                    content: Cow::Borrowed(system),
                }),
            }
            .into()
        }
        None => match json_mode {
            ModelInferenceRequestJsonMode::On => Some(OpenRouterRequestMessage::System(
                OpenRouterSystemRequestMessage {
                    content: Cow::Owned("Respond using JSON.".to_string()),
                },
            )),
            _ => None,
        },
    }
}

async fn prepare_openrouter_file_content_block(
    file: &LazyFile,
    fetch_and_encode_input_files_before_inference: bool,
) -> Result<OpenRouterContentBlock<'static>, Error> {
    match file {
        LazyFile::Url {
            file_url:
                FileUrl {
                    mime_type,
                    url,
                    detail,
                },
            future: _,
        } if !fetch_and_encode_input_files_before_inference
            && matches!(
                mime_type.as_ref().map(mime::MediaType::type_),
                Some(mime::IMAGE) | None
            ) =>
        {
            if detail.is_some() {
                tracing::warn!(
                    "The image detail parameter is not supported by OpenRouter. The `detail` field will be ignored."
                );
            }
            warn_cannot_forward_url_if_missing_mime_type(
                file,
                fetch_and_encode_input_files_before_inference,
                PROVIDER_TYPE,
            );
            Ok(OpenRouterContentBlock::ImageUrl {
                image_url: OpenRouterImageUrl {
                    url: url.to_string(),
                },
            })
        }
        _ => {
            let resolved_file = file.resolve().await?;
            let ObjectStorageFile { file, data } = &*resolved_file;
            let base64_url = format!("data:{};base64,{}", file.mime_type, data);

            if file.mime_type.type_() == mime::IMAGE {
                if file.detail.is_some() {
                    tracing::warn!(
                        "The image detail parameter is not supported by OpenRouter. The `detail` field will be ignored."
                    );
                }
                Ok(OpenRouterContentBlock::ImageUrl {
                    image_url: OpenRouterImageUrl { url: base64_url },
                })
            } else if file.mime_type.type_() == mime::AUDIO {
                let format = mime_type_to_audio_format(&file.mime_type)?;
                Ok(OpenRouterContentBlock::InputAudio {
                    input_audio: OpenRouterInputAudio {
                        data: Cow::Owned(data.clone()),
                        format: Cow::Owned(format.to_string()),
                    },
                })
            } else {
                let filename = if let Some(ref user_filename) = file.filename {
                    // Use the user-provided filename if available
                    Cow::Owned(user_filename.clone())
                } else {
                    // Otherwise, generate a filename with the appropriate extension
                    let suffix = mime_type_to_ext(&file.mime_type)?.ok_or_else(|| {
                        Error::new(ErrorDetails::InvalidMessage {
                            message: format!("Mime type {} has no filetype suffix", file.mime_type),
                        })
                    })?;
                    Cow::Owned(format!("input.{suffix}"))
                };
                Ok(OpenRouterContentBlock::File {
                    file: OpenRouterFile {
                        file_data: Some(Cow::Owned(base64_url)),
                        filename: Some(filename),
                    },
                })
            }
        }
    }
}

pub(super) async fn tensorzero_to_openrouter_messages(
    message: &RequestMessage,
    fetch_and_encode_input_files_before_inference: bool,
) -> Result<Vec<OpenRouterRequestMessage<'_>>, Error> {
    match message.role {
        Role::User => {
            tensorzero_to_openrouter_user_messages(
                &message.content,
                fetch_and_encode_input_files_before_inference,
            )
            .await
        }
        Role::Assistant => {
            tensorzero_to_openrouter_assistant_messages(
                &message.content,
                fetch_and_encode_input_files_before_inference,
            )
            .await
        }
    }
}

async fn tensorzero_to_openrouter_user_messages(
    content_blocks: &[ContentBlock],
    fetch_and_encode_input_files_before_inference: bool,
) -> Result<Vec<OpenRouterRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the user content blocks.

    let mut messages = Vec::new();
    let mut user_content_blocks = Vec::new();

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                user_content_blocks.push(OpenRouterContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool calls are not supported in user messages".to_string(),
                }));
            }
            ContentBlock::ToolResult(tool_result) => {
                messages.push(OpenRouterRequestMessage::Tool(
                    OpenRouterToolRequestMessage {
                        content: &tool_result.result,
                        tool_call_id: &tool_result.id,
                    },
                ));
            }
            ContentBlock::File(file) => {
                user_content_blocks.push(
                    prepare_openrouter_file_content_block(
                        file,
                        fetch_and_encode_input_files_before_inference,
                    )
                    .await?,
                );
            }
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(PROVIDER_TYPE, thought);
            }
            ContentBlock::Unknown(Unknown { data, .. }) => {
                user_content_blocks.push(OpenRouterContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
        };
    }

    // If there are any user content blocks, combine them into a single user message.
    if !user_content_blocks.is_empty() {
        messages.push(OpenRouterRequestMessage::User(
            OpenRouterUserRequestMessage {
                content: user_content_blocks,
            },
        ));
    }

    Ok(messages)
}

async fn tensorzero_to_openrouter_assistant_messages(
    content_blocks: &[ContentBlock],
    fetch_and_encode_input_files_before_inference: bool,
) -> Result<Vec<OpenRouterRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the assistant content blocks.
    let mut assistant_content_blocks = Vec::new();
    let mut assistant_tool_calls = Vec::new();

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                assistant_content_blocks.push(OpenRouterContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(tool_call) => {
                let tool_call = OpenRouterRequestToolCall {
                    id: &tool_call.id,
                    r#type: OpenRouterToolType::Function,
                    function: OpenRouterRequestFunctionCall {
                        name: &tool_call.name,
                        arguments: &tool_call.arguments,
                    },
                };

                assistant_tool_calls.push(tool_call);
            }
            ContentBlock::ToolResult(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool results are not supported in assistant messages".to_string(),
                }));
            }
            ContentBlock::File(file) => {
                assistant_content_blocks.push(
                    prepare_openrouter_file_content_block(
                        file,
                        fetch_and_encode_input_files_before_inference,
                    )
                    .await?,
                );
            }
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(PROVIDER_TYPE, thought);
            }
            ContentBlock::Unknown(Unknown { data, .. }) => {
                assistant_content_blocks.push(OpenRouterContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
        }
    }

    let content = match assistant_content_blocks.len() {
        0 => None,
        _ => Some(assistant_content_blocks),
    };

    let tool_calls = match assistant_tool_calls.len() {
        0 => None,
        _ => Some(assistant_tool_calls),
    };

    if content.is_none() && tool_calls.is_none() {
        return Ok(vec![]);
    }

    let message = OpenRouterRequestMessage::Assistant(OpenRouterAssistantRequestMessage {
        content,
        tool_calls,
    });

    Ok(vec![message])
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum OpenRouterResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: Value,
    },
}

impl OpenRouterResponseFormat {
    fn new(
        json_mode: ModelInferenceRequestJsonMode,
        output_schema: Option<&Value>,
        model: &str,
    ) -> Option<Self> {
        if model.contains("3.5") && json_mode == ModelInferenceRequestJsonMode::Strict {
            return Some(OpenRouterResponseFormat::JsonObject);
        }

        match json_mode {
            ModelInferenceRequestJsonMode::On => Some(OpenRouterResponseFormat::JsonObject),
            // For now, we never explicitly send `OpenRouterResponseFormat::Text`
            ModelInferenceRequestJsonMode::Off => None,
            ModelInferenceRequestJsonMode::Strict => match output_schema {
                Some(schema) => {
                    let json_schema = json!({"name": "response", "strict": true, "schema": schema});
                    Some(OpenRouterResponseFormat::JsonSchema { json_schema })
                }
                None => Some(OpenRouterResponseFormat::JsonObject),
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterToolType {
    Function,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct OpenRouterFunction<'a> {
    pub(super) name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<&'a str>,
    pub parameters: &'a Value,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct OpenRouterTool<'a> {
    pub(super) r#type: OpenRouterToolType,
    pub(super) function: OpenRouterFunction<'a>,
    pub(super) strict: bool,
}

impl<'a> From<&'a FunctionToolConfig> for OpenRouterTool<'a> {
    fn from(tool: &'a FunctionToolConfig) -> Self {
        OpenRouterTool {
            r#type: OpenRouterToolType::Function,
            function: OpenRouterFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
            strict: tool.strict(),
        }
    }
}

impl<'a> From<ChatCompletionTool<'a>> for OpenRouterTool<'a> {
    fn from(tool: ChatCompletionTool<'a>) -> Self {
        OpenRouterTool {
            r#type: OpenRouterToolType::Function,
            function: OpenRouterFunction {
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters,
            },
            strict: tool.strict,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(super) enum OpenRouterToolChoice<'a> {
    String(OpenRouterToolChoiceString),
    Specific(SpecificToolChoice<'a>),
    AllowedTools(OpenAIAllowedToolsChoice<'a>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum OpenRouterToolChoiceString {
    None,
    Auto,
    Required,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SpecificToolChoice<'a> {
    pub(super) r#type: OpenRouterToolType,
    pub(super) function: SpecificToolFunction<'a>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct SpecificToolFunction<'a> {
    pub(super) name: &'a str,
}

impl Default for OpenRouterToolChoice<'_> {
    fn default() -> Self {
        OpenRouterToolChoice::String(OpenRouterToolChoiceString::None)
    }
}

impl<'a> From<&'a ToolChoice> for OpenRouterToolChoice<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::None => OpenRouterToolChoice::String(OpenRouterToolChoiceString::None),
            ToolChoice::Auto => OpenRouterToolChoice::String(OpenRouterToolChoiceString::Auto),
            ToolChoice::Required => {
                OpenRouterToolChoice::String(OpenRouterToolChoiceString::Required)
            }
            ToolChoice::Specific(tool_name) => OpenRouterToolChoice::Specific(SpecificToolChoice {
                r#type: OpenRouterToolType::Function,
                function: SpecificToolFunction { name: tool_name },
            }),
        }
    }
}

impl<'a> From<ChatCompletionToolChoice<'a>> for OpenRouterToolChoice<'a> {
    fn from(tool_choice: ChatCompletionToolChoice<'a>) -> Self {
        match tool_choice {
            ChatCompletionToolChoice::String(tc_string) => match tc_string {
                ChatCompletionToolChoiceString::None => {
                    OpenRouterToolChoice::String(OpenRouterToolChoiceString::None)
                }
                ChatCompletionToolChoiceString::Auto => {
                    OpenRouterToolChoice::String(OpenRouterToolChoiceString::Auto)
                }
                ChatCompletionToolChoiceString::Required => {
                    OpenRouterToolChoice::String(OpenRouterToolChoiceString::Required)
                }
            },
            ChatCompletionToolChoice::Specific(specific) => {
                OpenRouterToolChoice::Specific(SpecificToolChoice {
                    r#type: OpenRouterToolType::Function,
                    function: SpecificToolFunction {
                        name: specific.function.name,
                    },
                })
            }
            ChatCompletionToolChoice::AllowedTools(allowed_tools) => {
                // Convert from common ChatCompletionAllowedToolsChoice to OpenAI AllowedToolsChoice
                OpenRouterToolChoice::AllowedTools(OpenAIAllowedToolsChoice {
                    r#type: allowed_tools.r#type,
                    allowed_tools: OpenAIAllowedToolsConstraint {
                        mode: match allowed_tools.allowed_tools.mode {
                            ChatCompletionAllowedToolsMode::Auto => AllowedToolsMode::Auto,
                            ChatCompletionAllowedToolsMode::Required => AllowedToolsMode::Required,
                        },
                        tools: allowed_tools
                            .allowed_tools
                            .tools
                            .into_iter()
                            .map(|tool_ref| ToolReference::Function {
                                function: OpenAISpecificToolFunction {
                                    name: tool_ref.function.name,
                                },
                            })
                            .collect(),
                    },
                })
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct StreamOptions {
    pub(super) include_usage: bool,
}

/// This struct defines the supported parameters for the OpenRouter API
/// See the [OpenRouter API documentation](https://platform.openrouter.com/docs/api-reference/chat/create)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Serialize)]
struct OpenRouterRequest<'a> {
    messages: Vec<OpenRouterRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
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
    response_format: Option<OpenRouterResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenRouterTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenRouterToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<String>,
}

fn apply_inference_params(
    request: &mut OpenRouterRequest,
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
        request.verbosity = verbosity.clone();
    }
}

impl<'a> OpenRouterRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<OpenRouterRequest<'a>, Error> {
        let response_format =
            OpenRouterResponseFormat::new(request.json_mode, request.output_schema, model);
        let stream_options = if request.stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        };
        let mut messages = prepare_openrouter_messages(request).await?;

        let (tools, tool_choice, mut parallel_tool_calls) = prepare_openrouter_tools(request)?;
        if model.to_lowercase().starts_with("o1") && parallel_tool_calls == Some(false) {
            parallel_tool_calls = None;
        }

        if model.to_lowercase().starts_with("o1-mini") {
            if let Some(OpenRouterRequestMessage::System(_)) = messages.first() {
                if let OpenRouterRequestMessage::System(system_msg) = messages.remove(0) {
                    let user_msg = OpenRouterRequestMessage::User(OpenRouterUserRequestMessage {
                        content: vec![OpenRouterContentBlock::Text {
                            text: system_msg.content,
                        }],
                    });
                    messages.insert(0, user_msg);
                }
            }
        }

        let mut openrouter_request = OpenRouterRequest {
            messages,
            model,
            temperature: request.temperature,
            max_completion_tokens: request.max_tokens,
            seed: request.seed,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            stream: request.stream,
            stream_options,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
            stop: request.borrow_stop_sequences(),
            reasoning_effort: None,
            verbosity: None,
        };

        apply_inference_params(&mut openrouter_request, &request.inference_params_v2);

        Ok(openrouter_request)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct OpenRouterUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

impl From<OpenRouterUsage> for Usage {
    fn from(usage: OpenRouterUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenRouterResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub(super) struct OpenRouterResponseToolCall {
    id: String,
    r#type: OpenRouterToolType,
    function: OpenRouterResponseFunctionCall,
}

impl From<OpenRouterResponseToolCall> for ToolCall {
    fn from(openrouter_tool_call: OpenRouterResponseToolCall) -> Self {
        ToolCall {
            id: openrouter_tool_call.id,
            name: openrouter_tool_call.function.name,
            arguments: openrouter_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct OpenRouterResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_calls: Option<Vec<OpenRouterResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum OpenRouterFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

impl From<OpenRouterFinishReason> for FinishReason {
    fn from(finish_reason: OpenRouterFinishReason) -> Self {
        match finish_reason {
            OpenRouterFinishReason::Stop => FinishReason::Stop,
            OpenRouterFinishReason::Length => FinishReason::Length,
            OpenRouterFinishReason::ContentFilter => FinishReason::ContentFilter,
            OpenRouterFinishReason::ToolCalls => FinishReason::ToolCall,
            OpenRouterFinishReason::FunctionCall => FinishReason::ToolCall,
            OpenRouterFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct OpenRouterResponseChoice {
    pub(super) index: u8,
    pub(super) message: OpenRouterResponseMessage,
    pub(super) finish_reason: OpenRouterFinishReason,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct OpenRouterResponse {
    pub(super) choices: Vec<OpenRouterResponseChoice>,
    pub(super) usage: OpenRouterUsage,
}

struct OpenRouterResponseWithMetadata<'a> {
    response: OpenRouterResponse,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    raw_response: String,
}

impl<'a> TryFrom<OpenRouterResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: OpenRouterResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let OpenRouterResponseWithMetadata {
            mut response,
            latency,
            raw_request,
            raw_response,
            generic_request,
        } = value;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                raw_request: Some(raw_request),
                raw_response: Some(serde_json::to_string(&response).unwrap_or_default()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }
        let OpenRouterResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(serde_json::to_string(&response).unwrap_or_default()),
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
        };
        let usage = response.usage.into();
        let system = generic_request.system.clone();
        let messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages: messages,
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
struct OpenRouterFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenRouterToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: OpenRouterFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenRouterDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenRouterToolCallChunk>>,
}

// Custom deserializer function for empty string to None
// This is required because SGLang (which depends on this code) returns "" in streaming chunks instead of null
fn empty_string_as_none<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    if let Some(s) = opt {
        if s.is_empty() {
            return Ok(None);
        }
        // Convert serde_json::Error to D::Error
        Ok(Some(
            T::deserialize(serde_json::Value::String(s).into_deserializer())
                .map_err(|e| serde::de::Error::custom(e.to_string()))?,
        ))
    } else {
        Ok(None)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenRouterChatChunkChoice {
    delta: OpenRouterDelta,
    #[serde(default)]
    #[serde(deserialize_with = "empty_string_as_none")]
    finish_reason: Option<OpenRouterFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenRouterChatChunk {
    choices: Vec<OpenRouterChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenRouterUsage>,
}

/// Maps an OpenRouter chunk to a TensorZero chunk for streaming inferences
fn openrouter_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: OpenRouterChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        inference::types::{FunctionType, RequestMessage},
        providers::test_helpers::{
            MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL, WEATHER_TOOL_CONFIG,
        },
        tool::ToolCallConfig,
    };
    use serde_json::json;
    use std::borrow::Cow;

    #[test]
    fn test_get_chat_url() {
        // Test with custom base URL
        let custom_base = "https://custom.openrouter.com/api/";
        let custom_url = get_chat_url(&Url::parse(custom_base).unwrap()).unwrap();
        assert_eq!(
            custom_url.as_str(),
            "https://custom.openrouter.com/api/chat/completions"
        );

        // Test with URL without trailing slash
        let unjoinable_url = get_chat_url(&Url::parse("https://example.com").unwrap());
        assert!(unjoinable_url.is_ok());
        assert_eq!(
            unjoinable_url.unwrap().as_str(),
            "https://example.com/chat/completions"
        );
        // Test with URL that can't be joined
        let unjoinable_url = get_chat_url(&Url::parse("https://example.com/foo").unwrap());
        assert!(unjoinable_url.is_ok());
        assert_eq!(
            unjoinable_url.unwrap().as_str(),
            "https://example.com/foo/chat/completions"
        );
    }

    #[test]
    fn test_handle_openrouter_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_openrouter_error(
            "Request Body",
            StatusCode::UNAUTHORIZED,
            "Unauthorized access",
            PROVIDER_TYPE,
        );
        let details = unauthorized.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            status_code,
            message,
            raw_request,
            raw_response,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Unauthorized access");
            assert_eq!(*status_code, Some(StatusCode::UNAUTHORIZED));
            assert_eq!(provider, PROVIDER_TYPE);
            assert_eq!(*raw_request, Some("Request Body".to_string()));
            assert_eq!(*raw_response, Some("Unauthorized access".to_string()));
        }

        // Test forbidden error
        let forbidden = handle_openrouter_error(
            "Request Body",
            StatusCode::FORBIDDEN,
            "Forbidden access",
            PROVIDER_TYPE,
        );
        let details = forbidden.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            raw_request,
            raw_response,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Forbidden access");
            assert_eq!(*status_code, Some(StatusCode::FORBIDDEN));
            assert_eq!(provider, PROVIDER_TYPE);
            assert_eq!(*raw_request, Some("Request Body".to_string()));
            assert_eq!(*raw_response, Some("Forbidden access".to_string()));
        }

        // Test rate limit error
        let rate_limit = handle_openrouter_error(
            "Request Body",
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded",
            PROVIDER_TYPE,
        );
        let details = rate_limit.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            raw_request,
            raw_response,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Rate limit exceeded");
            assert_eq!(*status_code, Some(StatusCode::TOO_MANY_REQUESTS));
            assert_eq!(provider, PROVIDER_TYPE);
            assert_eq!(*raw_request, Some("Request Body".to_string()));
            assert_eq!(*raw_response, Some("Rate limit exceeded".to_string()));
        }

        // Test server error
        let server_error = handle_openrouter_error(
            "Request Body",
            StatusCode::INTERNAL_SERVER_ERROR,
            "Server error",
            PROVIDER_TYPE,
        );
        let details = server_error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        if let ErrorDetails::InferenceServer {
            message,
            raw_request,
            raw_response,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Server error");
            assert_eq!(provider, PROVIDER_TYPE);
            assert_eq!(*raw_request, Some("Request Body".to_string()));
            assert_eq!(*raw_response, Some("Server error".to_string()));
        }
    }

    #[tokio::test]
    async fn test_openrouter_request_new() {
        // Test basic request
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

        let openrouter_request = OpenRouterRequest::new("gpt-4.1-mini", &basic_request)
            .await
            .unwrap();

        assert_eq!(openrouter_request.model, "gpt-4.1-mini");
        assert_eq!(openrouter_request.messages.len(), 2);
        assert_eq!(openrouter_request.temperature, Some(0.7));
        assert_eq!(openrouter_request.max_completion_tokens, Some(100));
        assert_eq!(openrouter_request.seed, Some(69));
        assert_eq!(openrouter_request.top_p, Some(0.9));
        assert_eq!(openrouter_request.presence_penalty, Some(0.1));
        assert_eq!(openrouter_request.frequency_penalty, Some(0.2));
        assert!(openrouter_request.stream);
        assert_eq!(openrouter_request.response_format, None);
        assert!(openrouter_request.tools.is_none());
        assert_eq!(openrouter_request.tool_choice, None);
        assert!(openrouter_request.parallel_tool_calls.is_none());

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

        let openrouter_request = OpenRouterRequest::new("gpt-4", &request_with_tools)
            .await
            .unwrap();

        assert_eq!(openrouter_request.model, "gpt-4");
        assert_eq!(openrouter_request.messages.len(), 2); // We'll add a system message containing Json to fit OpenRouter requirements
        assert_eq!(openrouter_request.temperature, None);
        assert_eq!(openrouter_request.max_completion_tokens, None);
        assert_eq!(openrouter_request.seed, None);
        assert_eq!(openrouter_request.top_p, None);
        assert_eq!(openrouter_request.presence_penalty, None);
        assert_eq!(openrouter_request.frequency_penalty, None);
        assert!(!openrouter_request.stream);
        assert_eq!(
            openrouter_request.response_format,
            Some(OpenRouterResponseFormat::JsonObject)
        );
        assert!(openrouter_request.tools.is_some());
        let tools = openrouter_request.tools.as_ref().unwrap();
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            openrouter_request.tool_choice,
            Some(OpenRouterToolChoice::Specific(SpecificToolChoice {
                r#type: OpenRouterToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
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

        let openrouter_request = OpenRouterRequest::new("gpt-4", &request_with_tools)
            .await
            .unwrap();

        assert_eq!(openrouter_request.model, "gpt-4");
        assert_eq!(openrouter_request.messages.len(), 1);
        assert_eq!(openrouter_request.temperature, None);
        assert_eq!(openrouter_request.max_completion_tokens, None);
        assert_eq!(openrouter_request.seed, None);
        assert!(!openrouter_request.stream);
        assert_eq!(openrouter_request.top_p, None);
        assert_eq!(openrouter_request.presence_penalty, None);
        assert_eq!(openrouter_request.frequency_penalty, None);
        // Resolves to normal JSON mode since no schema is provided (this shouldn't really happen in practice)
        assert_eq!(
            openrouter_request.response_format,
            Some(OpenRouterResponseFormat::JsonObject)
        );

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

        let openrouter_request = OpenRouterRequest::new("gpt-4", &request_with_tools)
            .await
            .unwrap();

        assert_eq!(openrouter_request.model, "gpt-4");
        assert_eq!(openrouter_request.messages.len(), 1);
        assert_eq!(openrouter_request.temperature, None);
        assert_eq!(openrouter_request.max_completion_tokens, None);
        assert_eq!(openrouter_request.seed, None);
        assert!(!openrouter_request.stream);
        assert_eq!(openrouter_request.top_p, None);
        assert_eq!(openrouter_request.presence_penalty, None);
        assert_eq!(openrouter_request.frequency_penalty, None);
        let expected_schema = serde_json::json!({"name": "response", "strict": true, "schema": {}});
        assert_eq!(
            openrouter_request.response_format,
            Some(OpenRouterResponseFormat::JsonSchema {
                json_schema: expected_schema,
            })
        );
    }

    #[tokio::test]
    async fn test_openrouter_new_request_o1() {
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["Hello".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            seed: Some(69),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let openrouter_request = OpenRouterRequest::new("o1-preview", &request)
            .await
            .unwrap();

        assert_eq!(openrouter_request.model, "o1-preview");
        assert_eq!(openrouter_request.messages.len(), 1);
        assert!(!openrouter_request.stream);
        assert_eq!(openrouter_request.response_format, None);
        assert_eq!(openrouter_request.temperature, Some(0.5));
        assert_eq!(openrouter_request.max_completion_tokens, Some(100));
        assert_eq!(openrouter_request.seed, Some(69));
        assert_eq!(openrouter_request.top_p, Some(0.9));
        assert_eq!(openrouter_request.presence_penalty, Some(0.1));
        assert_eq!(openrouter_request.frequency_penalty, Some(0.2));
        assert!(openrouter_request.tools.is_none());

        // Test case: System message is converted to User message
        let request_with_system = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["Hello".to_string().into()],
            }],
            system: Some("This is the system message".to_string()),
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            seed: Some(69),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let openrouter_request_with_system =
            OpenRouterRequest::new("o1-mini", &request_with_system)
                .await
                .unwrap();

        // Check that the system message was converted to a user message
        assert_eq!(openrouter_request_with_system.messages.len(), 2);
        assert!(
            matches!(
                openrouter_request_with_system.messages[0],
                OpenRouterRequestMessage::User(ref msg) if msg.content == [OpenRouterContentBlock::Text { text: "This is the system message".into() }]
            ),
            "Unexpected messages: {:?}",
            openrouter_request_with_system.messages
        );

        assert_eq!(openrouter_request_with_system.model, "o1-mini");
        assert!(!openrouter_request_with_system.stream);
        assert_eq!(openrouter_request_with_system.response_format, None);
        assert_eq!(openrouter_request_with_system.temperature, Some(0.5));
        assert_eq!(
            openrouter_request_with_system.max_completion_tokens,
            Some(100)
        );
        assert_eq!(openrouter_request_with_system.seed, Some(69));
        assert!(openrouter_request_with_system.tools.is_none());
        assert_eq!(openrouter_request_with_system.top_p, Some(0.9));
        assert_eq!(openrouter_request_with_system.presence_penalty, Some(0.1));
        assert_eq!(openrouter_request_with_system.frequency_penalty, Some(0.2));
    }

    #[test]
    fn test_try_from_openrouter_response() {
        // Test case 1: Valid response with content
        let valid_response = OpenRouterResponse {
            choices: vec![OpenRouterResponseChoice {
                index: 0,
                message: OpenRouterResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
                finish_reason: OpenRouterFinishReason::Stop,
            }],
            usage: OpenRouterUsage {
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

        let request_body = OpenRouterRequest {
            messages: vec![],
            model: "gpt-4.1-mini",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(OpenRouterResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
            reasoning_effort: None,
            verbosity: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test_response".to_string();
        let result = ProviderInferenceResponse::try_from(OpenRouterResponseWithMetadata {
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
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(100)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.raw_response, raw_response);
        assert_eq!(inference_response.system, None);
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );
        // Test case 2: Valid response with tool calls
        let valid_response_with_tools = OpenRouterResponse {
            choices: vec![OpenRouterResponseChoice {
                index: 0,
                finish_reason: OpenRouterFinishReason::ToolCalls,
                message: OpenRouterResponseMessage {
                    content: None,
                    tool_calls: Some(vec![OpenRouterResponseToolCall {
                        id: "call1".to_string(),
                        r#type: OpenRouterToolType::Function,
                        function: OpenRouterResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
            }],
            usage: OpenRouterUsage {
                prompt_tokens: Some(15),
                completion_tokens: Some(25),
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
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            seed: Some(69),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let request_body = OpenRouterRequest {
            messages: vec![],
            model: "gpt-4.1-mini",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(OpenRouterResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
            reasoning_effort: None,
            verbosity: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(OpenRouterResponseWithMetadata {
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
            inference_response.finish_reason,
            Some(FinishReason::ToolCall)
        );
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
        let invalid_response_no_choices = OpenRouterResponse {
            choices: vec![],
            usage: OpenRouterUsage {
                prompt_tokens: Some(5),
                completion_tokens: Some(0),
            },
        };
        let request_body = OpenRouterRequest {
            messages: vec![],
            model: "gpt-4.1-mini",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(OpenRouterResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
            reasoning_effort: None,
            verbosity: None,
        };
        let result = ProviderInferenceResponse::try_from(OpenRouterResponseWithMetadata {
            response: invalid_response_no_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(120),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        let details = err.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));

        // Test case 4: Invalid response with multiple choices
        let invalid_response_multiple_choices = OpenRouterResponse {
            choices: vec![
                OpenRouterResponseChoice {
                    index: 0,
                    message: OpenRouterResponseMessage {
                        content: Some("Choice 1".to_string()),
                        tool_calls: None,
                    },
                    finish_reason: OpenRouterFinishReason::Stop,
                },
                OpenRouterResponseChoice {
                    index: 1,
                    finish_reason: OpenRouterFinishReason::Stop,
                    message: OpenRouterResponseMessage {
                        content: Some("Choice 2".to_string()),
                        tool_calls: None,
                    },
                },
            ],
            usage: OpenRouterUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(10),
            },
        };

        let request_body = OpenRouterRequest {
            messages: vec![],
            model: "gpt-4.1-mini",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(OpenRouterResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
            reasoning_effort: None,
            verbosity: None,
        };
        let result = ProviderInferenceResponse::try_from(OpenRouterResponseWithMetadata {
            response: invalid_response_multiple_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(130),
            },
            raw_request: serde_json::to_string(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        let details = err.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
    }

    #[test]
    fn test_prepare_openrouter_tools() {
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
            tool_config: Some(Cow::Borrowed(&MULTI_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let (tools, tool_choice, parallel_tool_calls) =
            prepare_openrouter_tools(&request_with_tools).unwrap();
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(tools[1].function.name, QUERY_TOOL.name());
        assert_eq!(tools[1].function.parameters, QUERY_TOOL.parameters());
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            OpenRouterToolChoice::String(OpenRouterToolChoiceString::Required)
        );
        let parallel_tool_calls = parallel_tool_calls.unwrap();
        assert!(parallel_tool_calls);
        let tool_config = ToolCallConfig {
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: Some(true),
            ..Default::default()
        };

        // Test no tools but a tool choice and make sure tool choice output is None
        let request_without_tools = ModelInferenceRequest {
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
            tool_config: Some(Cow::Borrowed(&tool_config)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let (tools, tool_choice, parallel_tool_calls) =
            prepare_openrouter_tools(&request_without_tools).unwrap();
        assert!(tools.is_none());
        assert!(tool_choice.is_none());
        assert!(parallel_tool_calls.is_none());
    }

    #[tokio::test]
    async fn test_tensorzero_to_openrouter_messages() {
        let content_blocks = vec!["Hello".to_string().into()];
        let openrouter_messages = tensorzero_to_openrouter_user_messages(&content_blocks, true)
            .await
            .unwrap();
        assert_eq!(openrouter_messages.len(), 1);
        match &openrouter_messages[0] {
            OpenRouterRequestMessage::User(content) => {
                assert_eq!(
                    content.content,
                    &[OpenRouterContentBlock::Text {
                        text: "Hello".into()
                    }]
                );
            }
            _ => panic!("Expected a user message"),
        }

        // Message with multiple blocks
        let content_blocks = vec![
            "Hello".to_string().into(),
            "How are you?".to_string().into(),
        ];
        let openrouter_messages = tensorzero_to_openrouter_user_messages(&content_blocks, true)
            .await
            .unwrap();
        assert_eq!(openrouter_messages.len(), 1);
        match &openrouter_messages[0] {
            OpenRouterRequestMessage::User(content) => {
                assert_eq!(
                    content.content,
                    vec![
                        OpenRouterContentBlock::Text {
                            text: "Hello".into()
                        },
                        OpenRouterContentBlock::Text {
                            text: "How are you?".into()
                        }
                    ]
                );
            }
            _ => panic!("Expected a user message"),
        }

        // User message with one string and one tool call block
        // Since user messages in OpenRouter land can't contain tool calls (nor should they honestly),
        // We split the tool call out into a separate assistant message
        let tool_block = ContentBlock::ToolCall(ToolCall {
            id: "call1".to_string(),
            name: "test_function".to_string(),
            arguments: "{}".to_string(),
        });
        let content_blocks = vec!["Hello".to_string().into(), tool_block];
        let openrouter_messages =
            tensorzero_to_openrouter_assistant_messages(&content_blocks, true)
                .await
                .unwrap();
        assert_eq!(openrouter_messages.len(), 1);
        match &openrouter_messages[0] {
            OpenRouterRequestMessage::Assistant(content) => {
                assert_eq!(
                    content.content,
                    Some(vec![OpenRouterContentBlock::Text {
                        text: "Hello".into()
                    }])
                );
                let tool_calls = content.tool_calls.as_ref().unwrap();
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call1");
                assert_eq!(tool_calls[0].function.name, "test_function");
                assert_eq!(tool_calls[0].function.arguments, "{}");
            }
            _ => panic!("Expected an assistant message"),
        }
    }

    #[test]
    fn test_openrouter_to_tensorzero_chunk() {
        let chunk = OpenRouterChatChunk {
            choices: vec![OpenRouterChatChunkChoice {
                delta: OpenRouterDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some(OpenRouterFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let message = openrouter_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
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
        let chunk = OpenRouterChatChunk {
            choices: vec![OpenRouterChatChunkChoice {
                finish_reason: Some(OpenRouterFinishReason::ToolCalls),
                delta: OpenRouterDelta {
                    content: None,
                    tool_calls: Some(vec![OpenRouterToolCallChunk {
                        index: 0,
                        id: None,
                        function: OpenRouterFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message = openrouter_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
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
        let chunk = OpenRouterChatChunk {
            choices: vec![OpenRouterChatChunkChoice {
                finish_reason: None,
                delta: OpenRouterDelta {
                    content: None,
                    tool_calls: Some(vec![OpenRouterToolCallChunk {
                        index: 1,
                        id: None,
                        function: OpenRouterFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let error = openrouter_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
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
        let chunk = OpenRouterChatChunk {
            choices: vec![OpenRouterChatChunkChoice {
                finish_reason: Some(OpenRouterFinishReason::Stop),
                delta: OpenRouterDelta {
                    content: None,
                    tool_calls: Some(vec![OpenRouterToolCallChunk {
                        index: 1,
                        id: Some("id2".to_string()),
                        function: OpenRouterFunctionCallChunk {
                            name: Some("name2".to_string()),
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message = openrouter_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
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
        assert_eq!(message.finish_reason, Some(FinishReason::Stop));
        // Check that the lists were updated
        assert_eq!(tool_call_ids, vec!["id1".to_string(), "id2".to_string()]);

        // Check a chunk with no choices and only usage
        // Test a correct new tool chunk
        let chunk = OpenRouterChatChunk {
            choices: vec![],
            usage: Some(OpenRouterUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
            }),
        };
        let message = openrouter_to_tensorzero_chunk(
            "my_raw_chunk".to_string(),
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
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
    }

    #[test]
    fn test_new_openrouter_response_format() {
        // Test JSON mode On
        let json_mode = ModelInferenceRequestJsonMode::On;
        let output_schema = None;
        let format = OpenRouterResponseFormat::new(json_mode, output_schema, "gpt-4o");
        assert_eq!(format, Some(OpenRouterResponseFormat::JsonObject));

        // Test JSON mode Off
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let format = OpenRouterResponseFormat::new(json_mode, output_schema, "gpt-4o");
        assert_eq!(format, None);

        // Test JSON mode Strict with no schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let format = OpenRouterResponseFormat::new(json_mode, output_schema, "gpt-4o");
        assert_eq!(format, Some(OpenRouterResponseFormat::JsonObject));

        // Test JSON mode Strict with schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"type": "string"}
            }
        });
        let output_schema = Some(&schema);
        let format = OpenRouterResponseFormat::new(json_mode, output_schema, "gpt-4o");
        match format {
            Some(OpenRouterResponseFormat::JsonSchema { json_schema }) => {
                assert_eq!(json_schema["schema"], schema);
                assert_eq!(json_schema["name"], "response");
                assert_eq!(json_schema["strict"], true);
            }
            _ => panic!("Expected JsonSchema format"),
        }

        // Test JSON mode Strict with schema but gpt-3.5-turbo (does not support strict mode)
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"type": "string"}
            }
        });
        let output_schema = Some(&schema);
        let format = OpenRouterResponseFormat::new(json_mode, output_schema, "gpt-3.5-turbo");
        assert_eq!(format, Some(OpenRouterResponseFormat::JsonObject));
    }

    #[test]
    fn test_openrouter_api_base() {
        assert_eq!(
            OPENROUTER_DEFAULT_BASE_URL.as_str(),
            "https://openrouter.ai/api/v1"
        );
    }

    #[test]
    fn test_tensorzero_to_openrouter_system_message() {
        // Test Case 1: system is None, json_mode is Off
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages: Vec<OpenRouterRequestMessage> = vec![];
        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert_eq!(result, None);

        // Test Case 2: system is Some, json_mode is On, messages contain "json"
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            OpenRouterRequestMessage::User(OpenRouterUserRequestMessage {
                content: vec![OpenRouterContentBlock::Text {
                    text: "Please respond in JSON format.".into(),
                }],
            }),
            OpenRouterRequestMessage::Assistant(OpenRouterAssistantRequestMessage {
                content: Some(vec![OpenRouterContentBlock::Text {
                    text: "Sure, here is the data.".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(OpenRouterRequestMessage::System(
            OpenRouterSystemRequestMessage {
                content: Cow::Borrowed("System instructions"),
            },
        ));
        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 3: system is Some, json_mode is On, messages do not contain "json"
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            OpenRouterRequestMessage::User(OpenRouterUserRequestMessage {
                content: vec![OpenRouterContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            OpenRouterRequestMessage::Assistant(OpenRouterAssistantRequestMessage {
                content: Some(vec![OpenRouterContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected_content = "Respond using JSON.\n\nSystem instructions".to_string();
        let expected = Some(OpenRouterRequestMessage::System(
            OpenRouterSystemRequestMessage {
                content: Cow::Owned(expected_content),
            },
        ));
        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 4: system is Some, json_mode is Off
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages = vec![
            OpenRouterRequestMessage::User(OpenRouterUserRequestMessage {
                content: vec![OpenRouterContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            OpenRouterRequestMessage::Assistant(OpenRouterAssistantRequestMessage {
                content: Some(vec![OpenRouterContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(OpenRouterRequestMessage::System(
            OpenRouterSystemRequestMessage {
                content: Cow::Borrowed("System instructions"),
            },
        ));
        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 5: system is Some, json_mode is Strict
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let messages = vec![
            OpenRouterRequestMessage::User(OpenRouterUserRequestMessage {
                content: vec![OpenRouterContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            OpenRouterRequestMessage::Assistant(OpenRouterAssistantRequestMessage {
                content: Some(vec![OpenRouterContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(OpenRouterRequestMessage::System(
            OpenRouterSystemRequestMessage {
                content: Cow::Borrowed("System instructions"),
            },
        ));
        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 6: system contains "json", json_mode is On
        let system = Some("Respond using JSON.\n\nSystem instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![OpenRouterRequestMessage::User(
            OpenRouterUserRequestMessage {
                content: vec![OpenRouterContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            },
        )];
        let expected = Some(OpenRouterRequestMessage::System(
            OpenRouterSystemRequestMessage {
                content: Cow::Borrowed("Respond using JSON.\n\nSystem instructions"),
            },
        ));
        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 7: system is None, json_mode is On
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            OpenRouterRequestMessage::User(OpenRouterUserRequestMessage {
                content: vec![OpenRouterContentBlock::Text {
                    text: "Tell me a joke.".into(),
                }],
            }),
            OpenRouterRequestMessage::Assistant(OpenRouterAssistantRequestMessage {
                content: Some(vec![OpenRouterContentBlock::Text {
                    text: "Sure, here's one for you.".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(OpenRouterRequestMessage::System(
            OpenRouterSystemRequestMessage {
                content: Cow::Owned("Respond using JSON.".to_string()),
            },
        ));
        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 8: system is None, json_mode is Strict
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let messages = vec![
            OpenRouterRequestMessage::User(OpenRouterUserRequestMessage {
                content: vec![OpenRouterContentBlock::Text {
                    text: "Provide a summary of the news.".into(),
                }],
            }),
            OpenRouterRequestMessage::Assistant(OpenRouterAssistantRequestMessage {
                content: Some(vec![OpenRouterContentBlock::Text {
                    text: "Here's the summary.".into(),
                }]),
                tool_calls: None,
            }),
        ];

        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert!(result.is_none());

        // Test Case 9: system is None, json_mode is On, with empty messages
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages: Vec<OpenRouterRequestMessage> = vec![];
        let expected = Some(OpenRouterRequestMessage::System(
            OpenRouterSystemRequestMessage {
                content: Cow::Owned("Respond using JSON.".to_string()),
            },
        ));
        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 10: system is None, json_mode is Off, with messages containing "json"
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages = vec![OpenRouterRequestMessage::User(
            OpenRouterUserRequestMessage {
                content: vec![OpenRouterContentBlock::Text {
                    text: "Please include JSON in your response.".into(),
                }],
            },
        )];
        let expected = None;
        let result = tensorzero_to_openrouter_system_message(system, json_mode, &messages);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_create_file_url() {
        use url::Url;

        // Test Case 1: Base URL without trailing slash
        let base_url = Url::parse("https://openrouter.ai/api/v1").unwrap();
        let file_id = Some("file123");
        let result = get_file_url(&base_url, file_id).unwrap();
        assert_eq!(
            result.as_str(),
            "https://openrouter.ai/api/v1/files/file123/content"
        );

        // Test Case 2: Base URL with trailing slash
        let base_url = Url::parse("https://openrouter.ai/api/v1/").unwrap();
        let file_id = Some("file456");
        let result = get_file_url(&base_url, file_id).unwrap();
        assert_eq!(
            result.as_str(),
            "https://openrouter.ai/api/v1/files/file456/content"
        );

        // Test Case 3: Base URL with custom domain
        let base_url = Url::parse("https://custom-openrouter.example.com").unwrap();
        let file_id = Some("file789");
        let result = get_file_url(&base_url, file_id).unwrap();
        assert_eq!(
            result.as_str(),
            "https://custom-openrouter.example.com/files/file789/content"
        );

        // Test Case 4: Base URL without trailing slash, no file ID
        let base_url = Url::parse("https://openrouter.ai/api/v1").unwrap();
        let result = get_file_url(&base_url, None).unwrap();
        assert_eq!(result.as_str(), "https://openrouter.ai/api/v1/files");

        // Test Case 5: Base URL with trailing slash, no file ID
        let base_url = Url::parse("https://openrouter.ai/api/v1/").unwrap();
        let result = get_file_url(&base_url, None).unwrap();
        assert_eq!(result.as_str(), "https://openrouter.ai/api/v1/files");

        // Test Case 6: Custom domain base URL, no file ID
        let base_url = Url::parse("https://custom-openrouter.example.com").unwrap();
        let result = get_file_url(&base_url, None).unwrap();
        assert_eq!(
            result.as_str(),
            "https://custom-openrouter.example.com/files"
        );
    }

    #[test]
    fn test_try_from_openrouter_credentials() {
        // Test Static credentials
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = OpenRouterCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenRouterCredentials::Static(_)));

        // Test Dynamic credentials
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = OpenRouterCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenRouterCredentials::Dynamic(_)));

        // Test None credentials
        let generic = Credential::None;
        let creds = OpenRouterCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenRouterCredentials::None));

        // Test Missing credentials
        let generic = Credential::Missing;
        let creds = OpenRouterCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenRouterCredentials::None));

        // Test invalid credential type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = OpenRouterCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_serialize_user_messages() {
        // Test that a single message is serialized as 'content: string'
        let message = OpenRouterUserRequestMessage {
            content: vec![OpenRouterContentBlock::Text {
                text: "My single message".into(),
            }],
        };
        let serialized = serde_json::to_string(&message).unwrap();
        assert_eq!(serialized, r#"{"content":"My single message"}"#);

        // Test that a multiple messages are serialized as an array of content blocks
        let message = OpenRouterUserRequestMessage {
            content: vec![
                OpenRouterContentBlock::Text {
                    text: "My first message".into(),
                },
                OpenRouterContentBlock::Text {
                    text: "My second message".into(),
                },
            ],
        };
        let serialized = serde_json::to_string(&message).unwrap();
        assert_eq!(
            serialized,
            r#"{"content":[{"type":"text","text":"My first message"},{"type":"text","text":"My second message"}]}"#
        );
    }

    #[test]
    fn test_openrouter_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = OpenRouterRequest {
            messages: vec![],
            model: "test-model",
            temperature: None,
            max_completion_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            stream_options: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stop: None,
            reasoning_effort: None,
            verbosity: None,
        };

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort is applied correctly
        assert_eq!(request.reasoning_effort, Some("high".to_string()));

        // Test that thinking_budget_tokens warns with tip about reasoning_effort
        assert!(logs_contain(
            "OpenRouter does not support the inference parameter `thinking_budget_tokens`, so it will be ignored. Tip: You might want to use `reasoning_effort` for this provider."
        ));

        // Test that verbosity is applied correctly
        assert_eq!(request.verbosity, Some("low".to_string()));
    }
}
