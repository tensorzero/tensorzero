use async_trait::async_trait;
use futures::future::try_join_all;
use futures::{Stream, StreamExt, TryStreamExt};
use lazy_static::lazy_static;
use reqwest::multipart::{Form, Part};
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use responses::stream_openai_responses;
use secrecy::{ExposeSecret, SecretString};
use serde::de::IntoDeserializer;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::io::Write;
use std::pin::Pin;
use std::time::Duration;
use tokio::time::Instant;
use tracing::instrument;
use url::Url;
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
use crate::http::TensorzeroHttpClient;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::batch::{
    ProviderBatchInferenceOutput, ProviderBatchInferenceResponse,
};
use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2, ServiceTier,
};
use crate::inference::types::extra_body::FullExtraBodyConfig;
use crate::inference::types::file::{mime_type_to_audio_format, mime_type_to_ext, Detail};
use crate::inference::types::resolved_input::{FileUrl, LazyFile};
use crate::inference::types::ObjectStorageFile;
use crate::inference::types::{
    batch::{BatchStatus, StartBatchProviderInferenceResponse},
    ContentBlock, ContentBlockChunk, ContentBlockOutput, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, RequestMessage, Role, Text,
    TextChunk, Unknown, Usage,
};
use crate::inference::types::{
    FinishReason, ProviderInferenceResponseArgs, ProviderInferenceResponseStreamInner, ThoughtChunk,
};
use crate::inference::InferenceProvider;
use crate::model::{Credential, ModelProvider};
use crate::providers::openai::responses::{
    get_responses_url, OpenAIResponsesInput, OpenAIResponsesInputInner,
    OpenAIResponsesInputMessage, OpenAIResponsesInputMessageContent, OpenAIResponsesRequest,
    OpenAIResponsesResponse,
};
use crate::tool::{
    FunctionTool, FunctionToolConfig, OpenAICustomTool, ToolCall, ToolCallChunk, ToolCallConfig,
    ToolChoice, ToolConfigRef,
};

use crate::providers::helpers::{
    convert_stream_error, inject_extra_request_data_and_send,
    inject_extra_request_data_and_send_eventsource,
};

use super::helpers::{parse_jsonl_batch_file, JsonlBatchFileInfo};
use crate::inference::TensorZeroEventError;
use crate::inference::WrappedProvider;

pub mod grader;
mod responses;

lazy_static! {
    pub static ref OPENAI_DEFAULT_BASE_URL: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("https://api.openai.com/v1/").expect("Failed to parse OPENAI_DEFAULT_BASE_URL")
    };
}

const PROVIDER_NAME: &str = "OpenAI";
pub const PROVIDER_TYPE: &str = "openai";

type PreparedOpenAIToolsResult<'a> = (
    Option<Vec<OpenAITool<'a>>>,
    Option<OpenAIToolChoice<'a>>,
    Option<bool>,
);

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum OpenAIAPIType {
    #[default]
    ChatCompletions,
    Responses,
}

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenAIProvider {
    model_name: String,
    api_base: Option<Url>,
    #[serde(skip)]
    credentials: OpenAICredentials,
    include_encrypted_reasoning: bool,
    api_type: OpenAIAPIType,
    provider_tools: Vec<Value>,
}

impl OpenAIProvider {
    pub fn new(
        model_name: String,
        api_base: Option<Url>,
        credentials: OpenAICredentials,
        api_type: OpenAIAPIType,
        include_encrypted_reasoning: bool,
        provider_tools: Vec<Value>,
    ) -> Result<Self, Error> {
        if !matches!(api_type, OpenAIAPIType::Responses) && include_encrypted_reasoning {
            return Err(Error::new(ErrorDetails::Config {
                message:
                    "include_encrypted_reasoning is only supported when api_type = \"responses\""
                        .to_string(),
            }));
        }
        // Check if the api_base has the `/chat/completions` suffix and warn if it does
        if let Some(api_base) = &api_base {
            check_api_base_suffix(api_base);
        }

        if !provider_tools.is_empty() && !matches!(api_type, OpenAIAPIType::Responses) {
            return Err(ErrorDetails::Config{message: "`provider_tools` are provided for an OpenAI provider but Responses API is not enabled. These will be ignored.".to_string()}.into());
        }

        Ok(OpenAIProvider {
            model_name,
            api_base,
            credentials,

            include_encrypted_reasoning,
            api_type,

            provider_tools,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Checks if the provided OpenAI API base URL has `/chat/completions` suffix and warns if it does.
///
/// This check exists because a common mistake when configuring OpenAI API endpoints is to include
/// `/chat/completions` in the base URL. The gateway automatically appends this path when making requests,
/// so including it in the base URL results in an invalid endpoint like:
/// `http://localhost:1234/v1/chat/completions/chat/completions`
///
/// For example:
/// - Correct: `http://localhost:1234/v1` or `http://localhost:1234/openai/v1/`
/// - Incorrect: `http://localhost:1234/v1/chat/completions`
pub fn check_api_base_suffix(api_base: &Url) {
    let path = api_base.path();
    if path.ends_with("/chat/completions") || path.ends_with("/chat/completions/") {
        tracing::warn!(
            "The gateway automatically appends `/chat/completions` to the `api_base`. You provided `{api_base}` which is likely incorrect. Please remove the `/chat/completions` suffix from `api_base`.",
        );
    }
}

#[derive(Clone, Debug)]
pub enum OpenAICredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<OpenAICredentials>,
        fallback: Box<OpenAICredentials>,
    },
}

impl TryFrom<Credential> for OpenAICredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(OpenAICredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(OpenAICredentials::Dynamic(key_name)),
            Credential::None => Ok(OpenAICredentials::None),
            Credential::Missing => Ok(OpenAICredentials::None),
            Credential::WithFallback { default, fallback } => Ok(OpenAICredentials::WithFallback {
                default: Box::new((*default).try_into()?),
                fallback: Box::new((*fallback).try_into()?),
            }),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for OpenAI provider".to_string(),
            })),
        }
    }
}

impl OpenAICredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, DelayedError> {
        match self {
            OpenAICredentials::Static(api_key) => Ok(Some(api_key)),
            OpenAICredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                }))
                .transpose()
            }
            OpenAICredentials::WithFallback { default, fallback } => {
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
            OpenAICredentials::None => Ok(None),
        }
    }
}

#[async_trait]
impl WrappedProvider for OpenAIProvider {
    fn thought_block_provider_type_suffix(&self) -> Cow<'static, str> {
        Cow::Borrowed("openai")
    }

    async fn make_body<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
    ) -> Result<serde_json::Value, Error> {
        match self.api_type {
            OpenAIAPIType::Responses => Ok(serde_json::to_value(
                OpenAIResponsesRequest::new(
                    &self.model_name,
                    request,
                    self.include_encrypted_reasoning,
                    &self.provider_tools,
                    model_name,
                    provider_name,
                )
                .await?,
            )
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing OpenAI request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?),
            OpenAIAPIType::ChatCompletions => Ok(serde_json::to_value(
                OpenAIRequest::new(&self.model_name, request).await?,
            )
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing OpenAI request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?),
        }
    }

    fn parse_response(
        &self,
        request: &ModelInferenceRequest,
        raw_request: String,
        raw_response: String,
        latency: Latency,
        model_name: &str,
        provider_name: &str,
    ) -> Result<ProviderInferenceResponse, Error> {
        match self.api_type {
            OpenAIAPIType::Responses => {
                // TODO - include 'responses' somewhere in the error message
                let response: OpenAIResponsesResponse = serde_json::from_str(&raw_response)
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

                response.into_provider_response(
                    latency,
                    raw_request,
                    raw_response.clone(),
                    request,
                    model_name,
                    provider_name,
                )
            }
            OpenAIAPIType::ChatCompletions => {
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

                OpenAIResponseWithMetadata {
                    response,
                    raw_response,
                    latency,
                    raw_request,
                    generic_request: request,
                }
                .try_into()
            }
        }
    }

    fn stream_events(
        &self,
        event_source: Pin<
            Box<dyn Stream<Item = Result<Event, TensorZeroEventError>> + Send + 'static>,
        >,
        start_time: Instant,
        raw_request: &str,
    ) -> ProviderInferenceResponseStreamInner {
        stream_openai(
            PROVIDER_TYPE.to_string(),
            event_source,
            start_time,
            raw_request,
        )
    }
}

impl InferenceProvider for OpenAIProvider {
    async fn infer<'a>(
        &'a self,
        request: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_url = match self.api_type {
            OpenAIAPIType::Responses => {
                get_responses_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?
            }
            OpenAIAPIType::ChatCompletions => {
                get_chat_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?
            }
        };
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let request_body = self.make_body(request).await?;
        let mut request_builder = http_client.post(request_url);

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

            match self.api_type {
                OpenAIAPIType::Responses => {
                    let response: OpenAIResponsesResponse = serde_json::from_str(&raw_response)
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
                    response.into_provider_response(
                        latency,
                        raw_request,
                        raw_response.clone(),
                        request.request,
                        request.model_name,
                        request.provider_name,
                    )
                }
                OpenAIAPIType::ChatCompletions => {
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
                    Ok(OpenAIResponseWithMetadata {
                        response,
                        raw_response,
                        latency,
                        raw_request: raw_request.clone(),
                        generic_request: request.request,
                    }
                    .try_into()?)
                }
            }
        } else {
            Err(handle_openai_error(
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
            provider_name,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();

        match self.api_type {
            OpenAIAPIType::Responses => {
                let request_url =
                    get_responses_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;

                // TODO - support encrypted reasoning in streaming
                let request_body = serde_json::to_value(
                    OpenAIResponsesRequest::new(
                        &self.model_name,
                        request,
                        false,
                        &self.provider_tools,
                        model_name,
                        provider_name,
                    )
                    .await?,
                )
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing OpenAI Responses request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;

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

                let stream = stream_openai_responses(
                    PROVIDER_TYPE.to_string(),
                    event_source.map_err(TensorZeroEventError::EventSource),
                    start_time,
                    model_provider.discard_unknown_chunks,
                    model_name,
                    provider_name,
                    &raw_request,
                )
                .peekable();
                Ok((stream, raw_request))
            }
            OpenAIAPIType::ChatCompletions => {
                // Use Chat Completions API for streaming
                let request_url =
                    get_chat_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;

                let request_body =
                    serde_json::to_value(OpenAIRequest::new(&self.model_name, request).await?)
                        .map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: format!(
                                    "Error serializing OpenAI request: {}",
                                    DisplayOrDebugGateway::new(e)
                                ),
                            })
                        })?;

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

                let stream = stream_openai(
                    PROVIDER_TYPE.to_string(),
                    event_source.map_err(TensorZeroEventError::EventSource),
                    start_time,
                    &raw_request,
                )
                .peekable();
                Ok((stream, raw_request))
            }
        }
    }

    /// 1. Upload the requests to OpenAI as a File
    /// 2. Start the batch inference
    ///    We do them in sequence here.
    async fn start_batch_inference<'a>(
        &'a self,
        requests: &'a [ModelInferenceRequest<'_>],
        client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let mut batch_requests = Vec::with_capacity(requests.len());
        for request in requests {
            batch_requests.push(
                OpenAIBatchFileInput::new(request.inference_id, &self.model_name, request).await?,
            );
        }
        let raw_requests: Result<Vec<String>, serde_json::Error> = batch_requests
            .iter()
            .map(|b| serde_json::to_string(&b.body))
            .collect();
        let raw_requests = raw_requests.map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: e.to_string(),
            })
        })?;
        let file_id = upload_openai_file(
            &batch_requests,
            client,
            api_key,
            self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL),
            "batch".to_string(),
        )
        .await?;
        let batch_request = OpenAIBatchRequest::new(&file_id);
        let raw_request = serde_json::to_string(&batch_request).map_err(|_| Error::new(ErrorDetails::Serialization { message: "Error serializing OpenAI batch request. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string() }))?;
        let request_url =
            get_batch_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;
        let mut request_builder = client.post(request_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        // Now let's actually start the batch inference
        let res = request_builder
            .json(&batch_request)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!(
                        "Error sending request to OpenAI: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&batch_request).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        let text = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error retrieving batch response: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                raw_request: Some(serde_json::to_string(&batch_request).unwrap_or_default()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let response: OpenAIBatchResponse = serde_json::from_str(&text).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing JSON response: {e}, text: {text}"),
                raw_request: Some(serde_json::to_string(&batch_request).unwrap_or_default()),
                raw_response: Some(text.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let batch_params = OpenAIBatchParams {
            file_id: Cow::Owned(file_id),
            batch_id: Cow::Owned(response.id),
        };
        let batch_params = serde_json::to_value(batch_params).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing OpenAI batch params: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let errors = match response.errors {
            Some(errors) => errors
                .data
                .into_iter()
                .map(|error| {
                    serde_json::to_value(&error).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Error serializing batch error: {}",
                                DisplayOrDebugGateway::new(e)
                            ),
                        })
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => vec![],
        };
        Ok(StartBatchProviderInferenceResponse {
            batch_id: Uuid::now_v7(),
            batch_params,
            raw_requests,
            raw_request,
            raw_response: text,
            status: BatchStatus::Pending,
            errors,
        })
    }

    #[instrument(skip_all, fields(batch_request = ?batch_request))]
    async fn poll_batch_inference<'a>(
        &'a self,
        batch_request: &'a BatchRequestRow<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        let batch_params = OpenAIBatchParams::from_ref(&batch_request.batch_params)?;
        let mut request_url =
            get_batch_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;
        request_url
            .path_segments_mut()
            .map_err(|()| {
                Error::new(ErrorDetails::Inference {
                    message: "Failed to get mutable path segments".to_string(),
                })
            })?
            .push(&batch_params.batch_id);
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let raw_request = request_url.to_string();
        let mut request_builder = http_client.get(request_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let res = request_builder.send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error sending request to OpenAI: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&batch_request).unwrap_or_default()),
                raw_response: None,
            })
        })?;
        let text = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error parsing JSON response: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                raw_request: Some(serde_json::to_string(&batch_request).unwrap_or_default()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let response: OpenAIBatchResponse = serde_json::from_str(&text).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing JSON response: {e}."),
                raw_request: Some(serde_json::to_string(&batch_request).unwrap_or_default()),
                raw_response: Some(text.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let status: BatchStatus = response.status.into();
        let raw_response = text;
        match status {
            BatchStatus::Pending => Ok(PollBatchInferenceResponse::Pending {
                raw_request,
                raw_response,
            }),
            BatchStatus::Completed => {
                let output_file_id = response.output_file_id.as_ref().ok_or_else(|| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: "Output file ID is missing".to_string(),
                        raw_request: Some(
                            serde_json::to_string(&batch_request).unwrap_or_default(),
                        ),
                        raw_response: Some(raw_response.clone()),
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?;
                let response = self
                    .collect_finished_batch(
                        output_file_id,
                        http_client,
                        dynamic_api_keys,
                        raw_request,
                        raw_response,
                    )
                    .await?;
                Ok(PollBatchInferenceResponse::Completed(response))
            }
            BatchStatus::Failed => Ok(PollBatchInferenceResponse::Failed {
                raw_request,
                raw_response,
            }),
        }
    }
}

fn apply_inference_params(
    request: &mut OpenAIRequest,
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
        request.service_tier = service_tier.clone();
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

impl EmbeddingProvider for OpenAIProvider {
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
        let request_body = OpenAIEmbeddingRequest::new(
            &self.model_name,
            &request.input,
            request.dimensions,
            request.encoding_format,
        );
        let request_url =
            get_embedding_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;
        let start_time = Instant::now();
        let mut request_builder = client.post(request_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }

        let request_body_value = serde_json::to_value(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing OpenAI embedding request: {}",
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

            let response: OpenAIEmbeddingResponse =
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

            Ok(OpenAIEmbeddingResponseWithMetadata {
                response,
                latency,
                request: request_body,
                raw_response,
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

pub fn stream_openai(
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
                        let data: Result<OpenAIChatChunk, Error> =
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
                            openai_to_tensorzero_chunk(message.data, d, latency, &mut tool_call_ids)
                        });
                        yield stream_message;
                    }
                },
            }
        }
    })
}

impl OpenAIProvider {
    // Once a batch has been completed we need to retrieve the results from OpenAI using the files API
    #[instrument(skip_all, fields(file_id = file_id))]
    async fn collect_finished_batch(
        &self,
        file_id: &str,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        raw_request: String,
        raw_response: String,
    ) -> Result<ProviderBatchInferenceResponse, Error> {
        let file_url = get_file_url(
            self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL),
            Some(file_id),
        )?;
        let api_key = self
            .credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;
        let mut request_builder = client.get(file_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let res = request_builder.send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error downloading batch results from OpenAI for file {file_id}: {e}"
                ),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        if res.status() != StatusCode::OK {
            return Err(handle_openai_error(
                &raw_request,
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!(
                            "Error parsing error response for file {file_id}: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                        raw_request: None,
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?,
                PROVIDER_TYPE,
            ));
        }

        parse_jsonl_batch_file::<OpenAIBatchFileRow, _>(
            res.bytes().await,
            JsonlBatchFileInfo {
                file_id: file_id.to_string(),
                raw_request,
                raw_response,
                provider_type: PROVIDER_TYPE.to_string(),
            },
            TryInto::try_into,
        )
        .await
    }
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

fn get_batch_url(base_url: &Url) -> Result<Url, Error> {
    let mut url = base_url.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    url.join("batches").map_err(|e| {
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

pub(super) fn handle_openai_error(
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAISystemRequestMessage<'a> {
    pub content: Cow<'a, str>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAIUserRequestMessage<'a> {
    #[serde(serialize_with = "serialize_text_content_vec")]
    pub content: Vec<OpenAIContentBlock<'a>>,
}

pub type OpenAIFileID = String;

pub async fn upload_openai_file<T>(
    items: &[T],
    client: &TensorzeroHttpClient,
    api_key: Option<&SecretString>,
    api_base: &Url,
    purpose: String,
) -> Result<OpenAIFileID, Error>
where
    T: Serialize,
{
    let mut jsonl_data = Vec::new();
    for item in items {
        serde_json::to_writer(&mut jsonl_data, item).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error writing to JSONL: {}", DisplayOrDebugGateway::new(e)),
            })
        })?;
        jsonl_data.write_all(b"\n").map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error writing to JSONL: {}", DisplayOrDebugGateway::new(e)),
            })
        })?;
    }
    let form = Form::new().text("purpose", purpose).part(
        "file",
        Part::bytes(jsonl_data)
            .file_name("data.jsonl")
            .mime_str("application/json")
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error setting MIME type: {}", DisplayOrDebugGateway::new(e)),
                })
            })?,
    );
    let request_url = get_file_url(api_base, None)?;
    let mut request_builder = client.post(request_url);
    if let Some(api_key) = api_key {
        request_builder = request_builder.bearer_auth(api_key.expose_secret());
    }
    let res = request_builder.multipart(form).send().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            status_code: e.status(),
            message: format!(
                "Error sending request to OpenAI: {}",
                DisplayOrDebugGateway::new(e)
            ),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;
    let text = res.text().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!(
                "Error retrieving text response: {}",
                DisplayOrDebugGateway::new(e)
            ),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;
    let response: OpenAIFileResponse = serde_json::from_str(&text).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error parsing JSON response: {e}, text: {text}"),
            raw_request: None,
            raw_response: Some(text.clone()),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;
    Ok(response.id)
}

fn serialize_text_content_vec<S>(
    content: &Vec<OpenAIContentBlock<'_>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // If we have a single text block, serialize it as a string
    // to stay compatible with older providers which may not support content blocks
    if let [OpenAIContentBlock::Text { text }] = &content.as_slice() {
        text.serialize(serializer)
    } else {
        content.serialize(serializer)
    }
}

fn serialize_optional_text_content_vec<S>(
    content: &Option<Vec<OpenAIContentBlock<'_>>>,
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

#[derive(Clone, Deserialize, Debug, PartialEq, Serialize)]
pub struct OpenAIFile<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    file_data: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    filename: Option<Cow<'a, str>>,
}

#[derive(Clone, Deserialize, Debug, PartialEq, Serialize)]
pub struct OpenAIInputAudio<'a> {
    data: Cow<'a, str>,
    format: Cow<'a, str>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OpenAIContentBlock<'a> {
    Text { text: Cow<'a, str> },
    ImageUrl { image_url: OpenAIImageUrl },
    File { file: OpenAIFile<'a> },
    InputAudio { input_audio: OpenAIInputAudio<'a> },
    Unknown { data: Cow<'a, Value> },
}

impl Serialize for OpenAIContentBlock<'_> {
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
                image_url: &'a OpenAIImageUrl,
            },
            File {
                file: &'a OpenAIFile<'a>,
            },
            InputAudio {
                input_audio: &'a OpenAIInputAudio<'a>,
            },
        }
        match self {
            OpenAIContentBlock::Text { text } => Helper::Text { text }.serialize(serializer),
            OpenAIContentBlock::ImageUrl { image_url } => {
                Helper::ImageUrl { image_url }.serialize(serializer)
            }
            OpenAIContentBlock::File { file } => Helper::File { file }.serialize(serializer),
            OpenAIContentBlock::InputAudio { input_audio } => {
                Helper::InputAudio { input_audio }.serialize(serializer)
            }
            OpenAIContentBlock::Unknown { data } => data.serialize(serializer),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct OpenAIImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Detail>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAIRequestFunctionCall<'a> {
    pub name: Cow<'a, str>,
    pub arguments: Cow<'a, str>,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct OpenAIRequestToolCall<'a> {
    pub id: Cow<'a, str>,
    pub r#type: OpenAIToolType,
    pub function: OpenAIRequestFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for OpenAIRequestToolCall<'a> {
    fn from(tool_call: &'a ToolCall) -> Self {
        OpenAIRequestToolCall {
            id: Cow::Borrowed(&tool_call.id),
            r#type: OpenAIToolType::Function,
            function: OpenAIRequestFunctionCall {
                name: Cow::Borrowed(&tool_call.name),
                arguments: Cow::Borrowed(&tool_call.arguments),
            },
        }
    }
}

impl From<ToolCall> for OpenAIRequestToolCall<'static> {
    fn from(tool_call: ToolCall) -> Self {
        OpenAIRequestToolCall {
            id: Cow::Owned(tool_call.id),
            r#type: OpenAIToolType::Function,
            function: OpenAIRequestFunctionCall {
                name: Cow::Owned(tool_call.name),
                arguments: Cow::Owned(tool_call.arguments),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAIAssistantRequestMessage<'a> {
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_text_content_vec"
    )]
    pub content: Option<Vec<OpenAIContentBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIRequestToolCall<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAIToolRequestMessage<'a> {
    pub content: &'a str,
    pub tool_call_id: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum OpenAIRequestMessage<'a> {
    Developer(OpenAISystemRequestMessage<'a>),
    System(OpenAISystemRequestMessage<'a>),
    User(OpenAIUserRequestMessage<'a>),
    Assistant(OpenAIAssistantRequestMessage<'a>),
    Tool(OpenAIToolRequestMessage<'a>),
}

impl OpenAIRequestMessage<'_> {
    pub fn no_content(&self) -> bool {
        match self {
            OpenAIRequestMessage::System(_) => false,
            OpenAIRequestMessage::Developer(_) => false,
            OpenAIRequestMessage::User(OpenAIUserRequestMessage { content }) => content.is_empty(),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content,
                tool_calls,
            }) => content.is_none() && tool_calls.is_none(),
            OpenAIRequestMessage::Tool(_) => false,
        }
    }
    pub fn content_contains_case_insensitive(&self, value: &str) -> bool {
        match self {
            OpenAIRequestMessage::System(msg) => msg.content.to_lowercase().contains(value),
            OpenAIRequestMessage::Developer(msg) => msg.content.to_lowercase().contains(value),
            OpenAIRequestMessage::User(msg) => msg.content.iter().any(|c| match c {
                OpenAIContentBlock::Text { text } => text.to_lowercase().contains(value),
                OpenAIContentBlock::ImageUrl { .. }
                | OpenAIContentBlock::File { .. }
                | OpenAIContentBlock::InputAudio { .. } => false,
                // Don't inspect the contents of 'unknown' blocks
                OpenAIContentBlock::Unknown { data: _ } => false,
            }),
            OpenAIRequestMessage::Assistant(msg) => {
                if let Some(content) = &msg.content {
                    content.iter().any(|c| match c {
                        OpenAIContentBlock::Text { text } => text.to_lowercase().contains(value),
                        OpenAIContentBlock::ImageUrl { .. }
                        | OpenAIContentBlock::File { .. }
                        | OpenAIContentBlock::InputAudio { .. } => false,
                        // Don't inspect the contents of 'unknown' blocks
                        OpenAIContentBlock::Unknown { data: _ } => false,
                    })
                } else {
                    false
                }
            }
            OpenAIRequestMessage::Tool(msg) => msg.content.to_lowercase().contains(value),
        }
    }
}

impl<'a> SystemOrDeveloper<'a> {
    pub fn into_openai_request_message(self) -> OpenAIRequestMessage<'a> {
        match self {
            SystemOrDeveloper::System(msg) => {
                OpenAIRequestMessage::System(OpenAISystemRequestMessage { content: msg })
            }
            SystemOrDeveloper::Developer(msg) => {
                OpenAIRequestMessage::Developer(OpenAISystemRequestMessage { content: msg })
            }
        }
    }
    pub fn into_openai_responses_input(self) -> OpenAIResponsesInput<'a> {
        let role = match self {
            SystemOrDeveloper::System(_) => "system",
            SystemOrDeveloper::Developer(_) => "developer",
        };
        let text = match self {
            SystemOrDeveloper::System(msg) => msg,
            SystemOrDeveloper::Developer(msg) => msg,
        };
        OpenAIResponsesInput::Known(OpenAIResponsesInputInner::Message(
            OpenAIResponsesInputMessage {
                role,
                id: None,
                content: vec![OpenAIResponsesInputMessageContent::InputText { text }],
            },
        ))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct OpenAIMessagesConfig<'a> {
    pub json_mode: Option<&'a ModelInferenceRequestJsonMode>,
    pub provider_type: &'a str,
    pub fetch_and_encode_input_files_before_inference: bool,
}

fn supports_detail_parameter(provider_type: &str) -> bool {
    matches!(provider_type, "openai" | "azure" | "xai")
}

pub async fn prepare_openai_messages<'a>(
    system_or_developer: Option<SystemOrDeveloper<'a>>,
    messages: &'a [RequestMessage],
    config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let mut openai_messages: Vec<_> = try_join_all(
        messages
            .iter()
            .map(|msg| tensorzero_to_openai_messages(msg, config)),
    )
    .await?
    .into_iter()
    .flatten()
    .collect();

    if let Some(system_msg) =
        prepare_system_or_developer_message(system_or_developer, config.json_mode, &openai_messages)
    {
        openai_messages.insert(0, system_msg);
    }

    Ok(openai_messages)
}

/// Helper function to prepare allowed_tools constraint when dynamic allowed_tools are set.
/// This returns the AllowedToolsChoice struct with the appropriate mode and tool references.
///
/// This is shared logic across OpenAI-compatible providers (OpenAI, Groq, OpenRouter).
pub(crate) fn prepare_allowed_tools_constraint<'a>(
    tool_config: &'a ToolCallConfig,
) -> Option<AllowedToolsChoice<'a>> {
    // OpenAI-compatible providers don't allow both tool-choice "none" and tool-choice "allowed_tools",
    // since they're both set via the top-level "tool_choice" field.
    // We make `ToolChoice::None` take priority - that is, we allow "none" of the allowed tools.
    if tool_config.tool_choice == ToolChoice::None {
        return None;
    }
    let allowed_tools_list = tool_config.allowed_tools.as_dynamic_allowed_tools()?;

    // Construct the OpenAI spec-compliant allowed_tools structure
    let mode = match &tool_config.tool_choice {
        ToolChoice::Required => AllowedToolsMode::Required,
        _ => AllowedToolsMode::Auto,
    };

    // For each allowed tool name, determine if it's a function or custom tool
    let tool_refs: Vec<ToolReference> = allowed_tools_list
        .iter()
        .map(|name| {
            // Check if this tool name belongs to a custom tool
            let is_custom = tool_config
                .openai_custom_tools
                .iter()
                .any(|custom_tool| custom_tool.name == *name);

            if is_custom {
                ToolReference::Custom {
                    custom: SpecificToolCustom { name },
                }
            } else {
                ToolReference::Function {
                    function: SpecificToolFunction { name },
                }
            }
        })
        .collect();

    Some(AllowedToolsChoice {
        r#type: "allowed_tools",
        allowed_tools: AllowedToolsConstraint {
            mode,
            tools: tool_refs,
        },
    })
}

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to OpenAI format
fn prepare_openai_tools<'a>(request: &'a ModelInferenceRequest) -> PreparedOpenAIToolsResult<'a> {
    match &request.tool_config {
        None => (None, None, None),
        Some(tool_config) => {
            if !tool_config.any_tools_available() && tool_config.openai_custom_tools.is_empty() {
                return (None, None, None);
            }
            // This is the only place where we add OpenAI custom tools
            let tools = Some(
                tool_config
                    .tools_available_with_openai_custom()
                    .map(|tool_ref| match tool_ref {
                        ToolConfigRef::Function(func) => func.into(),
                        ToolConfigRef::OpenAICustom(custom) => OpenAITool::Custom { custom },
                    })
                    .collect(),
            );
            let parallel_tool_calls = tool_config.parallel_tool_calls;

            let tool_choice =
                if let Some(allowed_tools_choice) = prepare_allowed_tools_constraint(tool_config) {
                    Some(OpenAIToolChoice::AllowedTools(allowed_tools_choice))
                } else {
                    // No allowed_tools constraint, use regular tool_choice
                    Some((&tool_config.tool_choice).into())
                };

            (tools, tool_choice, parallel_tool_calls)
        }
    }
}

pub enum SystemOrDeveloper<'a> {
    System(Cow<'a, str>),
    Developer(Cow<'a, str>),
}

/// Prepares a system or developer message for OpenAI APIs with JSON mode handling.
///
/// When JSON mode is `On`, OpenAI/Azure require "JSON" to appear in either the system/developer
/// message or conversation. This function adds "Respond using JSON." when needed.
///
/// # System vs Developer Role
/// OpenAI is transitioning from "system" to "developer" role. Both work on most endpoints,
/// but newer features (e.g., reinforcement fine-tuning) only accept "developer".
/// This function preserves the specified role type for backward compatibility.
///
/// # Returns
/// * `Some(message)` - When content exists or JSON mode is On
/// * `None` - When no content and JSON mode is Off/Strict/None
///
/// # Behavior
/// - Checks for existing "JSON" mentions before adding instructions
/// - Only adds "Respond using JSON." prefix when necessary
/// - Preserves the original message role type
///
/// # Example
/// ```rust,ignore
/// let system_msg = prepare_system_or_developer_message(
///     Some(SystemOrDeveloper::System("You are a helpful assistant".into())),
///     Some(&ModelInferenceRequestJsonMode::On),
///     &messages
/// );
/// // Returns: System message with "Respond using JSON.\n\nYou are a helpful assistant"
/// ```
pub(super) fn prepare_system_or_developer_message<'a>(
    system_or_developer: Option<SystemOrDeveloper<'a>>,
    json_mode: Option<&'_ ModelInferenceRequestJsonMode>,
    messages: &[OpenAIRequestMessage<'a>],
) -> Option<OpenAIRequestMessage<'a>> {
    prepare_system_or_developer_message_helper(system_or_developer, json_mode, |content| {
        should_add_json_instruction_chat_completion(content, messages)
    })
    .map(SystemOrDeveloper::into_openai_request_message)
}

pub(super) fn prepare_system_or_developer_message_helper<'a>(
    system_or_developer: Option<SystemOrDeveloper<'a>>,
    json_mode: Option<&'_ ModelInferenceRequestJsonMode>,
    contains_content: impl FnOnce(&str) -> bool,
) -> Option<SystemOrDeveloper<'a>> {
    let (content, is_system) = match system_or_developer {
        Some(SystemOrDeveloper::System(content)) => (Some(content), true),
        Some(SystemOrDeveloper::Developer(content)) => (Some(content), false),
        None => (None, true), // Default to system message for JSON mode fallback
    };

    let final_content = match (content, json_mode) {
        // No content and no JSON mode - return None
        (
            None,
            None | Some(ModelInferenceRequestJsonMode::Off | ModelInferenceRequestJsonMode::Strict),
        ) => return None,

        // No content but JSON mode is on - create JSON instruction
        (None, Some(ModelInferenceRequestJsonMode::On)) => {
            Cow::Owned("Respond using JSON.".to_string())
        }

        // Has content and JSON mode is on - conditionally add JSON instruction
        (Some(content), Some(ModelInferenceRequestJsonMode::On)) => {
            if contains_content(&content) {
                Cow::Owned(format!("Respond using JSON.\n\n{content}"))
            } else {
                content
            }
        }

        // Has content, no JSON mode or JSON mode off/strict - use as-is
        (Some(content), _) => content,
    };

    Some(if is_system {
        SystemOrDeveloper::System(final_content)
    } else {
        SystemOrDeveloper::Developer(final_content)
    })
}

fn should_add_json_instruction_chat_completion(
    content: &str,
    messages: &[OpenAIRequestMessage<'_>],
) -> bool {
    !content.to_lowercase().contains("json")
        && !messages
            .iter()
            .any(|msg| msg.content_contains_case_insensitive("json"))
}

pub(super) async fn tensorzero_to_openai_messages<'a>(
    message: &'a RequestMessage,
    messages_config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    match message.role {
        Role::User => tensorzero_to_openai_user_messages(&message.content, messages_config).await,
        Role::Assistant => {
            let message = tensorzero_to_openai_assistant_message(
                Cow::Borrowed(&message.content),
                messages_config,
            )
            .await?;
            if message.no_content() {
                Ok(vec![])
            } else {
                Ok(vec![message])
            }
        }
    }
}

pub(super) async fn prepare_file_message(
    file: &LazyFile,
    messages_config: OpenAIMessagesConfig<'_>,
) -> Result<OpenAIContentBlock<'static>, Error> {
    match file {
        // If we have all of the following:
        // * The user passed in an image URL (not base64-encoded file data)
        // * The user explicitly specified an image mime type
        // * The `fetch_and_encode_input_files_before_inference` config setting is off (so we're allowed to forward image urls)
        //
        // Then, we can forward the image url directly to OpenAI. Unfortunately, we need to know the mime type for this to work,
        // since we need to map images to "image_url" content blocks.
        // Without downloading the file, we cannot guarantee that we guess the mime type correctly, so we don't try.
        //
        // OpenAI doesn't support passing in urls for 'file' content blocks, so we can only forward image urls.
        LazyFile::Url {
            file_url:
                FileUrl {
                    mime_type,
                    url,
                    detail,
                },
            future: _,
        } if !messages_config.fetch_and_encode_input_files_before_inference
        // If the mime type was provided by the caller we know we should only forward image URLs and fetch the rest
        && matches!(mime_type.as_ref().map(mime::MediaType::type_), Some(mime::IMAGE) | None) =>
        {
            let detail_to_use = if detail.is_some()
                && !supports_detail_parameter(messages_config.provider_type)
            {
                tracing::warn!(
                    "The image detail setting is not supported by `{}`. The `detail` field will be ignored.",
                    messages_config.provider_type
                );
                None
            } else {
                detail.clone()
            };
            Ok(OpenAIContentBlock::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: url.to_string(),
                    detail: detail_to_use,
                },
            })
        }
        _ => {
            let resolved_file = file.resolve().await?;
            let ObjectStorageFile { file, data } = &*resolved_file;
            let base64_url = format!("data:{};base64,{}", file.mime_type, data);
            if file.mime_type.type_() == mime::IMAGE {
                let detail_to_use = if file.detail.is_some()
                    && !supports_detail_parameter(messages_config.provider_type)
                {
                    tracing::warn!(
                        "The image detail setting is not supported by `{}`. The `detail` field will be ignored.",
                        messages_config.provider_type
                    );
                    None
                } else {
                    file.detail.clone()
                };
                Ok(OpenAIContentBlock::ImageUrl {
                    image_url: OpenAIImageUrl {
                        // This will only produce an error if we pass in a bad
                        // `Base64File` (with missing file data)
                        url: base64_url,
                        detail: detail_to_use,
                    },
                })
            } else if file.mime_type.type_() == mime::AUDIO {
                // Audio files use the input_audio format with unprefixed base64 and format field
                let format = mime_type_to_audio_format(&file.mime_type)?;
                Ok(OpenAIContentBlock::InputAudio {
                    input_audio: OpenAIInputAudio {
                        data: Cow::Owned(data.clone()),
                        format: Cow::Owned(format.to_string()),
                    },
                })
            } else {
                // OpenAI doesn't document how they determine the content type of the base64 blob
                // - let's try to pick a good suffix for the filename, in case they don't sniff
                // the mime type from the actual file content.
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
                Ok(OpenAIContentBlock::File {
                    file: OpenAIFile {
                        file_data: Some(Cow::Owned(base64_url)),
                        filename: Some(filename),
                    },
                })
            }
        }
    }
}

async fn tensorzero_to_openai_user_messages<'a>(
    content_blocks: &'a [ContentBlock],
    messages_config: OpenAIMessagesConfig<'a>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    // We need to separate the tool result messages from the user content blocks.

    let mut messages = Vec::new();
    let mut user_content_blocks = Vec::new();

    for block in content_blocks {
        match block {
            ContentBlock::Text(Text { text }) => {
                user_content_blocks.push(OpenAIContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(_) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool calls are not supported in user messages".to_string(),
                }));
            }
            ContentBlock::ToolResult(tool_result) => {
                messages.push(OpenAIRequestMessage::Tool(OpenAIToolRequestMessage {
                    content: &tool_result.result,
                    tool_call_id: &tool_result.id,
                }));
            }
            ContentBlock::File(file) => {
                user_content_blocks.push(prepare_file_message(file, messages_config).await?);
            }
            ContentBlock::Thought(thought) => {
                warn_discarded_thought_block(messages_config.provider_type, thought);
            }
            ContentBlock::Unknown(Unknown { data, .. }) => {
                user_content_blocks.push(OpenAIContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
        };
    }

    // If there are any user content blocks, combine them into a single user message.
    if !user_content_blocks.is_empty() {
        messages.push(OpenAIRequestMessage::User(OpenAIUserRequestMessage {
            content: user_content_blocks,
        }));
    }

    Ok(messages)
}

pub async fn tensorzero_to_openai_assistant_message<'a>(
    content_blocks: Cow<'a, [ContentBlock]>,
    messages_config: OpenAIMessagesConfig<'a>,
) -> Result<OpenAIRequestMessage<'a>, Error> {
    let content_block_cows: Vec<Cow<'_, ContentBlock>> = match content_blocks {
        Cow::Borrowed(content_blocks) => content_blocks.iter().map(Cow::Borrowed).collect(),
        Cow::Owned(content_blocks) => content_blocks.into_iter().map(Cow::Owned).collect(),
    };

    // We need to separate the tool result messages from the assistant content blocks.
    let mut assistant_content_blocks = Vec::new();
    let mut assistant_tool_calls = Vec::new();

    for block in content_block_cows {
        match block {
            Cow::Borrowed(ContentBlock::Text(Text { text })) => {
                assistant_content_blocks.push(OpenAIContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            Cow::Owned(ContentBlock::Text(Text { text })) => {
                assistant_content_blocks.push(OpenAIContentBlock::Text {
                    text: Cow::Owned(text),
                });
            }
            Cow::Borrowed(ContentBlock::ToolCall(tool_call)) => {
                let tool_call = OpenAIRequestToolCall {
                    id: Cow::Borrowed(&tool_call.id),
                    r#type: OpenAIToolType::Function,
                    function: OpenAIRequestFunctionCall {
                        name: Cow::Borrowed(&tool_call.name),
                        arguments: Cow::Borrowed(&tool_call.arguments),
                    },
                };

                assistant_tool_calls.push(tool_call);
            }
            Cow::Owned(ContentBlock::ToolCall(tool_call)) => {
                let tool_call = OpenAIRequestToolCall {
                    id: Cow::Owned(tool_call.id),
                    r#type: OpenAIToolType::Function,
                    function: OpenAIRequestFunctionCall {
                        name: Cow::Owned(tool_call.name),
                        arguments: Cow::Owned(tool_call.arguments),
                    },
                };

                assistant_tool_calls.push(tool_call);
            }
            Cow::Borrowed(ContentBlock::ToolResult(_))
            | Cow::Owned(ContentBlock::ToolResult(_)) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message: "Tool results are not supported in assistant messages".to_string(),
                }));
            }
            Cow::Borrowed(ContentBlock::File(ref file))
            | Cow::Owned(ContentBlock::File(ref file)) => {
                assistant_content_blocks.push(prepare_file_message(file, messages_config).await?);
            }
            Cow::Borrowed(ContentBlock::Thought(ref thought))
            | Cow::Owned(ContentBlock::Thought(ref thought)) => {
                warn_discarded_thought_block(messages_config.provider_type, thought);
            }
            Cow::Borrowed(ContentBlock::Unknown(Unknown { data, .. })) => {
                assistant_content_blocks.push(OpenAIContentBlock::Unknown {
                    data: Cow::Borrowed(data),
                });
            }
            Cow::Owned(ContentBlock::Unknown(Unknown { data, .. })) => {
                assistant_content_blocks.push(OpenAIContentBlock::Unknown {
                    data: Cow::Owned(data),
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

    let message = OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
        content,
        tool_calls,
    });

    Ok(message)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum OpenAIResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: Value,
    },
}

impl OpenAIResponseFormat {
    fn new(
        json_mode: ModelInferenceRequestJsonMode,
        output_schema: Option<&Value>,
        model: &str,
    ) -> Option<Self> {
        if model.contains("3.5") && json_mode == ModelInferenceRequestJsonMode::Strict {
            return Some(OpenAIResponseFormat::JsonObject);
        }

        match json_mode {
            ModelInferenceRequestJsonMode::On => Some(OpenAIResponseFormat::JsonObject),
            // For now, we never explicitly send `OpenAIResponseFormat::Text`
            ModelInferenceRequestJsonMode::Off => None,
            ModelInferenceRequestJsonMode::Strict => match output_schema {
                Some(schema) => {
                    let json_schema = json!({"name": "response", "strict": true, "schema": schema});
                    Some(OpenAIResponseFormat::JsonSchema { json_schema })
                }
                None => Some(OpenAIResponseFormat::JsonObject),
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIToolType {
    Function,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct OpenAIFunction<'a> {
    pub name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<&'a str>,
    pub parameters: &'a Value,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAITool<'a> {
    Function {
        function: OpenAIFunction<'a>,
        strict: bool,
    },
    Custom {
        custom: &'a OpenAICustomTool,
    },
}

impl<'a> From<&'a FunctionToolConfig> for OpenAITool<'a> {
    fn from(tool: &'a FunctionToolConfig) -> Self {
        OpenAITool::Function {
            function: OpenAIFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
            strict: tool.strict(),
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
pub struct OpenAISFTTool<'a> {
    pub r#type: OpenAIToolType,
    pub function: OpenAIFunction<'a>,
}

impl<'a> From<&'a FunctionTool> for OpenAISFTTool<'a> {
    fn from(tool: &'a FunctionTool) -> Self {
        OpenAISFTTool {
            r#type: OpenAIToolType::Function,
            function: OpenAIFunction {
                name: &tool.name,
                description: Some(&tool.description),
                parameters: &tool.parameters,
            },
        }
    }
}

impl<'a> From<&'a FunctionTool> for OpenAITool<'a> {
    fn from(tool: &'a FunctionTool) -> Self {
        OpenAITool::Function {
            function: OpenAIFunction {
                name: &tool.name,
                description: Some(&tool.description),
                parameters: &tool.parameters,
            },
            strict: tool.strict,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIBatchParams<'a> {
    file_id: Cow<'a, str>,
    batch_id: Cow<'a, str>,
}

impl<'a> OpenAIBatchParams<'a> {
    #[instrument(name = "OpenAIBatchParams::from_ref", skip_all, fields(%value))]
    fn from_ref(value: &'a Value) -> Result<Self, Error> {
        let file_id = value
            .get("file_id")
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidBatchParams {
                    message: "Missing file_id in batch params".to_string(),
                })
            })?
            .as_str()
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidBatchParams {
                    message: "file_id must be a string".to_string(),
                })
            })?;
        let batch_id = value
            .get("batch_id")
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidBatchParams {
                    message: "Missing batch_id in batch params".to_string(),
                })
            })?
            .as_str()
            .ok_or_else(|| {
                Error::new(ErrorDetails::InvalidBatchParams {
                    message: "batch_id must be a string".to_string(),
                })
            })?;
        Ok(Self {
            file_id: Cow::Borrowed(file_id),
            batch_id: Cow::Borrowed(batch_id),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(super) enum OpenAIToolChoice<'a> {
    String(OpenAIToolChoiceString),
    Specific(SpecificToolChoice<'a>),
    AllowedTools(AllowedToolsChoice<'a>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum OpenAIToolChoiceString {
    None,
    Auto,
    Required,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct SpecificToolChoice<'a> {
    pub(super) r#type: OpenAIToolType,
    pub(super) function: SpecificToolFunction<'a>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SpecificToolFunction<'a> {
    pub name: &'a str,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SpecificToolCustom<'a> {
    pub name: &'a str,
}

/// Represents the OpenAI API's allowed_tools constraint for tool_choice.
/// This matches the OpenAI spec structure:
/// {
///   "type": "allowed_tools",
///   "allowed_tools": {
///     "mode": "auto" | "required",
///     "tools": [
///       {"type": "function", "function": {"name": "..."}},
///       {"type": "custom", "custom": {"name": "..."}}
///     ]
///   }
/// }
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AllowedToolsChoice<'a> {
    pub r#type: &'static str, // Always "allowed_tools"
    pub allowed_tools: AllowedToolsConstraint<'a>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AllowedToolsConstraint<'a> {
    pub mode: AllowedToolsMode,
    pub tools: Vec<ToolReference<'a>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AllowedToolsMode {
    Auto,
    Required,
}

/// A reference to a tool by name, used in allowed_tools constraint.
/// Can reference either a function tool or a custom tool.
/// Serializes as:
///   - Function: {"type": "function", "function": {"name": "tool_name"}}
///   - Custom: {"type": "custom", "custom": {"name": "tool_name"}}
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolReference<'a> {
    Function { function: SpecificToolFunction<'a> },
    Custom { custom: SpecificToolCustom<'a> },
}

impl Default for OpenAIToolChoice<'_> {
    fn default() -> Self {
        OpenAIToolChoice::String(OpenAIToolChoiceString::None)
    }
}

impl<'a> From<&'a ToolChoice> for OpenAIToolChoice<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::None => OpenAIToolChoice::String(OpenAIToolChoiceString::None),
            ToolChoice::Auto => OpenAIToolChoice::String(OpenAIToolChoiceString::Auto),
            ToolChoice::Required => OpenAIToolChoice::String(OpenAIToolChoiceString::Required),
            ToolChoice::Specific(tool_name) => OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction { name: tool_name },
            }),
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct StreamOptions {
    pub(super) include_usage: bool,
}

/// This struct defines the supported parameters for the OpenAI API
/// See the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Default, Serialize)]
struct OpenAIRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
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
    response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<String>,
}

impl<'a> OpenAIRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<OpenAIRequest<'a>, Error> {
        let response_format =
            OpenAIResponseFormat::new(request.json_mode, request.output_schema, model);
        let stream_options = if request.stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        };
        let mut messages = prepare_openai_messages(
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

        let (tools, tool_choice, mut parallel_tool_calls) = prepare_openai_tools(request);
        if model.to_lowercase().starts_with("o1") && parallel_tool_calls == Some(false) {
            parallel_tool_calls = None;
        }

        if model.to_lowercase().starts_with("o1-mini") {
            if let Some(OpenAIRequestMessage::System(_)) = messages.first() {
                if let OpenAIRequestMessage::System(system_msg) = messages.remove(0) {
                    let user_msg = OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                        content: vec![OpenAIContentBlock::Text {
                            text: system_msg.content,
                        }],
                    });
                    messages.insert(0, user_msg);
                }
            }
        }

        let mut openai_request = OpenAIRequest {
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
            // allowed_tools is now part of tool_choice (AllowedToolsChoice variant)
            stop: request.borrow_stop_sequences(),
            reasoning_effort: None, // handled below
            service_tier: None,     // handled below
            verbosity: None,        // handled below
        };

        apply_inference_params(&mut openai_request, &request.inference_params_v2);

        Ok(openai_request)
    }
}

#[derive(Debug, Serialize)]
struct OpenAIBatchFileInput<'a> {
    custom_id: String,
    method: String,
    url: String,
    body: OpenAIRequest<'a>,
}

impl<'a> OpenAIBatchFileInput<'a> {
    async fn new(
        inference_id: Uuid,
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<Self, Error> {
        let body = OpenAIRequest::new(model, request).await?;
        Ok(Self {
            custom_id: inference_id.to_string(),
            method: "POST".to_string(),
            url: "/v1/chat/completions".to_string(),
            body,
        })
    }
}

#[derive(Debug, Serialize)]
struct OpenAIBatchRequest<'a> {
    input_file_id: &'a str,
    endpoint: &'a str,
    completion_window: &'a str,
    // metadata: HashMap<String, String>
}

impl<'a> OpenAIBatchRequest<'a> {
    fn new(input_file_id: &'a str) -> Self {
        Self {
            input_file_id,
            endpoint: "/v1/chat/completions",
            completion_window: "24h",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct OpenAIUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

impl From<OpenAIUsage> for Usage {
    fn from(usage: OpenAIUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct OpenAIEmbeddingUsage {
    pub prompt_tokens: Option<u32>,
}

impl From<OpenAIEmbeddingUsage> for Usage {
    fn from(usage: OpenAIEmbeddingUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: Some(0), // this is always zero for embeddings
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub(super) struct OpenAIResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub(super) struct OpenAIResponseCustomCall {
    name: String,
    input: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum OpenAIResponseToolCall {
    Function {
        id: String,
        function: OpenAIResponseFunctionCall,
    },
    Custom {
        id: String,
        custom: OpenAIResponseCustomCall,
    },
}

impl From<OpenAIResponseToolCall> for ToolCall {
    fn from(openai_tool_call: OpenAIResponseToolCall) -> Self {
        match openai_tool_call {
            OpenAIResponseToolCall::Function { id, function } => ToolCall {
                id,
                name: function.name,
                arguments: function.arguments,
            },
            OpenAIResponseToolCall::Custom { id, custom } => ToolCall {
                id,
                name: custom.name,
                arguments: custom.input,
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct OpenAIResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) content: Option<String>,
    // OpenAI doesn't currently set this field, but some OpenAI-compatible
    // providers (e.g. VLLM) do.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum OpenAIFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
    #[serde(other)]
    Unknown,
}

impl From<OpenAIFinishReason> for FinishReason {
    fn from(finish_reason: OpenAIFinishReason) -> Self {
        match finish_reason {
            OpenAIFinishReason::Stop => FinishReason::Stop,
            OpenAIFinishReason::Length => FinishReason::Length,
            OpenAIFinishReason::ContentFilter => FinishReason::ContentFilter,
            OpenAIFinishReason::ToolCalls => FinishReason::ToolCall,
            OpenAIFinishReason::FunctionCall => FinishReason::ToolCall,
            OpenAIFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct OpenAIResponseChoice {
    pub(super) index: u8,
    pub(super) message: OpenAIResponseMessage,
    pub(super) finish_reason: OpenAIFinishReason,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct OpenAIResponse {
    pub(super) choices: Vec<OpenAIResponseChoice>,
    pub(super) usage: OpenAIUsage,
}

struct OpenAIResponseWithMetadata<'a> {
    response: OpenAIResponse,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
    raw_response: String,
}

impl<'a> TryFrom<OpenAIResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: OpenAIResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let OpenAIResponseWithMetadata {
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
struct OpenAIFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenAIToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: OpenAIFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenAIDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    // OpenAI doesn't currently set this field, but some OpenAI-compatible
    // providers (e.g. VLLM) do.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCallChunk>>,
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
struct OpenAIChatChunkChoice {
    delta: OpenAIDelta,
    #[serde(default)]
    #[serde(deserialize_with = "empty_string_as_none")]
    finish_reason: Option<OpenAIFinishReason>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenAIChatChunk {
    choices: Vec<OpenAIChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAIUsage>,
}

/// Maps an OpenAI chunk to a TensorZero chunk for streaming inferences
fn openai_to_tensorzero_chunk(
    raw_message: String,
    mut chunk: OpenAIChatChunk,
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
        if let Some(reasoning) = choice.delta.reasoning_content {
            // We don't have real chunk ids, so always use chunk id 1 for reasoning content
            // (which should get concatenated into a single ContentBlock by the client)
            content.push(ContentBlockChunk::Thought(ThoughtChunk {
                text: Some(reasoning),
                signature: None,
                provider_type: Some(PROVIDER_TYPE.to_string()),
                id: "1".to_string(),
                summary_id: None,
                summary_text: None,
            }));
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

#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a EmbeddingInput,
    dimensions: Option<u32>,
    encoding_format: EmbeddingEncodingFormat,
}

impl<'a> OpenAIEmbeddingRequest<'a> {
    fn new(
        model: &'a str,
        input: &'a EmbeddingInput,
        dimensions: Option<u32>,
        encoding_format: EmbeddingEncodingFormat,
    ) -> Self {
        Self {
            model,
            input,
            dimensions,
            encoding_format,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    usage: Option<OpenAIEmbeddingUsage>,
}

struct OpenAIEmbeddingResponseWithMetadata<'a> {
    response: OpenAIEmbeddingResponse,
    latency: Latency,
    request: OpenAIEmbeddingRequest<'a>,
    raw_response: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIEmbeddingData {
    embedding: Embedding,
}

impl<'a> TryFrom<OpenAIEmbeddingResponseWithMetadata<'a>> for EmbeddingProviderResponse {
    type Error = Error;
    fn try_from(response: OpenAIEmbeddingResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let OpenAIEmbeddingResponseWithMetadata {
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

#[derive(Debug, Deserialize)]
struct OpenAIFileResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIBatchResponse {
    id: String,
    // object: String,
    // endpoint: String,
    errors: Option<OpenAIBatchErrors>,
    // input_file_id: String,
    // completion_window: String,
    status: OpenAIBatchStatus,
    output_file_id: Option<String>,
    // error_file_id: String,
    // created_at: i64,
    // in_progress_at: Option<i64>,
    // expires_at: i64,
    // finalizing_at: Option<i64>,
    // completed_at: Option<i64>,
    // failed_at: Option<i64>,
    // expired_at: Option<i64>,
    // cancelling_at: Option<i64>,
    // cancelled_at: Option<i64>,
    // request_counts: OpenAIBatchRequestCounts,
    // metadata: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OpenAIBatchStatus {
    Validating,
    Failed,
    InProgress,
    Finalizing,
    Completed,
    Expired,
    Cancelling,
    Cancelled,
}

impl From<OpenAIBatchStatus> for BatchStatus {
    fn from(status: OpenAIBatchStatus) -> Self {
        match status {
            OpenAIBatchStatus::Completed => BatchStatus::Completed,
            OpenAIBatchStatus::Validating
            | OpenAIBatchStatus::InProgress
            | OpenAIBatchStatus::Finalizing => BatchStatus::Pending,
            OpenAIBatchStatus::Failed
            | OpenAIBatchStatus::Expired
            | OpenAIBatchStatus::Cancelling
            | OpenAIBatchStatus::Cancelled => BatchStatus::Failed,
        }
    }
}

impl TryFrom<OpenAIBatchFileRow> for ProviderBatchInferenceOutput {
    type Error = Error;

    fn try_from(row: OpenAIBatchFileRow) -> Result<Self, Self::Error> {
        let mut response = row.response.body;
        // Validate we have exactly one choice
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                raw_request: None,
                raw_response: Some(serde_json::to_string(&response).unwrap_or_default()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }

        // Convert response to raw string for storage
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error parsing response: {}", DisplayOrDebugGateway::new(e)),
            })
        })?;

        // Extract the message from choices
        let OpenAIResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: None,
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }))?;

        // Convert message content to ContentBlocks
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(text) = message.content {
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        }

        Ok(Self {
            id: row.inference_id,
            output: content,
            raw_response,
            usage: response.usage.into(),
            finish_reason: Some(finish_reason.into()),
        })
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIBatchErrors {
    // object: String,
    data: Vec<OpenAIBatchError>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIBatchError {
    code: String,
    message: String,
    param: Option<String>,
    line: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct OpenAIBatchFileRow {
    #[serde(rename = "custom_id")]
    inference_id: Uuid,
    response: OpenAIBatchFileResponse,
}

#[derive(Debug, Deserialize)]
struct OpenAIBatchFileResponse {
    // status_code: u16,
    // request_id: String,
    body: OpenAIResponse,
}

#[cfg(test)]
mod tests {
    use base64::prelude::*;
    use base64::Engine;
    use futures::FutureExt;
    use serde_json::json;
    use std::borrow::Cow;

    use crate::inference::types::file::Detail;
    use crate::inference::types::storage::{StorageKind, StoragePath};
    use crate::inference::types::{
        FunctionType, ObjectStorageFile, ObjectStoragePointer, PendingObjectStoreFile,
        RequestMessage,
    };
    use crate::providers::test_helpers::{
        MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL, WEATHER_TOOL_CONFIG,
    };
    use crate::tool::ToolCallConfig;
    use crate::utils::testing::capture_logs;

    use super::*;

    static FERRIS_PNG: &[u8] = include_bytes!("../../../tests/e2e/providers/ferris.png");

    #[test]
    fn test_get_chat_url() {
        // Test with custom base URL
        let custom_base = "https://custom.openai.com/api/";
        let custom_url = get_chat_url(&Url::parse(custom_base).unwrap()).unwrap();
        assert_eq!(
            custom_url.as_str(),
            "https://custom.openai.com/api/chat/completions"
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
    fn test_handle_openai_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_openai_error(
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
        let forbidden = handle_openai_error(
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
        let rate_limit = handle_openai_error(
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
        let server_error = handle_openai_error(
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
    async fn test_openai_request_new() {
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

        let openai_request = OpenAIRequest::new("gpt-4.1-mini", &basic_request)
            .await
            .unwrap();

        assert_eq!(openai_request.model, "gpt-4.1-mini");
        assert_eq!(openai_request.messages.len(), 2);
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_completion_tokens, Some(100));
        assert_eq!(openai_request.seed, Some(69));
        assert_eq!(openai_request.top_p, Some(0.9));
        assert_eq!(openai_request.presence_penalty, Some(0.1));
        assert_eq!(openai_request.frequency_penalty, Some(0.2));
        assert!(openai_request.stream);
        assert_eq!(openai_request.response_format, None);
        assert!(openai_request.tools.is_none());
        assert_eq!(openai_request.tool_choice, None);
        assert!(openai_request.parallel_tool_calls.is_none());

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

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools)
            .await
            .unwrap();

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 2); // We'll add a system message containing Json to fit OpenAI requirements
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_completion_tokens, None);
        assert_eq!(openai_request.seed, None);
        assert_eq!(openai_request.top_p, None);
        assert_eq!(openai_request.presence_penalty, None);
        assert_eq!(openai_request.frequency_penalty, None);
        assert!(!openai_request.stream);
        assert_eq!(
            openai_request.response_format,
            Some(OpenAIResponseFormat::JsonObject)
        );
        assert!(openai_request.tools.is_some());
        let tools = openai_request.tools.as_ref().unwrap();
        match &tools[0] {
            OpenAITool::Function { function, .. } => {
                assert_eq!(function.name, WEATHER_TOOL.name());
                assert_eq!(function.parameters, WEATHER_TOOL.parameters());
            }
            OpenAITool::Custom { .. } => panic!("Expected Function tool"),
        }
        assert_eq!(
            openai_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
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

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools)
            .await
            .unwrap();

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 1);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_completion_tokens, None);
        assert_eq!(openai_request.seed, None);
        assert!(!openai_request.stream);
        assert_eq!(openai_request.top_p, None);
        assert_eq!(openai_request.presence_penalty, None);
        assert_eq!(openai_request.frequency_penalty, None);
        // Resolves to normal JSON mode since no schema is provided (this shouldn't really happen in practice)
        assert_eq!(
            openai_request.response_format,
            Some(OpenAIResponseFormat::JsonObject)
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

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools)
            .await
            .unwrap();

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 1);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_completion_tokens, None);
        assert_eq!(openai_request.seed, None);
        assert!(!openai_request.stream);
        assert_eq!(openai_request.top_p, None);
        assert_eq!(openai_request.presence_penalty, None);
        assert_eq!(openai_request.frequency_penalty, None);
        let expected_schema = serde_json::json!({"name": "response", "strict": true, "schema": {}});
        assert_eq!(
            openai_request.response_format,
            Some(OpenAIResponseFormat::JsonSchema {
                json_schema: expected_schema,
            })
        );
    }

    #[tokio::test]
    async fn test_openai_new_request_o1() {
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

        let openai_request = OpenAIRequest::new("o1-preview", &request).await.unwrap();

        assert_eq!(openai_request.model, "o1-preview");
        assert_eq!(openai_request.messages.len(), 1);
        assert!(!openai_request.stream);
        assert_eq!(openai_request.response_format, None);
        assert_eq!(openai_request.temperature, Some(0.5));
        assert_eq!(openai_request.max_completion_tokens, Some(100));
        assert_eq!(openai_request.seed, Some(69));
        assert_eq!(openai_request.top_p, Some(0.9));
        assert_eq!(openai_request.presence_penalty, Some(0.1));
        assert_eq!(openai_request.frequency_penalty, Some(0.2));
        assert!(openai_request.tools.is_none());

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

        let openai_request_with_system = OpenAIRequest::new("o1-mini", &request_with_system)
            .await
            .unwrap();

        // Check that the system message was converted to a user message
        assert_eq!(openai_request_with_system.messages.len(), 2);
        assert!(
            matches!(
                openai_request_with_system.messages[0],
                OpenAIRequestMessage::User(ref msg) if msg.content == [OpenAIContentBlock::Text { text: "This is the system message".into() }]
            ),
            "Unexpected messages: {:?}",
            openai_request_with_system.messages
        );

        assert_eq!(openai_request_with_system.model, "o1-mini");
        assert!(!openai_request_with_system.stream);
        assert_eq!(openai_request_with_system.response_format, None);
        assert_eq!(openai_request_with_system.temperature, Some(0.5));
        assert_eq!(openai_request_with_system.max_completion_tokens, Some(100));
        assert_eq!(openai_request_with_system.seed, Some(69));
        assert!(openai_request_with_system.tools.is_none());
        assert_eq!(openai_request_with_system.top_p, Some(0.9));
        assert_eq!(openai_request_with_system.presence_penalty, Some(0.1));
        assert_eq!(openai_request_with_system.frequency_penalty, Some(0.2));
    }

    #[test]
    fn test_try_from_openai_response() {
        // Test case 1: Valid response with content
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

        let request_body = OpenAIRequest {
            messages: vec![],
            model: "gpt-4.1-mini",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            response_format: Some(OpenAIResponseFormat::Text),
            ..Default::default()
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test_response".to_string();
        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithMetadata {
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
        let valid_response_with_tools = OpenAIResponse {
            choices: vec![OpenAIResponseChoice {
                index: 0,
                finish_reason: OpenAIFinishReason::ToolCalls,
                message: OpenAIResponseMessage {
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![OpenAIResponseToolCall::Function {
                        id: "call1".to_string(),
                        function: OpenAIResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
            }],
            usage: OpenAIUsage {
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

        let request_body = OpenAIRequest {
            messages: vec![],
            model: "gpt-4.1-mini",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            response_format: Some(OpenAIResponseFormat::Text),
            ..Default::default()
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithMetadata {
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
        let invalid_response_no_choices = OpenAIResponse {
            choices: vec![],
            usage: OpenAIUsage {
                prompt_tokens: Some(5),
                completion_tokens: Some(0),
            },
        };
        let request_body = OpenAIRequest {
            messages: vec![],
            model: "gpt-4.1-mini",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            response_format: Some(OpenAIResponseFormat::Text),
            ..Default::default()
        };
        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithMetadata {
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
        let invalid_response_multiple_choices = OpenAIResponse {
            choices: vec![
                OpenAIResponseChoice {
                    index: 0,
                    message: OpenAIResponseMessage {
                        content: Some("Choice 1".to_string()),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    finish_reason: OpenAIFinishReason::Stop,
                },
                OpenAIResponseChoice {
                    index: 1,
                    finish_reason: OpenAIFinishReason::Stop,
                    message: OpenAIResponseMessage {
                        content: Some("Choice 2".to_string()),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                },
            ],
            usage: OpenAIUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(10),
            },
        };

        let request_body = OpenAIRequest {
            messages: vec![],
            model: "gpt-4.1-mini",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            response_format: Some(OpenAIResponseFormat::Text),
            ..Default::default()
        };
        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithMetadata {
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
    fn test_prepare_openai_tools() {
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
        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(&request_with_tools);
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2);
        match &tools[0] {
            OpenAITool::Function { function, .. } => {
                assert_eq!(function.name, WEATHER_TOOL.name());
                assert_eq!(function.parameters, WEATHER_TOOL.parameters());
            }
            OpenAITool::Custom { .. } => panic!("Expected Function tool"),
        }
        match &tools[1] {
            OpenAITool::Function { function, .. } => {
                assert_eq!(function.name, QUERY_TOOL.name());
                assert_eq!(function.parameters, QUERY_TOOL.parameters());
            }
            OpenAITool::Custom { .. } => panic!("Expected Function tool"),
        }
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            OpenAIToolChoice::String(OpenAIToolChoiceString::Required)
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
            prepare_openai_tools(&request_without_tools);
        assert!(tools.is_none());
        assert!(tool_choice.is_none());
        assert!(parallel_tool_calls.is_none());
    }

    #[tokio::test]
    async fn test_tensorzero_to_openai_messages() {
        let content_blocks = vec!["Hello".to_string().into()];
        let openai_messages = tensorzero_to_openai_user_messages(
            &content_blocks,
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(openai_messages.len(), 1);
        match &openai_messages[0] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(
                    content.content,
                    &[OpenAIContentBlock::Text {
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
        let openai_messages = tensorzero_to_openai_user_messages(
            &content_blocks,
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(openai_messages.len(), 1);
        match &openai_messages[0] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(
                    content.content,
                    vec![
                        OpenAIContentBlock::Text {
                            text: "Hello".into()
                        },
                        OpenAIContentBlock::Text {
                            text: "How are you?".into()
                        }
                    ]
                );
            }
            _ => panic!("Expected a user message"),
        }

        // User message with one string and one tool call block
        // Since user messages in OpenAI land can't contain tool calls (nor should they honestly),
        // We split the tool call out into a separate assistant message
        let tool_block = ContentBlock::ToolCall(ToolCall {
            id: "call1".to_string(),
            name: "test_function".to_string(),
            arguments: "{}".to_string(),
        });
        let content_blocks = vec!["Hello".to_string().into(), tool_block];
        let openai_message = tensorzero_to_openai_assistant_message(
            Cow::Borrowed(&content_blocks),
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                fetch_and_encode_input_files_before_inference: false,
            },
        )
        .await
        .unwrap();
        match &openai_message {
            OpenAIRequestMessage::Assistant(content) => {
                assert_eq!(
                    content.content,
                    Some(vec![OpenAIContentBlock::Text {
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
    fn test_openai_to_tensorzero_chunk() {
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
                delta: OpenAIDelta {
                    content: Some("Hello".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: Some(OpenAIFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let message = openai_to_tensorzero_chunk(
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
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
                finish_reason: Some(OpenAIFinishReason::ToolCalls),
                delta: OpenAIDelta {
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![OpenAIToolCallChunk {
                        index: 0,
                        id: None,
                        function: OpenAIFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message = openai_to_tensorzero_chunk(
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
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
                finish_reason: None,
                delta: OpenAIDelta {
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![OpenAIToolCallChunk {
                        index: 1,
                        id: None,
                        function: OpenAIFunctionCallChunk {
                            name: None,
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let error = openai_to_tensorzero_chunk(
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
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
                finish_reason: Some(OpenAIFinishReason::Stop),
                delta: OpenAIDelta {
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![OpenAIToolCallChunk {
                        index: 1,
                        id: Some("id2".to_string()),
                        function: OpenAIFunctionCallChunk {
                            name: Some("name2".to_string()),
                            arguments: Some("{\"hello\":\"world\"}".to_string()),
                        },
                    }]),
                },
            }],
            usage: None,
        };
        let message = openai_to_tensorzero_chunk(
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
        let chunk = OpenAIChatChunk {
            choices: vec![],
            usage: Some(OpenAIUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
            }),
        };
        let message = openai_to_tensorzero_chunk(
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
    fn test_new_openai_response_format() {
        // Test JSON mode On
        let json_mode = ModelInferenceRequestJsonMode::On;
        let output_schema = None;
        let format = OpenAIResponseFormat::new(json_mode, output_schema, "gpt-4o");
        assert_eq!(format, Some(OpenAIResponseFormat::JsonObject));

        // Test JSON mode Off
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let format = OpenAIResponseFormat::new(json_mode, output_schema, "gpt-4o");
        assert_eq!(format, None);

        // Test JSON mode Strict with no schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let format = OpenAIResponseFormat::new(json_mode, output_schema, "gpt-4o");
        assert_eq!(format, Some(OpenAIResponseFormat::JsonObject));

        // Test JSON mode Strict with schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"type": "string"}
            }
        });
        let output_schema = Some(&schema);
        let format = OpenAIResponseFormat::new(json_mode, output_schema, "gpt-4o");
        match format {
            Some(OpenAIResponseFormat::JsonSchema { json_schema }) => {
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
        let format = OpenAIResponseFormat::new(json_mode, output_schema, "gpt-3.5-turbo");
        assert_eq!(format, Some(OpenAIResponseFormat::JsonObject));
    }

    #[test]
    fn test_openai_api_base() {
        assert_eq!(
            OPENAI_DEFAULT_BASE_URL.as_str(),
            "https://api.openai.com/v1/"
        );
    }

    #[test]
    fn test_prepare_system_or_developer_message() {
        // Test Case 1: system_or_developer is None, json_mode is Off
        let system_or_developer = None;
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages: Vec<OpenAIRequestMessage> = vec![];
        let result =
            prepare_system_or_developer_message(system_or_developer, Some(&json_mode), &messages);
        assert_eq!(result, None);

        // Test Case 2: system is Some, json_mode is On, messages contain "json"
        let system_or_developer = Some(SystemOrDeveloper::System("System instructions".into()));
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                content: vec![OpenAIContentBlock::Text {
                    text: "Please respond in JSON format.".into(),
                }],
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some(vec![OpenAIContentBlock::Text {
                    text: "Sure, here is the data.".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed("System instructions"),
        }));
        let result =
            prepare_system_or_developer_message(system_or_developer, Some(&json_mode), &messages);
        assert_eq!(result, expected);

        // Test Case 3: system is Some, json_mode is On, messages do not contain "json"
        let system_or_developer = Some(SystemOrDeveloper::System("System instructions".into()));
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                content: vec![OpenAIContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some(vec![OpenAIContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected_content = "Respond using JSON.\n\nSystem instructions".to_string();
        let expected = Some(OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Owned(expected_content),
        }));
        let result =
            prepare_system_or_developer_message(system_or_developer, Some(&json_mode), &messages);
        assert_eq!(result, expected);

        // Test Case 4: developer is Some, json_mode is Off
        let system_or_developer = Some(SystemOrDeveloper::Developer(
            "Developer instructions".into(),
        ));
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages = vec![
            OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                content: vec![OpenAIContentBlock::Text {
                    text: "Hello, how are you?".into(),
                }],
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some(vec![OpenAIContentBlock::Text {
                    text: "I am fine, thank you!".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(OpenAIRequestMessage::Developer(
            OpenAISystemRequestMessage {
                content: Cow::Borrowed("Developer instructions"),
            },
        ));
        let result =
            prepare_system_or_developer_message(system_or_developer, Some(&json_mode), &messages);
        assert_eq!(result, expected);

        // Test Case 5: developer is Some, json_mode is On, messages do not contain "json"
        let system_or_developer = Some(SystemOrDeveloper::Developer(
            "Developer instructions".into(),
        ));
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![OpenAIRequestMessage::User(OpenAIUserRequestMessage {
            content: vec![OpenAIContentBlock::Text {
                text: "Hello, how are you?".into(),
            }],
        })];
        let expected_content = "Respond using JSON.\n\nDeveloper instructions".to_string();
        let expected = Some(OpenAIRequestMessage::Developer(
            OpenAISystemRequestMessage {
                content: Cow::Owned(expected_content),
            },
        ));
        let result =
            prepare_system_or_developer_message(system_or_developer, Some(&json_mode), &messages);
        assert_eq!(result, expected);

        // Test Case 6: system is None, json_mode is On
        let system_or_developer = None;
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![
            OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                content: vec![OpenAIContentBlock::Text {
                    text: "Tell me a joke.".into(),
                }],
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some(vec![OpenAIContentBlock::Text {
                    text: "Sure, here's one for you.".into(),
                }]),
                tool_calls: None,
            }),
        ];
        let expected = Some(OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Owned("Respond using JSON.".to_string()),
        }));
        let result =
            prepare_system_or_developer_message(system_or_developer, Some(&json_mode), &messages);
        assert_eq!(result, expected);

        // Test Case 7: system is None, json_mode is Strict
        let system_or_developer = None;
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let messages = vec![
            OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                content: vec![OpenAIContentBlock::Text {
                    text: "Provide a summary of the news.".into(),
                }],
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some(vec![OpenAIContentBlock::Text {
                    text: "Here's the summary.".into(),
                }]),
                tool_calls: None,
            }),
        ];

        let result =
            prepare_system_or_developer_message(system_or_developer, Some(&json_mode), &messages);
        assert!(result.is_none());
    }

    #[test]
    fn test_create_file_url() {
        use url::Url;

        // Test Case 1: Base URL without trailing slash
        let base_url = Url::parse("https://api.openai.com/v1").unwrap();
        let file_id = Some("file123");
        let result = get_file_url(&base_url, file_id).unwrap();
        assert_eq!(
            result.as_str(),
            "https://api.openai.com/v1/files/file123/content"
        );

        // Test Case 2: Base URL with trailing slash
        let base_url = Url::parse("https://api.openai.com/v1/").unwrap();
        let file_id = Some("file456");
        let result = get_file_url(&base_url, file_id).unwrap();
        assert_eq!(
            result.as_str(),
            "https://api.openai.com/v1/files/file456/content"
        );

        // Test Case 3: Base URL with custom domain
        let base_url = Url::parse("https://custom-openai.example.com").unwrap();
        let file_id = Some("file789");
        let result = get_file_url(&base_url, file_id).unwrap();
        assert_eq!(
            result.as_str(),
            "https://custom-openai.example.com/files/file789/content"
        );

        // Test Case 4: Base URL without trailing slash, no file ID
        let base_url = Url::parse("https://api.openai.com/v1").unwrap();
        let result = get_file_url(&base_url, None).unwrap();
        assert_eq!(result.as_str(), "https://api.openai.com/v1/files");

        // Test Case 5: Base URL with trailing slash, no file ID
        let base_url = Url::parse("https://api.openai.com/v1/").unwrap();
        let result = get_file_url(&base_url, None).unwrap();
        assert_eq!(result.as_str(), "https://api.openai.com/v1/files");

        // Test Case 6: Custom domain base URL, no file ID
        let base_url = Url::parse("https://custom-openai.example.com").unwrap();
        let result = get_file_url(&base_url, None).unwrap();
        assert_eq!(result.as_str(), "https://custom-openai.example.com/files");
    }

    #[test]
    fn test_try_from_openai_credentials() {
        // Test Static credentials
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = OpenAICredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenAICredentials::Static(_)));

        // Test Dynamic credentials
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = OpenAICredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenAICredentials::Dynamic(_)));

        // Test None credentials
        let generic = Credential::None;
        let creds = OpenAICredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenAICredentials::None));

        // Test Missing credentials
        let generic = Credential::Missing;
        let creds = OpenAICredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenAICredentials::None));

        // Test invalid credential type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = OpenAICredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_serialize_user_messages() {
        // Test that a single message is serialized as 'content: string'
        let message = OpenAIUserRequestMessage {
            content: vec![OpenAIContentBlock::Text {
                text: "My single message".into(),
            }],
        };
        let serialized = serde_json::to_string(&message).unwrap();
        assert_eq!(serialized, r#"{"content":"My single message"}"#);

        // Test that a multiple messages are serialized as an array of content blocks
        let message = OpenAIUserRequestMessage {
            content: vec![
                OpenAIContentBlock::Text {
                    text: "My first message".into(),
                },
                OpenAIContentBlock::Text {
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

    #[tokio::test]
    async fn test_prepare_resolved_file_message() {
        let dummy_storage_path = StoragePath {
            kind: StorageKind::Disabled,
            path: object_store::path::Path::parse("dummy-path").unwrap(),
        };
        let file = LazyFile::Base64(PendingObjectStoreFile(ObjectStorageFile {
            file: ObjectStoragePointer {
                source_url: None,
                mime_type: mime::TEXT_PLAIN,
                storage_path: dummy_storage_path.clone(),
                detail: None,
                filename: None,
            },
            data: BASE64_STANDARD.encode(b"Hello, world!"),
        }));
        let first_res = prepare_file_message(
            &file,
            OpenAIMessagesConfig {
                fetch_and_encode_input_files_before_inference: true,
                json_mode: None,
                provider_type: PROVIDER_TYPE,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            first_res,
            OpenAIContentBlock::File {
                file: OpenAIFile {
                    file_data: Some(Cow::Owned(format!(
                        "data:text/plain;base64,{}",
                        BASE64_STANDARD.encode("Hello, world!")
                    ))),
                    filename: Some(Cow::Owned("input.txt".to_string())),
                },
            }
        );

        let second_res = prepare_file_message(
            &file,
            OpenAIMessagesConfig {
                fetch_and_encode_input_files_before_inference: false,
                json_mode: None,
                provider_type: PROVIDER_TYPE,
            },
        )
        .await
        .unwrap();

        // Since the file is already resolved, 'fetch_and_encode_input_files_before_inference' should have no effect
        assert_eq!(second_res, first_res);
    }

    #[tokio::test]
    async fn test_prepare_resolved_file_message_with_detail() {
        let dummy_storage_path = StoragePath {
            kind: StorageKind::Disabled,
            path: object_store::path::Path::parse("dummy-path").unwrap(),
        };
        let file = LazyFile::Base64(PendingObjectStoreFile(ObjectStorageFile {
            file: ObjectStoragePointer {
                source_url: None,
                mime_type: mime::IMAGE_PNG,
                storage_path: dummy_storage_path.clone(),
                detail: Some(Detail::High),
                filename: None,
            },
            data: BASE64_STANDARD.encode(b"fake image data"),
        }));
        let res = prepare_file_message(
            &file,
            OpenAIMessagesConfig {
                fetch_and_encode_input_files_before_inference: true,
                json_mode: None,
                provider_type: PROVIDER_TYPE,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            res,
            OpenAIContentBlock::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: format!(
                        "data:image/png;base64,{}",
                        BASE64_STANDARD.encode(b"fake image data")
                    ),
                    detail: Some(Detail::High),
                },
            }
        );
    }

    #[tokio::test]
    async fn test_file_url_no_mime_type_fetch_and_encode() {
        let logs_contain = capture_logs();
        let fetch_and_encode = OpenAIMessagesConfig {
            fetch_and_encode_input_files_before_inference: true,
            json_mode: None,
            provider_type: PROVIDER_TYPE,
        };
        let dummy_storage_path = StoragePath {
            kind: StorageKind::Disabled,
            path: object_store::path::Path::parse("dummy-path").unwrap(),
        };
        let url = Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png").unwrap();
        let res = prepare_file_message(
            &LazyFile::Url {
                file_url: FileUrl {
                    url: url.clone(),
                    mime_type: None,
                    detail: None,
                },
                future: async move {
                    Ok(ObjectStorageFile {
                        file: ObjectStoragePointer {
                            source_url: None,
                            // Deliberately use a different mime type to make sure we adjust the input filename
                            mime_type: mime::IMAGE_JPEG,
                            storage_path: dummy_storage_path.clone(),
                            detail: None,
                            filename: None,
                        },
                        data: BASE64_STANDARD.encode(FERRIS_PNG),
                    })
                }
                .boxed()
                .shared(),
            },
            fetch_and_encode,
        )
        .await
        .unwrap();

        assert_eq!(
            res,
            OpenAIContentBlock::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: format!(
                        "data:image/jpeg;base64,{}",
                        BASE64_STANDARD.encode(FERRIS_PNG)
                    ),
                    detail: None,
                },
            }
        );

        // We're encoding the file, so don't produce a warning about the user not providing an explicit mime_type
        assert!(!logs_contain("mime_type"));
    }

    #[tokio::test]
    async fn test_file_url_warn_mime_type() {
        let fetch_and_encode = OpenAIMessagesConfig {
            fetch_and_encode_input_files_before_inference: false,
            json_mode: None,
            provider_type: PROVIDER_TYPE,
        };
        let dummy_storage_path = StoragePath {
            kind: StorageKind::Disabled,
            path: object_store::path::Path::parse("dummy-path").unwrap(),
        };
        let url = Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png").unwrap();
        let res = prepare_file_message(
            &LazyFile::Url {
                file_url: FileUrl {
                    url: url.clone(),
                    mime_type: None,
                    detail: None,
                },
                future: async move {
                    Ok(ObjectStorageFile {
                        file: ObjectStoragePointer {
                            source_url: None,
                            // Deliberately use a different mime type to make sure we adjust the input filename
                            mime_type: mime::IMAGE_JPEG,
                            storage_path: dummy_storage_path.clone(),
                            detail: None,
                            filename: None,
                        },
                        data: BASE64_STANDARD.encode(FERRIS_PNG),
                    })
                }
                .boxed()
                .shared(),
            },
            fetch_and_encode,
        )
        .await
        .unwrap();

        // We didn't provide an input mime_type, so we'll go ahead and forward it
        assert_eq!(
            res,
            OpenAIContentBlock::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: url.to_string(),
                    detail: None,
                },
            }
        );
    }

    #[tokio::test]
    async fn test_forward_image_url() {
        let logs_contain = capture_logs();
        let fetch_and_encode = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };
        let url = Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png").unwrap();
        let res = prepare_file_message(
            &LazyFile::Url {
                file_url: FileUrl {
                    url: url.clone(),
                    mime_type: Some(mime::IMAGE_JPEG),
                    detail: None,
                },
                future: async { panic!("File future should not be resolved") }
                    .boxed()
                    .shared(),
            },
            fetch_and_encode,
        )
        .await
        .unwrap();

        // We provided an input mime_type, so we should forward the image url
        assert_eq!(
            res,
            OpenAIContentBlock::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: url.to_string(),
                    detail: None,
                },
            }
        );

        assert!(!logs_contain("mime_type"));
    }

    #[tokio::test]
    async fn test_forward_image_url_with_detail_low() {
        let fetch_and_encode = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };
        let url = Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png").unwrap();
        let res = prepare_file_message(
            &LazyFile::Url {
                file_url: FileUrl {
                    url: url.clone(),
                    mime_type: Some(mime::IMAGE_JPEG),
                    detail: Some(Detail::Low),
                },
                future: async { panic!("File future should not be resolved") }
                    .boxed()
                    .shared(),
            },
            fetch_and_encode,
        )
        .await
        .unwrap();

        assert_eq!(
            res,
            OpenAIContentBlock::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: url.to_string(),
                    detail: Some(Detail::Low),
                },
            }
        );
    }

    #[tokio::test]
    async fn test_forward_image_url_with_detail_high() {
        let fetch_and_encode = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };
        let url = Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png").unwrap();
        let res = prepare_file_message(
            &LazyFile::Url {
                file_url: FileUrl {
                    url: url.clone(),
                    mime_type: Some(mime::IMAGE_JPEG),
                    detail: Some(Detail::High),
                },
                future: async { panic!("File future should not be resolved") }
                    .boxed()
                    .shared(),
            },
            fetch_and_encode,
        )
        .await
        .unwrap();

        assert_eq!(
            res,
            OpenAIContentBlock::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: url.to_string(),
                    detail: Some(Detail::High),
                },
            }
        );
    }

    #[tokio::test]
    async fn test_forward_image_url_with_detail_auto() {
        let fetch_and_encode = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };
        let url = Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png").unwrap();
        let res = prepare_file_message(
            &LazyFile::Url {
                file_url: FileUrl {
                    url: url.clone(),
                    mime_type: Some(mime::IMAGE_JPEG),
                    detail: Some(Detail::Auto),
                },
                future: async { panic!("File future should not be resolved") }
                    .boxed()
                    .shared(),
            },
            fetch_and_encode,
        )
        .await
        .unwrap();

        assert_eq!(
            res,
            OpenAIContentBlock::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: url.to_string(),
                    detail: Some(Detail::Auto),
                },
            }
        );
    }

    #[tokio::test]
    async fn test_cannot_forward_file_url() {
        let logs_contain = capture_logs();
        let fetch_and_encode = OpenAIMessagesConfig {
            json_mode: None,
            provider_type: PROVIDER_TYPE,
            fetch_and_encode_input_files_before_inference: false,
        };
        let url = Url::parse("https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png").unwrap();
        let res = prepare_file_message(
            &LazyFile::Url {
                file_url: FileUrl {
                    url: url.clone(),
                    // By specifying a non-image mime type, we should end up using a 'file' content block
                    mime_type: Some(mime::APPLICATION_PDF),
                    detail: None,
                },
                future: async {
                    Ok(ObjectStorageFile {
                        file: ObjectStoragePointer {
                            source_url: None,
                            mime_type: mime::APPLICATION_PDF,
                            storage_path: StoragePath {
                                kind: StorageKind::Disabled,
                                path: object_store::path::Path::parse("dummy-path").unwrap(),
                            },
                            detail: None,
                            filename: None,
                        },
                        data: BASE64_STANDARD.encode(FERRIS_PNG),
                    })
                }
                .boxed()
                .shared(),
            },
            fetch_and_encode,
        )
        .await
        .unwrap();

        // We provided an input mime_type, so we should forward the image url
        assert_eq!(
            res,
            OpenAIContentBlock::File {
                file: OpenAIFile {
                    file_data: Some(Cow::Owned(format!(
                        "data:application/pdf;base64,{}",
                        BASE64_STANDARD.encode(FERRIS_PNG)
                    ))),
                    filename: Some(Cow::Owned("input.pdf".to_string())),
                },
            }
        );

        assert!(!logs_contain("mime_type"));
    }

    #[test]
    fn test_check_api_base_suffix() {
        let logs_contain = capture_logs();
        // Valid cases (should not warn)
        check_api_base_suffix(&Url::parse("http://localhost:1234/").unwrap());
        check_api_base_suffix(&Url::parse("http://localhost:1234/openai/").unwrap());
        check_api_base_suffix(&Url::parse("http://localhost:1234/openai/v1").unwrap());
        check_api_base_suffix(&Url::parse("http://localhost:1234/openai/v1/").unwrap());
        check_api_base_suffix(&Url::parse("http://localhost:1234/v1/").unwrap());
        check_api_base_suffix(&Url::parse("http://localhost:1234/v2").unwrap());
        check_api_base_suffix(&Url::parse("http://localhost:1234/v2/").unwrap());

        // Invalid cases (should warn)
        let url1 = Url::parse("http://localhost:1234/chat/completions").unwrap();
        check_api_base_suffix(&url1);
        assert!(logs_contain("automatically appends `/chat/completions`"));
        assert!(logs_contain(url1.as_ref()));

        let url2 = Url::parse("http://localhost:1234/chat/completions/").unwrap();
        check_api_base_suffix(&url2);
        assert!(logs_contain("automatically appends `/chat/completions`"));
        assert!(logs_contain(url2.as_ref()));

        let url3 = Url::parse("http://localhost:1234/v1/chat/completions").unwrap();
        check_api_base_suffix(&url3);
        assert!(logs_contain("automatically appends `/chat/completions`"));
        assert!(logs_contain(url3.as_ref()));

        let url4 = Url::parse("http://localhost:1234/v1/chat/completions/").unwrap();
        check_api_base_suffix(&url4);
        assert!(logs_contain("automatically appends `/chat/completions`"));
        assert!(logs_contain(url4.as_ref()));
    }

    #[test]
    fn test_openai_provider_new_api_base_check() {
        let logs_contain = capture_logs();
        let model_name = "test-model".to_string();

        // Valid cases (should not warn)
        let _ = OpenAIProvider::new(
            model_name.clone(),
            Some(Url::parse("http://localhost:1234/v1/").unwrap()),
            OpenAICredentials::None,
            OpenAIAPIType::ChatCompletions,
            false,
            Vec::new(),
        );

        let _ = OpenAIProvider::new(
            model_name.clone(),
            Some(Url::parse("http://localhost:1234/v1").unwrap()),
            OpenAICredentials::None,
            OpenAIAPIType::ChatCompletions,
            false,
            Vec::new(),
        );

        // Invalid cases (should warn)
        let invalid_url_1 = Url::parse("http://localhost:1234/chat/completions").unwrap();
        let _ = OpenAIProvider::new(
            model_name.clone(),
            Some(invalid_url_1.clone()),
            OpenAICredentials::None,
            OpenAIAPIType::ChatCompletions,
            false,
            Vec::new(),
        );
        assert!(logs_contain("automatically appends `/chat/completions`"));
        assert!(logs_contain(invalid_url_1.as_ref()));

        let invalid_url_2 = Url::parse("http://localhost:1234/v1/chat/completions/").unwrap();
        let _ = OpenAIProvider::new(
            model_name.clone(),
            Some(invalid_url_2.clone()),
            OpenAICredentials::None,
            OpenAIAPIType::ChatCompletions,
            false,
            Vec::new(),
        );
        assert!(logs_contain("automatically appends `/chat/completions`"));
        assert!(logs_contain(invalid_url_2.as_ref()));
    }

    #[tokio::test]
    async fn test_openai_apply_inference_params_called() {
        let logs_contain = capture_logs();
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["Test".to_string().into()],
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
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            inference_params_v2: ChatCompletionInferenceParamsV2 {
                reasoning_effort: Some("high".to_string()),
                service_tier: None,
                thinking_budget_tokens: Some(1024),
                verbosity: Some("low".to_string()),
            },
            extra_body: Default::default(),
            ..Default::default()
        };

        let openai_request = OpenAIRequest::new("gpt-4o", &request)
            .await
            .expect("Failed to create OpenAI request");

        // Test that reasoning_effort is applied correctly
        assert_eq!(openai_request.reasoning_effort, Some("high".to_string()));

        // Test that thinking_budget_tokens warns with tip about reasoning_effort
        assert!(logs_contain(
            "OpenAI does not support the inference parameter `thinking_budget_tokens`, so it will be ignored. Tip: You might want to use `reasoning_effort` for this provider."
        ));

        // Test that verbosity is applied correctly
        assert_eq!(openai_request.verbosity, Some("low".to_string()));
    }

    #[tokio::test]
    async fn test_prepare_file_message_with_custom_filename() {
        // Test that custom filename is preserved and used instead of fallback
        let dummy_storage_path = StoragePath {
            kind: StorageKind::Disabled,
            path: object_store::path::Path::parse("dummy-path").unwrap(),
        };
        let file = LazyFile::Base64(PendingObjectStoreFile(ObjectStorageFile {
            file: ObjectStoragePointer {
                source_url: None,
                mime_type: mime::TEXT_PLAIN,
                storage_path: dummy_storage_path.clone(),
                detail: None,
                filename: Some("my_document.txt".to_string()),
            },
            data: BASE64_STANDARD.encode(b"Hello, world!"),
        }));

        let res = prepare_file_message(
            &file,
            OpenAIMessagesConfig {
                fetch_and_encode_input_files_before_inference: true,
                json_mode: None,
                provider_type: PROVIDER_TYPE,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            res,
            OpenAIContentBlock::File {
                file: OpenAIFile {
                    file_data: Some(Cow::Owned(format!(
                        "data:text/plain;base64,{}",
                        BASE64_STANDARD.encode("Hello, world!")
                    ))),
                    filename: Some(Cow::Owned("my_document.txt".to_string())),
                },
            }
        );
    }

    #[tokio::test]
    async fn test_prepare_file_message_with_custom_pdf_filename() {
        // Test custom filename with PDF file
        let dummy_storage_path = StoragePath {
            kind: StorageKind::Disabled,
            path: object_store::path::Path::parse("dummy-path").unwrap(),
        };
        let file = LazyFile::Base64(PendingObjectStoreFile(ObjectStorageFile {
            file: ObjectStoragePointer {
                source_url: None,
                mime_type: mime::APPLICATION_PDF,
                storage_path: dummy_storage_path.clone(),
                detail: None,
                filename: Some("report_2024.pdf".to_string()),
            },
            data: BASE64_STANDARD.encode(b"%PDF-1.4"),
        }));

        let res = prepare_file_message(
            &file,
            OpenAIMessagesConfig {
                fetch_and_encode_input_files_before_inference: true,
                json_mode: None,
                provider_type: PROVIDER_TYPE,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            res,
            OpenAIContentBlock::File {
                file: OpenAIFile {
                    file_data: Some(Cow::Owned(format!(
                        "data:application/pdf;base64,{}",
                        BASE64_STANDARD.encode(b"%PDF-1.4")
                    ))),
                    filename: Some(Cow::Owned("report_2024.pdf".to_string())),
                },
            }
        );
    }

    #[tokio::test]
    async fn test_prepare_file_message_fallback_various_mime_types() {
        // Test that None filename falls back to "input.{ext}" for various MIME types
        let test_cases = vec![
            (mime::APPLICATION_PDF, "pdf"),
            (mime::IMAGE_JPEG, "jpg"),
            (mime::IMAGE_PNG, "png"),
        ];

        for (mime_type, expected_ext) in test_cases {
            let dummy_storage_path = StoragePath {
                kind: StorageKind::Disabled,
                path: object_store::path::Path::parse("dummy-path").unwrap(),
            };
            let file = LazyFile::Base64(PendingObjectStoreFile(ObjectStorageFile {
                file: ObjectStoragePointer {
                    source_url: None,
                    mime_type: mime_type.clone(),
                    storage_path: dummy_storage_path.clone(),
                    detail: None,
                    filename: None,
                },
                data: BASE64_STANDARD.encode(b"test data"),
            }));

            let res = prepare_file_message(
                &file,
                OpenAIMessagesConfig {
                    fetch_and_encode_input_files_before_inference: true,
                    json_mode: None,
                    provider_type: PROVIDER_TYPE,
                },
            )
            .await
            .unwrap();

            let expected_filename = format!("input.{expected_ext}");
            match res {
                OpenAIContentBlock::File { file } => {
                    assert_eq!(
                        file.filename,
                        Some(Cow::Owned(expected_filename)),
                        "Failed for MIME type: {mime_type}"
                    );
                }
                OpenAIContentBlock::ImageUrl { .. } => {
                    // Images use data URLs without filename field, which is fine
                    continue;
                }
                _ => panic!("Unexpected content block type for MIME type: {mime_type}"),
            }
        }
    }

    #[test]
    fn test_openai_chunk_missing_usage_block() {
        // Test that an OpenAI streaming chunk with no usage field is handled correctly
        let chunk_json = json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                "finish_reason": null
            }]
        });

        // Parse as OpenAIChatChunk
        let chunk: OpenAIChatChunk = serde_json::from_value(chunk_json).unwrap();

        // Verify the chunk was parsed successfully
        assert_eq!(chunk.choices.len(), 1);

        // Verify usage is None when the field is missing
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_openai_chunk_null_token_values() {
        // Test that an OpenAI chunk with null prompt_tokens and/or completion_tokens is handled correctly

        // Test with both tokens null
        let chunk_both_null = json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": null,
                "completion_tokens": null
            }
        });

        let chunk: OpenAIChatChunk = serde_json::from_value(chunk_both_null).unwrap();
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, None);
        assert_eq!(usage.completion_tokens, None);

        // Test with only prompt_tokens null
        let chunk_input_null = json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": null,
                "completion_tokens": 20
            }
        });

        let chunk2: OpenAIChatChunk = serde_json::from_value(chunk_input_null).unwrap();
        let usage2 = chunk2.usage.unwrap();
        assert_eq!(usage2.prompt_tokens, None);
        assert_eq!(usage2.completion_tokens, Some(20));

        // Test with only completion_tokens null
        let chunk_output_null = json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": null
            }
        });

        let chunk3: OpenAIChatChunk = serde_json::from_value(chunk_output_null).unwrap();
        let usage3 = chunk3.usage.unwrap();
        assert_eq!(usage3.prompt_tokens, Some(10));
        assert_eq!(usage3.completion_tokens, None);
    }

    #[test]
    fn test_prepare_openai_tools_with_no_tools() {
        // Test when no tool_config is provided
        let request = ModelInferenceRequest {
            tool_config: None,
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(&request);

        assert_eq!(tools, None);
        assert_eq!(tool_choice, None);
        assert_eq!(parallel_tool_calls, None);
    }

    #[test]
    fn test_prepare_openai_tools_with_allowed_tools_auto() {
        use crate::tool::{AllowedTools, AllowedToolsChoice};
        use std::borrow::Cow;

        // Create a tool config with explicit allowed_tools and auto tool choice
        let mut tool_config = MULTI_TOOL_CONFIG.clone();
        tool_config.tool_choice = ToolChoice::Auto;
        tool_config.allowed_tools = AllowedTools {
            tools: vec!["get_temperature".to_string()],
            choice: AllowedToolsChoice::Explicit,
        };

        let request = ModelInferenceRequest {
            tool_config: Some(Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(&request);

        // Verify tools are present
        assert!(tools.is_some());
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 2); // get_temperature and query_articles

        // Verify tool_choice is AllowedTools variant with Auto mode
        assert!(tool_choice.is_some());
        let tool_choice = tool_choice.unwrap();
        match tool_choice {
            OpenAIToolChoice::AllowedTools(allowed_tools_choice) => {
                assert_eq!(allowed_tools_choice.r#type, "allowed_tools");
                assert_eq!(
                    allowed_tools_choice.allowed_tools.mode,
                    AllowedToolsMode::Auto
                );
                assert_eq!(allowed_tools_choice.allowed_tools.tools.len(), 1);
                match &allowed_tools_choice.allowed_tools.tools[0] {
                    ToolReference::Function { function } => {
                        assert_eq!(function.name, "get_temperature");
                    }
                    ToolReference::Custom { .. } => panic!("Expected Function variant"),
                }
            }
            _ => panic!("Expected AllowedTools variant"),
        }

        assert_eq!(parallel_tool_calls, Some(true));
    }

    #[test]
    fn test_prepare_openai_tools_with_allowed_tools_required() {
        use crate::tool::{AllowedTools, AllowedToolsChoice};
        use std::borrow::Cow;

        // Create a tool config with explicit allowed_tools and required tool choice
        let mut tool_config = MULTI_TOOL_CONFIG.clone();
        tool_config.tool_choice = ToolChoice::Required;
        tool_config.allowed_tools = AllowedTools {
            tools: vec!["query_articles".to_string(), "get_temperature".to_string()],
            choice: AllowedToolsChoice::Explicit,
        };

        let request = ModelInferenceRequest {
            tool_config: Some(Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (tools, tool_choice, _parallel_tool_calls) = prepare_openai_tools(&request);

        // Verify tools are present
        assert!(tools.is_some());

        // Verify tool_choice is AllowedTools variant with Required mode
        assert!(tool_choice.is_some());
        let tool_choice = tool_choice.unwrap();
        match tool_choice {
            OpenAIToolChoice::AllowedTools(allowed_tools_choice) => {
                assert_eq!(allowed_tools_choice.r#type, "allowed_tools");
                assert_eq!(
                    allowed_tools_choice.allowed_tools.mode,
                    AllowedToolsMode::Required
                );
                assert_eq!(allowed_tools_choice.allowed_tools.tools.len(), 2);
                // Verify both tools are in the list
                let tool_names: Vec<&str> = allowed_tools_choice
                    .allowed_tools
                    .tools
                    .iter()
                    .map(|t| match t {
                        ToolReference::Function { function } => function.name,
                        ToolReference::Custom { .. } => panic!("Expected Function variant"),
                    })
                    .collect();
                assert!(tool_names.contains(&"query_articles"));
                assert!(tool_names.contains(&"get_temperature"));
            }
            _ => panic!("Expected AllowedTools variant"),
        }
    }

    #[test]
    fn test_prepare_openai_tools_with_allowed_tools_none_tool_choice() {
        use crate::tool::{AllowedTools, AllowedToolsChoice};
        use std::borrow::Cow;

        // Test that when tool_choice is None but allowed_tools is set,
        // we still use AllowedTools variant with Auto mode
        let mut tool_config = MULTI_TOOL_CONFIG.clone();
        tool_config.tool_choice = ToolChoice::None;
        tool_config.allowed_tools = AllowedTools {
            tools: vec!["get_temperature".to_string()],
            choice: AllowedToolsChoice::Explicit,
        };

        let request = ModelInferenceRequest {
            tool_config: Some(Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (_tools, tool_choice, _parallel_tool_calls) = prepare_openai_tools(&request);

        assert!(tool_choice.is_some());
        let tool_choice = tool_choice.unwrap();
        // ToolChoice::None with allowed_tools should map to None mode
        assert_eq!(
            tool_choice,
            OpenAIToolChoice::String(OpenAIToolChoiceString::None)
        );
    }

    #[test]
    fn test_prepare_openai_tools_with_allowed_tools_specific_tool_choice() {
        use crate::tool::{AllowedTools, AllowedToolsChoice};
        use std::borrow::Cow;

        // Test that Specific tool choice with allowed_tools uses Auto mode
        let mut tool_config = MULTI_TOOL_CONFIG.clone();
        tool_config.tool_choice = ToolChoice::Specific("get_temperature".to_string());
        tool_config.allowed_tools = AllowedTools {
            tools: vec!["get_temperature".to_string()],
            choice: AllowedToolsChoice::Explicit,
        };

        let request = ModelInferenceRequest {
            tool_config: Some(Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (_tools, tool_choice, _parallel_tool_calls) = prepare_openai_tools(&request);

        assert!(tool_choice.is_some());
        let tool_choice = tool_choice.unwrap();
        match tool_choice {
            OpenAIToolChoice::AllowedTools(allowed_tools_choice) => {
                // ToolChoice::Specific with allowed_tools should map to Auto mode
                assert_eq!(
                    allowed_tools_choice.allowed_tools.mode,
                    AllowedToolsMode::Auto
                );
            }
            _ => panic!("Expected AllowedTools variant"),
        }
    }

    #[test]
    fn test_prepare_openai_tools_without_allowed_tools() {
        use std::borrow::Cow;

        // Test that when allowed_tools is not set (FunctionDefault),
        // we use the regular tool_choice conversion
        let tool_config = MULTI_TOOL_CONFIG.clone();
        // MULTI_TOOL_CONFIG has ToolChoice::Required but no explicit allowed_tools

        let request = ModelInferenceRequest {
            tool_config: Some(Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (_tools, tool_choice, _parallel_tool_calls) = prepare_openai_tools(&request);

        assert!(tool_choice.is_some());
        let tool_choice = tool_choice.unwrap();
        // Without allowed_tools, should use regular String variant
        match tool_choice {
            OpenAIToolChoice::String(OpenAIToolChoiceString::Required) => {
                // This is expected
            }
            _ => panic!("Expected String(Required) variant, got {tool_choice:?}"),
        }
    }

    #[test]
    fn test_prepare_openai_tools_with_specific_tool_without_allowed_tools() {
        use std::borrow::Cow;

        // Test regular specific tool choice without allowed_tools
        let tool_config = WEATHER_TOOL_CONFIG.clone();
        // This has ToolChoice::Specific and no explicit allowed_tools

        let request = ModelInferenceRequest {
            tool_config: Some(Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (_tools, tool_choice, _parallel_tool_calls) = prepare_openai_tools(&request);

        assert!(tool_choice.is_some());
        let tool_choice = tool_choice.unwrap();
        // Without allowed_tools, Specific should convert to Specific variant
        match tool_choice {
            OpenAIToolChoice::Specific(specific) => {
                assert_eq!(specific.function.name, "get_temperature");
                assert_eq!(specific.r#type, OpenAIToolType::Function);
            }
            _ => panic!("Expected Specific variant, got {tool_choice:?}"),
        }
    }

    #[test]
    fn test_prepare_openai_tools_empty_allowed_tools_list() {
        use crate::tool::{AllowedTools, AllowedToolsChoice};
        use std::borrow::Cow;

        // Test edge case: explicit allowed_tools but empty list
        let mut tool_config = MULTI_TOOL_CONFIG.clone();
        tool_config.allowed_tools = AllowedTools {
            tools: vec![],
            choice: AllowedToolsChoice::Explicit,
        };

        let request = ModelInferenceRequest {
            tool_config: Some(Cow::Owned(tool_config)),
            ..Default::default()
        };

        let (_tools, tool_choice, _parallel_tool_calls) = prepare_openai_tools(&request);

        assert!(tool_choice.is_some());
        let tool_choice = tool_choice.unwrap();
        match tool_choice {
            OpenAIToolChoice::AllowedTools(allowed_tools_choice) => {
                assert_eq!(allowed_tools_choice.allowed_tools.tools.len(), 0);
            }
            _ => panic!("Expected AllowedTools variant with empty list"),
        }
    }
}
