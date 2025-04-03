use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest::multipart::{Form, Part};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::de::IntoDeserializer;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Write;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::time::Instant;
use tracing::instrument;
use url::Url;
use uuid::Uuid;

use crate::cache::ModelProviderRequest;
use crate::embeddings::{EmbeddingProvider, EmbeddingProviderResponse, EmbeddingRequest};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::batch::{
    ProviderBatchInferenceOutput, ProviderBatchInferenceResponse,
};
use crate::inference::types::resolved_input::ImageWithPath;
use crate::inference::types::{
    batch::{BatchStatus, StartBatchProviderInferenceResponse},
    ContentBlock, ContentBlockChunk, ContentBlockOutput, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, RequestMessage, Role, Text,
    TextChunk, Usage,
};
use crate::inference::types::{
    FinishReason, ProviderInferenceResponseArgs, ProviderInferenceResponseStreamInner,
};
use crate::model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

use crate::inference::providers::helpers::inject_extra_request_data;

lazy_static! {
    static ref OPENAI_DEFAULT_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.openai.com/v1/").expect("Failed to parse OPENAI_DEFAULT_BASE_URL")
    };
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("OPENAI_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "OpenAI";
const PROVIDER_TYPE: &str = "openai";

#[derive(Debug)]
pub struct OpenAIProvider {
    model_name: String,
    api_base: Option<Url>,
    credentials: OpenAICredentials,
}

static DEFAULT_CREDENTIALS: OnceLock<OpenAICredentials> = OnceLock::new();

impl OpenAIProvider {
    pub fn new(
        model_name: String,
        api_base: Option<Url>,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credentials = build_creds_caching_default(
            api_key_location,
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;
        Ok(OpenAIProvider {
            model_name,
            api_base,
            credentials,
        })
    }
}

#[derive(Clone, Debug)]
pub enum OpenAICredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for OpenAICredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(OpenAICredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(OpenAICredentials::Dynamic(key_name)),
            Credential::None => Ok(OpenAICredentials::None),
            Credential::Missing => Ok(OpenAICredentials::None),
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
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            OpenAICredentials::Static(api_key) => Ok(Some(api_key)),
            OpenAICredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                }))
                .transpose()
            }
            OpenAICredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for OpenAIProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let mut request_body = serde_json::to_value(OpenAIRequest::new(&self.model_name, request)?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error serializing OpenAI request: {e}"),
                })
            })?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let request_url = get_chat_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json");
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let res = request_builder
            .json(&request_body)
            .headers(headers)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to OpenAI: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
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
                request: request_body,
                generic_request: request,
            }
            .try_into()?)
        } else {
            Err(handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing error response: {e}"),
                        raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
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
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let mut request_body = serde_json::to_value(OpenAIRequest::new(&self.model_name, request)?)
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error serializing OpenAI request: {e}"),
                })
            })?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let request_url = get_chat_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .headers(headers);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let event_source = request_builder
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to OpenAI: {e}"),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

        let stream = stream_openai(event_source, start_time).peekable();
        Ok((stream, raw_request))
    }

    // Get a single chunk from the stream and make sure it is OK then send to client.
    // We want to do this here so that we can tell that the request is working.
    /// 1. Upload the requests to OpenAI as a File
    /// 2. Start the batch inference
    ///    We do them in sequence here.
    async fn start_batch_inference<'a>(
        &'a self,
        requests: &'a [ModelInferenceRequest<'_>],
        client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = get_file_url(
            self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL),
            None,
        )?;
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
        let mut jsonl_data = Vec::new();
        for item in batch_requests {
            serde_json::to_writer(&mut jsonl_data, &item).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error serializing request: {e}"),
                })
            })?;
            jsonl_data.write_all(b"\n").map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error writing to JSONL: {e}"),
                })
            })?;
        }
        // Create the multipart form
        let form = Form::new().text("purpose", "batch").part(
            "file",
            Part::bytes(jsonl_data)
                .file_name("data.jsonl")
                .mime_str("application/json")
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Error setting MIME type: {e}"),
                    })
                })?,
        );
        let mut request_builder = client.post(request_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        // Actually upload the file to OpenAI
        let res = request_builder.multipart(form).send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Error sending request to OpenAI: {e}"),
                status_code: e.status(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
        let text = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error retrieving text response: {e}"),
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
        let file_id = response.id;
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
                    message: format!("Error sending request to OpenAI: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&batch_request).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        let text = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error retrieving batch response: {e}"),
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
                message: format!("Error serializing OpenAI batch params: {e}"),
            })
        })?;
        let errors = match response.errors {
            Some(errors) => errors
                .data
                .into_iter()
                .map(|error| {
                    serde_json::to_value(&error).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!("Error serializing batch error: {e}"),
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
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        let batch_params = OpenAIBatchParams::from_ref(&batch_request.batch_params)?;
        let mut request_url =
            get_batch_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;
        request_url
            .path_segments_mut()
            .map_err(|_| {
                Error::new(ErrorDetails::Inference {
                    message: "Failed to get mutable path segments".to_string(),
                })
            })?
            .push(&batch_params.batch_id);
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let raw_request = request_url.to_string();
        let mut request_builder = http_client
            .get(request_url)
            .header("Content-Type", "application/json");
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let res = request_builder.send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Error sending request to OpenAI: {e}"),
                status_code: e.status(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&batch_request).unwrap_or_default()),
                raw_response: None,
            })
        })?;
        let text = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing JSON response: {e}"),
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

impl EmbeddingProvider for OpenAIProvider {
    async fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &reqwest::Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<EmbeddingProviderResponse, Error> {
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_body = OpenAIEmbeddingRequest::new(&self.model_name, &request.input);
        let request_url =
            get_embedding_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;
        let start_time = Instant::now();
        let mut request_builder = client
            .post(request_url)
            .header("Content-Type", "application/json");
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let res = request_builder
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to OpenAI: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response: OpenAIEmbeddingResponse =
                serde_json::from_str(&raw_response).map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing JSON response: {e}"),
                        raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
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
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing error response: {e}"),
                        raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?,
                PROVIDER_TYPE,
            ))
        }
    }
}

pub async fn convert_stream_error(provider_type: String, e: reqwest_eventsource::Error) -> Error {
    let message = e.to_string();
    let mut raw_response = None;
    if let reqwest_eventsource::Error::InvalidStatusCode(_, resp) = e {
        raw_response = resp.text().await.ok();
    }
    ErrorDetails::InferenceServer {
        message,
        raw_request: None,
        raw_response,
        provider_type,
    }
    .into()
}

pub fn stream_openai(
    mut event_source: EventSource,
    start_time: Instant,
) -> ProviderInferenceResponseStreamInner {
    let mut tool_call_ids = Vec::new();
    let mut tool_call_names = Vec::new();
    Box::pin(async_stream::stream! {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(convert_stream_error(PROVIDER_TYPE.to_string(), e).await);
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
                                    "Error parsing chunk. Error: {}",
                                    e,
                                ),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            openai_to_tensorzero_chunk(d, latency, &mut tool_call_ids, &mut tool_call_names)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    })
}

impl OpenAIProvider {
    // Once a batch has been completed we need to retrieve the results from OpenAI using the files API
    #[instrument(skip_all, fields(file_id = file_id))]
    async fn collect_finished_batch(
        &self,
        file_id: &str,
        client: &reqwest::Client,
        credentials: &InferenceCredentials,
        raw_request: String,
        raw_response: String,
    ) -> Result<ProviderBatchInferenceResponse, Error> {
        let file_url = get_file_url(
            self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL),
            Some(file_id),
        )?;
        let api_key = self.credentials.get_api_key(credentials)?;
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
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing error response for file {file_id}: {e}"),
                        raw_request: None,
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?,
                PROVIDER_TYPE,
            ));
        }

        let bytes = res.bytes().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error reading batch results response for file {file_id}: {e}"),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let mut elements: HashMap<Uuid, ProviderBatchInferenceOutput> = HashMap::new();
        let text = std::str::from_utf8(&bytes).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing batch results response for file {file_id}: {e}"),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        for line in text.lines() {
            let row = match serde_json::from_str::<OpenAIBatchFileRow>(line) {
                Ok(row) => row,
                Err(e) => {
                    // Construct error for logging but don't return it
                    let _ = Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing batch results row for file {file_id}: {e}"),
                        raw_request: None,
                        raw_response: Some(line.to_string()),
                        provider_type: PROVIDER_TYPE.to_string(),
                    });
                    continue;
                }
            };
            let output = match ProviderBatchInferenceOutput::try_from(row) {
                Ok(output) => output,
                Err(_) => {
                    // Construct error for logging but don't return it
                    continue;
                }
            };
            elements.insert(output.id, output);
        }

        Ok(ProviderBatchInferenceResponse {
            elements,
            raw_request,
            raw_response,
        })
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
        format!("files/{}/content", id)
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
            raw_request: None,
            raw_response: Some(response_body.to_string()),
            provider_type: provider_type.to_string(),
        }
        .into(),
        _ => ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            raw_request: None,
            raw_response: Some(response_body.to_string()),
            provider_type: provider_type.to_string(),
        }
        .into(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenAISystemRequestMessage<'a> {
    pub content: Cow<'a, str>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenAIUserRequestMessage<'a> {
    #[serde(serialize_with = "serialize_text_content_vec")]
    pub(super) content: Vec<OpenAIContentBlock<'a>>,
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

#[derive(Clone, Debug, PartialEq)]
pub enum OpenAIContentBlock<'a> {
    Text { text: Cow<'a, str> },
    ImageUrl { image_url: OpenAIImageUrl },
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
            Text { text: &'a str },
            ImageUrl { image_url: &'a OpenAIImageUrl },
        }
        match self {
            OpenAIContentBlock::Text { text } => Helper::Text { text }.serialize(serializer),
            OpenAIContentBlock::ImageUrl { image_url } => {
                Helper::ImageUrl { image_url }.serialize(serializer)
            }
            OpenAIContentBlock::Unknown { data } => data.serialize(serializer),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct OpenAIImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenAIRequestFunctionCall<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub struct OpenAIRequestToolCall<'a> {
    pub id: &'a str,
    pub r#type: OpenAIToolType,
    pub function: OpenAIRequestFunctionCall<'a>,
}

impl<'a> From<&'a ToolCall> for OpenAIRequestToolCall<'a> {
    fn from(tool_call: &'a ToolCall) -> Self {
        OpenAIRequestToolCall {
            id: &tool_call.id,
            r#type: OpenAIToolType::Function,
            function: OpenAIRequestFunctionCall {
                name: &tool_call.name,
                arguments: &tool_call.arguments,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenAIAssistantRequestMessage<'a> {
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_text_content_vec"
    )]
    pub content: Option<Vec<OpenAIContentBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIRequestToolCall<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenAIToolRequestMessage<'a> {
    pub content: &'a str,
    pub tool_call_id: &'a str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub(super) enum OpenAIRequestMessage<'a> {
    System(OpenAISystemRequestMessage<'a>),
    User(OpenAIUserRequestMessage<'a>),
    Assistant(OpenAIAssistantRequestMessage<'a>),
    Tool(OpenAIToolRequestMessage<'a>),
}

impl OpenAIRequestMessage<'_> {
    pub fn content_contains_case_insensitive(&self, value: &str) -> bool {
        match self {
            OpenAIRequestMessage::System(msg) => msg.content.to_lowercase().contains(value),
            OpenAIRequestMessage::User(msg) => msg.content.iter().any(|c| match c {
                OpenAIContentBlock::Text { text } => text.to_lowercase().contains(value),
                OpenAIContentBlock::ImageUrl { .. } => false,
                // Don't inspect the contents of 'unknown' blocks
                OpenAIContentBlock::Unknown { data: _ } => false,
            }),
            OpenAIRequestMessage::Assistant(msg) => {
                if let Some(content) = &msg.content {
                    content.iter().any(|c| match c {
                        OpenAIContentBlock::Text { text } => text.to_lowercase().contains(value),
                        OpenAIContentBlock::ImageUrl { .. } => false,
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

pub(super) fn prepare_openai_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Result<Vec<OpenAIRequestMessage<'a>>, Error> {
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in request.messages.iter() {
        messages.extend(tensorzero_to_openai_messages(message)?);
    }
    if let Some(system_msg) = tensorzero_to_openai_system_message(
        request.system.as_deref(),
        &request.json_mode,
        &messages,
    ) {
        messages.insert(0, system_msg);
    }
    Ok(messages)
}

/// If there are no tools passed or the tools are empty, return None for both tools and tool_choice
/// Otherwise convert the tool choice and tools to OpenAI format
pub(super) fn prepare_openai_tools<'a>(
    request: &'a ModelInferenceRequest,
) -> (
    Option<Vec<OpenAITool<'a>>>,
    Option<OpenAIToolChoice<'a>>,
    Option<bool>,
) {
    match &request.tool_config {
        None => (None, None, None),
        Some(tool_config) => {
            if tool_config.tools_available.is_empty() {
                return (None, None, None);
            }
            let tools = Some(
                tool_config
                    .tools_available
                    .iter()
                    .map(|tool| tool.into())
                    .collect(),
            );
            let tool_choice = Some((&tool_config.tool_choice).into());
            let parallel_tool_calls = tool_config.parallel_tool_calls;
            (tools, tool_choice, parallel_tool_calls)
        }
    }
}

/// This function is complicated only by the fact that OpenAI and Azure require
/// different instructions depending on the json mode and the content of the messages.
///
/// If ModelInferenceRequestJsonMode::On and the system message or instructions does not contain "JSON"
/// the request will return an error.
/// So, we need to format the instructions to include "Respond using JSON." if it doesn't already.
pub(super) fn tensorzero_to_openai_system_message<'a>(
    system: Option<&'a str>,
    json_mode: &ModelInferenceRequestJsonMode,
    messages: &[OpenAIRequestMessage<'a>],
) -> Option<OpenAIRequestMessage<'a>> {
    match system {
        Some(system) => {
            match json_mode {
                ModelInferenceRequestJsonMode::On => {
                    if messages
                        .iter()
                        .any(|msg| msg.content_contains_case_insensitive("json"))
                        || system.to_lowercase().contains("json")
                    {
                        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
                            content: Cow::Borrowed(system),
                        })
                    } else {
                        let formatted_instructions = format!("Respond using JSON.\n\n{system}");
                        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
                            content: Cow::Owned(formatted_instructions),
                        })
                    }
                }

                // If JSON mode is either off or strict, we don't need to do anything special
                _ => OpenAIRequestMessage::System(OpenAISystemRequestMessage {
                    content: Cow::Borrowed(system),
                }),
            }
            .into()
        }
        None => match *json_mode {
            ModelInferenceRequestJsonMode::On => {
                Some(OpenAIRequestMessage::System(OpenAISystemRequestMessage {
                    content: Cow::Owned("Respond using JSON.".to_string()),
                }))
            }
            _ => None,
        },
    }
}

pub(super) fn tensorzero_to_openai_messages(
    message: &RequestMessage,
) -> Result<Vec<OpenAIRequestMessage<'_>>, Error> {
    match message.role {
        Role::User => tensorzero_to_openai_user_messages(&message.content),
        Role::Assistant => tensorzero_to_openai_assistant_messages(&message.content),
    }
}

fn tensorzero_to_openai_user_messages(
    content_blocks: &[ContentBlock],
) -> Result<Vec<OpenAIRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the user content blocks.

    let mut messages = Vec::new();
    let mut user_content_blocks = Vec::new();

    for block in content_blocks.iter() {
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
            ContentBlock::Image(ImageWithPath {
                image,
                storage_path: _,
            }) => {
                user_content_blocks.push(OpenAIContentBlock::ImageUrl {
                    image_url: OpenAIImageUrl {
                        // This will only produce an error if we pass in a bad
                        // `Base64Image` (with missing image data)
                        url: format!("data:{};base64,{}", image.mime_type, image.data()?),
                    },
                });
            }
            ContentBlock::Thought(_) => {
                // OpenAI doesn't support thought blocks.
                // This can only happen if the thought block was generated by another model provider.
                // At this point, we can either convert the thought blocks to text or drop them.
                // We chose to drop them, because it's more consistent with the behavior that OpenAI expects.

                // TODO (#1361): test that this warning is logged when we drop thought blocks
                tracing::warn!(
                    "Dropping `thought` content block from user message. OpenAI does not support them."
                );
            }
            ContentBlock::Unknown {
                data,
                model_provider_name: _,
            } => {
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

fn tensorzero_to_openai_assistant_messages(
    content_blocks: &[ContentBlock],
) -> Result<Vec<OpenAIRequestMessage<'_>>, Error> {
    // We need to separate the tool result messages from the assistant content blocks.
    let mut assistant_content_blocks = Vec::new();
    let mut assistant_tool_calls = Vec::new();

    for block in content_blocks.iter() {
        match block {
            ContentBlock::Text(Text { text }) => {
                assistant_content_blocks.push(OpenAIContentBlock::Text {
                    text: Cow::Borrowed(text),
                });
            }
            ContentBlock::ToolCall(tool_call) => {
                let tool_call = OpenAIRequestToolCall {
                    id: &tool_call.id,
                    r#type: OpenAIToolType::Function,
                    function: OpenAIRequestFunctionCall {
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
            ContentBlock::Image(ImageWithPath {
                image,
                storage_path: _,
            }) => {
                assistant_content_blocks.push(OpenAIContentBlock::ImageUrl {
                    image_url: OpenAIImageUrl {
                        // This will only produce an error if we pass in a bad
                        // `Base64Image` (with missing image data)
                        url: format!("data:{};base64,{}", image.mime_type, image.data()?),
                    },
                });
            }
            ContentBlock::Thought(_) => {
                // OpenAI doesn't support thought blocks.
                // This can only happen if the thought block was generated by another model provider.
                // At this point, we can either convert the thought blocks to text or drop them.
                // We chose to drop them, because it's more consistent with the behavior that OpenAI expects.

                // TODO (#1361): test that this warning is logged when we drop thought blocks
                tracing::warn!(
                    "Dropping `thought` content block from assistant message. OpenAI does not support them."
                );
            }
            ContentBlock::Unknown {
                data,
                model_provider_name: _,
            } => {
                assistant_content_blocks.push(OpenAIContentBlock::Unknown {
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

    let message = OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
        content,
        tool_calls,
    });

    Ok(vec![message])
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
        json_mode: &ModelInferenceRequestJsonMode,
        output_schema: Option<&Value>,
        model: &str,
    ) -> Self {
        if model.contains("3.5") && *json_mode == ModelInferenceRequestJsonMode::Strict {
            return OpenAIResponseFormat::JsonObject;
        }

        match json_mode {
            ModelInferenceRequestJsonMode::On => OpenAIResponseFormat::JsonObject,
            ModelInferenceRequestJsonMode::Off => OpenAIResponseFormat::Text,
            ModelInferenceRequestJsonMode::Strict => match output_schema {
                Some(schema) => {
                    let json_schema = json!({"name": "response", "strict": true, "schema": schema});
                    OpenAIResponseFormat::JsonSchema { json_schema }
                }
                None => OpenAIResponseFormat::JsonObject,
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
pub(super) struct OpenAIFunction<'a> {
    pub(super) name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<&'a str>,
    pub parameters: &'a Value,
}

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct OpenAITool<'a> {
    pub(super) r#type: OpenAIToolType,
    pub(super) function: OpenAIFunction<'a>,
    pub(super) strict: bool,
}

impl<'a> From<&'a ToolConfig> for OpenAITool<'a> {
    fn from(tool: &'a ToolConfig) -> Self {
        OpenAITool {
            r#type: OpenAIToolType::Function,
            function: OpenAIFunction {
                name: tool.name(),
                description: Some(tool.description()),
                parameters: tool.parameters(),
            },
            strict: tool.strict(),
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
pub(super) struct SpecificToolFunction<'a> {
    pub(super) name: &'a str,
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
#[derive(Debug, Serialize)]
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
}

impl<'a> OpenAIRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<OpenAIRequest<'a>, Error> {
        let response_format = Some(OpenAIResponseFormat::new(
            &request.json_mode,
            request.output_schema,
            model,
        ));
        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };
        let mut messages = prepare_openai_messages(request)?;

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

        Ok(OpenAIRequest {
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
        })
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
        let body = OpenAIRequest::new(model, request)?;
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct OpenAIUsage {
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<OpenAIUsage> for Usage {
    fn from(usage: OpenAIUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenAIResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
pub(super) struct OpenAIResponseToolCall {
    id: String,
    r#type: OpenAIToolType,
    function: OpenAIResponseFunctionCall,
}

impl From<OpenAIResponseToolCall> for ToolCall {
    fn from(openai_tool_call: OpenAIResponseToolCall) -> Self {
        ToolCall {
            id: openai_tool_call.id,
            name: openai_tool_call.function.name,
            arguments: openai_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct OpenAIResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) content: Option<String>,
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
    request: serde_json::Value,
    generic_request: &'a ModelInferenceRequest<'a>,
    raw_response: String,
}

impl<'a> TryFrom<OpenAIResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: OpenAIResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let OpenAIResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            raw_response,
            generic_request,
        } = value;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
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
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
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
        }
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request body as JSON: {e}"),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(serde_json::to_string(&response).unwrap_or_default()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
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
    mut chunk: OpenAIChatChunk,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    tool_names: &mut Vec<String>,
) -> Result<ProviderInferenceResponseChunk, Error> {
    let raw_message = serde_json::to_string(&chunk).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error parsing response from OpenAI: {e}"),
            raw_request: None,
            raw_response: Some(serde_json::to_string(&chunk).unwrap_or_default()),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            raw_request: None,
            raw_response: Some(serde_json::to_string(&chunk).unwrap_or_default()),
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into());
    }
    let usage = chunk.usage.map(|u| u.into());
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
                let name = match tool_call.function.name {
                    Some(name) => {
                        tool_names.push(name.clone());
                        name
                    }
                    None => {
                        tool_names
                            .get(index as usize)
                            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                                message: "Tool call index out of bounds (meaning we haven't seen this many names in the stream)".to_string(),
                                raw_request: None,
                                raw_response: None,
                                provider_type: PROVIDER_TYPE.to_string(),
                            }))?
                            .clone()
                    }
                };
                content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                    id,
                    raw_name: name,
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
    input: &'a str,
}

impl<'a> OpenAIEmbeddingRequest<'a> {
    fn new(model: &'a str, input: &'a str) -> Self {
        Self { model, input }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    usage: OpenAIUsage,
}

struct OpenAIEmbeddingResponseWithMetadata<'a> {
    response: OpenAIEmbeddingResponse,
    latency: Latency,
    request: OpenAIEmbeddingRequest<'a>,
    raw_response: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
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
                message: format!("Error serializing request body as JSON: {e}"),
                raw_request: Some(serde_json::to_string(&request).unwrap_or_default()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        if response.data.len() != 1 {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: "Expected exactly one embedding in response".to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }));
        }
        let embedding = response
            .data
            .into_iter()
            .next()
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Expected exactly one embedding in response".to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?
            .embedding;

        Ok(EmbeddingProviderResponse::new(
            embedding,
            request.input.to_string(),
            raw_request,
            raw_response,
            response.usage.into(),
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
                message: format!("Error parsing response: {e}"),
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
struct OpenAIBatchRequestCounts {
    // total: u32,
    // completed: u32,
    // failed: u32,
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

    use std::borrow::Cow;

    use serde_json::json;

    use crate::{
        inference::{
            providers::test_helpers::{
                MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL, WEATHER_TOOL_CONFIG,
            },
            types::{FunctionType, RequestMessage},
        },
        tool::ToolCallConfig,
    };

    use super::*;

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
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, Some("Unauthorized access".to_string()));
        }

        // Test forbidden error
        let forbidden =
            handle_openai_error(StatusCode::FORBIDDEN, "Forbidden access", PROVIDER_TYPE);
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
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, Some("Forbidden access".to_string()));
        }

        // Test rate limit error
        let rate_limit = handle_openai_error(
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
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, Some("Rate limit exceeded".to_string()));
        }

        // Test server error
        let server_error = handle_openai_error(
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
            assert_eq!(*raw_request, None);
            assert_eq!(*raw_response, Some("Server error".to_string()));
        }
    }

    #[test]
    fn test_openai_request_new() {
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

        let openai_request = OpenAIRequest::new("gpt-3.5-turbo", &basic_request).unwrap();

        assert_eq!(openai_request.model, "gpt-3.5-turbo");
        assert_eq!(openai_request.messages.len(), 2);
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_completion_tokens, Some(100));
        assert_eq!(openai_request.seed, Some(69));
        assert_eq!(openai_request.top_p, Some(0.9));
        assert_eq!(openai_request.presence_penalty, Some(0.1));
        assert_eq!(openai_request.frequency_penalty, Some(0.2));
        assert!(openai_request.stream);
        assert_eq!(
            openai_request.response_format,
            Some(OpenAIResponseFormat::Text)
        );
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

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools).unwrap();

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
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
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

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools).unwrap();

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

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools).unwrap();

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

    #[test]
    fn test_openai_new_request_o1() {
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

        let openai_request = OpenAIRequest::new("o1-preview", &request).unwrap();

        assert_eq!(openai_request.model, "o1-preview");
        assert_eq!(openai_request.messages.len(), 1);
        assert!(!openai_request.stream);
        assert_eq!(
            openai_request.response_format,
            Some(OpenAIResponseFormat::Text)
        );
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

        let openai_request_with_system =
            OpenAIRequest::new("o1-mini", &request_with_system).unwrap();

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
        assert_eq!(
            openai_request_with_system.response_format,
            Some(OpenAIResponseFormat::Text)
        );
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
                    tool_calls: None,
                },
                finish_reason: OpenAIFinishReason::Stop,
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
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
            model: "gpt-3.5-turbo",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(OpenAIResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test_response".to_string();
        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithMetadata {
            response: valid_response,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            request: serde_json::to_value(&request_body).unwrap(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec!["Hello, world!".to_string().into()]
        );
        assert_eq!(inference_response.usage.input_tokens, 10);
        assert_eq!(inference_response.usage.output_tokens, 20);
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
                    tool_calls: Some(vec![OpenAIResponseToolCall {
                        id: "call1".to_string(),
                        r#type: OpenAIToolType::Function,
                        function: OpenAIResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
            }],
            usage: OpenAIUsage {
                prompt_tokens: 15,
                completion_tokens: 25,
                total_tokens: 40,
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
            model: "gpt-3.5-turbo",
            temperature: Some(0.5),
            top_p: Some(0.5),
            presence_penalty: Some(0.5),
            frequency_penalty: Some(0.5),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(OpenAIResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithMetadata {
            response: valid_response_with_tools,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(110),
            },
            request: serde_json::to_value(&request_body).unwrap(),
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
        assert_eq!(inference_response.usage.input_tokens, 15);
        assert_eq!(inference_response.usage.output_tokens, 25);
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
                prompt_tokens: 5,
                completion_tokens: 0,
                total_tokens: 5,
            },
        };
        let request_body = OpenAIRequest {
            messages: vec![],
            model: "gpt-3.5-turbo",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(OpenAIResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        };
        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithMetadata {
            response: invalid_response_no_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(120),
            },
            request: serde_json::to_value(&request_body).unwrap(),
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
                        tool_calls: None,
                    },
                    finish_reason: OpenAIFinishReason::Stop,
                },
                OpenAIResponseChoice {
                    index: 1,
                    finish_reason: OpenAIFinishReason::Stop,
                    message: OpenAIResponseMessage {
                        content: Some("Choice 2".to_string()),
                        tool_calls: None,
                    },
                },
            ],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
                total_tokens: 20,
            },
        };

        let request_body = OpenAIRequest {
            messages: vec![],
            model: "gpt-3.5-turbo",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_completion_tokens: Some(100),
            seed: Some(69),
            stream: false,
            response_format: Some(OpenAIResponseFormat::Text),
            stream_options: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        };
        let result = ProviderInferenceResponse::try_from(OpenAIResponseWithMetadata {
            response: invalid_response_multiple_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(130),
            },
            request: serde_json::to_value(&request_body).unwrap(),
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
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(tools[1].function.name, QUERY_TOOL.name());
        assert_eq!(tools[1].function.parameters, QUERY_TOOL.parameters());
        let tool_choice = tool_choice.unwrap();
        assert_eq!(
            tool_choice,
            OpenAIToolChoice::String(OpenAIToolChoiceString::Required)
        );
        let parallel_tool_calls = parallel_tool_calls.unwrap();
        assert!(parallel_tool_calls);
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: Some(true),
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

    #[test]
    fn test_tensorzero_to_openai_messages() {
        let content_blocks = vec!["Hello".to_string().into()];
        let openai_messages = tensorzero_to_openai_user_messages(&content_blocks).unwrap();
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
        let openai_messages = tensorzero_to_openai_user_messages(&content_blocks).unwrap();
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
        let openai_messages = tensorzero_to_openai_assistant_messages(&content_blocks).unwrap();
        assert_eq!(openai_messages.len(), 1);
        match &openai_messages[0] {
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
                    tool_calls: None,
                },
                finish_reason: Some(OpenAIFinishReason::Stop),
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let mut tool_call_names = vec!["name1".to_string()];
        let message = openai_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
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
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "id1".to_string(),
                raw_name: "name1".to_string(),
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
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
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
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
        )
        .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "id2".to_string(),
                raw_name: "name2".to_string(),
                raw_arguments: "{\"hello\":\"world\"}".to_string(),
            })]
        );
        assert_eq!(message.finish_reason, Some(FinishReason::Stop));
        // Check that the lists were updated
        assert_eq!(tool_call_ids, vec!["id1".to_string(), "id2".to_string()]);
        assert_eq!(
            tool_call_names,
            vec!["name1".to_string(), "name2".to_string()]
        );

        // Check a chunk with no choices and only usage
        // Test a correct new tool chunk
        let chunk = OpenAIChatChunk {
            choices: vec![],
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };
        let message = openai_to_tensorzero_chunk(
            chunk.clone(),
            Duration::from_millis(50),
            &mut tool_call_ids,
            &mut tool_call_names,
        )
        .unwrap();
        assert_eq!(message.content, vec![]);
        assert_eq!(
            message.usage,
            Some(Usage {
                input_tokens: 10,
                output_tokens: 20,
            })
        );
    }

    #[test]
    fn test_new_openai_response_format() {
        // Test JSON mode On
        let json_mode = ModelInferenceRequestJsonMode::On;
        let output_schema = None;
        let format = OpenAIResponseFormat::new(&json_mode, output_schema, "gpt-4o");
        assert_eq!(format, OpenAIResponseFormat::JsonObject);

        // Test JSON mode Off
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let format = OpenAIResponseFormat::new(&json_mode, output_schema, "gpt-4o");
        assert_eq!(format, OpenAIResponseFormat::Text);

        // Test JSON mode Strict with no schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let format = OpenAIResponseFormat::new(&json_mode, output_schema, "gpt-4o");
        assert_eq!(format, OpenAIResponseFormat::JsonObject);

        // Test JSON mode Strict with schema
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"type": "string"}
            }
        });
        let output_schema = Some(&schema);
        let format = OpenAIResponseFormat::new(&json_mode, output_schema, "gpt-4o");
        match format {
            OpenAIResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema["schema"], schema);
                assert_eq!(json_schema["name"], "response");
                assert_eq!(json_schema["strict"], true);
            }
            _ => panic!("Expected JsonSchema format"),
        }

        // Test JSON mode Strict with schema but gpt-3.5
        let json_mode = ModelInferenceRequestJsonMode::Strict;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "foo": {"type": "string"}
            }
        });
        let output_schema = Some(&schema);
        let format = OpenAIResponseFormat::new(&json_mode, output_schema, "gpt-3.5-turbo");
        assert_eq!(format, OpenAIResponseFormat::JsonObject);
    }

    #[test]
    fn test_openai_api_base() {
        assert_eq!(
            OPENAI_DEFAULT_BASE_URL.as_str(),
            "https://api.openai.com/v1/"
        );
    }

    #[test]
    fn test_tensorzero_to_openai_system_message() {
        // Test Case 1: system is None, json_mode is Off
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages: Vec<OpenAIRequestMessage> = vec![];
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, None);

        // Test Case 2: system is Some, json_mode is On, messages contain "json"
        let system = Some("System instructions");
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
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 3: system is Some, json_mode is On, messages do not contain "json"
        let system = Some("System instructions");
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
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 4: system is Some, json_mode is Off
        let system = Some("System instructions");
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
        let expected = Some(OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed("System instructions"),
        }));
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 5: system is Some, json_mode is Strict
        let system = Some("System instructions");
        let json_mode = ModelInferenceRequestJsonMode::Strict;
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
        let expected = Some(OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed("System instructions"),
        }));
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 6: system contains "json", json_mode is On
        let system = Some("Respond using JSON.\n\nSystem instructions");
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages = vec![OpenAIRequestMessage::User(OpenAIUserRequestMessage {
            content: vec![OpenAIContentBlock::Text {
                text: "Hello, how are you?".into(),
            }],
        })];
        let expected = Some(OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed("Respond using JSON.\n\nSystem instructions"),
        }));
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 7: system is None, json_mode is On
        let system = None;
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
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 8: system is None, json_mode is Strict
        let system = None;
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

        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert!(result.is_none());

        // Test Case 9: system is None, json_mode is On, with empty messages
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::On;
        let messages: Vec<OpenAIRequestMessage> = vec![];
        let expected = Some(OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Owned("Respond using JSON.".to_string()),
        }));
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);

        // Test Case 10: system is None, json_mode is Off, with messages containing "json"
        let system = None;
        let json_mode = ModelInferenceRequestJsonMode::Off;
        let messages = vec![OpenAIRequestMessage::User(OpenAIUserRequestMessage {
            content: vec![OpenAIContentBlock::Text {
                text: "Please include JSON in your response.".into(),
            }],
        })];
        let expected = None;
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);
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
            result.unwrap_err().get_owned_details(),
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
}
