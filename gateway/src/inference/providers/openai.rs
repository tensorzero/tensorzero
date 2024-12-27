use futures::stream::Stream;
use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest::multipart::{Form, Part};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::io::Write;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::embeddings::{EmbeddingProvider, EmbeddingProviderResponse, EmbeddingRequest};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    batch::{BatchProviderInferenceResponse, BatchStatus},
    ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
    RequestMessage, Role, Text, TextChunk, Usage,
};
use crate::model::{Credential, CredentialLocation};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

lazy_static! {
    static ref OPENAI_DEFAULT_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.openai.com/v1/").expect("Failed to parse OPENAI_DEFAULT_BASE_URL")
    };
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("OPENAI_API_KEY".to_string())
}

#[derive(Debug)]
pub struct OpenAIProvider {
    pub model_name: String,
    pub api_base: Option<Url>,
    pub credentials: OpenAICredentials,
}

impl OpenAIProvider {
    pub fn new(
        model_name: String,
        api_base: Option<Url>,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, "OpenAI"))?;
        let provider_credentials = OpenAICredentials::try_from(generic_credentials)?;
        Ok(OpenAIProvider {
            model_name,
            api_base,
            credentials: provider_credentials,
        })
    }
}

#[derive(Debug)]
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
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => {
                Ok(OpenAICredentials::None)
            },
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
                        provider_name: "OpenAI".to_string(),
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
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = OpenAIRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(self.api_base.as_ref())?;
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
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to OpenAI: {e}"),
                    status_code: e.status(),
                    provider_type: "OpenAI".to_string(),
                })
            })?;
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: "OpenAI".to_string(),
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: "OpenAI".to_string(),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(OpenAIResponseWithMetadata {
                response,
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
                        provider_type: "OpenAI".to_string(),
                    })
                })?,
            ))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
        if self.model_name.to_lowercase().starts_with("o1") {
            return Err(ErrorDetails::InvalidRequest {
                message: "The OpenAI o1 family of models does not support streaming.".to_string(),
            }
            .into());
        }
        let request_body = OpenAIRequest::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request: {e}"),
                provider_type: "OpenAI".to_string(),
            })
        })?;
        let request_url = get_chat_url(self.api_base.as_ref())?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json");
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
                    provider_type: "OpenAI".to_string(),
                })
            })?;

        let mut stream = Box::pin(stream_openai(event_source, start_time));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(ErrorDetails::InferenceServer {
                    message: "Stream ended before first chunk".to_string(),
                    provider_type: "OpenAI".to_string(),
                }
                .into())
            }
        };
        Ok((chunk, stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        requests: &'a [ModelInferenceRequest<'_>],
        client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<BatchProviderInferenceResponse, Error> {
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = get_files_url(self.api_base.as_ref())?;
        let inference_ids: Vec<Uuid> = requests.iter().map(|_| Uuid::now_v7()).collect();
        let batch_requests: Result<Vec<OpenAIBatchFileInput>, Error> = requests
            .iter()
            .zip(&inference_ids)
            .map(|(request, inference_id)| {
                OpenAIBatchFileInput::new(*inference_id, &self.model_name, request)
            })
            .collect();
        let batch_requests = batch_requests?;
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
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error serializing request: {e}"),
                    provider_type: "OpenAI".to_string(),
                })
            })?;
            jsonl_data.write_all(b"\n").map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error writing to JSONL: {e}"),
                    provider_type: "OpenAI".to_string(),
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
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error setting MIME type: {e}"),
                        provider_type: "OpenAI".to_string(),
                    })
                })?,
        );
        let mut request_builder = client.post(request_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let res = request_builder.multipart(form).send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Error sending request to OpenAI: {e}"),
                status_code: e.status(),
                provider_type: "OpenAI".to_string(),
            })
        })?;
        let response: OpenAIFileResponse = res.json().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing JSON response: {e}"),
                provider_type: "OpenAI".to_string(),
            })
        })?;
        let file_id = response.id;
        let batch_request = OpenAIBatchRequest::new(&file_id);
        let request_url = get_batch_url(self.api_base.as_ref())?;
        let mut request_builder = client.post(request_url);
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let res = request_builder
            .json(&batch_request)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to OpenAI: {e}"),
                    status_code: e.status(),
                    provider_type: "OpenAI".to_string(),
                })
            })?;
        let response: OpenAIBatchResponse = res.json().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing JSON response: {e}"),
                provider_type: "OpenAI".to_string(),
            })
        })?;
        Ok(BatchProviderInferenceResponse {
            batch_id: Uuid::now_v7(),
            inference_ids,
            batch_params: json!({"file_id": file_id, "batch_id": response.id}),
            raw_requests,
            status: BatchStatus::Pending,
        })
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
        let request_url = get_embedding_url(self.api_base.as_ref())?;
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
                    provider_type: "OpenAI".to_string(),
                })
            })?;
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: "OpenAI".to_string(),
                })
            })?;

            let response: OpenAIEmbeddingResponse =
                serde_json::from_str(&response).map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing JSON response: {e}: {response}"),
                        provider_type: "OpenAI".to_string(),
                    })
                })?;
            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

            Ok(OpenAIEmbeddingResponseWithMetadata {
                response,
                latency,
                request: request_body,
            }
            .try_into()?)
        } else {
            Err(handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing error response: {e}"),
                        provider_type: "OpenAI".to_string(),
                    })
                })?,
            ))
        }
    }
}

pub fn stream_openai(
    mut event_source: EventSource,
    start_time: Instant,
) -> impl Stream<Item = Result<ProviderInferenceResponseChunk, Error>> {
    let mut tool_call_ids = Vec::new();
    let mut tool_call_names = Vec::new();
    async_stream::stream! {
        let inference_id = Uuid::now_v7();
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(ErrorDetails::InferenceServer {
                        message: e.to_string(),
                        provider_type: "OpenAI".to_string(),
                    }.into());
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
                                    "Error parsing chunk. Error: {}, Data: {}",
                                    e, message.data
                                ),
                                provider_type: "OpenAI".to_string(),
                            }));

                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            openai_to_tensorzero_chunk(d, inference_id, latency, &mut tool_call_ids, &mut tool_call_names)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    }
}

pub(super) fn get_chat_url(base_url: Option<&Url>) -> Result<Url, Error> {
    let base_url = base_url.unwrap_or(&OPENAI_DEFAULT_BASE_URL);
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

fn get_files_url(base_url: Option<&Url>) -> Result<Url, Error> {
    let base_url = base_url.unwrap_or(&OPENAI_DEFAULT_BASE_URL);
    let mut url = base_url.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    url.join("files").map_err(|e| {
        Error::new(ErrorDetails::InvalidBaseUrl {
            message: e.to_string(),
        })
    })
}

fn get_batch_url(base_url: Option<&Url>) -> Result<Url, Error> {
    let base_url = base_url.unwrap_or(&OPENAI_DEFAULT_BASE_URL);
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

fn get_embedding_url(base_url: Option<&Url>) -> Result<Url, Error> {
    let base_url = base_url.unwrap_or(&OPENAI_DEFAULT_BASE_URL);
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

pub(super) fn handle_openai_error(response_code: StatusCode, response_body: &str) -> Error {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => ErrorDetails::InferenceClient {
            message: response_body.to_string(),
            status_code: Some(response_code),
            provider_type: "OpenAI".to_string(),
        }
        .into(),
        _ => ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            provider_type: "OpenAI".to_string(),
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
    content: Cow<'a, str>, // NOTE: this could be an array including images and stuff according to API spec (not supported yet)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenAIRequestFunctionCall<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenAIRequestToolCall<'a> {
    id: &'a str,
    r#type: OpenAIToolType,
    function: OpenAIRequestFunctionCall<'a>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIRequestToolCall<'a>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct OpenAIToolRequestMessage<'a> {
    content: &'a str,
    tool_call_id: &'a str,
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
    pub fn content(&self) -> Option<&str> {
        match self {
            OpenAIRequestMessage::System(msg) => Some(&msg.content),
            OpenAIRequestMessage::User(msg) => Some(&msg.content),
            OpenAIRequestMessage::Assistant(msg) => msg.content,
            OpenAIRequestMessage::Tool(msg) => Some(msg.content),
        }
    }
}

pub(super) fn prepare_openai_messages<'a>(
    request: &'a ModelInferenceRequest,
) -> Vec<OpenAIRequestMessage<'a>> {
    let mut messages: Vec<OpenAIRequestMessage> = request
        .messages
        .iter()
        .flat_map(tensorzero_to_openai_messages)
        .collect();
    if let Some(system_msg) = tensorzero_to_openai_system_message(
        request.system.as_deref(),
        &request.json_mode,
        &messages,
    ) {
        messages.insert(0, system_msg);
    }
    messages
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
            let parallel_tool_calls = Some(tool_config.parallel_tool_calls);
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
fn tensorzero_to_openai_system_message<'a>(
    system: Option<&'a str>,
    json_mode: &ModelInferenceRequestJsonMode,
    messages: &[OpenAIRequestMessage<'a>],
) -> Option<OpenAIRequestMessage<'a>> {
    match system {
        Some(system) => {
            match json_mode {
                ModelInferenceRequestJsonMode::On => {
                    if messages.iter().any(|msg| {
                        msg.content()
                            .map(|c| c.to_lowercase().contains("json"))
                            .unwrap_or(false)
                    }) || system.to_lowercase().contains("json")
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
) -> Vec<OpenAIRequestMessage<'_>> {
    let mut messages = Vec::new();

    for block in message.content.iter() {
        match block {
            ContentBlock::Text(Text { text }) => match message.role {
                Role::User => {
                    messages.push(OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                        content: Cow::Borrowed(text),
                    }));
                }
                Role::Assistant => {
                    messages.push(OpenAIRequestMessage::Assistant(
                        OpenAIAssistantRequestMessage {
                            content: Some(text),
                            tool_calls: None,
                        },
                    ));
                }
            },
            ContentBlock::ToolCall(tool_call) => {
                let tool_call = OpenAIRequestToolCall {
                    id: &tool_call.id,
                    r#type: OpenAIToolType::Function,
                    function: OpenAIRequestFunctionCall {
                        name: &tool_call.name,
                        arguments: &tool_call.arguments,
                    },
                };

                messages.push(OpenAIRequestMessage::Assistant(
                    OpenAIAssistantRequestMessage {
                        content: None,
                        tool_calls: Some(vec![tool_call]),
                    },
                ));
            }
            ContentBlock::ToolResult(tool_result) => {
                let message = OpenAIRequestMessage::Tool(OpenAIToolRequestMessage {
                    content: &tool_result.result,
                    tool_call_id: &tool_result.id,
                });
                messages.push(message);
            }
        }
    }

    messages
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
pub(super) enum OpenAIToolType {
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
        request: &'a ModelInferenceRequest,
    ) -> Result<OpenAIRequest<'a>, Error> {
        if model.to_lowercase().starts_with("o1") {
            return OpenAIRequest::new_o1(model, request);
        }
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
        let messages = prepare_openai_messages(request);

        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(request);
        Ok(OpenAIRequest {
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
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
        })
    }

    fn new_o1(
        model: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<OpenAIRequest<'a>, Error> {
        let response_format = None;
        let stream_options = None;
        let mut messages = prepare_openai_messages(request);
        if let Some(OpenAIRequestMessage::System(_)) = messages.first() {
            if let OpenAIRequestMessage::System(system_msg) = messages.remove(0) {
                let user_msg = OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                    content: system_msg.content,
                });
                messages.insert(0, user_msg);
            }
        }
        if request.tool_config.is_some() {
            return Err(ErrorDetails::InvalidRequest {
                message: "The OpenAI o1 family of models does not support tools.".to_string(),
            }
            .into());
        }
        let tools = None;
        let tool_choice = None;
        let parallel_tool_calls = None;

        // Temperature, top_p are fixed at 1
        // Presence and frequency penalty are fixed at 0
        // See [this guide](https://platform.openai.com/docs/guides/reasoning/quickstart) for more details
        Ok(OpenAIRequest {
            messages,
            model,
            temperature: None,
            max_tokens: request.max_tokens,
            seed: request.seed,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
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
    fn new(
        inference_id: Uuid,
        model: &'a str,
        request: &'a ModelInferenceRequest,
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
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    total_tokens: u32,
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

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct OpenAIResponseChoice {
    pub(super) index: u8,
    pub(super) message: OpenAIResponseMessage,
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
    request: OpenAIRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<OpenAIResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: OpenAIResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let OpenAIResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
        } = value;
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing response: {e}"),
                provider_type: "OpenAI".to_string(),
            })
        })?;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: "OpenAI".to_string(),
            }
            .into());
        }
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: "OpenAI".to_string(),
            }))?
            .message;
        let mut content: Vec<ContentBlock> = Vec::new();
        if let Some(text) = message.content {
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlock::ToolCall(tool_call.into()));
            }
        }
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request body as JSON: {e}"),
                provider_type: "OpenAI".to_string(),
            })
        })?;
        let system = generic_request.system.clone();
        let messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            content,
            system,
            messages,
            raw_request,
            raw_response,
            usage,
            latency,
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

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenAIChatChunkChoice {
    delta: OpenAIDelta,
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
    inference_id: Uuid,
    latency: Duration,
    tool_call_ids: &mut Vec<String>,
    tool_names: &mut Vec<String>,
) -> Result<ProviderInferenceResponseChunk, Error> {
    let raw_message = serde_json::to_string(&chunk).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error parsing response from OpenAI: {e}"),
            provider_type: "OpenAI".to_string(),
        })
    })?;
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            provider_type: "OpenAI".to_string(),
        }
        .into());
    }
    let usage = chunk.usage.map(|u| u.into());
    let mut content = vec![];
    if let Some(choice) = chunk.choices.pop() {
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
                                provider_type: "OpenAI".to_string(),
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
                                provider_type: "OpenAI".to_string(),
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
        inference_id,
        content,
        usage,
        raw_message,
        latency,
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
        } = response;
        let raw_request = serde_json::to_string(&request).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request body as JSON: {e}"),
                provider_type: "OpenAI".to_string(),
            })
        })?;
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing response from OpenAI: {e}"),
                provider_type: "OpenAI".to_string(),
            })
        })?;
        if response.data.len() != 1 {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: "Expected exactly one embedding in response".to_string(),
                provider_type: "OpenAI".to_string(),
            }));
        }
        let embedding = response
            .data
            .into_iter()
            .next()
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Expected exactly one embedding in response".to_string(),
                    provider_type: "OpenAI".to_string(),
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
    // errors: OpenAIBatchErrors,
    // input_file_id: String,
    // completion_window: String,
    // status: String,
    // output_file_id: String,
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
struct OpenAIBatchErrors {
    // object: String,
    // data: Vec<OpenAIBatchError>,
}

#[derive(Debug, Deserialize)]
struct OpenAIBatchError {
    // code: String,
    // message: String,
    // param: Option<String>,
    // line: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct OpenAIBatchRequestCounts {
    // total: u32,
    // completed: u32,
    // failed: u32,
}

#[cfg(test)]
mod tests {

    use std::borrow::Cow;

    use serde_json::json;

    use crate::{
        inference::{
            providers::common::{MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL, WEATHER_TOOL_CONFIG},
            types::FunctionType,
        },
        tool::ToolCallConfig,
    };

    use super::*;

    #[test]
    fn test_get_chat_url() {
        // Test with default URL
        let default_url = get_chat_url(None).unwrap();
        assert_eq!(
            default_url.as_str(),
            "https://api.openai.com/v1/chat/completions"
        );

        // Test with custom base URL
        let custom_base = "https://custom.openai.com/api/";
        let custom_url = get_chat_url(Some(&Url::parse(custom_base).unwrap())).unwrap();
        assert_eq!(
            custom_url.as_str(),
            "https://custom.openai.com/api/chat/completions"
        );

        // Test with URL without trailing slash
        let unjoinable_url = get_chat_url(Some(&Url::parse("https://example.com").unwrap()));
        assert!(unjoinable_url.is_ok());
        assert_eq!(
            unjoinable_url.unwrap().as_str(),
            "https://example.com/chat/completions"
        );
        // Test with URL that can't be joined
        let unjoinable_url = get_chat_url(Some(&Url::parse("https://example.com/foo").unwrap()));
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
        let unauthorized = handle_openai_error(StatusCode::UNAUTHORIZED, "Unauthorized access");
        let details = unauthorized.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Unauthorized access");
            assert_eq!(*status_code, Some(StatusCode::UNAUTHORIZED));
            assert_eq!(provider, "OpenAI");
        }

        // Test forbidden error
        let forbidden = handle_openai_error(StatusCode::FORBIDDEN, "Forbidden access");
        let details = forbidden.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Forbidden access");
            assert_eq!(*status_code, Some(StatusCode::FORBIDDEN));
            assert_eq!(provider, "OpenAI");
        }

        // Test rate limit error
        let rate_limit = handle_openai_error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded");
        let details = rate_limit.get_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Rate limit exceeded");
            assert_eq!(*status_code, Some(StatusCode::TOO_MANY_REQUESTS));
            assert_eq!(provider, "OpenAI")
        }

        // Test server error
        let server_error = handle_openai_error(StatusCode::INTERNAL_SERVER_ERROR, "Server error");
        let details = server_error.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        if let ErrorDetails::InferenceServer {
            message,
            provider_type: provider,
        } = details
        {
            assert_eq!(message, "Server error");
            assert_eq!(provider, "OpenAI");
        }
    }

    #[test]
    fn test_openai_request_new() {
        // Test basic request
        let basic_request = ModelInferenceRequest {
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
        };

        let openai_request = OpenAIRequest::new("gpt-3.5-turbo", &basic_request).unwrap();

        assert_eq!(openai_request.model, "gpt-3.5-turbo");
        assert_eq!(openai_request.messages.len(), 2);
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_tokens, Some(100));
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
        };

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools).unwrap();

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 2); // We'll add a system message containing Json to fit OpenAI requirements
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_tokens, None);
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
        };

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools).unwrap();

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 1);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_tokens, None);
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
        };

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools).unwrap();

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 1);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_tokens, None);
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
        };

        let openai_request = OpenAIRequest::new("o1-preview", &request).unwrap();

        assert_eq!(openai_request.model, "o1-preview");
        assert_eq!(openai_request.messages.len(), 1);
        assert!(!openai_request.stream);
        assert_eq!(openai_request.response_format, None);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_tokens, Some(100));
        assert_eq!(openai_request.seed, Some(69));
        assert_eq!(openai_request.top_p, None);
        assert_eq!(openai_request.presence_penalty, None);
        assert_eq!(openai_request.frequency_penalty, None);
        assert!(openai_request.tools.is_none());

        // Test case: System message is converted to User message
        let request_with_system = ModelInferenceRequest {
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
        };

        let openai_request_with_system =
            OpenAIRequest::new("o1-preview", &request_with_system).unwrap();

        // Check that the system message was converted to a user message
        assert_eq!(openai_request_with_system.messages.len(), 2);
        assert!(matches!(
            openai_request_with_system.messages[0],
            OpenAIRequestMessage::User(ref msg) if msg.content == "This is the system message"
        ));

        assert_eq!(openai_request_with_system.model, "o1-preview");
        assert!(!openai_request_with_system.stream);
        assert_eq!(openai_request_with_system.response_format, None);
        assert_eq!(openai_request_with_system.temperature, None);
        assert_eq!(openai_request_with_system.max_tokens, Some(100));
        assert_eq!(openai_request_with_system.seed, Some(69));
        assert!(openai_request_with_system.tools.is_none());
        assert_eq!(openai_request_with_system.top_p, None);
        assert_eq!(openai_request_with_system.presence_penalty, None);
        assert_eq!(openai_request_with_system.frequency_penalty, None);

        // Test case: Tool config errors on O1 models
        let request_with_tools = ModelInferenceRequest {
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
            tool_config: Some(Cow::Owned(ToolCallConfig {
                tools_available: vec![],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: false,
            })),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let openai_request_with_tools = OpenAIRequest::new("o1-preview", &request_with_tools);

        // Check that it returns an error
        assert!(openai_request_with_tools.is_err());
        let err = openai_request_with_tools.unwrap_err();
        let details = err.get_details();
        if let ErrorDetails::InvalidRequest { message } = details {
            assert_eq!(
                message,
                "The OpenAI o1 family of models does not support tools."
            );
        } else {
            panic!("Expected InvalidRequest error");
        }
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
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        };
        let generic_request = ModelInferenceRequest {
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
        };

        let request_body = OpenAIRequest {
            messages: vec![],
            model: "gpt-3.5-turbo",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
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
            response: valid_response,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            request: request_body,
            generic_request: &generic_request,
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec!["Hello, world!".to_string().into()]
        );
        assert_eq!(inference_response.usage.input_tokens, 10);
        assert_eq!(inference_response.usage.output_tokens, 20);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(100)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
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
        };

        let request_body = OpenAIRequest {
            messages: vec![],
            model: "gpt-3.5-turbo",
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
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
            request: request_body,
            generic_request: &generic_request,
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.output,
            vec![ContentBlock::ToolCall(ToolCall {
                id: "call1".to_string(),
                name: "test_function".to_string(),
                arguments: "{}".to_string(),
            })]
        );
        assert_eq!(inference_response.usage.input_tokens, 15);
        assert_eq!(inference_response.usage.output_tokens, 25);
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_millis(110)
            }
        );
        assert_eq!(inference_response.raw_request, raw_request);
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
            max_tokens: Some(100),
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
            request: request_body,
            generic_request: &generic_request,
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
                },
                OpenAIResponseChoice {
                    index: 1,
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
            max_tokens: Some(100),
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
            request: request_body,
            generic_request: &generic_request,
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        let details = err.get_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
    }

    #[test]
    fn test_prepare_openai_tools() {
        let request_with_tools = ModelInferenceRequest {
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
            parallel_tool_calls: true,
        };

        // Test no tools but a tool choice and make sure tool choice output is None
        let request_without_tools = ModelInferenceRequest {
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
        };
        let (tools, tool_choice, parallel_tool_calls) =
            prepare_openai_tools(&request_without_tools);
        assert!(tools.is_none());
        assert!(tool_choice.is_none());
        assert!(parallel_tool_calls.is_none());
    }

    #[test]
    fn test_tensorzero_to_openai_messages() {
        let simple_request_message = RequestMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into()],
        };
        let openai_messages = tensorzero_to_openai_messages(&simple_request_message);
        assert_eq!(openai_messages.len(), 1);
        match &openai_messages[0] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(content.content, "Hello");
            }
            _ => panic!("Expected a user message"),
        }

        // Message with multiple blocks
        let multi_block_message = RequestMessage {
            role: Role::User,
            content: vec![
                "Hello".to_string().into(),
                "How are you?".to_string().into(),
            ],
        };
        let openai_messages = tensorzero_to_openai_messages(&multi_block_message);
        assert_eq!(openai_messages.len(), 2);
        match &openai_messages[0] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(content.content, "Hello");
            }
            _ => panic!("Expected a user message"),
        }
        match &openai_messages[1] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(content.content, "How are you?");
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
        let multi_block_message = RequestMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into(), tool_block],
        };
        let openai_messages = tensorzero_to_openai_messages(&multi_block_message);
        assert_eq!(openai_messages.len(), 2);
        match &openai_messages[0] {
            OpenAIRequestMessage::User(content) => {
                assert_eq!(content.content, "Hello");
            }
            _ => panic!("Expected a user message"),
        }
        match &openai_messages[1] {
            OpenAIRequestMessage::Assistant(content) => {
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
            }],
            usage: None,
        };
        let mut tool_call_ids = vec!["id1".to_string()];
        let mut tool_call_names = vec!["name1".to_string()];
        let inference_id = Uuid::now_v7();
        let message = openai_to_tensorzero_chunk(
            chunk.clone(),
            inference_id,
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
        // Test what an intermediate tool chunk should look like
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
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
            inference_id,
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
        // Test what a bad tool chunk would do (new ID but no names)
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
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
            inference_id,
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
                provider_type: "OpenAI".to_string(),
            }
        );
        // Test a correct new tool chunk
        let chunk = OpenAIChatChunk {
            choices: vec![OpenAIChatChunkChoice {
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
            inference_id,
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
            inference_id,
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
                content: Cow::Borrowed("Please respond in JSON format."),
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some("Sure, here is the data."),
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
                content: Cow::Borrowed("Hello, how are you?"),
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some("I am fine, thank you!"),
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
                content: Cow::Borrowed("Hello, how are you?"),
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some("I am fine, thank you!"),
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
                content: Cow::Borrowed("Hello, how are you?"),
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some("I am fine, thank you!"),
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
            content: Cow::Borrowed("Hello, how are you?"),
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
                content: Cow::Borrowed("Tell me a joke."),
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some("Sure, here's one for you."),
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
                content: Cow::Borrowed("Provide a summary of the news."),
            }),
            OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                content: Some("Here's the summary."),
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
            content: Cow::Borrowed("Please include JSON in your response."),
        })];
        let expected = None;
        let result = tensorzero_to_openai_system_message(system, &json_mode, &messages);
        assert_eq!(result, expected);
    }
}
