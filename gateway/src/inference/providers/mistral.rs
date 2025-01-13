use std::{borrow::Cow, time::Duration};

use futures::stream::Stream;
use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::types::{
        batch::{BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse},
        ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequest,
        ModelInferenceRequestJsonMode, ProviderInferenceResponse, ProviderInferenceResponseChunk,
        ProviderInferenceResponseStream, TextChunk, Usage,
    },
    model::{Credential, CredentialLocation},
    tool::{ToolCall, ToolCallChunk, ToolChoice},
};

use super::{
    openai::{
        get_chat_url, tensorzero_to_openai_messages, OpenAIFunction, OpenAIRequestMessage,
        OpenAISystemRequestMessage, OpenAITool, OpenAIToolType,
    },
    provider_trait::InferenceProvider,
};

lazy_static! {
    static ref MISTRAL_API_BASE: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.mistral.ai/v1/").expect("Failed to parse MISTRAL_API_BASE")
    };
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("MISTRAL_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "Mistral";
const PROVIDER_TYPE: &str = "mistral";

#[derive(Debug)]
pub struct MistralProvider {
    pub model_name: String,
    pub credentials: MistralCredentials,
}

impl MistralProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, PROVIDER_TYPE))?;
        let provider_credentials = MistralCredentials::try_from(generic_credentials)?;
        Ok(MistralProvider {
            model_name,
            credentials: provider_credentials,
        })
    }
}

#[derive(Debug)]
pub enum MistralCredentials {
    Static(SecretString),
    Dynamic(String),
    #[cfg(any(test, feature = "e2e_tests"))]
    None,
}

impl TryFrom<Credential> for MistralCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(MistralCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(MistralCredentials::Dynamic(key_name)),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(MistralCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Mistral provider".to_string(),
            })),
        }
    }
}

impl MistralCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            MistralCredentials::Static(api_key) => Ok(api_key),
            MistralCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                })
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            MistralCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            }
            .into()),
        }
    }
}

impl InferenceProvider for MistralProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = MistralRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(Some(&MISTRAL_API_BASE))?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to Mistral: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(response.clone()),
                })
            })?;

            MistralResponseWithMetadata {
                response,
                latency,
                request: request_body,
                generic_request: request,
            }
            .try_into()
        } else {
            handle_mistral_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing error response: {e}"),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                        raw_response: None,
                    })
                })?,
            )
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
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
        let request_body = MistralRequest::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let request_url = get_chat_url(Some(&MISTRAL_API_BASE))?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to Mistral: {e}"),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
        let mut stream = Box::pin(stream_mistral(event_source, start_time));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::new(ErrorDetails::InferenceServer {
                    message: "Stream ended before first chunk".to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                }))
            }
        };
        Ok((chunk, stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Mistral".to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

fn handle_mistral_error(
    response_code: StatusCode,
    response_body: &str,
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => Err(ErrorDetails::InferenceClient {
            message: response_body.to_string(),
            status_code: Some(response_code),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into()),
        _ => Err(ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into()),
    }
}

pub fn stream_mistral(
    mut event_source: EventSource,
    start_time: Instant,
) -> impl Stream<Item = Result<ProviderInferenceResponseChunk, Error>> {
    async_stream::stream! {
        let inference_id = Uuid::now_v7();
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    yield Err(ErrorDetails::InferenceServer {
                        message: e.to_string(),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: None,
                    }.into());
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<MistralChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing chunk. Error: {}, Data: {}",
                                    e, message.data
                                ),
                                provider_type: PROVIDER_TYPE.to_string(),
                                raw_request: None,
                                raw_response: None,
                            }.into());
                        let latency = start_time.elapsed();
                        let stream_message = data.and_then(|d| {
                            mistral_to_tensorzero_chunk(d, inference_id, latency)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    }
}

pub(super) fn prepare_mistral_messages<'a>(
    request: &'a ModelInferenceRequest,
) -> Vec<OpenAIRequestMessage<'a>> {
    let mut messages: Vec<OpenAIRequestMessage> = request
        .messages
        .iter()
        .flat_map(tensorzero_to_openai_messages)
        .collect();
    if let Some(system_msg) = tensorzero_to_mistral_system_message(request.system.as_deref()) {
        messages.insert(0, system_msg);
    }
    messages
}

fn tensorzero_to_mistral_system_message(system: Option<&str>) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum MistralResponseFormat {
    JsonObject,
    #[default]
    Text,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum MistralToolChoice {
    Auto,
    None,
    Any,
}

#[derive(Debug, PartialEq, Serialize)]
struct MistralTool<'a> {
    r#type: OpenAIToolType,
    function: OpenAIFunction<'a>,
}

impl<'a> From<OpenAITool<'a>> for MistralTool<'a> {
    fn from(tool: OpenAITool<'a>) -> Self {
        MistralTool {
            r#type: tool.r#type,
            function: tool.function,
        }
    }
}

fn prepare_mistral_tools<'a>(
    request: &'a ModelInferenceRequest<'a>,
) -> Result<(Option<Vec<MistralTool<'a>>>, Option<MistralToolChoice>), Error> {
    match &request.tool_config {
        None => Ok((None, None)),
        Some(tool_config) => match &tool_config.tool_choice {
            ToolChoice::Specific(tool_name) => {
                let tool = tool_config
                    .tools_available
                    .iter()
                    .find(|t| t.name() == tool_name)
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::ToolNotFound {
                            name: tool_name.clone(),
                        })
                    })?;
                let tools = vec![MistralTool::from(OpenAITool::from(tool))];
                Ok((Some(tools), Some(MistralToolChoice::Any)))
            }
            ToolChoice::Auto | ToolChoice::Required => {
                let tools = tool_config
                    .tools_available
                    .iter()
                    .map(|t| MistralTool::from(OpenAITool::from(t)))
                    .collect();
                let tool_choice = match tool_config.tool_choice {
                    ToolChoice::Auto => MistralToolChoice::Auto,
                    ToolChoice::Required => MistralToolChoice::Any,
                    _ => {
                        return Err(ErrorDetails::InvalidTool {
                            message: "Tool choice must be Auto or Required. This is impossible."
                                .to_string(),
                        }
                        .into())
                    }
                };
                Ok((Some(tools), Some(tool_choice)))
            }
            ToolChoice::None => Ok((None, Some(MistralToolChoice::None))),
        },
    }
}

/// This struct defines the supported parameters for the Mistral inference API
/// See the [Mistral API documentation](https://docs.mistral.ai/api/#tag/chat)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, service_tier, stop, user,
/// or context_length_exceeded_behavior.
/// NOTE: Mistral does not support seed.
#[derive(Debug, Serialize)]
struct MistralRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    random_seed: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<MistralResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MistralTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<MistralToolChoice>,
}

impl<'a> MistralRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<MistralRequest<'a>, Error> {
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(MistralResponseFormat::JsonObject)
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_mistral_messages(request);
        let (tools, tool_choice) = prepare_mistral_tools(request)?;

        Ok(MistralRequest {
            messages,
            model,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            random_seed: request.seed,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct MistralUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<MistralUsage> for Usage {
    fn from(usage: MistralUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct MistralResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct MistralResponseToolCall {
    id: String,
    function: MistralResponseFunctionCall,
}

impl From<MistralResponseToolCall> for ToolCall {
    fn from(mistral_tool_call: MistralResponseToolCall) -> Self {
        ToolCall {
            id: mistral_tool_call.id,
            name: mistral_tool_call.function.name,
            arguments: mistral_tool_call.function.arguments,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralResponseToolCall>>,
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct MistralResponseChoice {
    index: u8,
    message: MistralResponseMessage,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct MistralResponse {
    choices: Vec<MistralResponseChoice>,
    usage: MistralUsage,
}

struct MistralResponseWithMetadata<'a> {
    response: MistralResponse,
    latency: Latency,
    request: MistralRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<MistralResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: MistralResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let MistralResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
        } = value;
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error parsing response: {e}"),
            })
        })?;
        if response.choices.len() != 1 {
            return Err(Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_response.clone()),
            }));
        }
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_response.clone())
            }))?
            .message;
        let mut content: Vec<ContentBlock> = Vec::new();
        if let Some(text) = message.content {
            if !text.is_empty() {
                content.push(text.into());
            }
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlock::ToolCall(tool_call.into()));
            }
        }
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request body as JSON: {e}"),
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
struct MistralFunctionCallChunk {
    name: String,
    arguments: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralToolCallChunk {
    // #[serde(skip_serializing_if = "Option::is_none")]
    id: String,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: MistralFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralToolCallChunk>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralChatChunkChoice {
    delta: MistralDelta,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MistralChatChunk {
    choices: Vec<MistralChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<MistralUsage>,
}

/// Maps a Mistral chunk to a TensorZero chunk for streaming inferences
fn mistral_to_tensorzero_chunk(
    mut chunk: MistralChatChunk,
    inference_id: Uuid,
    latency: Duration,
) -> Result<ProviderInferenceResponseChunk, Error> {
    let raw_message = serde_json::to_string(&chunk).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Error parsing response from Mistral: {e}"),
        })
    })?;
    if chunk.choices.len() > 1 {
        return Err(ErrorDetails::InferenceServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: Some(raw_message.clone()),
        }
        .into());
    }
    let usage = chunk.usage.map(|u| u.into());
    let mut content = vec![];
    if let Some(choice) = chunk.choices.pop() {
        if let Some(text) = choice.delta.content {
            if !text.is_empty() {
                content.push(ContentBlockChunk::Text(TextChunk {
                    text,
                    id: "0".to_string(),
                }));
            }
        }
        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: tool_call.id,
                    raw_name: tool_call.function.name,
                    raw_arguments: tool_call.function.arguments,
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;

    use super::*;

    use crate::inference::providers::common::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::types::{FunctionType, RequestMessage, Role};

    #[test]
    fn test_mistral_request_new() {
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let mistral_request =
            MistralRequest::new("mistral-small-latest", &request_with_tools).unwrap();

        assert_eq!(mistral_request.model, "mistral-small-latest");
        assert_eq!(mistral_request.messages.len(), 1);
        assert_eq!(mistral_request.temperature, Some(0.5));
        assert_eq!(mistral_request.max_tokens, Some(100));
        assert!(!mistral_request.stream);
        assert_eq!(
            mistral_request.response_format,
            Some(MistralResponseFormat::JsonObject)
        );
        assert!(mistral_request.tools.is_some());
        let tools = mistral_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(mistral_request.tool_choice, Some(MistralToolChoice::Any));
    }

    #[test]
    fn test_try_from_mistral_response() {
        // Test case 1: Valid response with content
        let valid_response = MistralResponse {
            choices: vec![MistralResponseChoice {
                index: 0,
                message: MistralResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                },
            }],
            usage: MistralUsage {
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
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
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
        let valid_response_with_tools = MistralResponse {
            choices: vec![MistralResponseChoice {
                index: 0,
                message: MistralResponseMessage {
                    content: None,
                    tool_calls: Some(vec![MistralResponseToolCall {
                        id: "call1".to_string(),
                        function: MistralResponseFunctionCall {
                            name: "test_function".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
            }],
            usage: MistralUsage {
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
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
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
        let invalid_response_no_choices = MistralResponse {
            choices: vec![],
            usage: MistralUsage {
                prompt_tokens: 5,
                completion_tokens: 0,
                total_tokens: 5,
            },
        };
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
        };
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: invalid_response_no_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(120),
            },
            request: request_body,
            generic_request: &generic_request,
        });
        let details = result.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        // Test case 4: Invalid response with multiple choices
        let invalid_response_multiple_choices = MistralResponse {
            choices: vec![
                MistralResponseChoice {
                    index: 0,
                    message: MistralResponseMessage {
                        content: Some("Choice 1".to_string()),
                        tool_calls: None,
                    },
                },
                MistralResponseChoice {
                    index: 1,
                    message: MistralResponseMessage {
                        content: Some("Choice 2".to_string()),
                        tool_calls: None,
                    },
                },
            ],
            usage: MistralUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
                total_tokens: 20,
            },
        };
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: false,
            response_format: Some(MistralResponseFormat::JsonObject),
            tools: None,
            tool_choice: None,
        };
        let result = ProviderInferenceResponse::try_from(MistralResponseWithMetadata {
            response: invalid_response_multiple_choices,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(130),
            },
            request: request_body,
            generic_request: &generic_request,
        });
        let details = result.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
    }

    #[test]
    fn test_handle_mistral_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_mistral_error(StatusCode::UNAUTHORIZED, "Unauthorized access");
        let details = unauthorized.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(message, "Unauthorized access");
            assert_eq!(status_code, Some(StatusCode::UNAUTHORIZED));
            assert_eq!(provider, PROVIDER_TYPE.to_string());
            assert_eq!(raw_request, None);
            assert_eq!(raw_response, None);
        }

        // Test forbidden error
        let forbidden = handle_mistral_error(StatusCode::FORBIDDEN, "Forbidden access");
        let details = forbidden.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(message, "Forbidden access");
            assert_eq!(status_code, Some(StatusCode::FORBIDDEN));
            assert_eq!(provider, PROVIDER_TYPE.to_string());
            assert_eq!(raw_request, None);
            assert_eq!(raw_response, None);
        }

        // Test rate limit error
        let rate_limit = handle_mistral_error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded");
        let details = rate_limit.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceClient { .. }));
        if let ErrorDetails::InferenceClient {
            message,
            status_code,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(message, "Rate limit exceeded");
            assert_eq!(status_code, Some(StatusCode::TOO_MANY_REQUESTS));
            assert_eq!(provider, PROVIDER_TYPE.to_string());
            assert_eq!(raw_request, None);
            assert_eq!(raw_response, None);
        }

        // Test server error
        let server_error = handle_mistral_error(StatusCode::INTERNAL_SERVER_ERROR, "Server error");
        let details = server_error.unwrap_err().get_owned_details();
        assert!(matches!(details, ErrorDetails::InferenceServer { .. }));
        if let ErrorDetails::InferenceServer {
            message,
            provider_type: provider,
            raw_request,
            raw_response,
        } = details
        {
            assert_eq!(message, "Server error");
            assert_eq!(provider, PROVIDER_TYPE.to_string());
            assert_eq!(raw_request, None);
            assert_eq!(raw_response, None);
        }
    }

    #[test]
    fn test_mistral_api_base() {
        assert_eq!(MISTRAL_API_BASE.as_str(), "https://api.mistral.ai/v1/");
    }

    #[test]
    fn test_credential_to_mistral_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: MistralCredentials = MistralCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MistralCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = MistralCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, MistralCredentials::Dynamic(_)));

        // Test Missing credential (test mode)
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            let generic = Credential::Missing;
            let creds = MistralCredentials::try_from(generic).unwrap();
            assert!(matches!(creds, MistralCredentials::None));
        }

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = MistralCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
}
