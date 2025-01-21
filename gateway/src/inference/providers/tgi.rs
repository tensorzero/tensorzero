/// TGI integration for TensorZero
///
/// Here, we list known limitations of TGI in our experience
///
/// First, TGI doesn't handle streaming tool calls correctly. It mangles the tool name field entirely and also doesn't catch EOS tokens properly.
/// Second, TGI doesn't handle multiple tools in a single request, because it doesn't return correct tool names. See the docs [here](https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/using_guidance#the-tools-parameter)
/// for an example.
/// Third, TGI doesn't support tool responses being sent back at all.
/// Fourth, TGI only supports JSON mode through a tool call. Luckily, we do this out of the box with `implicit_tool` as the json mode
///
/// In light of this, we have decided to not explicitly support tool calling for TGI and only support JSON mode via `implicit_tool`.
/// Our implementation currently allows you to use a tool in TGI (nonstreaming), but YMMV.
use futures::{Stream, StreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use super::openai::{
    get_chat_url, prepare_openai_messages, prepare_openai_tools, OpenAIRequestMessage, OpenAITool,
    OpenAIToolChoice, OpenAIToolType, StreamOptions,
};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::batch::{
    BatchRequestRow, PollBatchInferenceResponse, StartBatchProviderInferenceResponse,
};
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
    TextChunk, Usage,
};
use crate::model::{Credential, CredentialLocation};
use crate::tool::ToolCall;

const PROVIDER_NAME: &str = "TGI";
const PROVIDER_TYPE: &str = "tgi";

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("TGI_API_KEY".to_string())
}

#[derive(Debug)]
pub struct TGIProvider {
    api_base: Url,
    credentials: TGICredentials,
}

impl TGIProvider {
    pub fn new(api_base: Url, api_key_location: Option<CredentialLocation>) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, PROVIDER_TYPE))?;
        let provider_credentials = TGICredentials::try_from(generic_credentials)?;
        Ok(TGIProvider {
            api_base,
            credentials: provider_credentials,
        })
    }
}

#[derive(Debug)]
pub enum TGICredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for TGICredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(TGICredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(TGICredentials::Dynamic(key_name)),
            Credential::None => Ok(TGICredentials::None),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(TGICredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for TGI provider".to_string(),
            })),
        }
    }
}

impl TGICredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            TGICredentials::Static(api_key) => Ok(Some(api_key)),
            TGICredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                }))
                .transpose()
            }
            TGICredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for TGIProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        // TGI doesn't care about this field, so we can hardcode it to "tgi"
        let model_name = PROVIDER_TYPE.to_string();
        let request_body = TGIRequest::new(&model_name, request)?;
        let request_url = get_chat_url(&self.api_base)?;
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
                    message: format!("Error sending request to TGI: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: serde_json::to_string(&request_body).ok(),
                    raw_response: None,
                })
            })?;

        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: serde_json::to_string(&request_body).ok(),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: serde_json::to_string(&request_body).ok(),
                    raw_response: Some(response.clone()),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(TGIResponseWithMetadata {
                response,
                latency,
                request: request_body,
                generic_request: request,
            }
            .try_into()?)
        } else {
            Err(handle_tgi_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing error response: {e}"),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: serde_json::to_string(&request_body).ok(),
                        raw_response: None,
                    })
                })?,
            ))
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
        let model_name = PROVIDER_NAME.to_lowercase().clone();
        let request_body = TGIRequest::new(&model_name, request)?;
        // TGI integration does not support tools in streaming mode
        if request_body.tools.is_some() {
            return Err(ErrorDetails::InvalidTool {
                message: "TGI does not support tools in streaming mode".to_string(),
            }
            .into());
        }
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let request_url = get_chat_url(&self.api_base)?;
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
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: serde_json::to_string(&request_body).ok(),
                    raw_response: None,
                })
            })?;

        let mut stream = Box::pin(stream_tgi(event_source, start_time));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(ErrorDetails::InferenceServer {
                    message: "Stream ended before first chunk".to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: serde_json::to_string(&request_body).ok(),
                    raw_response: None,
                }
                .into())
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
            provider_type: PROVIDER_TYPE.to_string(),
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

fn stream_tgi(
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
                        raw_request: None,
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    }.into());
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<TGIChatChunk, Error> =
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
                            tgi_to_tensorzero_chunk(d, inference_id, latency)
                        });
                        yield stream_message;
                    }
                },
            }
        }

        event_source.close();
    }
}

/// This struct defines the supported parameters for the TGI API
/// See the [TGI documentation](https://huggingface.co/docs/text-generation-inference/en/reference/api_reference#openai-messages-api)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Serialize)]
struct TGIRequest<'a> {
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
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> TGIRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<TGIRequest<'a>, Error> {
        // TGI doesn't support JSON mode at all (only through tools [https://huggingface.co/docs/text-generation-inference/en/conceptual/guidance])
        // So we log a warning and ignore the JSON mode
        // You can get JSON mode through `implicit_tool` instead.
        if request.json_mode != ModelInferenceRequestJsonMode::Off {
            tracing::warn!("TGI does not support JSON mode. Ignoring JSON mode.");
        }

        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };

        let messages = prepare_openai_messages(request);

        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(request);

        Ok(TGIRequest {
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
            tools,
            tool_choice,
            parallel_tool_calls,
        })
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenAIRequestToolCall<'a> {
    id: &'a str,
    r#type: OpenAIToolType,
    function: OpenAIRequestFunctionCall<'a>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenAIRequestFunctionCall<'a> {
    name: &'a str,
    arguments: &'a str,
}

struct TGIResponseWithMetadata<'a> {
    response: TGIResponse,
    latency: Latency,
    request: TGIRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<TGIResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: TGIResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let TGIResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
        } = value;
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing response: {e}"),
            })
        })?;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: serde_json::to_string(&request_body).ok(),
                raw_response: Some(raw_response),
            }
            .into());
        }
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: serde_json::to_string(&request_body).ok(),
                raw_response: Some(raw_response.clone()),
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

pub(super) fn handle_tgi_error(response_code: StatusCode, response_body: &str) -> Error {
    match response_code {
        StatusCode::BAD_REQUEST
        | StatusCode::UNAUTHORIZED
        | StatusCode::FORBIDDEN
        | StatusCode::TOO_MANY_REQUESTS => ErrorDetails::InferenceClient {
            message: response_body.to_string(),
            status_code: Some(response_code),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into(),
        _ => ErrorDetails::InferenceServer {
            message: response_body.to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into(),
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct TGIUsage {
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<TGIUsage> for Usage {
    fn from(usage: TGIUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct TGIResponseFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct TGIResponseToolCall {
    id: String,
    r#type: OpenAIToolType,
    function: TGIResponseFunctionCall,
}

impl From<TGIResponseToolCall> for ToolCall {
    fn from(tgi_tool_call: TGIResponseToolCall) -> Self {
        ToolCall {
            id: tgi_tool_call.id,
            name: tgi_tool_call.function.name,
            arguments: serde_json::to_string(&tgi_tool_call.function.arguments).unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIResponseMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<TGIResponseToolCall>>,
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct TGIResponseChoice {
    index: u8,
    message: TGIResponseMessage,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct TGIResponse {
    choices: Vec<TGIResponseChoice>,
    usage: TGIUsage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIFunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIToolCallChunk {
    index: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    // NOTE: these are externally tagged enums, for now we're gonna just keep this hardcoded as there's only one option
    // If we were to do this better, we would need to check the `type` field
    function: TGIFunctionCallChunk,
}

// This doesn't include role
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<TGIToolCallChunk>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIChatChunkChoice {
    delta: TGIDelta,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct TGIChatChunk {
    choices: Vec<TGIChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<TGIUsage>,
}

/// Maps an TGI chunk to a TensorZero chunk for streaming inferences
fn tgi_to_tensorzero_chunk(
    mut chunk: TGIChatChunk,
    inference_id: Uuid,
    latency: Duration,
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
    if let Some(choice) = chunk.choices.pop() {
        if let Some(text) = choice.delta.content {
            content.push(ContentBlockChunk::Text(TextChunk {
                text,
                id: "0".to_string(),
            }));
        }
        if let Some(_tool_calls) = choice.delta.tool_calls {
            return Err(ErrorDetails::InferenceServer {
                message: "TGI returned a tool call but we don't make streaming tool call requests for this provider"
                    .to_string(),
                raw_request: None,
                raw_response: Some(raw_message),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
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

    use serde_json::json;

    use crate::inference::{
        providers::{
            common::{WEATHER_TOOL, WEATHER_TOOL_CONFIG},
            openai::{OpenAIToolType, SpecificToolChoice, SpecificToolFunction},
        },
        types::{FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role},
    };

    use super::*;

    #[test]
    fn test_tgi_request_new() {
        let model_name = PROVIDER_TYPE.to_string();
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
        let tgi_request = TGIRequest::new(&model_name, &basic_request).unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 2);
        assert_eq!(tgi_request.temperature, Some(0.7));
        assert_eq!(tgi_request.max_tokens, Some(100));
        assert_eq!(tgi_request.seed, Some(69));
        assert_eq!(tgi_request.top_p, Some(0.9));
        assert_eq!(tgi_request.presence_penalty, Some(0.1));
        assert_eq!(tgi_request.frequency_penalty, Some(0.2));
        assert!(tgi_request.stream);
        assert!(tgi_request.tools.is_none());
        assert_eq!(tgi_request.tool_choice, None);
        assert!(tgi_request.parallel_tool_calls.is_none());

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

        let tgi_request = TGIRequest::new(&model_name, &request_with_tools).unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 2);
        assert_eq!(tgi_request.temperature, None);
        assert_eq!(tgi_request.max_tokens, None);
        assert_eq!(tgi_request.seed, None);
        assert_eq!(tgi_request.top_p, None);
        assert_eq!(tgi_request.presence_penalty, None);
        assert_eq!(tgi_request.frequency_penalty, None);
        assert!(!tgi_request.stream);
        assert!(tgi_request.tools.is_some());
        let tools = tgi_request.tools.as_ref().unwrap();
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            tgi_request.tool_choice,
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

        let tgi_request = TGIRequest::new(&model_name, &request_with_tools).unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 1);
        assert_eq!(tgi_request.temperature, None);
        assert_eq!(tgi_request.max_tokens, None);
        assert_eq!(tgi_request.seed, None);
        assert!(!tgi_request.stream);
        assert_eq!(tgi_request.top_p, None);
        assert_eq!(tgi_request.presence_penalty, None);
        assert_eq!(tgi_request.frequency_penalty, None);

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

        let tgi_request = TGIRequest::new(&model_name, &request_with_tools).unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 1);
        assert_eq!(tgi_request.temperature, None);
        assert_eq!(tgi_request.max_tokens, None);
        assert_eq!(tgi_request.seed, None);
        assert!(!tgi_request.stream);
        assert_eq!(tgi_request.top_p, None);
        assert_eq!(tgi_request.presence_penalty, None);
        assert_eq!(tgi_request.frequency_penalty, None);
    }

    #[test]
    fn test_tgi_to_tensorzero_chunk() {
        let chunk = TGIChatChunk {
            choices: vec![TGIChatChunkChoice {
                delta: TGIDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
            }],
            usage: None,
        };
        let inference_id = Uuid::now_v7();
        let message =
            tgi_to_tensorzero_chunk(chunk.clone(), inference_id, Duration::from_millis(50))
                .unwrap();
        assert_eq!(
            message.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "Hello".to_string(),
                id: "0".to_string(),
            })],
        );
    }
}
