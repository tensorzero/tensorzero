use std::time::Duration;

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
    error::Error,
    inference::types::{
        ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequest,
        ModelInferenceRequestJsonMode, ProviderInferenceResponse, ProviderInferenceResponseChunk,
        ProviderInferenceResponseStream, TextChunk, Usage,
    },
    tool::{ToolCall, ToolCallChunk, ToolChoice},
};

use super::{
    openai::{
        get_chat_url, prepare_openai_messages, OpenAIFunction, OpenAIRequestMessage, OpenAITool,
        OpenAIToolType,
    },
    provider_trait::InferenceProvider,
};

lazy_static! {
    static ref MISTRAL_API_BASE: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.mistral.ai/v1/").expect("Failed to parse MISTRAL_API_BASE")
    };
}

#[derive(Debug)]
pub struct MistralProvider {
    pub model_name: String,
    pub api_key: Option<SecretString>,
}

impl InferenceProvider for MistralProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<ProviderInferenceResponse, Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Mistral".to_string(),
        })?;
        let request_body = MistralRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(Some(&MISTRAL_API_BASE))?;
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to Mistral: {e}"),
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| Error::MistralServer {
                message: format!("Error parsing text response: {e}"),
            })?;

            let response = serde_json::from_str(&response).map_err(|e| Error::MistralServer {
                message: format!("Error parsing JSON response: {e}: {response}"),
            })?;

            MistralResponseWithMetadata {
                response,
                latency,
                request: request_body,
            }
            .try_into()
        } else {
            handle_mistral_error(
                res.status(),
                &res.text().await.map_err(|e| Error::MistralServer {
                    message: format!("Error parsing error response: {e}"),
                })?,
            )
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Mistral".to_string(),
        })?;
        let request_body = MistralRequest::new(&self.model_name, request)?;
        let raw_request =
            serde_json::to_string(&request_body).map_err(|e| Error::MistralServer {
                message: format!("Error serializing request: {e}"),
            })?;
        let request_url = get_chat_url(Some(&MISTRAL_API_BASE))?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to Mistral: {e}"),
            })?;
        let mut stream = Box::pin(stream_mistral(event_source, start_time));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::MistralServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };
        Ok((chunk, stream, raw_request))
    }

    fn has_credentials(&self) -> bool {
        self.api_key.is_some()
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
        | StatusCode::TOO_MANY_REQUESTS => Err(Error::MistralClient {
            message: response_body.to_string(),
            status_code: response_code,
        }),
        _ => Err(Error::MistralServer {
            message: response_body.to_string(),
        }),
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
                    yield Err(Error::OpenAIServer {
                        message: e.to_string(),
                    });
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<MistralChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::OpenAIServer {
                                message: format!(
                                    "Error parsing chunk. Error: {}, Data: {}",
                                    e, message.data
                                ),
                            });
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
                    .ok_or_else(|| Error::ToolNotFound {
                        name: tool_name.clone(),
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
                        return Err(Error::InvalidTool {
                            message: "Tool choice must be Auto or Required. This is impossible."
                                .to_string(),
                        })
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
        // NB: Fireworks will throw an error if you give FireworksResponseFormat::Text and then also include tools.
        // So we just don't include it as Text is the same as None anyway.
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(MistralResponseFormat::JsonObject)
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_openai_messages(request);
        let (tools, tool_choice) = prepare_mistral_tools(request)?;

        Ok(MistralRequest {
            messages,
            model,
            temperature: request.temperature,
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
}

impl<'a> TryFrom<MistralResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: MistralResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let MistralResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
        } = value;
        let raw_response = serde_json::to_string(&response).map_err(|e| Error::MistralServer {
            message: format!("Error parsing response: {e}"),
        })?;
        if response.choices.len() != 1 {
            return Err(Error::MistralServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
            });
        }
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or(Error::MistralServer {
                message: "Response has no choices (this should never happen)".to_string(),
            })?
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
        let raw_request =
            serde_json::to_string(&request_body).map_err(|e| Error::MistralServer {
                message: format!("Error serializing request body as JSON: {e}"),
            })?;

        Ok(ProviderInferenceResponse::new(
            content,
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
    let raw_message = serde_json::to_string(&chunk).map_err(|e| Error::MistralServer {
        message: format!("Error parsing response from Mistral: {e}"),
    })?;
    if chunk.choices.len() > 1 {
        return Err(Error::MistralServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
        });
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
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
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
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.content,
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
        let request_body = MistralRequest {
            messages: vec![],
            model: "mistral-small-latest",
            temperature: Some(0.5),
            max_tokens: Some(100),
            random_seed: Some(69),
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
        });
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.content,
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
        });
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::MistralServer { .. }));
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
        });
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::MistralServer { .. }));
    }

    #[test]
    fn test_handle_mistral_error() {
        use reqwest::StatusCode;

        // Test unauthorized error
        let unauthorized = handle_mistral_error(StatusCode::UNAUTHORIZED, "Unauthorized access");
        assert!(matches!(unauthorized, Err(Error::MistralClient { .. })));
        if let Err(Error::MistralClient {
            message,
            status_code,
        }) = unauthorized
        {
            assert_eq!(message, "Unauthorized access");
            assert_eq!(status_code, StatusCode::UNAUTHORIZED);
        }

        // Test forbidden error
        let forbidden = handle_mistral_error(StatusCode::FORBIDDEN, "Forbidden access");
        assert!(matches!(forbidden, Err(Error::MistralClient { .. })));
        if let Err(Error::MistralClient {
            message,
            status_code,
        }) = forbidden
        {
            assert_eq!(message, "Forbidden access");
            assert_eq!(status_code, StatusCode::FORBIDDEN);
        }

        // Test rate limit error
        let rate_limit = handle_mistral_error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded");
        assert!(matches!(rate_limit, Err(Error::MistralClient { .. })));
        if let Err(Error::MistralClient {
            message,
            status_code,
        }) = rate_limit
        {
            assert_eq!(message, "Rate limit exceeded");
            assert_eq!(status_code, StatusCode::TOO_MANY_REQUESTS);
        }

        // Test server error
        let server_error = handle_mistral_error(StatusCode::INTERNAL_SERVER_ERROR, "Server error");
        assert!(matches!(server_error, Err(Error::MistralServer { .. })));
        if let Err(Error::MistralServer { message }) = server_error {
            assert_eq!(message, "Server error");
        }
    }

    #[test]
    fn test_mistral_api_base() {
        assert_eq!(MISTRAL_API_BASE.as_str(), "https://api.mistral.ai/v1/");
    }
}
