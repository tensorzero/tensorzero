use futures::{Stream, StreamExt};
use lazy_static::lazy_static;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::batch::BatchRequestRow;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, ContentBlock, ContentBlockChunk, FunctionType,
    Latency, ModelInferenceRequestJsonMode, Role, Text,
};
use crate::inference::types::{
    ModelInferenceRequest, ProviderInferenceResponse, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStream, RequestMessage, TextChunk, Usage,
};
use crate::model::{Credential, CredentialLocation};
use crate::tool::{ToolCall, ToolCallChunk, ToolCallConfig, ToolChoice, ToolConfig};

lazy_static! {
    static ref ANTHROPIC_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.anthropic.com/v1/messages")
            .expect("Failed to parse ANTHROPIC_BASE_URL")
    };
}
const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const PROVIDER_NAME: &str = "Anthropic";
const PROVIDER_TYPE: &str = "anthropic";

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("ANTHROPIC_API_KEY".to_string())
}

#[derive(Debug)]
pub struct AnthropicProvider {
    model_name: String,
    credentials: AnthropicCredentials,
}

impl AnthropicProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, PROVIDER_TYPE))?;
        let provider_credentials = AnthropicCredentials::try_from(generic_credentials)?;
        Ok(AnthropicProvider {
            model_name,
            credentials: provider_credentials,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum AnthropicCredentials {
    Static(SecretString),
    Dynamic(String),
    #[cfg(any(test, feature = "e2e_tests"))]
    None,
}

impl TryFrom<Credential> for AnthropicCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(AnthropicCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(AnthropicCredentials::Dynamic(key_name)),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(AnthropicCredentials::None),
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
    ) -> Result<&'a SecretString, Error> {
        match self {
            AnthropicCredentials::Static(api_key) => Ok(api_key),
            AnthropicCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                })
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            AnthropicCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            }
            .into()),
        }
    }
}

impl InferenceProvider for AnthropicProvider {
    /// Anthropic non-streaming API request
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = AnthropicRequestBody::new(&self.model_name, request)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let res = http_client
            .post(ANTHROPIC_BASE_URL.as_ref())
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("x-api-key", api_key.expose_secret())
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request: {e}"),
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
                    raw_response: Some(response.to_string()),
                })
            })?;

            let response_with_latency = AnthropicResponseWithMetadata {
                response,
                latency,
                request: request_body,
                input_messages: request.messages.clone(),
                function_type: &request.function_type,
                json_mode: &request.json_mode,
            };
            Ok(response_with_latency.try_into()?)
        } else {
            let response_code = res.status();
            let response_text = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
            let error_body: AnthropicError = serde_json::from_str(&response_text).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(response_text),
                })
            })?;
            handle_anthropic_error(
                response_code,
                error_body.error,
                serde_json::to_string(&request_body).unwrap_or_default(),
            )
        }
    }

    /// Anthropic streaming API request
    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        api_key: &'a InferenceCredentials,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
        let request_body = AnthropicRequestBody::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request body as JSON: {e}"),
            })
        })?;
        let start_time = Instant::now();
        let api_key = self.credentials.get_api_key(api_key)?;
        let event_source = http_client
            .post(ANTHROPIC_BASE_URL.as_ref())
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("content-type", "application/json")
            .header("x-api-key", api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request: {e}"),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
        let mut stream = Box::pin(stream_anthropic(event_source, start_time));
        let mut chunk = match stream.next().await {
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
        if matches!(
            request.json_mode,
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict
        ) && matches!(request.function_type, FunctionType::Json)
        {
            chunk = prefill_json_chunk_response(chunk);
        }
        Ok((chunk, stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
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
        _http_client: &'a reqwest::Client,
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
    mut event_source: EventSource,
    start_time: Instant,
) -> impl Stream<Item = Result<ProviderInferenceResponseChunk, Error>> {
    async_stream::stream! {
        let inference_id = Uuid::now_v7();
        let mut current_tool_id : Option<String> = None;
        let mut current_tool_name: Option<String> = None;
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
                        let data: Result<AnthropicStreamMessage, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing message: {}, Data: {}",
                                    e, message.data
                                ),
                                provider_type: PROVIDER_TYPE.to_string(),
                                raw_request: None,
                                raw_response: Some(message.data.clone()),
                            }));
                        // Anthropic streaming API docs specify that this is the last message
                        if let Ok(AnthropicStreamMessage::MessageStop) = data {
                            break;
                        }

                        let response = data.and_then(|data| {
                            anthropic_to_tensorzero_stream_message(
                                data,
                                inference_id,
                                start_time.elapsed(),
                                &mut current_tool_id,
                                &mut current_tool_name,
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
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
/// Anthropic doesn't handle the system message in this way
/// It's a field of the POST body instead
enum AnthropicRole {
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
enum AnthropicToolChoice<'a> {
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
        let disable_parallel_tool_use = Some(!tool_call_config.parallel_tool_calls);
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
struct AnthropicTool<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    input_schema: &'a Value,
}

impl<'a> From<&'a ToolConfig> for AnthropicTool<'a> {
    fn from(value: &'a ToolConfig) -> Self {
        // In case we add more tool types in the future, the compiler will complain here.
        AnthropicTool {
            name: value.name(),
            description: Some(value.description()),
            input_schema: value.parameters(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
// NB: Anthropic also supports Image blocks here but we won't for now
enum AnthropicMessageContent<'a> {
    Text {
        text: &'a str,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: Vec<AnthropicMessageContent<'a>>,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: Value,
    },
}

impl<'a> TryFrom<&'a ContentBlock> for AnthropicMessageContent<'a> {
    type Error = Error;

    fn try_from(block: &'a ContentBlock) -> Result<Self, Self::Error> {
        match block {
            ContentBlock::Text(Text { text }) => Ok(AnthropicMessageContent::Text { text }),
            ContentBlock::ToolCall(tool_call) => {
                // Convert the tool call arguments from String to JSON Value (Anthropic expects an object)
                let input: Value = serde_json::from_str(&tool_call.arguments).map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: format!("Error parsing tool call arguments as JSON Value: {e}"),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: Some(tool_call.arguments.clone()),
                    })
                })?;

                if !input.is_object() {
                    return Err(Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: "Tool call arguments must be a JSON object".to_string(),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: Some(tool_call.arguments.clone()),
                    }));
                }

                Ok(AnthropicMessageContent::ToolUse {
                    id: &tool_call.id,
                    name: &tool_call.name,
                    input,
                })
            }
            ContentBlock::ToolResult(tool_result) => Ok(AnthropicMessageContent::ToolResult {
                tool_use_id: &tool_result.id,
                content: vec![AnthropicMessageContent::Text {
                    text: &tool_result.result,
                }],
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct AnthropicMessage<'a> {
    role: AnthropicRole,
    content: Vec<AnthropicMessageContent<'a>>,
}

impl<'a> TryFrom<&'a RequestMessage> for AnthropicMessage<'a> {
    type Error = Error;

    fn try_from(
        inference_message: &'a RequestMessage,
    ) -> Result<AnthropicMessage<'a>, Self::Error> {
        let content: Vec<AnthropicMessageContent> = inference_message
            .content
            .iter()
            .map(|block| block.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(AnthropicMessage {
            role: inference_message.role.into(),
            content,
        })
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct AnthropicRequestBody<'a> {
    model: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    // This is the system message
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
}

impl<'a> AnthropicRequestBody<'a> {
    fn new(
        model_name: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<AnthropicRequestBody<'a>, Error> {
        if request.messages.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            }
            .into());
        }
        let system = request.system.as_deref();
        let request_messages: Vec<AnthropicMessage> = request
            .messages
            .iter()
            .map(AnthropicMessage::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let messages = prepare_messages(request_messages)?;
        let messages = if matches!(
            request.json_mode,
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict
        ) && matches!(request.function_type, FunctionType::Json)
        {
            prefill_json_message(messages)
        } else {
            messages
        };

        // Workaround for Anthropic API limitation: they don't support explicitly specifying "none"
        // for tool choice. When ToolChoice::None is specified, we don't send any tools in the
        // request payload to achieve the same effect.
        let tools = request.tool_config.as_ref().and_then(|c| {
            if matches!(c.tool_choice, ToolChoice::None) {
                None
            } else {
                Some(
                    c.tools_available
                        .iter()
                        .map(|tool| tool.into())
                        .collect::<Vec<_>>(),
                )
            }
        });

        // `tool_choice` should only be set if tools are set and non-empty
        let tool_choice: Option<AnthropicToolChoice> = tools
            .as_ref()
            .filter(|t| !t.is_empty())
            .and(request.tool_config.as_ref())
            .and_then(|c| c.as_ref().try_into().ok());
        // NOTE: Anthropic does not support seed
        Ok(AnthropicRequestBody {
            model: model_name,
            messages,
            max_tokens: request.max_tokens.unwrap_or(4096),
            stream: Some(request.stream),
            system,
            temperature: request.temperature,
            top_p: request.top_p,
            tool_choice,
            tools,
        })
    }
}

/// Modifies the message array to satisfy Anthropic API requirements by:
/// - Prepending a default User message with "[listening]" if the first message is not from a User
/// - Appending a default User message with "[listening]" if the last message is from an Assistant
fn prepare_messages(mut messages: Vec<AnthropicMessage>) -> Result<Vec<AnthropicMessage>, Error> {
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
                    content: vec![AnthropicMessageContent::Text {
                        text: "[listening]",
                    }],
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
                content: vec![AnthropicMessageContent::Text {
                    text: "[listening]",
                }],
            });
        }
    }
    Ok(messages)
}

fn prefill_json_message(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    let mut messages = messages;
    // Add a JSON-prefill message for Anthropic's JSON mode
    messages.push(AnthropicMessage {
        role: AnthropicRole::Assistant,
        content: vec![AnthropicMessageContent::Text {
            text: "Here is the JSON requested:\n{",
        }],
    });
    messages
}

pub(crate) fn prefill_json_response(
    content: Vec<ContentBlock>,
) -> Result<Vec<ContentBlock>, Error> {
    // Check if the content is a single text block
    if content.len() == 1 {
        if let ContentBlock::Text(text) = &content[0] {
            // If it's a single text block, add a "{" to the beginning
            return Ok(vec![ContentBlock::Text(Text {
                text: format!("{{{}", text.text.trim()),
            })]);
        }
    }
    // If it's not a single text block, return content as-is but log an error
    Error::new(ErrorDetails::OutputParsing {
        message: "Expected a single text block in the response from Anthropic".to_string(),
        raw_output: serde_json::to_string(&content).map_err(|e| Error::new(ErrorDetails::Inference {
            message: format!("Error serializing content as JSON: {e}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new"),
        }))?,
    });
    Ok(content)
}

pub(crate) fn prefill_json_chunk_response(
    chunk: ProviderInferenceResponseChunk,
) -> ProviderInferenceResponseChunk {
    let mut chunk = chunk;
    if chunk.content.is_empty() {
        chunk.content = vec![ContentBlockChunk::Text(TextChunk {
            text: "{".to_string(),
            id: chunk.inference_id.to_string(),
        })];
    } else if chunk.content.len() == 1 {
        if let ContentBlockChunk::Text(TextChunk { text, .. }) = &chunk.content[0] {
            // Add a "{" to the beginning of the text
            chunk.content = vec![ContentBlockChunk::Text(TextChunk {
                text: format!("{{{}", text.trim_start()),
                id: chunk.inference_id.to_string(),
            })];
        }
    } else {
        Error::new(ErrorDetails::OutputParsing {
            message: "Expected a single text block in the response from Anthropic".to_string(),
            raw_output: serde_json::to_string(&chunk.content).map_err(|e| Error::new(ErrorDetails::Inference {
                message: format!("Error serializing content as JSON: {e}. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new"),
            })).unwrap_or_default()
        });
    }
    chunk
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct AnthropicError {
    error: AnthropicErrorBody,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct AnthropicErrorBody {
    r#type: String,
    message: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

impl TryFrom<AnthropicContentBlock> for ContentBlock {
    type Error = Error;
    fn try_from(block: AnthropicContentBlock) -> Result<Self, Self::Error> {
        match block {
            AnthropicContentBlock::Text { text } => Ok(text.into()),
            AnthropicContentBlock::ToolUse { id, name, input } => {
                Ok(ContentBlock::ToolCall(ToolCall {
                    id,
                    name,
                    arguments: serde_json::to_string(&input).map_err(|e| {
                        Error::new(ErrorDetails::InferenceServer {
                            message: format!("Error parsing input for tool call: {e}"),
                            provider_type: PROVIDER_TYPE.to_string(),
                            raw_request: None,
                            raw_response: Some(serde_json::to_string(&input).unwrap_or_default()),
                        })
                    })?,
                }))
            }
        }
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
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct AnthropicResponse {
    id: String,
    r#type: String, // this is always "message"
    role: String,   // this is always "assistant"
    content: Vec<AnthropicContentBlock>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, PartialEq)]
struct AnthropicResponseWithMetadata<'a> {
    response: AnthropicResponse,
    latency: Latency,
    request: AnthropicRequestBody<'a>,
    input_messages: Vec<RequestMessage>,
    function_type: &'a FunctionType,
    json_mode: &'a ModelInferenceRequestJsonMode,
}

impl<'a> TryFrom<AnthropicResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: AnthropicResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let AnthropicResponseWithMetadata {
            response,
            latency,
            request: request_body,
            input_messages,
            function_type,
            json_mode,
        } = value;

        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request body as JSON: {e}"),
            })
        })?;

        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error parsing response from Anthropic: {e}"),
            })
        })?;

        let output: Vec<ContentBlock> = response
            .content
            .into_iter()
            .map(|block| block.try_into())
            .collect::<Result<Vec<_>, _>>()?;
        let content = if matches!(
            json_mode,
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict
        ) && matches!(function_type, FunctionType::Json)
        {
            prefill_json_response(output)?
        } else {
            output
        };

        Ok(ProviderInferenceResponse::new(
            content,
            request_body.system.map(String::from),
            input_messages,
            raw_request,
            raw_response,
            response.usage.into(),
            latency,
        ))
    }
}

fn handle_anthropic_error(
    response_code: StatusCode,
    response_body: AnthropicErrorBody,
    raw_request: String,
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED
        | StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::TOO_MANY_REQUESTS => Err(ErrorDetails::InferenceClient {
            status_code: Some(response_code),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(raw_request),
            raw_response: serde_json::to_string(&response_body).ok(),
            message: response_body.message,
        }
        .into()),
        // StatusCode::NOT_FOUND | StatusCode::FORBIDDEN | StatusCode::INTERNAL_SERVER_ERROR | 529: Overloaded
        // These are all captured in _ since they have the same error behavior
        _ => Err(ErrorDetails::InferenceServer {
            raw_response: serde_json::to_string(&response_body).ok(),
            message: response_body.message,
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: Some(raw_request),
        }
        .into()),
    }
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicMessageBlock {
    Text {
        text: String,
    },
    TextDelta {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    InputJsonDelta {
        partial_json: String,
    },
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamMessage {
    ContentBlockDelta {
        delta: AnthropicMessageBlock,
        index: u32,
    },
    ContentBlockStart {
        content_block: AnthropicMessageBlock,
        index: u32,
    },
    ContentBlockStop {
        index: u32,
    },
    Error {
        error: Value,
    },
    MessageDelta {
        delta: Value,
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
fn anthropic_to_tensorzero_stream_message(
    message: AnthropicStreamMessage,
    inference_id: Uuid,
    message_latency: Duration,
    current_tool_id: &mut Option<String>,
    current_tool_name: &mut Option<String>,
) -> Result<Option<ProviderInferenceResponseChunk>, Error> {
    let raw_message = serde_json::to_string(&message).map_err(|e| {
        Error::new(ErrorDetails::Serialization {
            message: format!("Error serializing stream message from Anthropic: {e}"),
        })
    })?;
    match message {
        AnthropicStreamMessage::ContentBlockDelta { delta, index } => match delta {
            AnthropicMessageBlock::TextDelta { text } => {
                Ok(Some(ProviderInferenceResponseChunk::new(
                    inference_id,
                    vec![ContentBlockChunk::Text(TextChunk {
                        text,
                        id: index.to_string(),
                    })],
                    None,
                    raw_message,
                    message_latency,
                )))
            }
            AnthropicMessageBlock::InputJsonDelta { partial_json } => {
                Ok(Some(ProviderInferenceResponseChunk::new(
                    inference_id,
                    // Take the current tool name and ID and use them to create a ToolCallChunk
                    // This is necessary because the ToolCallChunk must always contain the tool name and ID
                    // even though Anthropic only sends the tool ID and name in the ToolUse chunk and not InputJSONDelta
                    vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                        raw_name: current_tool_name.clone().ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                            message: "Got InputJsonDelta chunk from Anthropic without current tool name being set by a ToolUse".to_string(),
                            provider_type: PROVIDER_TYPE.to_string(),
                            raw_request: None,
                            raw_response: None,
                        }))?,
                        id: current_tool_id.clone().ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                            message: "Got InputJsonDelta chunk from Anthropic without current tool id being set by a ToolUse".to_string(),
                            provider_type: PROVIDER_TYPE.to_string(),
                            raw_request: None,
                            raw_response: None,
                        }))?,
                        raw_arguments: partial_json,
                    })],
                    None,
                    raw_message,
                    message_latency,
                )))
            }
            _ => Err(ErrorDetails::InferenceServer {
                message: "Unsupported content block type for ContentBlockDelta".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(serde_json::to_string(&delta).unwrap_or_default()),
            }
            .into()),
        },
        AnthropicStreamMessage::ContentBlockStart {
            content_block,
            index,
        } => match content_block {
            AnthropicMessageBlock::Text { text } => {
                let text_chunk = ContentBlockChunk::Text(TextChunk {
                    text,
                    id: index.to_string(),
                });
                Ok(Some(ProviderInferenceResponseChunk::new(
                    inference_id,
                    vec![text_chunk],
                    None,
                    raw_message,
                    message_latency,
                )))
            }
            AnthropicMessageBlock::ToolUse { id, name, .. } => {
                // This is a new tool call, update the ID for future chunks
                *current_tool_id = Some(id.clone());
                *current_tool_name = Some(name.clone());
                Ok(Some(ProviderInferenceResponseChunk::new(
                    inference_id,
                    vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                        id,
                        raw_name: name,
                        // As far as I can tell this is always {} so we ignore
                        raw_arguments: "".to_string(),
                    })],
                    None,
                    raw_message,
                    message_latency,
                )))
            }
            _ => Err(ErrorDetails::InferenceServer {
                message: "Unsupported content block type for ContentBlockStart".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(serde_json::to_string(&content_block).unwrap_or_default()),
            }
            .into()),
        },
        AnthropicStreamMessage::ContentBlockStop { .. } => Ok(None),
        AnthropicStreamMessage::Error { error } => Err(ErrorDetails::InferenceServer {
            message: error.to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        }
        .into()),
        AnthropicStreamMessage::MessageDelta { usage, .. } => {
            let usage = parse_usage_info(&usage);
            Ok(Some(ProviderInferenceResponseChunk::new(
                inference_id,
                vec![],
                Some(usage.into()),
                raw_message,
                message_latency,
            )))
        }
        AnthropicStreamMessage::MessageStart { message } => {
            if let Some(usage_info) = message.get("usage") {
                let usage = parse_usage_info(usage_info);
                Ok(Some(ProviderInferenceResponseChunk::new(
                    inference_id,
                    vec![],
                    Some(usage.into()),
                    raw_message,
                    message_latency,
                )))
            } else {
                Ok(None)
            }
        }
        AnthropicStreamMessage::MessageStop | AnthropicStreamMessage::Ping {} => Ok(None),
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

    use super::*;
    use crate::inference::providers::common::WEATHER_TOOL_CONFIG;
    use crate::inference::types::{FunctionType, ModelInferenceRequestJsonMode};
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::tool::{DynamicToolConfig, ToolConfig, ToolResult};
    use serde_json::json;

    #[test]
    fn test_try_from_tool_call_config() {
        // Need to cover all 4 cases
        let tool_call_config = ToolCallConfig {
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: false,
            tools_available: vec![],
        };
        let anthropic_tool_choice = AnthropicToolChoice::try_from(&tool_call_config);
        assert!(matches!(
            anthropic_tool_choice.unwrap(),
            AnthropicToolChoice::Auto {
                disable_parallel_tool_use: Some(true)
            }
        ));

        let tool_call_config = ToolCallConfig {
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: true,
            tools_available: vec![],
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
            parallel_tool_calls: true,
            tools_available: vec![],
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
            parallel_tool_calls: false,
            tools_available: vec![],
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
        let tool = ToolConfig::Dynamic(DynamicToolConfig {
            name: "test".to_string(),
            description: "test".to_string(),
            parameters: DynamicJSONSchema::new(parameters.clone()),
            strict: false,
        });
        let anthropic_tool: AnthropicTool = (&tool).into();
        assert_eq!(
            anthropic_tool,
            AnthropicTool {
                name: "test",
                description: Some("test"),
                input_schema: &parameters,
            }
        );
    }

    #[test]
    fn test_try_from_content_block() {
        let text_content_block = "test".to_string().into();
        let anthropic_content_block =
            AnthropicMessageContent::try_from(&text_content_block).unwrap();
        assert_eq!(
            anthropic_content_block,
            AnthropicMessageContent::Text { text: "test" }
        );

        let tool_call_content_block = ContentBlock::ToolCall(ToolCall {
            id: "test_id".to_string(),
            name: "test_name".to_string(),
            arguments: serde_json::to_string(&json!({"type": "string"})).unwrap(),
        });
        let anthropic_content_block =
            AnthropicMessageContent::try_from(&tool_call_content_block).unwrap();
        assert_eq!(
            anthropic_content_block,
            AnthropicMessageContent::ToolUse {
                id: "test_id",
                name: "test_name",
                input: json!({"type": "string"})
            }
        );
    }

    #[test]
    fn test_try_from_request_message() {
        // Test a User message
        let inference_request_message = RequestMessage {
            role: Role::User,
            content: vec!["test".to_string().into()],
        };
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message).unwrap();
        assert_eq!(
            anthropic_message,
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "test" }],
            }
        );

        // Test an Assistant message
        let inference_request_message = RequestMessage {
            role: Role::Assistant,
            content: vec!["test_assistant".to_string().into()],
        };
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message).unwrap();
        assert_eq!(
            anthropic_message,
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    text: "test_assistant",
                }],
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
        let anthropic_message = AnthropicMessage::try_from(&inference_request_message).unwrap();
        assert_eq!(
            anthropic_message,
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::ToolResult {
                    tool_use_id: "test_tool_call_id",
                    content: vec![AnthropicMessageContent::Text {
                        text: "test_tool_response"
                    }],
                }],
            }
        );
    }

    #[test]
    fn test_initialize_anthropic_request_body() {
        let model = "claude".to_string();
        let listening_message = AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text {
                text: "[listening]",
            }],
        };

        // Test Case 1: Empty message list
        let inference_request = ModelInferenceRequest {
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
        };
        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        let details = anthropic_request_body.unwrap_err().get_owned_details();
        assert_eq!(
            details,
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
        };
        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: &model,
                messages: vec![
                    listening_message.clone(),
                    AnthropicMessage::try_from(&inference_request.messages[0]).unwrap(),
                    listening_message.clone(),
                ],
                max_tokens: 4096,
                stream: Some(false),
                system: Some("test_system"),
                temperature: None,
                top_p: None,
                tool_choice: None,
                tools: None,
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
        };
        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: &model,
                messages: vec![
                    AnthropicMessage::try_from(&inference_request.messages[0]).unwrap(),
                    AnthropicMessage::try_from(&inference_request.messages[1]).unwrap(),
                    listening_message.clone(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some("test_system"),
                temperature: Some(0.5),
                top_p: None,
                tool_choice: None,
                tools: None,
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
        };
        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            AnthropicRequestBody {
                model: &model,
                messages: inference_request
                    .messages
                    .iter()
                    .map(|m| AnthropicMessage::try_from(m).unwrap())
                    .collect(),
                max_tokens: 4096,
                stream: Some(false),
                system: None,
                temperature: None,
                top_p: None,
                tool_choice: None,
                tools: None,
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
        };
        let anthropic_request_body = AnthropicRequestBody::new(&model, &inference_request);
        assert!(anthropic_request_body.is_ok());
        let result = anthropic_request_body.unwrap();
        assert_eq!(result.messages.len(), 4); // Original 2 messages + listening message + JSON prefill
        assert_eq!(
            result.messages[0],
            AnthropicMessage::try_from(&inference_request.messages[0]).unwrap()
        );
        assert_eq!(
            result.messages[1],
            AnthropicMessage::try_from(&inference_request.messages[1]).unwrap()
        );
        assert_eq!(result.messages[2], listening_message);
        assert_eq!(
            result.messages[3],
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    text: "Here is the JSON requested:\n{",
                }],
            }
        );
    }

    #[test]
    fn test_prepare_messages() {
        let listening_message = AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text {
                text: "[listening]",
            }],
        };

        // Test case 1: Empty messages - should add listening message
        let messages = vec![];
        let result = prepare_messages(messages).unwrap();
        assert_eq!(result, vec![listening_message.clone()]);

        // Test case 2: First message is Assistant - should prepend listening message
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text { text: "Hi" }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "Hello" }],
            },
        ];
        let result = prepare_messages(messages).unwrap();
        assert_eq!(
            result,
            vec![
                listening_message.clone(),
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![AnthropicMessageContent::Text { text: "Hi" }],
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![AnthropicMessageContent::Text { text: "Hello" }],
                },
            ]
        );

        // Test case 3: Last message is Assistant - should append listening message
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "Hello" }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text { text: "Hi" }],
            },
        ];
        let result = prepare_messages(messages).unwrap();
        assert_eq!(
            result,
            vec![
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![AnthropicMessageContent::Text { text: "Hello" }],
                },
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![AnthropicMessageContent::Text { text: "Hi" }],
                },
                listening_message.clone(),
            ]
        );

        // Test case 4: Valid message sequence - no changes needed
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "Hello" }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text { text: "Hi" }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text {
                    text: "How are you?",
                }],
            },
        ];
        let result = prepare_messages(messages.clone()).unwrap();
        assert_eq!(result, messages);

        // Test case 5: Both first Assistant and last Assistant - should add listening messages at both ends
        let messages = vec![
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text { text: "Hi" }],
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: vec![AnthropicMessageContent::Text { text: "Hello" }],
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: vec![AnthropicMessageContent::Text {
                    text: "How can I help?",
                }],
            },
        ];
        let result = prepare_messages(messages).unwrap();
        assert_eq!(
            result,
            vec![
                listening_message.clone(),
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![AnthropicMessageContent::Text { text: "Hi" }],
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: vec![AnthropicMessageContent::Text { text: "Hello" }],
                },
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![AnthropicMessageContent::Text {
                        text: "How can I help?"
                    }],
                },
                listening_message.clone(),
            ]
        );

        // Test case 6: Single Assistant message - should add listening messages at both ends
        let messages = vec![AnthropicMessage {
            role: AnthropicRole::Assistant,
            content: vec![AnthropicMessageContent::Text { text: "Hi" }],
        }];
        let result = prepare_messages(messages).unwrap();
        assert_eq!(
            result,
            vec![
                listening_message.clone(),
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: vec![AnthropicMessageContent::Text { text: "Hi" }],
                },
                listening_message.clone(),
            ]
        );

        // Test case 7: Single User message - no changes needed
        let messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text { text: "Hello" }],
        }];
        let result = prepare_messages(messages.clone()).unwrap();
        assert_eq!(result, messages);
    }

    #[test]
    fn test_handle_anthropic_error() {
        let error_body = AnthropicErrorBody {
            r#type: "error".to_string(),
            message: "test_message".to_string(),
        };
        let response_code = StatusCode::BAD_REQUEST;
        let result =
            handle_anthropic_error(response_code, error_body.clone(), "raw request".to_string());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceClient {
                message: "test_message".to_string(),
                status_code: Some(response_code),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("{\"type\":\"error\",\"message\":\"test_message\"}".to_string()),
            }
        );
        let response_code = StatusCode::UNAUTHORIZED;
        let result =
            handle_anthropic_error(response_code, error_body.clone(), "raw request".to_string());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceClient {
                message: "test_message".to_string(),
                status_code: Some(response_code),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("{\"type\":\"error\",\"message\":\"test_message\"}".to_string()),
            }
        );
        let response_code = StatusCode::TOO_MANY_REQUESTS;
        let result =
            handle_anthropic_error(response_code, error_body.clone(), "raw request".to_string());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceClient {
                message: "test_message".to_string(),
                status_code: Some(response_code),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("{\"type\":\"error\",\"message\":\"test_message\"}".to_string()),
            }
        );
        let response_code = StatusCode::NOT_FOUND;
        let result =
            handle_anthropic_error(response_code, error_body.clone(), "raw request".to_string());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceServer {
                message: "test_message".to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("{\"type\":\"error\",\"message\":\"test_message\"}".to_string()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
        );
        let response_code = StatusCode::INTERNAL_SERVER_ERROR;
        let result =
            handle_anthropic_error(response_code, error_body.clone(), "raw request".to_string());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceServer {
                message: "test_message".to_string(),
                raw_request: Some("raw request".to_string()),
                raw_response: Some("{\"type\":\"error\",\"message\":\"test_message\"}".to_string()),
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

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
    }

    #[test]
    fn test_anthropic_response_conversion() {
        // Test case 1: Text response
        let anthropic_response_body = AnthropicResponse {
            id: "1".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock::Text {
                text: "Response text".to_string(),
            }],
            model: "model-name".to_string(),
            stop_reason: Some("stop reason".to_string()),
            stop_sequence: Some("stop sequence".to_string()),
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
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
            top_p: Some(0.9),
            temperature: None,
            tool_choice: None,
            tools: None,
        };
        let input_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into()],
        }];
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = AnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            latency: latency.clone(),
            request: request_body,
            input_messages: input_messages.clone(),
            function_type: &FunctionType::Chat,
            json_mode: &ModelInferenceRequestJsonMode::Off,
        };

        let inference_response = ProviderInferenceResponse::try_from(body_with_latency).unwrap();
        assert_eq!(
            inference_response.output,
            vec!["Response text".to_string().into()]
        );

        let raw_json = json!(anthropic_response_body).to_string();
        assert_eq!(raw_json, inference_response.raw_response);
        assert_eq!(inference_response.usage.input_tokens, 100);
        assert_eq!(inference_response.usage.output_tokens, 50);
        assert_eq!(inference_response.latency, latency);
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.input_messages, input_messages);

        // Test case 2: Tool call response
        let anthropic_response_body = AnthropicResponse {
            id: "2".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock::ToolUse {
                id: "tool_call_1".to_string(),
                name: "get_temperature".to_string(),
                input: json!({"location": "New York"}),
            }],
            model: "model-name".to_string(),
            stop_reason: Some("tool_call".to_string()),
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
            system: None,
            temperature: None,
            top_p: Some(0.9),
            tool_choice: None,
            tools: None,
        };
        let input_messages = vec![RequestMessage {
            role: Role::Assistant,
            content: vec!["Hello".to_string().into()],
        }];
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = AnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            latency: latency.clone(),
            request: request_body,
            input_messages: input_messages.clone(),
            function_type: &FunctionType::Chat,
            json_mode: &ModelInferenceRequestJsonMode::Off,
        };

        let inference_response: ProviderInferenceResponse = body_with_latency.try_into().unwrap();
        assert!(inference_response.output.len() == 1);
        assert_eq!(
            inference_response.output[0],
            ContentBlock::ToolCall(ToolCall {
                id: "tool_call_1".to_string(),
                name: "get_temperature".to_string(),
                arguments: r#"{"location":"New York"}"#.to_string(),
            })
        );

        let raw_json = json!(anthropic_response_body).to_string();
        assert_eq!(raw_json, inference_response.raw_response);
        assert_eq!(inference_response.usage.input_tokens, 100);
        assert_eq!(inference_response.usage.output_tokens, 50);
        assert_eq!(inference_response.latency, latency);
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.input_messages, input_messages);

        // Test case 3: Mixed response (text and tool call)
        let anthropic_response_body = AnthropicResponse {
            id: "3".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                AnthropicContentBlock::Text {
                    text: "Here's the weather:".to_string(),
                },
                AnthropicContentBlock::ToolUse {
                    id: "tool_call_2".to_string(),
                    name: "get_temperature".to_string(),
                    input: json!({"location": "London"}),
                },
            ],
            model: "model-name".to_string(),
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
            system: None,
            temperature: None,
            top_p: Some(0.9),
            tool_choice: None,
            tools: None,
        };
        let input_messages = vec![RequestMessage {
            role: Role::User,
            content: vec!["Helloooo".to_string().into()],
        }];
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = AnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            latency: latency.clone(),
            request: request_body,
            input_messages: input_messages.clone(),
            function_type: &FunctionType::Chat,
            json_mode: &ModelInferenceRequestJsonMode::Off,
        };
        let inference_response = ProviderInferenceResponse::try_from(body_with_latency).unwrap();
        assert_eq!(
            inference_response.output[0],
            "Here's the weather:".to_string().into()
        );
        assert!(inference_response.output.len() == 2);
        assert_eq!(
            inference_response.output[1],
            ContentBlock::ToolCall(ToolCall {
                id: "tool_call_2".to_string(),
                name: "get_temperature".to_string(),
                arguments: r#"{"location":"London"}"#.to_string(),
            })
        );

        let raw_json = json!(anthropic_response_body).to_string();
        assert_eq!(raw_json, inference_response.raw_response);

        assert_eq!(inference_response.usage.input_tokens, 100);
        assert_eq!(inference_response.usage.output_tokens, 50);
        assert_eq!(inference_response.latency, latency);
        assert_eq!(inference_response.raw_request, raw_request);
        assert_eq!(inference_response.input_messages, input_messages);
    }

    #[test]
    fn test_anthropic_to_tensorzero_stream_message() {
        use serde_json::json;
        use uuid::Uuid;

        let inference_id = Uuid::now_v7();

        // Test ContentBlockDelta with TextDelta
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_delta = AnthropicStreamMessage::ContentBlockDelta {
            delta: AnthropicMessageBlock::TextDelta {
                text: "Hello".to_string(),
            },
            index: 0,
        };
        let latency = Duration::from_millis(100);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_delta,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
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
            delta: AnthropicMessageBlock::InputJsonDelta {
                partial_json: "aaaa: bbbbb".to_string(),
            },
            index: 0,
        };
        let latency = Duration::from_millis(100);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_delta,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceServer {
                message: "Got InputJsonDelta chunk from Anthropic without current tool name being set by a ToolUse".to_string(),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            }
        );

        // Test ContentBlockDelta with InputJsonDelta and previous tool info
        let mut current_tool_id = Some("tool_id".to_string());
        let mut current_tool_name = Some("tool_name".to_string());
        let content_block_delta = AnthropicStreamMessage::ContentBlockDelta {
            delta: AnthropicMessageBlock::InputJsonDelta {
                partial_json: "aaaa: bbbbb".to_string(),
            },
            index: 0,
        };
        let latency = Duration::from_millis(100);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_delta,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "tool_id".to_string());
                assert_eq!(tool_call.raw_name, "tool_name".to_string());
                assert_eq!(tool_call.raw_arguments, "aaaa: bbbbb".to_string());
            }
            _ => panic!("Expected a tool call content block"),
        }
        assert_eq!(chunk.latency, latency);

        // Test ContentBlockStart with ToolUse
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_start = AnthropicStreamMessage::ContentBlockStart {
            content_block: AnthropicMessageBlock::ToolUse {
                id: "tool1".to_string(),
                name: "calculator".to_string(),
                input: json!({}),
            },
            index: 1,
        };
        let latency = Duration::from_millis(110);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_start,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 1);
        match &chunk.content[0] {
            ContentBlockChunk::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "tool1".to_string());
                assert_eq!(tool_call.raw_name, "calculator".to_string());
                assert_eq!(tool_call.raw_arguments, "".to_string());
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
            content_block: AnthropicMessageBlock::Text {
                text: "Hello".to_string(),
            },
            index: 2,
        };
        let latency = Duration::from_millis(120);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_start,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
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

        // Test ContentBlockStart with InputJsonDelta (should fail)
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_start = AnthropicStreamMessage::ContentBlockStart {
            content_block: AnthropicMessageBlock::InputJsonDelta {
                partial_json: "aaaa: bbbbb".to_string(),
            },
            index: 3,
        };
        let latency = Duration::from_millis(130);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_start,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceServer {
                message: "Unsupported content block type for ContentBlockStart".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(
                    "{\"type\":\"input_json_delta\",\"partial_json\":\"aaaa: bbbbb\"}".to_string()
                ),
            }
        );

        // Test ContentBlockStop
        let content_block_stop = AnthropicStreamMessage::ContentBlockStop { index: 2 };
        let latency = Duration::from_millis(120);
        let result = anthropic_to_tensorzero_stream_message(
            content_block_stop,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test Error
        let error_message = AnthropicStreamMessage::Error {
            error: json!({"message": "Test error"}),
        };
        let latency = Duration::from_millis(130);
        let result = anthropic_to_tensorzero_stream_message(
            error_message,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceServer {
                message: r#"{"message":"Test error"}"#.to_string(),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            }
        );

        // Test MessageDelta with usage
        let message_delta = AnthropicStreamMessage::MessageDelta {
            delta: json!({}),
            usage: json!({"input_tokens": 10, "output_tokens": 20}),
        };
        let latency = Duration::from_millis(140);
        let result = anthropic_to_tensorzero_stream_message(
            message_delta,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 0);
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(chunk.latency, latency);

        // Test MessageStart with usage
        let message_start = AnthropicStreamMessage::MessageStart {
            message: json!({"usage": {"input_tokens": 5, "output_tokens": 15}}),
        };
        let latency = Duration::from_millis(150);
        let result = anthropic_to_tensorzero_stream_message(
            message_start,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        let chunk = result.unwrap().unwrap();
        assert_eq!(chunk.content.len(), 0);
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 15);
        assert_eq!(chunk.latency, latency);

        // Test MessageStop
        let message_stop = AnthropicStreamMessage::MessageStop;
        let latency = Duration::from_millis(160);
        let result = anthropic_to_tensorzero_stream_message(
            message_stop,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test Ping
        let ping = AnthropicStreamMessage::Ping {};
        let latency = Duration::from_millis(170);
        let result = anthropic_to_tensorzero_stream_message(
            ping,
            inference_id,
            latency,
            &mut current_tool_id,
            &mut current_tool_name,
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
            ANTHROPIC_BASE_URL.as_str(),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn test_prefill_json_message() {
        // Create a sample input message
        let input_messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicMessageContent::Text {
                text: "Generate some JSON",
            }],
        }];

        // Call the function
        let result = prefill_json_message(input_messages);

        // Assert that the result has one more message than the input
        assert_eq!(result.len(), 2);

        // Check the original message is unchanged
        assert_eq!(result[0].role, AnthropicRole::User);
        assert_eq!(
            result[0].content,
            vec![AnthropicMessageContent::Text {
                text: "Generate some JSON",
            }]
        );

        // Check the new message is correct
        assert_eq!(result[1].role, AnthropicRole::Assistant);
        assert_eq!(
            result[1].content,
            vec![AnthropicMessageContent::Text {
                text: "Here is the JSON requested:\n{",
            }]
        );
    }

    #[test]
    fn test_prefill_json_response() {
        // Test case 1: Single text block
        let input = vec![ContentBlock::Text(Text {
            text: "  \"key\": \"value\"}".to_string(),
        })];
        let result = prefill_json_response(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0],
            ContentBlock::Text(Text {
                text: "{\"key\": \"value\"}".to_string(),
            })
        );

        // Test case 2: Multiple blocks
        let input = vec![
            ContentBlock::Text(Text {
                text: "Block 1".to_string(),
            }),
            ContentBlock::Text(Text {
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
        let input = vec![ContentBlock::ToolCall(ToolCall {
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
        let inference_id = Uuid::now_v7();
        let chunk = ProviderInferenceResponseChunk {
            inference_id,
            content: vec![],
            created: 0,
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(0),
        };
        let result = prefill_json_chunk_response(chunk);
        assert_eq!(
            result.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "{".to_string(),
                id: inference_id.to_string(),
            })]
        );
        // Test case 2: Single text block
        let inference_id = Uuid::now_v7();
        let chunk = ProviderInferenceResponseChunk {
            inference_id,
            created: 0,
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(0),
            content: vec![ContentBlockChunk::Text(TextChunk {
                text: "\"key\": \"value ".to_string(),
                id: inference_id.to_string(),
            })],
        };
        let result = prefill_json_chunk_response(chunk);
        assert_eq!(
            result.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: "{\"key\": \"value ".to_string(),
                id: inference_id.to_string(),
            })]
        );

        // Test case 3: Multiple blocks (should remain unchanged)
        let chunk = ProviderInferenceResponseChunk {
            inference_id: Uuid::now_v7(),
            created: 0,
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(0),
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
        let result = prefill_json_chunk_response(chunk.clone());
        assert_eq!(result, chunk);

        // Test case 4: Non-text block (should remain unchanged)
        let chunk = ProviderInferenceResponseChunk {
            inference_id: Uuid::now_v7(),
            created: 0,
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(0),
            content: vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "1".to_string(),
                raw_name: "test_tool".to_string(),
                raw_arguments: "{}".to_string(),
            })],
        };
        let result = prefill_json_chunk_response(chunk.clone());
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

        // Test Missing credential (test mode)
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            let generic = Credential::Missing;
            let creds = AnthropicCredentials::try_from(generic).unwrap();
            assert!(matches!(creds, AnthropicCredentials::None));
        }

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = AnthropicCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
}
