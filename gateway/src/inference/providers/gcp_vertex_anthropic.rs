use std::time::Duration;

use futures::{Stream, StreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use uuid::Uuid;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    batch::BatchProviderInferenceResponse, ContentBlock, ContentBlockChunk, FunctionType, Latency,
    ModelInferenceRequestJsonMode, Role, Text, TextChunk,
};
use crate::inference::types::{
    ModelInferenceRequest, ProviderInferenceResponse, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStream, RequestMessage, Usage,
};
use crate::model::{Credential, CredentialLocation};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

use super::anthropic::{prefill_json_chunk_response, prefill_json_response};
use super::gcp_vertex_gemini::{
    default_api_key_location, GCPVertexCredentials,
};

/// Implements a subset of the GCP Vertex Gemini API as documented [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/generateContent) for non-streaming
/// and [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/streamGenerateContent) for streaming

#[derive(Debug)]
pub struct GCPVertexAnthropicProvider {
    pub request_url: String,
    pub streaming_request_url: String,
    pub audience: String,
    pub credentials: GCPVertexCredentials,
    pub model_id: String,
}

impl GCPVertexAnthropicProvider {
    pub fn new(
        model_id: String,
        location: String,
        project_id: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, "GCPVertexAnthropic"))?;
        let provider_credentials= GCPVertexCredentials::try_from((generic_credentials, "GCPVertexAnthropic"))?;
        let request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:rawPredict");
        let streaming_request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/anthropic/models/{model_id}:streamRawPredict");
        let audience = format!("https://{location}-aiplatform.googleapis.com/");

        Ok(GCPVertexAnthropicProvider {
            request_url,
            streaming_request_url,
            audience,
            credentials: provider_credentials,
            model_id,
        })
    }
}

const ANTHROPIC_API_VERSION: &str = "vertex-2023-10-16";

impl InferenceProvider for GCPVertexAnthropicProvider {
    /// Anthropic non-streaming API request
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = GCPVertexAnthropicRequestBody::new(request)?;
        let api_key = self
            .credentials
            .get_api_key(&self.audience, dynamic_api_keys)?;
        let start_time = Instant::now();
        let res = http_client
            .post(&self.request_url)
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request: {e}"),
                    status_code: e.status(),
                    provider_type: "GCP Vertex Anthropic".to_string(),
                })
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: "GCP Vertex Anthropic".to_string(),
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: "GCP Vertex Anthropic".to_string(),
                })
            })?;

            let response_with_latency = GCPVertexAnthropicResponseWithMetadata {
                response,
                latency,
                request: request_body,
                function_type: &request.function_type,
                json_mode: &request.json_mode,
                generic_request: request,
            };
            Ok(response_with_latency.try_into()?)
        } else {
            let response_code = res.status();
            let error_body = res.json::<GCPVertexAnthropicError>().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing response: {e}"),
                    provider_type: "GCP Vertex Anthropic".to_string(),
                })
            })?;
            handle_anthropic_error(response_code, error_body.error)
        }
    }

    /// Anthropic streaming API request
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
        let request_body = GCPVertexAnthropicRequestBody::new(request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request body as JSON: {e}"),
                provider_type: "GCP Vertex Anthropic".to_string(),
            })
        })?;
        let api_key = self
            .credentials
            .get_api_key(&self.audience, dynamic_api_keys)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(&self.streaming_request_url)
            .bearer_auth(api_key.expose_secret())
            .header("content-type", "application/json")
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to Anthropic: {e}"),
                    status_code: None,
                    provider_type: "GCP Vertex Anthropic".to_string(),
                })
            })?;
        let mut stream = Box::pin(stream_anthropic(event_source, start_time));
        let mut chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(ErrorDetails::InferenceServer {
                    message: "Stream ended before first chunk".to_string(),
                    provider_type: "GCP Vertex Anthropic".to_string(),
                }
                .into())
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
        _requests: &'a [ModelInferenceRequest<'a>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<BatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "GCP Vertex Anthropic".to_string(),
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
                        provider_type: "GCP Vertex Anthropic".to_string()
                    }.into());
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<GCPVertexAnthropicStreamMessage, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::new(ErrorDetails::InferenceServer {
                                message: format!(
                                    "Error parsing message: {}, Data: {}",
                                    e, message.data
                                ),
                                provider_type: "GCP Vertex Anthropic".to_string()
                            }));
                        // Anthropic streaming API docs specify that this is the last message
                        if let Ok(GCPVertexAnthropicStreamMessage::MessageStop) = data {
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
enum GCPVertexAnthropicRole {
    User,
    Assistant,
}

impl From<Role> for GCPVertexAnthropicRole {
    fn from(role: Role) -> Self {
        match role {
            Role::User => GCPVertexAnthropicRole::User,
            Role::Assistant => GCPVertexAnthropicRole::Assistant,
        }
    }
}

/// We can instruct Anthropic to use a particular tool,
/// any tool (but to use one), or to use a tool if needed.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum GCPVertexAnthropicToolChoice<'a> {
    Auto,
    Any,
    Tool { name: &'a str },
}

// We map our ToolChoice enum to the Anthropic one that serializes properly
impl<'a> TryFrom<&'a ToolChoice> for GCPVertexAnthropicToolChoice<'a> {
    type Error = Error;
    fn try_from(tool_choice: &'a ToolChoice) -> Result<Self, Error> {
        match tool_choice {
            ToolChoice::Auto => Ok(GCPVertexAnthropicToolChoice::Auto),
            ToolChoice::Required => Ok(GCPVertexAnthropicToolChoice::Any),
            ToolChoice::Specific(name) => Ok(GCPVertexAnthropicToolChoice::Tool { name }),
            // Workaround for Anthropic API limitation: they don't support explicitly specifying "none"
            // for tool choice. Instead, we return Auto but the request construction will ensure
            // that no tools are sent in the request payload. This achieves the same effect
            // as explicitly telling the model not to use tools, since without any tools
            // being provided, the model cannot make tool calls.
            ToolChoice::None => Ok(GCPVertexAnthropicToolChoice::Auto),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct GCPVertexAnthropicTool<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    input_schema: &'a Value,
}

impl<'a> From<&'a ToolConfig> for GCPVertexAnthropicTool<'a> {
    fn from(value: &'a ToolConfig) -> Self {
        // In case we add more tool types in the future, the compiler will complain here.
        GCPVertexAnthropicTool {
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
enum GCPVertexAnthropicMessageContent<'a> {
    Text {
        text: &'a str,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: Vec<GCPVertexAnthropicMessageContent<'a>>,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: Value,
    },
}

impl<'a> TryFrom<&'a ContentBlock> for GCPVertexAnthropicMessageContent<'a> {
    type Error = Error;

    fn try_from(block: &'a ContentBlock) -> Result<Self, Self::Error> {
        match block {
            ContentBlock::Text(Text { text }) => {
                Ok(GCPVertexAnthropicMessageContent::Text { text })
            }
            ContentBlock::ToolCall(tool_call) => {
                // Convert the tool call arguments from String to JSON Value (Anthropic expects an object)
                let input: Value = serde_json::from_str(&tool_call.arguments).map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: format!("Error parsing tool call arguments as JSON Value: {e}"),
                        provider_type: "GCP Vertex Anthropic".to_string(),
                    })
                })?;

                if !input.is_object() {
                    return Err(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: "Tool call arguments must be a JSON object".to_string(),
                        provider_type: "GCP Vertex Anthropic".to_string(),
                    }
                    .into());
                }

                Ok(GCPVertexAnthropicMessageContent::ToolUse {
                    id: &tool_call.id,
                    name: &tool_call.name,
                    input,
                })
            }
            ContentBlock::ToolResult(tool_result) => {
                Ok(GCPVertexAnthropicMessageContent::ToolResult {
                    tool_use_id: &tool_result.id,
                    content: vec![GCPVertexAnthropicMessageContent::Text {
                        text: &tool_result.result,
                    }],
                })
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct GCPVertexAnthropicMessage<'a> {
    role: GCPVertexAnthropicRole,
    content: Vec<GCPVertexAnthropicMessageContent<'a>>,
}

impl<'a> TryFrom<&'a RequestMessage> for GCPVertexAnthropicMessage<'a> {
    type Error = Error;

    fn try_from(
        inference_message: &'a RequestMessage,
    ) -> Result<GCPVertexAnthropicMessage<'a>, Self::Error> {
        let content: Vec<GCPVertexAnthropicMessageContent> = inference_message
            .content
            .iter()
            .map(|block| block.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(GCPVertexAnthropicMessage {
            role: inference_message.role.into(),
            content,
        })
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexAnthropicRequestBody<'a> {
    anthropic_version: &'static str,
    messages: Vec<GCPVertexAnthropicMessage<'a>>,
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
    tool_choice: Option<GCPVertexAnthropicToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GCPVertexAnthropicTool<'a>>>,
}

impl<'a> GCPVertexAnthropicRequestBody<'a> {
    fn new(request: &'a ModelInferenceRequest) -> Result<GCPVertexAnthropicRequestBody<'a>, Error> {
        if request.messages.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message: "Anthropic requires at least one message".to_string(),
            }
            .into());
        }
        let system = request.system.as_deref();
        let request_messages: Vec<GCPVertexAnthropicMessage> = request
            .messages
            .iter()
            .map(GCPVertexAnthropicMessage::try_from)
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

        // Workaround for GCP Vertex AI Anthropic API limitation: they don't support explicitly specifying "none"
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
        let tool_choice: Option<GCPVertexAnthropicToolChoice> = tools
            .as_ref()
            .filter(|t| !t.is_empty())
            .and(request.tool_config.as_ref())
            .and_then(|c| (&c.tool_choice).try_into().ok());
        // NOTE: Anthropic does not support seed
        Ok(GCPVertexAnthropicRequestBody {
            anthropic_version: ANTHROPIC_API_VERSION,
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

/// Anthropic API doesn't support consecutive messages from the same role.
/// This function consolidates messages from the same role into a single message
/// so as to satisfy the API.
/// It also makes modifications to the messages to make Anthropic happy.
/// For example, it will prepend a default User message if the first message is an Assistant message.
/// It will also append a default User message if the last message is an Assistant message.
fn prepare_messages(
    messages: Vec<GCPVertexAnthropicMessage>,
) -> Result<Vec<GCPVertexAnthropicMessage>, Error> {
    let mut consolidated_messages: Vec<GCPVertexAnthropicMessage> = Vec::new();
    let mut last_role: Option<GCPVertexAnthropicRole> = None;
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
        last_role = Some(this_role)
    }
    // Anthropic also requires that there is at least one message and it is a User message.
    // If it's not we will prepend a default User message.
    match consolidated_messages.first() {
        Some(&GCPVertexAnthropicMessage {
            role: GCPVertexAnthropicRole::User,
            ..
        }) => {}
        _ => {
            consolidated_messages.insert(
                0,
                GCPVertexAnthropicMessage {
                    role: GCPVertexAnthropicRole::User,
                    content: vec![GCPVertexAnthropicMessageContent::Text {
                        text: "[listening]",
                    }],
                },
            );
        }
    }
    // Anthropic will continue any assistant messages passed in.
    // Since we don't want to do that, we'll append a default User message in the case that the last message was
    // an assistant message
    if let Some(last_message) = consolidated_messages.last() {
        if last_message.role == GCPVertexAnthropicRole::Assistant {
            consolidated_messages.push(GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text {
                    text: "[listening]",
                }],
            });
        }
    }
    Ok(consolidated_messages)
}

fn prefill_json_message(
    mut messages: Vec<GCPVertexAnthropicMessage>,
) -> Vec<GCPVertexAnthropicMessage> {
    // Add a JSON-prefill message for Anthropic's JSON mode
    messages.push(GCPVertexAnthropicMessage {
        role: GCPVertexAnthropicRole::Assistant,
        content: vec![GCPVertexAnthropicMessageContent::Text {
            text: "Here is the JSON requested:\n{",
        }],
    });
    messages
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct GCPVertexAnthropicError {
    error: GCPVertexAnthropicErrorBody,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct GCPVertexAnthropicErrorBody {
    r#type: String,
    message: String,
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
}

impl TryFrom<GCPVertexAnthropicContentBlock> for ContentBlock {
    type Error = Error;
    fn try_from(block: GCPVertexAnthropicContentBlock) -> Result<Self, Self::Error> {
        match block {
            GCPVertexAnthropicContentBlock::Text { text } => Ok(text.into()),
            GCPVertexAnthropicContentBlock::ToolUse { id, name, input } => {
                Ok(ContentBlock::ToolCall(ToolCall {
                    id,
                    name,
                    arguments: serde_json::to_string(&input).map_err(|e| {
                        Error::new(ErrorDetails::InferenceServer {
                            message: format!("Error parsing input for tool call: {e}"),
                            provider_type: "GCP Vertex Anthropic".to_string(),
                        })
                    })?,
                }))
            }
        }
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
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct GCPVertexAnthropicResponse {
    id: String,
    r#type: String, // this is always "message"
    role: String,   // this is always "assistant"
    content: Vec<GCPVertexAnthropicContentBlock>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<String>,
    usage: GCPVertexAnthropic,
}

#[derive(Debug, PartialEq)]
struct GCPVertexAnthropicResponseWithMetadata<'a> {
    response: GCPVertexAnthropicResponse,
    latency: Latency,
    request: GCPVertexAnthropicRequestBody<'a>,
    function_type: &'a FunctionType,
    json_mode: &'a ModelInferenceRequestJsonMode,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<GCPVertexAnthropicResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: GCPVertexAnthropicResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let GCPVertexAnthropicResponseWithMetadata {
            response,
            latency,
            request,
            function_type,
            json_mode,
            generic_request,
        } = value;

        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing response from GCP Vertex Anthropic: {e}"),
                provider_type: "GCP Vertex Anthropic".to_string(),
            })
        })?;

        let content: Vec<ContentBlock> = response
            .content
            .into_iter()
            .map(|block| block.try_into())
            .collect::<Result<Vec<_>, _>>()?;
        let raw_request = serde_json::to_string(&request).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request to GCP Vertex Anthropic: {e}"),
                provider_type: "GCP Vertex Anthropic".to_string(),
            })
        })?;

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
            content,
            system,
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
    response_body: GCPVertexAnthropicErrorBody,
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED
        | StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::TOO_MANY_REQUESTS => Err(ErrorDetails::InferenceClient {
            message: response_body.message,
            status_code: Some(response_code),
            provider_type: "GCP Vertex Anthropic".to_string(),
        }
        .into()),
        // StatusCode::NOT_FOUND | StatusCode::FORBIDDEN | StatusCode::INTERNAL_SERVER_ERROR | 529: Overloaded
        // These are all captured in _ since they have the same error behavior
        _ => Err(ErrorDetails::InferenceServer {
            message: response_body.message,
            provider_type: "GCP Vertex Anthropic".to_string(),
        }
        .into()),
    }
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GCPVertexAnthropicMessageBlock {
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
enum GCPVertexAnthropicStreamMessage {
    ContentBlockDelta {
        delta: GCPVertexAnthropicMessageBlock,
        index: u32,
    },
    ContentBlockStart {
        content_block: GCPVertexAnthropicMessageBlock,
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
    message: GCPVertexAnthropicStreamMessage,
    inference_id: Uuid,
    message_latency: Duration,
    current_tool_id: &mut Option<String>,
    current_tool_name: &mut Option<String>,
) -> Result<Option<ProviderInferenceResponseChunk>, Error> {
    let raw_message = serde_json::to_string(&message).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!("Error parsing response from Anthropic: {e}"),
            provider_type: "GCP Vertex Anthropic".to_string(),
        })
    })?;
    match message {
        GCPVertexAnthropicStreamMessage::ContentBlockDelta { delta, index } => match delta {
            GCPVertexAnthropicMessageBlock::TextDelta { text } => {
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
            GCPVertexAnthropicMessageBlock::InputJsonDelta { partial_json } => {
                Ok(Some(ProviderInferenceResponseChunk::new(
                    inference_id,
                    // Take the current tool name and ID and use them to create a ToolCallChunk
                    // This is necessary because the ToolCallChunk must always contain the tool name and ID
                    // even though Anthropic only sends the tool ID and name in the ToolUse chunk and not InputJSONDelta
                    vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                        raw_name: current_tool_name.clone().ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                            message: "Got InputJsonDelta chunk from Anthropic without current tool name being set by a ToolUse".to_string(),
                            provider_type: "GCP Vertex Anthropic".to_string()
                        }))?,
                        id: current_tool_id.clone().ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                            message: "Got InputJsonDelta chunk from Anthropic without current tool id being set by a ToolUse".to_string(),
                            provider_type: "GCP Vertex Anthropic".to_string()
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
                provider_type: "GCP Vertex Anthropic".to_string(),
            }
            .into()),
        },
        GCPVertexAnthropicStreamMessage::ContentBlockStart {
            content_block,
            index,
        } => match content_block {
            GCPVertexAnthropicMessageBlock::Text { text } => {
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
            GCPVertexAnthropicMessageBlock::ToolUse { id, name, .. } => {
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
                provider_type: "GCP Vertex Anthropic".to_string(),
            }
            .into()),
        },
        GCPVertexAnthropicStreamMessage::ContentBlockStop { .. } => Ok(None),
        GCPVertexAnthropicStreamMessage::Error { error } => Err(ErrorDetails::InferenceServer {
            message: error.to_string(),
            provider_type: "GCP Vertex Anthropic".to_string(),
        }
        .into()),
        GCPVertexAnthropicStreamMessage::MessageDelta { usage, .. } => {
            let usage = parse_usage_info(&usage);
            Ok(Some(ProviderInferenceResponseChunk::new(
                inference_id,
                vec![],
                Some(usage.into()),
                raw_message,
                message_latency,
            )))
        }
        GCPVertexAnthropicStreamMessage::MessageStart { message } => {
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
        GCPVertexAnthropicStreamMessage::MessageStop | GCPVertexAnthropicStreamMessage::Ping {} => {
            Ok(None)
        }
    }
}

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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    use serde_json::json;

    use crate::inference::providers::common::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::types::{FunctionType, ModelInferenceRequestJsonMode};
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::tool::{DynamicToolConfig, ToolConfig, ToolResult};

    #[test]
    fn test_try_from_tool_choice() {
        // Test conversion of ToolChoice::None - now maps to Auto
        let tool_choice = ToolChoice::None;
        let anthropic_tool_choice = GCPVertexAnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            GCPVertexAnthropicToolChoice::Auto
        );

        let tool_choice = ToolChoice::Auto;
        let anthropic_tool_choice = GCPVertexAnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            GCPVertexAnthropicToolChoice::Auto
        );

        let tool_choice = ToolChoice::Required;
        let anthropic_tool_choice = GCPVertexAnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            GCPVertexAnthropicToolChoice::Any
        );

        let tool_choice = ToolChoice::Specific("test".to_string());
        let anthropic_tool_choice = GCPVertexAnthropicToolChoice::try_from(&tool_choice);
        assert!(anthropic_tool_choice.is_ok());
        assert_eq!(
            anthropic_tool_choice.unwrap(),
            GCPVertexAnthropicToolChoice::Tool { name: "test" }
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
        let anthropic_tool: GCPVertexAnthropicTool = (&tool).into();
        assert_eq!(
            anthropic_tool,
            GCPVertexAnthropicTool {
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
            GCPVertexAnthropicMessageContent::try_from(&text_content_block).unwrap();
        assert_eq!(
            anthropic_content_block,
            GCPVertexAnthropicMessageContent::Text { text: "test" }
        );

        let tool_call_content_block = ContentBlock::ToolCall(ToolCall {
            id: "test_id".to_string(),
            name: "test_name".to_string(),
            arguments: serde_json::to_string(&json!({"type": "string"})).unwrap(),
        });
        let anthropic_content_block =
            GCPVertexAnthropicMessageContent::try_from(&tool_call_content_block).unwrap();
        assert_eq!(
            anthropic_content_block,
            GCPVertexAnthropicMessageContent::ToolUse {
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
        let anthropic_message =
            GCPVertexAnthropicMessage::try_from(&inference_request_message).unwrap();
        assert_eq!(
            anthropic_message,
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "test" }],
            }
        );

        // Test an Assistant message
        let inference_request_message = RequestMessage {
            role: Role::Assistant,
            content: vec!["test_assistant".to_string().into()],
        };
        let anthropic_message =
            GCPVertexAnthropicMessage::try_from(&inference_request_message).unwrap();
        assert_eq!(
            anthropic_message,
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::Assistant,
                content: vec![GCPVertexAnthropicMessageContent::Text {
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
        let anthropic_message =
            GCPVertexAnthropicMessage::try_from(&inference_request_message).unwrap();
        assert_eq!(
            anthropic_message,
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::ToolResult {
                    tool_use_id: "test_tool_call_id",
                    content: vec![GCPVertexAnthropicMessageContent::Text {
                        text: "test_tool_response"
                    }],
                }],
            }
        );
    }

    #[test]
    fn test_initialize_anthropic_request_body() {
        let listening_message = GCPVertexAnthropicMessage {
            role: GCPVertexAnthropicRole::User,
            content: vec![GCPVertexAnthropicMessageContent::Text {
                text: "[listening]",
            }],
        };
        // Test Case 1: Empty message list
        let inference_request = ModelInferenceRequest {
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
        };
        let anthropic_request_body = GCPVertexAnthropicRequestBody::new(&inference_request);
        let details = anthropic_request_body.unwrap_err().get_owned_details();
        assert_eq!(
            details,
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
        };
        let anthropic_request_body = GCPVertexAnthropicRequestBody::new(&inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            GCPVertexAnthropicRequestBody {
                anthropic_version: ANTHROPIC_API_VERSION,
                messages: vec![
                    GCPVertexAnthropicMessage::try_from(&messages[0]).unwrap(),
                    GCPVertexAnthropicMessage::try_from(&messages[1]).unwrap(),
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
        };
        let anthropic_request_body = GCPVertexAnthropicRequestBody::new(&inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            GCPVertexAnthropicRequestBody {
                anthropic_version: ANTHROPIC_API_VERSION,
                messages: vec![
                    GCPVertexAnthropicMessage {
                        role: GCPVertexAnthropicRole::User,
                        content: vec![
                            GCPVertexAnthropicMessageContent::Text { text: "test_user" },
                            GCPVertexAnthropicMessageContent::Text { text: "test_user2" }
                        ],
                    },
                    GCPVertexAnthropicMessage::try_from(&messages[2]).unwrap(),
                    listening_message.clone(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some("test_system"),
                temperature: Some(0.5),
                top_p: Some(0.9),
                tool_choice: None,
                tools: None,
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
        };

        let anthropic_request_body = GCPVertexAnthropicRequestBody::new(&inference_request);
        assert!(anthropic_request_body.is_ok());
        assert_eq!(
            anthropic_request_body.unwrap(),
            GCPVertexAnthropicRequestBody {
                anthropic_version: ANTHROPIC_API_VERSION,
                messages: vec![
                    GCPVertexAnthropicMessage::try_from(&messages[0]).unwrap(),
                    GCPVertexAnthropicMessage::try_from(&messages[1]).unwrap(),
                    GCPVertexAnthropicMessage::try_from(&messages[2]).unwrap(),
                ],
                max_tokens: 100,
                stream: Some(true),
                system: Some("test_system"),
                temperature: Some(0.5),
                top_p: Some(0.9),
                tool_choice: Some(GCPVertexAnthropicToolChoice::Tool {
                    name: "get_temperature",
                }),
                tools: Some(vec![GCPVertexAnthropicTool {
                    name: WEATHER_TOOL.name(),
                    description: Some(WEATHER_TOOL.description()),
                    input_schema: WEATHER_TOOL.parameters(),
                }]),
            }
        );
    }

    #[test]
    fn test_consolidate_messages() {
        let listening_message = GCPVertexAnthropicMessage {
            role: GCPVertexAnthropicRole::User,
            content: vec![GCPVertexAnthropicMessageContent::Text {
                text: "[listening]",
            }],
        };
        // Test case 1: No consolidation needed
        let messages = vec![
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hello" }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::Assistant,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hi" }],
            },
        ];
        let expected = vec![
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hello" }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::Assistant,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hi" }],
            },
            listening_message.clone(),
        ];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 2: Consolidation needed
        let messages = vec![
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hello" }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text {
                    text: "How are you?",
                }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::Assistant,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hi" }],
            },
        ];
        let expected = vec![
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![
                    GCPVertexAnthropicMessageContent::Text { text: "Hello" },
                    GCPVertexAnthropicMessageContent::Text {
                        text: "How are you?",
                    },
                ],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::Assistant,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hi" }],
            },
            listening_message.clone(),
        ];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 3: Multiple consolidations needed
        let messages = vec![
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hello" }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text {
                    text: "How are you?",
                }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::Assistant,
                content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hi" }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::Assistant,
                content: vec![GCPVertexAnthropicMessageContent::Text {
                    text: "I am here to help.",
                }],
            },
        ];
        let expected = vec![
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![
                    GCPVertexAnthropicMessageContent::Text { text: "Hello" },
                    GCPVertexAnthropicMessageContent::Text {
                        text: "How are you?",
                    },
                ],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::Assistant,
                content: vec![
                    GCPVertexAnthropicMessageContent::Text { text: "Hi" },
                    GCPVertexAnthropicMessageContent::Text {
                        text: "I am here to help.",
                    },
                ],
            },
            listening_message.clone(),
        ];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 4: No messages
        let messages: Vec<GCPVertexAnthropicMessage> = vec![];
        let expected: Vec<GCPVertexAnthropicMessage> = vec![listening_message.clone()];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 5: Single message
        let messages = vec![GCPVertexAnthropicMessage {
            role: GCPVertexAnthropicRole::User,
            content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hello" }],
        }];
        let expected = vec![GCPVertexAnthropicMessage {
            role: GCPVertexAnthropicRole::User,
            content: vec![GCPVertexAnthropicMessageContent::Text { text: "Hello" }],
        }];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 6: Consolidate tool uses
        let messages = vec![
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![GCPVertexAnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::ToolResult {
                    tool_use_id: "tool2",
                    content: vec![GCPVertexAnthropicMessageContent::Text {
                        text: "Tool call 2",
                    }],
                }],
            },
        ];
        let expected = vec![GCPVertexAnthropicMessage {
            role: GCPVertexAnthropicRole::User,
            content: vec![
                GCPVertexAnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![GCPVertexAnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                },
                GCPVertexAnthropicMessageContent::ToolResult {
                    tool_use_id: "tool2",
                    content: vec![GCPVertexAnthropicMessageContent::Text {
                        text: "Tool call 2",
                    }],
                },
            ],
        }];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);

        // Test case 7: Consolidate mixed text and tool use
        let messages = vec![
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text {
                    text: "User message 1",
                }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![GCPVertexAnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                }],
            },
            GCPVertexAnthropicMessage {
                role: GCPVertexAnthropicRole::User,
                content: vec![GCPVertexAnthropicMessageContent::Text {
                    text: "User message 2",
                }],
            },
        ];
        let expected = vec![GCPVertexAnthropicMessage {
            role: GCPVertexAnthropicRole::User,
            content: vec![
                GCPVertexAnthropicMessageContent::Text {
                    text: "User message 1",
                },
                GCPVertexAnthropicMessageContent::ToolResult {
                    tool_use_id: "tool1",
                    content: vec![GCPVertexAnthropicMessageContent::Text {
                        text: "Tool call 1",
                    }],
                },
                GCPVertexAnthropicMessageContent::Text {
                    text: "User message 2",
                },
            ],
        }];
        assert_eq!(prepare_messages(messages.clone()).unwrap(), expected);
    }

    #[test]
    fn test_handle_anthropic_error() {
        let error_body = GCPVertexAnthropicErrorBody {
            r#type: "error".to_string(),
            message: "test_message".to_string(),
        };
        let response_code = StatusCode::BAD_REQUEST;
        let result = handle_anthropic_error(response_code, error_body.clone());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceClient {
                message: "test_message".to_string(),
                status_code: Some(response_code),
                provider_type: "GCP Vertex Anthropic".to_string()
            }
        );
        let response_code = StatusCode::UNAUTHORIZED;
        let result = handle_anthropic_error(response_code, error_body.clone());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceClient {
                message: "test_message".to_string(),
                status_code: Some(response_code),
                provider_type: "GCP Vertex Anthropic".to_string()
            }
        );
        let response_code = StatusCode::TOO_MANY_REQUESTS;
        let result = handle_anthropic_error(response_code, error_body.clone());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceClient {
                message: "test_message".to_string(),
                status_code: Some(response_code),
                provider_type: "GCP Vertex Anthropic".to_string()
            }
        );
        let response_code = StatusCode::NOT_FOUND;
        let result = handle_anthropic_error(response_code, error_body.clone());
        assert!(result.is_err());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceServer {
                message: "test_message".to_string(),
                provider_type: "GCP Vertex Anthropic".to_string()
            }
        );
        let response_code = StatusCode::INTERNAL_SERVER_ERROR;
        let result = handle_anthropic_error(response_code, error_body.clone());
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InferenceServer {
                message: "test_message".to_string(),
                provider_type: "GCP Vertex Anthropic".to_string()
            }
        );
    }

    #[test]
    fn test_anthropic_usage_to_usage() {
        let anthropic_usage = GCPVertexAnthropic {
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
        let anthropic_response_body = GCPVertexAnthropicResponse {
            id: "1".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![GCPVertexAnthropicContentBlock::Text {
                text: "Response text".to_string(),
            }],
            model: "model-name".to_string(),
            stop_reason: Some("stop reason".to_string()),
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
        };
        let request_body = GCPVertexAnthropicRequestBody {
            anthropic_version: "1.0",
            system: None,
            messages: vec![],
            stream: Some(false),
            max_tokens: 1000,
            temperature: None,
            top_p: None,
            tool_choice: None,
            tools: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = GCPVertexAnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            latency: latency.clone(),
            request: request_body,
            function_type: &FunctionType::Chat,
            json_mode: &ModelInferenceRequestJsonMode::Off,
            generic_request: &generic_request,
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
        assert_eq!(inference_response.system, Some("system".to_string()));
        assert_eq!(
            inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["Hello".to_string().into()],
            }]
        );
        // Test case 2: Tool call response
        let anthropic_response_body = GCPVertexAnthropicResponse {
            id: "2".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![GCPVertexAnthropicContentBlock::ToolUse {
                id: "tool_call_1".to_string(),
                name: "get_temperature".to_string(),
                input: json!({"location": "New York"}),
            }],
            model: "model-name".to_string(),
            stop_reason: Some("tool_call".to_string()),
            stop_sequence: None,
            usage: GCPVertexAnthropic {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let generic_request = ModelInferenceRequest {
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
        };
        let request_body = GCPVertexAnthropicRequestBody {
            anthropic_version: "1.0",
            system: None,
            messages: vec![],
            stream: Some(false),
            max_tokens: 1000,
            temperature: None,
            top_p: None,
            tool_choice: None,
            tools: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = GCPVertexAnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            latency: latency.clone(),
            request: request_body,
            function_type: &FunctionType::Chat,
            json_mode: &ModelInferenceRequestJsonMode::Off,
            generic_request: &generic_request,
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
                GCPVertexAnthropicContentBlock::Text {
                    text: "Here's the weather:".to_string(),
                },
                GCPVertexAnthropicContentBlock::ToolUse {
                    id: "tool_call_2".to_string(),
                    name: "get_temperature".to_string(),
                    input: json!({"location": "London"}),
                },
            ],
            model: "model-name".to_string(),
            stop_reason: None,
            stop_sequence: None,
            usage: GCPVertexAnthropic {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let generic_request = ModelInferenceRequest {
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
        };
        let request_body = GCPVertexAnthropicRequestBody {
            anthropic_version: "1.0",
            system: None,
            messages: vec![],
            stream: Some(false),
            max_tokens: 1000,
            temperature: None,
            top_p: None,
            tool_choice: None,
            tools: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let body_with_latency = GCPVertexAnthropicResponseWithMetadata {
            response: anthropic_response_body.clone(),
            latency: latency.clone(),
            request: request_body,
            function_type: &FunctionType::Chat,
            json_mode: &ModelInferenceRequestJsonMode::Off,
            generic_request: &generic_request,
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
    fn test_anthropic_to_tensorzero_stream_message() {
        use serde_json::json;
        use uuid::Uuid;

        let inference_id = Uuid::now_v7();

        // Test ContentBlockDelta with TextDelta
        let mut current_tool_id = None;
        let mut current_tool_name = None;
        let content_block_delta = GCPVertexAnthropicStreamMessage::ContentBlockDelta {
            delta: GCPVertexAnthropicMessageBlock::TextDelta {
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
        let content_block_delta = GCPVertexAnthropicStreamMessage::ContentBlockDelta {
            delta: GCPVertexAnthropicMessageBlock::InputJsonDelta {
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
                provider_type: "GCP Vertex Anthropic".to_string()
            }
        );

        // Test ContentBlockDelta with InputJsonDelta and previous tool info
        let mut current_tool_id = Some("tool_id".to_string());
        let mut current_tool_name = Some("tool_name".to_string());
        let content_block_delta = GCPVertexAnthropicStreamMessage::ContentBlockDelta {
            delta: GCPVertexAnthropicMessageBlock::InputJsonDelta {
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
        let content_block_start = GCPVertexAnthropicStreamMessage::ContentBlockStart {
            content_block: GCPVertexAnthropicMessageBlock::ToolUse {
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
        let content_block_start = GCPVertexAnthropicStreamMessage::ContentBlockStart {
            content_block: GCPVertexAnthropicMessageBlock::Text {
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
        let content_block_start = GCPVertexAnthropicStreamMessage::ContentBlockStart {
            content_block: GCPVertexAnthropicMessageBlock::InputJsonDelta {
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
                provider_type: "GCP Vertex Anthropic".to_string()
            }
        );

        // Test ContentBlockStop
        let content_block_stop = GCPVertexAnthropicStreamMessage::ContentBlockStop { index: 2 };
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
        let error_message = GCPVertexAnthropicStreamMessage::Error {
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
                provider_type: "GCP Vertex Anthropic".to_string(),
            }
        );

        // Test MessageDelta with usage
        let message_delta = GCPVertexAnthropicStreamMessage::MessageDelta {
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
        let message_start = GCPVertexAnthropicStreamMessage::MessageStart {
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
        let message_stop = GCPVertexAnthropicStreamMessage::MessageStop;
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
        let ping = GCPVertexAnthropicStreamMessage::Ping {};
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
    fn test_prefill_json_message() {
        let input_messages = vec![GCPVertexAnthropicMessage {
            role: GCPVertexAnthropicRole::User,
            content: vec![GCPVertexAnthropicMessageContent::Text {
                text: "Generate some JSON",
            }],
        }];

        let result = prefill_json_message(input_messages);

        assert_eq!(result.len(), 2);

        assert_eq!(result[0].role, GCPVertexAnthropicRole::User);
        assert_eq!(
            result[0].content,
            vec![GCPVertexAnthropicMessageContent::Text {
                text: "Generate some JSON",
            }]
        );

        assert_eq!(result[1].role, GCPVertexAnthropicRole::Assistant);
        assert_eq!(
            result[1].content,
            vec![GCPVertexAnthropicMessageContent::Text {
                text: "Here is the JSON requested:\n{",
            }]
        );
    }
}
