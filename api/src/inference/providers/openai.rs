use futures::StreamExt;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;
use uuid::Uuid;

use crate::config_parser::ProviderConfig;
use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    InferenceRequestMessage, InferenceResponseChunk, InferenceResponseStream,
    ModelInferenceRequest, ModelInferenceResponse, Tool, ToolCall, ToolCallChunk, ToolChoice,
    ToolType, Usage,
};

const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1/";

pub struct OpenAIProvider;

impl InferenceProvider for OpenAIProvider {
    async fn infer(
        &self,
        request: &ModelInferenceRequest,
        model: &ProviderConfig,
        http_client: &reqwest::Client,
        api_key: &SecretString,
    ) -> Result<ModelInferenceResponse, Error> {
        let (model_name, api_base) = match model {
            ProviderConfig::OpenAI {
                model_name,
                api_base,
            } => (model_name, api_base.as_deref()),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected OpenAI provider config".to_string(),
                })
            }
        };
        let request_body = OpenAIRequest::new(model_name, request);
        let request_url = get_chat_url(api_base)?;
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!("Bearer {}", api_key.expose_secret()),
            )
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to OpenAI: {e}"),
            })?;
        if res.status().is_success() {
            let response_body =
                res.json::<OpenAIResponse>()
                    .await
                    .map_err(|e| Error::OpenAIServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            Ok(response_body.try_into()?)
        } else {
            handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::OpenAIServer {
                    message: format!("Error parsing error response: {e}"),
                })?,
            )
        }
    }

    async fn infer_stream(
        &self,
        request: &ModelInferenceRequest,
        model: &ProviderConfig,
        http_client: &reqwest::Client,
        api_key: &SecretString,
    ) -> Result<InferenceResponseStream, Error> {
        let (model_name, api_base) = match model {
            ProviderConfig::OpenAI {
                model_name,
                api_base,
            } => (model_name, api_base.as_deref()),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected OpenAI provider config".to_string(),
                })
            }
        };
        let request_body = OpenAIRequest::new(model_name, request);
        let request_url = get_chat_url(api_base)?;
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!("Bearer {}", api_key.expose_secret()),
            )
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to OpenAI: {e}"),
            })?;
        Ok(stream_openai(event_source).await)
    }
}

async fn stream_openai(mut event_source: EventSource) -> InferenceResponseStream {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let inference_id = Uuid::now_v7();
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    if let Err(_e) = tx.send(Err(Error::AnthropicServer {
                        message: e.to_string(),
                    })) {
                        // rx dropped
                        break;
                    }
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        let data: Result<OpenAIChatChunk, Error> =
                            serde_json::from_str(&message.data).map_err(|e| Error::OpenAIServer {
                                message: format!(
                                    "Error parsing chunk. Error: {}, Data: {}",
                                    e, message.data
                                ),
                            });
                        let stream_message =
                            data.and_then(|d| openai_to_tensorzero_stream_message(d, inference_id));
                        if tx.send(stream_message).is_err() {
                            // rx dropped
                            break;
                        }
                    }
                },
            }
        }

        event_source.close();
    });

    Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}

fn get_chat_url(base_url: Option<&str>) -> Result<Url, Error> {
    let base_url = base_url.unwrap_or(OPENAI_DEFAULT_BASE_URL);
    let base_url = if base_url.ends_with('/') {
        base_url.to_string()
    } else {
        format!("{}/", base_url)
    };
    let url = Url::parse(&base_url)
        .map_err(|e| Error::InvalidBaseUrl {
            message: e.to_string(),
        })?
        .join("chat/completions")
        .map_err(|e| Error::InvalidBaseUrl {
            message: e.to_string(),
        })?;
    Ok(url)
}

fn handle_openai_error(
    response_code: StatusCode,
    response_body: &str,
) -> Result<ModelInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::TOO_MANY_REQUESTS => {
            Err(Error::OpenAIClient {
                message: response_body.to_string(),
                status_code: response_code,
            })
        }
        _ => Err(Error::OpenAIServer {
            message: response_body.to_string(),
        }),
    }
}

#[derive(Serialize, Debug, Clone, PartialEq)]
struct OpenAISystemRequestMessage<'a> {
    content: &'a str,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
struct OpenAIUserRequestMessage<'a> {
    content: &'a str, // TODO: this could be an array including images and stuff according to API spec, we don't support
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
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

#[derive(Serialize, Debug, Clone, PartialEq)]
struct OpenAIAssistantRequestMessage<'a> {
    content: Option<&'a str>,
    tool_calls: Option<Vec<OpenAIRequestToolCall<'a>>>,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
struct OpenAIToolRequestMessage<'a> {
    content: &'a str,
    tool_call_id: &'a str,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
enum OpenAIRequestMessage<'a> {
    System(OpenAISystemRequestMessage<'a>),
    User(OpenAIUserRequestMessage<'a>),
    Assistant(OpenAIAssistantRequestMessage<'a>),
    Tool(OpenAIToolRequestMessage<'a>),
}

impl<'a> From<&'a InferenceRequestMessage> for OpenAIRequestMessage<'a> {
    fn from(inference_message: &'a InferenceRequestMessage) -> Self {
        match inference_message {
            InferenceRequestMessage::System(message) => {
                OpenAIRequestMessage::System(OpenAISystemRequestMessage {
                    content: &message.content,
                })
            }
            InferenceRequestMessage::User(message) => {
                OpenAIRequestMessage::User(OpenAIUserRequestMessage {
                    content: &message.content,
                })
            }
            InferenceRequestMessage::Assistant(message) => {
                OpenAIRequestMessage::Assistant(OpenAIAssistantRequestMessage {
                    content: message.content.as_deref(),
                    tool_calls: message
                        .tool_calls
                        .as_ref()
                        .map(|t| t.iter().map(|t| t.into()).collect()),
                })
            }
            InferenceRequestMessage::Tool(message) => {
                OpenAIRequestMessage::Tool(OpenAIToolRequestMessage {
                    content: &message.content,
                    tool_call_id: &message.tool_call_id,
                })
            }
        }
    }
}

#[derive(Default, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum OpenAIResponseFormat {
    JsonObject,
    #[default]
    Text,
}

#[derive(Serialize, PartialEq, Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OpenAIToolType {
    Function,
}

// NB: if we ever add more complex tool types, should convert to references and lifetimes here
// as we do above.
impl From<ToolType> for OpenAIToolType {
    fn from(tool_type: ToolType) -> Self {
        match tool_type {
            ToolType::Function => OpenAIToolType::Function,
        }
    }
}

#[derive(Serialize, PartialEq, Debug)]
struct OpenAIFunction<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    parameters: &'a Value,
}

#[derive(Serialize, Debug, PartialEq)]
struct OpenAITool<'a> {
    r#type: OpenAIToolType,
    function: OpenAIFunction<'a>,
}

impl<'a> From<&'a Tool> for OpenAITool<'a> {
    fn from(tool: &'a Tool) -> Self {
        OpenAITool {
            r#type: tool.r#type.clone().into(),
            function: OpenAIFunction {
                name: &tool.name,
                description: tool.description.as_deref(),
                parameters: &tool.parameters,
            },
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
enum OpenAIToolChoice<'a> {
    String(OpenAIToolChoiceString),
    Specific(SpecificToolChoice<'a>),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
enum OpenAIToolChoiceString {
    None,
    Auto,
    Required,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
struct SpecificToolChoice<'a> {
    r#type: OpenAIToolType,
    function: SpecificToolFunction<'a>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct SpecificToolFunction<'a> {
    name: &'a str,
}

impl<'a> Default for OpenAIToolChoice<'a> {
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
            ToolChoice::Tool(tool_name) => OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction { name: tool_name },
            }),
        }
    }
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// This struct defines the supported parameters for the OpenAI API
/// See the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create)
/// for more details.
/// We are not handling logprobs, top_logprobs, max_tokens, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Serialize)]
struct OpenAIRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    response_format: OpenAIResponseFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> OpenAIRequest<'a> {
    pub fn new(model: &'a str, request: &'a ModelInferenceRequest) -> OpenAIRequest<'a> {
        let response_format = match request.json_mode {
            true => OpenAIResponseFormat::JsonObject,
            false => OpenAIResponseFormat::Text,
        };
        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };
        OpenAIRequest {
            messages: request.messages.iter().map(|m| m.into()).collect(),
            model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream,
            stream_options,
            response_format,
            tools: request
                .tools_available
                .as_ref()
                .map(|t| t.iter().map(|t| t.into()).collect()),
            tool_choice: request.tool_choice.as_ref().map(OpenAIToolChoice::from),
            parallel_tool_calls: request.parallel_tool_calls,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<OpenAIUsage> for Usage {
    fn from(usage: OpenAIUsage) -> Self {
        Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenAIResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenAIResponseToolCall {
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

impl From<OpenAIResponseToolCall> for ToolCallChunk {
    fn from(tool_call: OpenAIResponseToolCall) -> Self {
        ToolCallChunk {
            id: Some(tool_call.id),
            name: Some(tool_call.function.name),
            arguments: Some(tool_call.function.arguments),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct OpenAIResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

// Leaving out logprobs and finish_reason for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct OpenAIResponseChoice {
    index: u8,
    message: OpenAIResponseMessage,
}

// Leaving out id, created, model, service_tier, system_fingerprint, object for now
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct OpenAIResponse {
    choices: Vec<OpenAIResponseChoice>,
    usage: OpenAIUsage,
}

impl TryFrom<OpenAIResponse> for ModelInferenceResponse {
    type Error = Error;
    fn try_from(mut response: OpenAIResponse) -> Result<Self, Self::Error> {
        let raw = serde_json::to_string(&response).map_err(|e| Error::OpenAIServer {
            message: format!("Error parsing response: {e}"),
        })?;
        if response.choices.len() != 1 {
            return Err(Error::OpenAIServer {
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
            .ok_or(Error::OpenAIServer {
                message: "Response has no choices (this should never happen)".to_string(),
            })?
            .message;

        Ok(ModelInferenceResponse::new(
            message.content,
            message
                .tool_calls
                .map(|t| t.into_iter().map(|t| t.into()).collect()),
            raw,
            usage,
        ))
    }
}

// This doesn't include role
#[derive(Deserialize, Debug, PartialEq)]
struct OpenAIDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

// This doesn't include logprobs, finish_reason, and index
#[derive(Deserialize, Debug, PartialEq)]
struct OpenAIChatChunkChoice {
    delta: OpenAIDelta,
}

#[derive(Deserialize, Debug, PartialEq)]
struct OpenAIChatChunk {
    choices: Vec<OpenAIChatChunkChoice>,
    usage: Option<OpenAIUsage>,
}

fn openai_to_tensorzero_stream_message(
    mut chunk: OpenAIChatChunk,
    inference_id: Uuid,
) -> Result<InferenceResponseChunk, Error> {
    if chunk.choices.len() > 1 {
        return Err(Error::OpenAIServer {
            message: "Response has invalid number of choices: {}. Expected 1.".to_string(),
        });
    }
    let (content, tool_calls) = match chunk.choices.pop() {
        Some(choice) => (
            choice.delta.content,
            choice
                .delta
                .tool_calls
                .map(|t| t.into_iter().map(|t| t.into()).collect()),
        ),
        None => (None, None),
    };
    let usage = chunk.usage.map(|u| u.into());
    Ok(InferenceResponseChunk::new(
        inference_id,
        content,
        tool_calls,
        usage,
    ))
}

#[cfg(test)]
mod tests {
    use crate::inference::types::{
        AssistantInferenceRequestMessage, FunctionType, UserInferenceRequestMessage,
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
        let custom_url = get_chat_url(Some(custom_base)).unwrap();
        assert_eq!(
            custom_url.as_str(),
            "https://custom.openai.com/api/chat/completions"
        );

        // Test with invalid URL
        let invalid_url = get_chat_url(Some("not a url"));
        assert!(invalid_url.is_err());

        // Test with URL without trailing slash
        let unjoinable_url = get_chat_url(Some("https://example.com"));
        assert!(unjoinable_url.is_ok());
        assert_eq!(
            unjoinable_url.unwrap().as_str(),
            "https://example.com/chat/completions"
        );
        // Test with URL that can't be joined
        let unjoinable_url = get_chat_url(Some("https://example.com/foo"));
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
        assert!(matches!(unauthorized, Err(Error::OpenAIClient { .. })));
        if let Err(Error::OpenAIClient {
            message,
            status_code,
        }) = unauthorized
        {
            assert_eq!(message, "Unauthorized access");
            assert_eq!(status_code, StatusCode::UNAUTHORIZED);
        }

        // Test forbidden error
        let forbidden = handle_openai_error(StatusCode::FORBIDDEN, "Forbidden access");
        assert!(matches!(forbidden, Err(Error::OpenAIClient { .. })));
        if let Err(Error::OpenAIClient {
            message,
            status_code,
        }) = forbidden
        {
            assert_eq!(message, "Forbidden access");
            assert_eq!(status_code, StatusCode::FORBIDDEN);
        }

        // Test rate limit error
        let rate_limit = handle_openai_error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded");
        assert!(matches!(rate_limit, Err(Error::OpenAIClient { .. })));
        if let Err(Error::OpenAIClient {
            message,
            status_code,
        }) = rate_limit
        {
            assert_eq!(message, "Rate limit exceeded");
            assert_eq!(status_code, StatusCode::TOO_MANY_REQUESTS);
        }

        // Test server error
        let server_error = handle_openai_error(StatusCode::INTERNAL_SERVER_ERROR, "Server error");
        assert!(matches!(server_error, Err(Error::OpenAIServer { .. })));
        if let Err(Error::OpenAIServer { message }) = server_error {
            assert_eq!(message, "Server error");
        }
    }

    #[test]
    fn test_openai_request_new() {
        use crate::inference::types::{
            InferenceRequestMessage, ModelInferenceRequest, Tool, ToolChoice,
        };
        use serde_json::json;

        // Test basic request
        let basic_request = ModelInferenceRequest {
            messages: vec![
                InferenceRequestMessage::User(UserInferenceRequestMessage {
                    content: "Hello".to_string(),
                }),
                InferenceRequestMessage::Assistant(AssistantInferenceRequestMessage {
                    content: Some("Hi there!".to_string()),
                    tool_calls: None,
                }),
            ],
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: true,
            json_mode: false,
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let openai_request = OpenAIRequest::new("gpt-3.5-turbo", &basic_request);

        assert_eq!(openai_request.model, "gpt-3.5-turbo");
        assert_eq!(openai_request.messages.len(), 2);
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_tokens, Some(100));
        assert!(openai_request.stream);
        assert_eq!(openai_request.response_format, OpenAIResponseFormat::Text);
        assert!(openai_request.tools.is_none());
        assert!(openai_request.tool_choice.is_none());
        assert!(openai_request.parallel_tool_calls.is_none());

        // Test request with tools and JSON mode
        let tool = Tool {
            name: "get_weather".to_string(),
            description: Some("Get the current weather".to_string()),
            r#type: ToolType::Function,
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }),
        };

        let request_with_tools = ModelInferenceRequest {
            messages: vec![InferenceRequestMessage::User(UserInferenceRequestMessage {
                content: "What's the weather?".to_string(),
            })],
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: true,
            tools_available: Some(vec![tool]),
            tool_choice: Some(ToolChoice::Auto),
            parallel_tool_calls: Some(true),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let openai_request = OpenAIRequest::new("gpt-4", &request_with_tools);

        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.messages.len(), 1);
        assert_eq!(openai_request.temperature, None);
        assert_eq!(openai_request.max_tokens, None);
        assert!(!openai_request.stream);
        assert_eq!(
            openai_request.response_format,
            OpenAIResponseFormat::JsonObject
        );
        assert!(openai_request.tools.is_some());
        assert_eq!(openai_request.tools.as_ref().unwrap().len(), 1);
        assert_eq!(
            openai_request.tool_choice,
            Some(OpenAIToolChoice::String(OpenAIToolChoiceString::Auto))
        );
        assert_eq!(openai_request.parallel_tool_calls, Some(true));
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

        let result = ModelInferenceResponse::try_from(valid_response);
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(
            inference_response.content,
            Some("Hello, world!".to_string())
        );
        assert_eq!(inference_response.tool_calls, None);
        assert_eq!(inference_response.usage.prompt_tokens, 10);
        assert_eq!(inference_response.usage.completion_tokens, 20);

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

        let result = ModelInferenceResponse::try_from(valid_response_with_tools);
        assert!(result.is_ok());
        let inference_response = result.unwrap();
        assert_eq!(inference_response.content, None);
        assert!(inference_response.tool_calls.is_some());
        assert_eq!(inference_response.tool_calls.unwrap().len(), 1);
        assert_eq!(inference_response.usage.prompt_tokens, 15);
        assert_eq!(inference_response.usage.completion_tokens, 25);

        // Test case 3: Invalid response with no choices
        let invalid_response_no_choices = OpenAIResponse {
            choices: vec![],
            usage: OpenAIUsage {
                prompt_tokens: 5,
                completion_tokens: 0,
                total_tokens: 5,
            },
        };

        let result = ModelInferenceResponse::try_from(invalid_response_no_choices);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::OpenAIServer { .. }));

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

        let result = ModelInferenceResponse::try_from(invalid_response_multiple_choices);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::OpenAIServer { .. }));
    }
}
