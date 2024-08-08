use futures::{StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::ExposeSecret;
use serde::Serialize;
use serde_json::Value;
use tokio::time::Instant;

use crate::{
    error::Error,
    inference::types::{
        InferenceResponseStream, JSONMode, Latency, ModelInferenceRequest, ModelInferenceResponse,
        ModelInferenceResponseChunk,
    },
    model::ProviderConfig,
};

use super::{
    openai::{
        get_chat_url, handle_openai_error, prepare_openai_messages, prepare_openai_tools,
        stream_openai, OpenAIRequestMessage, OpenAIResponse, OpenAIResponseWithLatency, OpenAITool,
        OpenAIToolChoice,
    },
    provider_trait::InferenceProvider,
};

pub struct TogetherProvider;

// TODO (#80): Add support for Llama 3.1 function calling as discussed [here](https://docs.together.ai/docs/llama-3-function-calling)

impl InferenceProvider for TogetherProvider {
    async fn infer<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let (model_name, api_key) = match model {
            ProviderConfig::Together {
                model_name,
                api_key,
            } => (
                model_name,
                api_key.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "Together".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected Together provider config".to_string(),
                })
            }
        };
        let api_base = Some("https://api.together.xyz/v1");
        let request_body = TogetherRequest::new(model_name, request);
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::TogetherClient {
                message: format!("{e}"),
                status_code: e.status().unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            })?;
        if res.status().is_success() {
            let response_body =
                res.json::<OpenAIResponse>()
                    .await
                    .map_err(|e| Error::TogetherServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            Ok(OpenAIResponseWithLatency {
                response: response_body,
                latency: Latency::NonStreaming {
                    response_time: start_time.elapsed(),
                },
            }
            .try_into()
            .map_err(map_openai_to_together_error)?)
        } else {
            handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::TogetherServer {
                    message: format!("Error parsing error response: {e}"),
                })?,
            )
            .map_err(map_openai_to_together_error)
        }
    }

    async fn infer_stream<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let (model_name, api_key) = match model {
            ProviderConfig::Together {
                model_name,
                api_key,
            } => (
                model_name,
                api_key.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "Together".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected Together provider config".to_string(),
                })
            }
        };
        let request_body = TogetherRequest::new(model_name, request);
        let api_base = Some("https://api.together.xyz/v1");
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to Together: {e}"),
            })?;
        let mut stream =
            Box::pin(stream_openai(event_source, start_time).map_err(map_openai_to_together_error));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::TogetherServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };
        Ok((chunk, stream))
    }
}

fn map_openai_to_together_error(e: Error) -> Error {
    match e {
        Error::OpenAIServer { message } => Error::TogetherServer { message },
        Error::OpenAIClient {
            message,
            status_code,
        } => Error::TogetherClient {
            message,
            status_code,
        },
        _ => e,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum TogetherResponseFormat<'a> {
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<&'a Value>, // the desired JSON schema
    },
}

/// This struct defines the supported parameters for the Together inference API
/// See the [Together API documentation](https://docs.together.ai/docs/chat-overview)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, seed, service_tier, stop, user,
/// or context_length_exceeded_behavior
#[derive(Serialize)]
struct TogetherRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<TogetherResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> TogetherRequest<'a> {
    pub fn new(model: &'a str, request: &'a ModelInferenceRequest) -> TogetherRequest<'a> {
        let response_format = match request.json_mode {
            JSONMode::On | JSONMode::Strict => Some(TogetherResponseFormat::JsonObject {
                schema: request.output_schema,
            }),
            JSONMode::Off => None,
        };
        let messages = prepare_openai_messages(request);
        let (tools, tool_choice) = prepare_openai_tools(request);
        TogetherRequest {
            messages,
            model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls: request.parallel_tool_calls,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::inference::{
        providers::openai::OpenAIToolChoiceString,
        types::{FunctionType, RequestMessage, Role, Tool, ToolChoice},
    };

    #[test]
    fn test_together_request_new() {
        let tool = Tool::Function {
            name: "get_weather".to_string(),
            description: Some("Get the current weather".to_string()),
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
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system_instructions: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: JSONMode::On,
            tools_available: Some(vec![tool]),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: Some(true),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let together_request =
            TogetherRequest::new("togethercomputer/llama-v3-8b", &request_with_tools);

        assert_eq!(together_request.model, "togethercomputer/llama-v3-8b");
        assert_eq!(together_request.messages.len(), 1);
        assert_eq!(together_request.temperature, None);
        assert_eq!(together_request.max_tokens, None);
        assert!(!together_request.stream);
        assert_eq!(
            together_request.response_format,
            Some(TogetherResponseFormat::JsonObject {
                schema: request_with_tools.output_schema,
            })
        );
        assert!(together_request.tools.is_some());
        assert_eq!(together_request.tools.as_ref().unwrap().len(), 1);
        assert_eq!(
            together_request.tool_choice,
            Some(OpenAIToolChoice::String(OpenAIToolChoiceString::Auto))
        );
        assert_eq!(together_request.parallel_tool_calls, Some(true));
    }
}
