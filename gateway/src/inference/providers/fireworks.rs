use futures::{StreamExt, TryStreamExt};
use reqwest_eventsource::RequestBuilderExt;
use secrecy::ExposeSecret;
use serde::Serialize;
use serde_json::Value;
use tokio::time::Instant;

use crate::{
    error::Error,
    inference::types::{
        InferenceResponseStream, Latency, ModelInferenceRequest, ModelInferenceResponse,
        ModelInferenceResponseChunk,
    },
    model::ProviderConfig,
};

use super::{
    openai::{
        get_chat_url, handle_openai_error, stream_openai, OpenAIRequestMessage, OpenAIResponse,
        OpenAIResponseWithLatency, OpenAITool, OpenAIToolChoice,
    },
    provider_trait::InferenceProvider,
};

pub struct FireworksProvider;

/// Key differences between Fireworks and OpenAI inference:
/// - Fireworks allows you to specify output format in JSON mode
/// - Fireworks automatically returns usage in streaming inference, we don't have to ask for it
/// - Fireworks allows you to auto-truncate requests that have too many tokens
///   (there are 2 ways to do it, we have the default of auto-truncation to the max window size)
impl InferenceProvider for FireworksProvider {
    async fn infer<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let (model_name, api_key) = match model {
            ProviderConfig::Fireworks {
                model_name,
                api_key,
            } => (
                model_name,
                api_key.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "Fireworks".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected Fireworks provider config".to_string(),
                })
            }
        };
        let api_base = Some("https://api.fireworks.ai/inference/v1/");
        let request_body = FireworksRequest::new(model_name, request);
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
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
                message: format!("Error sending request to Fireworks: {e}"),
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response_body =
                res.json::<OpenAIResponse>()
                    .await
                    .map_err(|e| Error::FireworksServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            Ok(OpenAIResponseWithLatency {
                response: response_body,
                latency,
            }
            .try_into()
            .map_err(map_openai_to_fireworks_error)?)
        } else {
            handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::FireworksServer {
                    message: format!("Error parsing error response: {e}"),
                })?,
            )
            .map_err(map_openai_to_fireworks_error)
        }
    }

    async fn infer_stream<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let (model_name, api_key) = match model {
            ProviderConfig::Fireworks {
                model_name,
                api_key,
            } => (
                model_name,
                api_key.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "Fireworks".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected Fireworks provider config".to_string(),
                })
            }
        };
        let request_body = FireworksRequest::new(model_name, request);
        let api_base = Some("https://api.fireworks.ai/inference/v1/");
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
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
                message: format!("Error sending request to Fireworks: {e}"),
            })?;
        let mut stream = Box::pin(
            stream_openai(event_source, start_time).map_err(map_openai_to_fireworks_error),
        );
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::FireworksServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };
        Ok((chunk, stream))
    }
}

fn map_openai_to_fireworks_error(e: Error) -> Error {
    match e {
        Error::OpenAIServer { message } => Error::FireworksServer { message },
        Error::OpenAIClient {
            message,
            status_code,
        } => Error::FireworksClient {
            message,
            status_code,
        },
        _ => e,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum FireworksResponseFormat<'a> {
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<&'a Value>, // the desired JSON schema
    },
    #[default]
    Text,
}

/// This struct defines the supported parameters for the Fireworks inference API
/// See the [Fireworks API documentation](https://docs.fireworks.ai/api-reference/post-chatcompletions)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, seed, service_tier, stop, user,
/// or context_length_exceeded_behavior
#[derive(Serialize)]
struct FireworksRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<FireworksResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> FireworksRequest<'a> {
    pub fn new(model: &'a str, request: &'a ModelInferenceRequest) -> FireworksRequest<'a> {
        let tools = request
            .tools_available
            .as_ref()
            .map(|t| t.iter().map(|t| t.into()).collect());
        // NB: Fireworks will throw an error if you give FireworksResponseFormat::Text and then also include tools.
        // So we just don't include it as Text is the same as None anyway.
        let response_format = match request.json_mode {
            true => Some(FireworksResponseFormat::JsonObject {
                schema: request.output_schema,
            }),
            false => None,
        };
        FireworksRequest {
            messages: request.messages.iter().map(|m| m.into()).collect(),
            model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream,
            response_format,
            tools,
            tool_choice: request.tool_choice.as_ref().map(OpenAIToolChoice::from),
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
        types::{
            FunctionType, InferenceRequestMessage, Tool, ToolChoice, ToolType,
            UserInferenceRequestMessage,
        },
    };

    #[test]
    fn test_fireworks_request_new() {
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

        let fireworks_request =
            FireworksRequest::new("accounts/fireworks/models/llama-v3-8b", &request_with_tools);

        assert_eq!(
            fireworks_request.model,
            "accounts/fireworks/models/llama-v3-8b"
        );
        assert_eq!(fireworks_request.messages.len(), 1);
        assert_eq!(fireworks_request.temperature, None);
        assert_eq!(fireworks_request.max_tokens, None);
        assert!(!fireworks_request.stream);
        assert_eq!(
            fireworks_request.response_format,
            Some(FireworksResponseFormat::JsonObject {
                schema: request_with_tools.output_schema,
            })
        );
        assert!(fireworks_request.tools.is_some());
        assert_eq!(fireworks_request.tools.as_ref().unwrap().len(), 1);
        assert_eq!(
            fireworks_request.tool_choice,
            Some(OpenAIToolChoice::String(OpenAIToolChoiceString::Auto))
        );
        assert_eq!(fireworks_request.parallel_tool_calls, Some(true));
    }
}
