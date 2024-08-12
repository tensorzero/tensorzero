use futures::{StreamExt, TryStreamExt};
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use serde_json::Value;
use tokio::time::Instant;

use crate::{
    error::Error,
    inference::types::{
        InferenceResponseStream, JSONMode, Latency, ModelInferenceRequest, ModelInferenceResponse,
        ModelInferenceResponseChunk,
    },
};

use super::{
    openai::{
        get_chat_url, handle_openai_error, prepare_openai_messages, prepare_openai_tools,
        stream_openai, OpenAIRequestMessage, OpenAIResponse, OpenAIResponseWithLatency, OpenAITool,
        OpenAIToolChoice,
    },
    provider_trait::InferenceProvider,
};

#[derive(Clone, Debug)]
pub struct FireworksProvider {
    pub model_name: String,
    pub api_key: Option<SecretString>,
}

/// Key differences between Fireworks and OpenAI inference:
/// - Fireworks allows you to specify output format in JSON mode
/// - Fireworks automatically returns usage in streaming inference, we don't have to ask for it
/// - Fireworks allows you to auto-truncate requests that have too many tokens
///   (there are 2 ways to do it, we have the default of auto-truncation to the max window size)
impl InferenceProvider for FireworksProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Fireworks".to_string(),
        })?;
        let api_base = Some("https://api.fireworks.ai/inference/v1/");
        let request_body = FireworksRequest::new(&self.model_name, request);
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
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
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Fireworks".to_string(),
        })?;
        let request_body = FireworksRequest::new(&self.model_name, request);
        let api_base = Some("https://api.fireworks.ai/inference/v1/");
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
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
        // NB: Fireworks will throw an error if you give FireworksResponseFormat::Text and then also include tools.
        // So we just don't include it as Text is the same as None anyway.
        let response_format = match request.json_mode {
            JSONMode::On | JSONMode::Strict => Some(FireworksResponseFormat::JsonObject {
                schema: request.output_schema,
            }),
            JSONMode::Off => None,
        };
        let messages = prepare_openai_messages(request);
        let (tools, tool_choice) = prepare_openai_tools(request);
        FireworksRequest {
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
        types::{FunctionType, RequestMessage, Role},
    };
    use crate::tool::{Tool, ToolChoice};

    #[test]
    fn test_fireworks_request_new() {
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
            system: None,
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
