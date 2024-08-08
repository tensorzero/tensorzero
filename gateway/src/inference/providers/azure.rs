use futures::{StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use tokio::time::Instant;

use crate::error::Error;
use crate::inference::types::{
    InferenceResponseStream, Latency, ModelInferenceRequest, ModelInferenceResponse,
    ModelInferenceResponseChunk,
};

use super::openai::{
    handle_openai_error, stream_openai, OpenAIRequestMessage, OpenAIResponse,
    OpenAIResponseWithLatency, OpenAITool, OpenAIToolChoice,
};
use super::provider_trait::InferenceProvider;

#[derive(Clone, Debug)]
pub struct AzureProvider {
    pub model_name: String,
    pub api_base: String,
    pub deployment_id: String,
    pub api_key: Option<SecretString>,
}

impl InferenceProvider for AzureProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Azure".to_string(),
        })?;
        let request_body = AzureRequest::new(&self.model_name, request);
        let request_url = get_azure_chat_url(&self.api_base, &self.deployment_id);
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header("api-key", api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::AzureClient {
                message: e.to_string(),
                status_code: e.status().unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            })?;
        if res.status().is_success() {
            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            let response_body =
                res.json::<OpenAIResponse>()
                    .await
                    .map_err(|e| Error::AzureServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            Ok(OpenAIResponseWithLatency {
                response: response_body,
                latency,
            }
            .try_into()
            .map_err(map_openai_to_azure_error)?)
        } else {
            handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::AzureServer {
                    message: format!("Error parsing error response: {e}"),
                })?,
            )
            .map_err(map_openai_to_azure_error)
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Azure".to_string(),
        })?;
        let request_body = AzureRequest::new(&self.model_name, request);
        let request_url = get_azure_chat_url(&self.api_base, &self.deployment_id);
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header("api-key", api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to Azure: {e}"),
            })?;
        let mut stream =
            Box::pin(stream_openai(event_source, start_time).map_err(map_openai_to_azure_error));
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

fn map_openai_to_azure_error(e: Error) -> Error {
    match e {
        Error::OpenAIServer { message } => Error::AzureServer { message },
        Error::OpenAIClient {
            message,
            status_code,
        } => Error::AzureClient {
            message,
            status_code,
        },
        _ => e,
    }
}

fn get_azure_chat_url(api_base: &str, deployment_id: &str) -> String {
    let api_version = "2024-02-01";
    format!(
        "{api_base}/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}"
    )
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum AzureResponseFormat {
    JsonObject,
    #[default]
    Text,
}

/// This struct defines the supported parameters for the Azure OpenAI inference API
/// See the [API documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, seed, service_tier, stop, user,
/// or context_length_exceeded_behavior
#[derive(Serialize)]
struct AzureRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    response_format: AzureResponseFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> AzureRequest<'a> {
    pub fn new(model: &'a str, request: &'a ModelInferenceRequest) -> AzureRequest<'a> {
        let response_format = match request.json_mode {
            true => AzureResponseFormat::JsonObject,
            false => AzureResponseFormat::Text,
        };
        AzureRequest {
            messages: request.messages.iter().map(|m| m.into()).collect(),
            model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream,
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
    fn test_azure_request_new() {
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

        let azure_request = AzureRequest::new("togethercomputer/llama-v3-8b", &request_with_tools);

        assert_eq!(azure_request.model, "togethercomputer/llama-v3-8b");
        assert_eq!(azure_request.messages.len(), 1);
        assert_eq!(azure_request.temperature, None);
        assert_eq!(azure_request.max_tokens, None);
        assert!(!azure_request.stream);
        assert_eq!(
            azure_request.response_format,
            AzureResponseFormat::JsonObject
        );
        assert!(azure_request.tools.is_some());
        assert_eq!(azure_request.tools.as_ref().unwrap().len(), 1);
        assert_eq!(
            azure_request.tool_choice,
            Some(OpenAIToolChoice::String(OpenAIToolChoiceString::Auto))
        );
        assert_eq!(azure_request.parallel_tool_calls, Some(true));
    }
}
