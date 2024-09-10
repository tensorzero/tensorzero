use futures::{StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use tokio::time::Instant;
use url::Url;

use crate::error::Error;
use crate::inference::types::{
    Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode, ProviderInferenceResponse,
    ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
};

use super::openai::{
    handle_openai_error, prepare_openai_messages, prepare_openai_tools, stream_openai,
    OpenAIRequestMessage, OpenAIResponseWithMetadata, OpenAITool, OpenAIToolChoice,
    OpenAIToolChoiceString, SpecificToolChoice,
};
use super::provider_trait::InferenceProvider;

#[derive(Debug)]
pub struct AzureProvider {
    pub deployment_id: String,
    pub endpoint: Url,
    pub api_key: Option<SecretString>,
}

impl InferenceProvider for AzureProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<ProviderInferenceResponse, Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Azure".to_string(),
        })?;
        let request_body = AzureRequest::new(request);
        let raw_request = serde_json::to_string(&request_body).map_err(|e| Error::AzureServer {
            message: format!("Error serializing request body as JSON: {e}"),
        })?;
        let request_url = get_azure_chat_url(&self.endpoint, &self.deployment_id)?;
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

            let response = res.text().await.map_err(|e| Error::AnthropicServer {
                message: format!("Error parsing text response: {e}"),
            })?;

            let response = serde_json::from_str(&response).map_err(|e| Error::AnthropicServer {
                message: format!("Error parsing JSON response: {e}: {response}"),
            })?;

            Ok(OpenAIResponseWithMetadata {
                response,
                latency,
                raw_request,
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
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Azure".to_string(),
        })?;
        let request_body = AzureRequest::new(request);
        let raw_request = serde_json::to_string(&request_body).map_err(|e| Error::AzureServer {
            message: format!("Error serializing request body as JSON: {e}"),
        })?;
        let request_url = get_azure_chat_url(&self.endpoint, &self.deployment_id)?;
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
        Ok((chunk, stream, raw_request))
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

fn get_azure_chat_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .map_err(|e| Error::AzureServer {
            message: format!("Error parsing URL: {e:?}"),
        })?
        .push("openai")
        .push("deployments")
        .push(deployment_id)
        .push("chat")
        .push("completions");
    url.query_pairs_mut()
        .append_pair("api-version", "2024-06-01");
    Ok(url)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum AzureResponseFormat {
    #[allow(dead_code)]
    JsonObject,
    #[default]
    Text,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(untagged)]
enum AzureToolChoice<'a> {
    String(AzureToolChoiceString),
    Specific(SpecificToolChoice<'a>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum AzureToolChoiceString {
    None,
    Auto,
    // Note: Azure doesn't support required tool choice.
}

impl<'a> From<OpenAIToolChoice<'a>> for AzureToolChoice<'a> {
    fn from(tool_choice: OpenAIToolChoice<'a>) -> Self {
        match tool_choice {
            OpenAIToolChoice::String(tool_choice) => {
                match tool_choice {
                    OpenAIToolChoiceString::None => {
                        AzureToolChoice::String(AzureToolChoiceString::None)
                    }
                    OpenAIToolChoiceString::Auto => {
                        AzureToolChoice::String(AzureToolChoiceString::Auto)
                    }
                    OpenAIToolChoiceString::Required => {
                        AzureToolChoice::String(AzureToolChoiceString::Auto)
                    } // Azure doesn't support required
                }
            }
            OpenAIToolChoice::Specific(tool_choice) => AzureToolChoice::Specific(tool_choice),
        }
    }
}

/// This struct defines the supported parameters for the Azure OpenAI inference API
/// See the [API documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, seed, service_tier, stop, user,
/// or context_length_exceeded_behavior
#[derive(Debug, Serialize)]
struct AzureRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    stream: bool,
    response_format: AzureResponseFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AzureToolChoice<'a>>,
}

impl<'a> AzureRequest<'a> {
    pub fn new(request: &'a ModelInferenceRequest) -> AzureRequest<'a> {
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                AzureResponseFormat::JsonObject
            }
            ModelInferenceRequestJsonMode::Off => AzureResponseFormat::Text,
        };
        let messages = prepare_openai_messages(request);
        let (tools, tool_choice, _) = prepare_openai_tools(request);
        AzureRequest {
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream,
            response_format,
            seed: request.seed,
            tools,
            tool_choice: tool_choice.map(AzureToolChoice::from),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    use crate::inference::providers::common::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::providers::openai::{OpenAIToolType, SpecificToolFunction};
    use crate::inference::types::{
        FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };

    #[test]
    fn test_azure_request_new() {
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let azure_request = AzureRequest::new(&request_with_tools);

        assert_eq!(azure_request.messages.len(), 1);
        assert_eq!(azure_request.temperature, Some(0.5));
        assert_eq!(azure_request.max_tokens, Some(100));
        assert!(!azure_request.stream);
        assert_eq!(azure_request.seed, Some(69));
        assert_eq!(
            azure_request.response_format,
            AzureResponseFormat::JsonObject
        );
        assert!(azure_request.tools.is_some());
        let tools = azure_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            azure_request.tool_choice,
            Some(AzureToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );
    }

    #[test]
    fn test_azure_tool_choice_from() {
        // Required is converted to Auto
        let tool_choice = OpenAIToolChoice::String(OpenAIToolChoiceString::Required);
        let azure_tool_choice = AzureToolChoice::from(tool_choice);
        assert_eq!(
            azure_tool_choice,
            AzureToolChoice::String(AzureToolChoiceString::Auto)
        );

        // Specific tool choice is converted to Specific
        let specific_tool_choice = OpenAIToolChoice::Specific(SpecificToolChoice {
            r#type: OpenAIToolType::Function,
            function: SpecificToolFunction {
                name: "test_function",
            },
        });
        let azure_specific_tool_choice = AzureToolChoice::from(specific_tool_choice);
        assert_eq!(
            azure_specific_tool_choice,
            AzureToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: "test_function",
                }
            })
        );

        // None is converted to None
        let none_tool_choice = OpenAIToolChoice::String(OpenAIToolChoiceString::None);
        let azure_none_tool_choice = AzureToolChoice::from(none_tool_choice);
        assert_eq!(
            azure_none_tool_choice,
            AzureToolChoice::String(AzureToolChoiceString::None)
        );

        // Auto is converted to Auto
        let auto_tool_choice = OpenAIToolChoice::String(OpenAIToolChoiceString::Auto);
        let azure_auto_tool_choice = AzureToolChoice::from(auto_tool_choice);
        assert_eq!(
            azure_auto_tool_choice,
            AzureToolChoice::String(AzureToolChoiceString::Auto)
        );
    }
}
