use std::borrow::Cow;

use futures::{StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use url::Url;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::inference::types::{
    ContentBlock, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
};
use crate::model::ProviderCredentials;

use super::openai::{
    handle_openai_error, prepare_openai_messages, prepare_openai_tools, stream_openai,
    OpenAIRequestMessage, OpenAIResponse, OpenAITool, OpenAIToolChoice, OpenAIToolChoiceString,
    SpecificToolChoice,
};
use super::provider_trait::{HasCredentials, InferenceProvider};

#[derive(Debug)]
pub struct AzureProvider {
    pub deployment_id: String,
    pub endpoint: Url,
    pub api_key: Option<SecretString>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AzureCredentials<'a> {
    pub api_key: Cow<'a, SecretString>,
}

impl InferenceProvider for AzureProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        api_key: ProviderCredentials<'a>,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = AzureRequest::new(request);
        let request_url = get_azure_chat_url(&self.endpoint, &self.deployment_id)?;
        let start_time = Instant::now();
        let api_key = match &api_key {
            ProviderCredentials::Azure(credentials) => &credentials.api_key,
            _ => {
                return Err(Error::BadCredentialsPreInference {
                    provider_name: "Azure".to_string(),
                })
            }
        };
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

            Ok(AzureResponseWithMetadata {
                response,
                latency,
                request: request_body,
                generic_request: request,
            }
            .try_into()
            .map_err(map_openai_to_azure_error)?)
        } else {
            Err(map_openai_to_azure_error(handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::AzureServer {
                    message: format!("Error parsing error response: {e}"),
                })?,
            )))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        api_key: ProviderCredentials<'a>,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
        let request_body = AzureRequest::new(request);
        let raw_request = serde_json::to_string(&request_body).map_err(|e| Error::AzureServer {
            message: format!("Error serializing request body as JSON: {e}"),
        })?;
        let request_url = get_azure_chat_url(&self.endpoint, &self.deployment_id)?;
        let api_key = match &api_key {
            ProviderCredentials::Azure(credentials) => &credentials.api_key,
            _ => {
                return Err(Error::BadCredentialsPreInference {
                    provider_name: "Azure".to_string(),
                })
            }
        };
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

impl HasCredentials for AzureProvider {
    fn has_credentials(&self) -> bool {
        self.api_key.is_some()
    }

    fn get_credentials<'a>(
        &'a self,
        api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderCredentials<'a>, Error> {
        if let Some(api_key) = &self.api_key {
            if api_keys.azure.is_some() {
                return Err(Error::UnexpectedDynamicCredentials {
                    provider_name: "Azure".to_string(),
                });
            }
            return Ok(ProviderCredentials::Azure(Cow::Owned(AzureCredentials {
                api_key: Cow::Borrowed(api_key),
            })));
        } else {
            match &api_keys.azure {
                Some(credentials) => Ok(ProviderCredentials::Azure(Cow::Borrowed(credentials))),
                None => Err(Error::ApiKeyMissing {
                    provider_name: "Azure".to_string(),
                }),
            }
        }
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
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
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
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            stream: request.stream,
            response_format,
            seed: request.seed,
            tools,
            tool_choice: tool_choice.map(AzureToolChoice::from),
        }
    }
}

struct AzureResponseWithMetadata<'a> {
    response: OpenAIResponse,
    latency: Latency,
    request: AzureRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<AzureResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: AzureResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let AzureResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
        } = value;
        let raw_response = serde_json::to_string(&response).map_err(|e| Error::OpenAIServer {
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
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or(Error::OpenAIServer {
                message: "Response has no choices (this should never happen)".to_string(),
            })?
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
        let raw_request = serde_json::to_string(&request_body).map_err(|e| Error::AzureServer {
            message: format!("Error serializing request body as JSON: {e}"),
        })?;

        Ok(ProviderInferenceResponse::new(
            content,
            system,
            input_messages,
            raw_request,
            raw_response,
            usage,
            latency,
        ))
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
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::Off,
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
        assert_eq!(azure_request.response_format, AzureResponseFormat::Text);
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

        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Json,
            output_schema: None,
        };

        let azure_request = AzureRequest::new(&request_with_tools);

        assert_eq!(azure_request.messages.len(), 2);
        assert_eq!(azure_request.temperature, Some(0.5));
        assert_eq!(azure_request.max_tokens, Some(100));
        assert_eq!(azure_request.top_p, Some(0.9));
        assert_eq!(azure_request.presence_penalty, Some(0.1));
        assert_eq!(azure_request.frequency_penalty, Some(0.2));
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

    #[test]
    fn test_get_credentials() {
        let provider_no_credentials = AzureProvider {
            api_key: None,
            endpoint: Url::parse("https://example.com").unwrap(),
            deployment_id: "deployment_id".to_string(),
        };
        let credentials = InferenceCredentials::default();
        let result = provider_no_credentials
            .get_credentials(&credentials)
            .unwrap_err();
        assert_eq!(
            result,
            Error::ApiKeyMissing {
                provider_name: "Azure".to_string(),
            }
        );
        let credentials = InferenceCredentials {
            azure: Some(AzureCredentials {
                api_key: Cow::Owned(SecretString::from("test_api_key".to_string())),
            }),
            ..Default::default()
        };
        let result = provider_no_credentials
            .get_credentials(&credentials)
            .unwrap();
        match result {
            ProviderCredentials::Azure(creds) => {
                assert_eq!(creds.api_key.expose_secret(), "test_api_key".to_string());
            }
            _ => panic!("Expected Azure credentials"),
        }

        let provider_with_credentials = AzureProvider {
            api_key: Some(SecretString::from("test_api_key".to_string())),
            endpoint: Url::parse("https://example.com").unwrap(),
            deployment_id: "deployment_id".to_string(),
        };
        let result = provider_with_credentials
            .get_credentials(&credentials)
            .unwrap_err();
        assert_eq!(
            result,
            Error::UnexpectedDynamicCredentials {
                provider_name: "Azure".to_string(),
            }
        );
        let credentials = InferenceCredentials::default();
        let result = provider_with_credentials
            .get_credentials(&credentials)
            .unwrap();
        match result {
            ProviderCredentials::Azure(creds) => {
                assert_eq!(creds.api_key.expose_secret(), "test_api_key".to_string());
            }
            _ => panic!("Expected Azure credentials"),
        }
    }
}
