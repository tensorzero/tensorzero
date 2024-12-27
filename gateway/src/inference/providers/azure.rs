
use futures::StreamExt;
use reqwest::StatusCode;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use url::Url;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{
    batch::BatchProviderInferenceResponse, ContentBlock, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, ProviderInferenceResponse, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStream,
};
use crate::model::{Credential, CredentialLocation};

use super::openai::{
    handle_openai_error, prepare_openai_messages, prepare_openai_tools, stream_openai,
    OpenAIRequestMessage, OpenAIResponse, OpenAITool, OpenAIToolChoice, OpenAIToolChoiceString,
    SpecificToolChoice,
};
use super::provider_trait::InferenceProvider;

#[derive(Debug)]
pub struct AzureProvider {
    pub deployment_id: String,
    pub endpoint: Url,
    pub credentials: AzureCredentials,
}

impl AzureProvider {
    pub fn new(
        deployment_id: String,
        endpoint: Url,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, "azure"))?;
        let provider_credentials = AzureCredentials::try_from(generic_credentials)?;
        Ok(AzureProvider {
            deployment_id,
            endpoint,
            credentials: provider_credentials,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum AzureCredentials {
    Static(SecretString),
    Dynamic(String),
    #[cfg(any(test, feature = "e2e_tests"))]
    None
}

impl TryFrom<Credential> for AzureCredentials {
    type Error = Error;
    
    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(AzureCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(AzureCredentials::Dynamic(key_name)),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => {
                Ok(AzureCredentials::None)
            },
            _ => {
                Err(Error::new(ErrorDetails::Config {
                    message: "Invalid api_key_location for Azure provider".to_string(),
                }))
            } 
        }
    }
}

impl AzureCredentials {
    fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            AzureCredentials::Static(api_key) => Ok(api_key),
            AzureCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Azure".to_string(),
                    }
                    .into()
                })
            },
            #[cfg(any(test, feature = "e2e_tests"))]
            AzureCredentials::None => {
                Err(ErrorDetails::ApiKeyMissing {
                    provider_name: "Azure".to_string(),
                }
                .into())
            }
        }
    }
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("AZURE_OPENAI_API_KEY".to_string())
}

impl InferenceProvider for AzureProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        api_key: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = AzureRequest::new(request);
        let request_url = get_azure_chat_url(&self.endpoint, &self.deployment_id)?;
        let start_time = Instant::now();
        let api_key = self.credentials.get_api_key(api_key)?;
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header("api-key", api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: e.to_string(),
                    status_code: Some(e.status().unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)),
                    provider_type: "Azure".to_string(),
                })
            })?;
        if res.status().is_success() {
            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: "Azure".to_string(),
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: "Azure".to_string(),
                })
            })?;

            Ok(AzureResponseWithMetadata {
                response,
                latency,
                request: request_body,
                generic_request: request,
            }
            .try_into()?)
        } else {
            Err(handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error parsing error response: {e}"),
                        provider_type: "Azure".to_string(),
                    })
                })?,
            ))
        }
    }

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
        let request_body = AzureRequest::new(request);
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request body as JSON: {e}"),
                provider_type: "Azure".to_string(),
            })
        })?;
        let request_url = get_azure_chat_url(&self.endpoint, &self.deployment_id)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header("api-key", api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to Azure: {e}"),
                    status_code: None,
                    provider_type: "Azure".to_string(),
                })
            })?;
        let mut stream = Box::pin(stream_openai(event_source, start_time));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(ErrorDetails::InferenceServer {
                    message: "Stream ended before first chunk".to_string(),
                    provider_type: "Azure".to_string(),
                }
                .into())
            }
        };
        Ok((chunk, stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'a>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<BatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Azure".to_string(),
        }
        .into())
    }
}

fn get_azure_chat_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing URL: {e:?}"),
                provider_type: "Azure".to_string(),
            })
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
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing response: {e}"),
                provider_type: "Azure".to_string(),
            })
        })?;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: "Azure".to_string(),
            }
            .into());
        }
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Response has no choices (this should never happen)".to_string(),
                    provider_type: "Azure".to_string(),
                })
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
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request body as JSON: {e}"),
                provider_type: "Azure".to_string(),
            })
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
}
