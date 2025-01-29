use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, ContentBlock, Latency, ModelInferenceRequest,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
};
use crate::model::{Credential, CredentialLocation};
use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use tokio::time::Instant;
use url::Url;

use super::openai::{
    get_chat_url, handle_openai_error, prepare_openai_messages, stream_openai,
    OpenAIRequestMessage, OpenAIResponse,
};
use super::provider_trait::InferenceProvider;

lazy_static! {
    static ref OPENROUTER_DEFAULT_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://openrouter.ai/api/v1/")
            .expect("Failed to parse OPENROUTER_DEFAULT_BASE_URL")
    };
}

pub fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("OPENROUTER_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "OpenRouter";
const PROVIDER_TYPE: &str = "openrouter";

#[derive(Debug)]
pub struct OpenRouterProvider {
    model_name: String,
    credentials: OpenRouterCredentials,
}

impl OpenRouterProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, PROVIDER_TYPE))?;
        let provider_credentials = OpenRouterCredentials::try_from(generic_credentials)?;
        Ok(OpenRouterProvider {
            model_name,
            credentials: provider_credentials,
        })
    }
}

#[derive(Debug)]
pub enum OpenRouterCredentials {
    Static(SecretString),
    Dynamic(String),
    #[cfg(any(test, feature = "e2e_tests"))]
    None,
}

impl TryFrom<Credential> for OpenRouterCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(OpenRouterCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(OpenRouterCredentials::Dynamic(key_name)),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(OpenRouterCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for OpenRouter provider".to_string(),
            })),
        }
    }
}

impl OpenRouterCredentials {
    fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            OpenRouterCredentials::Static(api_key) => Ok(api_key),
            OpenRouterCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                })
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            OpenRouterCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            })?,
        }
    }
}

impl InferenceProvider for OpenRouterProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = OpenRouterRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(&OPENROUTER_DEFAULT_BASE_URL)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret());

        let res = request_builder
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to OpenRouter: {e}"),
                    status_code: e.status(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(response.clone()),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(OpenRouterResponseWithMetadata {
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
                        raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })
                })?,
                PROVIDER_TYPE,
            ))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
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
        let request_body = OpenRouterRequest::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let request_url = get_chat_url(&OPENROUTER_DEFAULT_BASE_URL)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();

        let request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret());

        let event_source = request_builder
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to OpenRouter: {e}"),
                    status_code: None,
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

        let mut stream = Box::pin(stream_openai(event_source, start_time));
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(ErrorDetails::InferenceServer {
                    message: "Stream ended before first chunk".to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                }
                .into())
            }
        };
        Ok((chunk, stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
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

#[derive(Debug, Serialize)]
struct OpenRouterRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

impl<'a> OpenRouterRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<OpenRouterRequest<'a>, Error> {
        let ModelInferenceRequest {
            temperature,
            max_tokens,
            seed,
            top_p,
            presence_penalty,
            frequency_penalty,
            stream,
            ..
        } = *request;

        let messages = prepare_openai_messages(request);
        Ok(OpenRouterRequest {
            messages,
            model,
            frequency_penalty,
            max_tokens,
            presence_penalty,
            seed,
            stream,
            temperature,
            top_p,
        })
    }
}

struct OpenRouterResponseWithMetadata<'a> {
    response: OpenAIResponse,
    latency: Latency,
    request: OpenRouterRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<OpenRouterResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: OpenRouterResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let OpenRouterResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
        } = value;

        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing response: {e}"),
            })
        })?;

        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices {}, Expected 1",
                    response.choices.len()
                ),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }

        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }))?
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
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request body as JSON: {e}"),
            })
        })?;
        let system = generic_request.system.clone();
        let messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            content,
            system,
            messages,
            raw_request,
            raw_response,
            usage,
            latency,
        ))
    }
}

// Add tests module
#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    use crate::inference::providers::common::WEATHER_TOOL_CONFIG;
    use crate::inference::types::{
        FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };

    #[test]
    fn test_openrouter_request_new() {
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

        let openrouter_request =
            OpenRouterRequest::new("anthropic/claude-3-opus-20240229", &request_with_tools)
                .expect("failed to create OpenRouter Request during test");

        assert_eq!(openrouter_request.messages.len(), 1);
        assert_eq!(openrouter_request.temperature, Some(0.5));
        assert_eq!(openrouter_request.max_tokens, Some(100));
        assert!(!openrouter_request.stream);
        assert_eq!(openrouter_request.seed, Some(69));
    }

    #[test]
    fn test_openrouter_api_base() {
        assert_eq!(
            OPENROUTER_DEFAULT_BASE_URL.as_str(),
            "https://openrouter.ai/api/v1/"
        );
    }

    #[test]
    fn test_credential_to_openrouter_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = OpenRouterCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenRouterCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = OpenRouterCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OpenRouterCredentials::Dynamic(_)));

        // Test Missing credential (test mode)
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            let generic = Credential::Missing;
            let creds = OpenRouterCredentials::try_from(generic).unwrap();
            assert!(matches!(creds, OpenRouterCredentials::None));
        }

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = OpenRouterCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
}
