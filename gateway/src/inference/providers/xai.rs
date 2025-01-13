use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use tokio::time::Instant;
use url::Url;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, ContentBlock, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, ProviderInferenceResponse, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStream,
};
use crate::model::{Credential, CredentialLocation};

use super::openai::{
    get_chat_url, handle_openai_error, prepare_openai_messages, prepare_openai_tools,
    stream_openai, OpenAIRequestMessage, OpenAIResponse, OpenAITool, OpenAIToolChoice,
    StreamOptions,
};

lazy_static! {
    static ref XAI_DEFAULT_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.x.ai/v1").expect("Failed to parse XAI_DEFAULT_BASE_URL")
    };
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("XAI_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "xAI";
const PROVIDER_TYPE: &str = "xai";

#[derive(Debug)]
pub struct XAIProvider {
    pub model_name: String,
    pub credentials: XAICredentials,
}

impl XAIProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, PROVIDER_TYPE))?;
        let provider_credentials = XAICredentials::try_from(generic_credentials)?;

        Ok(XAIProvider {
            model_name,
            credentials: provider_credentials,
        })
    }
}

#[derive(Debug)]
pub enum XAICredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for XAICredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(XAICredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(XAICredentials::Dynamic(key_name)),
            Credential::None => Ok(XAICredentials::None),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(XAICredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for xAI provider".to_string(),
            })),
        }
    }
}

impl XAICredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            XAICredentials::Static(api_key) => Ok(api_key),
            XAICredentials::Dynamic(key_name) => dynamic_api_keys.get(key_name).ok_or_else(|| {
                ErrorDetails::ApiKeyMissing {
                    provider_name: PROVIDER_NAME.to_string(),
                }
                .into()
            }),
            XAICredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            }
            .into()),
        }
    }
}

impl InferenceProvider for XAIProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = XAIRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(Some(&XAI_DEFAULT_BASE_URL))?;
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
                    message: format!("Error sending request to xAI: {e}"),
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
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(XAIResponseWithMetadata {
                response,
                latency,
                request: request_body,
                generic_request: request,
            }
            .try_into()?)
        } else {
            let status = res.status();

            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing error response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            Err(handle_openai_error(status, &response, PROVIDER_TYPE))
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
        let request_body = XAIRequest::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request: {e}"),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
        let request_url = get_chat_url(Some(&XAI_DEFAULT_BASE_URL))?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to xAI: {e}"),
                    status_code: None,
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
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
            provider_type: "xAI".to_string(),
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

/// This struct defines the supported parameters for the xAI API
/// See the [xAI API documentation](https://docs.x.ai/api/endpoints#chat-completions)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// logit_bias, seed, service_tier, stop, user or response_format.
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Serialize)]
struct XAIRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,

    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
}

impl<'a> XAIRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<XAIRequest<'a>, Error> {
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

        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };

        if request.json_mode == ModelInferenceRequestJsonMode::Strict {
            return Err(ErrorDetails::InvalidRequest {
                message: "The xAI models don't support strict JSON mode.".to_string(),
            }
            .into());
        }

        let messages = prepare_openai_messages(request);

        let (tools, tool_choice, _) = prepare_openai_tools(request);
        Ok(XAIRequest {
            messages,
            model,
            temperature,
            max_tokens,
            seed,
            top_p,
            presence_penalty,
            frequency_penalty,
            stream,
            stream_options,
            tools,
            tool_choice,
        })
    }
}

struct XAIResponseWithMetadata<'a> {
    response: OpenAIResponse,
    latency: Latency,
    request: XAIRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<XAIResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: XAIResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let XAIResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
        } = value;

        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error parsing response: {e}"),
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    use crate::inference::providers::common::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::providers::openai::{
        OpenAIToolType, SpecificToolChoice, SpecificToolFunction,
    };
    use crate::inference::types::{
        FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };

    #[test]
    fn test_xai_request_new() {
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

        let xai_request = XAIRequest::new("grok-beta", &request_with_tools)
            .expect("failed to create xAI Request during test");

        assert_eq!(xai_request.messages.len(), 1);
        assert_eq!(xai_request.temperature, Some(0.5));
        assert_eq!(xai_request.max_tokens, Some(100));
        assert!(!xai_request.stream);
        assert_eq!(xai_request.seed, Some(69));
        assert!(xai_request.tools.is_some());
        let tools = xai_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            xai_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
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

        let xai_request = XAIRequest::new("grok-beta", &request_with_tools)
            .expect("failed to create xAI Request");

        assert_eq!(xai_request.messages.len(), 2);
        assert_eq!(xai_request.temperature, Some(0.5));
        assert_eq!(xai_request.max_tokens, Some(100));
        assert_eq!(xai_request.top_p, Some(0.9));
        assert_eq!(xai_request.presence_penalty, Some(0.1));
        assert_eq!(xai_request.frequency_penalty, Some(0.2));
        assert!(!xai_request.stream);
        assert_eq!(xai_request.seed, Some(69));

        assert!(xai_request.tools.is_some());
        let tools = xai_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            xai_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );

        let request_with_tools = ModelInferenceRequest {
            json_mode: ModelInferenceRequestJsonMode::Strict,
            ..request_with_tools
        };

        let xai_request = XAIRequest::new("grok-beta", &request_with_tools);
        assert!(xai_request.is_err());
    }

    #[test]
    fn test_xai_api_base() {
        assert_eq!(XAI_DEFAULT_BASE_URL.as_str(), "https://api.x.ai/v1");
    }

    #[test]
    fn test_credential_to_xai_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: XAICredentials = XAICredentials::try_from(generic).unwrap();
        assert!(matches!(creds, XAICredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = XAICredentials::try_from(generic).unwrap();
        assert!(matches!(creds, XAICredentials::Dynamic(_)));

        // Test Missing credential (test mode)
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            let generic = Credential::Missing;
            let creds = XAICredentials::try_from(generic).unwrap();
            assert!(matches!(creds, XAICredentials::None));
        }

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = XAICredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
}
