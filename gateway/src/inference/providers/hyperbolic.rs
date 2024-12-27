
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{
    batch::BatchProviderInferenceResponse, ContentBlock, Latency, ModelInferenceRequest,
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
    static ref HYPERBOLIC_DEFAULT_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.hyperbolic.xyz/v1/")
            .expect("Failed to parse HYPERBOLIC_DEFAULT_BASE_URL")
    };
}

pub fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("HYPERBOLIC_API_KEY".to_string())
}

#[derive(Debug)]
pub struct HyperbolicProvider {
    pub model_name: String,
    pub credentials: HyperbolicCredentials,
}

impl HyperbolicProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, "Hyperbolic"))?;
        let provider_credentials = HyperbolicCredentials::try_from(generic_credentials)?; 
        Ok(HyperbolicProvider {
            model_name,
            credentials: provider_credentials,
        })
    }
}

#[derive(Debug)]
pub enum HyperbolicCredentials {
    Static(SecretString),
    Dynamic(String),
    #[cfg(any(test, feature = "e2e_tests"))]
    None
}

impl TryFrom<Credential> for HyperbolicCredentials {
    type Error = Error;
    
    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(HyperbolicCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(HyperbolicCredentials::Dynamic(key_name)),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(HyperbolicCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Hyperbolic provider".to_string(),
            }))
        }
    }
}


impl HyperbolicCredentials {
    fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            HyperbolicCredentials::Static(api_key) => Ok(api_key),
            HyperbolicCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "HyperbolicCredentials".to_string(),
                    }
                    .into()
                })
            },
            #[cfg(any(test, feature = "e2e_tests"))]
            HyperbolicCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: "Hyperbolic".to_string(),
            })?,
        }
    }
}

impl InferenceProvider for HyperbolicProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = HyperbolicRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(Some(&HYPERBOLIC_DEFAULT_BASE_URL))?;
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
                    message: format!("Error sending request to Hyperbolic: {e}"),
                    status_code: e.status(),
                    provider_type: "Hyperbolic".to_string(),
                })
            })?;

        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: "Hyperbolic".to_string(),
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: "Hyperbolic".to_string(),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(HyperbolicResponseWithMetadata {
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
                        provider_type: "Hyperbolic".to_string(),
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
        let request_body = HyperbolicRequest::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request: {e}"),
                provider_type: "Hyperbolic".to_string(),
            })
        })?;
        let request_url = get_chat_url(Some(&HYPERBOLIC_DEFAULT_BASE_URL))?;
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
                    message: format!("Error sending request to Hyperbolic: {e}"),
                    status_code: None,
                    provider_type: "Hyperbolic".to_string(),
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
                    provider_type: "Hyperbolic".to_string(),
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
            provider_type: "Hyperbolic".to_string(),
        }
        .into())
    }
}

/// This struct defines the supported parameters for the Hyperbolic text generation API
/// See the [API documentation](https://docs.hyperbolic.xyz/docs/rest-api)
/// for more details.
/// We are not handling logit_bias, logprobs, toplogprobs, n, stop, user, top_k, min_p, and repetition_penalty.
#[derive(Debug, Serialize)]
struct HyperbolicRequest<'a> {
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

impl<'a> HyperbolicRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<HyperbolicRequest<'a>, Error> {
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
        Ok(HyperbolicRequest {
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

struct HyperbolicResponseWithMetadata<'a> {
    response: OpenAIResponse,
    latency: Latency,
    request: HyperbolicRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<HyperbolicResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: HyperbolicResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let HyperbolicResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
        } = value;

        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing response: {e}"),
                provider_type: "Hyperbolic".to_string(),
            })
        })?;

        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices {}, Expected 1",
                    response.choices.len()
                ),
                provider_type: "Hyperbolic".to_string(),
            }
            .into());
        }

        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: "Hyperbolic".to_string(),
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
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request body as JSON: {e}"),
                provider_type: "Hyperbolic".to_string(),
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

    use crate::inference::providers::common::WEATHER_TOOL_CONFIG;
    use crate::inference::types::{
        FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };

    #[test]
    fn test_hyperbolic_request_new() {
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

        let hyperbolic_request =
            HyperbolicRequest::new("meta-llama/Meta-Llama-3-70B-Instruct", &request_with_tools)
                .expect("failed to create Hyperbolic Request during test");

        assert_eq!(hyperbolic_request.messages.len(), 1);
        assert_eq!(hyperbolic_request.temperature, Some(0.5));
        assert_eq!(hyperbolic_request.max_tokens, Some(100));
        assert!(!hyperbolic_request.stream);
        assert_eq!(hyperbolic_request.seed, Some(69));
    }

    #[test]
    fn test_hyperbolic_api_base() {
        assert_eq!(
            HYPERBOLIC_DEFAULT_BASE_URL.as_str(),
            "https://api.hyperbolic.xyz/v1/"
        );
    }
}
