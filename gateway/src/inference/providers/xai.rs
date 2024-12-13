use futures::stream::TryStreamExt;
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
use crate::inference::types::{
    ContentBlock, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
};
use crate::model::CredentialLocation;

use super::openai::{
    get_chat_url, handle_openai_error, prepare_openai_messages, prepare_openai_tools,
    stream_openai, OpenAIRequestMessage, OpenAIResponse, OpenAITool, OpenAIToolChoice,
    StreamOptions,
};

lazy_static! {
    static ref X_AI_DEFAULT_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.x.ai/v1").expect("Failed to parse X_AI_DEFAULT_BASE_URL")
    };
}

pub fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("X_AI_API_KEY".to_string())
}

#[derive(Debug)]
pub struct XAIProvider {
    pub model_name: String,
    pub credentials: XAICredentials,
}

#[derive(Debug)]
pub enum XAICredentials {
    Static(SecretString),
    Dynamic(String),
    None,
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
                    provider_name: "X AI".to_string(),
                }
                .into()
            }),
            XAICredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: "X AI".to_string(),
            }
            .into()),
        }
    }
}

impl InferenceProvider for XAIProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = XAIRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(Some(&X_AI_DEFAULT_BASE_URL))?;
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
                    message: format!("Error sending request to X AI: {e}"),
                })
            })?;
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::XAIServer {
                    message: format!("Error parsing text response: {e}"),
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::XAIServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
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
            .try_into()
            .map_err(map_openai_to_x_ai_error)?)
        } else {
            Err(map_openai_to_x_ai_error(handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::XAIServer {
                        message: format!("Error parsing error response: {e}"),
                    })
                })?,
            )))
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
        let request_body = XAIRequest::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::XAIServer {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let request_url = get_chat_url(Some(&X_AI_DEFAULT_BASE_URL))?;
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
                    message: format!("Error sending request to X AI: {e}"),
                })
            })?;

        let mut stream =
            Box::pin(stream_openai(event_source, start_time).map_err(map_openai_to_x_ai_error));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(ErrorDetails::OpenAIServer {
                    message: "Stream ended before first chunk".to_string(),
                }
                .into())
            }
        };
        Ok((chunk, stream, raw_request))
    }
}

/// This struct defines the supported parameters for the X AI API
/// See the [X AI API documentation](https://docs.x.ai/api/endpoints#chat-completions)
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
                message: "The X AI Grok beta faily of models does not support strict JSON"
                    .to_string(),
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
            Error::new(ErrorDetails::XAIServer {
                message: format!("Error parsing response: {e}"),
            })
        })?;

        if response.choices.len() != 1 {
            return Err(ErrorDetails::XAIServer {
                message: format!(
                    "Response has invalid number of choices {}, Expected 1",
                    response.choices.len()
                ),
            }
            .into());
        }

        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::XAIServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
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
            Error::new(ErrorDetails::XAIServer {
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

fn map_openai_to_x_ai_error(e: Error) -> Error {
    let details = e.get_owned_details();
    match details {
        ErrorDetails::OpenAIServer { message } => ErrorDetails::XAIServer { message },
        ErrorDetails::OpenAIClient {
            message,
            status_code,
        } => ErrorDetails::XAIClient {
            message,
            status_code,
        },
        e => e,
    }
    .into()
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

        let x_ai_request = XAIRequest::new("grok-beta", &request_with_tools)
            .expect("failed to create X AI Request during test");

        assert_eq!(x_ai_request.messages.len(), 1);
        assert_eq!(x_ai_request.temperature, Some(0.5));
        assert_eq!(x_ai_request.max_tokens, Some(100));
        assert!(!x_ai_request.stream);
        assert_eq!(x_ai_request.seed, Some(69));
        assert!(x_ai_request.tools.is_some());
        let tools = x_ai_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            x_ai_request.tool_choice,
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

        let x_ai_request = XAIRequest::new("grok-beta", &request_with_tools)
            .expect("failed to create X AI Request");

        assert_eq!(x_ai_request.messages.len(), 2);
        assert_eq!(x_ai_request.temperature, Some(0.5));
        assert_eq!(x_ai_request.max_tokens, Some(100));
        assert_eq!(x_ai_request.top_p, Some(0.9));
        assert_eq!(x_ai_request.presence_penalty, Some(0.1));
        assert_eq!(x_ai_request.frequency_penalty, Some(0.2));
        assert!(!x_ai_request.stream);
        assert_eq!(x_ai_request.seed, Some(69));

        assert!(x_ai_request.tools.is_some());
        let tools = x_ai_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            x_ai_request.tool_choice,
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

        let x_ai_request = XAIRequest::new("grok-beta", &request_with_tools);
        assert!(x_ai_request.is_err());
    }
}
