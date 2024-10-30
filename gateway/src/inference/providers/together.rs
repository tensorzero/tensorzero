use std::borrow::Cow;

use futures::{StreamExt, TryStreamExt};
use lazy_static::lazy_static;
use reqwest::StatusCode;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::Error,
    inference::types::{
        ContentBlock, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
        ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
    },
    model::ProviderCredentials,
};

use super::{
    openai::{
        get_chat_url, handle_openai_error, prepare_openai_tools, stream_openai,
        tensorzero_to_openai_messages, OpenAIRequestMessage, OpenAIResponse,
        OpenAISystemRequestMessage, OpenAITool, OpenAIToolChoice,
    },
    provider_trait::{HasCredentials, InferenceProvider},
};

lazy_static! {
    static ref TOGETHER_API_BASE: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.together.xyz/v1").expect("Failed to parse TOGETHER_API_BASE")
    };
}

#[derive(Debug)]
pub struct TogetherProvider {
    pub model_name: String,
    pub api_key: Option<SecretString>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TogetherCredentials<'a> {
    pub api_key: Cow<'a, SecretString>,
}

// TODO (#80): Add support for Llama 3.1 function calling as discussed [here](https://docs.together.ai/docs/llama-3-function-calling)

impl InferenceProvider for TogetherProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        api_key: ProviderCredentials<'a>,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = TogetherRequest::new(&self.model_name, request);
        let request_url = get_chat_url(Some(&TOGETHER_API_BASE))?;
        let api_key = match &api_key {
            ProviderCredentials::Together(credentials) => &credentials.api_key,
            _ => {
                return Err(Error::BadCredentialsPreInference {
                    provider_name: "Together".to_string(),
                })
            }
        };
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::TogetherClient {
                message: format!("{e}"),
                status_code: e.status().unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            })?;
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| Error::TogetherServer {
                message: format!("Error parsing text response: {e}"),
            })?;

            let response = serde_json::from_str(&response).map_err(|e| Error::TogetherServer {
                message: format!("Error parsing JSON response: {e}: {response}"),
            })?;

            Ok(TogetherResponseWithMetadata {
                response,
                latency: Latency::NonStreaming {
                    response_time: start_time.elapsed(),
                },
                request: request_body,
                generic_request: request,
            }
            .try_into()
            .map_err(map_openai_to_together_error)?)
        } else {
            Err(map_openai_to_together_error(handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::TogetherServer {
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
        let request_body = TogetherRequest::new(&self.model_name, request);
        let raw_request =
            serde_json::to_string(&request_body).map_err(|e| Error::TogetherServer {
                message: format!("Error serializing request: {e}"),
            })?;
        let api_key = match &api_key {
            ProviderCredentials::Together(credentials) => &credentials.api_key,
            _ => {
                return Err(Error::BadCredentialsPreInference {
                    provider_name: "Together".to_string(),
                })
            }
        };
        let request_url = get_chat_url(Some(&TOGETHER_API_BASE))?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to Together: {e}"),
            })?;
        let mut stream =
            Box::pin(stream_openai(event_source, start_time).map_err(map_openai_to_together_error));
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

impl HasCredentials for TogetherProvider {
    fn has_credentials(&self) -> bool {
        self.api_key.is_some()
    }

    fn get_credentials<'a>(
        &'a self,
        credentials: &'a InferenceCredentials,
    ) -> Result<ProviderCredentials<'a>, Error> {
        if let Some(api_key) = &self.api_key {
            if credentials.together.is_some() {
                return Err(Error::UnexpectedDynamicCredentials {
                    provider_name: "Together".to_string(),
                });
            }
            return Ok(ProviderCredentials::Together(Cow::Owned(
                TogetherCredentials {
                    api_key: Cow::Borrowed(api_key),
                },
            )));
        } else {
            match &credentials.together {
                Some(credentials) => Ok(ProviderCredentials::Together(Cow::Borrowed(credentials))),
                None => Err(Error::ApiKeyMissing {
                    provider_name: "Together".to_string(),
                }),
            }
        }
    }
}

fn map_openai_to_together_error(e: Error) -> Error {
    match e {
        Error::OpenAIServer { message } => Error::TogetherServer { message },
        Error::OpenAIClient {
            message,
            status_code,
        } => Error::TogetherClient {
            message,
            status_code,
        },
        _ => e,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum TogetherResponseFormat<'a> {
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<&'a Value>, // the desired JSON schema
    },
}

/// This struct defines the supported parameters for the Together inference API
/// See the [Together API documentation](https://docs.together.ai/docs/chat-overview)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, seed, service_tier, stop, user,
/// or context_length_exceeded_behavior
#[derive(Debug, Serialize)]
struct TogetherRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<TogetherResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> TogetherRequest<'a> {
    pub fn new(model: &'a str, request: &'a ModelInferenceRequest) -> TogetherRequest<'a> {
        let response_format = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                Some(TogetherResponseFormat::JsonObject {
                    schema: request.output_schema,
                })
            }
            ModelInferenceRequestJsonMode::Off => None,
        };
        let messages = prepare_together_messages(request);
        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(request);
        TogetherRequest {
            messages,
            model,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            seed: request.seed,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
        }
    }
}

pub(super) fn prepare_together_messages<'a>(
    request: &'a ModelInferenceRequest,
) -> Vec<OpenAIRequestMessage<'a>> {
    let mut messages: Vec<OpenAIRequestMessage> = request
        .messages
        .iter()
        .flat_map(tensorzero_to_openai_messages)
        .collect();
    if let Some(system_msg) = tensorzero_to_together_system_message(request.system.as_deref()) {
        messages.insert(0, system_msg);
    }
    messages
}

fn tensorzero_to_together_system_message(system: Option<&str>) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

struct TogetherResponseWithMetadata<'a> {
    response: OpenAIResponse,
    latency: Latency,
    request: TogetherRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<TogetherResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: TogetherResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let TogetherResponseWithMetadata {
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
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or(Error::OpenAIServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
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
        let raw_request =
            serde_json::to_string(&request_body).map_err(|e| Error::FireworksServer {
                message: format!("Error serializing request body as JSON: {e}"),
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
    use crate::inference::types::{FunctionType, RequestMessage, Role};

    #[test]
    fn test_together_request_new() {
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            max_tokens: Some(100),
            seed: Some(69),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let together_request =
            TogetherRequest::new("togethercomputer/llama-v3-8b", &request_with_tools);

        assert_eq!(together_request.model, "togethercomputer/llama-v3-8b");
        assert_eq!(together_request.messages.len(), 1);
        assert_eq!(together_request.temperature, Some(0.5));
        assert_eq!(together_request.top_p, Some(0.9));
        assert_eq!(together_request.presence_penalty, Some(0.1));
        assert_eq!(together_request.frequency_penalty, Some(0.1));
        assert_eq!(together_request.max_tokens, Some(100));
        assert_eq!(together_request.seed, Some(69));
        assert!(!together_request.stream);
        let tools = together_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            together_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );
        assert_eq!(together_request.parallel_tool_calls, Some(false));
    }

    #[test]
    fn test_together_api_base() {
        assert_eq!(TOGETHER_API_BASE.as_str(), "https://api.together.xyz/v1");
    }

    #[test]
    fn test_get_credentials() {
        let provider_no_credentials = TogetherProvider {
            api_key: None,
            model_name: "togethercomputer/llama-v3-8b".to_string(),
        };
        let credentials = InferenceCredentials::default();
        let result = provider_no_credentials
            .get_credentials(&credentials)
            .unwrap_err();
        assert_eq!(
            result,
            Error::ApiKeyMissing {
                provider_name: "Together".to_string(),
            }
        );
        let credentials = InferenceCredentials {
            together: Some(TogetherCredentials {
                api_key: Cow::Owned(SecretString::from("test_api_key".to_string())),
            }),
            ..Default::default()
        };
        let result = provider_no_credentials
            .get_credentials(&credentials)
            .unwrap();
        match result {
            ProviderCredentials::Together(creds) => {
                assert_eq!(creds.api_key.expose_secret(), "test_api_key".to_string());
            }
            _ => panic!("Expected Together credentials"),
        }

        let provider_with_credentials = TogetherProvider {
            api_key: Some(SecretString::from("test_api_key".to_string())),
            model_name: "togethercomputer/llama-v3-8b".to_string(),
        };
        let result = provider_with_credentials
            .get_credentials(&credentials)
            .unwrap_err();
        assert_eq!(
            result,
            Error::UnexpectedDynamicCredentials {
                provider_name: "Together".to_string(),
            }
        );
        let credentials = InferenceCredentials::default();
        let result = provider_with_credentials
            .get_credentials(&credentials)
            .unwrap();
        match result {
            ProviderCredentials::Together(creds) => {
                assert_eq!(creds.api_key.expose_secret(), "test_api_key".to_string());
            }
            _ => panic!("Expected Together credentials"),
        }
    }
}
