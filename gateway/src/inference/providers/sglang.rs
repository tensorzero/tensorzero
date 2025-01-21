use futures::StreamExt;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use serde_json::{json, Value};
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

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("SGLANG_API_KEY".to_string())
}

const PROVIDER_NAME: &str = "SGLang";
const PROVIDER_TYPE: &str = "sglang";

#[derive(Debug)]
pub struct SGLangProvider {
    model_name: String,
    api_base: Url,
    credentials: SGLangCredentials,
}

impl SGLangProvider {
    pub fn new(
        model_name: String,
        api_base: Url,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, PROVIDER_TYPE))?;
        let provider_credentials = SGLangCredentials::try_from(generic_credentials)?;
        Ok(SGLangProvider {
            model_name,
            api_base,
            credentials: provider_credentials,
        })
    }
}

#[derive(Debug)]
pub enum SGLangCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for SGLangCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(SGLangCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(SGLangCredentials::Dynamic(key_name)),
            Credential::None => Ok(SGLangCredentials::None),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(SGLangCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for SGLang provider".to_string(),
            })),
        }
    }
}

impl SGLangCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            SGLangCredentials::Static(api_key) => Ok(Some(api_key)),
            SGLangCredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                }))
                .transpose()
            }
            SGLangCredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for SGLangProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = SGLangRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(&self.api_base)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json");
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let res = request_builder
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to SGLang: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
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
            Ok(SGLangResponseWithMetadata {
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
        let request_body = SGLangRequest::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let request_url = get_chat_url(&self.api_base)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json");
        if let Some(api_key) = api_key {
            request_builder = request_builder.bearer_auth(api_key.expose_secret());
        }
        let event_source = request_builder
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to SGLang: {e}"),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
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

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum SGLangResponseFormat {
    #[default]
    Text,
    JsonSchema {
        json_schema: Value,
    },
}

impl SGLangResponseFormat {
    fn new(
        json_mode: &ModelInferenceRequestJsonMode,
        output_schema: Option<&Value>,
    ) -> Result<Self, Error> {
        match json_mode {
            ModelInferenceRequestJsonMode::Off => Ok(SGLangResponseFormat::Text),
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                if let Some(schema) = output_schema {
                    let json_schema = json!({"name": "response", "strict": true, "schema": schema});
                    return Ok(SGLangResponseFormat::JsonSchema { json_schema });
                }
                Err(ErrorDetails::InvalidRequest {
                    message: "The SGL models requires a schema to be provided in json mode"
                        .to_string(),
                }
                .into())
            }
        }
    }
}

/// This struct defines the supported parameters for the OpenAI API
/// See the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Serialize)]
struct SGLangRequest<'a> {
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
    response_format: Option<SGLangResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> SGLangRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<SGLangRequest<'a>, Error> {
        let response_format = Some(SGLangResponseFormat::new(
            &request.json_mode,
            request.output_schema,
        )?);
        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };
        let messages = prepare_openai_messages(request);

        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(request);
        Ok(SGLangRequest {
            messages,
            model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            seed: request.seed,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            stream: request.stream,
            stream_options,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
        })
    }
}

struct SGLangResponseWithMetadata<'a> {
    response: OpenAIResponse,
    latency: Latency,
    request: SGLangRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<SGLangResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: SGLangResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let SGLangResponseWithMetadata {
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
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(serde_json::to_string(&response).unwrap_or_default()),
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(serde_json::to_string(&response).unwrap_or_default()),
                provider_type: PROVIDER_TYPE.to_string(),
            }))?
            .message;
        let mut content: Vec<ContentBlock> = Vec::new();
        if let Some(text) = message.content {
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            // See https://github.com/tensorzero/tensorzero/discussions/776
            tracing::warn!("Tool calls are not yet supported for SGLang");
            for tool_call in tool_calls {
                content.push(ContentBlock::ToolCall(tool_call.into()));
            }
        }
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request body as JSON: {e}"),
            })
        })?;
        let usage = response.usage.into();
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

    use serde_json::json;

    use crate::inference::{
        providers::common::WEATHER_TOOL_CONFIG,
        types::{FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role},
    };

    use super::*;

    #[test]
    fn test_sglang_request_new() {
        let model_name = PROVIDER_TYPE.to_string();
        let basic_request = ModelInferenceRequest {
            messages: vec![
                RequestMessage {
                    role: Role::User,
                    content: vec!["Hello".to_string().into()],
                },
                RequestMessage {
                    role: Role::Assistant,
                    content: vec!["Hi there!".to_string().into()],
                },
            ],
            system: None,
            tool_config: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let sglang_request = SGLangRequest::new(&model_name, &basic_request).unwrap();

        assert_eq!(sglang_request.model, &model_name);
        assert_eq!(sglang_request.messages.len(), 2);
        assert_eq!(sglang_request.temperature, Some(0.7));
        assert_eq!(sglang_request.max_tokens, Some(100));
        assert_eq!(sglang_request.seed, Some(69));
        assert_eq!(sglang_request.top_p, Some(0.9));
        assert_eq!(sglang_request.presence_penalty, Some(0.1));
        assert_eq!(sglang_request.frequency_penalty, Some(0.2));
        assert!(sglang_request.stream);
        assert!(sglang_request.tools.is_none());
        assert_eq!(sglang_request.tool_choice, None);
        assert!(sglang_request.parallel_tool_calls.is_none());

        // Test that non-strict JSON mode requires an output schema
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        SGLangRequest::new(&model_name, &request_with_tools).expect_err("requires a schema");

        // Test request with in strict JSON mode requires an no output schema
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Strict,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        SGLangRequest::new(&model_name, &request_with_tools).expect_err("requires a schema");

        // Test request with strict JSON mode with an output schema
        let output_schema = json!({});
        let request_with_tools = ModelInferenceRequest {
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            seed: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Strict,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
        };

        let sglang_request = SGLangRequest::new(&model_name, &request_with_tools).unwrap();

        assert_eq!(sglang_request.model, &model_name);
        assert_eq!(sglang_request.messages.len(), 1);
        assert_eq!(sglang_request.temperature, None);
        assert_eq!(sglang_request.max_tokens, None);
        assert_eq!(sglang_request.seed, None);
        assert!(!sglang_request.stream);
        assert_eq!(sglang_request.top_p, None);
        assert_eq!(sglang_request.presence_penalty, None);
        assert_eq!(sglang_request.frequency_penalty, None);
    }
}
