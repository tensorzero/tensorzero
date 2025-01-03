use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use tokio::time::Instant;
use url::Url;

use super::openai::{
    get_chat_url, handle_openai_error, prepare_openai_messages, prepare_openai_tools,
    stream_openai, OpenAIRequestMessage, OpenAIResponse, OpenAITool, OpenAIToolChoice,
    OpenAIToolType, StreamOptions,
};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    batch::BatchProviderInferenceResponse, ContentBlock, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, ProviderInferenceResponse, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStream,
};
use crate::model::CredentialLocation;

lazy_static! {
    static ref TGI_DEFAULT_BASE_URL: Url = {
        #[allow(clippy::expect_used)]
        Url::parse("https://api.openai.com/v1").expect("Failed to parse TGI_DEFAULT_BASE_URL")
    };
    static ref TGI: String = "TGI".to_string();
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("TGI_API_KEY".to_string())
}

#[derive(Debug)]
pub struct TGIProvider {
    pub api_base: Url,
    pub credentials: TGICredentials,
}

impl TGIProvider {
    pub fn new(api_base: Url, api_key_location: Option<CredentialLocation>) -> Result<Self, Error> {
        let api_key_location = api_key_location.unwrap_or(default_api_key_location());
        let credentials = match api_key_location {
            CredentialLocation::Env(key_name) => {
                let api_key = env::var(key_name)
                    .map_err(|_| {
                        Error::new(ErrorDetails::ApiKeyMissing {
                            provider_name: TGI.clone(),
                        })
                    })?
                    .into();
                TGICredentials::Static(api_key)
            }
            CredentialLocation::Dynamic(key_name) => TGICredentials::Dynamic(key_name),
            CredentialLocation::None => TGICredentials::None,
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for TGI provider".to_string(),
            }))?,
        };
        Ok(TGIProvider {
            api_base,
            credentials,
        })
    }
}

#[derive(Debug)]
pub enum TGICredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TGICredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            TGICredentials::Static(api_key) => Ok(Some(api_key)),
            TGICredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: TGI.clone(),
                    }
                    .into()
                }))
                .transpose()
            }
            TGICredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for TGIProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let model_name = TGI.to_lowercase().clone();
        let request_body = TGIRequest::new(&model_name, request)?;
        let request_url = get_chat_url(Some(&TGI_DEFAULT_BASE_URL))?;
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
                    message: format!("Error sending request to TGI: {e}"),
                    status_code: e.status(),
                    provider_type: TGI.clone(),
                })
            })?;

        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: TGI.clone(),
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: TGI.clone(),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(TGIResponseWithMetadata {
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
                        provider_type: TGI.clone(),
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
        let model_name = TGI.to_lowercase().clone();
        let request_body = TGIRequest::new(&model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request: {e}"),
                provider_type: TGI.clone(),
            })
        })?;
        let request_url = get_chat_url(Some(&TGI_DEFAULT_BASE_URL))?;
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
                    message: format!("Error sending request to OpenAI: {e}"),
                    status_code: None,
                    provider_type: TGI.clone(),
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
                    provider_type: TGI.clone(),
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
            provider_type: TGI.clone(),
        }
        .into())
    }
}

/// This struct defines the supported parameters for the TGI API
/// See the [TGI documentation](https://huggingface.co/docs/text-generation-inference/en/reference/api_reference#openai-messages-api)
/// Since TGI is fully compatible with Open AI, you can also
/// See the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create)
/// for more details.
/// We are not handling logprobs, top_logprobs, n,
/// presence_penalty, seed, service_tier, stop, user,
/// or the deprecated function_call and functions arguments.
#[derive(Debug, Serialize)]
struct TGIRequest<'a> {
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
    response_format: Option<TGIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

impl<'a> TGIRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<TGIRequest<'a>, Error> {
        let response_format = Some(TGIResponseFormat::new(
            &request.json_mode,
            request.output_schema,
            model,
        ));

        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };

        let messages = prepare_openai_messages(request);

        let (tools, tool_choice, parallel_tool_calls) = prepare_openai_tools(request);

        Ok(TGIRequest {
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

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum TGIResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: Value,
    },
}

impl TGIResponseFormat {
    pub fn new(
        json_mode: &ModelInferenceRequestJsonMode,
        output_schema: Option<&Value>,
        model: &str,
    ) -> Self {
        if model.contains("3.5") && *json_mode == ModelInferenceRequestJsonMode::Strict {
            return TGIResponseFormat::JsonObject;
        }

        match json_mode {
            ModelInferenceRequestJsonMode::On => TGIResponseFormat::JsonObject,
            ModelInferenceRequestJsonMode::Off => TGIResponseFormat::Text,
            ModelInferenceRequestJsonMode::Strict => match output_schema {
                Some(schema) => {
                    let json_schema = json!({"name": "response", "strict": true, "schema": schema});
                    TGIResponseFormat::JsonSchema { json_schema }
                }
                None => TGIResponseFormat::JsonObject,
            },
        }
    }
}
#[derive(Serialize, Debug, Clone, PartialEq, Deserialize)]
struct OpenAIRequestToolCall<'a> {
    id: &'a str,
    r#type: OpenAIToolType,
    function: OpenAIRequestFunctionCall<'a>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct OpenAIRequestFunctionCall<'a> {
    name: &'a str,
    arguments: &'a str,
}

struct TGIResponseWithMetadata<'a> {
    response: OpenAIResponse,
    latency: Latency,
    request: TGIRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<TGIResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: TGIResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let TGIResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
        } = value;
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing response: {e}"),
                provider_type: TGI.clone(),
            })
        })?;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: TGI.clone(),
            }
            .into());
        }
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                provider_type: TGI.clone(),
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
                provider_type: TGI.clone(),
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

    use serde_json::json;

    use crate::inference::{
        providers::{
            common::{WEATHER_TOOL, WEATHER_TOOL_CONFIG},
            openai::{OpenAIToolType, SpecificToolChoice, SpecificToolFunction},
        },
        types::{FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role},
    };

    use super::*;

    #[test]
    fn test_tgi_request_new() {
        let model_name = TGI.to_lowercase().clone();
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
        let tgi_request = TGIRequest::new(&model_name, &basic_request).unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 2);
        assert_eq!(tgi_request.temperature, Some(0.7));
        assert_eq!(tgi_request.max_tokens, Some(100));
        assert_eq!(tgi_request.seed, Some(69));
        assert_eq!(tgi_request.top_p, Some(0.9));
        assert_eq!(tgi_request.presence_penalty, Some(0.1));
        assert_eq!(tgi_request.frequency_penalty, Some(0.2));
        assert!(tgi_request.stream);
        assert_eq!(tgi_request.response_format, Some(TGIResponseFormat::Text));
        assert!(tgi_request.tools.is_none());
        assert_eq!(tgi_request.tool_choice, None);
        assert!(tgi_request.parallel_tool_calls.is_none());

        // Test request with tools and JSON mode
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

        let tgi_request = TGIRequest::new(&model_name, &request_with_tools).unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 2);
        assert_eq!(tgi_request.temperature, None);
        assert_eq!(tgi_request.max_tokens, None);
        assert_eq!(tgi_request.seed, None);
        assert_eq!(tgi_request.top_p, None);
        assert_eq!(tgi_request.presence_penalty, None);
        assert_eq!(tgi_request.frequency_penalty, None);
        assert!(!tgi_request.stream);
        assert_eq!(
            tgi_request.response_format,
            Some(TGIResponseFormat::JsonObject)
        );
        assert!(tgi_request.tools.is_some());
        let tools = tgi_request.tools.as_ref().unwrap();
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            tgi_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );

        // Test request with strict JSON mode with no output schema
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

        let tgi_request = TGIRequest::new(&model_name, &request_with_tools).unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 1);
        assert_eq!(tgi_request.temperature, None);
        assert_eq!(tgi_request.max_tokens, None);
        assert_eq!(tgi_request.seed, None);
        assert!(!tgi_request.stream);
        assert_eq!(tgi_request.top_p, None);
        assert_eq!(tgi_request.presence_penalty, None);
        assert_eq!(tgi_request.frequency_penalty, None);
        // Resolves to normal JSON mode since no schema is provided (this shouldn't really happen in practice)
        assert_eq!(
            tgi_request.response_format,
            Some(TGIResponseFormat::JsonObject)
        );

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

        let tgi_request = TGIRequest::new(&model_name, &request_with_tools).unwrap();

        assert_eq!(tgi_request.model, &model_name);
        assert_eq!(tgi_request.messages.len(), 1);
        assert_eq!(tgi_request.temperature, None);
        assert_eq!(tgi_request.max_tokens, None);
        assert_eq!(tgi_request.seed, None);
        assert!(!tgi_request.stream);
        assert_eq!(tgi_request.top_p, None);
        assert_eq!(tgi_request.presence_penalty, None);
        assert_eq!(tgi_request.frequency_penalty, None);
        let expected_schema = serde_json::json!({"name": "response", "strict": true, "schema": {}});
        assert_eq!(
            tgi_request.response_format,
            Some(TGIResponseFormat::JsonSchema {
                json_schema: expected_schema,
            })
        );
    }
}
