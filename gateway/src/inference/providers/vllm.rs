use std::borrow::Cow;

use futures::{StreamExt, TryStreamExt};
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use super::openai::{
    get_chat_url, handle_openai_error, stream_openai, tensorzero_to_openai_messages,
    OpenAIRequestMessage, OpenAIResponse, OpenAISystemRequestMessage, StreamOptions,
};
use super::provider_trait::InferenceProvider;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, ContentBlock, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, ProviderInferenceResponse, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStream,
};
use crate::model::CredentialLocation;

#[derive(Debug)]
pub struct VLLMProvider {
    pub model_name: String,
    pub api_base: Url,
    pub credentials: VLLMCredentials,
}

pub fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("VLLM_API_KEY".to_string())
}

#[derive(Debug)]
pub enum VLLMCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl VLLMCredentials {
    fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            VLLMCredentials::Static(api_key) => Ok(Some(api_key)),
            VLLMCredentials::Dynamic(key_name) => {
                Ok(Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: "vLLM".to_string(),
                    })
                })?))
            }
            VLLMCredentials::None => Ok(None),
        }
    }
}

/// Key differences between vLLM and OpenAI inference:
/// - vLLM supports guided decoding
/// - vLLM only supports a specific tool and nothing else (and the implementation varies among LLMs)
///   **Today, we can't support tools** so we are leaving it as an open issue (#169).
impl InferenceProvider for VLLMProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = VLLMRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(Some(&self.api_base))?;
        let start_time = Instant::now();
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let mut request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json");
        if let Some(key) = api_key {
            request_builder = request_builder.bearer_auth(key.expose_secret());
        }
        let res = request_builder
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to vLLM: {e}"),
                })
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response_body = res.json::<OpenAIResponse>().await.map_err(|e| {
                Error::new(ErrorDetails::VLLMServer {
                    message: format!("Error parsing response: {e}"),
                })
            })?;
            Ok(VLLMResponseWithMetadata {
                response: response_body,
                latency,
                request: request_body,
                generic_request: request,
            }
            .try_into()
            .map_err(map_openai_to_vllm_error)?)
        } else {
            Err(map_openai_to_vllm_error(handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::VLLMServer {
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
        let request_body = VLLMRequest::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::VLLMServer {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = get_chat_url(Some(&self.api_base))?;
        let start_time = Instant::now();
        let mut request_builder = http_client
            .post(request_url)
            .header("Content-Type", "application/json");
        if let Some(key) = api_key {
            request_builder = request_builder.bearer_auth(key.expose_secret());
        }
        let event_source = request_builder
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to vLLM: {e}"),
                })
            })?;
        let mut stream =
            Box::pin(stream_openai(event_source, start_time).map_err(map_openai_to_vllm_error));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(ErrorDetails::VLLMServer {
                    message: "Stream ended before first chunk".to_string(),
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
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "vLLM".to_string(),
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
            provider_type: "GCP Vertex Gemini".to_string(),
        }
        .into())
    }
}

fn map_openai_to_vllm_error(e: Error) -> Error {
    match e.get_owned_details() {
        ErrorDetails::OpenAIServer { message } => Error::new(ErrorDetails::VLLMServer { message }),
        ErrorDetails::OpenAIClient {
            message,
            status_code,
        } => ErrorDetails::VLLMClient {
            message,
            status_code,
        }
        .into(),
        e => e.into(),
    }
}

/// This struct defines the supported parameters for the vLLM inference API
/// See the [vLLM API documentation](https://docs.vllm.ai/en/stable/index.html)
/// for more details.
/// We are not handling many features of the API here.
#[derive(Debug, Serialize)]
struct VLLMRequest<'a> {
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
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    guided_json: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
}

impl<'a> VLLMRequest<'a> {
    pub fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest,
    ) -> Result<VLLMRequest<'a>, Error> {
        let guided_json = match (&request.json_mode, request.output_schema) {
            (
                ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict,
                Some(schema),
            ) => Some(schema),
            _ => None,
        };
        let stream_options = match request.stream {
            true => Some(StreamOptions {
                include_usage: true,
            }),
            false => None,
        };
        let messages = prepare_vllm_messages(request);
        // TODO (#169): Implement tool calling.
        if request.tool_config.is_some() {
            return Err(ErrorDetails::VLLMClient {
                status_code: reqwest::StatusCode::BAD_REQUEST,
                message: "TensorZero does not support tool use with vLLM. Please use a different provider.".to_string(),
            }.into());
        }

        Ok(VLLMRequest {
            messages,
            model,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            stream: request.stream,
            stream_options,
            guided_json,
            seed: request.seed,
        })
    }
}

struct VLLMResponseWithMetadata<'a> {
    response: OpenAIResponse,
    latency: Latency,
    request: VLLMRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<VLLMResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: VLLMResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let VLLMResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
        } = value;
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::OpenAIServer {
                message: format!("Error parsing response: {e}"),
            })
        })?;
        if response.choices.len() != 1 {
            return Err(ErrorDetails::OpenAIServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
            }
            .into());
        }
        let usage = response.usage.into();
        let message = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::VLLMServer {
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
            Error::new(ErrorDetails::VLLMServer {
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

pub(super) fn prepare_vllm_messages<'a>(
    request: &'a ModelInferenceRequest,
) -> Vec<OpenAIRequestMessage<'a>> {
    let mut messages: Vec<OpenAIRequestMessage> = request
        .messages
        .iter()
        .flat_map(tensorzero_to_openai_messages)
        .collect();
    if let Some(system_msg) = tensorzero_to_vllm_system_message(request.system.as_deref()) {
        messages.insert(0, system_msg);
    }
    messages
}

fn tensorzero_to_vllm_system_message(system: Option<&str>) -> Option<OpenAIRequestMessage<'_>> {
    system.map(|instructions| {
        OpenAIRequestMessage::System(OpenAISystemRequestMessage {
            content: Cow::Borrowed(instructions),
        })
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use serde_json::json;

    use super::*;

    use crate::inference::{
        providers::common::WEATHER_TOOL_CONFIG,
        types::{FunctionType, RequestMessage, Role},
    };

    #[test]
    fn test_vllm_request_new() {
        let output_schema = json!({
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "location": {"type": "string"}
            }
        });
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
            seed: Some(69),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
        };

        let vllm_request = VLLMRequest::new("llama-v3-8b", &request_with_tools).unwrap();

        assert_eq!(vllm_request.model, "llama-v3-8b");
        assert_eq!(vllm_request.messages.len(), 1);
        assert_eq!(vllm_request.temperature, Some(0.5));
        assert_eq!(vllm_request.max_tokens, Some(100));
        assert!(!vllm_request.stream);
        assert_eq!(vllm_request.guided_json, Some(&output_schema));

        let output_schema = json!({
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "location": {"type": "string"},
            }
        });
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
            seed: Some(69),
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
        };

        let err = VLLMRequest::new("llama-v3-8b", &request_with_tools).unwrap_err();
        assert!(err
            .to_string()
            .contains("TensorZero does not support tool use with vLLM"));
    }
}
