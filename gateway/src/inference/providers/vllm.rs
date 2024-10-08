use std::borrow::Cow;

use futures::{StreamExt, TryStreamExt};
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use super::openai::{
    get_chat_url, handle_openai_error, stream_openai, tensorzero_to_openai_messages,
    OpenAIRequestMessage, OpenAIResponse, OpenAISystemRequestMessage, StreamOptions,
};
use super::provider_trait::{HasCredentials, InferenceProvider};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::inference::types::{
    ContentBlock, Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
};
use crate::model::ProviderCredentials;

#[derive(Debug)]
pub struct VLLMProvider {
    pub model_name: String,
    pub api_key: Option<SecretString>,
    pub api_base: Url,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VLLMCredentials<'a> {
    pub api_key: Cow<'a, SecretString>,
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
        api_key: ProviderCredentials<'a>,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body = VLLMRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(Some(&self.api_base))?;
        let start_time = Instant::now();
        let api_key = match &api_key {
            ProviderCredentials::VLLM(credentials) => &credentials.api_key,
            _ => {
                return Err(Error::ApiKeyMissing {
                    provider_name: "vLLM".to_string(),
                })
            }
        };
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to vLLM: {e}"),
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response_body =
                res.json::<OpenAIResponse>()
                    .await
                    .map_err(|e| Error::VLLMServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            Ok(VLLMResponseWithMetadata {
                response: response_body,
                latency,
                request: request_body,
            }
            .try_into()
            .map_err(map_openai_to_vllm_error)?)
        } else {
            Err(map_openai_to_vllm_error(handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::VLLMServer {
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
        let request_body = VLLMRequest::new(&self.model_name, request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| Error::VLLMServer {
            message: format!("Error serializing request: {e}"),
        })?;
        let api_key = match &api_key {
            ProviderCredentials::VLLM(credentials) => &credentials.api_key,
            _ => {
                return Err(Error::BadCredentialsPreInference {
                    provider_name: "vLLM".to_string(),
                })
            }
        };
        let request_url = get_chat_url(Some(&self.api_base))?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to vLLM: {e}"),
            })?;
        let mut stream =
            Box::pin(stream_openai(event_source, start_time).map_err(map_openai_to_vllm_error));
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::VLLMServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };
        Ok((chunk, stream, raw_request))
    }
}

impl HasCredentials for VLLMProvider {
    fn has_credentials(&self) -> bool {
        self.api_key.is_some()
    }
    fn get_credentials<'a>(
        &'a self,
        credentials: &'a InferenceCredentials,
    ) -> Result<ProviderCredentials<'a>, Error> {
        if let Some(api_key) = &self.api_key {
            if credentials.vllm.is_some() {
                return Err(Error::UnexpectedDynamicCredentials {
                    provider_name: "vLLM".to_string(),
                });
            }
            return Ok(ProviderCredentials::VLLM(Cow::Owned(VLLMCredentials {
                api_key: Cow::Borrowed(api_key),
            })));
        } else {
            match &credentials.vllm {
                Some(credentials) => Ok(ProviderCredentials::VLLM(Cow::Borrowed(credentials))),
                None => Err(Error::ApiKeyMissing {
                    provider_name: "vLLM".to_string(),
                }),
            }
        }
    }
}

fn map_openai_to_vllm_error(e: Error) -> Error {
    match e {
        Error::OpenAIServer { message } => Error::VLLMServer { message },
        Error::OpenAIClient {
            message,
            status_code,
        } => Error::VLLMClient {
            message,
            status_code,
        },
        _ => e,
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
            return Err(Error::VLLMClient {
                status_code: reqwest::StatusCode::BAD_REQUEST,
                message: "TensorZero does not support tool use with vLLM. Please use a different provider.".to_string(),
            });
        }

        Ok(VLLMRequest {
            messages,
            model,
            temperature: request.temperature,
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
}

impl<'a> TryFrom<VLLMResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: VLLMResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let VLLMResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
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
            .ok_or(Error::VLLMServer {
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

        Ok(ProviderInferenceResponse::new(
            content,
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
