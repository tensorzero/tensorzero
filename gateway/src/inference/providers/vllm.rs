use futures::{StreamExt, TryStreamExt};
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use super::openai::{
    get_chat_url, handle_openai_error, prepare_openai_messages, stream_openai,
    OpenAIRequestMessage, OpenAIResponse, OpenAIResponseWithLatency, StreamOptions,
};
use super::provider_trait::InferenceProvider;
use crate::error::Error;
use crate::inference::types::{
    Latency, ModelInferenceRequest, ModelInferenceRequestJsonMode, ProviderInferenceResponse,
    ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
};

#[derive(Debug)]
pub struct VLLMProvider {
    pub model_name: String,
    pub api_key: Option<SecretString>,
    pub api_base: Url,
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
    ) -> Result<ProviderInferenceResponse, Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "vLLM".to_string(),
        })?;
        let request_body = VLLMRequest::new(&self.model_name, request)?;
        let request_url = get_chat_url(Some(&self.api_base))?;
        let start_time = Instant::now();
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
            Ok(OpenAIResponseWithLatency {
                response: response_body,
                latency,
            }
            .try_into()
            .map_err(map_openai_to_vllm_error)?)
        } else {
            handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::VLLMServer {
                    message: format!("Error parsing error response: {e}"),
                })?,
            )
            .map_err(map_openai_to_vllm_error)
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
        ),
        Error,
    > {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "vLLM".to_string(),
        })?;
        let request_body = VLLMRequest::new(&self.model_name, request)?;
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
        Ok((chunk, stream))
    }

    fn has_credentials(&self) -> bool {
        self.api_key.is_some()
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
        let messages = prepare_openai_messages(request);
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

#[cfg(test)]
mod tests {
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
            tool_config: Some(&WEATHER_TOOL_CONFIG),
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
        };

        let err = VLLMRequest::new("llama-v3-8b", &request_with_tools).unwrap_err();
        assert!(err
            .to_string()
            .contains("TensorZero does not support tool use with vLLM"));
    }
}
