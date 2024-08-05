use std::time::Duration;

use futures::StreamExt;
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::Latency;
use crate::{
    inference::types::{
        InferenceRequestMessage, InferenceResponseStream, ModelInferenceRequest,
        ModelInferenceResponse, ModelInferenceResponseChunk, Tool, ToolCall, ToolCallChunk,
        ToolChoice, ToolType, Usage,
    },
    model::ProviderConfig,
};

fn get_request_url(location: &str, project_id: &str, model_id: &str) -> String {
    format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent")
}

pub struct GCPVertexGeminiProvider;

impl InferenceProvider for GCPVertexGeminiProvider {
    /// GCP Vertex Gemini non-streaming API request
    async fn infer<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let (model_id, location, project_id, credentials) = match model {
            ProviderConfig::GCPVertexGemini {
                model_id,
                location,
                project_id,
                credentials,
            } => (
                model_id,
                location,
                project_id,
                credentials.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "GCP Vertex Gemini".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected GCP Vertex Gemini provider config".to_string(),
                })
            }
        };

        let request_body = AnthropicRequestBody::new(model_name, request)?;
        let start_time = Instant::now();
        let res = http_client
            .post(ANTHROPIC_BASE_URL)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("x-api-key", api_key.expose_secret())
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request: {e}"),
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let body =
                res.json::<AnthropicResponseBody>()
                    .await
                    .map_err(|e| Error::AnthropicServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            let body_with_latency = AnthropicResponseBodyWithLatency { body, latency };
            Ok(body_with_latency.try_into()?)
        } else {
            let response_code = res.status();
            let error_body =
                res.json::<AnthropicError>()
                    .await
                    .map_err(|e| Error::AnthropicServer {
                        message: format!("Error parsing response: {e}"),
                    })?;
            handle_anthropic_error(response_code, error_body.error)
        }
    }

    /// Anthropic streaming API request
    async fn infer_stream<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let (model_name, api_key) = match model {
            ProviderConfig::Anthropic {
                model_name,
                api_key,
            } => (
                model_name,
                api_key.as_ref().ok_or(Error::ApiKeyMissing {
                    provider_name: "Anthropic".to_string(),
                })?,
            ),
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected Anthropic provider config".to_string(),
                })
            }
        };
        let request_body = AnthropicRequestBody::new(model_name, request)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(ANTHROPIC_BASE_URL)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("content-type", "application/json")
            .header("x-api-key", api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to Anthropic: {e}"),
            })?;
        let mut stream = stream_anthropic(event_source, start_time).await;
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::AnthropicServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };
        Ok((chunk, stream))
    }
}

/// Maps events from Anthropic into the TensorZero format
/// Modified from the example [here](https://github.com/64bit/async-openai/blob/5c9c817b095e3bacb2b6c9804864cdf8b15c795e/async-openai/src/client.rs#L433)
/// At a high level, this function is handling low-level EventSource details and mapping the objects returned by Anthropic into our `InferenceResponseChunk` type
async fn stream_anthropic(
    mut event_source: EventSource,
    start_time: Instant,
) -> InferenceResponseStream {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let inference_id = Uuid::now_v7();
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    if let Err(_e) = tx.send(Err(Error::AnthropicServer {
                        message: e.to_string(),
                    })) {
                        // rx dropped
                        break;
                    }
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<AnthropicStreamMessage, Error> =
                            serde_json::from_str(&message.data).map_err(|e| {
                                {
                                    Error::AnthropicServer {
                                        message: format!(
                                            "Error parsing message: {}, Data: {}",
                                            e, message.data
                                        ),
                                    }
                                }
                            });
                        let response = match data {
                            Err(e) => Err(e),
                            Ok(data) => {
                                // Anthropic streaming API docs specify that this is the last message
                                if let AnthropicStreamMessage::MessageStop = data {
                                    break;
                                }
                                anthropic_to_tensorzero_stream_message(
                                    data,
                                    inference_id,
                                    start_time.elapsed(),
                                )
                            }
                        }
                        .transpose();

                        if let Some(stream_message) = response {
                            if tx.send(stream_message).is_err() {
                                // rx dropped
                                break;
                            }
                        }
                    }
                },
            }
        }

        event_source.close();
    });

    Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}

#[derive(Serialize)]
#[serde(rename_all="lowercase")]
enum GCPVertexGeminiRole {
    User,
    Model,
}

struct GCPVertexGeminiFunctionCall<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", untagged)]
enum GCPVertexGeminiContentPart<'a> {
    Text { text: &'a str },
    // InlineData { inline_data: Blob },
    // FileData { file_data: FileData },
    FunctionCall { function_call: GCPVertexGeminiFunctionCall },
    // FunctionResponse { function_response: FunctionResponse },
    // VideoMetadata { video_metadata: VideoMetadata },
}

struct GCPVertexGeminiContent {
    role: GCPVertexGeminiRole,
    parts: Vec<
}