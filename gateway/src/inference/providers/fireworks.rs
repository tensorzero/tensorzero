use futures::{StreamExt, TryStreamExt};
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use serde_json::Value;
use tokio::time::Instant;

use crate::{
    error::Error,
    inference::types::{
        JSONMode, Latency, ModelInferenceRequest, ProviderInferenceResponse,
        ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
    },
};

use super::{
    openai::{
        get_chat_url, handle_openai_error, prepare_openai_messages, prepare_openai_tools,
        stream_openai, OpenAIFunction, OpenAIRequestMessage, OpenAIResponseWithLatency, OpenAITool,
        OpenAIToolChoice, OpenAIToolType,
    },
    provider_trait::InferenceProvider,
};

#[derive(Debug)]
pub struct FireworksProvider {
    pub model_name: String,
    pub api_key: Option<SecretString>,
}

/// Key differences between Fireworks and OpenAI inference:
/// - Fireworks allows you to specify output format in JSON mode
/// - Fireworks automatically returns usage in streaming inference, we don't have to ask for it
/// - Fireworks allows you to auto-truncate requests that have too many tokens
///   (there are 2 ways to do it, we have the default of auto-truncation to the max window size)
impl InferenceProvider for FireworksProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<ProviderInferenceResponse, Error> {
        let api_key = self.api_key.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "Fireworks".to_string(),
        })?;
        let api_base = Some("https://api.fireworks.ai/inference/v1/");
        let request_body = FireworksRequest::new(&self.model_name, request);
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to Fireworks: {e}"),
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| Error::FireworksServer {
                message: format!("Error parsing text response: {e}"),
            })?;

            let response = serde_json::from_str(&response).map_err(|e| Error::FireworksServer {
                message: format!("Error parsing JSON response: {e}: {response}"),
            })?;

            Ok(OpenAIResponseWithLatency { response, latency }
                .try_into()
                .map_err(map_openai_to_fireworks_error)?)
        } else {
            handle_openai_error(
                res.status(),
                &res.text().await.map_err(|e| Error::FireworksServer {
                    message: format!("Error parsing error response: {e}"),
                })?,
            )
            .map_err(map_openai_to_fireworks_error)
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
            provider_name: "Fireworks".to_string(),
        })?;
        let request_body = FireworksRequest::new(&self.model_name, request);
        let api_base = Some("https://api.fireworks.ai/inference/v1/");
        let request_url = get_chat_url(api_base)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to Fireworks: {e}"),
            })?;
        let mut stream = Box::pin(
            stream_openai(event_source, start_time).map_err(map_openai_to_fireworks_error),
        );
        // Get a single chunk from the stream and make sure it is OK then send to client.
        // We want to do this here so that we can tell that the request is working.
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::FireworksServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };
        Ok((chunk, stream))
    }
}

fn map_openai_to_fireworks_error(e: Error) -> Error {
    match e {
        Error::OpenAIServer { message } => Error::FireworksServer { message },
        Error::OpenAIClient {
            message,
            status_code,
        } => Error::FireworksClient {
            message,
            status_code,
        },
        _ => e,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum FireworksResponseFormat<'a> {
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<&'a Value>, // the desired JSON schema
    },
    #[default]
    Text,
}

/// This struct defines the supported parameters for the Fireworks inference API
/// See the [Fireworks API documentation](https://docs.fireworks.ai/api-reference/post-chatcompletions)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, service_tier, stop, user,
/// or context_length_exceeded_behavior.
/// NOTE: Fireworks does not support seed.
#[derive(Debug, Serialize)]
struct FireworksRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<FireworksResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<FireworksTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
}

impl<'a> FireworksRequest<'a> {
    pub fn new(model: &'a str, request: &'a ModelInferenceRequest) -> FireworksRequest<'a> {
        // NB: Fireworks will throw an error if you give FireworksResponseFormat::Text and then also include tools.
        // So we just don't include it as Text is the same as None anyway.
        let response_format = match request.json_mode {
            JSONMode::On | JSONMode::Strict => Some(FireworksResponseFormat::JsonObject {
                schema: request.output_schema,
            }),
            JSONMode::Off => None,
        };
        let messages = prepare_openai_messages(request);
        let (tools, tool_choice, _) = prepare_openai_tools(request);
        let tools = tools.map(|t| t.into_iter().map(|tool| tool.into()).collect());

        FireworksRequest {
            messages,
            model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream,
            response_format,
            tools,
            tool_choice,
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct FireworksTool<'a> {
    r#type: OpenAIToolType,
    function: OpenAIFunction<'a>,
}

impl<'a> From<OpenAITool<'a>> for FireworksTool<'a> {
    fn from(tool: OpenAITool<'a>) -> Self {
        FireworksTool {
            r#type: tool.r#type,
            function: tool.function,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::inference::providers::common::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::providers::openai::OpenAIToolType;
    use crate::inference::providers::openai::{SpecificToolChoice, SpecificToolFunction};
    use crate::inference::types::{FunctionType, RequestMessage, Role};

    #[test]
    fn test_fireworks_request_new() {
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
            json_mode: JSONMode::On,
            tool_config: Some(&WEATHER_TOOL_CONFIG),
            function_type: FunctionType::Chat,
            output_schema: None,
        };

        let fireworks_request =
            FireworksRequest::new("accounts/fireworks/models/llama-v3-8b", &request_with_tools);

        assert_eq!(
            fireworks_request.model,
            "accounts/fireworks/models/llama-v3-8b"
        );
        assert_eq!(fireworks_request.messages.len(), 1);
        assert_eq!(fireworks_request.temperature, Some(0.5));
        assert_eq!(fireworks_request.max_tokens, Some(100));
        assert!(!fireworks_request.stream);
        assert_eq!(
            fireworks_request.response_format,
            Some(FireworksResponseFormat::JsonObject {
                schema: request_with_tools.output_schema,
            })
        );
        assert!(fireworks_request.tools.is_some());
        let tools = fireworks_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            fireworks_request.tool_choice,
            Some(OpenAIToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );
    }
}
