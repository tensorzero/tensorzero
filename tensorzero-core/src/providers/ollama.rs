use std::borrow::Cow;

use futures::StreamExt;
use lazy_static::lazy_static;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;

use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DelayedError, DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::InferenceProvider;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::chat_completion_inference_params::{
    ChatCompletionInferenceParamsV2, warn_inference_parameter_not_supported,
};
use crate::inference::types::{
    ContentBlockChunk, ContentBlockOutput, FinishReason, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseArgs, ProviderInferenceResponseChunk,
    ProviderInferenceResponseStreamInner, TextChunk, Usage,
    batch::StartBatchProviderInferenceResponse, current_timestamp,
};
use crate::model::{Credential, ModelProvider};
use crate::providers::chat_completions::prepare_chat_completion_tools;
use crate::providers::helpers::inject_extra_request_data_and_send;
use crate::tool::ToolCallChunk;

lazy_static! {
    static ref OLLAMA_DEFAULT_BASE_URL: Url = {
        #[expect(clippy::expect_used)]
        Url::parse("http://localhost:11434").expect("Failed to parse OLLAMA_DEFAULT_BASE_URL")
    };
}

const PROVIDER_NAME: &str = "Ollama";
pub const PROVIDER_TYPE: &str = "ollama";

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OllamaProvider {
    model_name: String,
    #[serde(skip)]
    credentials: OllamaCredentials,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_base: Option<Url>,
}

impl OllamaProvider {
    pub fn new(model_name: String, api_base: Option<Url>, credentials: OllamaCredentials) -> Self {
        OllamaProvider {
            model_name,
            credentials,
            api_base,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    fn get_api_base(&self) -> &Url {
        self.api_base.as_ref().unwrap_or(&OLLAMA_DEFAULT_BASE_URL)
    }
}

fn get_chat_url(base_url: &Url) -> Result<Url, Error> {
    base_url.join("api/chat").map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: format!("Failed to join URL: {}", DisplayOrDebugGateway::new(e)),
        })
    })
}

#[derive(Clone, Debug)]
pub enum OllamaCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<OllamaCredentials>,
        fallback: Box<OllamaCredentials>,
    },
}

impl TryFrom<Credential> for OllamaCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(OllamaCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(OllamaCredentials::Dynamic(key_name)),
            Credential::None => Ok(OllamaCredentials::None),
            Credential::Missing => Ok(OllamaCredentials::None),
            Credential::WithFallback { default, fallback } => Ok(OllamaCredentials::WithFallback {
                default: Box::new((*default).try_into()?),
                fallback: Box::new((*fallback).try_into()?),
            }),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Ollama provider".to_string(),
            })),
        }
    }
}

impl OllamaCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, DelayedError> {
        match self {
            OllamaCredentials::Static(api_key) => Ok(Some(api_key)),
            OllamaCredentials::Dynamic(key_name) => {
                let key = dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })?;
                Ok(Some(key))
            }
            OllamaCredentials::WithFallback { default, fallback } => {
                match default.get_api_key(dynamic_api_keys) {
                    Ok(key) => Ok(key),
                    Err(e) => {
                        e.log_at_level(
                            "Using fallback credential, as default credential is unavailable: ",
                            tracing::Level::WARN,
                        );
                        fallback.get_api_key(dynamic_api_keys)
                    }
                }
            }
            OllamaCredentials::None => Ok(None),
        }
    }
}

impl InferenceProvider for OllamaProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body =
            serde_json::to_value(OllamaRequest::new(&self.model_name, request).await?)
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing Ollama request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;
        let request_url = get_chat_url(self.get_api_base())?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();

        let mut request_builder = http_client.post(request_url);
        if let Some(key) = api_key {
            request_builder = request_builder.bearer_auth(key.expose_secret());
        }

        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;

        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let response: OllamaResponse = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };
            Ok(OllamaResponseWithMetadata {
                response,
                raw_response,
                latency,
                raw_request,
                generic_request: request,
            }
            .try_into()?)
        } else {
            let status = res.status();
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            Err(handle_ollama_error(&raw_request, status, &response))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body =
            serde_json::to_value(OllamaRequest::new(&self.model_name, request).await?)
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing Ollama request: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?;

        let request_url = get_chat_url(self.get_api_base())?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();

        let mut request_builder = http_client.post(request_url);
        if let Some(key) = api_key {
            request_builder = request_builder.bearer_auth(key.expose_secret());
        }

        // For streaming, we need to send the request and get a response manually
        // since Ollama uses NDJSON, not SSE
        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            request_builder,
        )
        .await?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error reading error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            return Err(handle_ollama_error(&raw_request, status, &text));
        }

        let stream = stream_ollama(res, start_time, raw_request.clone()).peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_NAME.to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

/// Request format for Ollama's native /api/chat endpoint
/// See: https://docs.ollama.com/api/chat
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(Default))]
struct OllamaRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaRequestMessage<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<OllamaFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OllamaFormat {
    Text(String),
    Schema(Value),
}

impl OllamaFormat {
    fn new(
        json_mode: ModelInferenceRequestJsonMode,
        output_schema: Option<&Value>,
    ) -> Option<Self> {
        match json_mode {
            ModelInferenceRequestJsonMode::On => Some(OllamaFormat::Text("json".to_string())),
            ModelInferenceRequestJsonMode::Off => None,
            ModelInferenceRequestJsonMode::Strict => match output_schema {
                Some(schema) => Some(OllamaFormat::Schema(schema.clone())),
                None => Some(OllamaFormat::Text("json".to_string())),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct OllamaRequestMessage<'a> {
    role: &'a str,
    content: Cow<'a, str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OllamaToolCall {
    function: OllamaToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OllamaToolCallFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<Value>,
}

#[derive(Debug, Serialize)]
struct OllamaTool<'a> {
    r#type: &'static str,
    function: OllamaToolFunction<'a>,
}

#[derive(Debug, Serialize)]
struct OllamaToolFunction<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    parameters: &'a Value,
}

fn apply_inference_params(
    _request: &mut OllamaRequest,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "reasoning_effort", None);
    }

    if service_tier.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier", None);
    }

    if thinking_budget_tokens.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "thinking_budget_tokens", None);
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
}

impl<'a> OllamaRequest<'a> {
    pub async fn new(
        model: &'a str,
        request: &'a ModelInferenceRequest<'_>,
    ) -> Result<OllamaRequest<'a>, Error> {
        let messages = prepare_ollama_messages(request);
        let format = OllamaFormat::new(request.json_mode, request.output_schema);

        let options = if request.temperature.is_some()
            || request.top_p.is_some()
            || request.seed.is_some()
            || request.max_tokens.is_some()
            || request.presence_penalty.is_some()
            || request.frequency_penalty.is_some()
            || request.stop_sequences.is_some()
        {
            Some(OllamaOptions {
                temperature: request.temperature,
                top_p: request.top_p,
                seed: request.seed,
                num_predict: request.max_tokens,
                presence_penalty: request.presence_penalty,
                frequency_penalty: request.frequency_penalty,
                stop: request.stop_sequences.as_ref().map(|s| s.to_vec()),
            })
        } else {
            None
        };

        let tools = prepare_ollama_tools(request)?;

        let mut ollama_request = OllamaRequest {
            model,
            messages,
            stream: request.stream,
            format,
            options,
            tools,
            keep_alive: None,
        };

        apply_inference_params(&mut ollama_request, &request.inference_params_v2);

        Ok(ollama_request)
    }
}

fn prepare_ollama_messages<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Vec<OllamaRequestMessage<'a>> {
    use crate::inference::types::{ContentBlock, Role, Text};

    let mut messages = Vec::new();

    // Add system message if present
    if let Some(system) = &request.system {
        messages.push(OllamaRequestMessage {
            role: "system",
            content: Cow::Borrowed(system.as_str()),
            images: None,
            tool_calls: None,
        });
    }

    // Add conversation messages
    for msg in &request.messages {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };

        let mut content_parts = Vec::new();
        let images: Vec<String> = Vec::new(); // Placeholder for future image support
        let mut tool_calls = Vec::new();

        for block in &msg.content {
            match block {
                ContentBlock::Text(Text { text }) => {
                    content_parts.push(text.as_str());
                }
                ContentBlock::File(_file) => {
                    // Note: Ollama supports images via base64. For now, we'll skip files
                    // as properly handling LazyFile requires async resolution.
                    // A future enhancement could resolve the file and include the base64 data.
                    tracing::warn!(
                        "Ollama provider does not yet support file/image inputs. Skipping file block."
                    );
                }
                ContentBlock::ToolCall(tc) => {
                    tool_calls.push(OllamaToolCall {
                        function: OllamaToolCallFunction {
                            name: tc.name.clone(),
                            arguments: Some(
                                serde_json::from_str(&tc.arguments).unwrap_or(Value::Null),
                            ),
                        },
                    });
                }
                ContentBlock::ToolResult(tr) => {
                    // Tool results are sent as a separate message with role "tool"
                    messages.push(OllamaRequestMessage {
                        role: "tool",
                        content: Cow::Owned(tr.result.clone()),
                        images: None,
                        tool_calls: None,
                    });
                    continue;
                }
                ContentBlock::Thought(_thought) => {
                    // Skip thought blocks - Ollama doesn't support extended thinking
                }
                ContentBlock::Unknown(_) => {
                    // Skip unknown blocks
                }
            }
        }

        let content = if content_parts.is_empty() {
            Cow::Borrowed("")
        } else {
            Cow::Owned(content_parts.join("\n"))
        };

        messages.push(OllamaRequestMessage {
            role,
            content,
            images: if images.is_empty() { None } else { Some(images) },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
        });
    }

    messages
}

fn prepare_ollama_tools<'a>(
    request: &'a ModelInferenceRequest<'_>,
) -> Result<Option<Vec<OllamaTool<'a>>>, Error> {
    let (chat_tools, _tool_choice, _parallel_tool_calls) =
        prepare_chat_completion_tools(request, false)?;

    if let Some(tools) = chat_tools {
        let ollama_tools: Vec<OllamaTool<'a>> = tools
            .into_iter()
            .map(|tool| OllamaTool {
                r#type: "function",
                function: OllamaToolFunction {
                    name: tool.function.name,
                    description: tool.function.description,
                    parameters: tool.function.parameters,
                },
            })
            .collect();
        Ok(Some(ollama_tools))
    } else {
        Ok(None)
    }
}

/// Response format for Ollama's native /api/chat endpoint
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: OllamaResponseMessage,
    #[serde(default)]
    done: bool,
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
    #[cfg_attr(not(test), expect(dead_code))]
    role: String,
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

/// Streaming response chunk from Ollama
#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    message: OllamaStreamMessage,
    #[serde(default)]
    done: bool,
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

struct OllamaResponseWithMetadata<'a> {
    response: OllamaResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<OllamaResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;

    fn try_from(value: OllamaResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let OllamaResponseWithMetadata {
            response,
            latency,
            raw_request,
            generic_request,
            raw_response,
        } = value;

        let usage = Usage {
            input_tokens: response.prompt_eval_count,
            output_tokens: response.eval_count,
        };

        let mut content: Vec<ContentBlockOutput> = Vec::new();

        if !response.message.content.is_empty() {
            content.push(response.message.content.into());
        }

        if let Some(tool_calls) = response.message.tool_calls {
            for tc in tool_calls {
                content.push(ContentBlockOutput::ToolCall(crate::tool::ToolCall {
                    id: uuid::Uuid::now_v7().to_string(),
                    name: tc.function.name,
                    arguments: tc
                        .function
                        .arguments
                        .map(|a| a.to_string())
                        .unwrap_or_default(),
                }));
            }
        }

        let finish_reason = match response.done_reason.as_deref() {
            Some("stop") => Some(FinishReason::Stop),
            Some("length") => Some(FinishReason::Length),
            Some("tool_calls") => Some(FinishReason::ToolCall),
            _ => {
                if response.done {
                    Some(FinishReason::Stop)
                } else {
                    None
                }
            }
        };

        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response,
                usage,
                latency,
                finish_reason,
            },
        ))
    }
}

fn handle_ollama_error(
    raw_request: &str,
    status: reqwest::StatusCode,
    response_body: &str,
) -> Error {
    let error_message = match serde_json::from_str::<Value>(response_body) {
        Ok(json) => json
            .get("error")
            .and_then(|e| e.as_str())
            .unwrap_or(response_body)
            .to_string(),
        Err(_) => response_body.to_string(),
    };

    Error::new(ErrorDetails::InferenceServer {
        message: format!("Ollama error ({status}): {error_message}"),
        raw_request: Some(raw_request.to_string()),
        raw_response: Some(response_body.to_string()),
        provider_type: PROVIDER_TYPE.to_string(),
    })
}

fn stream_ollama(
    response: reqwest::Response,
    start_time: Instant,
    raw_request: String,
) -> ProviderInferenceResponseStreamInner {
    use futures::stream::unfold;

    let stream = unfold(
        (response, start_time, raw_request, false, String::new()),
        |(mut response, start_time, raw_request, mut done, mut buffer)| async move {
            if done {
                return None;
            }

            match response.chunk().await {
                Ok(Some(chunk)) => {
                    let chunk_str = match std::str::from_utf8(&chunk) {
                        Ok(s) => s,
                        Err(e) => {
                            return Some((
                                Err(Error::new(ErrorDetails::InferenceServer {
                                    message: format!("Invalid UTF-8 in response: {e}"),
                                    raw_request: Some(raw_request.clone()),
                                    raw_response: None,
                                    provider_type: PROVIDER_TYPE.to_string(),
                                })),
                                (response, start_time, raw_request, true, buffer),
                            ));
                        }
                    };

                    // Append to buffer and process complete lines
                    buffer.push_str(chunk_str);

                    // Process NDJSON - each line is a separate JSON object
                    if let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        if line.is_empty() {
                            // Empty line, return an empty chunk
                            return Some((
                                Ok(ProviderInferenceResponseChunk {
                                    content: vec![],
                                    raw_response: String::new(),
                                    created: current_timestamp(),
                                    usage: None,
                                    latency: start_time.elapsed(),
                                    finish_reason: None,
                                }),
                                (response, start_time, raw_request, done, buffer),
                            ));
                        }

                        match serde_json::from_str::<OllamaStreamChunk>(&line) {
                            Ok(ollama_chunk) => {
                                if ollama_chunk.done {
                                    done = true;
                                }
                                let chunk = ollama_stream_chunk_to_response(
                                    ollama_chunk,
                                    start_time.elapsed(),
                                    &line,
                                );
                                Some((
                                    Ok(chunk),
                                    (response, start_time, raw_request, done, buffer),
                                ))
                            }
                            Err(e) => Some((
                                Err(Error::new(ErrorDetails::InferenceServer {
                                    message: format!("Failed to parse Ollama stream chunk: {e}"),
                                    raw_request: Some(raw_request.clone()),
                                    raw_response: Some(line),
                                    provider_type: PROVIDER_TYPE.to_string(),
                                })),
                                (response, start_time, raw_request, true, buffer),
                            )),
                        }
                    } else {
                        // No complete line yet, return an empty chunk and continue
                        Some((
                            Ok(ProviderInferenceResponseChunk {
                                content: vec![],
                                raw_response: String::new(),
                                created: current_timestamp(),
                                usage: None,
                                latency: start_time.elapsed(),
                                finish_reason: None,
                            }),
                            (response, start_time, raw_request, done, buffer),
                        ))
                    }
                }
                Ok(None) => None,
                Err(e) => Some((
                    Err(Error::new(ErrorDetails::InferenceServer {
                        message: format!("Error reading stream: {e}"),
                        raw_request: Some(raw_request.clone()),
                        raw_response: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    })),
                    (response, start_time, raw_request, true, buffer),
                )),
            }
        },
    );

    Box::pin(stream)
}

fn ollama_stream_chunk_to_response(
    chunk: OllamaStreamChunk,
    latency: std::time::Duration,
    raw_response: &str,
) -> ProviderInferenceResponseChunk {
    let mut content = Vec::new();

    if let Some(text) = chunk.message.content
        && !text.is_empty()
    {
        content.push(ContentBlockChunk::Text(TextChunk {
            id: uuid::Uuid::now_v7().to_string(),
            text,
        }));
    }

    if let Some(tool_calls) = chunk.message.tool_calls {
        for tc in tool_calls {
            content.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                id: uuid::Uuid::now_v7().to_string(),
                raw_name: Some(tc.function.name),
                raw_arguments: tc
                    .function
                    .arguments
                    .map(|a| a.to_string())
                    .unwrap_or_default(),
            }));
        }
    }

    let finish_reason = match chunk.done_reason.as_deref() {
        Some("stop") => Some(FinishReason::Stop),
        Some("length") => Some(FinishReason::Length),
        Some("tool_calls") => Some(FinishReason::ToolCall),
        _ => {
            if chunk.done {
                Some(FinishReason::Stop)
            } else {
                None
            }
        }
    };

    let usage = if chunk.prompt_eval_count.is_some() || chunk.eval_count.is_some() {
        Some(Usage {
            input_tokens: chunk.prompt_eval_count,
            output_tokens: chunk.eval_count,
        })
    } else {
        None
    };

    ProviderInferenceResponseChunk {
        content,
        raw_response: raw_response.to_string(),
        created: current_timestamp(),
        usage,
        latency,
        finish_reason,
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;
    use uuid::Uuid;

    use super::*;

    use crate::inference::types::{FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role};
    use crate::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};

    #[tokio::test]
    async fn test_ollama_request_new() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
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
            extra_body: Default::default(),
            ..Default::default()
        };

        let ollama_request = OllamaRequest::new("llama3.2", &request_with_tools)
            .await
            .expect("failed to create Ollama Request during test");

        assert_eq!(ollama_request.model, "llama3.2");
        assert_eq!(ollama_request.messages.len(), 1);
        assert!(!ollama_request.stream);
        assert!(ollama_request.options.is_some());

        let options = ollama_request.options.as_ref().unwrap();
        assert_eq!(options.temperature, Some(0.5));
        assert_eq!(options.num_predict, Some(100));
        assert_eq!(options.seed, Some(69));

        assert!(ollama_request.tools.is_some());
        let tools = ollama_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool = &tools[0];
        assert_eq!(tool.function.name, WEATHER_TOOL.name());
        assert_eq!(tool.function.parameters, WEATHER_TOOL.parameters());
    }

    #[tokio::test]
    async fn test_ollama_request_with_system() {
        let request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["Hello".to_string().into()],
            }],
            system: Some("You are a helpful assistant.".to_string().into()),
            temperature: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: None,
            stream: false,
            seed: None,
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let ollama_request = OllamaRequest::new("llama3.2", &request)
            .await
            .expect("failed to create Ollama Request");

        assert_eq!(ollama_request.messages.len(), 2);
        assert_eq!(ollama_request.messages[0].role, "system");
        assert_eq!(
            ollama_request.messages[0].content,
            "You are a helpful assistant."
        );
        assert_eq!(ollama_request.messages[1].role, "user");
        assert_eq!(ollama_request.messages[1].content, "Hello");
    }

    #[test]
    fn test_ollama_api_base() {
        assert_eq!(OLLAMA_DEFAULT_BASE_URL.as_str(), "http://localhost:11434/");
    }

    #[test]
    fn test_credential_to_ollama_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds: OllamaCredentials = OllamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OllamaCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = OllamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OllamaCredentials::Dynamic(_)));

        // Test Missing credential - should map to None since Ollama doesn't require auth
        let generic = Credential::Missing;
        let creds = OllamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OllamaCredentials::None));

        // Test None credential
        let generic = Credential::None;
        let creds = OllamaCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, OllamaCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = OllamaCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_ollama_format() {
        // Off mode
        let format = OllamaFormat::new(ModelInferenceRequestJsonMode::Off, None);
        assert!(format.is_none());

        // On mode
        let format = OllamaFormat::new(ModelInferenceRequestJsonMode::On, None);
        assert!(matches!(format, Some(OllamaFormat::Text(ref s)) if s == "json"));

        // Strict mode with schema
        let schema = serde_json::json!({"type": "object"});
        let format = OllamaFormat::new(ModelInferenceRequestJsonMode::Strict, Some(&schema));
        assert!(matches!(format, Some(OllamaFormat::Schema(_))));

        // Strict mode without schema
        let format = OllamaFormat::new(ModelInferenceRequestJsonMode::Strict, None);
        assert!(matches!(format, Some(OllamaFormat::Text(ref s)) if s == "json"));
    }

    #[test]
    fn test_ollama_response_parsing() {
        let response_json = r#"{
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 20
        }"#;

        let response: OllamaResponse = serde_json::from_str(response_json).unwrap();
        assert_eq!(response.message.role, "assistant");
        assert_eq!(response.message.content, "Hello! How can I help you today?");
        assert!(response.done);
        assert_eq!(response.done_reason, Some("stop".to_string()));
        assert_eq!(response.prompt_eval_count, Some(10));
        assert_eq!(response.eval_count, Some(20));
    }

    #[tokio::test]
    async fn test_ollama_response_with_metadata_try_into() {
        let response = OllamaResponse {
            message: OllamaResponseMessage {
                role: "assistant".to_string(),
                content: "Hello, world!".to_string(),
                tool_calls: None,
            },
            done: true,
            done_reason: Some("stop".to_string()),
            prompt_eval_count: Some(10),
            eval_count: Some(20),
        };

        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
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
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let ollama_response_with_metadata = OllamaResponseWithMetadata {
            response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            raw_request: "test_request".to_string(),
            generic_request: &generic_request,
        };

        let inference_response: ProviderInferenceResponse =
            ollama_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 1);
        assert_eq!(
            inference_response.output[0],
            "Hello, world!".to_string().into()
        );
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(inference_response.raw_response, "test_response");
        assert_eq!(inference_response.usage.input_tokens, Some(10));
        assert_eq!(inference_response.usage.output_tokens, Some(20));
    }

    #[test]
    fn test_ollama_apply_inference_params() {
        use crate::inference::types::chat_completion_inference_params::ServiceTier;
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: Some(ServiceTier::Default),
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = OllamaRequest::default();

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort warns
        assert!(logs_contain(
            "Ollama does not support the inference parameter `reasoning_effort`, so it will be ignored."
        ));

        // Test that service_tier warns
        assert!(logs_contain(
            "Ollama does not support the inference parameter `service_tier`, so it will be ignored."
        ));

        // Test that thinking_budget_tokens warns
        assert!(logs_contain(
            "Ollama does not support the inference parameter `thinking_budget_tokens`, so it will be ignored."
        ));

        // Test that verbosity warns
        assert!(logs_contain(
            "Ollama does not support the inference parameter `verbosity`, so it will be ignored."
        ));
    }
}
