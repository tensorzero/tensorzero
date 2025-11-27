use std::borrow::Cow;
use std::time::Duration;

use futures::{future::try_join_all, StreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::Event;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use super::helpers::check_new_tool_call_name;
use super::helpers::inject_extra_request_data_and_send_eventsource;
use crate::cache::ModelProviderRequest;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::warn_discarded_thought_block;
use crate::error::warn_discarded_unknown_chunk;
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::error::{DelayedError, DisplayOrDebugGateway, Error, ErrorDetails};
use crate::http::TensorZeroEventSource;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::chat_completion_inference_params::{
    warn_inference_parameter_not_supported, ChatCompletionInferenceParamsV2,
};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, serialize_or_log, ModelInferenceRequest,
    ObjectStorageFile, PeekableProviderInferenceResponseStream, ProviderInferenceResponse,
    ProviderInferenceResponseChunk, RequestMessage, Usage,
};
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, ContentBlockOutput, Latency, ModelInferenceRequestJsonMode,
    ProviderInferenceResponseArgs, ProviderInferenceResponseStreamInner, Role, Text, TextChunk,
    Thought, ThoughtChunk, Unknown, UnknownChunk,
};
use crate::inference::types::{FinishReason, FlattenUnknown};
use crate::inference::InferenceProvider;
use crate::model::{Credential, ModelProvider};
use crate::tool::FunctionToolConfig;
#[cfg(test)]
use crate::tool::{AllowedTools, AllowedToolsChoice};
use crate::tool::{ToolCall, ToolCallChunk, ToolCallConfig, ToolChoice};

use super::gcp_vertex_gemini::process_jsonschema_for_gcp_vertex_gemini;
use super::helpers::{convert_stream_error, inject_extra_request_data_and_send};

const PROVIDER_NAME: &str = "Google AI Studio Gemini";
pub const PROVIDER_TYPE: &str = "google_ai_studio_gemini";

/// Implements a subset of the Google AI Studio Gemini API as documented [here](https://ai.google.dev/gemini-api/docs/text-generation?lang=rest)
/// See the `GCPVertexGeminiProvider` struct docs for information about our handling 'thought' and unknown blocks.
#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GoogleAIStudioGeminiProvider {
    model_name: String,
    request_url: Url,
    streaming_request_url: Url,
    #[serde(skip)]
    credentials: GoogleAIStudioCredentials,
}

impl GoogleAIStudioGeminiProvider {
    pub fn new(model_name: String, credentials: GoogleAIStudioCredentials) -> Result<Self, Error> {
        let request_url = Url::parse(&format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        ))
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to parse request URL: {e}"),
            })
        })?;
        let streaming_request_url = Url::parse(&format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse",
        ))
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to parse streaming request URL: {e}"),
            })
        })?;
        Ok(GoogleAIStudioGeminiProvider {
            model_name,
            request_url,
            streaming_request_url,
            credentials,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Clone, Debug)]
pub enum GoogleAIStudioCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
    WithFallback {
        default: Box<GoogleAIStudioCredentials>,
        fallback: Box<GoogleAIStudioCredentials>,
    },
}

impl TryFrom<Credential> for GoogleAIStudioCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(GoogleAIStudioCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(GoogleAIStudioCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(GoogleAIStudioCredentials::None),
            Credential::WithFallback { default, fallback } => {
                Ok(GoogleAIStudioCredentials::WithFallback {
                    default: Box::new((*default).try_into()?),
                    fallback: Box::new((*fallback).try_into()?),
                })
            }
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Google AI Studio Gemini provider"
                    .to_string(),
            })),
        }
    }
}

impl GoogleAIStudioCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, DelayedError> {
        match self {
            GoogleAIStudioCredentials::Static(api_key) => Ok(api_key),
            GoogleAIStudioCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                })
            }
            GoogleAIStudioCredentials::WithFallback { default, fallback } => {
                // Try default first, fall back to fallback if it fails
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
            GoogleAIStudioCredentials::None => {
                Err(DelayedError::new(ErrorDetails::ApiKeyMissing {
                    provider_name: PROVIDER_NAME.to_string(),
                    message: "No credentials are set".to_string(),
                }))
            }
        }
    }
}

impl InferenceProvider for GoogleAIStudioGeminiProvider {
    /// Google AI Studio Gemini non-streaming API request
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body =
            serde_json::to_value(GeminiRequest::new(request).await?).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Gemini request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let mut url = self.request_url.clone();
        url.query_pairs_mut()
            .append_pair("key", api_key.expose_secret());
        let builder = http_client.post(url);
        let (res, raw_request) = inject_extra_request_data_and_send(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            builder,
        )
        .await?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;
            let response_with_latency = GeminiResponseWithMetadata {
                response,
                latency,
                raw_response,
                raw_request,
                generic_request: request,
                model_name,
                provider_name,
            };
            Ok(response_with_latency.try_into()?)
        } else {
            let response_code = res.status();
            let error_body = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
            handle_google_ai_studio_error(response_code, error_body)
        }
    }

    /// Google AI Studio Gemini streaming API request
    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let request_body =
            serde_json::to_value(GeminiRequest::new(request).await?).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error serializing Gemini request: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                })
            })?;
        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        let start_time = Instant::now();
        let mut url = self.streaming_request_url.clone();
        url.query_pairs_mut()
            .append_pair("key", api_key.expose_secret());
        let builder = http_client.post(url);
        let (event_source, raw_request) = inject_extra_request_data_and_send_eventsource(
            PROVIDER_TYPE,
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            request_body,
            builder,
        )
        .await?;
        let stream = stream_google_ai_studio_gemini(
            event_source,
            start_time,
            model_provider,
            model_name,
            provider_name,
            &raw_request,
        )
        .peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Google AI Studio Gemini".to_string(),
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

fn stream_google_ai_studio_gemini(
    mut event_source: TensorZeroEventSource,
    start_time: Instant,
    model_provider: &ModelProvider,
    model_name: &str,
    provider_name: &str,
    raw_request: &str,
) -> ProviderInferenceResponseStreamInner {
    let raw_request = raw_request.to_string();
    let discard_unknown_chunks = model_provider.discard_unknown_chunks;
    let model_name = model_name.to_string();
    let provider_name = provider_name.to_string();
    Box::pin(async_stream::stream! {
        let mut last_tool_name = None;
        let mut last_tool_idx = None;
        let mut last_thought_id = 0;
        let mut last_unknown_chunk_id = 0;
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    if matches!(e, reqwest_eventsource::Error::StreamEnded) {
                        break;
                    }
                    yield Err(convert_stream_error(raw_request.clone(), PROVIDER_TYPE.to_string(), e).await);
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<GeminiResponse, Error> = serde_json::from_str(&message.data).map_err(|e| {
                            Error::new(ErrorDetails::InferenceServer {
                                message: format!("Error parsing streaming JSON response: {}", DisplayOrDebugGateway::new(e)),
                                provider_type: PROVIDER_TYPE.to_string(),
                                raw_request: Some(raw_request.clone()),
                                raw_response: Some(message.data.clone()),
                            })
                        });
                        let data = match data {
                            Ok(data) => data,
                            Err(e) => {
                                yield Err(e);
                                continue;
                            }
                        };
                        yield convert_stream_response_with_metadata_to_chunk(
                            ConvertStreamResponseArgs {
                                raw_response: message.data,
                                response: data,
                                latency: start_time.elapsed(),
                                last_tool_name: &mut last_tool_name,
                                last_tool_idx: &mut last_tool_idx,
                                last_thought_id: &mut last_thought_id,
                                last_unknown_chunk_id: &mut last_unknown_chunk_id,
                                discard_unknown_chunks,
                                model_name: &model_name,
                                provider_name: &provider_name,
                            },
                        )
                    }
                }
            }
         }
    })
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum GeminiRole {
    User,
    Model,
}

impl From<Role> for GeminiRole {
    fn from(role: Role) -> Self {
        match role {
            Role::User => GeminiRole::User,
            Role::Assistant => GeminiRole::Model,
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct GeminiFunctionCall<'a> {
    name: &'a str,
    args: Value,
}

#[derive(Debug, PartialEq, Serialize)]
struct GeminiFunctionResponse<'a> {
    name: &'a str,
    response: Value,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct GeminiContentPart<'a> {
    #[serde(default)]
    thought: bool,
    #[serde(default)]
    thought_signature: Option<String>,
    #[serde(flatten)]
    #[serde(default)]
    data: FlattenUnknown<'a, GeminiPartData<'a>>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", untagged)]
enum GeminiPartData<'a> {
    Text {
        text: &'a str,
    },
    InlineData {
        #[serde(rename = "inline_data")]
        inline_data: GeminiInlineData,
    },
    // TODO (if needed): FileData { file_data: FileData },
    FunctionCall {
        function_call: GeminiFunctionCall<'a>,
    },
    FunctionResponse {
        function_response: GeminiFunctionResponse<'a>,
    },
    // TODO (if needed): ExecutableCode [docs](https://ai.google.dev/api/caching#ExecutableCode)
    // TODO (if needed): ExecutableCodeResult [docs](https://ai.google.dev/api/caching#CodeExecutionResult)
}

#[derive(Debug, PartialEq, Serialize)]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, PartialEq, Serialize)]
struct GeminiContent<'a> {
    role: GeminiRole,
    parts: Vec<GeminiContentPart<'a>>,
}

impl<'a> GeminiContent<'a> {
    async fn from_request_message(message: &'a RequestMessage) -> Result<Self, Error> {
        let role = GeminiRole::from(message.role);
        let mut output = Vec::with_capacity(message.content.len());
        let mut iter = message.content.iter();
        while let Some(block) = iter.next() {
            match block {
                ContentBlock::Thought(
                    thought @ Thought {
                        text,
                        signature,
                        summary: _,
                        provider_type: _,
                    },
                ) => {
                    // Gemini never produces 'thought: true' at the moment, and there's no documentation
                    // on whether or not they should be passed back in.
                    // As a result, we don't attempt to feed `Thought.text` back to Gemini, as this would
                    // require us to set 'thought: true' in the request.
                    // Instead, we just warn and discard the content block.
                    if text.is_some() {
                        warn_discarded_thought_block(PROVIDER_TYPE, thought);
                    } else if let Some(signature) = signature {
                        let next_block = iter.next();
                        match next_block {
                            None => {
                                return Err(Error::new(ErrorDetails::InferenceServer {
                                    message: "Thought block with signature must be followed by a content block in Gemini".to_string(),
                                    provider_type: PROVIDER_TYPE.to_string(),
                                    raw_request: None,
                                    raw_response: None,
                                }));
                            }
                            Some(ContentBlock::Thought(Thought { .. })) => {
                                return Err(Error::new(ErrorDetails::InferenceServer {
                                    message: "Thought block with signature cannot be followed by another thought block in Gemini".to_string(),
                                    provider_type: PROVIDER_TYPE.to_string(),
                                    raw_request: None,
                                    raw_response: None,
                                }));
                            }
                            Some(ContentBlock::Unknown(_)) => {
                                return Err(Error::new(ErrorDetails::InferenceServer {
                                    message: "Thought block with signature cannot be followed by an unknown block in Gemini".to_string(),
                                    provider_type: PROVIDER_TYPE.to_string(),
                                    raw_request: None,
                                    raw_response: None,
                                }));
                            }
                            Some(next_block) => {
                                let gemini_part =
                                    convert_non_thought_content_block(next_block).await?;
                                match gemini_part {
                                    FlattenUnknown::Normal(part) => {
                                        output.push(GeminiContentPart {
                                            thought: false,
                                            thought_signature: Some(signature.clone()),
                                            data: FlattenUnknown::Normal(part),
                                        });
                                    }
                                    // We should have handled this case above with `Some(ContentBlock::Unknown(_))`
                                    FlattenUnknown::Unknown(_) => {
                                        return Err(Error::new(ErrorDetails::InternalError {
                                            message: format!("Got unknown block after thought block. {IMPOSSIBLE_ERROR_MESSAGE}"),
                                        }));
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {
                    let part = convert_non_thought_content_block(block).await?;
                    match part {
                        FlattenUnknown::Normal(part) => {
                            output.push(GeminiContentPart {
                                thought: false,
                                thought_signature: None,
                                data: FlattenUnknown::Normal(part),
                            });
                        }
                        FlattenUnknown::Unknown(data) => {
                            output.push(GeminiContentPart {
                                thought: false,
                                thought_signature: None,
                                data: FlattenUnknown::Unknown(data),
                            });
                        }
                    }
                }
            }
        }
        Ok(GeminiContent {
            role,
            parts: output,
        })
    }
}

/// Handles all `ContentBlock`s other than `ContentBlock::Thought` (which needs special handling
/// to merge the signature with the next block).
async fn convert_non_thought_content_block(
    block: &ContentBlock,
) -> Result<FlattenUnknown<'_, GeminiPartData<'_>>, Error> {
    match block {
        ContentBlock::Text(Text { text }) => {
            Ok(FlattenUnknown::Normal(GeminiPartData::Text { text }))
        }
        ContentBlock::ToolResult(tool_result) => {
            // Gemini expects the format below according to [the documentation](https://ai.google.dev/gemini-api/docs/function-calling#multi-turn-example-1)
            let response = serde_json::json!({
                "name": tool_result.name,
                "content": tool_result.result,
            });
            Ok(FlattenUnknown::Normal(
                GeminiPartData::FunctionResponse {
                    function_response: GeminiFunctionResponse {
                        name: &tool_result.name,
                        response,
                    },
                },
            ))
        }
        ContentBlock::ToolCall(tool_call) => {
            // Convert the tool call arguments from String to JSON Value (Gemini expects an object)
            let args: Value = serde_json::from_str(&tool_call.arguments).map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: Some(StatusCode::BAD_REQUEST),
                    message: format!(
                        "Error parsing tool call arguments as JSON Value: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: Some(tool_call.arguments.clone()),
                })
            })?;

            if !args.is_object() {
                return Err(ErrorDetails::InferenceClient {
                    status_code: Some(StatusCode::BAD_REQUEST),
                    message: "Tool call arguments must be a JSON object".to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: Some(tool_call.arguments.clone()),
                }
                .into());
            }

            Ok(FlattenUnknown::Normal(GeminiPartData::FunctionCall {
                function_call: GeminiFunctionCall {
                    name: &tool_call.name,
                    args,
                },
            }))
        }
        ContentBlock::File(file) => {
            let resolved_file = file.resolve().await?;
            let ObjectStorageFile { file, data } = &*resolved_file;
            if file.detail.is_some() {
                tracing::warn!(
                    "The image detail parameter is not supported by Google AI Studio Gemini. The `detail` field will be ignored."
                );
            }
            Ok(FlattenUnknown::Normal(GeminiPartData::InlineData {
                inline_data: GeminiInlineData {
                    mime_type: file.mime_type.to_string(),
                    data: data.to_string(),
                },
            }))
        }
        ContentBlock::Thought(_) => Err(Error::new(ErrorDetails::InternalError {
            message: format!("Got thought block in `convert_non_thought_content_block`. {IMPOSSIBLE_ERROR_MESSAGE}"),
        })),
        ContentBlock::Unknown(Unknown { data, .. }) => Ok(FlattenUnknown::Unknown(Cow::Borrowed(data))),
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct GeminiFunctionDeclaration<'a> {
    name: &'a str,
    description: &'a str,
    parameters: Value, // Should be a JSONSchema as a Value
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool<'a> {
    pub function_declarations: Vec<GeminiFunctionDeclaration<'a>>,
    // TODO (if needed): code_execution ([docs](https://ai.google.dev/api/caching#CodeExecution))
}

impl<'a> GeminiFunctionDeclaration<'a> {
    fn from_tool_config(tool: &'a FunctionToolConfig) -> Self {
        let mut parameters = tool.parameters().clone();
        if let Some(obj) = parameters.as_object_mut() {
            obj.remove("additionalProperties");
            obj.remove("$schema");
        }

        GeminiFunctionDeclaration {
            name: tool.name(),
            description: tool.description(),
            parameters,
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
enum GeminiFunctionCallingMode {
    Auto,
    Any,
    None,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFunctionCallingConfig<'a> {
    mode: GeminiFunctionCallingMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<&'a str>>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleAIStudioGeminiToolConfig<'a> {
    function_calling_config: GeminiFunctionCallingConfig<'a>,
}

impl<'a> GoogleAIStudioGeminiToolConfig<'a> {
    fn from_tool_config(tool_config: &'a ToolCallConfig) -> Self {
        match &tool_config.tool_choice {
            ToolChoice::None => GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::None,
                    allowed_function_names: None,
                },
            },
            ToolChoice::Auto => {
                let allowed_function_names = tool_config.allowed_tools.as_dynamic_allowed_tools();
                // If allowed_function_names is set, we need to use Any mode because
                // Gemini's Auto mode with allowed_function_names errors
                let mode = if allowed_function_names.is_some() {
                    GeminiFunctionCallingMode::Any
                } else {
                    GeminiFunctionCallingMode::Auto
                };
                GoogleAIStudioGeminiToolConfig {
                    function_calling_config: GeminiFunctionCallingConfig {
                        mode,
                        allowed_function_names,
                    },
                }
            }
            ToolChoice::Required => GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Any,
                    allowed_function_names: tool_config.allowed_tools.as_dynamic_allowed_tools(),
                },
            },
            ToolChoice::Specific(tool_name) => GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec![tool_name]),
                },
            },
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
enum GeminiResponseMimeType {
    #[serde(rename = "text/plain")]
    #[expect(dead_code)]
    TextPlain,
    #[serde(rename = "application/json")]
    ApplicationJson,
}

// TODO (if needed): add the other options [here](https://ai.google.dev/api/generate-content#v1beta.GenerationConfig)
#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiThinkingConfig {
    thinking_budget: i32,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Cow<'a, [String]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<GeminiResponseMimeType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<Value>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest<'a> {
    contents: Vec<GeminiContent<'a>>,
    tools: Option<Vec<GeminiTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<GoogleAIStudioGeminiToolConfig<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent<'a>>,
}

fn apply_inference_params(
    request: &mut GeminiRequest,
    inference_params: &ChatCompletionInferenceParamsV2,
) {
    let ChatCompletionInferenceParamsV2 {
        reasoning_effort,
        service_tier,
        thinking_budget_tokens,
        verbosity,
    } = inference_params;

    if reasoning_effort.is_some() {
        warn_inference_parameter_not_supported(
            PROVIDER_NAME,
            "reasoning_effort",
            Some("Tip: You might want to use `thinking_budget_tokens` for this provider."),
        );
    }

    if let Some(budget_tokens) = thinking_budget_tokens {
        if let Some(gen_config) = &mut request.generation_config {
            gen_config.thinking_config = Some(GeminiThinkingConfig {
                thinking_budget: *budget_tokens,
            });
        } else {
            request.generation_config = Some(GeminiGenerationConfig {
                stop_sequences: None,
                temperature: None,
                thinking_config: Some(GeminiThinkingConfig {
                    thinking_budget: *budget_tokens,
                }),
                top_p: None,
                presence_penalty: None,
                frequency_penalty: None,
                max_output_tokens: None,
                seed: None,
                response_mime_type: None,
                response_schema: None,
            });
        }
    }

    if service_tier.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "service_tier", None);
    }

    if verbosity.is_some() {
        warn_inference_parameter_not_supported(PROVIDER_NAME, "verbosity", None);
    }
}

impl<'a> GeminiRequest<'a> {
    pub async fn new(request: &'a ModelInferenceRequest<'a>) -> Result<Self, Error> {
        if request.messages.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message: "Google AI Studio Gemini requires at least one message".to_string(),
            }
            .into());
        }
        let system_instruction =
            request
                .system
                .as_ref()
                .map(|system_instruction| GeminiPartData::Text {
                    text: system_instruction,
                });
        let all_contents: Vec<GeminiContent> = try_join_all(
            request
                .messages
                .iter()
                .map(GeminiContent::from_request_message),
        )
        .await?;
        let contents: Vec<GeminiContent> = all_contents
            .into_iter()
            .filter(|m| !m.parts.is_empty())
            .collect();
        let (tools, tool_config) = prepare_tools(request)?;
        let (response_mime_type, response_schema) = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                match request.output_schema {
                    Some(output_schema) => (
                        Some(GeminiResponseMimeType::ApplicationJson),
                        Some(process_jsonschema_for_gcp_vertex_gemini(output_schema)),
                    ),
                    None => (Some(GeminiResponseMimeType::ApplicationJson), None),
                }
            }
            ModelInferenceRequestJsonMode::Off => (None, None),
        };
        let generation_config = Some(GeminiGenerationConfig {
            stop_sequences: request.borrow_stop_sequences(),
            temperature: request.temperature,
            thinking_config: None,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_output_tokens: request.max_tokens,
            seed: request.seed,
            response_mime_type,
            response_schema,
        });
        let mut gemini_request = GeminiRequest {
            contents,
            tools,
            tool_config,
            generation_config,
            system_instruction: system_instruction.map(|content| GeminiContent {
                role: GeminiRole::Model,
                parts: vec![GeminiContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(content),
                }],
            }),
        };

        apply_inference_params(&mut gemini_request, &request.inference_params_v2);

        Ok(gemini_request)
    }
}

fn prepare_tools<'a>(
    request: &'a ModelInferenceRequest<'a>,
) -> Result<
    (
        Option<Vec<GeminiTool<'a>>>,
        Option<GoogleAIStudioGeminiToolConfig<'a>>,
    ),
    Error,
> {
    match &request.tool_config {
        Some(tool_config) => {
            if !tool_config.any_tools_available() {
                return Ok((None, None));
            }
            let tools = Some(vec![GeminiTool {
                function_declarations: tool_config
                    .tools_available()?
                    .map(GeminiFunctionDeclaration::from_tool_config)
                    .collect(),
            }]);
            let tool_config_converted = Some(GoogleAIStudioGeminiToolConfig::from_tool_config(
                tool_config,
            ));
            Ok((tools, tool_config_converted))
        }
        None => Ok((None, None)),
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct GeminiResponseFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseContentPart {
    #[serde(default)]
    thought: bool,
    #[serde(default)]
    thought_signature: Option<String>,
    #[serde(flatten)]
    #[serde(default)]
    data: FlattenUnknown<'static, GeminiResponseContentPartData>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
enum GeminiResponseContentPartData {
    Text(String),
    // TODO (if needed): InlineData { inline_data: Blob },
    // TODO (if needed): FileData { file_data: FileData },
    FunctionCall(GeminiResponseFunctionCall),
    // TODO (if needed): FunctionResponse
    // TODO (if needed): VideoMetadata { video_metadata: VideoMetadata },
}

#[expect(clippy::too_many_arguments)]
fn content_part_to_tensorzero_chunk(
    part: GeminiResponseContentPart,
    last_tool_name: &mut Option<String>,
    last_tool_idx: &mut Option<u32>,
    last_thought_id: &mut u32,
    discard_unknown_chunks: bool,
    output: &mut Vec<ContentBlockChunk>,
    last_unknown_chunk_id: &mut u32,
    model_name: &str,
    provider_name: &str,
) -> Result<(), Error> {
    if part.thought {
        match part.data {
            FlattenUnknown::Normal(GeminiResponseContentPartData::Text(text)) => {
                *last_thought_id += 1;
                output.push(ContentBlockChunk::Thought(ThoughtChunk {
                    id: last_thought_id.to_string(),
                    text: Some(text),
                    signature: part.thought_signature,
                    summary_id: None,
                    summary_text: None,
                    provider_type: Some(PROVIDER_TYPE.to_string()),
                }));
            }
            // Handle 'thought/thoughtSignature' with no other fields
            FlattenUnknown::Unknown(obj)
                if obj.as_object().is_some_and(serde_json::Map::is_empty) =>
            {
                *last_thought_id += 1;
                output.push(ContentBlockChunk::Thought(ThoughtChunk {
                    id: last_thought_id.to_string(),
                    text: None,
                    signature: part.thought_signature,
                    summary_id: None,
                    summary_text: None,
                    provider_type: Some(PROVIDER_TYPE.to_string()),
                }));
            }
            _ => {
                return Err(Error::new(ErrorDetails::InferenceServer {
                        message:
                            format!(
                                "Thought part in Google AI Studio Gemini response must be a text block: {part:?}"
                            ),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: Some(serde_json::to_string(&part).unwrap_or_default()),
                    }));
            }
        }
        return Ok(());
    }

    // Google AI Studio can emit `thoughtSignature` attached to arbitrary parts (including function calls)
    // Their API expects us to pass back the part with the 'thoughtSignature' attached.
    // Since the TensorZero model only supports standalone thought blocks, we emit a Thought block
    // with just the signature, immediately following the original part.
    // When constructing the input, we merge these blocks with their predecessor
    if let Some(thought_signature) = part.thought_signature {
        // GCP doesn't have any concept of chunk ids. To make sure that our
        // `collect_chunks` code never tries to merge thought blocks, we assign
        // a fresh id to each 'thoughtSignature' that we see.
        *last_thought_id += 1;
        // Add a thought chunk to the output, then continue on to process 'part.data'
        output.push(ContentBlockChunk::Thought(ThoughtChunk {
            id: last_thought_id.to_string(),
            text: None,
            summary_id: None,
            summary_text: None,
            signature: Some(thought_signature),
            provider_type: Some(PROVIDER_TYPE.to_string()),
        }));
    }

    match part.data {
        FlattenUnknown::Normal(GeminiResponseContentPartData::Text(text)) => {
            output.push(ContentBlockChunk::Text(TextChunk {
                text,
                id: "0".to_string(),
            }));
        }
        FlattenUnknown::Normal(GeminiResponseContentPartData::FunctionCall(function_call)) => {
            let arguments = serialize_or_log(&function_call.args);
            let name = check_new_tool_call_name(function_call.name, last_tool_name);
            if name.is_some() {
                // If a name comes from check_new_tool_call_name, we need to increment the tool call index
                // because this is a new tool call.
                // This will be used as a new ID so we can differentiate between tool calls.
                let new_tool_idx = match last_tool_idx {
                    Some(idx) => *idx + 1,
                    None => 0,
                };
                *last_tool_idx = Some(new_tool_idx);
            }
            let id = match last_tool_idx {
                Some(idx) => idx.to_string(),
                None => return Err(Error::new(ErrorDetails::Inference {
                    message: "Tool call index is not set in Google AI Studio Gemini. This should never happen. Please file a bug report: https://github.com/tensorzero/tensorzero/discussions/categories/bug-reports".to_string(),
                })),
            };
            output.push(ContentBlockChunk::ToolCall(ToolCallChunk {
                raw_name: name,
                raw_arguments: arguments,
                id,
            }));
        }
        FlattenUnknown::Unknown(part) => {
            if discard_unknown_chunks {
                warn_discarded_unknown_chunk(PROVIDER_TYPE, &part.to_string());
                return Ok(());
            }
            output.push(ContentBlockChunk::Unknown(UnknownChunk {
                id: last_unknown_chunk_id.to_string(),
                data: part.into_owned(),
                model_name: Some(model_name.to_string()),
                provider_name: Some(provider_name.to_string()),
            }));
            *last_unknown_chunk_id += 1;
        }
    }
    Ok(())
}

fn convert_part_to_output(
    model_name: &str,
    provider_name: &str,
    part: GeminiResponseContentPart,
    output: &mut Vec<ContentBlockOutput>,
) -> Result<(), Error> {
    if part.thought {
        match part.data {
            FlattenUnknown::Normal(GeminiResponseContentPartData::Text(text)) => {
                output.push(ContentBlockOutput::Thought(Thought {
                    signature: part.thought_signature,
                    text: Some(text),
                    summary: None,
                    provider_type: Some(PROVIDER_TYPE.to_string()),
                }));
            }
            // Handle 'thought' with no other fields
            FlattenUnknown::Unknown(obj)
                if obj.as_object().is_some_and(serde_json::Map::is_empty) =>
            {
                output.push(ContentBlockOutput::Thought(Thought {
                    signature: part.thought_signature,
                    text: None,
                    summary: None,
                    provider_type: Some(PROVIDER_TYPE.to_string()),
                }));
            }
            _ => {
                output.push(ContentBlockOutput::Unknown(Unknown {
                    data: serde_json::to_value(part).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Error serializing thought part returned from GCP: {e}"
                            ),
                        })
                    })?,
                    model_name: Some(model_name.to_string()),
                    provider_name: Some(provider_name.to_string()),
                }));
            }
        }
        return Ok(());
    }

    // If we have a thought_signature but 'thought' is not set, then we emit a separate Thought block
    // containing just the signature. When we receive this as input, we'll attach the thought signature
    // to the GCP content part that we create for the following content block.
    // This is needed due to the fact that that TensorZero data model cannot represent thought signatures
    // attached to arbitrary content blocks.
    if let Some(thought_signature) = part.thought_signature {
        output.push(ContentBlockOutput::Thought(Thought {
            signature: Some(thought_signature),
            text: None,
            summary: None,
            provider_type: Some(PROVIDER_TYPE.to_string()),
        }));
    }
    match part.data {
        FlattenUnknown::Normal(GeminiResponseContentPartData::Text(text)) => {
            output.push(text.into());
        }
        FlattenUnknown::Normal(GeminiResponseContentPartData::FunctionCall(function_call)) => {
            output.push(ContentBlockOutput::ToolCall(ToolCall {
                name: function_call.name,
                arguments: serde_json::to_string(&function_call.args).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error serializing function call arguments returned from Gemini: {e}"
                        ),
                    })
                })?,
                // Gemini doesn't have the concept of tool call ID so we generate one for our bookkeeping
                id: Uuid::now_v7().to_string(),
            }));
        }
        FlattenUnknown::Unknown(part) => {
            output.push(ContentBlockOutput::Unknown(Unknown {
                data: part.into_owned(),
                model_name: Some(model_name.to_string()),
                provider_name: Some(provider_name.to_string()),
            }));
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
struct GeminiResponseContent {
    #[serde(default)]
    parts: Vec<GeminiResponseContentPart>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum GeminiFinishReason {
    FinishReasonUnspecified,
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Other,
    Blocklist,
    ProhibitedContent,
    #[serde(rename = "SPII")]
    Spii,
    MalformedFunctionCall,
    #[serde(other)]
    Unknown,
}

impl From<GeminiFinishReason> for FinishReason {
    fn from(finish_reason: GeminiFinishReason) -> Self {
        match finish_reason {
            GeminiFinishReason::Stop => FinishReason::Stop,
            GeminiFinishReason::MaxTokens => FinishReason::Length,
            GeminiFinishReason::Safety => FinishReason::ContentFilter,
            GeminiFinishReason::Recitation => FinishReason::ToolCall,
            GeminiFinishReason::Other => FinishReason::Unknown,
            GeminiFinishReason::Blocklist => FinishReason::ContentFilter,
            GeminiFinishReason::ProhibitedContent => FinishReason::ContentFilter,
            GeminiFinishReason::Spii => FinishReason::ContentFilter,
            GeminiFinishReason::MalformedFunctionCall => FinishReason::ToolCall,
            GeminiFinishReason::FinishReasonUnspecified => FinishReason::Unknown,
            GeminiFinishReason::Unknown => FinishReason::Unknown,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseCandidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<GeminiResponseContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<GeminiFinishReason>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: Option<u32>,
    // Gemini doesn't return output tokens in certain edge cases (e.g. generation blocked by safety settings)
    #[serde(skip_serializing_if = "Option::is_none")]
    candidates_token_count: Option<u32>,
}

impl From<GeminiUsageMetadata> for Usage {
    fn from(usage_metadata: GeminiUsageMetadata) -> Self {
        Usage {
            input_tokens: usage_metadata.prompt_token_count,
            output_tokens: usage_metadata.candidates_token_count,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<GeminiResponseCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

struct GeminiResponseWithMetadata<'a> {
    model_name: &'a str,
    provider_name: &'a str,
    response: GeminiResponse,
    raw_response: String,
    latency: Latency,
    raw_request: String,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<GeminiResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(response: GeminiResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let GeminiResponseWithMetadata {
            response,
            raw_response,
            latency,
            raw_request,
            generic_request,
            model_name,
            provider_name,
        } = response;

        // Google AI Studio Gemini response can contain multiple candidates and each of these can contain
        // multiple content parts. We will only use the first candidate but handle all parts of the response therein.
        let first_candidate = response.candidates.into_iter().next().ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "Google AI Studio Gemini response has no candidates".to_string(),
                raw_request: Some(raw_request.clone()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        // Gemini sometimes doesn't return content in the response (e.g. safety settings blocked the generation).
        let content: Vec<ContentBlockOutput> = match first_candidate.content {
            Some(content) => {
                let mut output = Vec::with_capacity(content.parts.len());
                for part in content.parts {
                    convert_part_to_output(model_name, provider_name, part, &mut output)?;
                }
                output
            }
            None => vec![],
        };

        let usage = response
            .usage_metadata
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Google AI Studio Gemini non-streaming response has no usage metadata"
                        .to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?
            .into();
        let system = generic_request.system.clone();
        let messages = generic_request.messages.clone();
        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages: messages,
                raw_request,
                raw_response: raw_response.clone(),
                usage,
                latency,
                finish_reason: first_candidate.finish_reason.map(Into::into),
            },
        ))
    }
}

struct ConvertStreamResponseArgs<'a> {
    raw_response: String,
    response: GeminiResponse,
    latency: Duration,
    last_tool_name: &'a mut Option<String>,
    last_tool_idx: &'a mut Option<u32>,
    last_thought_id: &'a mut u32,
    last_unknown_chunk_id: &'a mut u32,
    discard_unknown_chunks: bool,
    model_name: &'a str,
    provider_name: &'a str,
}

fn convert_stream_response_with_metadata_to_chunk(
    args: ConvertStreamResponseArgs,
) -> Result<ProviderInferenceResponseChunk, Error> {
    let ConvertStreamResponseArgs {
        raw_response,
        response,
        latency,
        last_tool_name,
        last_tool_idx,
        last_thought_id,
        last_unknown_chunk_id,
        discard_unknown_chunks,
        model_name,
        provider_name,
    } = args;
    let first_candidate = response.candidates.into_iter().next().ok_or_else(|| {
        Error::new(ErrorDetails::InferenceServer {
            message: "Google AI Studio Gemini response has no candidates".to_string(),
            raw_request: None,
            raw_response: Some(raw_response.clone()),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;

    // Gemini sometimes returns chunks without content (e.g. they might have usage only).
    let mut content: Vec<ContentBlockChunk> = match first_candidate.content {
        Some(content) => {
            let mut output = Vec::with_capacity(content.parts.len());
            for part in content.parts {
                content_part_to_tensorzero_chunk(
                    part,
                    last_tool_name,
                    last_tool_idx,
                    last_thought_id,
                    discard_unknown_chunks,
                    &mut output,
                    last_unknown_chunk_id,
                    model_name,
                    provider_name,
                )?;
            }
            output
        }
        None => vec![],
    };

    // Gemini occasionally spuriously returns empty text chunks. We filter these out.
    content.retain(|chunk| match chunk {
        ContentBlockChunk::Text(text) => !text.text.is_empty(),
        _ => true,
    });
    // Google AI Studio returns the running usage metadata in each chunk.
    // We only want to return the final usage metadata once the stream has ended.
    // So, we clear the usage metadata if the finish reason is not set.
    let usage = if first_candidate.finish_reason.as_ref().is_none() {
        None
    } else {
        response.usage_metadata.map(Into::into)
    };
    Ok(ProviderInferenceResponseChunk::new(
        content,
        usage,
        raw_response,
        latency,
        first_candidate.finish_reason.map(Into::into),
    ))
}

fn handle_google_ai_studio_error(
    response_code: StatusCode,
    response_body: String,
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED
        | StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::TOO_MANY_REQUESTS => Err(ErrorDetails::InferenceClient {
            status_code: Some(response_code),
            message: response_body.clone(),
            raw_request: None,
            raw_response: Some(response_body.clone()),
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into()),
        // StatusCode::NOT_FOUND | StatusCode::FORBIDDEN | StatusCode::INTERNAL_SERVER_ERROR | 529: Overloaded
        // These are all captured in _ since they have the same error behavior
        _ => Err(ErrorDetails::InferenceServer {
            message: response_body.clone(),
            raw_request: None,
            raw_response: Some(response_body.clone()),
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
    use base64::Engine;
    use serde_json::json;

    use super::*;
    use crate::inference::types::file::Detail;
    use crate::inference::types::resolved_input::LazyFile;
    use crate::inference::types::storage::{StorageKind, StoragePath};
    use crate::inference::types::{
        ContentBlock, FlattenUnknown, FunctionType, ModelInferenceRequestJsonMode,
        ObjectStorageFile, ObjectStoragePointer, PendingObjectStoreFile,
    };
    use crate::providers::test_helpers::{MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL};
    use crate::tool::{ToolCallConfig, ToolResult};
    use crate::utils::testing::capture_logs;

    #[test]
    fn test_convert_unknown_content_block_warn() {
        let logs_contain = capture_logs();
        use std::time::Duration;
        let content = GeminiResponseContent {
            parts: vec![GeminiResponseContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Unknown(Cow::Owned(
                    json!({"unknown_field": "unknown_value"}),
                )),
            }],
        };

        let response = GeminiResponse {
            candidates: vec![GeminiResponseCandidate {
                content: Some(content),
                finish_reason: Some(GeminiFinishReason::Stop),
            }],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
            }),
        };

        let latency = Duration::from_millis(100);
        let mut last_tool_name = None;

        let mut last_tool_idx = None;
        let mut last_thought_id = 0;
        let mut last_unknown_chunk_id = 0;
        let res = convert_stream_response_with_metadata_to_chunk(ConvertStreamResponseArgs {
            raw_response: "raw_response".to_string(),
            response,
            latency,
            last_tool_name: &mut last_tool_name,
            last_tool_idx: &mut last_tool_idx,
            last_thought_id: &mut last_thought_id,
            last_unknown_chunk_id: &mut last_unknown_chunk_id,
            discard_unknown_chunks: true,
            model_name: "test_model",
            provider_name: "test_provider",
        })
        .unwrap();
        assert_eq!(res.content, []);
        assert!(
            logs_contain("Discarding unknown chunk in google_ai_studio_gemini response"),
            "Missing warning in logs"
        );
    }

    #[tokio::test]
    async fn test_google_ai_studio_gemini_content_try_from() {
        let message = RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into()],
        };
        let content = GeminiContent::from_request_message(&message).await.unwrap();
        assert_eq!(content.role, GeminiRole::User);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::Text {
                    text: "Hello, world!"
                }),
            }
        );

        let message = RequestMessage {
            role: Role::Assistant,
            content: vec!["Hello, world!".to_string().into()],
        };
        let content = GeminiContent::from_request_message(&message).await.unwrap();
        assert_eq!(content.role, GeminiRole::Model);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::Text {
                    text: "Hello, world!"
                }),
            }
        );
        let message = RequestMessage {
            role: Role::Assistant,
            content: vec![
                "Here's the result of the function call:".to_string().into(),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "get_temperature".to_string(),
                    arguments: r#"{"location": "New York", "unit": "celsius"}"#.to_string(),
                }),
            ],
        };
        let content = GeminiContent::from_request_message(&message).await.unwrap();
        assert_eq!(content.role, GeminiRole::Model);
        assert_eq!(content.parts.len(), 2);
        assert_eq!(
            content.parts[0],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::Text {
                    text: "Here's the result of the function call:",
                }),
            }
        );
        assert_eq!(
            content.parts[1],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::FunctionCall {
                    function_call: GeminiFunctionCall {
                        name: "get_temperature",
                        args: json!({"location": "New York", "unit": "celsius"}),
                    }
                }),
            }
        );

        let message = RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "call_1".to_string(),
                name: "get_temperature".to_string(),
                result: r#"{"temperature": 25, "conditions": "sunny"}"#.to_string(),
            })],
        };
        let content = GeminiContent::from_request_message(&message).await.unwrap();
        assert_eq!(content.role, GeminiRole::User);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::FunctionResponse {
                    function_response: GeminiFunctionResponse {
                        name: "get_temperature",
                        response: json!({
                            "name": "get_temperature",
                            "content": r#"{"temperature": 25, "conditions": "sunny"}"#
                        }),
                    }
                }),
            }
        );
    }

    #[test]
    fn test_from_vec_tool() {
        let tools_vec: Vec<&FunctionToolConfig> =
            MULTI_TOOL_CONFIG.tools_available().unwrap().collect();
        let tool = GeminiTool {
            function_declarations: tools_vec
                .iter()
                .map(|&t| GeminiFunctionDeclaration::from_tool_config(t))
                .collect(),
        };
        assert_eq!(
            tool,
            GeminiTool {
                function_declarations: vec![
                    GeminiFunctionDeclaration {
                        name: "get_temperature",
                        description: "Get the current temperature in a given location",
                        parameters: tools_vec[0].parameters().clone(),
                    },
                    GeminiFunctionDeclaration {
                        name: "query_articles",
                        description: "Query articles from Wikipedia",
                        parameters: tools_vec[1].parameters().clone(),
                    }
                ]
            }
        );
    }

    #[test]
    fn test_from_tool_config() {
        // Test Auto mode
        let tool_call_config = ToolCallConfig {
            static_tools_available: vec![],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(),
        };
        let tool_config = GoogleAIStudioGeminiToolConfig::from_tool_config(&tool_call_config);
        assert_eq!(
            tool_config,
            GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                }
            }
        );

        let tool_call_config = ToolCallConfig {
            static_tools_available: vec![],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(),
        };
        let tool_config = GoogleAIStudioGeminiToolConfig::from_tool_config(&tool_call_config);
        assert_eq!(
            tool_config,
            GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Any,
                    allowed_function_names: None,
                }
            }
        );

        let tool_call_config = ToolCallConfig {
            static_tools_available: vec![],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Specific("get_temperature".to_string()),
            parallel_tool_calls: None,
            allowed_tools: AllowedTools {
                tools: vec!["get_temperature".to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
        };
        let tool_config = GoogleAIStudioGeminiToolConfig::from_tool_config(&tool_call_config);
        assert_eq!(
            tool_config,
            GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec!["get_temperature"]),
                }
            }
        );

        // Test Auto mode with specific allowed tools - should use Any mode
        let tool_call_config = ToolCallConfig {
            static_tools_available: vec![],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools {
                tools: vec!["tool1".to_string(), "tool2".to_string()]
                    .into_iter()
                    .collect(),
                choice: AllowedToolsChoice::Explicit,
            },
        };
        let tool_config = GoogleAIStudioGeminiToolConfig::from_tool_config(&tool_call_config);
        assert_eq!(
            tool_config.function_calling_config.mode,
            GeminiFunctionCallingMode::Any
        );
        let mut allowed_names = tool_config
            .function_calling_config
            .allowed_function_names
            .unwrap();
        allowed_names.sort();
        assert_eq!(allowed_names, vec!["tool1", "tool2"]);

        // Test Required mode with specific allowed tools (new behavior)
        let tool_call_config = ToolCallConfig {
            static_tools_available: vec![],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools {
                tools: vec!["allowed_tool".to_string()].into_iter().collect(),
                choice: AllowedToolsChoice::Explicit,
            },
        };
        let tool_config = GoogleAIStudioGeminiToolConfig::from_tool_config(&tool_call_config);
        assert_eq!(
            tool_config,
            GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec!["allowed_tool"]),
                }
            }
        );

        let tool_call_config = ToolCallConfig {
            static_tools_available: vec![],
            dynamic_tools_available: vec![],
            provider_tools: vec![],
            openai_custom_tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            allowed_tools: AllowedTools::default(),
        };
        let tool_config = GoogleAIStudioGeminiToolConfig::from_tool_config(&tool_call_config);
        assert_eq!(
            tool_config,
            GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::None,
                    allowed_function_names: None,
                }
            }
        );
    }

    #[tokio::test]
    async fn test_google_ai_studio_gemini_request_try_from() {
        // Test Case 1: Empty message list
        let tool_config = ToolCallConfig::default();
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![],
            system: None,
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let result = GeminiRequest::new(&inference_request).await;
        let error = result.unwrap_err();
        let details = error.get_details();
        assert_eq!(
            *details,
            ErrorDetails::InvalidRequest {
                message: "Google AI Studio Gemini requires at least one message".to_string()
            }
        );

        // Test Case 2: Messages with System instructions
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
        ];
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let result = GeminiRequest::new(&inference_request).await;
        let request = result.unwrap();
        assert_eq!(request.contents.len(), 2);
        assert_eq!(request.contents[0].role, GeminiRole::User);
        assert_eq!(
            request.contents[0].parts[0],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::Text { text: "test_user" }),
            }
        );
        assert_eq!(request.contents[1].role, GeminiRole::Model);
        assert_eq!(request.contents[1].parts.len(), 1);
        assert_eq!(
            request.contents[1].parts[0],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::Text {
                    text: "test_assistant"
                }),
            }
        );

        // Test case 3: Messages with system message and some of the optional fields are tested
        let messages = vec![
            RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            },
            RequestMessage {
                role: Role::User,
                content: vec!["test_user2".to_string().into()],
            },
            RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            },
        ];
        let output_schema = serde_json::json!({});
        let inference_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tool_config: Some(Cow::Borrowed(&tool_config)),
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stream: true,
            json_mode: ModelInferenceRequestJsonMode::On,
            function_type: FunctionType::Chat,
            output_schema: Some(&output_schema),
            extra_body: Default::default(),
            ..Default::default()
        };
        // JSON schema should be supported for Gemini Pro models
        let result = GeminiRequest::new(&inference_request).await;
        let request = result.unwrap();
        assert_eq!(request.contents.len(), 3);
        assert_eq!(request.contents[0].role, GeminiRole::User);
        assert_eq!(request.contents[1].role, GeminiRole::User);
        assert_eq!(request.contents[2].role, GeminiRole::Model);
        assert_eq!(request.contents[0].parts.len(), 1);
        assert_eq!(request.contents[1].parts.len(), 1);
        assert_eq!(request.contents[2].parts.len(), 1);
        assert_eq!(
            request.contents[0].parts[0],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::Text { text: "test_user" }),
            }
        );
        assert_eq!(
            request.contents[1].parts[0],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::Text { text: "test_user2" }),
            }
        );
        assert_eq!(
            request.contents[2].parts[0],
            GeminiContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(GeminiPartData::Text {
                    text: "test_assistant"
                }),
            }
        );
        assert_eq!(
            request.generation_config.as_ref().unwrap().temperature,
            Some(0.5)
        );
        assert_eq!(request.generation_config.as_ref().unwrap().top_p, Some(0.9));
        assert_eq!(
            request.generation_config.as_ref().unwrap().presence_penalty,
            Some(0.1)
        );
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .frequency_penalty,
            Some(0.1)
        );
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .max_output_tokens,
            Some(100)
        );
        assert_eq!(request.generation_config.as_ref().unwrap().seed, Some(69));
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .response_mime_type,
            Some(GeminiResponseMimeType::ApplicationJson)
        );
        assert_eq!(
            request.generation_config.as_ref().unwrap().response_schema,
            Some(output_schema.clone())
        );
    }

    #[test]
    fn test_google_ai_studio_gemini_to_t0_response() {
        let part = GeminiResponseContentPartData::Text("test_assistant".to_string());
        let content = GeminiResponseContent {
            parts: vec![GeminiResponseContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(part),
            }],
        };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
            finish_reason: Some(GeminiFinishReason::Stop),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(10),
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(1),
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = GeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let raw_response = "test response".to_string();
        let response_with_latency = GeminiResponseWithMetadata {
            model_name: "test_model",
            provider_name: "test_provider",
            response,
            latency: latency.clone(),
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        };
        let model_inference_response: ProviderInferenceResponse =
            response_with_latency.try_into().unwrap();
        assert_eq!(
            model_inference_response.output,
            vec!["test_assistant".to_string().into()]
        );
        assert_eq!(
            model_inference_response.usage,
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(10),
            }
        );
        assert_eq!(model_inference_response.latency, latency);
        assert_eq!(model_inference_response.raw_request, raw_request);
        assert_eq!(model_inference_response.raw_response, raw_response);
        assert_eq!(
            model_inference_response.finish_reason,
            Some(FinishReason::Stop)
        );
        assert_eq!(model_inference_response.system, None);
        assert_eq!(
            model_inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );
        let text_part =
            GeminiResponseContentPartData::Text("Here's the weather information:".to_string());
        let function_call_part =
            GeminiResponseContentPartData::FunctionCall(GeminiResponseFunctionCall {
                name: "get_temperature".to_string(),
                args: json!({"location": "New York", "unit": "celsius"}),
            });
        let content = GeminiResponseContent {
            parts: vec![
                GeminiResponseContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(text_part),
                },
                GeminiResponseContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(function_call_part),
                },
            ],
        };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
            finish_reason: Some(GeminiFinishReason::Stop),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(15),
                candidates_token_count: Some(20),
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(2),
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }],
            system: Some("test_system".to_string()),
            tool_config: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let request_body = GeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let response_with_latency = GeminiResponseWithMetadata {
            model_name: "test_model",
            provider_name: "test_provider",
            response,
            latency: latency.clone(),
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        };
        let model_inference_response: ProviderInferenceResponse =
            response_with_latency.try_into().unwrap();

        if let [ContentBlockOutput::Text(Text { text }), ContentBlockOutput::ToolCall(tool_call)] =
            &model_inference_response.output[..]
        {
            assert_eq!(text, "Here's the weather information:");
            assert_eq!(tool_call.name, "get_temperature");
            assert_eq!(
                tool_call.arguments,
                r#"{"location":"New York","unit":"celsius"}"#
            );
        } else {
            panic!("Expected a text and tool call content block");
        }

        assert_eq!(
            model_inference_response.usage,
            Usage {
                input_tokens: Some(15),
                output_tokens: Some(20),
            }
        );
        assert_eq!(model_inference_response.latency, latency);
        assert_eq!(
            model_inference_response.finish_reason,
            Some(FinishReason::Stop)
        );
        assert_eq!(model_inference_response.raw_request, raw_request);
        assert_eq!(
            model_inference_response.system,
            Some("test_system".to_string())
        );
        assert_eq!(
            model_inference_response.input_messages,
            vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]
        );

        let text_part1 =
            GeminiResponseContentPartData::Text("Here's the weather information:".to_string());
        let function_call_part =
            GeminiResponseContentPartData::FunctionCall(GeminiResponseFunctionCall {
                name: "get_temperature".to_string(),
                args: json!({"location": "New York", "unit": "celsius"}),
            });
        let text_part2 = GeminiResponseContentPartData::Text(
            "And here's a restaurant recommendation:".to_string(),
        );
        let function_call_part2 =
            GeminiResponseContentPartData::FunctionCall(GeminiResponseFunctionCall {
                name: "get_restaurant".to_string(),
                args: json!({"cuisine": "Italian", "price_range": "moderate"}),
            });
        let content = GeminiResponseContent {
            parts: vec![
                GeminiResponseContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(text_part1),
                },
                GeminiResponseContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(function_call_part),
                },
                GeminiResponseContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(text_part2),
                },
                GeminiResponseContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(function_call_part2),
                },
            ],
        };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
            finish_reason: Some(GeminiFinishReason::Stop),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(25),
                candidates_token_count: Some(40),
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(3),
        };
        let request_body = GeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let response_with_latency = GeminiResponseWithMetadata {
            model_name: "test_model",
            provider_name: "test_provider",
            response,
            latency: latency.clone(),
            raw_request: raw_request.clone(),
            generic_request: &generic_request,
            raw_response: raw_response.clone(),
        };
        let model_inference_response: ProviderInferenceResponse =
            response_with_latency.try_into().unwrap();
        assert_eq!(model_inference_response.raw_request, raw_request);

        assert_eq!(model_inference_response.raw_response, raw_response);
        if let [ContentBlockOutput::Text(Text { text: text1 }), ContentBlockOutput::ToolCall(tool_call1), ContentBlockOutput::Text(Text { text: text2 }), ContentBlockOutput::ToolCall(tool_call2)] =
            &model_inference_response.output[..]
        {
            assert_eq!(text1, "Here's the weather information:");
            assert_eq!(text2, "And here's a restaurant recommendation:");
            assert_eq!(tool_call1.name, "get_temperature");
            assert_eq!(
                tool_call1.arguments,
                r#"{"location":"New York","unit":"celsius"}"#
            );
            assert_eq!(tool_call2.name, "get_restaurant");
            assert_eq!(
                tool_call2.arguments,
                r#"{"cuisine":"Italian","price_range":"moderate"}"#
            );
        } else {
            panic!(
                "Content does not match expected structure: {:?}",
                model_inference_response.output
            );
        }

        assert_eq!(
            model_inference_response.usage,
            Usage {
                input_tokens: Some(25),
                output_tokens: Some(40),
            }
        );
        assert_eq!(model_inference_response.latency, latency);
        assert_eq!(
            model_inference_response.system,
            Some("test_system".to_string())
        );
        assert_eq!(
            model_inference_response.input_messages,
            vec![RequestMessage {
                role: Role::Assistant,
                content: vec!["test_assistant".to_string().into()],
            }]
        );
    }

    #[test]
    fn test_prepare_tools() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&MULTI_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let (tools, tool_choice) = prepare_tools(&request_with_tools).unwrap();
        let tools = tools.unwrap();
        let tool_config = tool_choice.unwrap();
        assert_eq!(
            tool_config.function_calling_config.mode,
            GeminiFunctionCallingMode::Any,
        );
        assert_eq!(tools.len(), 1);
        let GeminiTool {
            function_declarations,
        } = &tools[0];
        assert_eq!(function_declarations.len(), 2);
        assert_eq!(function_declarations[0].name, WEATHER_TOOL.name());
        assert_eq!(
            function_declarations[0].parameters,
            WEATHER_TOOL.parameters().clone()
        );
        assert_eq!(function_declarations[1].name, QUERY_TOOL.name());
        assert_eq!(
            function_declarations[1].parameters,
            QUERY_TOOL.parameters().clone()
        );
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stream: false,
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&MULTI_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let (tools, tool_choice) = prepare_tools(&request_with_tools).unwrap();
        let tools = tools.unwrap();
        let tool_config = tool_choice.unwrap();
        // Flash models do not support function calling mode Any
        assert_eq!(
            tool_config.function_calling_config.mode,
            // GeminiFunctionCallingMode::Auto,
            GeminiFunctionCallingMode::Any,
        );
        assert_eq!(tools.len(), 1);
        let GeminiTool {
            function_declarations,
        } = &tools[0];
        assert_eq!(function_declarations.len(), 2);
        assert_eq!(function_declarations[0].name, WEATHER_TOOL.name());
        assert_eq!(
            function_declarations[0].parameters,
            WEATHER_TOOL.parameters().clone()
        );
        assert_eq!(function_declarations[1].name, QUERY_TOOL.name());
        assert_eq!(
            function_declarations[1].parameters,
            QUERY_TOOL.parameters().clone()
        );
    }

    #[test]
    fn test_process_jsonschema_for_gcp_vertex_gemini() {
        let output_schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string", "format": "email"}
            }
        });
        let processed_schema = process_jsonschema_for_gcp_vertex_gemini(&output_schema);
        assert_eq!(processed_schema, output_schema);

        // Test with a schema that includes additionalProperties
        let output_schema_with_additional = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "additionalProperties": true
        });
        let output_schema_without_additional = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
        });
        let processed_schema_with_additional =
            process_jsonschema_for_gcp_vertex_gemini(&output_schema_with_additional);
        assert_eq!(
            processed_schema_with_additional,
            output_schema_without_additional
        );

        // Test with a schema that explicitly disallows additional properties
        let output_schema_no_additional = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "additionalProperties": false
        });
        let processed_schema_no_additional =
            process_jsonschema_for_gcp_vertex_gemini(&output_schema_no_additional);
        assert_eq!(
            processed_schema_no_additional,
            output_schema_without_additional
        );
        // Test with a schema that includes recursive additionalProperties
        let output_schema_recursive = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "children": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer", "minimum": 0}
                        },
                        "additionalProperties": {
                            "$ref": "#"
                        }
                    }
                }
            },
            "additionalProperties": {
                "$ref": "#"
            }
        });
        let expected_processed_schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "children": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer", "minimum": 0}
                        }
                    }
                }
            }
        });
        let processed_schema_recursive =
            process_jsonschema_for_gcp_vertex_gemini(&output_schema_recursive);
        assert_eq!(processed_schema_recursive, expected_processed_schema);
    }

    #[test]
    fn test_credential_to_google_ai_studio_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = GoogleAIStudioCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, GoogleAIStudioCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = GoogleAIStudioCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, GoogleAIStudioCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = GoogleAIStudioCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, GoogleAIStudioCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = GoogleAIStudioCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_try_from_with_content_and_finish_reason() {
        // Setup a response with content and finish reason
        let text_part = GeminiResponseContentPartData::Text("Hello, world!".to_string());
        let content = GeminiResponseContent {
            parts: vec![GeminiResponseContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(text_part),
            }],
        };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
            finish_reason: Some(GeminiFinishReason::Stop),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(20),
            }),
        };

        // Convert to ProviderInferenceResponseChunk
        let mut last_tool_name = None;
        let mut last_tool_idx = None;
        let mut last_thought_id = 0;
        let mut last_unknown_chunk_id = 0;
        let chunk: ProviderInferenceResponseChunk =
            convert_stream_response_with_metadata_to_chunk(ConvertStreamResponseArgs {
                raw_response: "my_raw_chunk".to_string(),
                response,
                latency: Duration::from_millis(100),
                last_tool_name: &mut last_tool_name,
                last_tool_idx: &mut last_tool_idx,
                last_thought_id: &mut last_thought_id,
                last_unknown_chunk_id: &mut last_unknown_chunk_id,
                discard_unknown_chunks: false,
                model_name: "test_model",
                provider_name: "test_provider",
            })
            .unwrap();

        // Verify tool call tracking state - should remain None for text chunks
        assert_eq!(last_tool_idx, None);

        // Verify content
        assert_eq!(chunk.content.len(), 1);
        if let ContentBlockChunk::Text(text) = &chunk.content[0] {
            assert_eq!(text.text, "Hello, world!");
            assert_eq!(text.id, "0");
        } else {
            panic!("Expected text content");
        }

        // Verify usage is included when finish_reason is set
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(20));

        // Verify finish reason
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_try_from_without_finish_reason() {
        // Setup a response without finish reason (streaming chunk)
        let text_part = GeminiResponseContentPartData::Text("Partial response".to_string());
        let content = GeminiResponseContent {
            parts: vec![GeminiResponseContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(text_part),
            }],
        };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
            finish_reason: None, // No finish reason for streaming chunks
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(15),
            }),
        };

        // Convert to ProviderInferenceResponseChunk
        let mut last_tool_name = None;
        let mut last_tool_idx = None;
        let mut last_thought_id = 0;
        let mut last_unknown_chunk_id = 0;
        let chunk: ProviderInferenceResponseChunk =
            convert_stream_response_with_metadata_to_chunk(ConvertStreamResponseArgs {
                raw_response: "my_raw_chunk".to_string(),
                response,
                latency: Duration::from_millis(50),
                last_tool_name: &mut last_tool_name,
                last_tool_idx: &mut last_tool_idx,
                last_thought_id: &mut last_thought_id,
                last_unknown_chunk_id: &mut last_unknown_chunk_id,
                discard_unknown_chunks: false,
                model_name: "test_model",
                provider_name: "test_provider",
            })
            .unwrap();

        // Verify tool call tracking state - should remain None for text chunks
        assert_eq!(last_tool_idx, None);

        // Verify content
        assert_eq!(chunk.content.len(), 1);
        if let ContentBlockChunk::Text(text) = &chunk.content[0] {
            assert_eq!(text.text, "Partial response");
        } else {
            panic!("Expected text content");
        }

        // Verify usage is None when finish_reason is not set
        assert!(chunk.usage.is_none());

        // Verify finish reason is None
        assert_eq!(chunk.finish_reason, None);
    }

    #[test]
    fn test_try_from_with_empty_text_chunks() {
        // Setup a response with empty text chunks that should be filtered out
        let empty_text = GeminiResponseContentPartData::Text(String::new());
        let non_empty_text = GeminiResponseContentPartData::Text("Non-empty text".to_string());
        let content = GeminiResponseContent {
            parts: vec![
                GeminiResponseContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(empty_text),
                },
                GeminiResponseContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(non_empty_text),
                },
            ],
        };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
            finish_reason: Some(GeminiFinishReason::Stop),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(5),
                candidates_token_count: Some(3),
            }),
        };

        // Convert to ProviderInferenceResponseChunk
        let mut last_tool_name = None;
        let mut last_tool_idx = None;
        let mut last_thought_id = 0;
        let mut last_unknown_chunk_id = 0;
        let chunk: ProviderInferenceResponseChunk =
            convert_stream_response_with_metadata_to_chunk(ConvertStreamResponseArgs {
                raw_response: "my_raw_chunk".to_string(),
                response,
                latency: Duration::from_millis(75),
                last_tool_name: &mut last_tool_name,
                last_tool_idx: &mut last_tool_idx,
                last_thought_id: &mut last_thought_id,
                last_unknown_chunk_id: &mut last_unknown_chunk_id,
                discard_unknown_chunks: false,
                model_name: "test_model",
                provider_name: "test_provider",
            })
            .unwrap();

        // Verify tool call tracking state - should remain None for text chunks
        assert_eq!(last_tool_idx, None);

        // Verify empty text chunks are filtered out
        assert_eq!(chunk.content.len(), 1);
        if let ContentBlockChunk::Text(text) = &chunk.content[0] {
            assert_eq!(text.text, "Non-empty text");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_try_from_with_function_call() {
        // Setup a response with a function call
        let function_call =
            GeminiResponseContentPartData::FunctionCall(GeminiResponseFunctionCall {
                name: "get_weather".to_string(),
                args: json!({"location": "New York", "unit": "celsius"}),
            });
        let content = GeminiResponseContent {
            parts: vec![GeminiResponseContentPart {
                thought: false,
                thought_signature: None,
                data: FlattenUnknown::Normal(function_call),
            }],
        };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
            finish_reason: Some(GeminiFinishReason::Recitation),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(15),
                candidates_token_count: Some(10),
            }),
        };

        // Convert to ProviderInferenceResponseChunk
        let mut last_tool_name = None;
        let mut last_tool_idx = None;
        let mut last_thought_id = 0;
        let mut last_unknown_chunk_id = 0;
        let chunk: ProviderInferenceResponseChunk =
            convert_stream_response_with_metadata_to_chunk(ConvertStreamResponseArgs {
                raw_response: "my_raw_chunk".to_string(),
                response,
                latency: Duration::from_millis(120),
                last_tool_name: &mut last_tool_name,
                last_tool_idx: &mut last_tool_idx,
                last_thought_id: &mut last_thought_id,
                last_unknown_chunk_id: &mut last_unknown_chunk_id,
                discard_unknown_chunks: false,
                model_name: "test_model",
                provider_name: "test_provider",
            })
            .unwrap();

        // Verify tool call tracking state - should be Some(0) for first tool call
        assert_eq!(last_tool_idx, Some(0));
        assert_eq!(last_tool_name, Some("get_weather".to_string()));

        // Verify function call content
        assert_eq!(chunk.content.len(), 1);
        if let ContentBlockChunk::ToolCall(tool_call) = &chunk.content[0] {
            assert_eq!(tool_call.raw_name, Some("get_weather".to_string()));
            assert_eq!(tool_call.id, "0");
            // Check that arguments were serialized correctly
            let args: serde_json::Value = serde_json::from_str(&tool_call.raw_arguments).unwrap();
            assert_eq!(args["location"], "New York");
            assert_eq!(args["unit"], "celsius");
        } else {
            panic!("Expected tool call content");
        }

        // Verify finish reason for tool calls
        assert_eq!(chunk.finish_reason, Some(FinishReason::ToolCall));
    }

    #[test]
    fn test_try_from_without_content() {
        // Setup a response without content (e.g., blocked by safety settings)
        let candidate = GeminiResponseCandidate {
            content: None,
            finish_reason: Some(GeminiFinishReason::Safety),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(8),
                candidates_token_count: None, // No output tokens when blocked
            }),
        };

        // Convert to ProviderInferenceResponseChunk
        let mut last_tool_name = None;
        let mut last_tool_idx = None;
        let mut last_thought_id = 0;
        let mut last_unknown_chunk_id = 0;
        let chunk: ProviderInferenceResponseChunk =
            convert_stream_response_with_metadata_to_chunk(ConvertStreamResponseArgs {
                raw_response: "my_raw_chunk".to_string(),
                response,
                latency: Duration::from_millis(60),
                last_tool_name: &mut last_tool_name,
                last_tool_idx: &mut last_tool_idx,
                last_thought_id: &mut last_thought_id,
                last_unknown_chunk_id: &mut last_unknown_chunk_id,
                discard_unknown_chunks: false,
                model_name: "test_model",
                provider_name: "test_provider",
            })
            .unwrap();

        // Verify tool call tracking state - should remain None for responses without content
        assert_eq!(last_tool_idx, None);

        // Verify empty content
        assert_eq!(chunk.content.len(), 0);

        // Verify usage is included (with zero output tokens)
        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(8));
        assert_eq!(usage.output_tokens, None);

        // Verify finish reason for safety blocks
        assert_eq!(chunk.finish_reason, Some(FinishReason::ContentFilter));
    }

    #[test]
    fn test_try_from_with_no_candidates() {
        // Setup a response with no candidates
        let response = GeminiResponse {
            candidates: vec![],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: Some(5),
                candidates_token_count: Some(0),
            }),
        };

        // Convert to ProviderInferenceResponseChunk
        let mut last_tool_name = None;
        let mut last_tool_idx = None;
        let mut last_thought_id = 0;
        let mut last_unknown_chunk_id = 0;
        let result = convert_stream_response_with_metadata_to_chunk(ConvertStreamResponseArgs {
            raw_response: "my_raw_chunk".to_string(),
            response,
            latency: Duration::from_millis(30),
            last_tool_name: &mut last_tool_name,
            last_tool_idx: &mut last_tool_idx,
            last_thought_id: &mut last_thought_id,
            last_unknown_chunk_id: &mut last_unknown_chunk_id,
            discard_unknown_chunks: false,
            model_name: "test_model",
            provider_name: "test_provider",
        });

        // Should remain None when there's an error
        assert_eq!(last_tool_idx, None);

        // Verify error is returned
        assert!(result.is_err());
        let error = result.unwrap_err();
        let details = error.get_details();
        if let ErrorDetails::InferenceServer { message, .. } = details {
            assert!(message.contains("no candidates"));
        } else {
            panic!("Expected InferenceServer error");
        }
    }

    #[test]
    fn test_try_from_with_various_finish_reasons() {
        // Test different finish reasons and their mappings
        let finish_reasons = vec![
            (GeminiFinishReason::Stop, FinishReason::Stop),
            (GeminiFinishReason::MaxTokens, FinishReason::Length),
            (GeminiFinishReason::Safety, FinishReason::ContentFilter),
            (GeminiFinishReason::Recitation, FinishReason::ToolCall),
            (GeminiFinishReason::Other, FinishReason::Unknown),
            (GeminiFinishReason::Blocklist, FinishReason::ContentFilter),
            (
                GeminiFinishReason::ProhibitedContent,
                FinishReason::ContentFilter,
            ),
            (GeminiFinishReason::Spii, FinishReason::ContentFilter),
            (
                GeminiFinishReason::MalformedFunctionCall,
                FinishReason::ToolCall,
            ),
            (
                GeminiFinishReason::FinishReasonUnspecified,
                FinishReason::Unknown,
            ),
            (GeminiFinishReason::Unknown, FinishReason::Unknown),
        ];

        for (gemini_reason, expected_reason) in finish_reasons {
            let text_part = GeminiResponseContentPartData::Text("Test".to_string());
            let content = GeminiResponseContent {
                parts: vec![GeminiResponseContentPart {
                    thought: false,
                    thought_signature: None,
                    data: FlattenUnknown::Normal(text_part),
                }],
            };
            let candidate = GeminiResponseCandidate {
                content: Some(content),
                finish_reason: Some(gemini_reason),
            };
            let response = GeminiResponse {
                candidates: vec![candidate],
                usage_metadata: Some(GeminiUsageMetadata {
                    prompt_token_count: Some(1),
                    candidates_token_count: Some(1),
                }),
            };

            let chunk: ProviderInferenceResponseChunk = {
                let mut last_tool_name = None;
                let mut last_tool_idx = None;
                let mut last_thought_id = 0;
                let mut last_unknown_chunk_id = 0;
                let result =
                    convert_stream_response_with_metadata_to_chunk(ConvertStreamResponseArgs {
                        raw_response: "my_raw_chunk".to_string(),
                        response,
                        latency: Duration::from_millis(10),
                        last_tool_name: &mut last_tool_name,
                        last_tool_idx: &mut last_tool_idx,
                        last_thought_id: &mut last_thought_id,
                        last_unknown_chunk_id: &mut last_unknown_chunk_id,
                        discard_unknown_chunks: false,
                        model_name: "test_model",
                        provider_name: "test_provider",
                    });
                // Verify tool call tracking state
                assert_eq!(last_tool_idx, None);
                result
            }
            .unwrap();
            assert_eq!(chunk.finish_reason, Some(expected_reason));
        }
    }

    #[test]
    fn test_google_ai_studio_gemini_apply_inference_params_called() {
        let logs_contain = crate::utils::testing::capture_logs();
        let inference_params = ChatCompletionInferenceParamsV2 {
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            thinking_budget_tokens: Some(1024),
            verbosity: Some("low".to_string()),
        };
        let mut request = GeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
        };

        apply_inference_params(&mut request, &inference_params);

        // Test that reasoning_effort warns with tip about thinking_budget_tokens
        assert!(logs_contain(
            "Google AI Studio Gemini does not support the inference parameter `reasoning_effort`, so it will be ignored. Tip: You might want to use `thinking_budget_tokens` for this provider."
        ));

        // Test that thinking_budget_tokens is applied correctly in generation_config
        assert!(request.generation_config.is_some());
        let gen_config = request.generation_config.unwrap();
        assert_eq!(
            gen_config.thinking_config,
            Some(GeminiThinkingConfig {
                thinking_budget: 1024,
            })
        );

        // Test that verbosity warns
        assert!(logs_contain(
            "Google AI Studio Gemini does not support the inference parameter `verbosity`"
        ));
    }

    #[tokio::test]
    async fn test_gemini_warns_on_detail() {
        let logs_contain = capture_logs();

        // Test with resolved file with detail
        let dummy_storage_path = StoragePath {
            kind: StorageKind::Disabled,
            path: object_store::path::Path::parse("dummy-path").unwrap(),
        };
        let content_block = ContentBlock::File(Box::new(LazyFile::Base64(PendingObjectStoreFile(
            ObjectStorageFile {
                file: ObjectStoragePointer {
                    source_url: None,
                    mime_type: mime::IMAGE_PNG,
                    storage_path: dummy_storage_path,
                    detail: Some(Detail::Auto),
                    filename: None,
                },
                data: BASE64_STANDARD.encode(b"fake image data"),
            },
        ))));

        let _result = convert_non_thought_content_block(&content_block).await;

        // Should log a warning about detail not being supported
        assert!(logs_contain(
            "The image detail parameter is not supported by Google AI Studio Gemini"
        ));
    }
}
