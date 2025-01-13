use std::time::Duration;

use futures::{Stream, StreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::batch::{BatchRequestRow, PollBatchInferenceResponse};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, serialize_or_log, ModelInferenceRequest,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
    RequestMessage, Usage,
};
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequestJsonMode, Role, Text, TextChunk,
};
use crate::model::{Credential, CredentialLocation};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

const PROVIDER_NAME: &str = "Google AI Studio Gemini";
const PROVIDER_TYPE: &str = "google_ai_studio_gemini";

/// Implements a subset of the Google AI Studio Gemini API as documented [here](https://ai.google.dev/gemini-api/docs/text-generation?lang=rest)
#[derive(Debug)]
pub struct GoogleAIStudioGeminiProvider {
    pub request_url: Url,
    pub streaming_request_url: Url,
    pub credentials: GoogleAIStudioCredentials,
}

impl GoogleAIStudioGeminiProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, PROVIDER_TYPE))?;
        let provider_credentials = GoogleAIStudioCredentials::try_from(generic_credentials)?;

        let request_url = Url::parse(&format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        ))
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to parse request URL: {}", e),
            })
        })?;
        let streaming_request_url = Url::parse(&format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse",
        ))
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Failed to parse streaming request URL: {}", e),
            })
        })?;
        Ok(GoogleAIStudioGeminiProvider {
            request_url,
            streaming_request_url,
            credentials: provider_credentials,
        })
    }
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("GOOGLE_AI_STUDIO_API_KEY".to_string())
}

#[derive(Debug)]
pub enum GoogleAIStudioCredentials {
    Static(SecretString),
    Dynamic(String),
    #[cfg(any(test, feature = "e2e_tests"))]
    None,
}

impl TryFrom<Credential> for GoogleAIStudioCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(GoogleAIStudioCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(GoogleAIStudioCredentials::Dynamic(key_name)),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(GoogleAIStudioCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Google AI Studio Gemini provider"
                    .to_string(),
            }))?,
        }
    }
}

impl GoogleAIStudioCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            GoogleAIStudioCredentials::Static(api_key) => Ok(api_key),
            GoogleAIStudioCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                })
            }
            #[cfg(any(test, feature = "e2e_tests"))]
            GoogleAIStudioCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            })?,
        }
    }
}

impl InferenceProvider for GoogleAIStudioGeminiProvider {
    /// Google AI Studio Gemini non-streaming API request
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body: GeminiRequest = GeminiRequest::new(request)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut url = self.request_url.clone();
        url.query_pairs_mut()
            .append_pair("key", api_key.expose_secret());
        let res = http_client
            .post(url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request: {e}"),
                    status_code: e.status(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(response.clone()),
                })
            })?;
            let response_with_latency = GeminiResponseWithMetadata {
                response,
                latency,
                request: request_body,
                generic_request: request,
            };
            Ok(response_with_latency.try_into()?)
        } else {
            let response_code = res.status();
            let error_body = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
            handle_google_ai_studio_error(response_code, error_body)
        }
    }

    /// Google AI Studio Gemini streaming API request
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
        let request_body: GeminiRequest = GeminiRequest::new(request)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request: {e}"),
            })
        })?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let mut url = self.streaming_request_url.clone();
        url.query_pairs_mut()
            .append_pair("key", api_key.expose_secret());
        let event_source = http_client
            .post(url)
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to Google AI Studio Gemini: {e}"),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        let mut stream = Box::pin(stream_google_ai_studio_gemini(event_source, start_time));
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(ErrorDetails::InferenceServer {
                    message: "Stream ended before first chunk".to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
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
            provider_type: "Google AI Studio Gemini".to_string(),
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

fn stream_google_ai_studio_gemini(
    mut event_source: EventSource,
    start_time: Instant,
) -> impl Stream<Item = Result<ProviderInferenceResponseChunk, Error>> {
    async_stream::stream! {
        let inference_id = Uuid::now_v7();
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    if matches!(e, reqwest_eventsource::Error::StreamEnded) {
                        break;
                    }
                    yield Err(ErrorDetails::InferenceServer {
                        message: e.to_string(),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: None,
                    }.into())
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<GeminiResponse, Error> = serde_json::from_str(&message.data).map_err(|e| {
                            Error::new(ErrorDetails::InferenceServer {
                                message: format!("Error parsing streaming JSON response: {e}"),
                                provider_type: PROVIDER_TYPE.to_string(),
                                raw_request: None,
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
                        let response = GoogleAIStudioGeminiResponseWithMetadata {
                            response: data,
                            latency: start_time.elapsed(),
                            inference_id,
                        }.try_into();
                        yield response
                    }
                }
            }
         }
    }
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

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", untagged)]
enum GeminiPart<'a> {
    Text {
        text: &'a str,
    },
    // TODO (if needed): InlineData { inline_data: Blob },
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

impl<'a> TryFrom<&'a ContentBlock> for GeminiPart<'a> {
    type Error = Error;

    fn try_from(block: &'a ContentBlock) -> Result<Self, Error> {
        match block {
            ContentBlock::Text(Text { text }) => Ok(GeminiPart::Text { text }),
            ContentBlock::ToolResult(tool_result) => {
                // Convert the tool result from String to JSON Value (Gemini expects an object)
                let response: Value = serde_json::from_str(&tool_result.result).map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: format!("Error parsing tool result as JSON Value: {e}"),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: Some(tool_result.result.clone()),
                    })
                })?;

                // Gemini expects the format below according to [the documentation](https://ai.google.dev/gemini-api/docs/function-calling#multi-turn-example-1)
                let response = serde_json::json!({
                    "name": tool_result.name,
                    "content": response,
                });

                Ok(GeminiPart::FunctionResponse {
                    function_response: GeminiFunctionResponse {
                        name: &tool_result.name,
                        response,
                    },
                })
            }
            ContentBlock::ToolCall(tool_call) => {
                // Convert the tool call arguments from String to JSON Value (Gemini expects an object)
                let args: Value = serde_json::from_str(&tool_call.arguments).map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: format!("Error parsing tool call arguments as JSON Value: {e}"),
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

                Ok(GeminiPart::FunctionCall {
                    function_call: GeminiFunctionCall {
                        name: &tool_call.name,
                        args,
                    },
                })
            }
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct GeminiContent<'a> {
    role: GeminiRole,
    parts: Vec<GeminiPart<'a>>,
}

impl<'a> TryFrom<&'a RequestMessage> for GeminiContent<'a> {
    type Error = Error;

    fn try_from(message: &'a RequestMessage) -> Result<Self, Self::Error> {
        let role = GeminiRole::from(message.role);
        let parts: Vec<GeminiPart> = message
            .content
            .iter()
            .map(|block| block.try_into())
            .collect::<Result<_, _>>()?;

        Ok(GeminiContent { role, parts })
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
    function_declarations: Vec<GeminiFunctionDeclaration<'a>>,
    // TODO (if needed): code_execution ([docs](https://ai.google.dev/api/caching#CodeExecution))
}

impl<'a> From<&'a ToolConfig> for GeminiFunctionDeclaration<'a> {
    fn from(tool: &'a ToolConfig) -> Self {
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

impl<'a> From<&'a Vec<ToolConfig>> for GeminiTool<'a> {
    fn from(tools: &'a Vec<ToolConfig>) -> Self {
        let function_declarations: Vec<GeminiFunctionDeclaration<'a>> =
            tools.iter().map(|tc| tc.into()).collect();
        GeminiTool {
            function_declarations,
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

impl<'a> From<&'a ToolChoice> for GoogleAIStudioGeminiToolConfig<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::None => GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::None,
                    allowed_function_names: None,
                },
            },
            ToolChoice::Auto => GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                },
            },
            ToolChoice::Required => GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Any,
                    allowed_function_names: None,
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
    #[allow(dead_code)]
    TextPlain,
    #[serde(rename = "application/json")]
    ApplicationJson,
}

// TODO (if needed): add the other options [here](https://ai.google.dev/api/generate-content#v1beta.GenerationConfig)
#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig<'a> {
    stop_sequences: Option<Vec<&'a str>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    max_output_tokens: Option<u32>,
    seed: Option<u32>,
    response_mime_type: Option<GeminiResponseMimeType>,
    response_schema: Option<Value>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest<'a> {
    contents: Vec<GeminiContent<'a>>,
    tools: Option<Vec<GeminiTool<'a>>>,
    tool_config: Option<GoogleAIStudioGeminiToolConfig<'a>>,
    generation_config: Option<GeminiGenerationConfig<'a>>,
    system_instruction: Option<GeminiContent<'a>>,
}

impl<'a> GeminiRequest<'a> {
    pub fn new(request: &'a ModelInferenceRequest<'a>) -> Result<Self, Error> {
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
                .map(|system_instruction| GeminiPart::Text {
                    text: system_instruction,
                });
        let contents: Vec<GeminiContent> = request
            .messages
            .iter()
            .map(GeminiContent::try_from)
            .collect::<Result<_, _>>()?;
        let (tools, tool_config) = prepare_tools(request);
        let (response_mime_type, response_schema) = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                match request.output_schema {
                    Some(output_schema) => (
                        Some(GeminiResponseMimeType::ApplicationJson),
                        Some(process_output_schema(output_schema)?),
                    ),
                    None => (Some(GeminiResponseMimeType::ApplicationJson), None),
                }
            }
            ModelInferenceRequestJsonMode::Off => (None, None),
        };
        let generation_config = Some(GeminiGenerationConfig {
            stop_sequences: None,
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            seed: request.seed,
            response_mime_type,
            response_schema,
        });
        Ok(GeminiRequest {
            contents,
            tools,
            tool_config,
            generation_config,
            system_instruction: system_instruction.map(|content| GeminiContent {
                role: GeminiRole::Model,
                parts: vec![content],
            }),
        })
    }
}

fn prepare_tools<'a>(
    request: &'a ModelInferenceRequest<'a>,
) -> (
    Option<Vec<GeminiTool<'a>>>,
    Option<GoogleAIStudioGeminiToolConfig<'a>>,
) {
    match &request.tool_config {
        Some(tool_config) => {
            if tool_config.tools_available.is_empty() {
                return (None, None);
            }
            let tools = Some(vec![(&tool_config.tools_available).into()]);
            let tool_config = Some((&tool_config.tool_choice).into());
            (tools, tool_config)
        }
        None => (None, None),
    }
}

fn process_output_schema(output_schema: &Value) -> Result<Value, Error> {
    let mut schema = output_schema.clone();

    /// Recursively remove all instances of "additionalProperties"
    fn remove_additional_properties(value: &mut Value) {
        match value {
            Value::Object(obj) => {
                obj.remove("additionalProperties");
                for (_, v) in obj.iter_mut() {
                    remove_additional_properties(v);
                }
            }
            Value::Array(arr) => {
                for v in arr.iter_mut() {
                    remove_additional_properties(v);
                }
            }
            _ => {}
        }
    }

    remove_additional_properties(&mut schema);
    Ok(schema)
}

#[derive(Debug, Deserialize, Serialize)]
struct GeminiResponseFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
enum GeminiResponseContentPart {
    Text(String),
    // TODO (if needed): InlineData { inline_data: Blob },
    // TODO (if needed): FileData { file_data: FileData },
    FunctionCall(GeminiResponseFunctionCall),
    // TODO (if needed): FunctionResponse
    // TODO (if needed): VideoMetadata { video_metadata: VideoMetadata },
}

impl From<GeminiResponseContentPart> for ContentBlockChunk {
    /// Google AI Studio Gemini does not support parallel tool calling or multiple content blocks as far as I can tell.
    /// So there is no issue with bookkeeping IDs for content blocks.
    /// We should revisit this if they begin to support it.
    fn from(part: GeminiResponseContentPart) -> Self {
        match part {
            GeminiResponseContentPart::Text(text) => ContentBlockChunk::Text(TextChunk {
                text,
                id: "0".to_string(),
            }),
            GeminiResponseContentPart::FunctionCall(function_call) => {
                let arguments = serialize_or_log(&function_call.args);
                ContentBlockChunk::ToolCall(ToolCallChunk {
                    raw_name: function_call.name,
                    raw_arguments: arguments,
                    id: "0".to_string(),
                })
            }
        }
    }
}

impl TryFrom<GeminiResponseContentPart> for ContentBlock {
    type Error = Error;
    fn try_from(part: GeminiResponseContentPart) -> Result<Self, Self::Error> {
        match part {
            GeminiResponseContentPart::Text(text) => Ok(text.into()),
            GeminiResponseContentPart::FunctionCall(function_call) => {
                Ok(ContentBlock::ToolCall(ToolCall {
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
                }))
            }
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponseContentPart>,
}

#[derive(Debug, Deserialize, Serialize)]
struct GeminiResponseCandidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<GeminiResponseContent>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    // Gemini doesn't return output tokens in certain edge cases (e.g. generation blocked by safety settings)
    #[serde(skip_serializing_if = "Option::is_none")]
    candidates_token_count: Option<u32>,
}

impl From<GeminiUsageMetadata> for Usage {
    fn from(usage_metadata: GeminiUsageMetadata) -> Self {
        Usage {
            input_tokens: usage_metadata.prompt_token_count,
            output_tokens: usage_metadata.candidates_token_count.unwrap_or(0),
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
    response: GeminiResponse,
    latency: Latency,
    request: GeminiRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<GeminiResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(response: GeminiResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let GeminiResponseWithMetadata {
            response,
            latency,
            request: request_body,
            generic_request,
        } = response;
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing response from Google AI Studio Gemini: {e}"),
            })
        })?;

        // Google AI Studio Gemini response can contain multiple candidates and each of these can contain
        // multiple content parts. We will only use the first candidate but handle all parts of the response therein.
        let first_candidate = response.candidates.into_iter().next().ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "Google AI Studio Gemini response has no candidates".to_string(),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        // Gemini sometimes doesn't return content in the response (e.g. safety settings blocked the generation).
        let content: Vec<ContentBlock> = match first_candidate.content {
            Some(content) => content
                .parts
                .into_iter()
                .map(|part| part.try_into())
                .collect::<Result<Vec<ContentBlock>, Error>>()?,
            None => vec![],
        };

        let usage = response
            .usage_metadata
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Google AI Studio Gemini non-streaming response has no usage metadata"
                        .to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?
            .into();
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error serializing request: {e}"),
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

struct GoogleAIStudioGeminiResponseWithMetadata {
    response: GeminiResponse,
    latency: Duration,
    inference_id: Uuid,
}

impl TryFrom<GoogleAIStudioGeminiResponseWithMetadata> for ProviderInferenceResponseChunk {
    type Error = Error;
    fn try_from(response: GoogleAIStudioGeminiResponseWithMetadata) -> Result<Self, Self::Error> {
        let GoogleAIStudioGeminiResponseWithMetadata {
            response,
            latency,
            inference_id,
        } = response;

        let raw = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing streaming response from Google AI Studio Gemini: {e}"
                ),
            })
        })?;

        let first_candidate = response.candidates.into_iter().next().ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "Google AI Studio Gemini response has no candidates".to_string(),
                raw_request: None,
                raw_response: Some(raw.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        // Gemini sometimes returns chunks without content (e.g. they might have usage only).
        let mut content: Vec<ContentBlockChunk> = match first_candidate.content {
            Some(content) => content.parts.into_iter().map(|part| part.into()).collect(),
            None => vec![],
        };

        // Gemini occasionally spuriously returns empty text chunks. We filter these out.
        content.retain(|chunk| match chunk {
            ContentBlockChunk::Text(text) => !text.text.is_empty(),
            _ => true,
        });
        Ok(ProviderInferenceResponseChunk::new(
            inference_id,
            content,
            response
                .usage_metadata
                .map(|usage_metadata| usage_metadata.into()),
            raw,
            latency,
        ))
    }
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

    use serde_json::json;

    use super::*;
    use crate::inference::providers::common::{MULTI_TOOL_CONFIG, QUERY_TOOL, WEATHER_TOOL};
    use crate::inference::types::{FunctionType, ModelInferenceRequestJsonMode};
    use crate::tool::{ToolCallConfig, ToolResult};

    #[test]
    fn test_google_ai_studio_gemini_content_try_from() {
        let message = RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into()],
        };
        let content = GeminiContent::try_from(&message).unwrap();
        assert_eq!(content.role, GeminiRole::User);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GeminiPart::Text {
                text: "Hello, world!"
            }
        );

        let message = RequestMessage {
            role: Role::Assistant,
            content: vec!["Hello, world!".to_string().into()],
        };
        let content = GeminiContent::try_from(&message).unwrap();
        assert_eq!(content.role, GeminiRole::Model);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GeminiPart::Text {
                text: "Hello, world!"
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
        let content = GeminiContent::try_from(&message).unwrap();
        assert_eq!(content.role, GeminiRole::Model);
        assert_eq!(content.parts.len(), 2);
        assert_eq!(
            content.parts[0],
            GeminiPart::Text {
                text: "Here's the result of the function call:"
            }
        );
        assert_eq!(
            content.parts[1],
            GeminiPart::FunctionCall {
                function_call: GeminiFunctionCall {
                    name: "get_temperature",
                    args: json!({"location": "New York", "unit": "celsius"}),
                }
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
        let content = GeminiContent::try_from(&message).unwrap();
        assert_eq!(content.role, GeminiRole::User);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GeminiPart::FunctionResponse {
                function_response: GeminiFunctionResponse {
                    name: "get_temperature",
                    response: json!({
                        "name": "get_temperature",
                        "content": json!({"temperature": 25, "conditions": "sunny"})
                    }),
                }
            }
        );
    }

    #[test]
    fn test_from_vec_tool() {
        let tool = GeminiTool::from(&MULTI_TOOL_CONFIG.tools_available);
        assert_eq!(
            tool,
            GeminiTool {
                function_declarations: vec![
                    GeminiFunctionDeclaration {
                        name: "get_temperature",
                        description: "Get the current temperature in a given location",
                        parameters: MULTI_TOOL_CONFIG.tools_available[0].parameters().clone(),
                    },
                    GeminiFunctionDeclaration {
                        name: "query_articles",
                        description: "Query articles from Wikipedia",
                        parameters: MULTI_TOOL_CONFIG.tools_available[1].parameters().clone(),
                    }
                ]
            }
        );
    }

    #[test]
    fn test_from_tool_choice() {
        let tool_choice = ToolChoice::Auto;
        let tool_config = GoogleAIStudioGeminiToolConfig::from(&tool_choice);
        assert_eq!(
            tool_config,
            GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                }
            }
        );

        let tool_choice = ToolChoice::Required;
        let tool_config = GoogleAIStudioGeminiToolConfig::from(&tool_choice);
        assert_eq!(
            tool_config,
            GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Any,
                    allowed_function_names: None,
                }
            }
        );

        let tool_choice = ToolChoice::Specific("get_temperature".to_string());
        let tool_config = GoogleAIStudioGeminiToolConfig::from(&tool_choice);
        assert_eq!(
            tool_config,
            GoogleAIStudioGeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: GeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec!["get_temperature"]),
                }
            }
        );

        let tool_choice = ToolChoice::None;
        let tool_config = GoogleAIStudioGeminiToolConfig::from(&tool_choice);
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

    #[test]
    fn test_google_ai_studio_gemini_request_try_from() {
        // Test Case 1: Empty message list
        let tool_config = ToolCallConfig {
            tools_available: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: false,
        };
        let inference_request = ModelInferenceRequest {
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
        };
        let result = GeminiRequest::new(&inference_request);
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
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
        };
        let result = GeminiRequest::new(&inference_request);
        let request = result.unwrap();
        assert_eq!(request.contents.len(), 2);
        assert_eq!(request.contents[0].role, GeminiRole::User);
        assert_eq!(
            request.contents[0].parts[0],
            GeminiPart::Text { text: "test_user" }
        );
        assert_eq!(request.contents[1].role, GeminiRole::Model);
        assert_eq!(request.contents[1].parts.len(), 1);
        assert_eq!(
            request.contents[1].parts[0],
            GeminiPart::Text {
                text: "test_assistant"
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
        };
        // JSON schema should be supported for Gemini Pro models
        let result = GeminiRequest::new(&inference_request);
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
            GeminiPart::Text { text: "test_user" }
        );
        assert_eq!(
            request.contents[1].parts[0],
            GeminiPart::Text { text: "test_user2" }
        );
        assert_eq!(
            request.contents[2].parts[0],
            GeminiPart::Text {
                text: "test_assistant"
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
        let part = GeminiResponseContentPart::Text("test_assistant".to_string());
        let content = GeminiResponseContent { parts: vec![part] };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: Some(10),
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(1),
        };
        let generic_request = ModelInferenceRequest {
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
            response,
            latency: latency.clone(),
            request: request_body,
            generic_request: &generic_request,
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
                input_tokens: 10,
                output_tokens: 10,
            }
        );
        assert_eq!(model_inference_response.latency, latency);
        assert_eq!(model_inference_response.raw_request, raw_request);
        assert_eq!(model_inference_response.system, None);
        assert_eq!(
            model_inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );
        let text_part =
            GeminiResponseContentPart::Text("Here's the weather information:".to_string());
        let function_call_part =
            GeminiResponseContentPart::FunctionCall(GeminiResponseFunctionCall {
                name: "get_temperature".to_string(),
                args: json!({"location": "New York", "unit": "celsius"}),
            });
        let content = GeminiResponseContent {
            parts: vec![text_part, function_call_part],
        };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: 15,
                candidates_token_count: Some(20),
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(2),
        };
        let generic_request = ModelInferenceRequest {
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
            response,
            latency: latency.clone(),
            request: request_body,
            generic_request: &generic_request,
        };
        let model_inference_response: ProviderInferenceResponse =
            response_with_latency.try_into().unwrap();

        if let [ContentBlock::Text(Text { text }), ContentBlock::ToolCall(tool_call)] =
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
                input_tokens: 15,
                output_tokens: 20,
            }
        );
        assert_eq!(model_inference_response.latency, latency);
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
            GeminiResponseContentPart::Text("Here's the weather information:".to_string());
        let function_call_part =
            GeminiResponseContentPart::FunctionCall(GeminiResponseFunctionCall {
                name: "get_temperature".to_string(),
                args: json!({"location": "New York", "unit": "celsius"}),
            });
        let text_part2 =
            GeminiResponseContentPart::Text("And here's a restaurant recommendation:".to_string());
        let function_call_part2 =
            GeminiResponseContentPart::FunctionCall(GeminiResponseFunctionCall {
                name: "get_restaurant".to_string(),
                args: json!({"cuisine": "Italian", "price_range": "moderate"}),
            });
        let content = GeminiResponseContent {
            parts: vec![
                text_part1,
                function_call_part,
                text_part2,
                function_call_part2,
            ],
        };
        let candidate = GeminiResponseCandidate {
            content: Some(content),
        };
        let response = GeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: 25,
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
            response,
            latency: latency.clone(),
            request: request_body,
            generic_request: &generic_request,
        };
        let model_inference_response: ProviderInferenceResponse =
            response_with_latency.try_into().unwrap();
        assert_eq!(model_inference_response.raw_request, raw_request);

        if let [ContentBlock::Text(Text { text: text1 }), ContentBlock::ToolCall(tool_call1), ContentBlock::Text(Text { text: text2 }), ContentBlock::ToolCall(tool_call2)] =
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
                input_tokens: 25,
                output_tokens: 40,
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
        };
        let (tools, tool_choice) = prepare_tools(&request_with_tools);
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
        };
        let (tools, tool_choice) = prepare_tools(&request_with_tools);
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
    fn test_process_output_schema() {
        let output_schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string", "format": "email"}
            }
        });
        let processed_schema = process_output_schema(&output_schema).unwrap();
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
            process_output_schema(&output_schema_with_additional).unwrap();
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
            process_output_schema(&output_schema_no_additional).unwrap();
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
        let processed_schema_recursive = process_output_schema(&output_schema_recursive).unwrap();
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

        // Test Missing credential (test mode)
        #[cfg(any(test, feature = "e2e_tests"))]
        {
            let generic = Credential::Missing;
            let creds = GoogleAIStudioCredentials::try_from(generic).unwrap();
            assert!(matches!(creds, GoogleAIStudioCredentials::None));
        }

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = GoogleAIStudioCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }
}
