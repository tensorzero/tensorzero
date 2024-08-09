use std::time::Duration;

use futures::{Stream, StreamExt};
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{ContentBlock, ContentBlockChunk, Latency, Role, Text, TextChunk};
use crate::inference::types::{
    InferenceResponseStream, ModelInferenceRequest, ModelInferenceResponse,
    ModelInferenceResponseChunk, RequestMessage, Tool, ToolCall, ToolChoice, Usage,
};

/// Implements a subset of the GCP Vertex Gemini API as documented [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/generateContent) for non-streaming
/// and [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/streamGenerateContent) for streaming

#[derive(Clone, Debug)]
pub struct GCPVertexGeminiProvider {
    pub request_url: String,
    pub streaming_request_url: String,
    pub audience: String,
    pub credentials: Option<GCPCredentials>,
}

/// Auth
///
/// We implement below the JWT request signing as documented [here](https://developers.google.com/identity/protocols/oauth2/service-account).
///
/// GCPCredentials contains the pieces of information required to successfully make a request using a service account JWT
/// key. The way this works is that there are "claims" about who is making the request and we sign those claims using the key.
#[derive(Clone)]
pub struct GCPCredentials {
    private_key_id: String,
    private_key: EncodingKey,
    client_email: String,
}

impl std::fmt::Debug for GCPCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GCPCredentials")
            .field("private_key_id", &self.private_key_id)
            .field("private_key", &"[redacted]")
            .field("client_email", &self.client_email)
            .finish()
    }
}

/// JWT standard claims that are used in GCP auth.
#[derive(Serialize)]
struct Claims<'a> {
    iss: &'a str, // Issuer
    sub: &'a str, // Subject
    aud: &'a str, // Audience
    iat: u64,     // Issued at
    exp: u64,     // Expiration time
}

impl<'a> Claims<'a> {
    fn new(iss: &'a str, sub: &'a str, aud: &'a str) -> Self {
        #[allow(clippy::expect_used)]
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards");
        let iat = current_time.as_secs();
        let exp = (current_time + Duration::from_secs(3600)).as_secs();
        Self {
            iss,
            sub,
            aud,
            iat,
            exp,
        }
    }
}

impl GCPCredentials {
    /// Given a path to a JSON key taken from a GCP service account, load the credentials needed to sign requests.
    pub fn from_env(path: &str) -> Result<Self, Error> {
        let credential_str = std::fs::read_to_string(path).map_err(|e| Error::GCPCredentials {
            message: format!("Failed to read GCP Vertex Gemini credentials: {e}"),
        })?;
        let credential_value: Value =
            serde_json::from_str(&credential_str).map_err(|e| Error::GCPCredentials {
                message: format!("Failed to parse GCP Vertex Gemini credentials: {e}"),
            })?;
        match (
            credential_value
                .get("private_key_id")
                .ok_or(Error::GCPCredentials {
                    message: "GCP Vertex Gemini: missing private_key_id".to_string(),
                })?
                .as_str(),
            credential_value
                .get("private_key")
                .ok_or(Error::GCPCredentials {
                    message: "GCP Vertex Gemini: missing private_key".to_string(),
                })?
                .as_str(),
            credential_value
                .get("client_email")
                .ok_or(Error::GCPCredentials {
                    message: "GCP Vertex Gemini: missing client_email".to_string(),
                })?
                .as_str(),
        ) {
            (Some(private_key_id), Some(private_key), Some(client_email)) => Ok(GCPCredentials {
                private_key_id: private_key_id.to_string(),
                private_key: EncodingKey::from_rsa_pem(private_key.as_bytes()).map_err(|_| {
                    Error::GCPCredentials {
                        message: "GCP Vertex Gemini: private_key failed to parse as RSA"
                            .to_string(),
                    }
                })?,
                client_email: client_email.to_string(),
            }),
            _ => Err(Error::GCPCredentials {
                message: "GCP Vertex Gemini: missing required credentials".to_string(),
            }),
        }
    }

    // Get a signed JWT token for the given audience valid from the current time.
    fn get_jwt_token(&self, audience: &str) -> Result<String, Error> {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(self.private_key_id.clone());
        let claims = Claims::new(&self.client_email, &self.client_email, audience);
        let token =
            encode(&header, &claims, &self.private_key).map_err(|e| Error::GCPCredentials {
                message: format!("Failed to encode JWT: {e}"),
            })?;
        Ok(token)
    }
}

impl InferenceProvider for GCPVertexGeminiProvider {
    /// GCP Vertex Gemini non-streaming API request
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let credentials = self.credentials.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "GCP Vertex Gemini".to_string(),
        })?;
        let request_body: GCPVertexGeminiRequest = request.try_into()?;
        let token = credentials.get_jwt_token(&self.audience)?;
        let start_time = Instant::now();
        let res = http_client
            .post(&self.request_url)
            .bearer_auth(token)
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
            let body = res.json::<GCPVertexGeminiResponse>().await.map_err(|e| {
                Error::GCPVertexServer {
                    message: format!("Error parsing response: {e}"),
                }
            })?;
            let body_with_latency = GCPVertexGeminiResponseWithLatency { body, latency };
            Ok(body_with_latency.try_into()?)
        } else {
            let response_code = res.status();
            let error_body = res.text().await.map_err(|e| Error::GCPVertexServer {
                message: format!("Error parsing response: {e}"),
            })?;
            handle_gcp_vertex_gemini_error(response_code, error_body)
        }
    }

    /// GCP Vertex Gemini streaming API request
    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let credentials = self.credentials.as_ref().ok_or(Error::ApiKeyMissing {
            provider_name: "GCP Vertex Gemini".to_string(),
        })?;
        let request_body: GCPVertexGeminiRequest = request.try_into()?;
        let token = credentials.get_jwt_token(&self.audience)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(&self.streaming_request_url)
            .bearer_auth(token)
            .json(&request_body)
            .eventsource()
            .map_err(|e| Error::InferenceClient {
                message: format!("Error sending request to GCP Vertex Gemini: {e}"),
            })?;
        let mut stream = Box::pin(stream_gcp_vertex_gemini(event_source, start_time));
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(Error::GCPVertexServer {
                    message: "Stream ended before first chunk".to_string(),
                })
            }
        };
        Ok((chunk, stream))
    }
}

fn stream_gcp_vertex_gemini(
    mut event_source: EventSource,
    start_time: Instant,
) -> impl Stream<Item = Result<ModelInferenceResponseChunk, Error>> {
    async_stream::stream! {
        let inference_id = Uuid::now_v7();
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    if matches!(e, reqwest_eventsource::Error::StreamEnded) {
                        break;
                    }
                    yield Err(Error::GCPVertexServer {
                        message: e.to_string(),
                    })
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<GCPVertexGeminiResponse, Error> = serde_json::from_str(&message.data).map_err(|e| Error::GCPVertexServer {
                            message: format!("Error parsing response: {e}"),
                        });
                        let data = match data {
                            Ok(data) => data,
                            Err(e) => {
                                yield Err(e);
                                continue;
                            }
                        };
                        let response = GCPVertexGeminiStreamResponseWithMetadata {
                            body: data,
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

#[derive(Serialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
enum GCPVertexGeminiRole {
    User,
    Model,
}

impl From<Role> for GCPVertexGeminiRole {
    fn from(role: Role) -> Self {
        match role {
            Role::User => GCPVertexGeminiRole::User,
            Role::Assistant => GCPVertexGeminiRole::Model,
        }
    }
}

#[derive(Serialize, PartialEq, Debug)]
struct GCPVertexGeminiFunctionCall<'a> {
    name: &'a str,
    args: &'a str, // JSON as string
}

#[derive(Serialize, PartialEq, Debug)]
struct GCPVertexGeminiFunctionResponse<'a> {
    name: &'a str,
    response: &'a str, // JSON as string
}

#[derive(Serialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase", untagged)]
enum GCPVertexGeminiContentPart<'a> {
    Text {
        text: &'a str,
    },
    // TODO (#19): InlineData { inline_data: Blob },
    // TODO (#19): FileData { file_data: FileData },
    FunctionCall {
        function_call: GCPVertexGeminiFunctionCall<'a>,
    },
    FunctionResponse {
        function_response: GCPVertexGeminiFunctionResponse<'a>,
    },
    // TODO (#19): VideoMetadata { video_metadata: VideoMetadata },
}

impl<'a> From<&'a ContentBlock> for GCPVertexGeminiContentPart<'a> {
    fn from(block: &'a ContentBlock) -> Self {
        match block {
            ContentBlock::Text(Text { text }) => GCPVertexGeminiContentPart::Text { text },
            ContentBlock::ToolResult(tool_result) => GCPVertexGeminiContentPart::FunctionResponse {
                function_response: GCPVertexGeminiFunctionResponse {
                    name: &tool_result.name,
                    response: &tool_result.result,
                },
            },
            ContentBlock::ToolCall(tool_call) => GCPVertexGeminiContentPart::FunctionCall {
                function_call: GCPVertexGeminiFunctionCall {
                    name: &tool_call.name,
                    args: &tool_call.arguments,
                },
            },
        }
    }
}

#[derive(Serialize, Debug, PartialEq)]
struct GCPVertexGeminiContent<'a> {
    role: GCPVertexGeminiRole,
    parts: Vec<GCPVertexGeminiContentPart<'a>>,
}

impl<'a> From<&'a RequestMessage> for GCPVertexGeminiContent<'a> {
    fn from(message: &'a RequestMessage) -> Self {
        let role = GCPVertexGeminiRole::from(message.role);
        let parts: Vec<GCPVertexGeminiContentPart> =
            message.content.iter().map(|block| block.into()).collect();
        GCPVertexGeminiContent { role, parts }
    }
}

#[derive(Serialize, PartialEq, Debug)]
struct GCPVertexGeminiFunctionDeclaration<'a> {
    name: &'a str,
    description: Option<&'a str>,
    parameters: Option<&'a Value>, // Should be a JSONSchema as a Value
}

// TODO (#19): implement [Retrieval](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Tool#Retrieval)
// and [GoogleSearchRetrieval](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Tool#GoogleSearchRetrieval)
// tools.
#[derive(Serialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
enum GCPVertexGeminiTool<'a> {
    FunctionDeclarations(Vec<GCPVertexGeminiFunctionDeclaration<'a>>),
}

impl<'a> From<&'a Tool> for GCPVertexGeminiFunctionDeclaration<'a> {
    fn from(tool: &'a Tool) -> Self {
        match tool {
            Tool::Function {
                description,
                name,
                parameters,
            } => GCPVertexGeminiFunctionDeclaration {
                name,
                description: description.as_deref(),
                parameters: Some(parameters),
            },
        }
    }
}

impl<'a> From<&'a Vec<Tool>> for GCPVertexGeminiTool<'a> {
    fn from(tools: &'a Vec<Tool>) -> Self {
        GCPVertexGeminiTool::FunctionDeclarations(tools.iter().map(|tool| tool.into()).collect())
    }
}

#[derive(Serialize, PartialEq, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum GCPVertexGeminiFunctionCallingMode {
    Auto,
    Any,
    None,
}

#[derive(Serialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiFunctionCallingConfig<'a> {
    mode: GCPVertexGeminiFunctionCallingMode,
    allowed_function_names: Option<Vec<&'a str>>,
}

#[derive(Serialize, PartialEq, Debug)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiToolConfig<'a> {
    function_calling_config: GCPVertexGeminiFunctionCallingConfig<'a>,
}

// Auto is the default mode where a tool could be called but it isn't required.
// Any is a mode where a tool is required and if allowed_function_names is Some it has to be from that list.
// See [the documentation](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/ToolConfig) for details.
impl<'a> From<&'a ToolChoice> for GCPVertexGeminiToolConfig<'a> {
    fn from(tool_choice: &'a ToolChoice) -> Self {
        match tool_choice {
            ToolChoice::None => GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::None,
                    allowed_function_names: None,
                },
            },
            ToolChoice::Auto => GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                },
            },
            ToolChoice::Required => GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: None,
                },
            },
            ToolChoice::Tool(tool_name) => GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec![tool_name]),
                },
            },
            ToolChoice::Implicit => GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec!["respond"]),
                },
            },
        }
    }
}

#[derive(Serialize, Debug, PartialEq)]
enum GCPVertexGeminiResponseMimeType {
    #[serde(rename = "text/plain")]
    #[allow(dead_code)]
    TextPlain,
    #[serde(rename = "application/json")]
    ApplicationJson,
}

// TODO (#19): add the other options [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerationConfig)
#[derive(Serialize, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiGenerationConfig<'a> {
    stop_sequences: Option<Vec<&'a str>>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    response_mime_type: Option<GCPVertexGeminiResponseMimeType>,
    response_schema: Option<&'a Value>,
}

#[derive(Serialize, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiRequest<'a> {
    contents: Vec<GCPVertexGeminiContent<'a>>,
    tools: Option<Vec<GCPVertexGeminiTool<'a>>>,
    tool_config: Option<GCPVertexGeminiToolConfig<'a>>,
    generation_config: Option<GCPVertexGeminiGenerationConfig<'a>>,
    system_instruction: Option<GCPVertexGeminiContent<'a>>,
    // TODO (#19): [Safety Settings](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/SafetySetting)
}

impl<'a> TryFrom<&'a ModelInferenceRequest<'a>> for GCPVertexGeminiRequest<'a> {
    type Error = Error;
    fn try_from(request: &'a ModelInferenceRequest<'a>) -> Result<Self, Self::Error> {
        if request.messages.is_empty() {
            return Err(Error::InvalidRequest {
                message: "GCP Vertex Gemini requires at least one message".to_string(),
            });
        }
        let system_instruction =
            request
                .system
                .as_ref()
                .map(|system_instruction| GCPVertexGeminiContentPart::Text {
                    text: system_instruction,
                });
        let contents: Vec<GCPVertexGeminiContent> = request
            .messages
            .iter()
            .map(GCPVertexGeminiContent::from)
            .collect();
        let tools = request
            .tools_available
            .as_ref()
            .map(|tools| vec![GCPVertexGeminiTool::from(tools)]);
        let tool_config = Some(GCPVertexGeminiToolConfig::from(&request.tool_choice));
        let (response_mime_type, response_schema) = match request.output_schema {
            Some(output_schema) => (
                Some(GCPVertexGeminiResponseMimeType::ApplicationJson),
                Some(output_schema),
            ),
            None => (None, None),
        };
        let generation_config = Some(GCPVertexGeminiGenerationConfig {
            stop_sequences: None,
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            response_mime_type,
            response_schema,
        });
        Ok(GCPVertexGeminiRequest {
            contents,
            tools,
            tool_config,
            generation_config,
            system_instruction: system_instruction.map(|content| GCPVertexGeminiContent {
                role: GCPVertexGeminiRole::Model,
                parts: vec![content],
            }),
        })
    }
}

#[derive(Deserialize, Serialize)]
struct GCPVertexGeminiResponseFunctionCall {
    name: String,
    args: String,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", untagged)]
enum GCPVertexGeminiResponseContentPart {
    Text {
        text: String,
    },
    // TODO (#19): InlineData { inline_data: Blob },
    // TODO (#19): FileData { file_data: FileData },
    FunctionCall {
        function_call: GCPVertexGeminiResponseFunctionCall,
    },
    // TODO (#19, if ever needed): FunctionResponse
    // TODO (#19): VideoMetadata { video_metadata: VideoMetadata },
}

impl From<GCPVertexGeminiResponseContentPart> for ContentBlockChunk {
    fn from(part: GCPVertexGeminiResponseContentPart) -> Self {
        match part {
            GCPVertexGeminiResponseContentPart::Text { text } => {
                ContentBlockChunk::Text(TextChunk {
                    text,
                    // TODO as below
                    id: "0".to_string(),
                })
            }
            GCPVertexGeminiResponseContentPart::FunctionCall { .. } => {
                unimplemented!()
                // TODO (#19, #30): figure out how GCP does bookkeeping for streaming tool calls and implement this here.
                // ContentBlock::ToolCall(ToolCall {
                //     name: function_call.name,
                //     arguments: function_call.args,
                //     id: Uuid::now_v7().to_string(),
                // })
            }
        }
    }
}

impl From<GCPVertexGeminiResponseContentPart> for ContentBlock {
    fn from(part: GCPVertexGeminiResponseContentPart) -> Self {
        match part {
            GCPVertexGeminiResponseContentPart::Text { text } => text.into(),
            GCPVertexGeminiResponseContentPart::FunctionCall { function_call } => {
                ContentBlock::ToolCall(ToolCall {
                    name: function_call.name,
                    arguments: function_call.args,
                    // GCP doesn't have the concept of tool call ID so we generate one for our bookkeeping
                    id: Uuid::now_v7().to_string(),
                })
            }
        }
    }
}

#[derive(Deserialize, Serialize)]
struct GCPVertexGeminiResponseContent {
    parts: Vec<GCPVertexGeminiResponseContentPart>,
}

#[derive(Deserialize, Serialize)]
struct GCPVertexGeminiResponseCandidate {
    content: Option<GCPVertexGeminiResponseContent>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
}

impl From<GCPVertexGeminiUsageMetadata> for Usage {
    fn from(usage_metadata: GCPVertexGeminiUsageMetadata) -> Self {
        Usage {
            prompt_tokens: usage_metadata.prompt_token_count,
            completion_tokens: usage_metadata.candidates_token_count,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiResponse {
    candidates: Vec<GCPVertexGeminiResponseCandidate>,
    usage_metadata: Option<GCPVertexGeminiUsageMetadata>,
}

struct GCPVertexGeminiResponseWithLatency {
    body: GCPVertexGeminiResponse,
    latency: Latency,
}

impl TryFrom<GCPVertexGeminiResponseWithLatency> for ModelInferenceResponse {
    type Error = Error;
    fn try_from(response: GCPVertexGeminiResponseWithLatency) -> Result<Self, Self::Error> {
        let GCPVertexGeminiResponseWithLatency { body, latency } = response;
        let raw = serde_json::to_string(&body).map_err(|e| Error::GCPVertexServer {
            message: format!("Error parsing response from GCP Vertex Gemini: {e}"),
        })?;
        // GCP Vertex Gemini response can contain multiple candidates and each of these can contain
        // multiple content parts. We will only use the first candidate but handle all parts of the response therein.
        let first_candidate = body
            .candidates
            .into_iter()
            .next()
            .ok_or(Error::GCPVertexServer {
                message: "GCP Vertex Gemini response has no candidates".to_string(),
            })?;
        let parts = match first_candidate.content {
            Some(content) => content.parts,
            None => {
                return Err(Error::GCPVertexServer {
                    message: "GCP Vertex Gemini response has no content".to_string(),
                })
            }
        };
        let content: Vec<ContentBlock> = parts.into_iter().map(|part| part.into()).collect();
        let usage = body
            .usage_metadata
            .ok_or(Error::GCPVertexServer {
                message: "GCP Vertex Gemini non-streaming response has no usage metadata"
                    .to_string(),
            })?
            .into();

        Ok(ModelInferenceResponse::new(content, raw, usage, latency))
    }
}

struct GCPVertexGeminiStreamResponseWithMetadata {
    body: GCPVertexGeminiResponse,
    latency: Duration,
    inference_id: Uuid,
}

impl TryFrom<GCPVertexGeminiStreamResponseWithMetadata> for ModelInferenceResponseChunk {
    type Error = Error;
    fn try_from(response: GCPVertexGeminiStreamResponseWithMetadata) -> Result<Self, Self::Error> {
        let GCPVertexGeminiStreamResponseWithMetadata {
            body,
            latency,
            inference_id,
        } = response;
        let raw = serde_json::to_string(&body).map_err(|e| Error::GCPVertexServer {
            message: format!("Error parsing response from GCP Vertex Gemini: {e}"),
        })?;
        let first_candidate = body
            .candidates
            .into_iter()
            .next()
            .ok_or(Error::GCPVertexServer {
                message: "GCP Vertex Gemini response has no candidates".to_string(),
            })?;
        let parts = match first_candidate.content {
            Some(content) => content.parts,
            None => {
                return Err(Error::GCPVertexServer {
                    message: "GCP Vertex Gemini response has no content".to_string(),
                })
            }
        };
        let content: Vec<ContentBlockChunk> = parts.into_iter().map(|part| part.into()).collect();
        Ok(ModelInferenceResponseChunk::new(
            inference_id,
            content,
            body.usage_metadata
                .map(|usage_metadata| usage_metadata.into()),
            raw,
            latency,
        ))
    }
}

fn handle_gcp_vertex_gemini_error(
    response_code: StatusCode,
    response_body: String,
) -> Result<ModelInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED
        | StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::TOO_MANY_REQUESTS => Err(Error::GCPVertexClient {
            message: response_body,
            status_code: response_code,
        }),
        // StatusCode::NOT_FOUND | StatusCode::FORBIDDEN | StatusCode::INTERNAL_SERVER_ERROR | 529: Overloaded
        // These are all captured in _ since they have the same error behavior
        _ => Err(Error::GCPVertexServer {
            message: response_body,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::{FunctionType, JSONMode, Tool, ToolResult};

    #[test]
    fn test_gcp_vertex_content_try_from() {
        let message = RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into()],
        };
        let content = GCPVertexGeminiContent::from(&message);
        assert_eq!(content.role, GCPVertexGeminiRole::User);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GCPVertexGeminiContentPart::Text {
                text: "Hello, world!"
            }
        );

        let message = RequestMessage {
            role: Role::Assistant,
            content: vec!["Hello, world!".to_string().into()],
        };
        let content = GCPVertexGeminiContent::from(&message);
        assert_eq!(content.role, GCPVertexGeminiRole::Model);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GCPVertexGeminiContentPart::Text {
                text: "Hello, world!"
            }
        );
        let message = RequestMessage {
            role: Role::Assistant,
            content: vec![
                "Here's the result of the function call:".to_string().into(),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"location": "New York", "unit": "celsius"}"#.to_string(),
                }),
            ],
        };
        let content = GCPVertexGeminiContent::from(&message);
        assert_eq!(content.role, GCPVertexGeminiRole::Model);
        assert_eq!(content.parts.len(), 2);
        assert_eq!(
            content.parts[0],
            GCPVertexGeminiContentPart::Text {
                text: "Here's the result of the function call:"
            }
        );
        assert_eq!(
            content.parts[1],
            GCPVertexGeminiContentPart::FunctionCall {
                function_call: GCPVertexGeminiFunctionCall {
                    name: "get_weather",
                    args: r#"{"location": "New York", "unit": "celsius"}"#,
                }
            }
        );

        let message = RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                result: r#"{"temperature": 25, "conditions": "sunny"}"#.to_string(),
            })],
        };
        let content = GCPVertexGeminiContent::from(&message);
        assert_eq!(content.role, GCPVertexGeminiRole::User);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GCPVertexGeminiContentPart::FunctionResponse {
                function_response: GCPVertexGeminiFunctionResponse {
                    name: "get_weather",
                    response: r#"{"temperature": 25, "conditions": "sunny"}"#,
                }
            }
        );
    }

    #[test]
    fn test_from_vec_tool() {
        let parameters = [
            serde_json::to_value(r#"{"location": {"type": "string"}, "unit": {"type": "string"}}"#)
                .unwrap(),
            serde_json::to_value(r#"{"timezone": {"type": "string"}}"#).unwrap(),
        ];
        let tools = vec![
            Tool::Function {
                name: "get_weather".to_string(),
                description: Some("Get the weather for a given location".to_string()),
                parameters: parameters[0].clone(),
            },
            Tool::Function {
                name: "get_time".to_string(),
                description: Some("Get the current time for a given timezone".to_string()),
                parameters: parameters[1].clone(),
            },
        ];
        let tool = GCPVertexGeminiTool::from(&tools);
        assert_eq!(
            tool,
            GCPVertexGeminiTool::FunctionDeclarations(vec![
                GCPVertexGeminiFunctionDeclaration {
                    name: "get_weather",
                    description: Some("Get the weather for a given location"),
                    parameters: Some(&parameters[0]),
                },
                GCPVertexGeminiFunctionDeclaration {
                    name: "get_time",
                    description: Some("Get the current time for a given timezone"),
                    parameters: Some(&parameters[1]),
                }
            ])
        );
    }

    #[test]
    fn test_from_tool_choice() {
        let tool_choice = ToolChoice::Auto;
        let tool_config = GCPVertexGeminiToolConfig::from(&tool_choice);
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                }
            }
        );

        let tool_choice = ToolChoice::Tool("get_weather".to_string());
        let tool_config = GCPVertexGeminiToolConfig::from(&tool_choice);
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec!["get_weather"]),
                }
            }
        );

        let tool_choice = ToolChoice::None;
        let tool_config = GCPVertexGeminiToolConfig::from(&tool_choice);
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::None,
                    allowed_function_names: None,
                }
            }
        );

        let tool_choice = ToolChoice::Required;
        let tool_config = GCPVertexGeminiToolConfig::from(&tool_choice);
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: None,
                }
            }
        );

        let tool_choice = ToolChoice::Implicit;
        let tool_config = GCPVertexGeminiToolConfig::from(&tool_choice);
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec!["respond"]),
                }
            }
        );
    }

    #[test]
    fn test_gcp_vertex_request_try_from() {
        // Test Case 1: Empty message list
        let inference_request = ModelInferenceRequest {
            messages: vec![],
            system: None,
            tools_available: None,
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let result = GCPVertexGeminiRequest::try_from(&inference_request);
        let error = result.unwrap_err();
        assert_eq!(
            error,
            Error::InvalidRequest {
                message: "GCP Vertex Gemini requires at least one message".to_string()
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
            tools_available: None,
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let result = GCPVertexGeminiRequest::try_from(&inference_request);
        let request = result.unwrap();
        assert_eq!(request.contents.len(), 2);
        assert_eq!(request.contents[0].role, GCPVertexGeminiRole::User);
        assert_eq!(
            request.contents[0].parts[0],
            GCPVertexGeminiContentPart::Text { text: "test_user" }
        );
        assert_eq!(request.contents[1].role, GCPVertexGeminiRole::Model);
        assert_eq!(request.contents[1].parts.len(), 1);
        assert_eq!(
            request.contents[1].parts[0],
            GCPVertexGeminiContentPart::Text {
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
        let inference_request = ModelInferenceRequest {
            messages: messages.clone(),
            system: Some("test_system".to_string()),
            tools_available: None,
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            temperature: Some(0.5),
            max_tokens: Some(100),
            stream: true,
            json_mode: JSONMode::On,
            function_type: FunctionType::Chat,
            output_schema: None,
        };
        let result = GCPVertexGeminiRequest::try_from(&inference_request);
        let request = result.unwrap();
        assert_eq!(request.contents.len(), 3);
        assert_eq!(request.contents[0].role, GCPVertexGeminiRole::User);
        assert_eq!(request.contents[1].role, GCPVertexGeminiRole::User);
        assert_eq!(request.contents[2].role, GCPVertexGeminiRole::Model);
        assert_eq!(request.contents[0].parts.len(), 1);
        assert_eq!(request.contents[1].parts.len(), 1);
        assert_eq!(request.contents[2].parts.len(), 1);
        assert_eq!(
            request.contents[0].parts[0],
            GCPVertexGeminiContentPart::Text { text: "test_user" }
        );
        assert_eq!(
            request.contents[1].parts[0],
            GCPVertexGeminiContentPart::Text { text: "test_user2" }
        );
        assert_eq!(
            request.contents[2].parts[0],
            GCPVertexGeminiContentPart::Text {
                text: "test_assistant"
            }
        );
        assert_eq!(
            request.generation_config.as_ref().unwrap().temperature,
            Some(0.5)
        );
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .max_output_tokens,
            Some(100)
        );
    }

    #[test]
    fn test_gcp_to_t0_response() {
        let part = GCPVertexGeminiResponseContentPart::Text {
            text: "test_assistant".to_string(),
        };
        let content = GCPVertexGeminiResponseContent { parts: vec![part] };
        let candidate = GCPVertexGeminiResponseCandidate {
            content: Some(content),
        };
        let response = GCPVertexGeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GCPVertexGeminiUsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 10,
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(1),
        };
        let response_with_latency = GCPVertexGeminiResponseWithLatency {
            body: response,
            latency: latency.clone(),
        };
        let model_inference_response: ModelInferenceResponse =
            response_with_latency.try_into().unwrap();
        assert_eq!(
            model_inference_response.content,
            vec!["test_assistant".to_string().into()]
        );
        assert_eq!(
            model_inference_response.usage,
            Usage {
                prompt_tokens: 10,
                completion_tokens: 10,
            }
        );
        assert_eq!(model_inference_response.latency, latency);

        let text_part = GCPVertexGeminiResponseContentPart::Text {
            text: "Here's the weather information:".to_string(),
        };
        let function_call_part = GCPVertexGeminiResponseContentPart::FunctionCall {
            function_call: GCPVertexGeminiResponseFunctionCall {
                name: "get_weather".to_string(),
                args: r#"{"location": "New York", "unit": "celsius"}"#.to_string(),
            },
        };
        let content = GCPVertexGeminiResponseContent {
            parts: vec![text_part, function_call_part],
        };
        let candidate = GCPVertexGeminiResponseCandidate {
            content: Some(content),
        };
        let response = GCPVertexGeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GCPVertexGeminiUsageMetadata {
                prompt_token_count: 15,
                candidates_token_count: 20,
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(2),
        };
        let response_with_latency = GCPVertexGeminiResponseWithLatency {
            body: response,
            latency: latency.clone(),
        };
        let model_inference_response: ModelInferenceResponse =
            response_with_latency.try_into().unwrap();

        if let [ContentBlock::Text(Text { text }), ContentBlock::ToolCall(tool_call)] =
            &model_inference_response.content[..]
        {
            assert_eq!(text, "Here's the weather information:");
            assert_eq!(tool_call.name, "get_weather");
            assert_eq!(
                tool_call.arguments,
                r#"{"location": "New York", "unit": "celsius"}"#
            );
        } else {
            unreachable!()
        }

        assert_eq!(
            model_inference_response.usage,
            Usage {
                prompt_tokens: 15,
                completion_tokens: 20,
            }
        );
        assert_eq!(model_inference_response.latency, latency);

        let text_part1 = GCPVertexGeminiResponseContentPart::Text {
            text: "Here's the weather information:".to_string(),
        };
        let function_call_part = GCPVertexGeminiResponseContentPart::FunctionCall {
            function_call: GCPVertexGeminiResponseFunctionCall {
                name: "get_weather".to_string(),
                args: r#"{"location": "New York", "unit": "celsius"}"#.to_string(),
            },
        };
        let text_part2 = GCPVertexGeminiResponseContentPart::Text {
            text: "And here's a restaurant recommendation:".to_string(),
        };
        let function_call_part2 = GCPVertexGeminiResponseContentPart::FunctionCall {
            function_call: GCPVertexGeminiResponseFunctionCall {
                name: "get_restaurant".to_string(),
                args: r#"{"cuisine": "Italian", "price_range": "moderate"}"#.to_string(),
            },
        };
        let content = GCPVertexGeminiResponseContent {
            parts: vec![
                text_part1,
                function_call_part,
                text_part2,
                function_call_part2,
            ],
        };
        let candidate = GCPVertexGeminiResponseCandidate {
            content: Some(content),
        };
        let response = GCPVertexGeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GCPVertexGeminiUsageMetadata {
                prompt_token_count: 25,
                candidates_token_count: 40,
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(3),
        };
        let response_with_latency = GCPVertexGeminiResponseWithLatency {
            body: response,
            latency: latency.clone(),
        };
        let model_inference_response: ModelInferenceResponse =
            response_with_latency.try_into().unwrap();

        if let [ContentBlock::Text(Text { text: text1 }), ContentBlock::ToolCall(tool_call1), ContentBlock::Text(Text { text: text2 }), ContentBlock::ToolCall(tool_call2)] =
            &model_inference_response.content[..]
        {
            assert_eq!(text1, "Here's the weather information:");
            assert_eq!(text2, "And here's a restaurant recommendation:");
            assert_eq!(tool_call1.name, "get_weather");
            assert_eq!(
                tool_call1.arguments,
                r#"{"location": "New York", "unit": "celsius"}"#
            );
            assert_eq!(tool_call2.name, "get_restaurant");
            assert_eq!(
                tool_call2.arguments,
                r#"{"cuisine": "Italian", "price_range": "moderate"}"#
            );
        } else {
            unreachable!(
                "Content does not match expected structure: {:?}",
                model_inference_response.content
            );
        }

        assert_eq!(
            model_inference_response.usage,
            Usage {
                prompt_tokens: 25,
                completion_tokens: 40,
            }
        );
        assert_eq!(model_inference_response.latency, latency);
    }
}
