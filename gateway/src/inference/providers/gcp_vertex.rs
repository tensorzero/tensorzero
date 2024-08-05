use std::time::Duration;

use futures::StreamExt;
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
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

/// Implements a subset ofthe GCP Vertex Gemini API as documented [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerateContentResponse)

pub struct GCPVertexGeminiProvider;

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

#[derive(Serialize)]
struct Claims<'a> {
    iss: &'a str,
    sub: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
}

impl<'a> Claims<'a> {
    fn new(iss: &'a str, sub: &'a str, aud: &'a str) -> Self {
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
    pub fn from_env(path: &str) -> Result<Self, Error> {
        let credential_str = std::fs::read_to_string(path).map_err(|e| Error::ApiKeyMissing {
            provider_name: "GCP Vertex Gemini".to_string(),
        })?;
        let credential_value: Value =
            serde_json::from_str(&credential_str).map_err(|e| Error::ApiKeyMissing {
                provider_name: "GCP Vertex Gemini".to_string(),
            })?;
        match (
            credential_value
                .get("private_key_id")
                .ok_or(Error::ApiKeyMissing {
                    provider_name: "GCP Vertex Gemini: private_key_id".to_string(),
                })?
                .as_str(),
            credential_value
                .get("private_key")
                .ok_or(Error::ApiKeyMissing {
                    provider_name: "GCP Vertex Gemini: private_key".to_string(),
                })?
                .as_str(),
            credential_value
                .get("client_email")
                .ok_or(Error::ApiKeyMissing {
                    provider_name: "GCP Vertex Gemini: client_email".to_string(),
                })?
                .as_str(),
        ) {
            (Some(private_key_id), Some(private_key), Some(client_email)) => Ok(GCPCredentials {
                private_key_id: private_key_id.to_string(),
                private_key: EncodingKey::from_rsa_pem(private_key.as_bytes()).map_err(|_| {
                    Error::ApiKeyMissing {
                        provider_name: "GCP Vertex Gemini: private_key failed to parse as RSA"
                            .to_string(),
                    }
                })?,
                client_email: client_email.to_string(),
            }),
            _ => Err(Error::ApiKeyMissing {
                provider_name: "GCP Vertex Gemini".to_string(),
            }),
        }
    }

    fn get_jwt_token(&self, audience: &str) -> String {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(self.private_key_id.clone());
        let claims = Claims::new(&self.client_email, &self.client_email, audience);
        let token = encode(&header, &claims, &self.private_key).unwrap();
        token
    }
}

impl InferenceProvider for GCPVertexGeminiProvider {
    /// GCP Vertex Gemini non-streaming API request
    async fn infer<'a>(
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let (request_url, audience, credentials) = match model {
            ProviderConfig::GCPVertexGemini {
                request_url,
                audience,
                credentials,
            } => (
                request_url,
                audience,
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

        let request_body: GCPVertexGeminiRequest = request.try_into()?;
        let token = credentials.get_jwt_token(audience);
        let start_time = Instant::now();
        let res = http_client
            .post(request_url)
            .header("Authorization", format!("Bearer {}", token))
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
                Error::AnthropicServer {
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
        request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        todo!()
    }
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum GCPVertexGeminiRole {
    User,
    Model,
}

#[derive(Serialize)]
struct GCPVertexGeminiFunctionCall<'a> {
    name: &'a str,
    args: &'a str, // JSON as string
}

#[derive(Serialize)]
struct GCPVertexGeminiFunctionResponse<'a> {
    name: &'a str,
    response: &'a str, // JSON as string
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase", untagged)]
enum GCPVertexGeminiContentPart<'a> {
    Text {
        text: &'a str,
    },
    // TODO: InlineData { inline_data: Blob },
    // TODO: FileData { file_data: FileData },
    FunctionCall {
        function_call: GCPVertexGeminiFunctionCall<'a>,
    },
    FunctionResponse {
        function_response: GCPVertexGeminiFunctionResponse<'a>,
    },
    // TODO: VideoMetadata { video_metadata: VideoMetadata },
}

#[derive(Serialize)]
struct GCPVertexGeminiContent<'a> {
    role: GCPVertexGeminiRole,
    parts: Vec<GCPVertexGeminiContentPart<'a>>,
}

impl<'a> TryFrom<&'a InferenceRequestMessage> for GCPVertexGeminiContent<'a> {
    type Error = Error;
    fn try_from(message: &'a InferenceRequestMessage) -> Result<Self, Self::Error> {
        Ok(match message {
            InferenceRequestMessage::User(message) => GCPVertexGeminiContent {
                role: GCPVertexGeminiRole::User,
                parts: vec![GCPVertexGeminiContentPart::Text {
                    text: &message.content,
                }],
            },
            InferenceRequestMessage::Assistant(message) => {
                let mut parts = vec![];
                if let Some(content) = &message.content {
                    parts.push(GCPVertexGeminiContentPart::Text { text: content });
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        let function_call = GCPVertexGeminiFunctionCall {
                            name: tool_call.name.as_str(),
                            args: tool_call.arguments.as_str(),
                        };
                        parts.push(GCPVertexGeminiContentPart::FunctionCall { function_call });
                    }
                }
                GCPVertexGeminiContent {
                    role: GCPVertexGeminiRole::Model,
                    parts,
                }
            }
            InferenceRequestMessage::System(message) => return Err(Error::InvalidMessage {
                message: "Can't convert System message to GCP Vertex Gemini message. Don't pass System message in except as the first message in the chat.".to_string(),
            }),
            InferenceRequestMessage::Tool(message) => GCPVertexGeminiContent {
                role: GCPVertexGeminiRole::User,
                parts: vec![GCPVertexGeminiContentPart::FunctionCall {
                    function_call: GCPVertexGeminiFunctionCall {
                        name: &message.tool_call_id,
                        args: &message.content,
                    },
                }],
            },
        })
    }
}

#[derive(Serialize)]
struct GCPVertexFunctionDeclaration<'a> {
    name: &'a str,
    description: Option<&'a str>,
    parameters: Option<&'a Value>, // Should be a JSONSchema as a Value
}

// TODO: implement [Retrieval](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Tool#Retrieval)
// and [GoogleSearchRetrieval](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Tool#GoogleSearchRetrieval)
// tools.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
enum GCPVertexGeminiTool<'a> {
    FunctionDeclarations(Vec<GCPVertexFunctionDeclaration<'a>>),
}

impl<'a> From<&'a Vec<Tool>> for GCPVertexGeminiTool<'a> {
    fn from(tools: &'a Vec<Tool>) -> Self {
        GCPVertexGeminiTool::FunctionDeclarations(
            tools
                .into_iter()
                .map(|tool| GCPVertexFunctionDeclaration {
                    name: &tool.name,
                    description: tool.description.as_deref(),
                    parameters: Some(&tool.parameters),
                })
                .collect(),
        )
    }
}

#[derive(Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum GCPVertexGeminiFunctionCallingMode {
    Auto,
    Any,
    None,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiFunctionCallingConfig<'a> {
    mode: GCPVertexGeminiFunctionCallingMode,
    allowed_function_names: Option<Vec<&'a str>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiToolConfig<'a> {
    function_calling_config: GCPVertexGeminiFunctionCallingConfig<'a>,
}

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
                    mode: GCPVertexGeminiFunctionCallingMode::Auto,
                    allowed_function_names: Some(vec![tool_name]),
                },
            },
        }
    }
}

#[derive(Serialize)]
enum GCPVertexGeminiResponseMimeType {
    #[serde(rename = "text/plain")]
    TextPlain,
    #[serde(rename = "application/json")]
    ApplicationJson,
}

// TODO: add the other options [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerationConfig)
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiGenerationConfig<'a> {
    stop_sequences: Option<Vec<&'a str>>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    response_mime_type: Option<GCPVertexGeminiResponseMimeType>,
    response_schema: Option<&'a Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiRequest<'a> {
    contents: Vec<GCPVertexGeminiContent<'a>>,
    tools: Option<Vec<GCPVertexGeminiTool<'a>>>,
    tool_config: Option<GCPVertexGeminiToolConfig<'a>>,
    generation_config: Option<GCPVertexGeminiGenerationConfig<'a>>,
    system_instruction: Option<GCPVertexGeminiContent<'a>>,
    // TODO: [Safety Settings](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/SafetySetting)
}

impl<'a> TryFrom<&'a ModelInferenceRequest<'a>> for GCPVertexGeminiRequest<'a> {
    type Error = Error;
    fn try_from(request: &'a ModelInferenceRequest<'a>) -> Result<Self, Self::Error> {
        if request.messages.is_empty() {
            return Err(Error::InvalidRequest {
                message: "GCP Vertex Gemini requires at least one message".to_string(),
            });
        }
        let first_message = &request.messages[0];
        let (system_instruction, request_messages) = match first_message {
            InferenceRequestMessage::System(message) => {
                let content = GCPVertexGeminiContent {
                    role: GCPVertexGeminiRole::Model,
                    parts: vec![GCPVertexGeminiContentPart::Text {
                        text: message.content.as_str(),
                    }],
                };
                (Some(content), &request.messages[1..])
            }
            _ => (None, &request.messages[..]),
        };
        let contents = request
            .messages
            .iter()
            .map(|message| GCPVertexGeminiContent::try_from(message))
            .collect::<Result<Vec<_>, _>>()?;
        let tools = request
            .tools_available
            .as_ref()
            .map(|tools| vec![GCPVertexGeminiTool::from(tools)]);
        let tool_config = request
            .tool_choice
            .as_ref()
            .map(|tool_choice| GCPVertexGeminiToolConfig::from(tool_choice));
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
            system_instruction,
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
    // TODO: InlineData { inline_data: Blob },
    // TODO: FileData { file_data: FileData },
    FunctionCall {
        function_call: GCPVertexGeminiResponseFunctionCall,
    },
    // TODO (if ever needed): FunctionResponse
    // TODO: VideoMetadata { video_metadata: VideoMetadata },
}

#[derive(Deserialize, Serialize)]
struct GCPVertexGeminiResponseContent {
    parts: Vec<GCPVertexGeminiResponseContentPart>,
}

#[derive(Deserialize, Serialize)]
struct GCPVertexGeminiResponseCandidate {
    index: u8,
    content: GCPVertexGeminiResponseContent,
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
    usage_metadata: GCPVertexGeminiUsageMetadata,
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
        let mut message_text: Option<String> = None;
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        // GCP Vertex Gemini response can contain multiple candidates and each of these can contain
        // multiple content parts. We will only use the first candidate but handle all parts of the response therein.
        let first_candidate = body
            .candidates
            .into_iter()
            .next()
            .ok_or(Error::GCPVertexServer {
                message: "GCP Vertex Gemini response has no candidates".to_string(),
            })?;
        for part in first_candidate.content.parts {
            match part {
                GCPVertexGeminiResponseContentPart::Text { text } => match message_text {
                    Some(message) => message_text = Some(format!("{}\n{}", message, text)),
                    None => message_text = Some(text),
                },
                GCPVertexGeminiResponseContentPart::FunctionCall { function_call } => {
                    let tool_call = ToolCall {
                        name: function_call.name.clone(),
                        arguments: function_call.args,
                        id: function_call.name,
                    };
                    if let Some(calls) = tool_calls.as_mut() {
                        calls.push(tool_call);
                    } else {
                        tool_calls = Some(vec![tool_call]);
                    }
                }
            }
        }

        Ok(ModelInferenceResponse::new(
            message_text,
            tool_calls,
            raw,
            body.usage_metadata.into(),
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
