use std::borrow::Cow;
use std::time::Duration;

use futures::{Stream, StreamExt};
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use reqwest::StatusCode;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::Instant;
use uuid::Uuid;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{
    batch::BatchProviderInferenceResponse, serialize_or_log, ModelInferenceRequest,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
    RequestMessage, Usage,
};
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequestJsonMode, Role, Text, TextChunk,
};
use crate::model::{Credential, CredentialLocation};
use crate::tool::{ToolCall, ToolCallChunk, ToolChoice, ToolConfig};

/// Implements a subset of the GCP Vertex Gemini API as documented [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/generateContent) for non-streaming
/// and [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/streamGenerateContent) for streaming

#[derive(Debug)]
pub struct GCPVertexGeminiProvider {
    pub request_url: String,
    pub streaming_request_url: String,
    pub audience: String,
    pub credentials: GCPVertexCredentials,
    pub model_id: String,
}

impl GCPVertexGeminiProvider {
    pub fn new(
        model_id: String,
        location: String,
        project_id: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credential_location = api_key_location.unwrap_or(default_api_key_location());
        let generic_credentials = Credential::try_from((credential_location, "GCPVertexGemini"))?;
        let provider_credentials= GCPVertexCredentials::try_from((generic_credentials, "GCPVertexGemini"))?;
        let request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent");
        let streaming_request_url = format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:streamGenerateContent?alt=sse");
        let audience = format!("https://{location}-aiplatform.googleapis.com/");

        Ok(GCPVertexGeminiProvider {
            request_url,
            streaming_request_url,
            audience,
            credentials: provider_credentials,
            model_id,
        })
    }
}

pub fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("GCP_VERTEX_CREDENTIALS_PATH".to_string())
}

#[derive(Debug)]
pub enum GCPVertexCredentials {
    Static(GCPServiceAccountCredentials),
    Dynamic(String),
    #[cfg(any(test, feature = "e2e_tests"))]
    None
}

impl TryFrom<(Credential, &str)> for GCPVertexCredentials {
    type Error = Error;
    
    fn try_from((credentials, model): (Credential, &str)) -> Result<Self, Error> {
        match credentials {
            Credential::Static(value) => Ok(GCPVertexCredentials::Static(
                GCPServiceAccountCredentials::from_path(value.expose_secret()).map_err(|e| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: format!("Failed to load GCP credentials: {}", e),
                    })
                })?,
            )),
            Credential::FileContents(file_content) => Ok(GCPVertexCredentials::Static(
                GCPServiceAccountCredentials::from_json_str(file_content.expose_secret()).map_err(|e| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: format!("Failed to load GCP credentials: {}", e),
                    })
                })?,
            )),
            Credential::Dynamic(key_name) => Ok(GCPVertexCredentials::Dynamic(key_name)),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => {
                Ok(GCPVertexCredentials::None)
            },
            _ => Err(Error::new(ErrorDetails::GCPCredentials {
                message: format!("Invalid credential_location for {} provider", model),
            }))?,

        }
    }
}

impl GCPVertexCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        audience: &'a str,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Cow<'a, SecretString>, Error> {
        match self {
            GCPVertexCredentials::Static(credentials) => {
                Ok(Cow::Owned(credentials.get_jwt_token(audience)?.into()))
            }
            GCPVertexCredentials::Dynamic(key_name) => Ok(Cow::Borrowed(
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    Error::new(ErrorDetails::ApiKeyMissing {
                        provider_name: "GCP Vertex Gemini".to_string(),
                    })
                })?,
            )),
            #[cfg(any(test, feature = "e2e_tests"))]
            GCPVertexCredentials::None => Err(Error::new(ErrorDetails::ApiKeyMissing {
                provider_name: "GCP Vertex Gemini".to_string(),
            }))
        }
    }
}

/// Auth
///
/// We implement below the JWT request signing as documented [here](https://developers.google.com/identity/protocols/oauth2/service-account).
///
/// GCPCredentials contains the pieces of information required to successfully make a request using a service account JWT
/// key. The way this works is that there are "claims" about who is making the request and we sign those claims using the key.
#[derive(Clone)]
pub struct GCPServiceAccountCredentials {
    pub private_key_id: String,
    pub private_key: EncodingKey,
    pub client_email: String,
}



impl std::fmt::Debug for GCPServiceAccountCredentials {
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

impl GCPServiceAccountCredentials {
    // Parse a JSON string into a GCPServiceAccountCredentials struct.
    pub fn from_json_str(credential_str: &str) -> Result<Self,Error> {
        let credential_value: Value = serde_json::from_str(credential_str).map_err(|e| {
            Error::new(ErrorDetails::GCPCredentials {
                message: format!("Failed to parse GCP Vertex Gemini credentials: {e}"),
            })
        })?;
        match (
            credential_value
                .get("private_key_id")
                .ok_or_else(|| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: "GCP Vertex Gemini: missing private_key_id".to_string(),
                    })
                })?
                .as_str(),
            credential_value
                .get("private_key")
                .ok_or_else(|| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: "GCP Vertex Gemini: missing private_key".to_string(),
                    })
                })?
                .as_str(),
            credential_value
                .get("client_email")
                .ok_or_else(|| {
                    Error::new(ErrorDetails::GCPCredentials {
                        message: "GCP Vertex Gemini: missing client_email".to_string(),
                    })
                })?
                .as_str(),
        ) {
            (Some(private_key_id), Some(private_key), Some(client_email)) => {
                Ok(GCPServiceAccountCredentials {
                    private_key_id: private_key_id.to_string(),
                    private_key: EncodingKey::from_rsa_pem(private_key.as_bytes()).map_err(
                        |_| {
                            Error::new(ErrorDetails::GCPCredentials {
                                message: "GCP Vertex Gemini: private_key failed to parse as RSA"
                                    .to_string(),
                            })
                        },
                    )?,
                    client_email: client_email.to_string(),
                })
            }
            _ => Err(ErrorDetails::GCPCredentials {
                message: "GCP Vertex Gemini: missing required credentials".to_string(),
            }
            .into()),
        }
    }

    /// Given a path to a JSON key taken from a GCP service account, load the credentials needed to sign requests.
    pub fn from_path(path: &str) -> Result<Self, Error> {
        let credential_str = std::fs::read_to_string(path).map_err(|e| {
            Error::new(ErrorDetails::GCPCredentials {
                message: format!("Failed to read GCP Vertex Gemini credentials: {e}"),
            })
        })?;

        Self::from_json_str(&credential_str)   
    }

    // Get a signed JWT token for the given audience valid from the current time.
    pub fn get_jwt_token(&self, audience: &str) -> Result<String, Error> {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(self.private_key_id.clone());
        let claims = Claims::new(&self.client_email, &self.client_email, audience);
        let token = encode(&header, &claims, &self.private_key).map_err(|e| {
            Error::new(ErrorDetails::GCPCredentials {
                message: format!("Failed to encode JWT: {e}"),
            })
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
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let request_body: GCPVertexGeminiRequest =
            GCPVertexGeminiRequest::new(request, &self.model_id)?;
        let api_key = self
            .credentials
            .get_api_key(&self.audience, dynamic_api_keys)?;
        let start_time = Instant::now();
        let res = http_client
            .post(&self.request_url)
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request: {e}"),
                    status_code: e.status(),
                    provider_type: "GCP Vertex Gemini".to_string(),
                })
            })?;
        let latency = Latency::NonStreaming {
            response_time: start_time.elapsed(),
        };
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    provider_type: "GCP Vertex Gemini".to_string(),
                })
            })?;

            let response = serde_json::from_str(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {response}"),
                    provider_type: "GCP Vertex Gemini".to_string(),
                })
            })?;
            let response_with_latency = GCPVertexGeminiResponseWithMetadata {
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
                    provider_type: "GCP Vertex Gemini".to_string(),
                })
            })?;
            handle_gcp_vertex_gemini_error(response_code, error_body)
        }
    }

    /// GCP Vertex Gemini streaming API request
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
        let request_body: GCPVertexGeminiRequest =
            GCPVertexGeminiRequest::new(request, &self.model_id)?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request: {e}"),
                provider_type: "GCP Vertex Gemini".to_string(),
            })
        })?;
        let api_key = self
            .credentials
            .get_api_key(&self.audience, dynamic_api_keys)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(&self.streaming_request_url)
            .bearer_auth(api_key.expose_secret())
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to GCP Vertex Gemini: {e}"),
                    status_code: None,
                    provider_type: "GCP Vertex Gemini".to_string(),
                })
            })?;
        let mut stream = Box::pin(stream_gcp_vertex_gemini(event_source, start_time));
        let chunk = match stream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(ErrorDetails::InferenceServer {
                    message: "Stream ended before first chunk".to_string(),
                    provider_type: "GCP Vertex Gemini".to_string(),
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
    ) -> Result<BatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "GCP Vertex Gemini".to_string(),
        }
        .into())
    }
}

fn stream_gcp_vertex_gemini(
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
                        provider_type: "GCP Vertex Gemini".to_string(),
                    }
                    .into());
                }
                Ok(event) => match event {
                    Event::Open => continue,
                    Event::Message(message) => {
                        let data: Result<GCPVertexGeminiResponse, Error> = serde_json::from_str(&message.data).map_err(|e| {
                            Error::new(ErrorDetails::InferenceServer {
                                message: format!("Error parsing streaming JSON response: {e}"),
                                provider_type: "GCP Vertex Gemini".to_string(),
                            })
                        });
                        let data = match data {
                            Ok(data) => data,
                            Err(e) => {
                                yield Err(e);
                                continue;
                            }
                        };
                        let response = GCPVertexGeminiStreamResponseWithMetadata {
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

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexGeminiFunctionCall<'a> {
    name: &'a str,
    args: Value,
}

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexGeminiFunctionResponse<'a> {
    name: &'a str,
    response: Value,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", untagged)]
enum GCPVertexGeminiContentPart<'a> {
    Text {
        text: &'a str,
    },
    // TODO (if needed): InlineData { inline_data: Blob },
    // TODO (if needed): FileData { file_data: FileData },
    FunctionCall {
        function_call: GCPVertexGeminiFunctionCall<'a>,
    },
    FunctionResponse {
        function_response: GCPVertexGeminiFunctionResponse<'a>,
    },
    // TODO (if needed): VideoMetadata { video_metadata: VideoMetadata },
}

impl<'a> TryFrom<&'a ContentBlock> for GCPVertexGeminiContentPart<'a> {
    type Error = Error;

    fn try_from(block: &'a ContentBlock) -> Result<Self, Error> {
        match block {
            ContentBlock::Text(Text { text }) => Ok(GCPVertexGeminiContentPart::Text { text }),
            ContentBlock::ToolResult(tool_result) => {
                // Convert the tool result from String to JSON Value (GCP expects an object)
                let response: Value = serde_json::from_str(&tool_result.result).map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: format!("Error parsing tool result as JSON Value: {e}"),
                        provider_type: "GCP Vertex Gemini".to_string(),
                    })
                })?;

                // GCP expects the format below according to [the documentation](https://ai.google.dev/gemini-api/docs/function-calling#multi-turn-example-1)
                let response = serde_json::json!({
                    "name": tool_result.name,
                    "content": response,
                });

                Ok(GCPVertexGeminiContentPart::FunctionResponse {
                    function_response: GCPVertexGeminiFunctionResponse {
                        name: &tool_result.name,
                        response,
                    },
                })
            }
            ContentBlock::ToolCall(tool_call) => {
                // Convert the tool call arguments from String to JSON Value (GCP expects an object)
                let args: Value = serde_json::from_str(&tool_call.arguments).map_err(|e| {
                    Error::new(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: format!("Error parsing tool call arguments as JSON Value: {e}"),
                        provider_type: "GCP Vertex Gemini".to_string(),
                    })
                })?;

                if !args.is_object() {
                    return Err(ErrorDetails::InferenceClient {
                        status_code: Some(StatusCode::BAD_REQUEST),
                        message: "Tool call arguments must be a JSON object".to_string(),
                        provider_type: "GCP Vertex Gemini".to_string(),
                    }
                    .into());
                };

                Ok(GCPVertexGeminiContentPart::FunctionCall {
                    function_call: GCPVertexGeminiFunctionCall {
                        name: &tool_call.name,
                        args,
                    },
                })
            }
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexGeminiContent<'a> {
    role: GCPVertexGeminiRole,
    parts: Vec<GCPVertexGeminiContentPart<'a>>,
}

impl<'a> TryFrom<&'a RequestMessage> for GCPVertexGeminiContent<'a> {
    type Error = Error;

    fn try_from(message: &'a RequestMessage) -> Result<Self, Self::Error> {
        let role = GCPVertexGeminiRole::from(message.role);
        let parts: Vec<GCPVertexGeminiContentPart> = message
            .content
            .iter()
            .map(|block| block.try_into())
            .collect::<Result<_, _>>()?;

        Ok(GCPVertexGeminiContent { role, parts })
    }
}

#[derive(Debug, PartialEq, Serialize)]
struct GCPVertexGeminiFunctionDeclaration<'a> {
    name: &'a str,
    description: Option<&'a str>,
    parameters: Option<Value>, // Should be a JSONSchema as a Value
}

// TODO (if needed): implement [Retrieval](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Tool#Retrieval)
// and [GoogleSearchRetrieval](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Tool#GoogleSearchRetrieval)
// tools.
#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
enum GCPVertexGeminiTool<'a> {
    FunctionDeclarations(Vec<GCPVertexGeminiFunctionDeclaration<'a>>),
}

impl<'a> From<&'a ToolConfig> for GCPVertexGeminiFunctionDeclaration<'a> {
    fn from(tool: &'a ToolConfig) -> Self {
        let mut parameters = tool.parameters().clone();
        if let Some(obj) = parameters.as_object_mut() {
            obj.remove("additionalProperties");
            obj.remove("$schema");
        }

        GCPVertexGeminiFunctionDeclaration {
            name: tool.name(),
            description: Some(tool.description()),
            parameters: Some(parameters),
        }
    }
}

impl<'a> From<&'a Vec<ToolConfig>> for GCPVertexGeminiTool<'a> {
    fn from(tools: &'a Vec<ToolConfig>) -> Self {
        let function_declarations: Vec<GCPVertexGeminiFunctionDeclaration<'a>> =
            tools.iter().map(|tc| tc.into()).collect();
        GCPVertexGeminiTool::FunctionDeclarations(function_declarations)
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum GCPVertexGeminiFunctionCallingMode {
    Auto,
    Any,
    None,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiFunctionCallingConfig<'a> {
    mode: GCPVertexGeminiFunctionCallingMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<&'a str>>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiToolConfig<'a> {
    function_calling_config: GCPVertexGeminiFunctionCallingConfig<'a>,
}

// Auto is the default mode where a tool could be called but it isn't required.
// Any is a mode where a tool is required and if allowed_function_names is Some it has to be from that list.
// See [the documentation](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/ToolConfig) for details.
const MODELS_SUPPORTING_ANY_MODE: &[&str] = &["gemini-1.5-pro-001"];

impl<'a> From<(&'a ToolChoice, &'a str)> for GCPVertexGeminiToolConfig<'a> {
    fn from(input: (&'a ToolChoice, &'a str)) -> Self {
        let (tool_choice, model_name) = input;
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
            ToolChoice::Required => {
                if MODELS_SUPPORTING_ANY_MODE.contains(&model_name) {
                    GCPVertexGeminiToolConfig {
                        function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                            mode: GCPVertexGeminiFunctionCallingMode::Any,
                            allowed_function_names: None,
                        },
                    }
                } else {
                    GCPVertexGeminiToolConfig {
                        function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                            mode: GCPVertexGeminiFunctionCallingMode::Auto,
                            allowed_function_names: None,
                        },
                    }
                }
            }
            ToolChoice::Specific(tool_name) => {
                if MODELS_SUPPORTING_ANY_MODE.contains(&model_name) {
                    GCPVertexGeminiToolConfig {
                        function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                            mode: GCPVertexGeminiFunctionCallingMode::Any,
                            allowed_function_names: Some(vec![tool_name]),
                        },
                    }
                } else {
                    GCPVertexGeminiToolConfig {
                        function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                            mode: GCPVertexGeminiFunctionCallingMode::Auto,
                            allowed_function_names: None,
                        },
                    }
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
enum GCPVertexGeminiResponseMimeType {
    #[serde(rename = "text/plain")]
    #[allow(dead_code)]
    TextPlain,
    #[serde(rename = "application/json")]
    ApplicationJson,
}

// TODO (if needed): add the other options [here](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerationConfig)
#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiGenerationConfig<'a> {
    stop_sequences: Option<Vec<&'a str>>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u32>,
    response_mime_type: Option<GCPVertexGeminiResponseMimeType>,
    response_schema: Option<Value>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiRequest<'a> {
    contents: Vec<GCPVertexGeminiContent<'a>>,
    tools: Option<Vec<GCPVertexGeminiTool<'a>>>,
    tool_config: Option<GCPVertexGeminiToolConfig<'a>>,
    generation_config: Option<GCPVertexGeminiGenerationConfig<'a>>,
    system_instruction: Option<GCPVertexGeminiContent<'a>>,
    // TODO (if needed): [Safety Settings](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/SafetySetting)
}

impl<'a> GCPVertexGeminiRequest<'a> {
    pub fn new(request: &'a ModelInferenceRequest<'a>, model_name: &'a str) -> Result<Self, Error> {
        if request.messages.is_empty() {
            return Err(ErrorDetails::InvalidRequest {
                message: "GCP Vertex Gemini requires at least one message".to_string(),
            }
            .into());
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
            .map(GCPVertexGeminiContent::try_from)
            .collect::<Result<_, _>>()?;
        let (tools, tool_config) = prepare_tools(request, model_name);
        let (response_mime_type, response_schema) = match request.json_mode {
            ModelInferenceRequestJsonMode::On | ModelInferenceRequestJsonMode::Strict => {
                match request.output_schema {
                    Some(output_schema) => {
                        // According to these [docs](https://ai.google.dev/gemini-api/docs/json-mode?lang=web),
                        // JSON mode is only supported for Gemini Pro models not Flash.
                        let strict_json_models = ["gemini-1.5-pro-001"];
                        let response_schema = if strict_json_models.contains(&model_name) {
                            Some(process_output_schema(output_schema)?)
                        } else {
                            None
                        };

                        (
                            Some(GCPVertexGeminiResponseMimeType::ApplicationJson),
                            response_schema,
                        )
                    }
                    None => (Some(GCPVertexGeminiResponseMimeType::ApplicationJson), None),
                }
            }
            ModelInferenceRequestJsonMode::Off => (None, None),
        };
        let generation_config = Some(GCPVertexGeminiGenerationConfig {
            stop_sequences: None,
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            seed: request.seed,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
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

fn prepare_tools<'a>(
    request: &'a ModelInferenceRequest<'a>,
    model_name: &'a str,
) -> (
    Option<Vec<GCPVertexGeminiTool<'a>>>,
    Option<GCPVertexGeminiToolConfig<'a>>,
) {
    match &request.tool_config {
        Some(tool_config) => {
            if tool_config.tools_available.is_empty() {
                return (None, None);
            }
            let tools = Some(vec![(&tool_config.tools_available).into()]);
            let tool_config = Some((&tool_config.tool_choice, model_name).into());
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
struct GCPVertexGeminiResponseFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
enum GCPVertexGeminiResponseContentPart {
    Text(String),
    // TODO (if needed): InlineData { inline_data: Blob },
    // TODO (if needed): FileData { file_data: FileData },
    FunctionCall(GCPVertexGeminiResponseFunctionCall),
    // TODO (if needed): FunctionResponse
    // TODO (if needed): VideoMetadata { video_metadata: VideoMetadata },
}

impl From<GCPVertexGeminiResponseContentPart> for ContentBlockChunk {
    /// GCP Vertex Gemini does not support parallel tool calling or multiple content blocks as far as I can tell.
    /// So there is no issue with bookkeeping IDs for content blocks.
    /// We should revisit this if they begin to support it.
    fn from(part: GCPVertexGeminiResponseContentPart) -> Self {
        match part {
            GCPVertexGeminiResponseContentPart::Text(text) => ContentBlockChunk::Text(TextChunk {
                text,
                id: "0".to_string(),
            }),
            GCPVertexGeminiResponseContentPart::FunctionCall(function_call) => {
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

impl TryFrom<GCPVertexGeminiResponseContentPart> for ContentBlock {
    type Error = Error;
    fn try_from(part: GCPVertexGeminiResponseContentPart) -> Result<Self, Self::Error> {
        match part {
            GCPVertexGeminiResponseContentPart::Text(text) => Ok(text.into()),
            GCPVertexGeminiResponseContentPart::FunctionCall(function_call) => {
                Ok(ContentBlock::ToolCall(ToolCall {
                    name: function_call.name,
                    arguments: serde_json::to_string(&function_call.args).map_err(|e| {
                        Error::new(ErrorDetails::Serialization {
                            message: format!(
                                "Error serializing function call arguments returned from GCP: {e}"
                            ),
                        })
                    })?,
                    // GCP doesn't have the concept of tool call ID so we generate one for our bookkeeping
                    id: Uuid::now_v7().to_string(),
                }))
            }
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct GCPVertexGeminiResponseContent {
    parts: Vec<GCPVertexGeminiResponseContentPart>,
}

#[derive(Debug, Deserialize, Serialize)]
struct GCPVertexGeminiResponseCandidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<GCPVertexGeminiResponseContent>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiUsageMetadata {
    prompt_token_count: Option<u32>,
    // GCP doesn't return output tokens in certain edge cases (e.g. generation blocked by safety settings)
    #[serde(skip_serializing_if = "Option::is_none")]
    candidates_token_count: Option<u32>,
}

impl From<GCPVertexGeminiUsageMetadata> for Usage {
    fn from(usage_metadata: GCPVertexGeminiUsageMetadata) -> Self {
        Usage {
            input_tokens: usage_metadata.prompt_token_count.unwrap_or(0),
            output_tokens: usage_metadata.candidates_token_count.unwrap_or(0),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct GCPVertexGeminiResponse {
    candidates: Vec<GCPVertexGeminiResponseCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage_metadata: Option<GCPVertexGeminiUsageMetadata>,
}

struct GCPVertexGeminiResponseWithMetadata<'a> {
    response: GCPVertexGeminiResponse,
    latency: Latency,
    request: GCPVertexGeminiRequest<'a>,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<GCPVertexGeminiResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(response: GCPVertexGeminiResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let GCPVertexGeminiResponseWithMetadata {
            response,
            latency,
            request: request_body,
            generic_request,
        } = response;
        let raw_response = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing response from GCP Vertex Gemini: {e}"),
                provider_type: "GCP Vertex Gemini".to_string(),
            })
        })?;

        // GCP Vertex Gemini response can contain multiple candidates and each of these can contain
        // multiple content parts. We will only use the first candidate but handle all parts of the response therein.
        let first_candidate = response.candidates.into_iter().next().ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "GCP Vertex Gemini response has no candidates".to_string(),
                provider_type: "GCP Vertex Gemini".to_string(),
            })
        })?;

        // GCP sometimes doesn't return content in the response (e.g. safety settings blocked the generation).
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
                    message: "GCP Vertex Gemini non-streaming response has no usage metadata"
                        .to_string(),
                    provider_type: "GCP Vertex Gemini".to_string(),
                })
            })?
            .into();
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error serializing request: {e}"),
                provider_type: "GCP Vertex Gemini".to_string(),
            })
        })?;
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();

        Ok(ProviderInferenceResponse::new(
            content,
            system,
            input_messages,
            raw_request,
            raw_response,
            usage,
            latency,
        ))
    }
}

struct GCPVertexGeminiStreamResponseWithMetadata {
    response: GCPVertexGeminiResponse,
    latency: Duration,
    inference_id: Uuid,
}

impl TryFrom<GCPVertexGeminiStreamResponseWithMetadata> for ProviderInferenceResponseChunk {
    type Error = Error;
    fn try_from(response: GCPVertexGeminiStreamResponseWithMetadata) -> Result<Self, Self::Error> {
        let GCPVertexGeminiStreamResponseWithMetadata {
            response,
            latency,
            inference_id,
        } = response;

        let raw = serde_json::to_string(&response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error serializing streaming response from GCP Vertex Gemini: {e}"
                ),
                provider_type: "GCP Vertex Gemini".to_string(),
            })
        })?;

        let first_candidate = response.candidates.into_iter().next().ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: "GCP Vertex Gemini response has no candidates".to_string(),
                provider_type: "GCP Vertex Gemini".to_string(),
            })
        })?;

        // GCP sometimes returns chunks without content (e.g. they might have usage only).
        let mut content: Vec<ContentBlockChunk> = match first_candidate.content {
            Some(content) => content.parts.into_iter().map(|part| part.into()).collect(),
            None => vec![],
        };

        // GCP occasionally spuriously returns empty text chunks. We filter these out.
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

fn handle_gcp_vertex_gemini_error(
    response_code: StatusCode,
    response_body: String,
) -> Result<ProviderInferenceResponse, Error> {
    match response_code {
        StatusCode::UNAUTHORIZED
        | StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::TOO_MANY_REQUESTS => Err(Error::new(ErrorDetails::InferenceClient {
            message: response_body,
            status_code: Some(response_code),
            provider_type: "GCP Vertex Gemini".to_string(),
        })),
        // StatusCode::NOT_FOUND | StatusCode::FORBIDDEN | StatusCode::INTERNAL_SERVER_ERROR | 529: Overloaded
        // These are all captured in _ since they have the same error behavior
        _ => Err(Error::new(ErrorDetails::InferenceServer {
            message: response_body,
            provider_type: "GCP Vertex Gemini".to_string(),
        })),
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
    fn test_gcp_vertex_content_try_from() {
        let message = RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into()],
        };
        let content = GCPVertexGeminiContent::try_from(&message).unwrap();
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
        let content = GCPVertexGeminiContent::try_from(&message).unwrap();
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
                    name: "get_temperature".to_string(),
                    arguments: r#"{"location": "New York", "unit": "celsius"}"#.to_string(),
                }),
            ],
        };
        let content = GCPVertexGeminiContent::try_from(&message).unwrap();
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
        let content = GCPVertexGeminiContent::try_from(&message).unwrap();
        assert_eq!(content.role, GCPVertexGeminiRole::User);
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0],
            GCPVertexGeminiContentPart::FunctionResponse {
                function_response: GCPVertexGeminiFunctionResponse {
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
        let tool = GCPVertexGeminiTool::from(&MULTI_TOOL_CONFIG.tools_available);
        assert_eq!(
            tool,
            GCPVertexGeminiTool::FunctionDeclarations(vec![
                GCPVertexGeminiFunctionDeclaration {
                    name: "get_temperature",
                    description: Some("Get the current temperature in a given location"),
                    parameters: Some(MULTI_TOOL_CONFIG.tools_available[0].parameters().clone()),
                },
                GCPVertexGeminiFunctionDeclaration {
                    name: "query_articles",
                    description: Some("Query articles from Wikipedia"),
                    parameters: Some(MULTI_TOOL_CONFIG.tools_available[1].parameters().clone()),
                }
            ])
        );
    }

    #[test]
    fn test_from_tool_choice() {
        let tool_choice = ToolChoice::Auto;
        let pro_model_name = "gemini-1.5-pro-001";
        let flash_model_name = "gemini-1.5-flash-001";
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, pro_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                }
            }
        );
        let tool_choice = ToolChoice::Auto;
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, flash_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                }
            }
        );

        // Flash is still Auto mode since it doesn't support Any mode
        let tool_choice = ToolChoice::Required;
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, flash_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                }
            }
        );

        // The Pro model supports Any mode
        let tool_choice = ToolChoice::Required;
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, pro_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: None,
                }
            }
        );

        let tool_choice = ToolChoice::Specific("get_temperature".to_string());
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, flash_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Auto,
                    allowed_function_names: None,
                }
            }
        );

        // The Pro model supports Any mode with allowed function names
        let tool_choice = ToolChoice::Specific("get_temperature".to_string());
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, pro_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::Any,
                    allowed_function_names: Some(vec!["get_temperature"]),
                }
            }
        );

        // Both Flash and Pro support None mode
        let tool_choice = ToolChoice::None;
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, flash_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::None,
                    allowed_function_names: None,
                }
            }
        );

        let tool_choice = ToolChoice::None;
        let tool_config = GCPVertexGeminiToolConfig::from((&tool_choice, pro_model_name));
        assert_eq!(
            tool_config,
            GCPVertexGeminiToolConfig {
                function_calling_config: GCPVertexGeminiFunctionCallingConfig {
                    mode: GCPVertexGeminiFunctionCallingMode::None,
                    allowed_function_names: None,
                }
            }
        );
    }

    #[test]
    fn test_gcp_vertex_request_try_from() {
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
        let result = GCPVertexGeminiRequest::new(&inference_request, "gemini-pro");
        let details = result.unwrap_err().get_owned_details();
        assert_eq!(
            details,
            ErrorDetails::InvalidRequest {
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
        let result = GCPVertexGeminiRequest::new(&inference_request, "gemini-pro");
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
        let result = GCPVertexGeminiRequest::new(&inference_request, "gemini-1.5-pro-001");
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
        assert_eq!(request.generation_config.as_ref().unwrap().seed, Some(69));
        assert_eq!(
            request
                .generation_config
                .as_ref()
                .unwrap()
                .response_mime_type,
            Some(GCPVertexGeminiResponseMimeType::ApplicationJson)
        );
        assert_eq!(
            request.generation_config.as_ref().unwrap().response_schema,
            Some(output_schema.clone())
        );

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
        // JSON mode should be supported for Gemini Flash models but without a schema
        let result = GCPVertexGeminiRequest::new(&inference_request, "gemini-flash");
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
            Some(GCPVertexGeminiResponseMimeType::ApplicationJson)
        );
        assert_eq!(
            request.generation_config.as_ref().unwrap().response_schema,
            None
        );
    }

    #[test]
    fn test_gcp_to_t0_response() {
        let part = GCPVertexGeminiResponseContentPart::Text("test_assistant".to_string());
        let content = GCPVertexGeminiResponseContent { parts: vec![part] };
        let candidate = GCPVertexGeminiResponseCandidate {
            content: Some(content),
        };
        let response = GCPVertexGeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GCPVertexGeminiUsageMetadata {
                prompt_token_count: None,
                candidates_token_count: None,
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(1),
        };
        let generic_request = ModelInferenceRequest {
            messages: vec![],
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
        let request_body = GCPVertexGeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let response_with_latency = GCPVertexGeminiResponseWithMetadata {
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
                input_tokens: 0,
                output_tokens: 0,
            }
        );
        assert_eq!(model_inference_response.latency, latency);
        assert_eq!(model_inference_response.raw_request, raw_request);
        assert_eq!(
            model_inference_response.system,
            Some("test_system".to_string())
        );
        assert_eq!(model_inference_response.input_messages, vec![]);
        let text_part =
            GCPVertexGeminiResponseContentPart::Text("Here's the weather information:".to_string());
        let function_call_part =
            GCPVertexGeminiResponseContentPart::FunctionCall(GCPVertexGeminiResponseFunctionCall {
                name: "get_temperature".to_string(),
                args: json!({"location": "New York", "unit": "celsius"}),
            });
        let content = GCPVertexGeminiResponseContent {
            parts: vec![text_part, function_call_part],
        };
        let candidate = GCPVertexGeminiResponseCandidate {
            content: Some(content),
        };
        let response = GCPVertexGeminiResponse {
            candidates: vec![candidate],
            usage_metadata: Some(GCPVertexGeminiUsageMetadata {
                prompt_token_count: Some(15),
                candidates_token_count: Some(20),
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(2),
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
        let request_body = GCPVertexGeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let response_with_latency = GCPVertexGeminiResponseWithMetadata {
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
        assert_eq!(model_inference_response.system, None);
        assert_eq!(
            model_inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }]
        );

        let text_part1 =
            GCPVertexGeminiResponseContentPart::Text("Here's the weather information:".to_string());
        let function_call_part =
            GCPVertexGeminiResponseContentPart::FunctionCall(GCPVertexGeminiResponseFunctionCall {
                name: "get_temperature".to_string(),
                args: json!({"location": "New York", "unit": "celsius"}),
            });
        let text_part2 = GCPVertexGeminiResponseContentPart::Text(
            "And here's a restaurant recommendation:".to_string(),
        );
        let function_call_part2 =
            GCPVertexGeminiResponseContentPart::FunctionCall(GCPVertexGeminiResponseFunctionCall {
                name: "get_restaurant".to_string(),
                args: json!({"cuisine": "Italian", "price_range": "moderate"}),
            });
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
                prompt_token_count: Some(25),
                candidates_token_count: Some(40),
            }),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_secs(3),
        };
        let request_body = GCPVertexGeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: None,
            tool_config: None,
            system_instruction: None,
        };
        let raw_request = serde_json::to_string(&request_body).unwrap();
        let response_with_latency = GCPVertexGeminiResponseWithMetadata {
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
        assert_eq!(model_inference_response.system, None);
        assert_eq!(
            model_inference_response.input_messages,
            vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
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
        let (tools, tool_choice) = prepare_tools(&request_with_tools, "gemini-1.5-pro-001");
        let tools = tools.unwrap();
        let tool_config = tool_choice.unwrap();
        assert_eq!(
            tool_config.function_calling_config.mode,
            GCPVertexGeminiFunctionCallingMode::Any,
        );
        assert_eq!(tools.len(), 1);
        match &tools[0] {
            GCPVertexGeminiTool::FunctionDeclarations(function_declarations) => {
                assert_eq!(function_declarations.len(), 2);
                assert_eq!(function_declarations[0].name, WEATHER_TOOL.name());
                assert_eq!(
                    function_declarations[0].parameters,
                    Some(WEATHER_TOOL.parameters().clone())
                );
                assert_eq!(function_declarations[1].name, QUERY_TOOL.name());
                assert_eq!(
                    function_declarations[1].parameters,
                    Some(QUERY_TOOL.parameters().clone())
                );
            }
        }
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
        let (tools, tool_choice) = prepare_tools(&request_with_tools, "gemini-1.5-flash-001");
        let tools = tools.unwrap();
        let tool_config = tool_choice.unwrap();
        // Flash models do not support function calling mode Any
        assert_eq!(
            tool_config.function_calling_config.mode,
            GCPVertexGeminiFunctionCallingMode::Auto,
        );
        assert_eq!(tools.len(), 1);
        match &tools[0] {
            GCPVertexGeminiTool::FunctionDeclarations(function_declarations) => {
                assert_eq!(function_declarations.len(), 2);
                assert_eq!(function_declarations[0].name, WEATHER_TOOL.name());
                assert_eq!(
                    function_declarations[0].parameters,
                    Some(WEATHER_TOOL.parameters().clone())
                );
                assert_eq!(function_declarations[1].name, QUERY_TOOL.name());
                assert_eq!(
                    function_declarations[1].parameters,
                    Some(QUERY_TOOL.parameters().clone())
                );
            }
        }
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
}
