#![allow(clippy::unwrap_used)]
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use lazy_static::lazy_static;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use serde_json::{json, Value};
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::inference::InferenceProvider;

use crate::cache::ModelProviderRequest;
use crate::embeddings::{
    Embedding, EmbeddingProvider, EmbeddingProviderRequestInfo, EmbeddingProviderResponse,
    EmbeddingRequest,
};

use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DelayedError, Error, ErrorDetails};
use crate::http::TensorzeroHttpClient;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::batch::{BatchRequestRow, BatchStatus};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, current_timestamp, ContentBlockChunk,
    ContentBlockOutput, Latency, ModelInferenceRequest, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, Usage,
};
use crate::inference::types::{ContentBlock, FinishReason, ProviderInferenceResponseStreamInner};
use crate::inference::types::{Text, TextChunk, Thought, ThoughtChunk};
use crate::model::{CredentialLocation, CredentialLocationWithFallback, ModelProvider};
use crate::providers::helpers::inject_extra_request_data;
use crate::rate_limiting::{
    ActiveRateLimitKey, FailedRateLimit, RateLimitResource, RateLimitingScopeKey,
};
use crate::tool::{ToolCall, ToolCallChunk};

const PROVIDER_NAME: &str = "Dummy";
pub const PROVIDER_TYPE: &str = "dummy";

#[derive(Debug, Default, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct DummyProvider {
    pub model_name: String,
    #[serde(skip)]
    pub credentials: DummyCredentials,
}

impl DummyProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocationWithFallback>,
    ) -> Result<Self, Error> {
        let api_key_location = api_key_location
            .map(|loc| loc.default_location().clone())
            .unwrap_or_else(default_api_key_location);
        match api_key_location {
            CredentialLocation::Dynamic(key_name) => Ok(DummyProvider {
                model_name,
                credentials: DummyCredentials::Dynamic(key_name),
            }),
            CredentialLocation::None => Ok(DummyProvider {
                model_name,
                credentials: DummyCredentials::None,
            }),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Dummy provider".to_string(),
            })),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    fn get_model_usage(&self, output_tokens: u32) -> Usage {
        match self.model_name.as_str() {
            "input_tokens_zero" => Usage {
                input_tokens: Some(0),
                output_tokens: Some(output_tokens),
            },
            "output_tokens_zero" => Usage {
                input_tokens: Some(10),
                output_tokens: Some(0),
            },
            "input_tokens_output_tokens_zero" => Usage {
                input_tokens: Some(0),
                output_tokens: Some(0),
            },
            "input_five_output_six" => Usage {
                input_tokens: Some(5),
                output_tokens: Some(6),
            },
            _ => Usage {
                input_tokens: Some(10),
                output_tokens: Some(output_tokens),
            },
        }
    }

    async fn create_streaming_reasoning_response(
        &self,
        thinking_chunks: Vec<&'static str>,
        response_chunks: Vec<&'static str>,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let thinking_chunks = thinking_chunks.into_iter().map(|chunk| {
            ContentBlockChunk::Thought(ThoughtChunk {
                text: Some(chunk.to_string()),
                signature: None,
                summary_id: None,
                summary_text: None,
                id: "0".to_string(),
                provider_type: None,
            })
        });
        let response_chunks = response_chunks.into_iter().map(|chunk| {
            ContentBlockChunk::Text(TextChunk {
                text: chunk.to_string(),
                id: "0".to_string(),
            })
        });
        let num_chunks = thinking_chunks.len() + response_chunks.len();
        let created = current_timestamp();
        let chained = thinking_chunks
            .into_iter()
            .chain(response_chunks.into_iter());
        let total_tokens = num_chunks as u32;
        let stream = tokio_stream::iter(chained.enumerate())
            .map(move |(i, chunk)| {
                Ok(ProviderInferenceResponseChunk {
                    created,
                    content: vec![chunk],
                    usage: None,
                    raw_response: String::new(),
                    latency: Duration::from_millis(50 + 10 * (i as u64 + 1)),
                    finish_reason: None,
                })
            })
            .chain(tokio_stream::once(Ok(ProviderInferenceResponseChunk {
                created,
                content: vec![],
                usage: Some(self.get_model_usage(total_tokens)),
                finish_reason: Some(FinishReason::Stop),
                raw_response: String::new(),
                latency: Duration::from_millis(50 + 10 * (num_chunks as u64)),
            })))
            .throttle(std::time::Duration::from_millis(10));

        Ok((
            futures::stream::StreamExt::peekable(Box::pin(stream)),
            DUMMY_RAW_REQUEST.to_string(),
        ))
    }
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::None
}

#[derive(Debug, Default)]
pub enum DummyCredentials {
    #[default]
    None,
    Dynamic(String),
}

impl DummyCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<Option<&'a SecretString>, DelayedError> {
        match self {
            DummyCredentials::None => Ok(None),
            DummyCredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    DelayedError::new(ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                        message: format!("Dynamic api key `{key_name}` is missing"),
                    })
                }))
                .transpose()
            }
        }
    }
}

pub static DUMMY_INFER_RESPONSE_CONTENT: &str = "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.";
pub static DUMMY_INFER_RESPONSE_RAW: &str = r#"{
  "id": "id",
  "object": "text.completion",
  "created": 1618870400,
  "model": "text-davinci-002",
  "choices": [
    {
      "text": "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.",
      "index": 0,
      "logprobs": null,
      "finish_reason": null
    }
  ]
}"#;

pub static ALTERNATE_INFER_RESPONSE_CONTENT: &str =
    "Megumin chanted her spell, but instead of an explosion, a gentle rain began to fall.";

lazy_static! {
    pub static ref DUMMY_TOOL_RESPONSE: Value = json!({"location": "Brooklyn", "units": "celsius"});
    // This is the same as DUMMY_TOOL_RESPONSE, but with the units capitalized
    // Since that field is an enum, this should fail validation
    pub static ref DUMMY_BAD_TOOL_RESPONSE: Value = json!({"location": "Brooklyn", "units": "Celsius"});
    static ref FLAKY_COUNTERS: Mutex<HashMap<String, u16>> = Mutex::new(HashMap::new());
}
pub static DUMMY_JSON_RESPONSE_RAW: &str = r#"{"answer":"Hello"}"#;
pub static DUMMY_JSON_GOODBYE_RESPONSE_RAW: &str = r#"{"answer":"Goodbye"}"#;
pub static DUMMY_JSON_RESPONSE_RAW_DIFF_SCHEMA: &str = r#"{"response":"Hello"}"#;
pub static DUMMY_JSON_COT_RESPONSE_RAW: &str =
    r#"{"thinking":"hmmm", "response": {"answer":"tokyo!"}}"#;
pub static DUMMY_STREAMING_THINKING: [&str; 2] = ["hmmm", "hmmm"];
pub static DUMMY_STREAMING_RESPONSE: [&str; 16] = [
    "Wally,",
    " the",
    " golden",
    " retriever,",
    " wagged",
    " his",
    " tail",
    " excitedly",
    " as",
    " he",
    " devoured",
    " a",
    " slice",
    " of",
    " cheese",
    " pizza.",
];
pub static DUMMY_STREAMING_TOOL_RESPONSE: [&str; 5] = [
    r#"{"location""#,
    r#":"Brooklyn""#,
    r#","units""#,
    r#":"celsius"#,
    r#""}"#,
];

pub static DUMMY_STREAMING_GOOD_TOOL_RESPONSE: [&str; 5] = [
    r#"{"sentiment""#,
    r#":"positive""#,
    r#","confidence""#,
    r":0.95",
    r"}",
];

pub static DUMMY_STREAMING_JSON_RESPONSE: [&str; 5] =
    [r#"{"name""#, r#":"John""#, r#","age""#, r":30", r"}"];

pub static DUMMY_RAW_REQUEST: &str = "raw request";

impl InferenceProvider for DummyProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        _http_client: &'a TensorzeroHttpClient,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        if self.model_name == "slow" {
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        // Just so they don't seem like 0ms inferences
        tokio::time::sleep(Duration::from_millis(1)).await;

        // Check for flaky models
        if self.model_name.starts_with("flaky_") {
            #[expect(clippy::expect_used)]
            let mut counters = FLAKY_COUNTERS
                .lock()
                .expect("FLAKY_COUNTERS mutex is poisoned");
            let counter = counters.entry(self.model_name.clone()).or_insert(0);
            *counter += 1;

            // Fail on even-numbered calls
            if counter.is_multiple_of(2) {
                if self.model_name.contains("rate_limit") {
                    return Err(ErrorDetails::RateLimitExceeded {
                        failed_rate_limits: vec![FailedRateLimit {
                            key: ActiveRateLimitKey(String::from("key")),
                            requested: 100,
                            available: 0,
                            resource: RateLimitResource::Token,
                            scope_key: vec![RateLimitingScopeKey::TagTotal {
                                key: "test".to_string(),
                            }],
                        }],
                    }
                    .into());
                }
                return Err(ErrorDetails::InferenceClient {
                    raw_request: Some("raw request".to_string()),
                    raw_response: None,
                    message: format!(
                        "Flaky model '{}' failed on call number {}",
                        self.model_name, *counter
                    ),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                }
                .into());
            }
        }

        if self.model_name.starts_with("error") {
            return Err(ErrorDetails::InferenceClient {
                message: format!(
                    "Error sending request to Dummy provider for model '{}'.",
                    self.model_name
                ),
                raw_request: Some("raw request".to_string()),
                raw_response: None,
                status_code: None,
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }
        if self.model_name == "multiple-text-blocks" {
            // The first message must have 2 text blocks or we error
            let first_message = &request.messages[0];
            let first_message_text_content = first_message
                .content
                .iter()
                .filter(|block| matches!(block, ContentBlock::Text(_)))
                .collect::<Vec<_>>();
            if first_message_text_content.len() != 2 {
                return Err(ErrorDetails::InferenceClient {
                    message: "First message must have exactly two text blocks".to_string(),
                    raw_request: Some("raw request".to_string()),
                    raw_response: None,
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                }
                .into());
            }
        }

        let api_key = self
            .credentials
            .get_api_key(dynamic_api_keys)
            .map_err(|e| e.log())?;
        if self.model_name == "test_key" {
            if let Some(api_key) = api_key {
                if api_key.expose_secret() != "good_key" {
                    return Err(ErrorDetails::InferenceClient {
                        message: "Invalid API key for Dummy provider".to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                        status_code: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    }
                    .into());
                }
            }
        }
        let id = Uuid::now_v7();
        let created = current_timestamp();
        let content = match self.model_name.as_str() {
            "null" => vec![],
            "tool" => vec![ContentBlockOutput::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                #[expect(clippy::unwrap_used)]
                arguments: serde_json::to_string(&*DUMMY_TOOL_RESPONSE).unwrap(),
                id: "0".to_string(),
            })],
            "good_tool" => vec![ContentBlockOutput::ToolCall(ToolCall {
                name: "output_schema_tool".to_string(),
                arguments: r#"{"sentiment":"positive","confidence":0.95}"#.to_string(),
                id: "0".to_string(),
            })],
            "invalid_tool_arguments" => vec![ContentBlockOutput::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                arguments: "Not valid 'JSON'".to_string(),
                id: "0".to_string(),
            })],
            "reasoner" => vec![
                ContentBlockOutput::Thought(Thought {
                    text: Some("hmmm".to_string()),
                    signature: None,
                    summary: None,
                    provider_type: None,
                }),
                ContentBlockOutput::Text(Text {
                    text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
                }),
            ],
            "reasoner_with_signature" => vec![
                ContentBlockOutput::Thought(Thought {
                    text: Some("hmmm".to_string()),
                    signature: Some("my_signature".to_string()),
                    summary: None,
                    provider_type: None,
                }),
                ContentBlockOutput::Text(Text {
                    text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
                }),
            ],
            "json_reasoner" => vec![
                ContentBlockOutput::Thought(Thought {
                    text: Some("hmmm".to_string()),
                    signature: None,
                    summary: None,
                    provider_type: None,
                }),
                ContentBlockOutput::Text(Text {
                    text: DUMMY_JSON_RESPONSE_RAW.to_string(),
                }),
            ],
            "bad_tool" => vec![ContentBlockOutput::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                #[expect(clippy::unwrap_used)]
                arguments: serde_json::to_string(&*DUMMY_BAD_TOOL_RESPONSE).unwrap(),
                id: "0".to_string(),
            })],
            "json" => vec![DUMMY_JSON_RESPONSE_RAW.to_string().into()],
            "json_goodbye" => vec![DUMMY_JSON_GOODBYE_RESPONSE_RAW.to_string().into()],
            "json_cot" => vec![DUMMY_JSON_COT_RESPONSE_RAW.to_string().into()],
            "json_diff_schema" => vec![DUMMY_JSON_RESPONSE_RAW_DIFF_SCHEMA.to_string().into()],
            "json_beatles_1" => vec![r#"{"names":["John", "George"]}"#.to_string().into()],
            "json_beatles_2" => vec![r#"{"names":["Paul", "Ringo"]}"#.to_string().into()],
            "best_of_n_0" => {
                vec![r#"{"thinking": "hmmm", "answer_choice": 0}"#.to_string().into()]
            }
            "best_of_n_1" => {
                vec![r#"{"thinking": "hmmm", "answer_choice": 1}"#.to_string().into()]
            }
            "best_of_n_big" => {
                vec![r#"{"thinking": "hmmm", "answer_choice": 100}"#.to_string().into()]
            }
            "flaky_best_of_n_judge" => {
                vec![r#"{"thinking": "hmmm", "answer_choice": 0}"#.to_string().into()]
            }
            "random_answer" => {
                vec![ContentBlockOutput::Text(Text {
                    text: serde_json::json!({
                        "answer": Uuid::now_v7().to_string()
                    })
                    .to_string(),
                })]
            }
            "alternate" => vec![ALTERNATE_INFER_RESPONSE_CONTENT.to_string().into()],
            "echo_extra_info" => {
                vec![ContentBlockOutput::Text(Text {
                    text: json!({
                        "extra_body": request.extra_body,
                        "extra_headers": request.extra_headers,
                    })
                    .to_string(),
                })]
            }
            "echo_injected_data" => {
                let mut body = serde_json::json!({});
                let headers = inject_extra_request_data(
                    &request.extra_body,
                    &request.extra_headers,
                    model_provider,
                    model_name,
                    &mut body,
                )?;
                vec![ContentBlockOutput::Text(Text {
                    text: json!({
                        "injected_body": body,
                        "injected_headers": headers.into_iter().map(|(k, v)| (k.unwrap().to_string(), v.to_str().unwrap().to_string())).collect::<Vec<_>>(),
                    })
                    .to_string(),
                })]
            }
            "echo_request_messages" => {
                vec![ContentBlockOutput::Text(Text {
                    text: json!({
                        "system": request.system,
                        "messages": request.messages,
                    })
                    .to_string(),
                })]
            }
            "extract_images" => {
                let images: Vec<_> = request
                    .messages
                    .iter()
                    .flat_map(|m| {
                        m.content.iter().flat_map(|block| {
                            if let ContentBlock::File(image) = block {
                                Some(image.clone())
                            } else {
                                None
                            }
                        })
                    })
                    .collect();
                vec![ContentBlockOutput::Text(Text {
                    text: serde_json::to_string(&images).map_err(|e| {
                        ErrorDetails::Serialization {
                            message: format!("Failed to serialize collected images: {e:?}"),
                        }
                    })?,
                })]
            }
            "require_pdf" => {
                let files: Vec<_> = request
                    .messages
                    .iter()
                    .flat_map(|m| {
                        m.content.iter().flat_map(|block| {
                            if let ContentBlock::File(file) = block {
                                Some(file.clone())
                            } else {
                                None
                            }
                        })
                    })
                    .collect();
                let mut found_pdf = false;
                for file in &files {
                    let resolved_file = file.resolve().await?;
                    if resolved_file.file.mime_type == mime::APPLICATION_PDF {
                        found_pdf = true;
                    }
                }
                if found_pdf {
                    vec![ContentBlockOutput::Text(Text {
                        text: serde_json::to_string(&files).map_err(|e| {
                            ErrorDetails::Serialization {
                                message: format!("Failed to serialize collected files: {e:?}"),
                            }
                        })?,
                    })]
                } else {
                    return Err(ErrorDetails::InferenceClient {
                        message: "PDF must be provided for require_pdf model".to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                        status_code: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    }
                    .into());
                }
            }
            "llm_judge::true" => vec![r#"{"score": true}"#.to_string().into()],
            "llm_judge::false" => vec![r#"{"score": false}"#.to_string().into()],
            "llm_judge::zero" => vec![r#"{"score": 0}"#.to_string().into()],
            "llm_judge::one" => vec![r#"{"score": 1}"#.to_string().into()],
            "llm_judge::error" => {
                return Err(ErrorDetails::InferenceClient {
                    message: "Dummy error in inference".to_string(),
                    raw_request: Some("raw request".to_string()),
                    raw_response: None,
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                }
                .into());
            }
            _ => vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()],
        };
        let raw_request = DUMMY_RAW_REQUEST.to_string();
        let raw_response = match self.model_name.as_str() {
            #[expect(clippy::unwrap_used)]
            "tool" => serde_json::to_string(&*DUMMY_TOOL_RESPONSE).unwrap(),
            "good_tool" => r#"{"sentiment":"positive","confidence":0.95}"#.to_string(),
            "json" => DUMMY_JSON_RESPONSE_RAW.to_string(),
            "json_goodbye" => DUMMY_JSON_GOODBYE_RESPONSE_RAW.to_string(),
            "json_cot" => DUMMY_JSON_COT_RESPONSE_RAW.to_string(),
            #[expect(clippy::unwrap_used)]
            "bad_tool" => serde_json::to_string(&*DUMMY_BAD_TOOL_RESPONSE).unwrap(),
            "best_of_n_0" => r#"{"thinking": "hmmm", "answer_choice": 0}"#.to_string(),
            "best_of_n_1" => r#"{"thinking": "hmmm", "answer_choice": 1}"#.to_string(),
            "best_of_n_big" => r#"{"thinking": "hmmm", "answer_choice": 100}"#.to_string(),
            _ => DUMMY_INFER_RESPONSE_RAW.to_string(),
        };
        let usage = self.get_model_usage(content.len() as u32);
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let system = request.system.clone();
        let input_messages = request.messages.clone();
        let finish_reason = if self.model_name.contains("tool") || self.model_name == "good_tool" {
            Some(FinishReason::ToolCall)
        } else {
            Some(FinishReason::Stop)
        };
        Ok(ProviderInferenceResponse {
            id,
            created,
            output: content,
            raw_request,
            raw_response,
            usage,
            latency,
            system,
            input_messages,
            finish_reason,
        })
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request: _,
            provider_name: _,
            model_name: _,
            otlp_config: _,
        }: ModelProviderRequest<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
        _model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        if self.model_name == "slow" {
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        // Just so they don't seem like 0ms inferences
        tokio::time::sleep(Duration::from_millis(1)).await;
        // Check for flaky models
        if self.model_name.starts_with("flaky_") {
            #[expect(clippy::expect_used)]
            let mut counters = FLAKY_COUNTERS
                .lock()
                .expect("FLAKY_COUNTERS mutex is poisoned");
            let counter = counters.entry(self.model_name.clone()).or_insert(0);
            *counter += 1;

            // Fail on even-numbered calls
            if counter.is_multiple_of(2) {
                return Err(ErrorDetails::InferenceClient {
                    raw_request: Some("raw request".to_string()),
                    raw_response: None,
                    message: format!(
                        "Flaky model '{}' failed on call number {}",
                        self.model_name, *counter
                    ),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                }
                .into());
            }
        }
        if self.model_name == "reasoner" {
            return self
                .create_streaming_reasoning_response(
                    DUMMY_STREAMING_THINKING.to_vec(),
                    DUMMY_STREAMING_RESPONSE.to_vec(),
                )
                .await;
        }
        if self.model_name == "json_reasoner" {
            return self
                .create_streaming_reasoning_response(
                    DUMMY_STREAMING_THINKING.to_vec(),
                    DUMMY_STREAMING_JSON_RESPONSE.to_vec(),
                )
                .await;
        }
        if self.model_name == "json" {
            return self
                .create_streaming_reasoning_response(vec![], DUMMY_STREAMING_JSON_RESPONSE.to_vec())
                .await;
        }

        if self.model_name.starts_with("error") {
            return Err(ErrorDetails::InferenceClient {
                message: format!(
                    "Error sending request to Dummy provider for model '{}'.",
                    self.model_name
                ),
                raw_request: Some("raw request".to_string()),
                raw_response: None,
                status_code: None,
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }

        let err_in_stream = self.model_name == "err_in_stream";
        let fatal_stream_error = self.model_name == "fatal_stream_error";

        let created = current_timestamp();

        let (content_chunks, is_tool_call) = match self.model_name.as_str() {
            "tool" | "tool_split_name" => (DUMMY_STREAMING_TOOL_RESPONSE.to_vec(), true),
            "good_tool" => (DUMMY_STREAMING_GOOD_TOOL_RESPONSE.to_vec(), true),
            "reasoner" => (DUMMY_STREAMING_RESPONSE.to_vec(), false),
            _ => (DUMMY_STREAMING_RESPONSE.to_vec(), false),
        };

        let content_chunk_len = content_chunks.len();
        let finish_reason = if is_tool_call {
            Some(FinishReason::ToolCall)
        } else {
            Some(FinishReason::Stop)
        };
        let split_tool_name = self.model_name == "tool_split_name";
        let slow_second_chunk = self.model_name == "slow_second_chunk";
        let stream = async_stream::stream! {
            for (i, chunk) in content_chunks.into_iter().enumerate() {
                if slow_second_chunk && i == 1 {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                if fatal_stream_error && i == 2 {
                    yield Err(Error::new(ErrorDetails::FatalStreamError {
                        message: "Dummy fatal error".to_string(),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                    }));
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    continue;
                }
                if err_in_stream && i == 3 {
                    yield Err(Error::new(ErrorDetails::InferenceClient {
                        message: "Dummy error in stream".to_string(),
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                        status_code: None,
                        provider_type: PROVIDER_TYPE.to_string(),
                    }));
                    continue;
                }
                // We want to simulate the tool name being in the first chunk, but not in the subsequent chunks.
                let tool_name = if i == 0 && !split_tool_name {
                    Some("get_temperature".to_string())
                } else if split_tool_name {
                    if i == 0 {
                        Some("get_temp".to_string())
                    } else if i == 1 {
                        Some("erature".to_string())
                    } else {
                        None
                    }
                } else {
                    None
                };
                yield Ok(ProviderInferenceResponseChunk {
                    created,
                    content: vec![if is_tool_call {
                        ContentBlockChunk::ToolCall(ToolCallChunk {
                            id: "0".to_string(),
                            raw_name: tool_name,
                            raw_arguments: chunk.to_string(),
                        })
                    } else {
                        ContentBlockChunk::Text(TextChunk {
                            text: chunk.to_string(),
                            id: "0".to_string(),
                        })
                    }],
                    usage: None,
                    finish_reason: None,
                    raw_response: chunk.to_string(),
                    latency: Duration::from_millis(50 + 10 * (i as u64 + 1)),
                });
            }
        };
        let stream: ProviderInferenceResponseStreamInner = Box::pin(
            stream
                .chain(tokio_stream::once(Ok(ProviderInferenceResponseChunk {
                    created,
                    content: vec![],
                    usage: Some(self.get_model_usage(content_chunk_len as u32)),
                    finish_reason,
                    raw_response: String::new(),
                    latency: Duration::from_millis(50 + 10 * (content_chunk_len as u64)),
                })))
                .throttle(std::time::Duration::from_millis(10)),
        );

        Ok((
            // We need this verbose path to avoid using `tokio_stream::StreamExt::peekable`,
            // which produces a different types
            futures::stream::StreamExt::peekable(stream),
            DUMMY_RAW_REQUEST.to_string(),
        ))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        let file_id = Uuid::now_v7();
        let batch_id = Uuid::now_v7();
        let raw_requests: Vec<String> =
            requests.iter().map(|_| "raw_request".to_string()).collect();
        Ok(StartBatchProviderInferenceResponse {
            batch_id,
            batch_params: json!({"file_id": file_id, "batch_id": batch_id}),
            status: BatchStatus::Pending,
            raw_requests,
            raw_request: "raw request".to_string(),
            raw_response: "raw response".to_string(),
            errors: vec![],
        })
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a TensorzeroHttpClient,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Dummy".to_string(),
        }
        .into())
    }
}
lazy_static! {
    static ref EMPTY_SECRET: SecretString = SecretString::from(String::new());
}

impl EmbeddingProvider for DummyProvider {
    async fn embed(
        &self,
        request: &EmbeddingRequest,
        _http_client: &TensorzeroHttpClient,
        _dynamic_api_keys: &InferenceCredentials,
        _model_provider_data: &EmbeddingProviderRequestInfo,
    ) -> Result<EmbeddingProviderResponse, Error> {
        if self.model_name.starts_with("error") {
            return Err(ErrorDetails::InferenceClient {
                message: format!(
                    "Error sending request to Dummy provider for model '{}'.",
                    self.model_name
                ),
                raw_request: Some("raw request".to_string()),
                raw_response: None,
                status_code: None,
                provider_type: PROVIDER_TYPE.to_string(),
            }
            .into());
        }
        if self.model_name.contains("slow") {
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
        let id = Uuid::now_v7();
        let created = current_timestamp();
        let embeddings = vec![Embedding::Float(vec![0.0; 1536]); request.input.num_inputs()];
        let raw_request = DUMMY_RAW_REQUEST.to_string();
        let raw_response = DUMMY_RAW_REQUEST.to_string();
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(0),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        Ok(EmbeddingProviderResponse {
            id,
            input: request.input.clone(),
            embeddings,
            created,
            raw_request,
            raw_response,
            usage,
            latency,
        })
    }
}
