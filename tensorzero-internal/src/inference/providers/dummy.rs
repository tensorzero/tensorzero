use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use lazy_static::lazy_static;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};
use tokio_stream::StreamExt;
use uuid::Uuid;

use super::provider_trait::InferenceProvider;

use crate::cache::ModelProviderRequest;
use crate::embeddings::{EmbeddingProvider, EmbeddingProviderResponse, EmbeddingRequest};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::batch::{BatchRequestRow, BatchStatus};
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, current_timestamp, ContentBlockChunk,
    ContentBlockOutput, Latency, ModelInferenceRequest, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, Usage,
};
use crate::inference::types::{ContentBlock, FinishReason, ProviderInferenceResponseStreamInner};
use crate::inference::types::{Text, TextChunk, Thought, ThoughtChunk};
use crate::model::{CredentialLocation, ModelProvider};
use crate::tool::{ToolCall, ToolCallChunk};

const PROVIDER_NAME: &str = "Dummy";
const PROVIDER_TYPE: &str = "dummy";

#[derive(Debug, Default)]
pub struct DummyProvider {
    pub model_name: String,
    pub credentials: DummyCredentials,
}

impl DummyProvider {
    pub fn new(
        model_name: String,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let api_key_location = api_key_location.unwrap_or(default_api_key_location());
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
    ) -> Result<Option<&'a SecretString>, Error> {
        match self {
            DummyCredentials::None => Ok(None),
            DummyCredentials::Dynamic(key_name) => {
                Some(dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
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
pub static DUMMY_INFER_USAGE: Usage = Usage {
    input_tokens: 10,
    output_tokens: 10,
};
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

pub static DUMMY_STREAMING_JSON_RESPONSE: [&str; 5] =
    [r#"{"name""#, r#":"John""#, r#","age""#, r#":30"#, r#"}"#];

pub static DUMMY_RAW_REQUEST: &str = "raw request";

impl InferenceProvider for DummyProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name: _,
        }: ModelProviderRequest<'a>,
        _http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        _model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        if self.model_name == "slow" {
            tokio::time::sleep(Duration::from_secs(5)).await;
        }

        // Check for flaky models
        if self.model_name.starts_with("flaky_") {
            #[allow(clippy::expect_used)]
            let mut counters = FLAKY_COUNTERS
                .lock()
                .expect("FLAKY_COUNTERS mutex is poisoned");
            let counter = counters.entry(self.model_name.clone()).or_insert(0);
            *counter += 1;

            // Fail on even-numbered calls
            if *counter % 2 == 0 {
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

        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
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
            "tool" => vec![ContentBlockOutput::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                #[allow(clippy::unwrap_used)]
                arguments: serde_json::to_string(&*DUMMY_TOOL_RESPONSE).unwrap(),
                id: "0".to_string(),
            })],
            "reasoner" => vec![
                ContentBlockOutput::Thought(Thought {
                    text: "hmmm".to_string(),
                    signature: None,
                }),
                ContentBlockOutput::Text(Text {
                    text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
                }),
            ],
            "json_reasoner" => vec![
                ContentBlockOutput::Thought(Thought {
                    text: "hmmm".to_string(),
                    signature: None,
                }),
                ContentBlockOutput::Text(Text {
                    text: DUMMY_JSON_RESPONSE_RAW.to_string(),
                }),
            ],
            "bad_tool" => vec![ContentBlockOutput::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                #[allow(clippy::unwrap_used)]
                arguments: serde_json::to_string(&*DUMMY_BAD_TOOL_RESPONSE).unwrap(),
                id: "0".to_string(),
            })],
            "json" => vec![DUMMY_JSON_RESPONSE_RAW.to_string().into()],
            "json_goodbye" => vec![DUMMY_JSON_GOODBYE_RESPONSE_RAW.to_string().into()],
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
            #[allow(clippy::unwrap_used)]
            "echo_request_messages" => vec![ContentBlockOutput::Text(Text {
                text: serde_json::to_string(&json!({
                    "system": request.system,
                    "messages": request.messages,
                }))
                .unwrap(),
            })],
            "extract_images" => {
                let images: Vec<_> = request
                    .messages
                    .iter()
                    .flat_map(|m| {
                        m.content.iter().flat_map(|block| {
                            if let ContentBlock::Image(image) = block {
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
            "llm_judge::true" => vec![r#"{"thinking": "hmmm", "score": true}"#.to_string().into()],
            "llm_judge::false" => {
                vec![r#"{"thinking": "hmmm", "score": false}"#.to_string().into()]
            }
            "llm_judge::zero" => vec![r#"{"thinking": "hmmm", "score": 0}"#.to_string().into()],
            "llm_judge::one" => {
                vec![r#"{"thinking": "hmmm", "score": 1}"#.to_string().into()]
            }
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
            #[allow(clippy::unwrap_used)]
            "tool" => serde_json::to_string(&*DUMMY_TOOL_RESPONSE).unwrap(),
            #[allow(clippy::unwrap_used)]
            "json" => DUMMY_JSON_RESPONSE_RAW.to_string(),
            #[allow(clippy::unwrap_used)]
            "json_goodbye" => DUMMY_JSON_GOODBYE_RESPONSE_RAW.to_string(),
            #[allow(clippy::unwrap_used)]
            "bad_tool" => serde_json::to_string(&*DUMMY_BAD_TOOL_RESPONSE).unwrap(),
            "best_of_n_0" => r#"{"thinking": "hmmm", "answer_choice": 0}"#.to_string(),
            "best_of_n_1" => r#"{"thinking": "hmmm", "answer_choice": 1}"#.to_string(),
            "best_of_n_big" => r#"{"thinking": "hmmm", "answer_choice": 100}"#.to_string(),
            _ => DUMMY_INFER_RESPONSE_RAW.to_string(),
        };
        let usage = match self.model_name.as_str() {
            "input_tokens_zero" => Usage {
                input_tokens: 0,
                output_tokens: 10,
            },
            "output_tokens_zero" => Usage {
                input_tokens: 10,
                output_tokens: 0,
            },
            "input_tokens_output_tokens_zero" => Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
            _ => DUMMY_INFER_USAGE.clone(),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let system = request.system.clone();
        let input_messages = request.messages.clone();
        let finish_reason = match self.model_name.contains("tool") {
            true => Some(FinishReason::ToolCall),
            false => Some(FinishReason::Stop),
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
        }: ModelProviderRequest<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
        _model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        if self.model_name == "slow" {
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        // Check for flaky models
        if self.model_name.starts_with("flaky_") {
            #[allow(clippy::expect_used)]
            let mut counters = FLAKY_COUNTERS
                .lock()
                .expect("FLAKY_COUNTERS mutex is poisoned");
            let counter = counters.entry(self.model_name.clone()).or_insert(0);
            *counter += 1;

            // Fail on even-numbered calls
            if *counter % 2 == 0 {
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
            return create_streaming_reasoning_response(
                DUMMY_STREAMING_THINKING.to_vec(),
                DUMMY_STREAMING_RESPONSE.to_vec(),
            )
            .await;
        }
        if self.model_name == "json_reasoner" {
            return create_streaming_reasoning_response(
                DUMMY_STREAMING_THINKING.to_vec(),
                DUMMY_STREAMING_JSON_RESPONSE.to_vec(),
            )
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

        let created = current_timestamp();

        let (content_chunks, is_tool_call) = match self.model_name.as_str() {
            "tool" => (DUMMY_STREAMING_TOOL_RESPONSE.to_vec(), true),
            "reasoner" => (DUMMY_STREAMING_RESPONSE.to_vec(), false),
            _ => (DUMMY_STREAMING_RESPONSE.to_vec(), false),
        };

        let total_tokens = content_chunks.len() as u32;
        let content_chunk_len = content_chunks.len();
        let finish_reason = match is_tool_call {
            true => Some(FinishReason::ToolCall),
            false => Some(FinishReason::Stop),
        };
        let stream: ProviderInferenceResponseStreamInner = Box::pin(
            tokio_stream::iter(content_chunks.into_iter().enumerate())
                .map(move |(i, chunk)| {
                    if err_in_stream && i == 3 {
                        return Err(Error::new(ErrorDetails::InferenceClient {
                            message: "Dummy error in stream".to_string(),
                            raw_request: Some("raw request".to_string()),
                            raw_response: None,
                            status_code: None,
                            provider_type: PROVIDER_TYPE.to_string(),
                        }));
                    }
                    Ok(ProviderInferenceResponseChunk {
                        created,
                        content: vec![if is_tool_call {
                            ContentBlockChunk::ToolCall(ToolCallChunk {
                                id: "0".to_string(),
                                raw_name: "get_temperature".to_string(),
                                raw_arguments: chunk.to_string(),
                            })
                        } else {
                            ContentBlockChunk::Text(crate::inference::types::TextChunk {
                                text: chunk.to_string(),
                                id: "0".to_string(),
                            })
                        }],
                        usage: None,
                        finish_reason: None,
                        raw_response: chunk.to_string(),
                        latency: Duration::from_millis(50 + 10 * (i as u64 + 1)),
                    })
                })
                .chain(tokio_stream::once(Ok(ProviderInferenceResponseChunk {
                    created,
                    content: vec![],
                    usage: Some(crate::inference::types::Usage {
                        input_tokens: 10,
                        output_tokens: total_tokens,
                    }),
                    finish_reason,
                    raw_response: "".to_string(),
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
        _client: &'a reqwest::Client,
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
        _http_client: &'a reqwest::Client,
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
        _http_client: &reqwest::Client,
        _dynamic_api_keys: &InferenceCredentials,
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
        let id = Uuid::now_v7();
        let created = current_timestamp();
        let embedding = vec![0.0; 1536];
        let raw_request = DUMMY_RAW_REQUEST.to_string();
        let raw_response = DUMMY_RAW_REQUEST.to_string();
        let usage = DUMMY_INFER_USAGE.clone();
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        Ok(EmbeddingProviderResponse {
            id,
            input: request.input.to_string(),
            embedding,
            created,
            raw_request,
            raw_response,
            usage,
            latency,
        })
    }
}

async fn create_streaming_reasoning_response(
    thinking_chunks: Vec<&'static str>,
    response_chunks: Vec<&'static str>,
) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
    let thinking_chunks = thinking_chunks.into_iter().map(|chunk| {
        ContentBlockChunk::Thought(ThoughtChunk {
            text: Some(chunk.to_string()),
            signature: None,
            id: "0".to_string(),
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
    let stream = tokio_stream::iter(chained.enumerate())
        .map(move |(i, chunk)| {
            Ok(ProviderInferenceResponseChunk {
                created,
                content: vec![chunk],
                usage: None,
                raw_response: "".to_string(),
                latency: Duration::from_millis(50 + 10 * (i as u64 + 1)),
                finish_reason: None,
            })
        })
        .chain(tokio_stream::once(Ok(ProviderInferenceResponseChunk {
            created,
            content: vec![],
            usage: Some(crate::inference::types::Usage {
                input_tokens: 10,
                output_tokens: 10,
            }),
            finish_reason: Some(FinishReason::Stop),
            raw_response: "".to_string(),
            latency: Duration::from_millis(50 + 10 * (num_chunks as u64)),
        })))
        .throttle(std::time::Duration::from_millis(10));

    Ok((
        futures::stream::StreamExt::peekable(Box::pin(stream)),
        DUMMY_RAW_REQUEST.to_string(),
    ))
}
