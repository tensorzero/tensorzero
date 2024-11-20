use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use lazy_static::lazy_static;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::StreamExt;
use uuid::Uuid;

use super::provider_trait::{HasCredentials, InferenceProvider};

use crate::embeddings::{EmbeddingProvider, EmbeddingProviderResponse, EmbeddingRequest};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{
    current_timestamp, ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequest,
    ProviderInferenceResponse, ProviderInferenceResponseChunk, ProviderInferenceResponseStream,
    Usage,
};
use crate::model::ProviderCredentials;
use crate::tool::{ToolCall, ToolCallChunk};

#[derive(Debug, Default)]
pub struct DummyProvider {
    pub model_name: String,
    pub dynamic_credentials: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DummyCredentials<'a> {
    pub api_key: Cow<'a, SecretString>,
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

pub static DUMMY_RAW_REQUEST: &str = "raw request";

impl InferenceProvider for DummyProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
        api_key: ProviderCredentials<'a>,
    ) -> Result<ProviderInferenceResponse, Error> {
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
                    message: format!(
                        "Flaky model '{}' failed on call number {}",
                        self.model_name, *counter
                    ),
                }
                .into());
            }
        }

        if self.model_name == "error" {
            return Err(ErrorDetails::InferenceClient {
                message: "Error sending request to Dummy provider.".to_string(),
            }
            .into());
        }
        let api_key = match &api_key {
            ProviderCredentials::Dummy(credentials) => &credentials.api_key,
            _ => {
                return Err(ErrorDetails::BadCredentialsPreInference {
                    provider_name: "Dummy".to_string(),
                }
                .into());
            }
        };
        if self.model_name == "test_key" && api_key.expose_secret() != "good_key" {
            return Err(ErrorDetails::InferenceClient {
                message: "Invalid API key for Dummy provider".to_string(),
            }
            .into());
        }
        let id = Uuid::now_v7();
        #[allow(clippy::expect_used)]
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        let content = match self.model_name.as_str() {
            "tool" => vec![ContentBlock::ToolCall(ToolCall {
                name: "get_temperature".to_string(),
                #[allow(clippy::unwrap_used)]
                arguments: serde_json::to_string(&*DUMMY_TOOL_RESPONSE).unwrap(),
                id: "0".to_string(),
            })],
            "bad_tool" => vec![ContentBlock::ToolCall(ToolCall {
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
            "alternate" => vec![ALTERNATE_INFER_RESPONSE_CONTENT.to_string().into()],
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
        let usage = DUMMY_INFER_USAGE.clone();
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let system = request.system.clone();
        let input_messages = request.messages.clone();
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
        })
    }

    async fn infer_stream<'a>(
        &'a self,
        _request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
        _api_key: ProviderCredentials<'a>,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
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
                    message: format!(
                        "Flaky model '{}' failed on call number {}",
                        self.model_name, *counter
                    ),
                }
                .into());
            }
        }

        if self.model_name == "error" {
            return Err(ErrorDetails::InferenceClient {
                message: "Error sending request to Dummy provider.".to_string(),
            }
            .into());
        }
        let id = Uuid::now_v7();
        #[allow(clippy::expect_used)]
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();

        let (content_chunks, is_tool_call) = if self.model_name == "tool" {
            (DUMMY_STREAMING_TOOL_RESPONSE.to_vec(), true)
        } else {
            (DUMMY_STREAMING_RESPONSE.to_vec(), false)
        };

        let total_tokens = content_chunks.len() as u32;

        let initial_chunk = ProviderInferenceResponseChunk {
            inference_id: id,
            created,
            content: vec![if is_tool_call {
                ContentBlockChunk::ToolCall(ToolCallChunk {
                    id: "0".to_string(),
                    raw_name: "get_temperature".to_string(),
                    raw_arguments: content_chunks[0].to_string(),
                })
            } else {
                ContentBlockChunk::Text(crate::inference::types::TextChunk {
                    text: content_chunks[0].to_string(),
                    id: "0".to_string(),
                })
            }],
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(100),
        };
        let content_chunk_len = content_chunks.len();
        let stream = tokio_stream::iter(content_chunks.into_iter().skip(1).enumerate())
            .map(move |(i, chunk)| {
                Ok(ProviderInferenceResponseChunk {
                    inference_id: id,
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
                    raw_response: "".to_string(),
                    latency: Duration::from_millis(50 + 10 * (i as u64 + 1)),
                })
            })
            .chain(tokio_stream::once(Ok(ProviderInferenceResponseChunk {
                inference_id: id,
                created,
                content: vec![],
                usage: Some(crate::inference::types::Usage {
                    input_tokens: 10,
                    output_tokens: total_tokens,
                }),
                raw_response: "".to_string(),
                latency: Duration::from_millis(50 + 10 * (content_chunk_len as u64)),
            })))
            .throttle(std::time::Duration::from_millis(10));

        Ok((
            initial_chunk,
            Box::pin(stream),
            DUMMY_RAW_REQUEST.to_string(),
        ))
    }
}
lazy_static! {
    static ref EMPTY_SECRET: SecretString = SecretString::from(String::new());
}

impl HasCredentials for DummyProvider {
    fn has_credentials(&self) -> bool {
        self.model_name != "bad_credentials"
    }

    fn get_credentials<'a>(
        &'a self,
        credentials: &'a InferenceCredentials,
    ) -> Result<ProviderCredentials<'a>, Error> {
        if self.dynamic_credentials {
            match &credentials.dummy {
                Some(credentials) => Ok(ProviderCredentials::Dummy(Cow::Borrowed(credentials))),
                None => Err(ErrorDetails::ApiKeyMissing {
                    provider_name: "Dummy".to_string(),
                }
                .into()),
            }
        } else {
            match &credentials.dummy {
                Some(_credentials) => Err(ErrorDetails::UnexpectedDynamicCredentials {
                    provider_name: "Dummy".to_string(),
                }
                .into()),
                None => Ok(ProviderCredentials::Dummy(Cow::Owned(DummyCredentials {
                    api_key: Cow::Borrowed(&EMPTY_SECRET),
                }))),
            }
        }
    }
}

impl EmbeddingProvider for DummyProvider {
    async fn embed(
        &self,
        request: &EmbeddingRequest,
        _http_client: &reqwest::Client,
    ) -> Result<EmbeddingProviderResponse, Error> {
        if self.model_name == "error" {
            return Err(ErrorDetails::InferenceClient {
                message: "Error sending request to Dummy provider.".to_string(),
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
