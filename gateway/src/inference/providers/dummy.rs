use std::time::Duration;

use tokio_stream::StreamExt;
use uuid::Uuid;

use super::provider_trait::InferenceProvider;

use crate::error::Error;
use crate::inference::types::{
    InferenceResponseStream, Latency, ModelInferenceRequest, ModelInferenceResponse,
    ModelInferenceResponseChunk, Usage,
};
use crate::model::ProviderConfig;

pub struct DummyProvider;

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
pub static DUMMY_JSON_RESPONSE_RAW: &str = r#"{"answer":"Hello"}"#;
pub static DUMMY_INFER_USAGE: Usage = Usage {
    prompt_tokens: 10,
    completion_tokens: 10,
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

impl InferenceProvider for DummyProvider {
    async fn infer<'a>(
        _request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        _http_client: &'a reqwest::Client,
    ) -> Result<ModelInferenceResponse, Error> {
        let model_name = match model {
            ProviderConfig::Dummy { model_name } => model_name,
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected Dummy provider config.".to_string(),
                })
            }
        };
        if model_name == "error" {
            return Err(Error::InferenceClient {
                message: "Error sending request to Dummy provider.".to_string(),
            });
        }
        let id = Uuid::now_v7();
        #[allow(clippy::expect_used)]
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        let content = match model_name.as_str() {
            "json" => Some(r#"{"answer":"Hello"}"#.to_string()),
            _ => Some(DUMMY_INFER_RESPONSE_CONTENT.to_string()),
        };
        let raw = match model_name.as_str() {
            "json" => DUMMY_JSON_RESPONSE_RAW.to_string(),
            _ => DUMMY_INFER_RESPONSE_RAW.to_string(),
        };
        let usage = DUMMY_INFER_USAGE.clone();
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        Ok(ModelInferenceResponse {
            id,
            created,
            content,
            tool_calls: None,
            raw_response: raw,
            usage,
            latency,
        })
    }

    async fn infer_stream<'a>(
        _request: &'a ModelInferenceRequest<'a>,
        model: &'a ProviderConfig,
        _http_client: &'a reqwest::Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let model_name = match model {
            ProviderConfig::Dummy { model_name } => model_name,
            _ => {
                return Err(Error::InvalidProviderConfig {
                    message: "Expected Dummy provider config.".to_string(),
                })
            }
        };
        if model_name == "error" {
            return Err(Error::InferenceClient {
                message: "Error sending request to Dummy provider.".to_string(),
            });
        }
        let id = Uuid::now_v7();
        #[allow(clippy::expect_used)]
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();

        let content_chunks = DUMMY_STREAMING_RESPONSE.to_vec();

        let total_tokens = content_chunks.len() as u32;

        let initial_chunk = ModelInferenceResponseChunk {
            inference_id: id,
            created,
            content: Some(content_chunks[0].to_string()),
            tool_calls: None,
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(100),
        };
        let content_chunk_len = content_chunks.len();
        let stream = tokio_stream::iter(content_chunks.into_iter().skip(1).enumerate())
            .map(move |(i, chunk)| {
                Ok(ModelInferenceResponseChunk {
                    inference_id: id,
                    created,
                    content: Some(chunk.to_string()),
                    tool_calls: None,
                    usage: None,
                    raw_response: "".to_string(),
                    latency: Duration::from_millis(50 + 10 * (i as u64 + 1)),
                })
            })
            .chain(tokio_stream::once(Ok(ModelInferenceResponseChunk {
                inference_id: id,
                created,
                content: None,
                tool_calls: None,
                usage: Some(crate::inference::types::Usage {
                    prompt_tokens: 10,
                    completion_tokens: total_tokens,
                }),
                raw_response: "".to_string(),
                latency: Duration::from_millis(50 + 10 * (content_chunk_len as u64)),
            })))
            .throttle(std::time::Duration::from_millis(10));

        Ok((initial_chunk, Box::pin(stream)))
    }
}
