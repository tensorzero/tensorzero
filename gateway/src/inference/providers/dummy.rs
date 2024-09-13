use std::time::Duration;

use lazy_static::lazy_static;
use serde_json::{json, Value};
use tokio_stream::StreamExt;
use uuid::Uuid;

use super::provider_trait::InferenceProvider;

use crate::error::Error;
use crate::inference::types::{
    ContentBlock, ContentBlockChunk, Latency, ModelInferenceRequest, ProviderInferenceResponse,
    ProviderInferenceResponseChunk, ProviderInferenceResponseStream, Usage,
};
use crate::tool::{ToolCall, ToolCallChunk};

#[derive(Debug)]
pub struct DummyProvider {
    pub model_name: String,
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

lazy_static! {
    pub static ref DUMMY_TOOL_RESPONSE: Value = json!({"location": "Brooklyn", "units": "celsius"});
    // This is the same as DUMMY_TOOL_RESPONSE, but with the units capitalized
    // Since that field is an enum, this should fail validation
    pub static ref DUMMY_BAD_TOOL_RESPONSE: Value = json!({"location": "Brooklyn", "units": "Celsius"});
}
pub static DUMMY_JSON_RESPONSE_RAW: &str = r#"{"answer":"Hello"}"#;
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
        _request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
    ) -> Result<ProviderInferenceResponse, Error> {
        if self.model_name == "error" {
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
            "json" => vec![r#"{"answer":"Hello"}"#.to_string().into()],
            _ => vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()],
        };
        let raw_request = DUMMY_RAW_REQUEST.to_string();
        let raw_response = match self.model_name.as_str() {
            #[allow(clippy::unwrap_used)]
            "tool" => serde_json::to_string(&*DUMMY_TOOL_RESPONSE).unwrap(),
            #[allow(clippy::unwrap_used)]
            "json" => DUMMY_JSON_RESPONSE_RAW.to_string(),
            #[allow(clippy::unwrap_used)]
            "bad_tool" => serde_json::to_string(&*DUMMY_BAD_TOOL_RESPONSE).unwrap(),
            _ => DUMMY_INFER_RESPONSE_RAW.to_string(),
        };
        let usage = DUMMY_INFER_USAGE.clone();
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        Ok(ProviderInferenceResponse {
            id,
            created,
            content,
            raw_request,
            raw_response,
            usage,
            latency,
        })
    }

    async fn infer_stream<'a>(
        &'a self,
        _request: &'a ModelInferenceRequest<'a>,
        _http_client: &'a reqwest::Client,
    ) -> Result<
        (
            ProviderInferenceResponseChunk,
            ProviderInferenceResponseStream,
            String,
        ),
        Error,
    > {
        if self.model_name == "error" {
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

    fn has_credentials(&self) -> bool {
        self.model_name != "bad_credentials"
    }
}
