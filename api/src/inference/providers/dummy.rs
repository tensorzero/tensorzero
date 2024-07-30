use crate::{
    error::Error,
    inference::types::{
        InferenceResponseStream, ModelInferenceRequest, ModelInferenceResponse,
        ModelInferenceResponseChunk,
    },
    model::ProviderConfig,
};
use tokio_stream::StreamExt;
use uuid::Uuid;

use super::provider_trait::InferenceProvider;

pub struct DummyProvider;

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
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        let content = Some("Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.".to_string());
        let raw = r#"{
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
}"#.to_string();
        let usage = crate::inference::types::Usage {
            prompt_tokens: 10,
            completion_tokens: 10,
        };
        Ok(ModelInferenceResponse {
            id,
            created,
            content,
            tool_calls: None,
            raw,
            usage,
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
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();

        let content_chunks = vec![
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

        let total_tokens = content_chunks.len() as u32;

        let initial_chunk = ModelInferenceResponseChunk {
            inference_id: id,
            created,
            content: Some(content_chunks[0].to_string()),
            tool_calls: None,
            usage: None,
            raw: "".to_string(),
        };
        let stream = tokio_stream::iter(content_chunks.into_iter().skip(1))
            .map(move |chunk| {
                Ok(ModelInferenceResponseChunk {
                    inference_id: id,
                    created,
                    content: Some(chunk.to_string()),
                    tool_calls: None,
                    usage: None,
                    raw: "".to_string(),
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
                raw: "".to_string(),
            })))
            .throttle(std::time::Duration::from_millis(10));

        Ok((initial_chunk, Box::pin(stream)))
    }
}
