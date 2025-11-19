//! Embedding types for OpenAI-compatible API.
//!
//! This module provides request and response types for the embeddings endpoint,
//! including parameter structures and conversion logic between OpenAI's embedding
//! format and TensorZero's internal embedding representations.

use serde::{Deserialize, Serialize};

use crate::cache::CacheParamsOptions;
use crate::embeddings::{Embedding, EmbeddingEncodingFormat, EmbeddingInput};
use crate::endpoints::embeddings::{EmbeddingResponse, Params as EmbeddingParams};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;

const TENSORZERO_EMBEDDING_MODEL_NAME_PREFIX: &str = "tensorzero::embedding_model_name::";

#[derive(Debug, Deserialize)]
pub struct OpenAICompatibleEmbeddingParams {
    input: EmbeddingInput,
    model: String,
    dimensions: Option<u32>,
    #[serde(default)]
    encoding_format: EmbeddingEncodingFormat,
    #[serde(default, rename = "tensorzero::credentials")]
    tensorzero_credentials: InferenceCredentials,
    #[serde(rename = "tensorzero::dryrun")]
    tensorzero_dryrun: Option<bool>,
    #[serde(rename = "tensorzero::cache_options")]
    tensorzero_cache_options: Option<CacheParamsOptions>,
}

impl TryFrom<OpenAICompatibleEmbeddingParams> for EmbeddingParams {
    type Error = Error;
    fn try_from(params: OpenAICompatibleEmbeddingParams) -> Result<Self, Self::Error> {
        let model_name = match params
            .model
            .strip_prefix(TENSORZERO_EMBEDDING_MODEL_NAME_PREFIX)
        {
            Some(model_name) => model_name.to_string(),
            None => {
                crate::utils::deprecation_warning("Model names in the OpenAI-compatible embeddings endpoint should be prefixed with 'tensorzero::embedding_model_name::'");
                params.model
            }
        };
        Ok(EmbeddingParams {
            input: params.input,
            model_name,
            dimensions: params.dimensions,
            encoding_format: params.encoding_format,
            credentials: params.tensorzero_credentials,
            dryrun: params.tensorzero_dryrun,
            cache_options: params.tensorzero_cache_options.unwrap_or_default(),
        })
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "object", rename_all = "lowercase")]
pub enum OpenAIEmbeddingResponse {
    List {
        data: Vec<OpenAIEmbedding>,
        model: String,
        usage: Option<OpenAIEmbeddingUsage>,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "object", rename_all = "lowercase")]
pub enum OpenAIEmbedding {
    Embedding { embedding: Embedding, index: usize },
}

#[derive(Debug, Serialize)]
pub struct OpenAIEmbeddingUsage {
    prompt_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

impl From<EmbeddingResponse> for OpenAIEmbeddingResponse {
    fn from(response: EmbeddingResponse) -> Self {
        OpenAIEmbeddingResponse::List {
            data: response
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(i, embedding)| OpenAIEmbedding::Embedding {
                    embedding,
                    index: i,
                })
                .collect(),
            model: format!("{TENSORZERO_EMBEDDING_MODEL_NAME_PREFIX}{}", response.model),
            usage: Some(OpenAIEmbeddingUsage {
                prompt_tokens: response.usage.input_tokens,
                total_tokens: response.usage.input_tokens, // there are no output tokens for embeddings
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_from_embedding_params_deprecated() {
        let logs_contain = crate::utils::testing::capture_logs();
        let openai_embedding_params = OpenAICompatibleEmbeddingParams {
            input: EmbeddingInput::Single("foo".to_string()),
            model: "text-embedding-ada-002".to_string(),
            dimensions: Some(15),
            encoding_format: EmbeddingEncodingFormat::Float,
            tensorzero_credentials: InferenceCredentials::default(),
            tensorzero_dryrun: None,
            tensorzero_cache_options: None,
        };
        let param: EmbeddingParams = openai_embedding_params.try_into().unwrap();
        assert_eq!(param.model_name, "text-embedding-ada-002");
        assert_eq!(param.dimensions, Some(15));
        assert_eq!(param.encoding_format, EmbeddingEncodingFormat::Float);
        assert!(logs_contain("Deprecation warning: Model names in the OpenAI-compatible embeddings endpoint should be prefixed with 'tensorzero::embedding_model_name::'"));
    }

    #[test]
    fn test_try_from_embedding_params_strip() {
        let logs_contain = crate::utils::testing::capture_logs();
        let openai_embedding_params = OpenAICompatibleEmbeddingParams {
            input: EmbeddingInput::Single("foo".to_string()),
            model: "tensorzero::embedding_model_name::text-embedding-ada-002".to_string(),
            dimensions: Some(15),
            encoding_format: EmbeddingEncodingFormat::Float,
            tensorzero_credentials: InferenceCredentials::default(),
            tensorzero_dryrun: None,
            tensorzero_cache_options: None,
        };
        let param: EmbeddingParams = openai_embedding_params.try_into().unwrap();
        assert_eq!(param.model_name, "text-embedding-ada-002");
        assert_eq!(param.dimensions, Some(15));
        assert_eq!(param.encoding_format, EmbeddingEncodingFormat::Float);
        assert!(!logs_contain("Deprecation warning: Model names in the OpenAI-compatible embeddings endpoint should be prefixed with 'tensorzero::embedding_model_name::'"));
    }
}
