use std::future::Future;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::credentials::ModelProviderRequestInfo;
use crate::extra_body::ExtraBodyConfig;
use crate::extra_headers::ExtraHeadersConfig;
use crate::{EmbeddingEncodingFormat, Latency, RawUsageEntry, Usage};
use tensorzero_error::Error;
use tensorzero_http::TensorzeroHttpClient;
use tensorzero_types::inference_params::InferenceCredentials;

// =============================================================================
// Embedding
// =============================================================================

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Embedding {
    Float(Vec<f32>),
    Base64(String),
}

impl Embedding {
    pub fn as_float(&self) -> Option<&Vec<f32>> {
        match self {
            Embedding::Float(vec) => Some(vec),
            Embedding::Base64(_) => None,
        }
    }

    pub fn ndims(&self) -> usize {
        match self {
            Embedding::Float(vec) => vec.len(),
            Embedding::Base64(encoded) => encoded.len() * 3 / 16,
        }
    }
}

// =============================================================================
// EmbeddingInput
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
    SingleTokens(Vec<u32>),
    BatchTokens(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    pub fn num_inputs(&self) -> usize {
        match self {
            EmbeddingInput::Single(_) => 1,
            EmbeddingInput::Batch(texts) => texts.len(),
            EmbeddingInput::SingleTokens(_) => 1,
            EmbeddingInput::BatchTokens(tokens) => tokens.len(),
        }
    }

    pub fn first(&self) -> Option<&String> {
        match self {
            EmbeddingInput::Single(text) => Some(text),
            EmbeddingInput::Batch(texts) => texts.first(),
            EmbeddingInput::SingleTokens(_) => None,
            EmbeddingInput::BatchTokens(_) => None,
        }
    }

    pub fn estimated_input_token_usage(&self) -> u64 {
        match self {
            EmbeddingInput::Single(text) => (text.len() as u64) / 2,
            EmbeddingInput::Batch(texts) => texts.iter().map(|t| (t.len() as u64) / 2).sum(),
            EmbeddingInput::SingleTokens(tokens) => tokens.len() as u64,
            EmbeddingInput::BatchTokens(token_arrays) => {
                token_arrays.iter().map(|t| t.len() as u64).sum()
            }
        }
    }
}

impl From<String> for EmbeddingInput {
    fn from(text: String) -> Self {
        EmbeddingInput::Single(text)
    }
}

// =============================================================================
// EmbeddingRequest, EmbeddingProviderRequest
// =============================================================================

#[derive(Debug, PartialEq, Serialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    pub dimensions: Option<u32>,
    pub encoding_format: EmbeddingEncodingFormat,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingProviderRequest<'request> {
    pub request: &'request EmbeddingRequest,
    pub model_name: &'request str,
    pub provider_name: &'request str,
}

// =============================================================================
// EmbeddingProviderResponse
// =============================================================================

#[derive(Debug, PartialEq)]
pub struct EmbeddingProviderResponse {
    pub id: Uuid,
    pub input: EmbeddingInput,
    pub embeddings: Vec<Embedding>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
    pub raw_usage: Option<Vec<RawUsageEntry>>,
}

impl EmbeddingProviderResponse {
    #[expect(clippy::missing_panics_doc)]
    pub fn new(
        embeddings: Vec<Embedding>,
        input: EmbeddingInput,
        raw_request: String,
        raw_response: String,
        usage: Usage,
        latency: Latency,
        raw_usage: Option<Vec<RawUsageEntry>>,
    ) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        #[expect(clippy::expect_used)]
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        Self {
            id: Uuid::now_v7(),
            input,
            embeddings,
            created,
            raw_request,
            raw_response,
            usage,
            latency,
            raw_usage,
        }
    }
}

// =============================================================================
// EmbeddingProviderRequestInfo
// =============================================================================

#[derive(Clone, Debug)]
pub struct EmbeddingProviderRequestInfo {
    pub provider_name: Arc<str>,
    pub extra_body: Option<ExtraBodyConfig>,
    pub extra_headers: Option<ExtraHeadersConfig>,
}

impl From<&EmbeddingProviderRequestInfo> for ModelProviderRequestInfo {
    fn from(val: &EmbeddingProviderRequestInfo) -> Self {
        ModelProviderRequestInfo {
            provider_name: val.provider_name.clone(),
            extra_headers: val.extra_headers.clone(),
            extra_body: val.extra_body.clone(),
            discard_unknown_chunks: false,
        }
    }
}

// =============================================================================
// EmbeddingProvider trait
// =============================================================================

pub trait EmbeddingProvider {
    fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &TensorzeroHttpClient,
        dynamic_api_keys: &InferenceCredentials,
        model_provider_data: &EmbeddingProviderRequestInfo,
    ) -> impl Future<Output = Result<EmbeddingProviderResponse, Error>> + Send;
}
