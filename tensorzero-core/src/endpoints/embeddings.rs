use std::sync::Arc;

use serde::Deserialize;
use tracing::instrument;

use crate::{
    cache::CacheParamsOptions,
    config::Config,
    db::clickhouse::ClickHouseConnectionInfo,
    embeddings::{Embedding, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingRequest},
    endpoints::inference::InferenceClients,
    error::{Error, ErrorDetails},
    inference::types::Usage,
};

use super::inference::InferenceCredentials;

#[derive(Debug, Clone, Deserialize)]
pub struct Params {
    pub input: EmbeddingInput,
    pub model_name: String,
    pub dimensions: Option<u32>,
    pub encoding_format: EmbeddingEncodingFormat,
    // if true, the embedding will not be stored
    pub dryrun: Option<bool>,
    #[serde(default)]
    pub credentials: InferenceCredentials,
    #[serde(default)]
    pub cache_options: CacheParamsOptions,
}

#[instrument(
    name = "embeddings",
    skip(config, http_client, params),
    fields(model, num_inputs)
)]
pub async fn embeddings(
    config: Arc<Config>,
    http_client: &reqwest::Client,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    params: Params,
) -> Result<EmbeddingResponse, Error> {
    let span = tracing::Span::current();
    span.record("model", &params.model_name);
    span.record("num_inputs", params.input.num_inputs());
    let embedding_model = config
        .embedding_models
        .get(&params.model_name)
        .await?
        .ok_or(Error::new(ErrorDetails::ModelNotFound {
            model_name: params.model_name.clone(),
        }))?;
    if let EmbeddingInput::Batch(array) = &params.input {
        if array.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Input cannot be empty".to_string(),
            }));
        }
    }

    let request = EmbeddingRequest {
        input: params.input,
        dimensions: params.dimensions,
        encoding_format: params.encoding_format,
    };
    let dryrun = params.dryrun.unwrap_or(false);
    let clients = InferenceClients {
        http_client,
        credentials: &params.credentials,
        cache_options: &(params.cache_options, dryrun).into(),
        clickhouse_connection_info: &clickhouse_connection_info,
    };
    let response = embedding_model
        .embed(&request, &params.model_name, &clients)
        .await?;
    let usage = response.usage_considering_cached();
    Ok(EmbeddingResponse {
        embeddings: response.embeddings,
        usage,
        model: params.model_name,
    })
}

pub struct EmbeddingResponse {
    pub embeddings: Vec<Embedding>,
    pub usage: Usage,
    pub model: String,
}
