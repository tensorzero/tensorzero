//! Embeddings endpoint handler for OpenAI-compatible API.
//!
//! This module implements the HTTP handler for the `/openai/v1/embeddings` endpoint,
//! providing compatibility with the OpenAI Embeddings API format. It converts between
//! OpenAI's embedding request format and TensorZero's internal embedding system.

use axum::{extract::State, Extension, Json};

use crate::endpoints::embeddings::embeddings;
use crate::endpoints::RequestApiKeyExtension;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::embeddings::{OpenAICompatibleEmbeddingParams, OpenAIEmbeddingResponse};

pub async fn embeddings_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleEmbeddingParams>,
) -> Result<Json<OpenAIEmbeddingResponse>, Error> {
    let embedding_params = openai_compatible_params.try_into()?;
    let response = embeddings(
        config,
        &http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        embedding_params,
        api_key_ext,
    )
    .await?;
    Ok(Json(response.into()))
}
