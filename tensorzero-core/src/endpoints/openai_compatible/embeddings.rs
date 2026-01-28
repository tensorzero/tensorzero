//! Embeddings endpoint handler for OpenAI-compatible API.
//!
//! This module implements the HTTP handler for the `/openai/v1/embeddings` endpoint,
//! providing compatibility with the OpenAI Embeddings API format. It converts between
//! OpenAI's embedding request format and TensorZero's internal embedding system.

use axum::{Extension, Json, extract::State};

use crate::endpoints::embeddings::embeddings;
use crate::utils::gateway::{AppState, AppStateData};
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::types::embeddings::{OpenAICompatibleEmbeddingParams, OpenAIEmbeddingResponse};
use super::{OpenAICompatibleError, OpenAIStructuredJson};

pub async fn embeddings_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        rate_limiting_manager,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    OpenAIStructuredJson(openai_compatible_params): OpenAIStructuredJson<
        OpenAICompatibleEmbeddingParams,
    >,
) -> Result<Json<OpenAIEmbeddingResponse>, OpenAICompatibleError> {
    let embedding_params = openai_compatible_params.try_into()?;
    let response = embeddings(
        config,
        &http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        rate_limiting_manager,
        embedding_params,
        api_key_ext,
    )
    .await?;
    Ok(Json(response.into()))
}
