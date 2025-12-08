//! Embeddings endpoint handler for OpenAI-compatible API.
//!
//! This module implements the HTTP handler for the `/openai/v1/embeddings` endpoint,
//! providing compatibility with the OpenAI Embeddings API format. It converts between
//! OpenAI's embedding request format and TensorZero's internal embedding system.

use axum::{Extension, Json, debug_handler, extract::State};

use crate::endpoints::embeddings::embeddings;
use crate::error::AxumResponseError;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::types::embeddings::{OpenAICompatibleEmbeddingParams, OpenAIEmbeddingResponse};

#[debug_handler(state = AppStateData)]
pub async fn embeddings_handler(
    State(app_state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleEmbeddingParams>,
) -> Result<Json<OpenAIEmbeddingResponse>, AxumResponseError> {
    async {
        let embedding_params = openai_compatible_params.try_into()?;
        let response = embeddings(
            app_state.config.clone(),
            &app_state.http_client,
            app_state.clickhouse_connection_info.clone(),
            app_state.postgres_connection_info.clone(),
            app_state.deferred_tasks.clone(),
            embedding_params,
            api_key_ext,
        )
        .await?;
        Ok(response.into())
    }
    .await
    .map(Json)
    .map_err(|e| AxumResponseError::new(e, app_state))
}
