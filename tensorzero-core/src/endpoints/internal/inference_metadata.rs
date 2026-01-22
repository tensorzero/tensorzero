//! Inference metadata endpoint for listing inference metadata from the InferenceById table.

use axum::extract::{Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::inferences::{
    DEFAULT_INFERENCE_QUERY_LIMIT, InferenceMetadata, InferenceQueries,
    ListInferenceMetadataParams, PaginationParams,
};
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the inference_metadata endpoint
#[derive(Debug, Deserialize)]
pub struct InferenceMetadataQueryParams {
    /// Cursor to fetch records before this ID (mutually exclusive with `after`)
    pub before: Option<Uuid>,
    /// Cursor to fetch records after this ID (mutually exclusive with `before`)
    pub after: Option<Uuid>,
    /// Maximum number of records to return (default: 20)
    pub limit: Option<u32>,
    /// Filter by function name
    pub function_name: Option<String>,
    /// Filter by variant name
    pub variant_name: Option<String>,
    /// Filter by episode ID
    pub episode_id: Option<Uuid>,
}

/// Response containing a list of inference metadata
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct ListInferenceMetadataResponse {
    pub inference_metadata: Vec<InferenceMetadata>,
}

/// HTTP handler for listing inference metadata
#[debug_handler(state = AppStateData)]
#[instrument(name = "get_inference_metadata_handler", skip_all)]
pub async fn get_inference_metadata_handler(
    State(app_state): AppState,
    Query(params): Query<InferenceMetadataQueryParams>,
) -> Result<Json<ListInferenceMetadataResponse>, Error> {
    let pagination = match (params.before, params.after) {
        (Some(id), None) => Some(PaginationParams::Before { id }),
        (None, Some(id)) => Some(PaginationParams::After { id }),
        (None, None) => None,
        (Some(_), Some(_)) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot specify both 'before' and 'after' parameters".to_string(),
            }));
        }
    };

    let list_params = ListInferenceMetadataParams {
        pagination,
        limit: params.limit.unwrap_or(DEFAULT_INFERENCE_QUERY_LIMIT),
        function_name: params.function_name,
        variant_name: params.variant_name,
        episode_id: params.episode_id,
    };

    let inference_metadata = app_state
        .clickhouse_connection_info
        .list_inference_metadata(&list_params)
        .await?;

    Ok(Json(ListInferenceMetadataResponse { inference_metadata }))
}
