//! Episode inferences endpoint for listing full inferences by episode ID.

use axum::extract::{Query, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use crate::db::inferences::{
    DEFAULT_INFERENCE_QUERY_LIMIT, InferenceQueries, ListEpisodeInferencesParams, PaginationParams,
};
use crate::error::{Error, ErrorDetails};
use crate::stored_inference::StoredInference;
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the episode_inferences endpoint
#[derive(Debug, Deserialize)]
pub struct EpisodeInferencesQueryParams {
    /// Episode ID to query (required)
    pub episode_id: Uuid,
    /// Cursor to fetch records before this ID (mutually exclusive with `after`)
    pub before: Option<Uuid>,
    /// Cursor to fetch records after this ID (mutually exclusive with `before`)
    pub after: Option<Uuid>,
    /// Maximum number of records to return (default: 20)
    pub limit: Option<u32>,
}

/// Response containing a list of inferences for an episode
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct ListEpisodeInferencesResponse {
    pub inferences: Vec<StoredInference>,
}

/// HTTP handler for listing inferences by episode ID
#[debug_handler(state = AppStateData)]
#[instrument(name = "get_episode_inferences_handler", skip_all)]
pub async fn get_episode_inferences_handler(
    State(app_state): AppState,
    Query(params): Query<EpisodeInferencesQueryParams>,
) -> Result<Json<ListEpisodeInferencesResponse>, Error> {
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

    let list_params = ListEpisodeInferencesParams {
        episode_id: params.episode_id,
        pagination,
        limit: params.limit.unwrap_or(DEFAULT_INFERENCE_QUERY_LIMIT),
    };

    let inferences_db = app_state
        .clickhouse_connection_info
        .list_episode_inferences(&list_params)
        .await?;

    // Convert from StoredInferenceDatabase to StoredInference (API type)
    let inferences: Vec<StoredInference> = inferences_db
        .into_iter()
        .map(|inf| inf.into_stored_inference())
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Json(ListEpisodeInferencesResponse { inferences }))
}
