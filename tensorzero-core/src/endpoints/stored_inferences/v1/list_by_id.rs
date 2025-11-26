/// Internal handler for UI that paginates inferences by ID.
use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use tracing::instrument;
use uuid::Uuid;

use crate::db::inferences::{InferenceQueries, ListInferencesByIdParams, PaginateByIdCondition};
use crate::endpoints::stored_inferences::v1::types::{
    InternalInferenceMetadata, InternalListInferencesByIdResponse,
};
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData};

/// Query parameters for the internal list inferences by ID endpoint.
/// Used by the `GET /internal/inferences` endpoint.
#[derive(Debug, Deserialize)]
pub struct InternalListInferencesByIdQueryParams {
    /// Maximum number of inferences to return.
    pub limit: u32,

    /// Optional inference ID to paginate before.
    pub before: Option<Uuid>,

    /// Optional inference ID to paginate after.
    pub after: Option<Uuid>,

    /// Optional function name to filter inferences by.
    pub function_name: Option<String>,

    /// Optional variant name to filter inferences by.
    pub variant_name: Option<String>,

    /// Optional episode ID to filter inferences by.
    pub episode_id: Option<Uuid>,
}

/// Handler for the GET `/internal/inferences` endpoint.
/// Lists inferences with pagination by ID.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "internal.inferences.list_by_id", skip(app_state))]
pub async fn list_inferences_by_id_handler(
    State(app_state): AppState,
    Query(query_params): Query<InternalListInferencesByIdQueryParams>,
) -> Result<Json<InternalListInferencesByIdResponse>, Error> {
    let pagination = match (query_params.before, query_params.after) {
        (Some(_), Some(_)) => {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message: "Cannot specify both before and after parameters".to_string(),
            }));
        }
        (Some(before), None) => Some(PaginateByIdCondition::Before { id: before }),
        (None, Some(after)) => Some(PaginateByIdCondition::After { id: after }),
        _ => None,
    };

    let params = ListInferencesByIdParams {
        limit: query_params.limit,
        function_name: query_params.function_name,
        variant_name: query_params.variant_name,
        episode_id: query_params.episode_id,
        pagination,
    };

    let inferences = app_state
        .clickhouse_connection_info
        .list_inferences_by_id(params)
        .await?;
    let response = InternalListInferencesByIdResponse {
        inferences: inferences
            .into_iter()
            .map(|inference| InternalInferenceMetadata {
                id: inference.id,
                function_name: inference.function_name,
                variant_name: inference.variant_name,
                episode_id: inference.episode_id,
                function_type: inference.function_type,
                timestamp: inference.timestamp,
            })
            .collect(),
    };

    Ok(Json(response))
}
