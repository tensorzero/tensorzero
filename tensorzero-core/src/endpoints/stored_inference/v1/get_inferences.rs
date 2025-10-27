use axum::extract::State;
use axum::Json;
use tracing::instrument;

use crate::config::Config;
use crate::db::inferences::{InferenceOutputSource, InferenceQueries, ListInferencesParams};
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{GetInferencesRequest, GetInferencesResponse, ListInferencesRequest};

const DEFAULT_PAGE_SIZE: u32 = 20;
const DEFAULT_OFFSET: u32 = 0;

/// Handler for the POST `/v1/inferences/get_inferences` endpoint.
/// Retrieves specific inferences by their IDs.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "inferences.v1.get_inferences", skip(app_state, request))]
pub async fn get_inferences_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<GetInferencesRequest>,
) -> Result<Json<GetInferencesResponse>, Error> {
    let response = get_inferences(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        request,
    )
    .await?;
    Ok(Json(response))
}

async fn get_inferences(
    config: &Config,
    clickhouse: &impl InferenceQueries,
    request: GetInferencesRequest,
) -> Result<GetInferencesResponse, Error> {
    // If no IDs are provided, return an empty response.
    if request.ids.is_empty() {
        return Ok(GetInferencesResponse { inferences: vec![] });
    }

    let params = ListInferencesParams {
        function_name: None,
        ids: Some(&request.ids),
        variant_name: None,
        episode_id: None,
        filters: None,
        // For get by ID, we return the inference output (not demonstration feedback)
        output_source: InferenceOutputSource::Inference,
        // Return all inferences matching the IDs.
        limit: Some(u64::MAX),
        offset: Some(0),
        order_by: None,
    };

    let inferences_storage = clickhouse.list_inferences(config, &params).await?;
    let inferences = inferences_storage
        .into_iter()
        .map(|x| x.into_stored_inference(config))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GetInferencesResponse { inferences })
}

/// Handler for the POST `/v1/inferences/list_inferences` endpoint.
/// Lists inferences with optional filtering, pagination, and sorting.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "inferences.v1.list_inferences", skip(app_state, request))]
pub async fn list_inferences_handler(
    State(app_state): AppState,
    StructuredJson(request): StructuredJson<ListInferencesRequest>,
) -> Result<Json<GetInferencesResponse>, Error> {
    let response = list_inferences(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        request,
    )
    .await?;

    Ok(Json(response))
}

async fn list_inferences(
    config: &Config,
    clickhouse: &impl InferenceQueries,
    request: ListInferencesRequest,
) -> Result<GetInferencesResponse, Error> {
    let page_size = request.page_size.unwrap_or(DEFAULT_PAGE_SIZE) as u64;
    let offset = request.offset.unwrap_or(DEFAULT_OFFSET) as u64;

    let params = ListInferencesParams {
        function_name: request.function_name.as_deref(),
        ids: None, // List all inferences, not filtering by ID
        variant_name: request.variant_name.as_deref(),
        episode_id: request.episode_id.as_ref(),
        filters: request.filter.as_ref(),
        output_source: request.output_source,
        limit: Some(page_size),
        offset: Some(offset),
        order_by: request.order_by.as_deref(),
    };

    let inferences_storage = clickhouse.list_inferences(config, &params).await?;
    let inferences = inferences_storage
        .into_iter()
        .map(|x| x.into_stored_inference(config))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GetInferencesResponse { inferences })
}
