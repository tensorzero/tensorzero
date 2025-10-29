use axum::extract::{Path, State};
use axum::Json;
use futures::future::try_join_all;
use tracing::instrument;
use uuid::Uuid;

use crate::config::Config;
use crate::db::datasets::{DatapointInsert, DatasetQueries};
use crate::endpoints::datasets::validate_dataset_name;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::FetchContext;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    CreateDatapointRequest, CreateDatapointsRequest, CreateDatapointsResponse,
};

/// Handler for the POST `/v1/datasets/{dataset_id}/datapoints` endpoint.
/// Creates manual datapoints in a dataset.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.create_datapoints", skip(app_state, request))]
pub async fn create_datapoints_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<CreateDatapointsRequest>,
) -> Result<Json<CreateDatapointsResponse>, Error> {
    let response = create_datapoints(&app_state, &dataset_name, request).await?;
    Ok(Json(response))
}

/// Business logic for creating datapoints manually in a dataset.
/// This function validates the request, converts inputs, validates schemas,
/// and inserts the new datapoints into ClickHouse.
///
/// Returns an error if there are no datapoints, or if validation fails.
pub async fn create_datapoints(
    app_state: &AppStateData,
    dataset_name: &str,
    request: CreateDatapointsRequest,
) -> Result<CreateDatapointsResponse, Error> {
    create_datapoints_impl(
        &app_state.config,
        &app_state.http_client,
        &app_state.clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await
}

/// Internal implementation that can be called from legacy code.
pub async fn create_datapoints_impl(
    config: &Config,
    http_client: &crate::http::TensorzeroHttpClient,
    clickhouse: &crate::db::clickhouse::ClickHouseConnectionInfo,
    dataset_name: &str,
    request: CreateDatapointsRequest,
) -> Result<CreateDatapointsResponse, Error> {
    validate_dataset_name(dataset_name)?;

    if request.datapoints.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "At least one datapoint must be provided".to_string(),
        }));
    }

    let fetch_context = FetchContext {
        client: http_client,
        object_store_info: &config.object_store_info,
    };

    // Prepare all datapoints in parallel
    let prepare_futures = request.datapoints.into_iter().map(|datapoint_request| async {
        let result: Result<(DatapointInsert, Uuid), Error> = match datapoint_request {
            CreateDatapointRequest::Chat(chat_request) => {
                let (insert, id) = chat_request.prepare(config, &fetch_context, dataset_name).await?;
                Ok((DatapointInsert::Chat(insert), id))
            }
            CreateDatapointRequest::Json(json_request) => {
                let (insert, id) = json_request.prepare(config, &fetch_context, dataset_name).await?;
                Ok((DatapointInsert::Json(insert), id))
            }
        };
        result
    });

    let results: Vec<(DatapointInsert, Uuid)> = try_join_all(prepare_futures).await?;

    // Separate inserts and IDs
    let (datapoints_to_insert, ids): (Vec<DatapointInsert>, Vec<Uuid>) = results.into_iter().unzip();

    // Insert all datapoints
    clickhouse.insert_datapoints(&datapoints_to_insert).await?;

    Ok(CreateDatapointsResponse { ids })
}
