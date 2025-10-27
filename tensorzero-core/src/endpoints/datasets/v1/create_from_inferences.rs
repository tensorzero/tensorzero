use std::collections::HashSet;

use axum::extract::{Path, State};
use axum::Json;
use tracing::instrument;

use crate::config::Config;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::datasets::DatasetQueries;
use crate::db::inferences::{InferenceOutputSource, InferenceQueries, ListInferencesParams};
use crate::endpoints::datasets::v1::types::CreateDatapointsFromInferenceOutputSource;
use crate::endpoints::datasets::validate_dataset_name;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    CreateDatapointsFromInferenceRequest, CreateDatapointsFromInferenceRequestParams,
    CreateDatapointsFromInferenceResponse,
};

/// Handler for the POST `/v1/datasets/{dataset_id}/from_inferences` endpoint.
/// Creates datapoints from inferences based on either specific inference IDs or an inference query.
#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.create_from_inferences", skip(app_state, request))]
pub async fn create_from_inferences_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<CreateDatapointsFromInferenceRequest>,
) -> Result<Json<CreateDatapointsFromInferenceResponse>, Error> {
    let response = create_from_inferences(
        &app_state.config,
        &app_state.clickhouse_connection_info,
        dataset_name,
        request,
    )
    .await?;

    Ok(Json(response))
}

async fn create_from_inferences(
    config: &Config,
    clickhouse: &ClickHouseConnectionInfo,
    dataset_name: String,
    request: CreateDatapointsFromInferenceRequest,
) -> Result<CreateDatapointsFromInferenceResponse, Error> {
    validate_dataset_name(&dataset_name)?;

    let output_source = match request.output_source {
        Some(CreateDatapointsFromInferenceOutputSource::None) => {
            // If we are not including any output in the datapoints, we use Inference for the query to
            // avoid doing a join with the DemonstrationFeedback table. Then, we will drop it when constructing the datapoints.
            InferenceOutputSource::Inference
        }
        Some(CreateDatapointsFromInferenceOutputSource::Inference) => {
            InferenceOutputSource::Inference
        }
        Some(CreateDatapointsFromInferenceOutputSource::Demonstration) => {
            InferenceOutputSource::Demonstration
        }
        None => InferenceOutputSource::Inference,
    };

    let list_inferences_params = match &request.params {
        CreateDatapointsFromInferenceRequestParams::InferenceIds { inference_ids } => {
            ListInferencesParams {
                ids: Some(inference_ids),
                output_source,
                ..Default::default()
            }
        }
        CreateDatapointsFromInferenceRequestParams::InferenceQuery {
            function_name,
            variant_name,
            filters,
        } => ListInferencesParams {
            function_name: Some(function_name),
            variant_name: variant_name.as_deref(),
            filters: filters.as_ref(),
            output_source,
            ..Default::default()
        },
    };
    let inferences = clickhouse
        .list_inferences(config, &list_inferences_params)
        .await?;

    if let CreateDatapointsFromInferenceRequestParams::InferenceIds {
        inference_ids: request_inference_ids,
    } = &request.params
    {
        // Check if all inferences are found. If not, we fail early without creating any datapoints for a pseudo-transactional behavior.
        let found_inference_ids = inferences
            .iter()
            .map(|inference| inference.id())
            .collect::<HashSet<_>>();
        for inference_id in request_inference_ids {
            if !found_inference_ids.contains(&inference_id) {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Inference {inference_id} not found"),
                }));
            }
        }
    }

    // Convert inferences to datapoints
    let mut ids = Vec::new();
    let mut datapoints_to_insert = Vec::new();

    for inference in inferences {
        let datapoint_insert =
            inference.into_datapoint_insert(&dataset_name, &request.output_source);
        ids.push(datapoint_insert.id());
        datapoints_to_insert.push(datapoint_insert);
    }

    // Batch insert all datapoints
    if !datapoints_to_insert.is_empty() {
        clickhouse.insert_datapoints(&datapoints_to_insert).await?;
    }

    Ok(CreateDatapointsFromInferenceResponse { ids })
}
