use axum::extract::{Path, State};
use axum::Json;
use tracing::instrument;

use crate::config::Config;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::datasets::DatasetQueries;
use crate::db::inferences::{InferenceOutputSource, InferenceQueries, ListInferencesParams};
use crate::endpoints::datasets::v1::types::CreateDatapointsFromInferenceOutputSource;
use crate::endpoints::datasets::validate_dataset_name;
use crate::error::Error;
use crate::stored_inference::StoredInference;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{
    CreateDatapointResult, CreateDatapointsFromInferenceRequest,
    CreateDatapointsFromInferenceRequestParams, CreateDatapointsFromInferenceResponse,
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

    // Convert inferences to datapoints
    let mut results = Vec::new();
    let mut datapoints_to_insert = Vec::new();

    for inference in inferences {
        let inference_id = match &inference {
            StoredInference::Chat(chat) => chat.inference_id,
            StoredInference::Json(json) => json.inference_id,
        };

        // Convert the inference to a datapoint
        let (datapoint_id, datapoint_insert) =
            inference.to_datapoint_insert(&dataset_name, &request.output_source);
        datapoints_to_insert.push(datapoint_insert);
        results.push(CreateDatapointResult::Success {
            id: datapoint_id,
            inference_id,
        });
    }

    // Batch insert all successful datapoints
    if !datapoints_to_insert.is_empty() {
        clickhouse.insert_datapoints(&datapoints_to_insert).await?;
    }

    Ok(CreateDatapointsFromInferenceResponse {
        datapoints: results,
    })
}
