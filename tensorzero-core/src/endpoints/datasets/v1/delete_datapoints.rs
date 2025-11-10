use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;
use tracing::instrument;

use crate::db::datasets::DatasetQueries;
use crate::endpoints::datasets::validate_dataset_name;
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

use super::types::{DeleteDatapointsRequest, DeleteDatapointsResponse};

#[derive(Debug, Deserialize)]
pub struct DeleteDatapointsPathParams {
    pub dataset_name: String,
}

#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.delete_datapoints", skip(app_state, request))]
pub async fn delete_datapoints_handler(
    State(app_state): AppState,
    Path(path_params): Path<DeleteDatapointsPathParams>,
    StructuredJson(request): StructuredJson<DeleteDatapointsRequest>,
) -> Result<Json<DeleteDatapointsResponse>, Error> {
    let response = delete_datapoints(
        &app_state.clickhouse_connection_info,
        &path_params.dataset_name,
        request,
    )
    .await?;
    Ok(Json(response))
}

#[axum::debug_handler(state = AppStateData)]
#[instrument(name = "datasets.v1.delete_dataset", skip(app_state))]
pub async fn delete_dataset_handler(
    State(app_state): AppState,
    Path(path_params): Path<DeleteDatapointsPathParams>,
) -> Result<Json<DeleteDatapointsResponse>, Error> {
    let response = delete_dataset(
        &app_state.clickhouse_connection_info,
        &path_params.dataset_name,
    )
    .await?;
    Ok(Json(response))
}

/// Business logic for deleting an entire dataset.
/// This function stales all datapoints in the dataset without fetching them.
///
/// Returns the number of deleted datapoints, or an error if the dataset name is invalid.
pub async fn delete_dataset(
    clickhouse: &impl DatasetQueries,
    dataset_name: &str,
) -> Result<DeleteDatapointsResponse, Error> {
    validate_dataset_name(dataset_name)?;

    let num_deleted_datapoints = clickhouse.delete_datapoints(dataset_name, None).await?;

    Ok(DeleteDatapointsResponse {
        num_deleted_datapoints,
    })
}

/// Business logic for deleting datapoints from a dataset.
/// This function validates the request and stales the datapoints.
///
/// Returns the number of deleted datapoints, or an error if there are no datapoints or if the dataset name is invalid.
pub async fn delete_datapoints(
    clickhouse: &impl DatasetQueries,
    dataset_name: &str,
    request: DeleteDatapointsRequest,
) -> Result<DeleteDatapointsResponse, Error> {
    validate_dataset_name(dataset_name)?;
    if request.ids.is_empty() {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: "ids must be a non-empty list".to_string(),
        }));
    }

    let num_deleted_datapoints = clickhouse
        .delete_datapoints(dataset_name, Some(request.ids.as_slice()))
        .await?;

    Ok(DeleteDatapointsResponse {
        num_deleted_datapoints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::datasets::MockDatasetQueries;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_delete_datapoints_calls_clickhouse_with_correct_params() {
        let mut mock_clickhouse = MockDatasetQueries::new();
        let dataset_name = "test_dataset";
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let ids = vec![id1, id2];

        // Set up expectation
        mock_clickhouse
            .expect_delete_datapoints()
            .withf(move |name, input_ids| {
                name == dataset_name && input_ids.unwrap() == ids.as_slice()
            })
            .times(1)
            .returning(|_, _| Box::pin(async { Ok(2) }));

        let request = DeleteDatapointsRequest {
            ids: vec![id1, id2],
        };

        let result = delete_datapoints(&mock_clickhouse, dataset_name, request).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().num_deleted_datapoints, 2);
    }

    #[tokio::test]
    async fn test_delete_dataset_calls_clickhouse_with_none_ids() {
        let mut mock_clickhouse = MockDatasetQueries::new();
        let dataset_name = "test_dataset";

        // Set up expectation - delete_dataset should call with None
        mock_clickhouse
            .expect_delete_datapoints()
            .withf(move |name, ids| name == dataset_name && ids.is_none())
            .times(1)
            .returning(|_, _| Box::pin(async { Ok(5) }));

        let result = delete_dataset(&mock_clickhouse, dataset_name).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().num_deleted_datapoints, 5);
    }

    #[tokio::test]
    async fn test_delete_datapoints_invalid_dataset_name() {
        let mock_clickhouse = MockDatasetQueries::new();
        let invalid_name = "tensorzero::dataset";

        let request = DeleteDatapointsRequest {
            ids: vec![Uuid::now_v7()],
        };

        let result = delete_datapoints(&mock_clickhouse, invalid_name, request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid dataset name"));
    }

    #[tokio::test]
    async fn test_delete_dataset_invalid_dataset_name() {
        let mock_clickhouse = MockDatasetQueries::new();
        let invalid_name = "tensorzero::dataset";

        let result = delete_dataset(&mock_clickhouse, invalid_name).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid dataset name"));
    }
}
