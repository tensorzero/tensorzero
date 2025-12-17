//! Endpoint for inserting inferences matching dataset query parameters into a dataset.
//!
//! This endpoint is used by the dataset builder to insert inferences matching the filter
//! criteria into a dataset.

use axum::extract::{Path, State};
use axum::{Json, debug_handler};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::db::datasets::{DatasetQueries, DatasetQueryParams};
use crate::endpoints::datasets::internal::types::FilterInferencesForDatasetBuilderRequest;
use crate::error::Error;
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};

/// Response containing the number of rows inserted
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct InsertFromMatchingInferencesResponse {
    /// The number of rows inserted into the dataset
    pub rows_inserted: u32,
}

/// Inserts inferences matching the provided query parameters into a dataset
pub async fn insert_from_matching_inferences(
    clickhouse: &impl DatasetQueries,
    params: DatasetQueryParams,
) -> Result<InsertFromMatchingInferencesResponse, Error> {
    let rows_inserted = clickhouse.insert_rows_for_dataset(&params).await?;

    Ok(InsertFromMatchingInferencesResponse { rows_inserted })
}

/// HTTP handler for the insert from matching inferences endpoint
#[debug_handler(state = AppStateData)]
#[instrument(
    name = "insert_from_matching_inferences_handler",
    skip_all,
    fields(
        dataset_name = %dataset_name,
        inference_type = ?request.inference_type,
        function_name = ?request.function_name,
    )
)]
pub async fn insert_from_matching_inferences_handler(
    State(app_state): AppState,
    Path(dataset_name): Path<String>,
    StructuredJson(request): StructuredJson<FilterInferencesForDatasetBuilderRequest>,
) -> Result<Json<InsertFromMatchingInferencesResponse>, Error> {
    let params = DatasetQueryParams {
        inference_type: request.inference_type,
        function_name: request.function_name,
        dataset_name: Some(dataset_name),
        variant_name: request.variant_name,
        extra_where: None,
        extra_params: None,
        metric_filter: request.metric_filter,
        output_source: request.output_source,
        limit: None,
        offset: None,
    };
    let response =
        insert_from_matching_inferences(&app_state.clickhouse_connection_info, params).await?;

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::clickhouse::MockClickHouseConnectionInfo;
    use crate::db::datasets::DatasetOutputSource;
    use crate::endpoints::datasets::DatapointKind;

    #[tokio::test]
    async fn test_insert_from_matching_inferences() {
        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();

        mock_clickhouse
            .dataset_queries
            .expect_insert_rows_for_dataset()
            .times(1)
            .returning(move |params| {
                assert_eq!(params.function_name.as_deref(), Some("test_function"));
                assert_eq!(params.dataset_name.as_deref(), Some("test_dataset"));
                Box::pin(async move { Ok(42) })
            });

        let params = DatasetQueryParams {
            inference_type: DatapointKind::Chat,
            function_name: Some("test_function".to_string()),
            dataset_name: Some("test_dataset".to_string()),
            variant_name: None,
            extra_where: None,
            extra_params: None,
            metric_filter: None,
            output_source: DatasetOutputSource::Inference,
            limit: None,
            offset: None,
        };

        let result = insert_from_matching_inferences(&mock_clickhouse, params).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.rows_inserted, 42);
    }

    #[tokio::test]
    async fn test_insert_from_matching_inferences_with_variant() {
        let mut mock_clickhouse = MockClickHouseConnectionInfo::new();

        mock_clickhouse
            .dataset_queries
            .expect_insert_rows_for_dataset()
            .times(1)
            .returning(move |params| {
                assert_eq!(params.function_name.as_deref(), Some("json_function"));
                assert_eq!(params.variant_name.as_deref(), Some("variant_a"));
                assert_eq!(params.dataset_name.as_deref(), Some("json_dataset"));
                Box::pin(async move { Ok(10) })
            });

        let params = DatasetQueryParams {
            inference_type: DatapointKind::Json,
            function_name: Some("json_function".to_string()),
            dataset_name: Some("json_dataset".to_string()),
            variant_name: Some("variant_a".to_string()),
            extra_where: None,
            extra_params: None,
            metric_filter: None,
            output_source: DatasetOutputSource::None,
            limit: None,
            offset: None,
        };

        let result = insert_from_matching_inferences(&mock_clickhouse, params).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.rows_inserted, 10);
    }
}
