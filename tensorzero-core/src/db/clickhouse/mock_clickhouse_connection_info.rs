/// A test-only struct that implements all mock ClickHouse queries. This allows us to use a single struct in tests that need to mock multiple traits.
use async_trait::async_trait;
use uuid::Uuid;

use crate::config::Config;
use crate::db::datasets::{
    AdjacentDatapointIds, CountDatapointsForDatasetFunctionParams, DatapointInsert,
    DatasetDetailRow, DatasetMetadata, DatasetQueries, DatasetQueryParams,
    GetAdjacentDatapointIdsParams, GetDatapointParams, GetDatapointsParams,
    GetDatasetMetadataParams, GetDatasetRowsParams, MockDatasetQueries, StaleDatapointParams,
};
use crate::db::inferences::{InferenceQueries, ListInferencesParams, MockInferenceQueries};
use crate::endpoints::datasets::StoredDatapoint;
use crate::error::Error;
use crate::stored_inference::StoredInferenceDatabase;

/// Mock struct that implements all traits on ClickHouseConnectionInfo.
/// Usage: in tests, create a new mutable instance of this struct, and use the appropriate expect_ methods on the fields inside to mock
/// the expected method call. This struct will delegate each method call to the appropriate mock.
/// Example:
/// ```
/// let mut mock_clickhouse = MockClickHouseConnectionInfo::new();
/// mock_clickhouse.inference_queries.expect_list_inferences().times(1).returning(move |_, _| {
///     Box::pin(async move { Ok(vec![inference]) })
/// });
/// ```
pub(crate) struct MockClickHouseConnectionInfo {
    pub(crate) inference_queries: MockInferenceQueries,
    pub(crate) dataset_queries: MockDatasetQueries,
}

impl MockClickHouseConnectionInfo {
    pub fn new() -> Self {
        Self {
            inference_queries: MockInferenceQueries::new(),
            dataset_queries: MockDatasetQueries::new(),
        }
    }
}

#[async_trait]
impl InferenceQueries for MockClickHouseConnectionInfo {
    async fn list_inferences(
        &self,
        config: &Config,
        params: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInferenceDatabase>, Error> {
        self.inference_queries.list_inferences(config, params).await
    }
}

#[async_trait]
impl DatasetQueries for MockClickHouseConnectionInfo {
    async fn count_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error> {
        self.dataset_queries.count_rows_for_dataset(params).await
    }

    async fn insert_rows_for_dataset(&self, params: &DatasetQueryParams) -> Result<u32, Error> {
        self.dataset_queries.insert_rows_for_dataset(params).await
    }

    async fn get_dataset_rows(
        &self,
        params: &GetDatasetRowsParams,
    ) -> Result<Vec<DatasetDetailRow>, Error> {
        self.dataset_queries.get_dataset_rows(params).await
    }

    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error> {
        self.dataset_queries.get_dataset_metadata(params).await
    }

    async fn count_datasets(&self) -> Result<u32, Error> {
        self.dataset_queries.count_datasets().await
    }

    async fn stale_datapoint(&self, params: &StaleDatapointParams) -> Result<(), Error> {
        self.dataset_queries.stale_datapoint(params).await
    }

    async fn insert_datapoints(&self, datapoints: &[DatapointInsert]) -> Result<u64, Error> {
        self.dataset_queries.insert_datapoints(datapoints).await
    }

    async fn count_datapoints_for_dataset_function(
        &self,
        params: &CountDatapointsForDatasetFunctionParams,
    ) -> Result<u32, Error> {
        self.dataset_queries
            .count_datapoints_for_dataset_function(params)
            .await
    }

    async fn get_adjacent_datapoint_ids(
        &self,
        params: &GetAdjacentDatapointIdsParams,
    ) -> Result<AdjacentDatapointIds, Error> {
        self.dataset_queries
            .get_adjacent_datapoint_ids(params)
            .await
    }

    async fn get_datapoint(&self, params: &GetDatapointParams) -> Result<StoredDatapoint, Error> {
        self.dataset_queries.get_datapoint(params).await
    }

    async fn get_datapoints(
        &self,
        params: &GetDatapointsParams,
    ) -> Result<Vec<StoredDatapoint>, Error> {
        self.dataset_queries.get_datapoints(params).await
    }

    async fn delete_datapoints(
        &self,
        dataset_name: &str,
        datapoint_ids: Option<&[Uuid]>,
    ) -> Result<u64, Error> {
        self.dataset_queries
            .delete_datapoints(dataset_name, datapoint_ids)
            .await
    }
}
