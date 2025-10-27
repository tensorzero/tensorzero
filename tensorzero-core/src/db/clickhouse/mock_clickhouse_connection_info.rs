/// A test-only struct that implements all mock ClickHouse queries. This allows us to use a single struct in tests that need to mock multiple traits.
use async_trait::async_trait;

use crate::config::Config;
use crate::db::datasets::{DatapointInsert, DatasetQueries, MockDatasetQueries};
use crate::db::inferences::{InferenceQueries, ListInferencesParams, MockInferenceQueries};
use crate::error::Error;
use crate::stored_inference::StoredInference;

/// Mock struct that implements all traits on ClickHouseConnectionInfo.
/// Usage: in tests, create a new mutable instance of this struct, and use the appropriate expect_ methods on the fields inside to mock
/// the expected method call. This struct will deliegate each method call to the appropriate mock.
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
    ) -> Result<Vec<StoredInference>, Error> {
        self.inference_queries.list_inferences(config, params).await
    }
}

#[async_trait]
impl DatasetQueries for MockClickHouseConnectionInfo {
    async fn count_rows_for_dataset(
        &self,
        params: &crate::db::datasets::DatasetQueryParams,
    ) -> Result<u32, Error> {
        self.dataset_queries.count_rows_for_dataset(params).await
    }

    async fn insert_rows_for_dataset(
        &self,
        params: &crate::db::datasets::DatasetQueryParams,
    ) -> Result<u32, Error> {
        self.dataset_queries.insert_rows_for_dataset(params).await
    }

    async fn get_dataset_rows(
        &self,
        params: &crate::db::datasets::GetDatasetRowsParams,
    ) -> Result<Vec<crate::db::datasets::DatasetDetailRow>, Error> {
        self.dataset_queries.get_dataset_rows(params).await
    }

    async fn get_dataset_metadata(
        &self,
        params: &crate::db::datasets::GetDatasetMetadataParams,
    ) -> Result<Vec<crate::db::datasets::DatasetMetadata>, Error> {
        self.dataset_queries.get_dataset_metadata(params).await
    }

    async fn count_datasets(&self) -> Result<u32, Error> {
        self.dataset_queries.count_datasets().await
    }

    async fn stale_datapoint(
        &self,
        params: &crate::db::datasets::StaleDatapointParams,
    ) -> Result<(), Error> {
        self.dataset_queries.stale_datapoint(params).await
    }

    async fn insert_datapoint(&self, datapoint: &DatapointInsert) -> Result<(), Error> {
        self.dataset_queries.insert_datapoint(datapoint).await
    }

    async fn insert_datapoints(&self, datapoints: &[DatapointInsert]) -> Result<u64, Error> {
        self.dataset_queries.insert_datapoints(datapoints).await
    }

    async fn count_datapoints_for_dataset_function(
        &self,
        params: &crate::db::datasets::CountDatapointsForDatasetFunctionParams,
    ) -> Result<u32, Error> {
        self.dataset_queries
            .count_datapoints_for_dataset_function(params)
            .await
    }

    async fn get_adjacent_datapoint_ids(
        &self,
        params: &crate::db::datasets::GetAdjacentDatapointIdsParams,
    ) -> Result<crate::db::datasets::AdjacentDatapointIds, Error> {
        self.dataset_queries
            .get_adjacent_datapoint_ids(params)
            .await
    }

    async fn get_datapoint(
        &self,
        params: &crate::db::datasets::GetDatapointParams,
    ) -> Result<crate::endpoints::datasets::Datapoint, Error> {
        self.dataset_queries.get_datapoint(params).await
    }

    async fn get_datapoints(
        &self,
        params: &crate::db::datasets::GetDatapointsParams,
    ) -> Result<Vec<crate::endpoints::datasets::Datapoint>, Error> {
        self.dataset_queries.get_datapoints(params).await
    }
}
