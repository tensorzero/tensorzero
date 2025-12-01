/// A test-only struct that implements all mock ClickHouse queries. This allows us to use a single struct in tests that need to mock multiple traits.
use async_trait::async_trait;
use uuid::Uuid;

use crate::config::Config;
use crate::db::datasets::{
    CountDatapointsForDatasetFunctionParams, DatapointInsert, DatasetMetadata, DatasetQueries,
    DatasetQueryParams, GetDatapointParams, GetDatapointsParams, GetDatasetMetadataParams,
    MockDatasetQueries,
};
use crate::db::inferences::{
    GetInferenceBoundsParams, InferenceBounds, InferenceMetadata, InferenceQueries,
    ListInferencesByIdParams, ListInferencesParams, MockInferenceQueries,
};
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

    async fn get_inference_bounds(
        &self,
        params: GetInferenceBoundsParams,
    ) -> Result<InferenceBounds, Error> {
        self.inference_queries.get_inference_bounds(params).await
    }

    async fn list_inferences_by_id(
        &self,
        params: ListInferencesByIdParams,
    ) -> Result<Vec<InferenceMetadata>, Error> {
        self.inference_queries.list_inferences_by_id(params).await
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

    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error> {
        self.dataset_queries.get_dataset_metadata(params).await
    }

    async fn count_datasets(&self) -> Result<u32, Error> {
        self.dataset_queries.count_datasets().await
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
