/// A test-only struct that implements all mock ClickHouse queries. This allows us to use a single struct in tests that need to mock multiple traits.
use async_trait::async_trait;
use uuid::Uuid;

use crate::config::Config;
use crate::config::MetricConfigLevel;
use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
use crate::db::datasets::{
    DatasetMetadata, DatasetQueries, GetDatapointParams, GetDatapointsParams,
    GetDatasetMetadataParams, MockDatasetQueries,
};
use crate::db::inference_count::{
    CountByVariant, CountInferencesParams, CountInferencesWithDemonstrationFeedbacksParams,
    CountInferencesWithFeedbackParams, FunctionInferenceCount,
    GetFunctionThroughputByVariantParams, InferenceCountQueries, MockInferenceCountQueries,
    VariantThroughput,
};
use crate::db::inferences::{
    CountInferencesParams as ListInferencesCountParams, FunctionInfo, InferenceMetadata,
    InferenceQueries, ListInferenceMetadataParams, ListInferencesParams, MockInferenceQueries,
};
use crate::db::model_inferences::{MockModelInferenceQueries, ModelInferenceQueries};
use crate::db::stored_datapoint::StoredDatapoint;
use crate::db::{ConfigQueries, MockConfigQueries};
use crate::error::Error;
use crate::inference::types::StoredModelInference;
use crate::stored_inference::StoredInferenceDatabase;
use crate::tool::ToolCallConfigDatabaseInsert;
use serde_json::Value;

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
    pub(crate) config_queries: MockConfigQueries,
    pub(crate) model_inference_queries: MockModelInferenceQueries,
    pub(crate) inference_count_queries: MockInferenceCountQueries,
}

impl MockClickHouseConnectionInfo {
    pub fn new() -> Self {
        Self {
            inference_queries: MockInferenceQueries::new(),
            dataset_queries: MockDatasetQueries::new(),
            config_queries: MockConfigQueries::new(),
            model_inference_queries: MockModelInferenceQueries::new(),
            inference_count_queries: MockInferenceCountQueries::new(),
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

    async fn list_inference_metadata(
        &self,
        params: &ListInferenceMetadataParams,
    ) -> Result<Vec<InferenceMetadata>, Error> {
        self.inference_queries.list_inference_metadata(params).await
    }

    async fn count_inferences(
        &self,
        config: &Config,
        params: &ListInferencesCountParams<'_>,
    ) -> Result<u64, Error> {
        self.inference_queries
            .count_inferences(config, params)
            .await
    }

    async fn get_function_info(
        &self,
        target_id: &Uuid,
        level: MetricConfigLevel,
    ) -> Result<Option<FunctionInfo>, Error> {
        self.inference_queries
            .get_function_info(target_id, level)
            .await
    }

    async fn get_chat_inference_tool_params(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<ToolCallConfigDatabaseInsert>, Error> {
        self.inference_queries
            .get_chat_inference_tool_params(function_name, inference_id)
            .await
    }

    async fn get_json_inference_output_schema(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<Value>, Error> {
        self.inference_queries
            .get_json_inference_output_schema(function_name, inference_id)
            .await
    }

    async fn get_inference_output(
        &self,
        function_info: &FunctionInfo,
        inference_id: Uuid,
    ) -> Result<Option<String>, Error> {
        self.inference_queries
            .get_inference_output(function_info, inference_id)
            .await
    }
}

#[async_trait]
impl DatasetQueries for MockClickHouseConnectionInfo {
    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error> {
        self.dataset_queries.get_dataset_metadata(params).await
    }

    async fn insert_datapoints(&self, datapoints: &[StoredDatapoint]) -> Result<u64, Error> {
        self.dataset_queries.insert_datapoints(datapoints).await
    }

    async fn count_datapoints_for_dataset(
        &self,
        dataset_name: &str,
        function_name: Option<&str>,
    ) -> Result<u64, Error> {
        self.dataset_queries
            .count_datapoints_for_dataset(dataset_name, function_name)
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

    async fn clone_datapoints(
        &self,
        target_dataset_name: &str,
        source_datapoint_ids: &[Uuid],
    ) -> Result<Vec<Option<Uuid>>, Error> {
        self.dataset_queries
            .clone_datapoints(target_dataset_name, source_datapoint_ids)
            .await
    }
}

impl ConfigQueries for MockClickHouseConnectionInfo {
    async fn get_config_snapshot(
        &self,
        snapshot_hash: SnapshotHash,
    ) -> Result<ConfigSnapshot, Error> {
        self.config_queries.get_config_snapshot(snapshot_hash).await
    }
}

#[async_trait]
impl ModelInferenceQueries for MockClickHouseConnectionInfo {
    async fn get_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<StoredModelInference>, Error> {
        self.model_inference_queries
            .get_model_inferences_by_inference_id(inference_id)
            .await
    }
}

#[async_trait]
impl InferenceCountQueries for MockClickHouseConnectionInfo {
    async fn count_inferences_for_function(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        self.inference_count_queries
            .count_inferences_for_function(params)
            .await
    }

    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error> {
        self.inference_count_queries
            .count_inferences_by_variant(params)
            .await
    }

    async fn count_inferences_with_feedback(
        &self,
        params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error> {
        self.inference_count_queries
            .count_inferences_with_feedback(params)
            .await
    }

    async fn count_inferences_with_demonstration_feedback(
        &self,
        params: CountInferencesWithDemonstrationFeedbacksParams<'_>,
    ) -> Result<u64, Error> {
        self.inference_count_queries
            .count_inferences_with_demonstration_feedback(params)
            .await
    }

    async fn count_inferences_for_episode(&self, episode_id: uuid::Uuid) -> Result<u64, Error> {
        self.inference_count_queries
            .count_inferences_for_episode(episode_id)
            .await
    }

    async fn get_function_throughput_by_variant(
        &self,
        params: GetFunctionThroughputByVariantParams<'_>,
    ) -> Result<Vec<VariantThroughput>, Error> {
        self.inference_count_queries
            .get_function_throughput_by_variant(params)
            .await
    }

    async fn list_functions_with_inference_count(
        &self,
    ) -> Result<Vec<FunctionInferenceCount>, Error> {
        self.inference_count_queries
            .list_functions_with_inference_count()
            .await
    }
}
