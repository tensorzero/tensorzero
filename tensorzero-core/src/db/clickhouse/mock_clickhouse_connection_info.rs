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
use crate::db::feedback::{
    BooleanMetricFeedbackInsert, CommentFeedbackInsert, CumulativeFeedbackTimeSeriesPoint,
    DemonstrationFeedbackInsert, DemonstrationFeedbackRow, FeedbackBounds, FeedbackByVariant,
    FeedbackQueries, FeedbackRow, FloatMetricFeedbackInsert, GetVariantPerformanceParams,
    LatestFeedbackRow, MetricWithFeedback, MockFeedbackQueries,
    StaticEvaluationHumanFeedbackInsert, VariantPerformanceRow,
};
use crate::db::inferences::{
    CountByVariant, CountInferencesForFunctionParams, CountInferencesParams,
    CountInferencesWithFeedbackParams, FunctionInferenceCount, FunctionInfo,
    GetFunctionThroughputByVariantParams, InferenceMetadata, InferenceQueries,
    ListInferenceMetadataParams, ListInferencesParams, MockInferenceQueries, VariantThroughput,
};
use crate::db::model_inferences::{MockModelInferenceQueries, ModelInferenceQueries};
use crate::db::resolve_uuid::{ResolveUuidQueries, ResolvedObject};
use crate::db::stored_datapoint::StoredDatapoint;
use crate::db::{
    ConfigQueries, MockConfigQueries, ModelLatencyDatapoint, ModelUsageTimePoint, TimeWindow,
};
use crate::error::Error;
use crate::function::FunctionConfig;
use crate::inference::types::StoredModelInference;
use crate::inference::types::{ChatInferenceDatabaseInsert, JsonInferenceDatabaseInsert};
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
    pub(crate) feedback_queries: MockFeedbackQueries,
}

impl MockClickHouseConnectionInfo {
    pub fn new() -> Self {
        Self {
            inference_queries: MockInferenceQueries::new(),
            dataset_queries: MockDatasetQueries::new(),
            config_queries: MockConfigQueries::new(),
            model_inference_queries: MockModelInferenceQueries::new(),
            feedback_queries: MockFeedbackQueries::new(),
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
        params: &CountInferencesParams<'_>,
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

    async fn insert_chat_inferences(
        &self,
        rows: &[ChatInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        self.inference_queries.insert_chat_inferences(rows).await
    }

    async fn insert_json_inferences(
        &self,
        rows: &[JsonInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        self.inference_queries.insert_json_inferences(rows).await
    }

    // ===== Inference count methods (merged from InferenceCountQueries trait) =====

    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesForFunctionParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error> {
        self.inference_queries
            .count_inferences_by_variant(params)
            .await
    }

    async fn count_inferences_with_feedback(
        &self,
        params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error> {
        self.inference_queries
            .count_inferences_with_feedback(params)
            .await
    }

    async fn get_function_throughput_by_variant(
        &self,
        params: GetFunctionThroughputByVariantParams<'_>,
    ) -> Result<Vec<VariantThroughput>, Error> {
        self.inference_queries
            .get_function_throughput_by_variant(params)
            .await
    }

    async fn list_functions_with_inference_count(
        &self,
    ) -> Result<Vec<FunctionInferenceCount>, Error> {
        self.inference_queries
            .list_functions_with_inference_count()
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
        id_mappings: &std::collections::HashMap<Uuid, Uuid>,
    ) -> Result<Vec<Option<Uuid>>, Error> {
        self.dataset_queries
            .clone_datapoints(target_dataset_name, source_datapoint_ids, id_mappings)
            .await
    }
}

#[async_trait]
impl ConfigQueries for MockClickHouseConnectionInfo {
    async fn get_config_snapshot(
        &self,
        snapshot_hash: SnapshotHash,
    ) -> Result<ConfigSnapshot, Error> {
        self.config_queries.get_config_snapshot(snapshot_hash).await
    }

    async fn write_config_snapshot(&self, snapshot: &ConfigSnapshot) -> Result<(), Error> {
        self.config_queries.write_config_snapshot(snapshot).await
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

    async fn insert_model_inferences(&self, rows: &[StoredModelInference]) -> Result<(), Error> {
        self.model_inference_queries
            .insert_model_inferences(rows)
            .await
    }

    async fn count_distinct_models_used(&self) -> Result<u32, Error> {
        self.model_inference_queries
            .count_distinct_models_used()
            .await
    }

    async fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<ModelUsageTimePoint>, Error> {
        self.model_inference_queries
            .get_model_usage_timeseries(time_window, max_periods)
            .await
    }

    async fn get_model_latency_quantiles(
        &self,
        time_window: TimeWindow,
    ) -> Result<Vec<ModelLatencyDatapoint>, Error> {
        self.model_inference_queries
            .get_model_latency_quantiles(time_window)
            .await
    }

    fn get_model_latency_quantile_function_inputs(&self) -> &[f64] {
        self.model_inference_queries
            .get_model_latency_quantile_function_inputs()
    }
}

#[async_trait]
impl FeedbackQueries for MockClickHouseConnectionInfo {
    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: &str,
        variant_names: Option<&Vec<String>>,
        namespace: Option<&str>,
        max_samples_per_variant: Option<u64>,
    ) -> Result<Vec<FeedbackByVariant>, Error> {
        self.feedback_queries
            .get_feedback_by_variant(
                metric_name,
                function_name,
                variant_names,
                namespace,
                max_samples_per_variant,
            )
            .await
    }

    async fn get_cumulative_feedback_timeseries(
        &self,
        function_name: String,
        metric_name: String,
        variant_names: Option<Vec<String>>,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<CumulativeFeedbackTimeSeriesPoint>, Error> {
        self.feedback_queries
            .get_cumulative_feedback_timeseries(
                function_name,
                metric_name,
                variant_names,
                time_window,
                max_periods,
            )
            .await
    }

    async fn query_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<Vec<FeedbackRow>, Error> {
        self.feedback_queries
            .query_feedback_by_target_id(target_id, before, after, limit)
            .await
    }

    async fn query_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<FeedbackBounds, Error> {
        self.feedback_queries
            .query_feedback_bounds_by_target_id(target_id)
            .await
    }

    async fn count_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        self.feedback_queries
            .count_feedback_by_target_id(target_id)
            .await
    }

    async fn query_demonstration_feedback_by_inference_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<Vec<DemonstrationFeedbackRow>, Error> {
        self.feedback_queries
            .query_demonstration_feedback_by_inference_id(target_id, before, after, limit)
            .await
    }

    async fn query_metrics_with_feedback(
        &self,
        function_name: &str,
        function_config: &FunctionConfig,
        variant_name: Option<&str>,
    ) -> Result<Vec<MetricWithFeedback>, Error> {
        self.feedback_queries
            .query_metrics_with_feedback(function_name, function_config, variant_name)
            .await
    }

    async fn query_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<Vec<LatestFeedbackRow>, Error> {
        self.feedback_queries
            .query_latest_feedback_id_by_metric(target_id)
            .await
    }

    async fn get_variant_performances(
        &self,
        params: GetVariantPerformanceParams<'_>,
    ) -> Result<Vec<VariantPerformanceRow>, Error> {
        self.feedback_queries.get_variant_performances(params).await
    }

    async fn insert_boolean_feedback(
        &self,
        row: &BooleanMetricFeedbackInsert,
    ) -> Result<(), Error> {
        self.feedback_queries.insert_boolean_feedback(row).await
    }

    async fn insert_float_feedback(&self, row: &FloatMetricFeedbackInsert) -> Result<(), Error> {
        self.feedback_queries.insert_float_feedback(row).await
    }

    async fn insert_comment_feedback(&self, row: &CommentFeedbackInsert) -> Result<(), Error> {
        self.feedback_queries.insert_comment_feedback(row).await
    }

    async fn insert_demonstration_feedback(
        &self,
        row: &DemonstrationFeedbackInsert,
    ) -> Result<(), Error> {
        self.feedback_queries
            .insert_demonstration_feedback(row)
            .await
    }

    async fn insert_static_eval_feedback(
        &self,
        row: &StaticEvaluationHumanFeedbackInsert,
    ) -> Result<(), Error> {
        self.feedback_queries.insert_static_eval_feedback(row).await
    }
}

#[async_trait]
impl ResolveUuidQueries for MockClickHouseConnectionInfo {
    async fn resolve_uuid(&self, _id: &Uuid) -> Result<Vec<ResolvedObject>, Error> {
        Ok(vec![])
    }
}
