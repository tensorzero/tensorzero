//! Delegating database connection that wraps both ClickHouse and Postgres.
//!
//! This module provides a database implementation that delegates operations
//! to both ClickHouse (primary) and Postgres (secondary) databases based on
//! feature flags.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
use crate::config::{Config, MetricConfigLevel};
use crate::db::BatchWriterHandle;
use crate::db::TimeWindow;
use crate::db::batch_inference::{BatchInferenceQueries, CompletedBatchInferenceRow};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::datasets::{
    DatasetMetadata, DatasetQueries, GetDatapointParams, GetDatapointsParams,
    GetDatasetMetadataParams,
};
use crate::db::evaluation_queries::{
    EvaluationQueries, EvaluationResultRow, EvaluationRunInfoByIdRow, EvaluationRunInfoRow,
    EvaluationRunSearchResult, EvaluationStatisticsRow, InferenceEvaluationHumanFeedbackRow,
};
use crate::db::feedback::{
    BooleanMetricFeedbackInsert, CommentFeedbackInsert, CumulativeFeedbackTimeSeriesPoint,
    DemonstrationFeedbackInsert, DemonstrationFeedbackRow, FeedbackBounds, FeedbackByVariant,
    FeedbackQueries, FeedbackRow, FloatMetricFeedbackInsert, GetVariantPerformanceParams,
    LatestFeedbackRow, MetricWithFeedback, StaticEvaluationHumanFeedbackInsert,
    VariantPerformanceRow,
};
use crate::db::inferences::{
    CountByVariant, CountInferencesForFunctionParams, CountInferencesParams,
    CountInferencesWithFeedbackParams, FunctionInferenceCount, FunctionInfo,
    GetFunctionThroughputByVariantParams, InferenceMetadata, InferenceQueries,
    ListInferenceMetadataParams, ListInferencesParams, VariantThroughput,
};
use crate::db::model_inferences::ModelInferenceQueries;
use crate::db::postgres::PostgresConnectionInfo;
use crate::db::resolve_uuid::{ResolveUuidQueries, ResolvedObject};
use crate::db::stored_datapoint::StoredDatapoint;
use crate::db::workflow_evaluation_queries::{
    GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow, WorkflowEvaluationProjectRow,
    WorkflowEvaluationQueries, WorkflowEvaluationRunEpisodeWithFeedbackRow,
    WorkflowEvaluationRunInfo, WorkflowEvaluationRunRow, WorkflowEvaluationRunStatisticsRow,
    WorkflowEvaluationRunWithEpisodeCountRow,
};
use crate::db::{
    ConfigQueries, DICLExampleWithDistance, DICLQueries, DeploymentIdQueries, EpisodeByIdRow,
    EpisodeQueries, HowdyFeedbackCounts, HowdyInferenceCounts, HowdyQueries, HowdyTokenUsage,
    ModelLatencyDatapoint, ModelUsageTimePoint, StoredDICLExample, TableBoundsWithCount,
};
use crate::endpoints::stored_inferences::v1::types::InferenceFilter;
use crate::error::Error;
use crate::feature_flags::ENABLE_POSTGRES_AS_PRIMARY_DATASTORE;
use crate::function::{FunctionConfig, FunctionConfigType};
use crate::inference::types::batch::{BatchModelInferenceRow, BatchRequestRow};
use crate::inference::types::{
    ChatInferenceDatabaseInsert, JsonInferenceDatabaseInsert, StoredModelInference,
};
use crate::stored_inference::StoredInferenceDatabase;
use crate::tool::ToolCallConfigDatabaseInsert;

/// A delegating database implementation that wraps both ClickHouse and Postgres.
///
/// Both ClickHouse and Postgres connections wrap an Arc<> under the hood, so this is safe and cheap to clone.
///
/// When `ENABLE_POSTGRES_AS_PRIMARY_DATASTORE` is set, all reads and writes are
/// delegated to Postgres. Otherwise, all operations go to ClickHouse.
#[derive(Clone)]
pub struct DelegatingDatabaseConnection {
    pub clickhouse: ClickHouseConnectionInfo,
    pub postgres: PostgresConnectionInfo,
}
/// A trait that allows us to express "The returned database supports all these queries"
/// via &(dyn DelegatingDatabaseQueries).
pub trait DelegatingDatabaseQueries:
    ConfigQueries
    + DeploymentIdQueries
    + HowdyQueries
    + FeedbackQueries
    + InferenceQueries
    + DatasetQueries
    + BatchInferenceQueries
    + ModelInferenceQueries
    + WorkflowEvaluationQueries
    + EvaluationQueries
    + ResolveUuidQueries
    + EpisodeQueries
    + DICLQueries
{
    fn batcher_join_handles(&self) -> Vec<BatchWriterHandle>;
}
impl DelegatingDatabaseQueries for ClickHouseConnectionInfo {
    fn batcher_join_handles(&self) -> Vec<BatchWriterHandle> {
        self.batcher_join_handle().into_iter().collect()
    }
}
impl DelegatingDatabaseQueries for PostgresConnectionInfo {
    fn batcher_join_handles(&self) -> Vec<BatchWriterHandle> {
        self.batcher_join_handle().into_iter().collect()
    }
}

impl DelegatingDatabaseQueries for DelegatingDatabaseConnection {
    fn batcher_join_handles(&self) -> Vec<BatchWriterHandle> {
        let mut handles = Vec::new();
        if ENABLE_POSTGRES_AS_PRIMARY_DATASTORE.get() {
            if let Some(h) = self.postgres.batcher_join_handle() {
                handles.push(h);
            }
        } else if let Some(h) = self.clickhouse.batcher_join_handle() {
            handles.push(h);
        }
        handles
    }
}

impl DelegatingDatabaseConnection {
    pub fn new(clickhouse: ClickHouseConnectionInfo, postgres: PostgresConnectionInfo) -> Self {
        Self {
            clickhouse,
            postgres,
        }
    }

    fn get_database(&self) -> &(dyn DelegatingDatabaseQueries + Sync) {
        if ENABLE_POSTGRES_AS_PRIMARY_DATASTORE.get() {
            &self.postgres
        } else {
            &self.clickhouse
        }
    }
}

#[async_trait]
impl ConfigQueries for DelegatingDatabaseConnection {
    async fn get_config_snapshot(
        &self,
        snapshot_hash: SnapshotHash,
    ) -> Result<ConfigSnapshot, Error> {
        self.get_database().get_config_snapshot(snapshot_hash).await
    }

    async fn write_config_snapshot(&self, snapshot: &ConfigSnapshot) -> Result<(), Error> {
        self.get_database().write_config_snapshot(snapshot).await
    }
}

#[async_trait]
impl DeploymentIdQueries for DelegatingDatabaseConnection {
    async fn get_deployment_id(&self) -> Result<String, Error> {
        self.get_database().get_deployment_id().await
    }
}

#[async_trait]
impl HowdyQueries for DelegatingDatabaseConnection {
    async fn count_inferences_for_howdy(&self) -> Result<HowdyInferenceCounts, Error> {
        self.get_database().count_inferences_for_howdy().await
    }

    async fn count_feedbacks_for_howdy(&self) -> Result<HowdyFeedbackCounts, Error> {
        self.get_database().count_feedbacks_for_howdy().await
    }

    async fn get_token_totals_for_howdy(&self) -> Result<HowdyTokenUsage, Error> {
        self.get_database().get_token_totals_for_howdy().await
    }
}

#[async_trait]
impl FeedbackQueries for DelegatingDatabaseConnection {
    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: &str,
        variant_names: Option<&Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, Error> {
        self.get_database()
            .get_feedback_by_variant(metric_name, function_name, variant_names)
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
        self.get_database()
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
        self.get_database()
            .query_feedback_by_target_id(target_id, before, after, limit)
            .await
    }

    async fn query_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<FeedbackBounds, Error> {
        self.get_database()
            .query_feedback_bounds_by_target_id(target_id)
            .await
    }

    async fn count_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        self.get_database()
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
        self.get_database()
            .query_demonstration_feedback_by_inference_id(target_id, before, after, limit)
            .await
    }

    async fn query_metrics_with_feedback(
        &self,
        function_name: &str,
        function_config: &FunctionConfig,
        variant_name: Option<&str>,
    ) -> Result<Vec<MetricWithFeedback>, Error> {
        self.get_database()
            .query_metrics_with_feedback(function_name, function_config, variant_name)
            .await
    }

    async fn query_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<Vec<LatestFeedbackRow>, Error> {
        self.get_database()
            .query_latest_feedback_id_by_metric(target_id)
            .await
    }

    async fn get_variant_performances(
        &self,
        params: GetVariantPerformanceParams<'_>,
    ) -> Result<Vec<VariantPerformanceRow>, Error> {
        self.get_database().get_variant_performances(params).await
    }

    async fn insert_boolean_feedback(
        &self,
        row: &BooleanMetricFeedbackInsert,
    ) -> Result<(), Error> {
        self.get_database().insert_boolean_feedback(row).await
    }

    async fn insert_float_feedback(&self, row: &FloatMetricFeedbackInsert) -> Result<(), Error> {
        self.get_database().insert_float_feedback(row).await
    }

    async fn insert_comment_feedback(&self, row: &CommentFeedbackInsert) -> Result<(), Error> {
        self.get_database().insert_comment_feedback(row).await
    }

    async fn insert_demonstration_feedback(
        &self,
        row: &DemonstrationFeedbackInsert,
    ) -> Result<(), Error> {
        self.get_database().insert_demonstration_feedback(row).await
    }

    async fn insert_static_eval_feedback(
        &self,
        row: &StaticEvaluationHumanFeedbackInsert,
    ) -> Result<(), Error> {
        self.get_database().insert_static_eval_feedback(row).await
    }
}

#[async_trait]
impl InferenceQueries for DelegatingDatabaseConnection {
    async fn list_inferences(
        &self,
        config: &Config,
        params: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInferenceDatabase>, Error> {
        self.get_database().list_inferences(config, params).await
    }

    async fn list_inference_metadata(
        &self,
        params: &ListInferenceMetadataParams,
    ) -> Result<Vec<InferenceMetadata>, Error> {
        self.get_database().list_inference_metadata(params).await
    }

    async fn count_inferences(
        &self,
        config: &Config,
        params: &CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        self.get_database().count_inferences(config, params).await
    }

    async fn get_function_info(
        &self,
        target_id: &Uuid,
        level: MetricConfigLevel,
    ) -> Result<Option<FunctionInfo>, Error> {
        self.get_database()
            .get_function_info(target_id, level)
            .await
    }

    async fn get_chat_inference_tool_params(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<ToolCallConfigDatabaseInsert>, Error> {
        self.get_database()
            .get_chat_inference_tool_params(function_name, inference_id)
            .await
    }

    async fn get_json_inference_output_schema(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<Value>, Error> {
        self.get_database()
            .get_json_inference_output_schema(function_name, inference_id)
            .await
    }

    async fn get_inference_output(
        &self,
        function_info: &FunctionInfo,
        inference_id: Uuid,
    ) -> Result<Option<String>, Error> {
        self.get_database()
            .get_inference_output(function_info, inference_id)
            .await
    }

    async fn insert_chat_inferences(
        &self,
        rows: &[ChatInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        self.get_database().insert_chat_inferences(rows).await
    }

    async fn insert_json_inferences(
        &self,
        rows: &[JsonInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        self.get_database().insert_json_inferences(rows).await
    }

    // ===== Inference count methods (merged from InferenceCountQueries trait) =====

    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesForFunctionParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error> {
        self.get_database()
            .count_inferences_by_variant(params)
            .await
    }

    async fn count_inferences_with_feedback(
        &self,
        params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error> {
        self.get_database()
            .count_inferences_with_feedback(params)
            .await
    }

    async fn get_function_throughput_by_variant(
        &self,
        params: GetFunctionThroughputByVariantParams<'_>,
    ) -> Result<Vec<VariantThroughput>, Error> {
        self.get_database()
            .get_function_throughput_by_variant(params)
            .await
    }

    async fn list_functions_with_inference_count(
        &self,
    ) -> Result<Vec<FunctionInferenceCount>, Error> {
        self.get_database()
            .list_functions_with_inference_count()
            .await
    }
}

#[async_trait]
impl DatasetQueries for DelegatingDatabaseConnection {
    async fn get_dataset_metadata(
        &self,
        params: &GetDatasetMetadataParams,
    ) -> Result<Vec<DatasetMetadata>, Error> {
        self.get_database().get_dataset_metadata(params).await
    }

    async fn count_datapoints_for_dataset(
        &self,
        dataset_name: &str,
        function_name: Option<&str>,
    ) -> Result<u64, Error> {
        self.get_database()
            .count_datapoints_for_dataset(dataset_name, function_name)
            .await
    }

    async fn get_datapoint(&self, params: &GetDatapointParams) -> Result<StoredDatapoint, Error> {
        self.get_database().get_datapoint(params).await
    }

    async fn get_datapoints(
        &self,
        params: &GetDatapointsParams,
    ) -> Result<Vec<StoredDatapoint>, Error> {
        self.get_database().get_datapoints(params).await
    }

    async fn insert_datapoints(&self, datapoints: &[StoredDatapoint]) -> Result<u64, Error> {
        self.get_database().insert_datapoints(datapoints).await
    }

    async fn delete_datapoints(
        &self,
        dataset_name: &str,
        datapoint_ids: Option<&[Uuid]>,
    ) -> Result<u64, Error> {
        self.get_database()
            .delete_datapoints(dataset_name, datapoint_ids)
            .await
    }

    async fn clone_datapoints(
        &self,
        target_dataset_name: &str,
        source_datapoint_ids: &[Uuid],
        id_mappings: &HashMap<Uuid, Uuid>,
    ) -> Result<Vec<Option<Uuid>>, Error> {
        self.get_database()
            .clone_datapoints(target_dataset_name, source_datapoint_ids, id_mappings)
            .await
    }
}
#[async_trait]
impl BatchInferenceQueries for DelegatingDatabaseConnection {
    async fn get_batch_request(
        &self,
        batch_id: Uuid,
        inference_id: Option<Uuid>,
    ) -> Result<Option<BatchRequestRow<'static>>, Error> {
        self.get_database()
            .get_batch_request(batch_id, inference_id)
            .await
    }

    async fn get_batch_model_inferences(
        &self,
        batch_id: Uuid,
        inference_ids: &[Uuid],
    ) -> Result<Vec<BatchModelInferenceRow<'static>>, Error> {
        self.get_database()
            .get_batch_model_inferences(batch_id, inference_ids)
            .await
    }

    async fn get_completed_chat_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error> {
        self.get_database()
            .get_completed_chat_batch_inferences(
                batch_id,
                function_name,
                variant_name,
                inference_id,
            )
            .await
    }

    async fn get_completed_json_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error> {
        self.get_database()
            .get_completed_json_batch_inferences(
                batch_id,
                function_name,
                variant_name,
                inference_id,
            )
            .await
    }

    async fn write_batch_request(&self, row: &BatchRequestRow<'_>) -> Result<(), Error> {
        self.get_database().write_batch_request(row).await
    }

    async fn write_batch_model_inferences(
        &self,
        rows: &[BatchModelInferenceRow<'_>],
    ) -> Result<(), Error> {
        self.get_database().write_batch_model_inferences(rows).await
    }
}

#[async_trait]
impl ModelInferenceQueries for DelegatingDatabaseConnection {
    async fn get_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<StoredModelInference>, Error> {
        self.get_database()
            .get_model_inferences_by_inference_id(inference_id)
            .await
    }

    async fn count_distinct_models_used(&self) -> Result<u32, Error> {
        self.get_database().count_distinct_models_used().await
    }

    async fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<ModelUsageTimePoint>, Error> {
        self.get_database()
            .get_model_usage_timeseries(time_window, max_periods)
            .await
    }

    async fn get_model_latency_quantiles(
        &self,
        time_window: TimeWindow,
    ) -> Result<Vec<ModelLatencyDatapoint>, Error> {
        self.get_database()
            .get_model_latency_quantiles(time_window)
            .await
    }

    fn get_model_latency_quantile_function_inputs(&self) -> &[f64] {
        self.get_database()
            .get_model_latency_quantile_function_inputs()
    }

    async fn insert_model_inferences(&self, rows: &[StoredModelInference]) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        self.get_database().insert_model_inferences(rows).await
    }
}
#[async_trait]
impl WorkflowEvaluationQueries for DelegatingDatabaseConnection {
    async fn list_workflow_evaluation_projects(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<WorkflowEvaluationProjectRow>, Error> {
        self.get_database()
            .list_workflow_evaluation_projects(limit, offset)
            .await
    }

    async fn count_workflow_evaluation_projects(&self) -> Result<u32, Error> {
        self.get_database()
            .count_workflow_evaluation_projects()
            .await
    }

    async fn search_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        project_name: Option<&str>,
        search_query: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error> {
        self.get_database()
            .search_workflow_evaluation_runs(limit, offset, project_name, search_query)
            .await
    }

    async fn list_workflow_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
        run_id: Option<Uuid>,
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunWithEpisodeCountRow>, Error> {
        self.get_database()
            .list_workflow_evaluation_runs(limit, offset, run_id, project_name)
            .await
    }

    async fn count_workflow_evaluation_runs(&self) -> Result<u32, Error> {
        self.get_database().count_workflow_evaluation_runs().await
    }

    async fn get_workflow_evaluation_runs(
        &self,
        run_ids: &[Uuid],
        project_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunRow>, Error> {
        self.get_database()
            .get_workflow_evaluation_runs(run_ids, project_name)
            .await
    }

    async fn get_workflow_evaluation_run_statistics(
        &self,
        run_id: Uuid,
        metric_name: Option<&str>,
    ) -> Result<Vec<WorkflowEvaluationRunStatisticsRow>, Error> {
        self.get_database()
            .get_workflow_evaluation_run_statistics(run_id, metric_name)
            .await
    }

    async fn list_workflow_evaluation_run_episodes_by_task_name(
        &self,
        run_ids: &[Uuid],
        limit: u32,
        offset: u32,
    ) -> Result<Vec<GroupedWorkflowEvaluationRunEpisodeWithFeedbackRow>, Error> {
        self.get_database()
            .list_workflow_evaluation_run_episodes_by_task_name(run_ids, limit, offset)
            .await
    }

    async fn count_workflow_evaluation_run_episodes_by_task_name(
        &self,
        run_ids: &[Uuid],
    ) -> Result<u32, Error> {
        self.get_database()
            .count_workflow_evaluation_run_episodes_by_task_name(run_ids)
            .await
    }

    async fn get_workflow_evaluation_run_episodes_with_feedback(
        &self,
        run_id: Uuid,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<WorkflowEvaluationRunEpisodeWithFeedbackRow>, Error> {
        self.get_database()
            .get_workflow_evaluation_run_episodes_with_feedback(run_id, limit, offset)
            .await
    }

    async fn count_workflow_evaluation_run_episodes(&self, run_id: Uuid) -> Result<u32, Error> {
        self.get_database()
            .count_workflow_evaluation_run_episodes(run_id)
            .await
    }

    async fn get_workflow_evaluation_run_by_episode_id(
        &self,
        episode_id: Uuid,
    ) -> Result<Option<WorkflowEvaluationRunInfo>, Error> {
        self.get_database()
            .get_workflow_evaluation_run_by_episode_id(episode_id)
            .await
    }

    async fn insert_workflow_evaluation_run(
        &self,
        run_id: Uuid,
        variant_pins: &HashMap<String, String>,
        tags: &HashMap<String, String>,
        project_name: Option<&str>,
        run_display_name: Option<&str>,
        snapshot_hash: &SnapshotHash,
    ) -> Result<(), Error> {
        self.get_database()
            .insert_workflow_evaluation_run(
                run_id,
                variant_pins,
                tags,
                project_name,
                run_display_name,
                snapshot_hash,
            )
            .await
    }

    async fn insert_workflow_evaluation_run_episode(
        &self,
        run_id: Uuid,
        episode_id: Uuid,
        task_name: Option<&str>,
        tags: &HashMap<String, String>,
        snapshot_hash: &SnapshotHash,
    ) -> Result<(), Error> {
        self.get_database()
            .insert_workflow_evaluation_run_episode(
                run_id,
                episode_id,
                task_name,
                tags,
                snapshot_hash,
            )
            .await
    }
}

#[async_trait]
impl EvaluationQueries for DelegatingDatabaseConnection {
    async fn count_total_evaluation_runs(&self) -> Result<u64, Error> {
        self.get_database().count_total_evaluation_runs().await
    }

    async fn list_evaluation_runs(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationRunInfoRow>, Error> {
        self.get_database()
            .list_evaluation_runs(limit, offset)
            .await
    }

    async fn count_datapoints_for_evaluation(
        &self,
        function_name: &str,
        evaluation_run_ids: &[Uuid],
    ) -> Result<u64, Error> {
        self.get_database()
            .count_datapoints_for_evaluation(function_name, evaluation_run_ids)
            .await
    }

    async fn search_evaluation_runs(
        &self,
        evaluation_name: &str,
        function_name: &str,
        query: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationRunSearchResult>, Error> {
        self.get_database()
            .search_evaluation_runs(evaluation_name, function_name, query, limit, offset)
            .await
    }

    async fn get_evaluation_run_infos(
        &self,
        evaluation_run_ids: &[Uuid],
        function_name: &str,
    ) -> Result<Vec<EvaluationRunInfoByIdRow>, Error> {
        self.get_database()
            .get_evaluation_run_infos(evaluation_run_ids, function_name)
            .await
    }

    async fn get_evaluation_run_infos_for_datapoint(
        &self,
        datapoint_id: &Uuid,
        function_name: &str,
        function_type: FunctionConfigType,
    ) -> Result<Vec<EvaluationRunInfoByIdRow>, Error> {
        self.get_database()
            .get_evaluation_run_infos_for_datapoint(datapoint_id, function_name, function_type)
            .await
    }

    async fn get_evaluation_statistics(
        &self,
        function_name: &str,
        function_type: FunctionConfigType,
        metric_names: &[String],
        evaluation_run_ids: &[Uuid],
    ) -> Result<Vec<EvaluationStatisticsRow>, Error> {
        self.get_database()
            .get_evaluation_statistics(
                function_name,
                function_type,
                metric_names,
                evaluation_run_ids,
            )
            .await
    }

    async fn get_evaluation_results(
        &self,
        function_name: &str,
        evaluation_run_ids: &[Uuid],
        function_type: FunctionConfigType,
        metric_names: &[String],
        datapoint_id: Option<&Uuid>,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<EvaluationResultRow>, Error> {
        self.get_database()
            .get_evaluation_results(
                function_name,
                evaluation_run_ids,
                function_type,
                metric_names,
                datapoint_id,
                limit,
                offset,
            )
            .await
    }

    async fn get_inference_evaluation_human_feedback(
        &self,
        metric_name: &str,
        datapoint_id: &Uuid,
        output: &str,
    ) -> Result<Option<InferenceEvaluationHumanFeedbackRow>, Error> {
        self.get_database()
            .get_inference_evaluation_human_feedback(metric_name, datapoint_id, output)
            .await
    }
}

#[async_trait]
impl ResolveUuidQueries for DelegatingDatabaseConnection {
    async fn resolve_uuid(&self, id: &Uuid) -> Result<Vec<ResolvedObject>, Error> {
        self.get_database().resolve_uuid(id).await
    }
}

#[async_trait]
impl EpisodeQueries for DelegatingDatabaseConnection {
    async fn query_episode_table(
        &self,
        config: &Config,
        limit: u32,
        before: Option<Uuid>,
        after: Option<Uuid>,
        function_name: Option<String>,
        filters: Option<InferenceFilter>,
    ) -> Result<Vec<EpisodeByIdRow>, Error> {
        self.get_database()
            .query_episode_table(config, limit, before, after, function_name, filters)
            .await
    }

    async fn query_episode_table_bounds(&self) -> Result<TableBoundsWithCount, Error> {
        self.get_database().query_episode_table_bounds().await
    }
}

#[async_trait]
impl DICLQueries for DelegatingDatabaseConnection {
    async fn insert_dicl_example(&self, example: &StoredDICLExample) -> Result<(), Error> {
        self.get_database().insert_dicl_example(example).await
    }

    async fn insert_dicl_examples(&self, examples: &[StoredDICLExample]) -> Result<u64, Error> {
        self.get_database().insert_dicl_examples(examples).await
    }

    async fn get_similar_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
        embedding: &[f32],
        limit: u32,
    ) -> Result<Vec<DICLExampleWithDistance>, Error> {
        self.get_database()
            .get_similar_dicl_examples(function_name, variant_name, embedding, limit)
            .await
    }

    async fn has_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
    ) -> Result<bool, Error> {
        self.get_database()
            .has_dicl_examples(function_name, variant_name)
            .await
    }

    async fn delete_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
        namespace: Option<&str>,
    ) -> Result<u64, Error> {
        self.get_database()
            .delete_dicl_examples(function_name, variant_name, namespace)
            .await
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
mod test_helpers_impl {
    use super::DelegatingDatabaseConnection;
    use crate::db::clickhouse::test_helpers::get_clickhouse;
    use crate::db::postgres::test_helpers::get_postgres;
    use crate::db::test_helpers::TestDatabaseHelpers;
    use crate::feature_flags::ENABLE_POSTGRES_AS_PRIMARY_DATASTORE;
    use async_trait::async_trait;

    impl DelegatingDatabaseConnection {
        pub async fn new_for_e2e_test() -> Self {
            let clickhouse = get_clickhouse().await;
            let postgres = get_postgres().await;
            Self::new(clickhouse, postgres)
        }
    }

    #[async_trait]
    impl TestDatabaseHelpers for DelegatingDatabaseConnection {
        async fn flush_pending_writes(&self) {
            if ENABLE_POSTGRES_AS_PRIMARY_DATASTORE.get() {
                self.postgres.flush_pending_writes().await;
            } else {
                self.clickhouse.flush_pending_writes().await;
            }
        }

        async fn sleep_for_writes_to_be_visible(&self) {
            if ENABLE_POSTGRES_AS_PRIMARY_DATASTORE.get() {
                self.postgres.sleep_for_writes_to_be_visible().await;
            } else {
                self.clickhouse.sleep_for_writes_to_be_visible().await;
            }
        }
    }
}
