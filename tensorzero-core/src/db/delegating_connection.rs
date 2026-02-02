//! Delegating database connection that wraps both ClickHouse and Postgres.
//!
//! This module provides a database implementation that delegates operations
//! to both ClickHouse (primary) and Postgres (secondary) databases based on
//! feature flags.

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use crate::config::{Config, MetricConfigLevel};
use crate::db::TimeWindow;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::feedback::{
    BooleanMetricFeedbackInsert, CommentFeedbackInsert, CumulativeFeedbackTimeSeriesPoint,
    DemonstrationFeedbackInsert, DemonstrationFeedbackRow, FeedbackBounds, FeedbackByVariant,
    FeedbackQueries, FeedbackRow, FloatMetricFeedbackInsert, GetVariantPerformanceParams,
    LatestFeedbackRow, MetricWithFeedback, StaticEvaluationHumanFeedbackInsert,
    VariantPerformanceRow,
};
use crate::db::inferences::{
    CountInferencesParams, FunctionInfo, InferenceMetadata, InferenceQueries,
    ListInferenceMetadataParams, ListInferencesParams,
};
use crate::db::postgres::PostgresConnectionInfo;
use crate::error::Error;
use crate::feature_flags::{ENABLE_POSTGRES_READ, ENABLE_POSTGRES_WRITE};
use crate::function::FunctionConfig;
use crate::inference::types::{ChatInferenceDatabaseInsert, JsonInferenceDatabaseInsert};
use crate::stored_inference::StoredInferenceDatabase;
use crate::tool::ToolCallConfigDatabaseInsert;

/// A delegating database implementation that wraps both ClickHouse and Postgres.
///
/// Both ClickHouse and Postgres connections wrap an Arc<> under the hood, so this is safe and cheap to clone.
///
/// This struct delegates database operations as follows:
/// - Read operations: Delegate to Postgres if `ENABLE_POSTGRES_READ` is set,
///   otherwise delegate to ClickHouse
/// - Write operations: Always write to ClickHouse, conditionally write to Postgres
///   based on the `ENABLE_POSTGRES_WRITE` feature flag
///
/// Postgres write errors are logged but do not cause the operation to fail,
/// as ClickHouse remains the source of truth.
///
/// TODO(#5691): Once we're ready to remove ClickHouse dependency, reason about
/// the write path and which write is required for the operation to succeed.
#[derive(Clone)]
pub struct DelegatingDatabaseConnection {
    pub clickhouse: ClickHouseConnectionInfo,
    pub postgres: PostgresConnectionInfo,
}
/// A trait that allows us to express "The returned database supports all these queries"
/// via &(dyn DelegatingDatabaseQueries).
pub trait DelegatingDatabaseQueries: FeedbackQueries + InferenceQueries {}
impl DelegatingDatabaseQueries for ClickHouseConnectionInfo {}
impl DelegatingDatabaseQueries for PostgresConnectionInfo {}

impl DelegatingDatabaseConnection {
    pub fn new(clickhouse: ClickHouseConnectionInfo, postgres: PostgresConnectionInfo) -> Self {
        Self {
            clickhouse,
            postgres,
        }
    }

    fn get_read_database(&self) -> &(dyn DelegatingDatabaseQueries + Sync) {
        if ENABLE_POSTGRES_READ.get() {
            &self.postgres
        } else {
            &self.clickhouse
        }
    }
}

#[async_trait]
impl FeedbackQueries for DelegatingDatabaseConnection {
    // ===== Read methods: delegate based on ENABLE_POSTGRES_READ =====

    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: &str,
        variant_names: Option<&Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, Error> {
        self.get_read_database()
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
        self.get_read_database()
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
        self.get_read_database()
            .query_feedback_by_target_id(target_id, before, after, limit)
            .await
    }

    async fn query_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<FeedbackBounds, Error> {
        self.get_read_database()
            .query_feedback_bounds_by_target_id(target_id)
            .await
    }

    async fn count_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        self.get_read_database()
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
        self.get_read_database()
            .query_demonstration_feedback_by_inference_id(target_id, before, after, limit)
            .await
    }

    async fn query_metrics_with_feedback(
        &self,
        function_name: &str,
        function_config: &FunctionConfig,
        variant_name: Option<&str>,
    ) -> Result<Vec<MetricWithFeedback>, Error> {
        self.get_read_database()
            .query_metrics_with_feedback(function_name, function_config, variant_name)
            .await
    }

    async fn query_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<Vec<LatestFeedbackRow>, Error> {
        self.get_read_database()
            .query_latest_feedback_id_by_metric(target_id)
            .await
    }

    async fn get_variant_performances(
        &self,
        params: GetVariantPerformanceParams<'_>,
    ) -> Result<Vec<VariantPerformanceRow>, Error> {
        self.get_read_database()
            .get_variant_performances(params)
            .await
    }

    // ===== Write methods: write to ClickHouse, conditionally write to Postgres =====

    async fn insert_boolean_feedback(
        &self,
        row: &BooleanMetricFeedbackInsert,
    ) -> Result<(), Error> {
        self.clickhouse.insert_boolean_feedback(row).await?;

        if ENABLE_POSTGRES_WRITE.get()
            && let Err(e) = self.postgres.insert_boolean_feedback(row).await
        {
            tracing::error!("Error writing boolean feedback to Postgres: {e}");
        }

        Ok(())
    }

    async fn insert_float_feedback(&self, row: &FloatMetricFeedbackInsert) -> Result<(), Error> {
        self.clickhouse.insert_float_feedback(row).await?;

        if ENABLE_POSTGRES_WRITE.get()
            && let Err(e) = self.postgres.insert_float_feedback(row).await
        {
            tracing::error!("Error writing float feedback to Postgres: {e}");
        }

        Ok(())
    }

    async fn insert_comment_feedback(&self, row: &CommentFeedbackInsert) -> Result<(), Error> {
        self.clickhouse.insert_comment_feedback(row).await?;

        if ENABLE_POSTGRES_WRITE.get()
            && let Err(e) = self.postgres.insert_comment_feedback(row).await
        {
            tracing::error!("Error writing comment feedback to Postgres: {e}");
        }

        Ok(())
    }

    async fn insert_demonstration_feedback(
        &self,
        row: &DemonstrationFeedbackInsert,
    ) -> Result<(), Error> {
        self.clickhouse.insert_demonstration_feedback(row).await?;

        if ENABLE_POSTGRES_WRITE.get()
            && let Err(e) = self.postgres.insert_demonstration_feedback(row).await
        {
            tracing::error!("Error writing demonstration feedback to Postgres: {e}");
        }

        Ok(())
    }

    async fn insert_static_eval_feedback(
        &self,
        row: &StaticEvaluationHumanFeedbackInsert,
    ) -> Result<(), Error> {
        self.clickhouse.insert_static_eval_feedback(row).await?;

        if ENABLE_POSTGRES_WRITE.get()
            && let Err(e) = self.postgres.insert_static_eval_feedback(row).await
        {
            tracing::error!("Error writing static eval feedback to Postgres: {e}");
        }

        Ok(())
    }
}

#[async_trait]
impl InferenceQueries for DelegatingDatabaseConnection {
    // ===== Read methods: delegate based on ENABLE_POSTGRES_READ =====
    // Note: Currently all read methods delegate to ClickHouse since Postgres
    // read implementations are not yet complete. This will change when Postgres
    // read support is added.

    async fn list_inferences(
        &self,
        config: &Config,
        params: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInferenceDatabase>, Error> {
        self.get_read_database()
            .list_inferences(config, params)
            .await
    }

    async fn list_inference_metadata(
        &self,
        params: &ListInferenceMetadataParams,
    ) -> Result<Vec<InferenceMetadata>, Error> {
        self.get_read_database()
            .list_inference_metadata(params)
            .await
    }

    async fn count_inferences(
        &self,
        config: &Config,
        params: &CountInferencesParams<'_>,
    ) -> Result<u64, Error> {
        self.get_read_database()
            .count_inferences(config, params)
            .await
    }

    async fn get_function_info(
        &self,
        target_id: &Uuid,
        level: MetricConfigLevel,
    ) -> Result<Option<FunctionInfo>, Error> {
        self.get_read_database()
            .get_function_info(target_id, level)
            .await
    }

    async fn get_chat_inference_tool_params(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<ToolCallConfigDatabaseInsert>, Error> {
        self.get_read_database()
            .get_chat_inference_tool_params(function_name, inference_id)
            .await
    }

    async fn get_json_inference_output_schema(
        &self,
        function_name: &str,
        inference_id: Uuid,
    ) -> Result<Option<Value>, Error> {
        self.get_read_database()
            .get_json_inference_output_schema(function_name, inference_id)
            .await
    }

    async fn get_inference_output(
        &self,
        function_info: &FunctionInfo,
        inference_id: Uuid,
    ) -> Result<Option<String>, Error> {
        self.get_read_database()
            .get_inference_output(function_info, inference_id)
            .await
    }

    // ===== Write methods: write to ClickHouse, conditionally write to Postgres =====

    async fn insert_chat_inferences(
        &self,
        rows: &[ChatInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        self.clickhouse.insert_chat_inferences(rows).await?;

        if ENABLE_POSTGRES_WRITE.get()
            && let Err(e) = self.postgres.insert_chat_inferences(rows).await
        {
            tracing::error!("Error writing chat inferences to Postgres: {e}");
        }

        Ok(())
    }

    async fn insert_json_inferences(
        &self,
        rows: &[JsonInferenceDatabaseInsert],
    ) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        self.clickhouse.insert_json_inferences(rows).await?;

        if ENABLE_POSTGRES_WRITE.get()
            && let Err(e) = self.postgres.insert_json_inferences(rows).await
        {
            tracing::error!("Error writing json inferences to Postgres: {e}");
        }

        Ok(())
    }
}
