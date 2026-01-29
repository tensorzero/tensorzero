use crate::config::snapshot::SnapshotHash;
use crate::config::{MetricConfig, MetricConfigLevel};
use crate::error::Error;
use crate::function::{FunctionConfig, FunctionConfigType};
use async_trait::async_trait;
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

use super::{TableBounds, TimeWindow};
use crate::serde_util::deserialize_u64;

// ===== Insert types for write operations =====

/// Row to insert into boolean_metric_feedback
#[derive(Debug, Serialize)]
pub struct BooleanMetricFeedbackInsert {
    pub id: Uuid,
    pub target_id: Uuid,
    pub metric_name: String,
    pub value: bool,
    pub tags: HashMap<String, String>,
    pub snapshot_hash: SnapshotHash,
}

/// Row to insert into float_metric_feedback
#[derive(Debug, Serialize)]
pub struct FloatMetricFeedbackInsert {
    pub id: Uuid,
    pub target_id: Uuid,
    pub metric_name: String,
    pub value: f64,
    pub tags: HashMap<String, String>,
    pub snapshot_hash: SnapshotHash,
}

/// Row to insert into comment_feedback
#[derive(Debug, Serialize)]
pub struct CommentFeedbackInsert {
    pub id: Uuid,
    pub target_id: Uuid,
    pub target_type: CommentTargetType,
    pub value: String,
    pub tags: HashMap<String, String>,
    pub snapshot_hash: SnapshotHash,
}

/// Row to insert into demonstration_feedback
#[derive(Debug, Serialize)]
pub struct DemonstrationFeedbackInsert {
    pub id: Uuid,
    pub inference_id: Uuid,
    pub value: String,
    pub tags: HashMap<String, String>,
    pub snapshot_hash: SnapshotHash,
}

/// Row to insert into static_evaluation_human_feedback
#[derive(Debug, Deserialize, Serialize)]
pub struct StaticEvaluationHumanFeedbackInsert {
    pub feedback_id: Uuid,
    pub metric_name: String,
    pub datapoint_id: Uuid,
    pub output: String,
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluator_inference_id: Option<Uuid>,
}

#[async_trait]
#[cfg_attr(test, automock)]
pub trait FeedbackQueries {
    /// Retrieves cumulative feedback statistics for a given metric and function, optionally filtered by variant names.
    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: &str,
        variant_names: Option<&Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, Error>;

    /// Retrieves a time series of feedback statistics for a given metric and function,
    /// optionally filtered by variant names. Returns cumulative statistics
    /// (mean, variance, count) for each variant at each time point - each time point
    /// includes all data from the beginning up to that point. This will return max_periods
    /// complete time periods worth of data if present as well as the current time period's data.
    /// So there are at most max_periods + 1 time periods worth of data returned.
    ///
    /// # Data Sources
    ///
    /// The function aggregates data from the `FeedbackByVariantStatistics` table, which includes:
    /// - All inference-level feedback (feedback submitted with an inference ID as the target)
    /// - Episode-level feedback **only for episodes where a single variant was used for the function**
    ///
    /// Episodes with mixed variant usage (where multiple variants were used for the same
    /// function within an episode) are excluded from the statistics. This is a conservative
    /// approach to ensure clean attribution of feedback to variants.
    ///
    /// # Parameters
    ///
    /// - `function_name`: The name of the function to query
    /// - `metric_name`: The name of the metric to query
    /// - `variant_names`: Optional filter for specific variants. If `None`, all variants are included.
    ///   If `Some(vec![])`, returns empty results.
    /// - `time_window`: The time granularity (Minute, Hour, Day, Week, or Month)
    /// - `max_periods`: Maximum number of complete time periods to return
    ///
    /// # Returns
    ///
    /// A vector of `CumulativeFeedbackTimeSeriesPoint` containing cumulative statistics
    /// for each variant at each time point, including asymptotic confidence sequences.
    async fn get_cumulative_feedback_timeseries(
        &self,
        function_name: String,
        metric_name: String,
        variant_names: Option<Vec<String>>,
        time_window: super::TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<CumulativeFeedbackTimeSeriesPoint>, Error>;

    /// Queries all feedback (boolean metrics, float metrics, comments, demonstrations) for a given target ID
    async fn query_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<Vec<FeedbackRow>, Error>;

    /// Queries feedback bounds for a given target ID
    async fn query_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<FeedbackBounds, Error>;

    /// Counts total feedback items for a given target ID
    async fn count_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error>;

    async fn query_demonstration_feedback_by_inference_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<Vec<DemonstrationFeedbackRow>, Error>;

    /// Query all metrics that have feedback for a function, optionally filtered by variant
    async fn query_metrics_with_feedback(
        &self,
        function_name: &str,
        function_config: &FunctionConfig,
        variant_name: Option<&str>,
    ) -> Result<Vec<MetricWithFeedback>, Error>;

    /// Query the latest feedback ID for each metric for a given target
    async fn query_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<Vec<LatestFeedbackRow>, Error>;

    /// Get variant performance statistics for a given function and metric.
    ///
    /// Returns performance statistics (average, stdev, count, confidence interval) for each
    /// variant, optionally grouped by time period.
    ///
    /// # Parameters
    ///
    /// - `params`: Parameters specifying the function, metric configuration, time window, and optional variant filter
    ///
    /// # Returns
    ///
    /// A vector of `VariantPerformanceRow` containing statistics for each (variant, time_period) combination.
    async fn get_variant_performances(
        &self,
        params: GetVariantPerformanceParams<'_>,
    ) -> Result<Vec<VariantPerformanceRow>, Error>;

    // ===== Write methods =====

    /// Insert a boolean metric feedback row
    async fn insert_boolean_feedback(&self, row: &BooleanMetricFeedbackInsert)
    -> Result<(), Error>;

    /// Insert a float metric feedback row
    async fn insert_float_feedback(&self, row: &FloatMetricFeedbackInsert) -> Result<(), Error>;

    /// Insert a comment feedback row
    async fn insert_comment_feedback(&self, row: &CommentFeedbackInsert) -> Result<(), Error>;

    /// Insert a demonstration feedback row
    async fn insert_demonstration_feedback(
        &self,
        row: &DemonstrationFeedbackInsert,
    ) -> Result<(), Error>;

    /// Insert a static evaluation human feedback row
    async fn insert_static_eval_feedback(
        &self,
        row: &StaticEvaluationHumanFeedbackInsert,
    ) -> Result<(), Error>;
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct FeedbackByVariant {
    pub variant_name: String,
    // Mean of feedback values for the variant
    pub mean: f32,
    // Variance of feedback values for the variant
    // Equal to None for sample size 1 because ClickHouse uses sample variance with (n - 1) in the denominator
    pub variance: Option<f32>,
    #[serde(deserialize_with = "deserialize_u64")]
    pub count: u64,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InternalCumulativeFeedbackTimeSeriesPoint {
    // Time point up to which cumulative statistics are computed
    pub period_end: DateTime<Utc>,
    pub variant_name: String,
    // Mean of feedback values up to time point `period_end`
    pub mean: f32,
    // Variance of feedback values up to time point `period_end`.
    // Equal to None for sample size 1 because ClickHouse uses sample variance with (n - 1) in the denominator
    pub variance: Option<f32>,
    #[serde(deserialize_with = "deserialize_u64")]
    // Number of feedback values up to time point `period_end`
    pub count: u64,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CumulativeFeedbackTimeSeriesPoint {
    // Time point up to which cumulative statistics are computed
    pub period_end: DateTime<Utc>,
    pub variant_name: String,
    // Mean of feedback values up to time point `period_end`
    pub mean: f32,
    // Variance of feedback values up to time point `period_end`
    pub variance: Option<f32>,
    #[serde(deserialize_with = "deserialize_u64")]
    // Number of feedback values up to time point `period_end`
    pub count: u64,
    // 1 - confidence level for the asymptotic confidence sequence
    pub alpha: f32,
    // Confidence sequence lower and upper bounds
    pub cs_lower: Option<f32>,
    pub cs_upper: Option<f32>,
}

// Feedback by target ID types
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct BooleanMetricFeedbackRow {
    pub id: Uuid,
    pub target_id: Uuid,
    pub metric_name: String,
    pub value: bool,
    pub tags: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct FloatMetricFeedbackRow {
    pub id: Uuid,
    pub target_id: Uuid,
    pub metric_name: String,
    pub value: f64,
    pub tags: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct CommentFeedbackRow {
    pub id: Uuid,
    pub target_id: Uuid,
    pub target_type: CommentTargetType,
    pub value: String,
    pub tags: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum CommentTargetType {
    Inference,
    Episode,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct DemonstrationFeedbackRow {
    pub id: Uuid,
    pub inference_id: Uuid,
    pub value: String,
    pub tags: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum FeedbackRow {
    Boolean(BooleanMetricFeedbackRow),
    Float(FloatMetricFeedbackRow),
    Comment(CommentFeedbackRow),
    Demonstration(DemonstrationFeedbackRow),
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct FeedbackBounds {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<Uuid>,
    pub by_type: FeedbackBoundsByType,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct FeedbackBoundsByType {
    pub boolean: TableBounds,
    pub float: TableBounds,
    pub comment: TableBounds,
    pub demonstration: TableBounds,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct MetricWithFeedback {
    pub function_name: String,
    pub metric_name: String,
    /// The type of metric (boolean, float, demonstration).
    /// None if the metric is not in the current config (e.g., was deleted).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub metric_type: Option<MetricType>,
    pub feedback_count: u32,
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum MetricType {
    Boolean,
    Float,
    Demonstration,
}

#[derive(Debug, Deserialize)]
pub struct LatestFeedbackRow {
    pub metric_name: String,
    pub latest_id: String,
}

/// Parameters for getting variant performance statistics.
#[derive(Debug)]
pub struct GetVariantPerformanceParams<'a> {
    /// The name of the function to query
    pub function_name: &'a str,
    /// The type of the function (Chat or Json) - determines inference table
    pub function_type: FunctionConfigType,
    /// The name of the metric to query
    pub metric_name: &'a str,
    /// Configuration for the metric - determines metric table and level
    pub metric_config: &'a MetricConfig,
    /// Time granularity for grouping performance data
    pub time_window: TimeWindow,
    /// Optional variant name filter. If provided, only returns data for this variant.
    pub variant_name: Option<&'a str>,
}

/// Row returned from the variant performance query.
/// Contains statistics for each (variant, time_period) combination.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct VariantPerformanceRow {
    /// Start datetime of the period in RFC 3339 format.
    /// For cumulative time window, this is '1970-01-01T00:00:00Z'.
    pub period_start: DateTime<Utc>,
    /// The variant name
    pub variant_name: String,
    /// Number of data points in this (variant, period) combination
    pub count: u32,
    /// Average metric value
    pub avg_metric: f64,
    /// Sample standard deviation (null if count < 2)
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdev: Option<f64>,
    /// 95% confidence interval error margin (1.96 * stdev / sqrt(count))
    /// Null if count < 2 (when stdev is null)
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_error: Option<f64>,
}

impl GetVariantPerformanceParams<'_> {
    /// Returns the ClickHouse table name for the inference table based on function type.
    pub fn inference_table_name(&self) -> &'static str {
        self.function_type.table_name()
    }

    /// Returns the ClickHouse table name for the metric feedback table based on metric type.
    pub fn metric_table_name(&self) -> &'static str {
        self.metric_config.r#type.to_clickhouse_table_name()
    }

    /// Returns the level of the metric (inference or episode).
    pub fn metric_level(&self) -> MetricConfigLevel {
        self.metric_config.level.clone()
    }
}
