use crate::error::Error;
use async_trait::async_trait;
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::TableBounds;
use crate::serde_util::deserialize_u64;

#[async_trait]
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
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
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

#[derive(Clone, Debug, ts_rs::TS, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, ts_rs::TS, Serialize, Deserialize, PartialEq)]
#[ts(export)]
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
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct BooleanMetricFeedbackRow {
    pub id: Uuid,
    pub target_id: Uuid,
    pub metric_name: String,
    pub value: bool,
    pub tags: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct FloatMetricFeedbackRow {
    pub id: Uuid,
    pub target_id: Uuid,
    pub metric_name: String,
    pub value: f64,
    pub tags: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct CommentFeedbackRow {
    pub id: Uuid,
    pub target_id: Uuid,
    pub target_type: CommentTargetType,
    pub value: String,
    pub tags: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}
#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum CommentTargetType {
    Inference,
    Episode,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct DemonstrationFeedbackRow {
    pub id: Uuid,
    pub inference_id: Uuid,
    pub value: String,
    pub tags: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum FeedbackRow {
    Boolean(BooleanMetricFeedbackRow),
    Float(FloatMetricFeedbackRow),
    Comment(CommentFeedbackRow),
    Demonstration(DemonstrationFeedbackRow),
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct FeedbackBounds {
    #[ts(optional)]
    pub first_id: Option<Uuid>,
    #[ts(optional)]
    pub last_id: Option<Uuid>,
    pub by_type: FeedbackBoundsByType,
}

#[derive(Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct FeedbackBoundsByType {
    pub boolean: TableBounds,
    pub float: TableBounds,
    pub comment: TableBounds,
    pub demonstration: TableBounds,
}
