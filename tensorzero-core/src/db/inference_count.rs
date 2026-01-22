//! Inference count types and trait definitions.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use mockall::automock;

use crate::config::MetricConfig;
use crate::db::TimeWindow;
use crate::error::Error;
use crate::function::FunctionConfigType;
use crate::serde_util::{deserialize_u64, serialize_utc_datetime_rfc_3339_with_millis};

/// Parameters for counting inferences for a function.
#[derive(Debug)]
pub struct CountInferencesParams<'a> {
    pub function_name: &'a str,
    pub function_type: FunctionConfigType,
    pub variant_name: Option<&'a str>,
}

/// Row returned from the count_inferences_by_variant query.
#[derive(Debug, Deserialize)]
pub struct CountByVariant {
    pub variant_name: String,
    /// Number of inferences for this variant
    #[serde(deserialize_with = "deserialize_u64")]
    pub inference_count: u64,
    /// ISO 8601 timestamp of the last inference for this variant
    pub last_used_at: String,
}

/// Parameters for counting inferences with feedback.
/// If `metric_threshold` is Some, only counts inferences with feedback meeting the threshold criteria.
pub struct CountInferencesWithFeedbackParams<'a> {
    pub function_name: &'a str,
    pub function_type: FunctionConfigType,
    pub metric_name: &'a str,
    pub metric_config: &'a MetricConfig,
    /// If present, only counts inferences with feedback meeting the threshold criteria.
    pub metric_threshold: Option<f64>,
}

/// Parameters for counting inferences with demonstration feedbacks
pub struct CountInferencesWithDemonstrationFeedbacksParams<'a> {
    pub function_name: &'a str,
    pub function_type: FunctionConfigType,
}

/// Parameters for getting function throughput by variant.
#[derive(Debug)]
pub struct GetFunctionThroughputByVariantParams<'a> {
    pub function_name: &'a str,
    pub time_window: TimeWindow,
    pub max_periods: u32,
}

/// Row returned from the get_function_throughput_by_variant query.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct VariantThroughput {
    /// Start datetime of the period in RFC 3339 format with milliseconds
    #[serde(serialize_with = "serialize_utc_datetime_rfc_3339_with_millis")]
    pub period_start: DateTime<Utc>,
    pub variant_name: String,
    /// Number of inferences for this (period, variant) combination
    pub count: u32,
}

/// Row returned from the list_functions_with_inference_count query.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct FunctionInferenceCount {
    pub function_name: String,
    /// ISO 8601 timestamp of the most recent inference for this function
    #[serde(serialize_with = "serialize_utc_datetime_rfc_3339_with_millis")]
    pub last_inference_timestamp: DateTime<Utc>,
    /// Total number of inferences for this function
    pub inference_count: u32,
}

/// Trait for inference count queries
#[async_trait]
#[cfg_attr(test, automock)]
pub trait InferenceCountQueries {
    /// Counts the number of inferences for a function, optionally filtered by variant.
    async fn count_inferences_for_function(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<u64, Error>;

    /// Counts inferences for a function, optionally filtered by variant, grouped by variant.
    async fn count_inferences_by_variant(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<Vec<CountByVariant>, Error>;

    /// Count the number of inferences with feedback for a metric.
    /// If `metric_threshold` is Some, only counts inferences with feedback meeting the threshold criteria.
    async fn count_inferences_with_feedback(
        &self,
        params: CountInferencesWithFeedbackParams<'_>,
    ) -> Result<u64, Error>;

    /// Count the number of inferences with demonstration feedbacks for a function.
    async fn count_inferences_with_demonstration_feedback(
        &self,
        params: CountInferencesWithDemonstrationFeedbacksParams<'_>,
    ) -> Result<u64, Error>;

    /// Counts the number of inferences for an episode.
    async fn count_inferences_for_episode(&self, episode_id: uuid::Uuid) -> Result<u64, Error>;

    /// Get function throughput (inference counts) grouped by variant and time period.
    /// Returns throughput data for the last `max_periods` time periods, grouped by variant.
    /// For cumulative time window, returns all-time data with a fixed period_start.
    async fn get_function_throughput_by_variant(
        &self,
        params: GetFunctionThroughputByVariantParams<'_>,
    ) -> Result<Vec<VariantThroughput>, Error>;

    /// List all functions with their inference counts, ordered by most recent inference.
    /// Returns the function name, count of inferences, and timestamp of the most recent inference.
    async fn list_functions_with_inference_count(
        &self,
    ) -> Result<Vec<FunctionInferenceCount>, Error>;
}
