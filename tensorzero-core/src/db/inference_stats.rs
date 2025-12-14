//! Inference statistics types and trait definitions.

use async_trait::async_trait;
use serde::Deserialize;

#[cfg(test)]
use mockall::automock;

use crate::config::MetricConfig;
use crate::error::Error;
use crate::function::FunctionConfigType;
use crate::serde_util::deserialize_u64;

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

/// Trait for inference statistics queries
#[async_trait]
#[cfg_attr(test, automock)]
pub trait InferenceStatsQueries {
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
}
