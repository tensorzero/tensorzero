//! Inference statistics types and trait definitions.

use async_trait::async_trait;

use crate::config::MetricConfig;
use crate::error::Error;
use crate::function::FunctionConfigType;

/// Parameters for counting inferences for a function.
#[derive(Debug)]
pub struct CountInferencesParams<'a> {
    pub function_name: &'a str,
    pub function_type: FunctionConfigType,
    pub variant_name: Option<&'a str>,
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
pub trait InferenceStatsQueries {
    /// Counts the number of inferences for a function, optionally filtered by variant.
    async fn count_inferences_for_function(
        &self,
        params: CountInferencesParams<'_>,
    ) -> Result<u64, Error>;

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
