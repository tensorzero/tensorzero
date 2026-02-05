//! Model inference query types and trait definitions.

use async_trait::async_trait;
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

use crate::db::{ModelLatencyDatapoint, ModelUsageTimePoint, TimeWindow};
use crate::error::Error;
use crate::inference::types::StoredModelInference;

/// Trait for model inference queries
#[async_trait]
#[cfg_attr(test, automock)]
pub trait ModelInferenceQueries {
    /// Get all model inferences for a given inference ID.
    async fn get_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<StoredModelInference>, Error>;

    /// Insert model inferences into the database.
    async fn insert_model_inferences(&self, rows: &[StoredModelInference]) -> Result<(), Error>;

    // TODO(#5691): Add a db e2e test for this after we isolate the database in tests.
    /// Count the number of distinct models used.
    async fn count_distinct_models_used(&self) -> Result<u32, Error>;

    /// Get model usage timeseries data.
    async fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<ModelUsageTimePoint>, Error>;

    /// Get model latency quantile distributions.
    async fn get_model_latency_quantiles(
        &self,
        time_window: TimeWindow,
    ) -> Result<Vec<ModelLatencyDatapoint>, Error>;

    /// Get the inputs used for the database's latency quantiles query.
    /// ([0.01, 0.5, 0.90, 0.99], etc - not the actual quantile values.)
    fn get_model_latency_quantile_function_inputs(&self) -> &[f64];
}
