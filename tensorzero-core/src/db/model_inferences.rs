//! Model inference query types and trait definitions.

use async_trait::async_trait;
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

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
}
