//! Model inference query types and trait definitions.

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Error;
use crate::inference::types::StoredModelInference;

/// Trait for model inference queries
#[async_trait]
pub trait ModelInferenceQueries {
    /// Get all model inferences for a given inference ID.
    async fn get_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<StoredModelInference>, Error>;
}
