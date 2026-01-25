/// Definitions for batch inference-related database queries.
use async_trait::async_trait;
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

use crate::error::Error;
use crate::inference::types::FinishReason;
use crate::inference::types::batch::{BatchModelInferenceRow, BatchRequestRow};

/// Response row for completed batch inferences (used for both Chat and Json).
#[derive(Debug, serde::Deserialize)]
pub struct CompletedBatchInferenceRow {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub output: String,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub finish_reason: Option<FinishReason>,
}

#[async_trait]
#[cfg_attr(test, automock)]
pub trait BatchInferenceQueries {
    /// Get a batch request by batch_id, optionally filtering by inference_id.
    /// If inference_id is provided, validates that the inference belongs to the batch.
    async fn get_batch_request(
        &self,
        batch_id: Uuid,
        inference_id: Option<Uuid>,
    ) -> Result<Option<BatchRequestRow<'static>>, Error>;

    /// Get batch model inferences for the given batch_id and inference_ids.
    async fn get_batch_model_inferences(
        &self,
        batch_id: Uuid,
        inference_ids: &[Uuid],
    ) -> Result<Vec<BatchModelInferenceRow<'static>>, Error>;

    /// Get completed batch inference responses for Chat functions.
    /// If `inference_id` is Some, fetch only that specific inference. If None, fetch all inferences in the batch.
    async fn get_completed_chat_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error>;

    /// Get completed batch inference responses for Json functions.
    /// If `inference_id` is Some, fetch only that specific inference. If None, fetch all inferences in the batch.
    async fn get_completed_json_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error>;
}
