//! Minimal TensorZero client trait for inference.
//!
//! This provides the subset of the full `durable_tools::TensorZeroClient` that
//! `ts-executor-pool` actually requires: just the `inference` method.
//!
//! [`PoolInferenceParams`] is a lightweight version of
//! `tensorzero_core::client::ClientInferenceParams` containing only the fields
//! that this crate reads or writes. Implementations convert to the full
//! `ClientInferenceParams` at the call-site boundary.

use std::collections::HashMap;

use async_trait::async_trait;
use tensorzero_core::endpoints::inference::InferenceResponse;
use tensorzero_types::Input;
use uuid::Uuid;

#[cfg(test)]
use mockall::automock;

/// Minimal inference parameters used within `ts-executor-pool`.
///
/// This contains only the fields that this crate actually accesses.
/// Implementations of [`TensorZeroClient`] are responsible for converting
/// this into the full `ClientInferenceParams` before calling the gateway.
#[derive(Debug, Clone, Default)]
pub struct PoolInferenceParams {
    pub function_name: Option<String>,
    pub episode_id: Option<Uuid>,
    pub input: Input,
    pub tags: HashMap<String, String>,
}

/// Minimal client trait for running TensorZero inference.
///
/// Callers provide an implementation that wraps the full `TensorZeroClient`
/// (or any other inference backend). Only `inference` is required.
#[async_trait]
#[cfg_attr(test, automock)]
pub trait TensorZeroClient: Send + Sync + 'static {
    /// Run inference with the given parameters.
    async fn inference(
        &self,
        params: PoolInferenceParams,
    ) -> Result<InferenceResponse, Box<dyn std::error::Error + Send + Sync>>;
}
