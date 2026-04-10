//! Minimal TensorZero client trait for inference.
//!
//! This provides the subset of the full `durable_tools::TensorZeroClient` that
//! `ts-executor-pool` actually requires: just the `inference` method.

use async_trait::async_trait;
use tensorzero_core::client::ClientInferenceParams;
use tensorzero_core::endpoints::inference::InferenceResponse;

#[cfg(test)]
use mockall::automock;

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
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, Box<dyn std::error::Error + Send + Sync>>;
}
